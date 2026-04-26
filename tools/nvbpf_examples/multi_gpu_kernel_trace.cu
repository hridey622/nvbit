/*
 * NV-BPF Example: Multi-GPU Kernel Trace
 *
 * Host-side launch/context trace for multi-GPU workloads. This intentionally
 * avoids transport assumptions: it reports which visible GPU, context, stream,
 * launch shape, and nearby CUDA API activity surrounds each matched kernel.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

struct LaunchInfo {
    unsigned int gx = 1;
    unsigned int gy = 1;
    unsigned int gz = 1;
    unsigned int bx = 1;
    unsigned int by = 1;
    unsigned int bz = 1;
    unsigned int dynamic_smem = 0;
    CUstream stream = nullptr;
};

struct ApiTotals {
    uint64_t copy = 0;
    uint64_t peer = 0;
    uint64_t sync = 0;
    uint64_t alloc = 0;
    uint64_t event = 0;
    uint64_t context = 0;
    uint64_t copy_bytes = 0;
    uint64_t peer_bytes = 0;
    uint64_t alloc_bytes = 0;
};

static pthread_mutex_t trace_mutex;
static std::string kernel_name_filter;
static bool full_names = false;
static bool trace_api = false;
static bool verbose = false;
static uint64_t event_counter = 0;
static uint64_t launch_counter = 0;
static ApiTotals totals;
static ApiTotals last_launch_totals;
static char last_api_name[96] = "none";
static uint64_t last_api_event = 0;
static uint64_t gpu_launches[128] = {};

static bool csv_or_substring_match(const char* name, const std::string& filter) {
    if (filter.empty()) return true;
    size_t start = 0;
    while (start <= filter.size()) {
        size_t end = filter.find(',', start);
        if (end == std::string::npos) end = filter.size();
        std::string token = filter.substr(start, end - start);
        if (!token.empty() && strstr(name, token.c_str()) != nullptr) return true;
        if (end == filter.size()) break;
        start = end + 1;
    }
    return false;
}

static std::string compact_kernel_name(const char* raw) {
    if (raw == nullptr) return std::string("<unknown>");
    if (full_names) return std::string(raw);
    std::string name = raw;
    if (name.rfind("void ", 0) == 0) name = name.substr(5);
    size_t paren = name.find('(');
    if (paren != std::string::npos) name = name.substr(0, paren);
    if (name.size() <= 80) return name;
    return name.substr(0, 36) + "..." + name.substr(name.size() - 36);
}

static void format_bytes(uint64_t bytes, char* out, size_t out_size) {
    if (bytes < 1024) {
        snprintf(out, out_size, "%lu B", bytes);
    } else if (bytes < 1024ULL * 1024ULL) {
        snprintf(out, out_size, "%.1f KiB", (double)bytes / 1024.0);
    } else {
        snprintf(out, out_size, "%.2f MiB", (double)bytes / (1024.0 * 1024.0));
    }
}

static void print_activity_line(const ApiTotals& delta) {
    uint64_t other_copy = delta.copy > delta.peer ? delta.copy - delta.peer : 0;
    uint64_t other_copy_bytes =
        delta.copy_bytes > delta.peer_bytes ? delta.copy_bytes - delta.peer_bytes : 0;

    char peer_bytes[32];
    char copy_bytes[32];
    char alloc_bytes[32];
    format_bytes(delta.peer_bytes, peer_bytes, sizeof(peer_bytes));
    format_bytes(other_copy_bytes, copy_bytes, sizeof(copy_bytes));
    format_bytes(delta.alloc_bytes, alloc_bytes, sizeof(alloc_bytes));

    bool printed = false;
    printf("    before: ");
    if (delta.peer > 0) {
        printf("peer_copy=%lu (%s)", delta.peer, peer_bytes);
        printed = true;
    }
    if (other_copy > 0) {
        printf("%scopy=%lu (%s)", printed ? ", " : "", other_copy, copy_bytes);
        printed = true;
    }
    if (delta.sync > 0) {
        printf("%ssync=%lu", printed ? ", " : "", delta.sync);
        printed = true;
    }
    if (delta.alloc > 0) {
        printf("%salloc=%lu (%s)", printed ? ", " : "", delta.alloc, alloc_bytes);
        printed = true;
    }
    if (!printed) {
        printf("no tracked CUDA API activity");
    }
    printf("\n");
}

static int device_for_context(CUcontext target_ctx) {
    (void)target_ctx;
    CUdevice dev = (CUdevice)-1;
    return cuCtxGetDevice(&dev) == CUDA_SUCCESS ? (int)dev : -1;
}

static int device_for_pointer(CUdeviceptr ptr) {
    if (ptr == 0) return -1;
    CUdevice dev = (CUdevice)-1;
    if (cuPointerGetAttribute(&dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr) == CUDA_SUCCESS) {
        return (int)dev;
    }
    return -1;
}

static LaunchInfo get_launch_info(nvbit_api_cuda_t cbid, void* params) {
    LaunchInfo info;
    if (cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
        if (p->config != nullptr) {
            info.gx = p->config->gridDimX;
            info.gy = p->config->gridDimY;
            info.gz = p->config->gridDimZ;
            info.bx = p->config->blockDimX;
            info.by = p->config->blockDimY;
            info.bz = p->config->blockDimZ;
            info.dynamic_smem = p->config->sharedMemBytes;
            info.stream = p->config->hStream;
        }
    } else if (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz) {
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
        info.gx = p->gridDimX;
        info.gy = p->gridDimY;
        info.gz = p->gridDimZ;
        info.bx = p->blockDimX;
        info.by = p->blockDimY;
        info.bz = p->blockDimZ;
        info.dynamic_smem = p->sharedMemBytes;
        info.stream = p->hStream;
    } else if (cbid == API_CUDA_cuLaunchGridAsync) {
        cuLaunchGridAsync_params* p = (cuLaunchGridAsync_params*)params;
        info.gx = p->grid_width;
        info.gy = p->grid_height;
        info.stream = p->hStream;
    } else if (cbid == API_CUDA_cuLaunchGrid) {
        cuLaunchGrid_params* p = (cuLaunchGrid_params*)params;
        info.gx = p->grid_width;
        info.gy = p->grid_height;
    }
    return info;
}

static uint64_t copy_bytes_for_cbid(nvbit_api_cuda_t cbid, void* params) {
    switch (cbid) {
        case API_CUDA_cuMemcpyHtoD:
            return ((cuMemcpyHtoD_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyHtoD_v2:
            return ((cuMemcpyHtoD_v2_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyHtoD_v2_ptds:
            return ((cuMemcpyHtoD_v2_ptds_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyHtoDAsync:
            return ((cuMemcpyHtoDAsync_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyHtoDAsync_v2:
            return ((cuMemcpyHtoDAsync_v2_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyHtoDAsync_v2_ptsz:
            return ((cuMemcpyHtoDAsync_v2_ptsz_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoH:
            return ((cuMemcpyDtoH_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoH_v2:
            return ((cuMemcpyDtoH_v2_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoH_v2_ptds:
            return ((cuMemcpyDtoH_v2_ptds_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoHAsync:
            return ((cuMemcpyDtoHAsync_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoHAsync_v2:
            return ((cuMemcpyDtoHAsync_v2_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoHAsync_v2_ptsz:
            return ((cuMemcpyDtoHAsync_v2_ptsz_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoD:
            return ((cuMemcpyDtoD_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoD_v2:
            return ((cuMemcpyDtoD_v2_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoD_v2_ptds:
            return ((cuMemcpyDtoD_v2_ptds_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoDAsync:
            return ((cuMemcpyDtoDAsync_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoDAsync_v2:
            return ((cuMemcpyDtoDAsync_v2_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoDAsync_v2_ptsz:
            return ((cuMemcpyDtoDAsync_v2_ptsz_params*)params)->ByteCount;
        case API_CUDA_cuMemcpy:
            return ((cuMemcpy_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyAsync:
            return ((cuMemcpyAsync_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyAsync_ptsz:
            return ((cuMemcpyAsync_ptsz_params*)params)->ByteCount;
        default:
            return 0;
    }
}

static uint64_t peer_bytes_for_cbid(nvbit_api_cuda_t cbid, void* params) {
    switch (cbid) {
        case API_CUDA_cuMemcpyPeer:
            return ((cuMemcpyPeer_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyPeer_ptds:
            return ((cuMemcpyPeer_ptds_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyPeerAsync:
            return ((cuMemcpyPeerAsync_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyPeerAsync_ptsz:
            return ((cuMemcpyPeerAsync_ptsz_params*)params)->ByteCount;
        default:
            return 0;
    }
}

static uint64_t alloc_bytes_for_cbid(nvbit_api_cuda_t cbid, void* params) {
    switch (cbid) {
        case API_CUDA_cuMemAlloc:
            return ((cuMemAlloc_params*)params)->bytesize;
        case API_CUDA_cuMemAlloc_v2:
            return ((cuMemAlloc_v2_params*)params)->bytesize;
        case API_CUDA_cuMemAllocAsync:
            return ((cuMemAllocAsync_params*)params)->bytesize;
        case API_CUDA_cuMemAllocAsync_ptsz:
            return ((cuMemAllocAsync_ptsz_params*)params)->bytesize;
        case API_CUDA_cuMemAllocFromPoolAsync:
            return ((cuMemAllocFromPoolAsync_params*)params)->bytesize;
        case API_CUDA_cuMemAllocFromPoolAsync_ptsz:
            return ((cuMemAllocFromPoolAsync_ptsz_params*)params)->bytesize;
        default:
            return 0;
    }
}

static bool is_copy_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuMemcpy ||
           cbid == API_CUDA_cuMemcpyAsync ||
           cbid == API_CUDA_cuMemcpyAsync_ptsz ||
           cbid == API_CUDA_cuMemcpyHtoD ||
           cbid == API_CUDA_cuMemcpyHtoD_v2 ||
           cbid == API_CUDA_cuMemcpyHtoD_v2_ptds ||
           cbid == API_CUDA_cuMemcpyHtoDAsync ||
           cbid == API_CUDA_cuMemcpyHtoDAsync_v2 ||
           cbid == API_CUDA_cuMemcpyHtoDAsync_v2_ptsz ||
           cbid == API_CUDA_cuMemcpyDtoH ||
           cbid == API_CUDA_cuMemcpyDtoH_v2 ||
           cbid == API_CUDA_cuMemcpyDtoH_v2_ptds ||
           cbid == API_CUDA_cuMemcpyDtoHAsync ||
           cbid == API_CUDA_cuMemcpyDtoHAsync_v2 ||
           cbid == API_CUDA_cuMemcpyDtoHAsync_v2_ptsz ||
           cbid == API_CUDA_cuMemcpyDtoD ||
           cbid == API_CUDA_cuMemcpyDtoD_v2 ||
           cbid == API_CUDA_cuMemcpyDtoD_v2_ptds ||
           cbid == API_CUDA_cuMemcpyDtoDAsync ||
           cbid == API_CUDA_cuMemcpyDtoDAsync_v2 ||
           cbid == API_CUDA_cuMemcpyDtoDAsync_v2_ptsz;
}

static bool is_peer_copy_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuMemcpyPeer ||
           cbid == API_CUDA_cuMemcpyPeer_ptds ||
           cbid == API_CUDA_cuMemcpyPeerAsync ||
           cbid == API_CUDA_cuMemcpyPeerAsync_ptsz;
}

static bool is_sync_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuCtxSynchronize ||
           cbid == API_CUDA_cuCtxSynchronize_v2 ||
           cbid == API_CUDA_cuStreamSynchronize ||
           cbid == API_CUDA_cuStreamSynchronize_ptsz ||
           cbid == API_CUDA_cuEventSynchronize;
}

static bool is_alloc_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuMemAlloc ||
           cbid == API_CUDA_cuMemAlloc_v2 ||
           cbid == API_CUDA_cuMemAllocAsync ||
           cbid == API_CUDA_cuMemAllocAsync_ptsz ||
           cbid == API_CUDA_cuMemAllocFromPoolAsync ||
           cbid == API_CUDA_cuMemAllocFromPoolAsync_ptsz ||
           cbid == API_CUDA_cuMemFree ||
           cbid == API_CUDA_cuMemFree_v2 ||
           cbid == API_CUDA_cuMemFreeAsync ||
           cbid == API_CUDA_cuMemFreeAsync_ptsz;
}

static bool is_event_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuEventRecord ||
           cbid == API_CUDA_cuEventRecord_ptsz ||
           cbid == API_CUDA_cuEventRecordWithFlags ||
           cbid == API_CUDA_cuEventRecordWithFlags_ptsz ||
           cbid == API_CUDA_cuStreamWaitEvent ||
           cbid == API_CUDA_cuStreamWaitEvent_ptsz;
}

static bool is_context_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuCtxSetCurrent ||
           cbid == API_CUDA_cuCtxPushCurrent ||
           cbid == API_CUDA_cuCtxPushCurrent_v2 ||
           cbid == API_CUDA_cuCtxPopCurrent ||
           cbid == API_CUDA_cuCtxPopCurrent_v2;
}

static void remember_api_event(const char* name) {
    snprintf(last_api_name, sizeof(last_api_name), "%s", name ? name : "<unknown>");
    last_api_event = event_counter;
}

static void update_api_totals(CUcontext ctx, nvbit_api_cuda_t cbid, const char* name, void* params) {
    if (is_peer_copy_cbid(cbid)) {
        uint64_t bytes = peer_bytes_for_cbid(cbid, params);
        totals.peer++;
        totals.copy++;
        totals.peer_bytes += bytes;
        totals.copy_bytes += bytes;
        remember_api_event(name);
        if (trace_api) {
            int src = -1;
            int dst = -1;
            if (cbid == API_CUDA_cuMemcpyPeer) {
                cuMemcpyPeer_params* p = (cuMemcpyPeer_params*)params;
                src = device_for_pointer(p->srcDevice);
                dst = device_for_pointer(p->dstDevice);
            } else if (cbid == API_CUDA_cuMemcpyPeer_ptds) {
                cuMemcpyPeer_ptds_params* p = (cuMemcpyPeer_ptds_params*)params;
                src = device_for_pointer(p->srcDevice);
                dst = device_for_pointer(p->dstDevice);
            } else if (cbid == API_CUDA_cuMemcpyPeerAsync) {
                cuMemcpyPeerAsync_params* p = (cuMemcpyPeerAsync_params*)params;
                src = device_for_pointer(p->srcDevice);
                dst = device_for_pointer(p->dstDevice);
            } else {
                cuMemcpyPeerAsync_ptsz_params* p = (cuMemcpyPeerAsync_ptsz_params*)params;
                src = device_for_pointer(p->srcDevice);
                dst = device_for_pointer(p->dstDevice);
            }
            printf("[NVBPF] api seq=%lu kind=peer_copy bytes=%lu src_gpu=%d dst_gpu=%d name=%s\n",
                   event_counter, bytes, src, dst, name);
        }
        return;
    }
    if (is_copy_cbid(cbid)) {
        uint64_t bytes = copy_bytes_for_cbid(cbid, params);
        totals.copy++;
        totals.copy_bytes += bytes;
        remember_api_event(name);
        if (trace_api) {
            printf("[NVBPF] api seq=%lu kind=copy bytes=%lu gpu=%d name=%s\n",
                   event_counter, bytes, device_for_context(ctx), name);
        }
        return;
    }
    if (is_sync_cbid(cbid)) {
        totals.sync++;
        remember_api_event(name);
        if (trace_api) {
            printf("[NVBPF] api seq=%lu kind=sync gpu=%d name=%s\n",
                   event_counter, device_for_context(ctx), name);
        }
        return;
    }
    if (is_alloc_cbid(cbid)) {
        uint64_t bytes = alloc_bytes_for_cbid(cbid, params);
        totals.alloc++;
        totals.alloc_bytes += bytes;
        remember_api_event(name);
        if (trace_api) {
            printf("[NVBPF] api seq=%lu kind=alloc bytes=%lu gpu=%d name=%s\n",
                   event_counter, bytes, device_for_context(ctx), name);
        }
        return;
    }
    if (is_event_cbid(cbid)) {
        totals.event++;
        remember_api_event(name);
        if (trace_api) {
            printf("[NVBPF] api seq=%lu kind=event gpu=%d name=%s\n",
                   event_counter, device_for_context(ctx), name);
        }
        return;
    }
    if (is_context_cbid(cbid)) {
        totals.context++;
        remember_api_event(name);
        if (trace_api) {
            printf("[NVBPF] api seq=%lu kind=context gpu=%d name=%s\n",
                   event_counter, device_for_context(ctx), name);
        }
        return;
    }
}

static ApiTotals delta_since_last_launch() {
    ApiTotals delta;
    delta.copy = totals.copy - last_launch_totals.copy;
    delta.peer = totals.peer - last_launch_totals.peer;
    delta.sync = totals.sync - last_launch_totals.sync;
    delta.alloc = totals.alloc - last_launch_totals.alloc;
    delta.event = totals.event - last_launch_totals.event;
    delta.context = totals.context - last_launch_totals.context;
    delta.copy_bytes = totals.copy_bytes - last_launch_totals.copy_bytes;
    delta.peer_bytes = totals.peer_bytes - last_launch_totals.peer_bytes;
    delta.alloc_bytes = totals.alloc_bytes - last_launch_totals.alloc_bytes;
    return delta;
}

void nvbit_at_init() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    pthread_mutex_init(&trace_mutex, nullptr);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    trace_api = getenv("NVBPF_TRACE_API") != nullptr;
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    printf("[NVBPF MGPU] loaded");
    if (!kernel_name_filter.empty()) {
        printf(" filter=%s", kernel_name_filter.c_str());
    }
    printf("\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    bool is_launch = nvbpf_is_launch_event(cbid);

    pthread_mutex_lock(&trace_mutex);
    event_counter++;

    if (!is_launch && is_exit) {
        update_api_totals(ctx, cbid, name, params);
        pthread_mutex_unlock(&trace_mutex);
        return;
    }

    if (!is_launch) {
        pthread_mutex_unlock(&trace_mutex);
        return;
    }

    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);
    bool match = csv_or_substring_match(func_name, kernel_name_filter);
    if (!match) {
        pthread_mutex_unlock(&trace_mutex);
        return;
    }

    int gpu = device_for_context(ctx);
    LaunchInfo launch = get_launch_info(cbid, params);
    func_config_t cfg{};
    nvbit_get_func_config(ctx, func, &cfg);
    std::string kernel = compact_kernel_name(func_name);

    if (!is_exit) {
        launch_counter++;
        if (gpu >= 0 && gpu < (int)(sizeof(gpu_launches) / sizeof(gpu_launches[0]))) {
            gpu_launches[gpu]++;
        }

        ApiTotals delta = delta_since_last_launch();
        uint64_t last_delta = last_api_event == 0 ? 0 : event_counter - last_api_event;
        char dynamic_smem[32];
        char static_smem[32];
        format_bytes(launch.dynamic_smem, dynamic_smem, sizeof(dynamic_smem));
        format_bytes(cfg.shmem_static_nbytes, static_smem, sizeof(static_smem));

        printf("[NVBPF MGPU] #%lu GPU%d %s\n", launch_counter, gpu, kernel.c_str());
        printf("    launch: grid=%ux%ux%u block=%ux%ux%u regs=%u dyn_smem=%s\n",
               launch.gx, launch.gy, launch.gz, launch.bx, launch.by, launch.bz,
               cfg.num_registers, dynamic_smem);
        print_activity_line(delta);
        if (verbose) {
            printf("    detail: ctx=%p stream=%p static_smem=%s event=%lu last_api=%s last_delta_events=%lu api_events=%lu context_events=%lu\n",
                   (void*)ctx, (void*)launch.stream, static_smem, event_counter,
                   last_api_name, last_delta, delta.event, delta.context);
        }
    } else {
        int status = pStatus ? (int)(*pStatus) : 0;
        if (verbose || status != 0) {
            printf("[NVBPF MGPU] #%lu exit GPU%d status=%d %s\n",
                   launch_counter, gpu, status, kernel.c_str());
        }
        last_launch_totals = totals;
    }

    pthread_mutex_unlock(&trace_mutex);
}

void nvbit_at_term() {
    uint64_t other_copy = totals.copy > totals.peer ? totals.copy - totals.peer : 0;
    uint64_t other_copy_bytes =
        totals.copy_bytes > totals.peer_bytes ? totals.copy_bytes - totals.peer_bytes : 0;
    char peer_bytes[32];
    char copy_bytes[32];
    char alloc_bytes[32];
    format_bytes(totals.peer_bytes, peer_bytes, sizeof(peer_bytes));
    format_bytes(other_copy_bytes, copy_bytes, sizeof(copy_bytes));
    format_bytes(totals.alloc_bytes, alloc_bytes, sizeof(alloc_bytes));

    printf("[NVBPF MGPU] summary: launches=%lu", launch_counter);
    for (size_t i = 0; i < sizeof(gpu_launches) / sizeof(gpu_launches[0]); i++) {
        if (gpu_launches[i] != 0) {
            printf(" gpu%zu=%lu", i, gpu_launches[i]);
        }
    }
    printf("\n");
    printf("[NVBPF MGPU] activity: peer_copy=%lu (%s), other_copy=%lu (%s), sync=%lu, alloc=%lu (%s)",
           totals.peer, peer_bytes, other_copy, copy_bytes, totals.sync, totals.alloc,
           alloc_bytes);
    if (verbose) {
        printf(", event=%lu, context=%lu", totals.event, totals.context);
    }
    printf("\n");
}
