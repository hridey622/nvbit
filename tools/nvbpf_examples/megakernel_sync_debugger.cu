/*
 * NV-BPF Example: Megakernel Sync Debugger
 *
 * Combined host/device debugger for long-running or synchronization-heavy
 * kernels. Host callbacks report multi-GPU placement and API activity around
 * launches; device hooks summarize sync-relevant instruction families inside
 * each matched kernel.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_set>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

enum DebugCategory {
    CAT_BARRIER = 0,
    CAT_MEMBAR = 1,
    CAT_ATOMIC = 2,
    CAT_REDUCTION = 3,
    CAT_BRANCH = 4,
    CAT_LOAD = 5,
    CAT_STORE = 6,
    CAT_TENSOR = 7,
    CAT_CP_ASYNC = 8,
    CAT_HEARTBEAT = 9,
    CAT_COUNT = 10,
};

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
    uint64_t copy_bytes = 0;
    uint64_t peer_bytes = 0;
    uint64_t alloc_bytes = 0;
};

BPF_ARRAY(category_counts, uint64_t, CAT_COUNT);
BPF_ARRAY(category_active_sum, uint64_t, CAT_COUNT);
BPF_ARRAY(category_low_active, uint64_t, CAT_COUNT);
BPF_ARRAY(pred_off_events, uint64_t, 1);
BPF_ARRAY(active_lane_hist, uint64_t, 33);
BPF_PERCPU_ARRAY(sm_events, uint64_t, 1);
BPF_ARRAY(active_sm_bitmap, uint64_t, 4);

extern "C" __device__ __noinline__ void msd_count_category(int pred,
                                                           uint64_t pcounts,
                                                           uint64_t pactive_sum,
                                                           uint64_t plow_active,
                                                           uint64_t ppred_off,
                                                           uint64_t phist,
                                                           uint64_t psm_events,
                                                           uint64_t pbitmap,
                                                           uint32_t category,
                                                           uint32_t threshold);

static pthread_mutex_t debugger_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static std::string kernel_name_filter;
static bool full_names = false;
static bool verbose = false;
static bool trace_api = false;
static bool trace_memory = false;
static bool trace_branches = false;
static bool trace_tensor = false;
static bool trace_cp_async = true;
static uint32_t low_active_threshold = 16;
static uint64_t event_counter = 0;
static uint64_t launch_counter = 0;
static ApiTotals api_totals;
static ApiTotals last_launch_api_totals;
static uint64_t gpu_launches[128] = {};

static bool opcode_starts_with(const char* opcode, const char* prefix) {
    return strncmp(opcode, prefix, strlen(prefix)) == 0;
}

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
    if (name.size() <= 82) return name;
    return name.substr(0, 38) + "..." + name.substr(name.size() - 38);
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

static int current_device() {
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

static bool is_barrier_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "BAR") ||
           opcode_starts_with(opcode, "DEPBAR") ||
           opcode_starts_with(opcode, "ERRBAR") ||
           opcode_starts_with(opcode, "WARPSYNC") ||
           opcode_starts_with(opcode, "BSSY") ||
           opcode_starts_with(opcode, "BSYNC");
}

static bool is_membar_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "MEMBAR");
}

static bool is_atomic_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "ATOM");
}

static bool is_reduction_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "RED");
}

static bool is_branch_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "BRA") ||
           opcode_starts_with(opcode, "JMP") ||
           opcode_starts_with(opcode, "JMX") ||
           opcode_starts_with(opcode, "BRX") ||
           opcode_starts_with(opcode, "CALL") ||
           opcode_starts_with(opcode, "RET") ||
           opcode_starts_with(opcode, "EXIT");
}

static bool is_tensor_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "HMMA") ||
           opcode_starts_with(opcode, "MMA") ||
           opcode_starts_with(opcode, "WGMMA") ||
           opcode_starts_with(opcode, "BMMA");
}

static bool is_cp_async_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "CPASYNC") ||
           opcode_starts_with(opcode, "LDGSTS");
}

static void reset_state() {
    category_counts.reset();
    category_active_sum.reset();
    category_low_active.reset();
    pred_off_events.reset();
    active_lane_hist.reset();
    sm_events.reset();
    active_sm_bitmap.reset();
}

static void inject_category_counter(Instr* instr, DebugCategory category) {
    nvbit_insert_call(instr, "msd_count_category", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(instr);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&category_counts.data[0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&category_active_sum.data[0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&category_low_active.data[0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&pred_off_events.data[0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&active_lane_hist.data[0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&sm_events.data[0][0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&active_sm_bitmap.data[0]);
    nvbit_add_call_arg_const_val32(instr, (uint32_t)category);
    nvbit_add_call_arg_const_val32(instr, low_active_threshold);
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        bool heartbeat_inserted = false;
        for (auto* instr : instrs) {
            const char* opcode = instr->getOpcodeShort();
            if (!heartbeat_inserted) {
                inject_category_counter(instr, CAT_HEARTBEAT);
                heartbeat_inserted = true;
            }
            if (is_barrier_opcode(opcode)) inject_category_counter(instr, CAT_BARRIER);
            if (is_membar_opcode(opcode)) inject_category_counter(instr, CAT_MEMBAR);
            if (is_atomic_opcode(opcode)) inject_category_counter(instr, CAT_ATOMIC);
            if (is_reduction_opcode(opcode)) inject_category_counter(instr, CAT_REDUCTION);
            if (trace_branches && is_branch_opcode(opcode)) {
                inject_category_counter(instr, CAT_BRANCH);
            }
            if (trace_memory && instr->isLoad() &&
                instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) {
                inject_category_counter(instr, CAT_LOAD);
            }
            if (trace_memory && instr->isStore()) inject_category_counter(instr, CAT_STORE);
            if (trace_tensor && is_tensor_opcode(opcode)) inject_category_counter(instr, CAT_TENSOR);
            if (trace_cp_async && is_cp_async_opcode(opcode)) {
                inject_category_counter(instr, CAT_CP_ASYNC);
            }
        }
    }
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

static void update_api_totals(CUcontext ctx, nvbit_api_cuda_t cbid, const char* name, void* params) {
    if (is_peer_copy_cbid(cbid)) {
        uint64_t bytes = peer_bytes_for_cbid(cbid, params);
        api_totals.peer++;
        api_totals.copy++;
        api_totals.peer_bytes += bytes;
        api_totals.copy_bytes += bytes;
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
            printf("[NVBPF MEGA] api peer_copy bytes=%lu src_gpu=%d dst_gpu=%d name=%s\n",
                   bytes, src, dst, name);
        }
    } else if (is_copy_cbid(cbid)) {
        uint64_t bytes = copy_bytes_for_cbid(cbid, params);
        api_totals.copy++;
        api_totals.copy_bytes += bytes;
        if (trace_api) {
            printf("[NVBPF MEGA] api copy bytes=%lu gpu=%d name=%s\n",
                   bytes, current_device(), name);
        }
    } else if (is_sync_cbid(cbid)) {
        api_totals.sync++;
        if (trace_api) {
            printf("[NVBPF MEGA] api sync gpu=%d name=%s\n", current_device(), name);
        }
    } else if (is_alloc_cbid(cbid)) {
        uint64_t bytes = alloc_bytes_for_cbid(cbid, params);
        api_totals.alloc++;
        api_totals.alloc_bytes += bytes;
        if (trace_api) {
            printf("[NVBPF MEGA] api alloc bytes=%lu gpu=%d name=%s\n",
                   bytes, current_device(), name);
        }
    }
}

static ApiTotals delta_since_last_launch() {
    ApiTotals delta;
    delta.copy = api_totals.copy - last_launch_api_totals.copy;
    delta.peer = api_totals.peer - last_launch_api_totals.peer;
    delta.sync = api_totals.sync - last_launch_api_totals.sync;
    delta.alloc = api_totals.alloc - last_launch_api_totals.alloc;
    delta.copy_bytes = api_totals.copy_bytes - last_launch_api_totals.copy_bytes;
    delta.peer_bytes = api_totals.peer_bytes - last_launch_api_totals.peer_bytes;
    delta.alloc_bytes = api_totals.alloc_bytes - last_launch_api_totals.alloc_bytes;
    return delta;
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
    if (!printed) printf("no tracked API activity");
    printf("\n");
}

static uint64_t read_category(DebugCategory category) {
    uint64_t* val = category_counts.lookup((uint32_t)category);
    return val ? *val : 0;
}

static uint64_t read_low_active(DebugCategory category) {
    uint64_t* val = category_low_active.lookup((uint32_t)category);
    return val ? *val : 0;
}

static double avg_active_lanes(DebugCategory category) {
    uint64_t* count = category_counts.lookup((uint32_t)category);
    uint64_t* active = category_active_sum.lookup((uint32_t)category);
    if (!count || !active || *count == 0) return 0.0;
    return (double)(*active) / (double)(*count);
}

static int active_sm_count() {
    int active_sms = 0;
    for (int word = 0; word < 4; word++) {
        uint64_t* bm = active_sm_bitmap.lookup(word);
        if (bm) active_sms += __builtin_popcountll(*bm);
    }
    return active_sms;
}

static void print_hints(const ApiTotals& delta, uint64_t sync_ops,
                        uint64_t low_active_total, uint64_t selected_ops,
                        int active_sms) {
    bool any = false;
    printf("    hints: ");
    if (delta.peer > 0) {
        printf("%speer data arrived before launch", any ? "; " : "");
        any = true;
    }
    if (delta.sync > 0) {
        printf("%shost sync before launch", any ? "; " : "");
        any = true;
    }
    if (sync_ops == 0) {
        printf("%sno barrier/membar/atomic/reduction ops observed", any ? "; " : "");
        any = true;
    }
    if (selected_ops > 0 && low_active_total * 5 > selected_ops) {
        printf("%smany low-active warp events", any ? "; " : "");
        any = true;
    }
    if (active_sms <= 1 && selected_ops > 0) {
        printf("%sonly %d active SM(s)", any ? "; " : "", active_sms);
        any = true;
    }
    if (delta.alloc > 0) {
        printf("%sallocation churn before launch", any ? "; " : "");
        any = true;
    }
    if (!any) printf("no obvious sync smell from these counters");
    printf("\n");
}

void nvbit_at_init() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    if (getenv("NVBPF_FORCE_DEVICE_ALLOC") != nullptr) {
        setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    }
    pthread_mutex_init(&debugger_mutex, nullptr);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    if (const char* env = getenv("NVBPF_MEGA_LOW_LANES")) {
        low_active_threshold = (uint32_t)strtoul(env, nullptr, 0);
        if (low_active_threshold > 32) low_active_threshold = 32;
    }
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    trace_api = getenv("NVBPF_TRACE_API") != nullptr;
    trace_memory = getenv("NVBPF_MEGA_MEMORY") != nullptr;
    trace_branches = getenv("NVBPF_MEGA_BRANCH") != nullptr;
    trace_tensor = getenv("NVBPF_MEGA_TENSOR") != nullptr;
    trace_cp_async = getenv("NVBPF_MEGA_NO_CP_ASYNC") == nullptr;
    printf("[NVBPF MEGA] loaded");
    if (!kernel_name_filter.empty()) printf(" filter=%s", kernel_name_filter.c_str());
    printf(" low_lanes<%u", low_active_threshold);
    if (trace_memory || trace_branches || trace_tensor || !trace_cp_async) {
        printf(" modes:");
        if (trace_memory) printf(" memory");
        if (trace_branches) printf(" branch");
        if (trace_tensor) printf(" tensor");
        if (!trace_cp_async) printf(" no_cp_async");
    }
    printf("\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    bool is_launch = nvbpf_is_launch_event(cbid);

    pthread_mutex_lock(&debugger_mutex);
    event_counter++;

    if (!is_launch && is_exit) {
        update_api_totals(ctx, cbid, name, params);
        pthread_mutex_unlock(&debugger_mutex);
        return;
    }
    if (!is_launch) {
        pthread_mutex_unlock(&debugger_mutex);
        return;
    }

    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);
    bool match = csv_or_substring_match(func_name, kernel_name_filter);
    if (!is_exit) {
        if (match) {
            instrument_function_if_needed(ctx, func);
            reset_state();
            nvbit_enable_instrumented(ctx, func, true);
        } else {
            nvbit_enable_instrumented(ctx, func, false);
        }
        pthread_mutex_unlock(&debugger_mutex);
        return;
    }

    if (!match) {
        pthread_mutex_unlock(&debugger_mutex);
        return;
    }

    cudaDeviceSynchronize();

    launch_counter++;
    int gpu = current_device();
    if (gpu >= 0 && gpu < (int)(sizeof(gpu_launches) / sizeof(gpu_launches[0]))) {
        gpu_launches[gpu]++;
    }

    LaunchInfo launch = get_launch_info(cbid, params);
    func_config_t cfg{};
    nvbit_get_func_config(ctx, func, &cfg);
    ApiTotals delta = delta_since_last_launch();
    std::string kernel = compact_kernel_name(func_name);

    uint64_t barriers = read_category(CAT_BARRIER);
    uint64_t membars = read_category(CAT_MEMBAR);
    uint64_t atomics = read_category(CAT_ATOMIC);
    uint64_t reductions = read_category(CAT_REDUCTION);
    uint64_t branches = read_category(CAT_BRANCH);
    uint64_t loads = read_category(CAT_LOAD);
    uint64_t stores = read_category(CAT_STORE);
    uint64_t tensor = read_category(CAT_TENSOR);
    uint64_t cp_async = read_category(CAT_CP_ASYNC);
    uint64_t heartbeat = read_category(CAT_HEARTBEAT);
    uint64_t sync_ops = barriers + membars + atomics + reductions;
    uint64_t selected_ops = sync_ops + branches + loads + stores + tensor + cp_async;
    uint64_t low_active_total = 0;
    for (uint32_t i = 0; i < CAT_HEARTBEAT; i++) {
        low_active_total += read_low_active((DebugCategory)i);
    }
    int active_sms = active_sm_count();
    uint64_t pred_off = *pred_off_events.lookup(0);

    char dyn_smem[32];
    char static_smem[32];
    format_bytes(launch.dynamic_smem, dyn_smem, sizeof(dyn_smem));
    format_bytes(cfg.shmem_static_nbytes, static_smem, sizeof(static_smem));

    printf("[NVBPF MEGA] #%lu GPU%d %s\n", launch_counter, gpu, kernel.c_str());
    printf("    launch: grid=%ux%ux%u block=%ux%ux%u regs=%u dyn_smem=%s active_sms=%d\n",
           launch.gx, launch.gy, launch.gz, launch.bx, launch.by, launch.bz,
           cfg.num_registers, dyn_smem, active_sms);
    print_activity_line(delta);
    printf("    inside: sync_ops=%lu [barrier=%lu membar=%lu atomic=%lu red=%lu] low_active=%lu pred_off=%lu",
           sync_ops, barriers, membars, atomics, reductions, low_active_total, pred_off);
    if (trace_branches) printf(" branch=%lu", branches);
    printf("\n");
    if (trace_memory || trace_tensor || cp_async > 0) {
        printf("    counters:");
        if (trace_memory) {
            printf(" load=%lu store=%lu avg_lanes(load/store)=%.1f/%.1f",
                   loads, stores, avg_active_lanes(CAT_LOAD), avg_active_lanes(CAT_STORE));
        }
        if (trace_tensor) printf(" tensor=%lu", tensor);
        if (cp_async > 0) printf(" cp_async=%lu", cp_async);
        printf("\n");
    }
    print_hints(delta, sync_ops, low_active_total, selected_ops, active_sms);
    if (verbose) {
        printf("    detail: ctx=%p stream=%p static_smem=%s status=%d events=%lu\n",
               (void*)ctx, (void*)launch.stream, static_smem,
               pStatus ? (int)(*pStatus) : 0, event_counter);
        printf("    heartbeat=%lu\n", heartbeat);
        printf("    active_lane_hist:");
        for (int i = 0; i <= 32; i++) {
            uint64_t* val = active_lane_hist.lookup(i);
            if (val && *val > 0) printf(" %d:%lu", i, *val);
        }
        printf("\n");
    }

    last_launch_api_totals = api_totals;
    pthread_mutex_unlock(&debugger_mutex);
}

void nvbit_at_term() {
    uint64_t other_copy = api_totals.copy > api_totals.peer ?
        api_totals.copy - api_totals.peer : 0;
    uint64_t other_copy_bytes = api_totals.copy_bytes > api_totals.peer_bytes ?
        api_totals.copy_bytes - api_totals.peer_bytes : 0;
    char peer_bytes[32];
    char copy_bytes[32];
    char alloc_bytes[32];
    format_bytes(api_totals.peer_bytes, peer_bytes, sizeof(peer_bytes));
    format_bytes(other_copy_bytes, copy_bytes, sizeof(copy_bytes));
    format_bytes(api_totals.alloc_bytes, alloc_bytes, sizeof(alloc_bytes));

    printf("[NVBPF MEGA] summary: launches=%lu", launch_counter);
    for (size_t i = 0; i < sizeof(gpu_launches) / sizeof(gpu_launches[0]); i++) {
        if (gpu_launches[i] != 0) printf(" gpu%zu=%lu", i, gpu_launches[i]);
    }
    printf("\n");
    printf("[NVBPF MEGA] api: peer_copy=%lu (%s), other_copy=%lu (%s), sync=%lu, alloc=%lu (%s)\n",
           api_totals.peer, peer_bytes, other_copy, copy_bytes,
           api_totals.sync, api_totals.alloc, alloc_bytes);
}
