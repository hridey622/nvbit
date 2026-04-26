/*
 * NV-BPF Example: NVLink Trace
 *
 * Host-side topology and peer-copy correlation trace.
 * This is topology/API aware, not a raw NVLink hardware-counter tool.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

static bool topology_printed = false;
static uint64_t api_event_counter = 0;
static std::string kernel_name_filter;

struct RecentPeerCopy {
    uint64_t event_id = 0;
    CUdevice src_dev = 0;
    CUdevice dst_dev = 0;
    size_t bytes = 0;
    bool valid = false;
} recent_peer_copy;

static CUdevice device_for_context(CUcontext target_ctx) {
    if (target_ctx == nullptr) return (CUdevice)-1;

    CUcontext pushed = nullptr;
    CUdevice dev = (CUdevice)-1;
    if (cuCtxPushCurrent(target_ctx) == CUDA_SUCCESS) {
        cuCtxGetDevice(&dev);
        cuCtxPopCurrent(&pushed);
    }
    return dev;
}

static CUdevice device_for_pointer(CUdeviceptr ptr) {
    if (ptr == 0) return (CUdevice)-1;

    CUdevice dev = (CUdevice)-1;
    if (cuPointerGetAttribute(&dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr) == CUDA_SUCCESS) {
        return dev;
    }

    CUcontext ptr_ctx = nullptr;
    if (cuPointerGetAttribute(&ptr_ctx, CU_POINTER_ATTRIBUTE_CONTEXT, ptr) == CUDA_SUCCESS) {
        return device_for_context(ptr_ctx);
    }
    return dev;
}

static CUdevice device_for_context_or_pointer(CUcontext target_ctx, CUdeviceptr ptr) {
    CUdevice dev = device_for_context(target_ctx);
    if (dev != (CUdevice)-1) return dev;
    return device_for_pointer(ptr);
}

static void print_topology_once() {
    if (topology_printed) return;
    topology_printed = true;

    int count = 0;
    cuDeviceGetCount(&count);
    printf("[NVBPF NVLINK_TRACE] Device topology:\n");
    for (int i = 0; i < count; i++) {
        char name[128] = {};
        cuDeviceGetName(name, sizeof(name), i);
        printf("  GPU %d: %s\n", i, name);
    }
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < count; j++) {
            if (i == j) continue;
            int can = 0;
            cuDeviceCanAccessPeer(&can, i, j);
            if (can) {
                printf("  peer_access: GPU %d -> GPU %d enabled_capable=1\n", i, j);
            }
        }
    }
}

static bool is_launch_event_id(nvbit_api_cuda_t cbid) {
    return nvbpf_is_launch_event(cbid);
}

static const char* classify_framework(const char* func_name) {
    if (strstr(func_name, "nccl") || strstr(func_name, "allreduce") ||
        strstr(func_name, "reduce_scatter") || strstr(func_name, "all_gather")) {
        return "nccl";
    }
    if (strstr(func_name, "triton")) return "triton";
    if (strstr(func_name, "cutlass")) return "cutlass";
    if (strstr(func_name, "cublas") || strstr(func_name, "sgemm") ||
        strstr(func_name, "gemm")) {
        return "gemm";
    }
    if (strstr(func_name, "aten") || strstr(func_name, "at::")) return "pytorch";
    if (strstr(func_name, "flash") || strstr(func_name, "attention") ||
        strstr(func_name, "attn")) {
        return "attention";
    }
    return "other";
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    printf("[NVBPF NVLINK_TRACE] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!is_exit) return;
    print_topology_once();
    api_event_counter++;

    if (cbid == API_CUDA_cuMemcpyPeer || cbid == API_CUDA_cuMemcpyPeer_ptds) {
        CUcontext src_ctx = nullptr;
        CUcontext dst_ctx = nullptr;
        CUdeviceptr src_ptr = 0;
        CUdeviceptr dst_ptr = 0;
        size_t bytes = 0;
        if (cbid == API_CUDA_cuMemcpyPeer) {
            cuMemcpyPeer_params* p = (cuMemcpyPeer_params*)params;
            src_ctx = p->srcContext;
            dst_ctx = p->dstContext;
            src_ptr = p->srcDevice;
            dst_ptr = p->dstDevice;
            bytes = p->ByteCount;
        } else {
            cuMemcpyPeer_ptds_params* p = (cuMemcpyPeer_ptds_params*)params;
            src_ctx = p->srcContext;
            dst_ctx = p->dstContext;
            src_ptr = p->srcDevice;
            dst_ptr = p->dstDevice;
            bytes = p->ByteCount;
        }
        recent_peer_copy.src_dev = device_for_context_or_pointer(src_ctx, src_ptr);
        recent_peer_copy.dst_dev = device_for_context_or_pointer(dst_ctx, dst_ptr);
        recent_peer_copy.bytes = bytes;
        recent_peer_copy.event_id = api_event_counter;
        recent_peer_copy.valid = true;
        printf("[NVBPF] peer_copy bytes=%zu src_gpu=%d dst_gpu=%d sync\n",
               bytes, (int)recent_peer_copy.src_dev, (int)recent_peer_copy.dst_dev);
        return;
    }
    if (cbid == API_CUDA_cuMemcpyPeerAsync ||
        cbid == API_CUDA_cuMemcpyPeerAsync_ptsz) {
        CUcontext src_ctx = nullptr;
        CUcontext dst_ctx = nullptr;
        CUdeviceptr src_ptr = 0;
        CUdeviceptr dst_ptr = 0;
        size_t bytes = 0;
        if (cbid == API_CUDA_cuMemcpyPeerAsync) {
            cuMemcpyPeerAsync_params* p = (cuMemcpyPeerAsync_params*)params;
            src_ctx = p->srcContext;
            dst_ctx = p->dstContext;
            src_ptr = p->srcDevice;
            dst_ptr = p->dstDevice;
            bytes = p->ByteCount;
        } else {
            cuMemcpyPeerAsync_ptsz_params* p = (cuMemcpyPeerAsync_ptsz_params*)params;
            src_ctx = p->srcContext;
            dst_ctx = p->dstContext;
            src_ptr = p->srcDevice;
            dst_ptr = p->dstDevice;
            bytes = p->ByteCount;
        }
        recent_peer_copy.src_dev = device_for_context_or_pointer(src_ctx, src_ptr);
        recent_peer_copy.dst_dev = device_for_context_or_pointer(dst_ctx, dst_ptr);
        recent_peer_copy.bytes = bytes;
        recent_peer_copy.event_id = api_event_counter;
        recent_peer_copy.valid = true;
        printf("[NVBPF] peer_copy bytes=%zu src_gpu=%d dst_gpu=%d async\n",
               bytes, (int)recent_peer_copy.src_dev, (int)recent_peer_copy.dst_dev);
        return;
    }
    if (cbid == API_CUDA_cuMemcpyDtoD_v2 ||
        cbid == API_CUDA_cuMemcpyDtoDAsync_v2 ||
        cbid == API_CUDA_cuMemcpyDtoD_v2_ptds ||
        cbid == API_CUDA_cuMemcpyDtoDAsync_v2_ptsz) {
        size_t bytes = 0;
        if (cbid == API_CUDA_cuMemcpyDtoD_v2) {
            bytes = ((cuMemcpyDtoD_v2_params*)params)->ByteCount;
        } else if (cbid == API_CUDA_cuMemcpyDtoDAsync_v2) {
            bytes = ((cuMemcpyDtoDAsync_v2_params*)params)->ByteCount;
        } else if (cbid == API_CUDA_cuMemcpyDtoD_v2_ptds) {
            bytes = ((cuMemcpyDtoD_v2_ptds_params*)params)->ByteCount;
        } else {
            bytes = ((cuMemcpyDtoDAsync_v2_ptsz_params*)params)->ByteCount;
        }
        printf("[NVBPF] device_to_device_copy bytes=%zu\n", bytes);
        return;
    }

    if (is_launch_event_id(cbid)) {
        CUfunction func = nvbpf_get_launch_func(cbid, params);
        const char* func_name = nvbit_get_func_name(ctx, func);
        if (!kernel_name_filter.empty() &&
            strstr(func_name, kernel_name_filter.c_str()) == nullptr) {
            return;
        }

        CUdevice dev = 0;
        cuCtxGetDevice(&dev);
        printf("[NVBPF] kernel_launch gpu=%d framework=%s kernel=%s\n",
               (int)dev, classify_framework(func_name), func_name);
        if (recent_peer_copy.valid && api_event_counter - recent_peer_copy.event_id <= 8) {
            printf("        recent_peer_copy bytes=%zu src_gpu=%d dst_gpu=%d delta_events=%lu\n",
                   recent_peer_copy.bytes, (int)recent_peer_copy.src_dev,
                   (int)recent_peer_copy.dst_dev,
                   api_event_counter - recent_peer_copy.event_id);
        }
        if (strstr(func_name, "nccl") || strstr(func_name, "allreduce")) {
            printf("        kernel_name suggests communication activity\n");
        }
    }
}

void nvbit_at_term() {
    printf("[NVBPF NVLINK_TRACE] Tool terminated\n");
}
