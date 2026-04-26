/*
 * NV-BPF Example: CTA Role Classifier - Device Hooks
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "utils/utils.h"

static constexpr int kCtaProfileSlots = 4096;

struct CtaProfile {
    uint64_t linear_id;
    uint32_t math_sites;
    uint32_t mem_sites;
    uint32_t branch_sites;
    uint16_t cta_x;
    uint16_t cta_y;
    uint16_t cta_z;
    uint8_t edge;
    uint8_t valid;
    uint8_t _pad[6];
};

static __device__ __forceinline__ uint64_t linear_cta_id() {
    return (uint64_t)blockIdx.x +
           (uint64_t)gridDim.x *
               ((uint64_t)blockIdx.y + (uint64_t)gridDim.y * (uint64_t)blockIdx.z);
}

static __device__ __forceinline__ bool is_edge_cta() {
    return blockIdx.x == 0 || blockIdx.y == 0 || blockIdx.z == 0 ||
           blockIdx.x + 1 == gridDim.x ||
           blockIdx.y + 1 == gridDim.y ||
           blockIdx.z + 1 == gridDim.z;
}

extern "C" __device__ __noinline__ void crc_mark_cta(int pred,
                                                     uint64_t pprofiles,
                                                     uint64_t pdropped) {
    if (!pred) return;
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) return;

    uint64_t linear = linear_cta_id();
    uint32_t slot = (uint32_t)(linear & (kCtaProfileSlots - 1));
    CtaProfile* profiles = (CtaProfile*)pprofiles;
    CtaProfile* profile = &profiles[slot];
    if (!profile->valid || profile->linear_id == linear) {
        if (!profile->valid) {
            profile->linear_id = linear;
            profile->cta_x = (uint16_t)blockIdx.x;
            profile->cta_y = (uint16_t)blockIdx.y;
            profile->cta_z = (uint16_t)blockIdx.z;
            profile->edge = is_edge_cta() ? 1 : 0;
            profile->math_sites = 0;
            profile->mem_sites = 0;
            profile->branch_sites = 0;
            profile->valid = 1;
        }
        return;
    }
    uint64_t* dropped = (uint64_t*)pdropped;
    atomicAdd((unsigned long long*)&dropped[0], 1ULL);
}

extern "C" __device__ __noinline__ void crc_trace_site(int pred,
                                                       uint64_t pprofiles,
                                                       uint64_t pdropped,
                                                       uint32_t kind) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int active_threads = __popc(predicate_mask);
    if (first_laneid != laneid || active_threads == 0) return;

    uint64_t linear = linear_cta_id();
    uint32_t slot = (uint32_t)(linear & (kCtaProfileSlots - 1));
    CtaProfile* profiles = (CtaProfile*)pprofiles;
    CtaProfile* profile = &profiles[slot];
    if (!profile->valid || profile->linear_id != linear) {
        uint64_t* dropped = (uint64_t*)pdropped;
        atomicAdd((unsigned long long*)&dropped[0], 1ULL);
        return;
    }

    if (kind == 0) {
        atomicAdd((unsigned int*)&profile->mem_sites, 1U);
    } else if (kind == 1) {
        atomicAdd((unsigned int*)&profile->math_sites, 1U);
    } else {
        atomicAdd((unsigned int*)&profile->branch_sites, 1U);
    }
}
