/*
 * NV-BPF Example: Tile Lifetime Tracker - Device Hooks
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "utils/utils.h"

struct TileState {
    uint64_t start_seq;
    uint64_t math_count;
    uint8_t active;
    uint8_t _pad[7];
};

extern "C" __device__ __noinline__ void tlt_trace_event(int pred,
                                                        uint64_t psegments,
                                                        uint64_t plifetime_sum,
                                                        uint64_t plifetime_max,
                                                        uint64_t pmath_sum,
                                                        uint64_t phist,
                                                        uint64_t pwarp_seq,
                                                        uint64_t pstates,
                                                        uint32_t kind) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int active_threads = __popc(predicate_mask);
    if (first_laneid != laneid || active_threads == 0) return;

    uint32_t sm = bpf_get_current_sm_id();
    uint32_t warp = bpf_get_current_warp_id();
    uint32_t warp_key = sm * 64u + warp;
    if (warp_key >= 256u * 64u) return;

    uint64_t* warp_seq = (uint64_t*)pwarp_seq;
    TileState* states = (TileState*)pstates;
    uint64_t seq = atomicAdd((unsigned long long*)&warp_seq[warp_key], 1ULL) + 1ULL;
    TileState* state = &states[warp_key];

    if (kind == 0) {
        if (!state->active) {
            state->active = 1;
            state->start_seq = seq;
            state->math_count = 0;
        }
        return;
    }

    if (!state->active) return;

    if (kind == 1) {
        state->math_count++;
        return;
    }

    uint64_t lifetime = seq >= state->start_seq ? (seq - state->start_seq + 1ULL) : 1ULL;
    uint64_t* segments = (uint64_t*)psegments;
    uint64_t* lifetime_sum = (uint64_t*)plifetime_sum;
    uint64_t* lifetime_max = (uint64_t*)plifetime_max;
    uint64_t* math_sum = (uint64_t*)pmath_sum;
    uint64_t* hist = (uint64_t*)phist;

    atomicAdd((unsigned long long*)&segments[0], 1ULL);
    atomicAdd((unsigned long long*)&lifetime_sum[0], (unsigned long long)lifetime);
    atomicAdd((unsigned long long*)&math_sum[0], (unsigned long long)state->math_count);
    atomicMax((unsigned long long*)&lifetime_max[0], (unsigned long long)lifetime);

    if (lifetime <= 4) {
        atomicAdd((unsigned long long*)&hist[0], 1ULL);
    } else if (lifetime <= 16) {
        atomicAdd((unsigned long long*)&hist[1], 1ULL);
    } else if (lifetime <= 64) {
        atomicAdd((unsigned long long*)&hist[2], 1ULL);
    } else if (lifetime <= 256) {
        atomicAdd((unsigned long long*)&hist[3], 1ULL);
    } else {
        atomicAdd((unsigned long long*)&hist[4], 1ULL);
    }

    state->active = 0;
    state->start_seq = 0;
    state->math_count = 0;
}
