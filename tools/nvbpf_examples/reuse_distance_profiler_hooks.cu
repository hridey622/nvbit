/*
 * NV-BPF Example: Reuse Distance Profiler - Device Hooks
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "utils/utils.h"

struct ReuseEntry {
    uint64_t tag;
    uint64_t seq;
    uint32_t warp_key;
    uint8_t kind;
    uint8_t valid;
    uint16_t _pad;
};

extern "C" __device__ __noinline__ void rd_trace_mem(int pred,
                                                     uint64_t addr,
                                                     uint64_t pentries,
                                                     uint64_t pwarp_seq,
                                                     uint64_t psampled,
                                                     uint64_t phits,
                                                     uint64_t pgap_sum,
                                                     uint64_t pgap_max,
                                                     uint64_t phist,
                                                     uint32_t sample_every,
                                                     uint32_t line_shift,
                                                     uint32_t slot_mask,
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

    uint64_t line_tag = addr >> line_shift;
    uint64_t sample_key = line_tag ^ (uint64_t)sm ^ ((uint64_t)warp << 11);
    if (sample_every > 1 && (sample_key % sample_every) != 0) return;

    uint64_t* warp_seq = (uint64_t*)pwarp_seq;
    uint64_t seq = atomicAdd((unsigned long long*)&warp_seq[warp_key], 1ULL) + 1ULL;

    uint64_t* sampled = (uint64_t*)psampled;
    uint64_t* hits = (uint64_t*)phits;
    uint64_t* gap_sum = (uint64_t*)pgap_sum;
    uint64_t* gap_max = (uint64_t*)pgap_max;
    uint64_t* hist = (uint64_t*)phist;
    ReuseEntry* entries = (ReuseEntry*)pentries;

    atomicAdd((unsigned long long*)&sampled[kind], 1ULL);

    uint32_t slot = (uint32_t)(sample_key & (uint64_t)slot_mask);
    ReuseEntry* entry = &entries[slot];

    if (entry->valid && entry->warp_key == warp_key &&
        entry->tag == line_tag && entry->kind == (uint8_t)kind) {
        uint64_t gap = seq - entry->seq;
        atomicAdd((unsigned long long*)&hits[kind], 1ULL);
        atomicAdd((unsigned long long*)&gap_sum[0], (unsigned long long)gap);
        atomicMax((unsigned long long*)&gap_max[0], (unsigned long long)gap);
        if (gap <= 1) {
            atomicAdd((unsigned long long*)&hist[1], 1ULL);
        } else if (gap <= 4) {
            atomicAdd((unsigned long long*)&hist[2], 1ULL);
        } else if (gap <= 16) {
            atomicAdd((unsigned long long*)&hist[3], 1ULL);
        } else if (gap <= 64) {
            atomicAdd((unsigned long long*)&hist[4], 1ULL);
        } else {
            atomicAdd((unsigned long long*)&hist[5], 1ULL);
        }
    } else {
        atomicAdd((unsigned long long*)&hist[0], 1ULL);
    }

    entry->tag = line_tag;
    entry->seq = seq;
    entry->warp_key = warp_key;
    entry->kind = (uint8_t)kind;
    entry->valid = 1;
}
