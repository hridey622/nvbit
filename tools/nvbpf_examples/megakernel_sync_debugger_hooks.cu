/*
 * NV-BPF Example: Megakernel Sync Debugger - Device Hooks
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "utils/utils.h"

static constexpr uint32_t kNvbpfMaxSms = 256;
static constexpr uint32_t kBitmapWords = 4;

extern "C" __device__ __noinline__ void msd_count_category(int pred,
                                                           uint64_t pcounts,
                                                           uint64_t pactive_sum,
                                                           uint64_t plow_active,
                                                           uint64_t ppred_off,
                                                           uint64_t phist,
                                                           uint64_t psm_events,
                                                           uint64_t pbitmap,
                                                           uint32_t category,
                                                           uint32_t threshold) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    if (first_laneid != laneid) return;

    const int active_threads = __popc(predicate_mask);
    uint64_t* counts = (uint64_t*)pcounts;
    uint64_t* active_sum = (uint64_t*)pactive_sum;
    uint64_t* low_active = (uint64_t*)plow_active;
    uint64_t* pred_off = (uint64_t*)ppred_off;
    uint64_t* hist = (uint64_t*)phist;

    atomicAdd((unsigned long long*)&counts[category], 1ULL);
    if (active_threads >= 0 && active_threads <= 32) {
        atomicAdd((unsigned long long*)&hist[active_threads], 1ULL);
    }
    if (active_threads == 0) {
        atomicAdd((unsigned long long*)pred_off, 1ULL);
        return;
    }

    atomicAdd((unsigned long long*)&active_sum[category], (unsigned long long)active_threads);
    if ((uint32_t)active_threads < threshold) {
        atomicAdd((unsigned long long*)&low_active[category], 1ULL);
    }

    uint32_t sm = bpf_get_current_sm_id();
    if (sm >= kNvbpfMaxSms) return;
    uint64_t* sm_events = (uint64_t*)psm_events;
    atomicAdd((unsigned long long*)&sm_events[sm], 1ULL);
    uint32_t word = sm / 64;
    if (word < kBitmapWords) {
        uint64_t* bitmap = (uint64_t*)pbitmap;
        atomicOr((unsigned long long*)&bitmap[word], 1ULL << (sm % 64));
    }
}
