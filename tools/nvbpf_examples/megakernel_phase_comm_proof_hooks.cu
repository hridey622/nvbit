/*
 * NV-BPF Example: Megakernel Phase/Communication Proof - Device Hooks
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "utils/utils.h"

enum ProofPhase {
    PHASE_ENTRY = 0,
    PHASE_TMA = 1,
    PHASE_TMEM = 2,
    PHASE_TENSOR = 3,
    PHASE_SCALE = 4,
    PHASE_MULTICAST = 5,
    PHASE_SYNC = 6,
    PHASE_PRODUCER = 7,
    PHASE_HANDOFF = 8,
    PHASE_CONSUMER = 9,
    PHASE_EPILOGUE = 10,
    PHASE_BRANCH = 11,
    PHASE_COUNT = 12,
};

static constexpr uint32_t kNvbpfMaxSms = 256;
static constexpr uint32_t kBitmapWords = 4;

extern "C" __device__ __noinline__ void mpcp_record_phase(int pred,
                                                           uint64_t pcounts,
                                                           uint64_t pfirst_clock,
                                                           uint64_t plast_clock,
                                                           uint64_t pactive_sum,
                                                           uint64_t plow_active,
                                                           uint64_t psm_bitmap,
                                                           uint32_t phase,
                                                           uint32_t threshold) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    if (first_laneid != laneid) return;

    const int active_threads = __popc(predicate_mask);
    if (active_threads == 0 || phase >= PHASE_COUNT) return;

    uint64_t* counts = (uint64_t*)pcounts;
    uint64_t* first_clock = (uint64_t*)pfirst_clock;
    uint64_t* last_clock = (uint64_t*)plast_clock;
    uint64_t* active_sum = (uint64_t*)pactive_sum;
    uint64_t* low_active = (uint64_t*)plow_active;

    uint64_t now = clock64();
    atomicAdd((unsigned long long*)&counts[phase], 1ULL);
    atomicCAS((unsigned long long*)&first_clock[phase], 0ULL, (unsigned long long)now);
    atomicMax((unsigned long long*)&last_clock[phase], (unsigned long long)now);
    atomicAdd((unsigned long long*)&active_sum[phase], (unsigned long long)active_threads);
    if ((uint32_t)active_threads < threshold) {
        atomicAdd((unsigned long long*)&low_active[phase], 1ULL);
    }

    uint32_t sm = bpf_get_current_sm_id();
    if (sm >= kNvbpfMaxSms) return;
    uint32_t word = sm / 64;
    if (word < kBitmapWords) {
        uint64_t* bitmap = (uint64_t*)psm_bitmap;
        atomicOr((unsigned long long*)&bitmap[phase * kBitmapWords + word],
                 1ULL << (sm % 64));
    }
}
