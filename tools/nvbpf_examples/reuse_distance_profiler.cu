/*
 * NV-BPF Example: Reuse Distance Profiler
 *
 * Sampled per-warp memory-line reuse estimator. This is an approximation:
 * it tracks sampled memory lines inside a warp-local direct-mapped table and
 * measures how many sampled memory references elapse before the same line is
 * seen again.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_set>
#include <vector>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

static constexpr int kWarpStates = 256 * 64;
static constexpr int kReuseSlots = 8192;
static constexpr int kReuseHistBuckets = 6;

enum ReuseKind {
    RK_LOAD = 0,
    RK_STORE = 1,
    RK_COUNT = 2,
};

struct ReuseEntry {
    uint64_t tag;
    uint64_t seq;
    uint32_t warp_key;
    uint8_t kind;
    uint8_t valid;
    uint16_t _pad;
};

struct ReuseSummary {
    std::string kernel_name;
    uint64_t launches = 0;
    uint64_t sampled[RK_COUNT] = {};
    uint64_t hits[RK_COUNT] = {};
    uint64_t gap_sum = 0;
    uint64_t gap_max = 0;
    uint64_t hist[kReuseHistBuckets] = {};
};

BPF_ARRAY(rd_sampled_by_kind, uint64_t, RK_COUNT);
BPF_ARRAY(rd_hits_by_kind, uint64_t, RK_COUNT);
BPF_ARRAY(rd_gap_sum, uint64_t, 1);
BPF_ARRAY(rd_gap_max, uint64_t, 1);
BPF_ARRAY(rd_hist, uint64_t, kReuseHistBuckets);
BPF_ARRAY(rd_entries, ReuseEntry, kReuseSlots);
BPF_ARRAY(rd_warp_seq, uint64_t, kWarpStates);

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
                                                     uint32_t kind);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static std::vector<ReuseSummary> summaries;
static std::string kernel_name_filter;
static uint32_t sample_every = 32;
static uint32_t line_shift = 7;
static bool verbose = false;
static bool full_names = false;
static uint64_t matched_launches = 0;

static uint64_t parse_u64_env(const char* name, uint64_t def) {
    const char* env = getenv(name);
    if (!env || !*env) return def;
    return strtoull(env, nullptr, 0);
}

static std::string compact_kernel_name(const std::string& raw) {
    if (full_names) return raw;
    std::string name = raw;
    if (name.rfind("void ", 0) == 0) name = name.substr(5);
    size_t paren = name.find('(');
    if (paren != std::string::npos) name = name.substr(0, paren);
    if (name.size() <= 56) return name;
    return name.substr(0, 24) + "..." + name.substr(name.size() - 24);
}

static ReuseSummary* find_summary(const char* func_name) {
    for (auto& summary : summaries) {
        if (summary.kernel_name == func_name) return &summary;
    }
    return nullptr;
}

static const char* bucket_name(int idx) {
    switch (idx) {
        case 0: return "miss";
        case 1: return "h1";
        case 2: return "h4";
        case 3: return "h16";
        case 4: return "h64";
        default: return "hfar";
    }
}

static void reset_state() {
    rd_sampled_by_kind.reset();
    rd_hits_by_kind.reset();
    rd_gap_sum.reset();
    rd_gap_max.reset();
    rd_hist.reset();
    rd_entries.reset();
    rd_warp_seq.reset();
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            if (instr->isLoad() &&
                instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) {
                nvbit_insert_call(instr, "rd_trace_mem", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_mref_addr64(instr, 0);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_entries.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_warp_seq.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_sampled_by_kind.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_hits_by_kind.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_gap_sum.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_gap_max.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_hist.data[0]);
                nvbit_add_call_arg_const_val32(instr, sample_every);
                nvbit_add_call_arg_const_val32(instr, line_shift);
                nvbit_add_call_arg_const_val32(instr, kReuseSlots - 1);
                nvbit_add_call_arg_const_val32(instr, RK_LOAD);
            }
            if (instr->isStore()) {
                nvbit_insert_call(instr, "rd_trace_mem", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_mref_addr64(instr, 0);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_entries.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_warp_seq.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_sampled_by_kind.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_hits_by_kind.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_gap_sum.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_gap_max.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&rd_hist.data[0]);
                nvbit_add_call_arg_const_val32(instr, sample_every);
                nvbit_add_call_arg_const_val32(instr, line_shift);
                nvbit_add_call_arg_const_val32(instr, kReuseSlots - 1);
                nvbit_add_call_arg_const_val32(instr, RK_STORE);
            }
        }
    }
}

static double pct(uint64_t num, uint64_t den) {
    return den ? (100.0 * (double)num / (double)den) : 0.0;
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    sample_every = (uint32_t)parse_u64_env("NVBPF_REUSE_SAMPLE_EVERY", 32);
    if (sample_every == 0) sample_every = 1;
    line_shift = (uint32_t)parse_u64_env("NVBPF_REUSE_LINE_SHIFT", 7);
    if (line_shift > 20) line_shift = 20;
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF REUSE_DISTANCE_PROFILER] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!nvbpf_is_launch_event(cbid)) return;
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);

    if (!is_exit) {
        bool match = kernel_name_filter.empty() ||
                     strstr(func_name, kernel_name_filter.c_str()) != nullptr;
        pthread_mutex_lock(&launch_mutex);
        if (match) {
            instrument_function_if_needed(ctx, func);
            reset_state();
            nvbit_enable_instrumented(ctx, func, true);
        } else {
            nvbit_enable_instrumented(ctx, func, false);
        }
        if (!match) pthread_mutex_unlock(&launch_mutex);
    } else {
        cudaDeviceSynchronize();
        if (!kernel_name_filter.empty() &&
            strstr(func_name, kernel_name_filter.c_str()) == nullptr) {
            return;
        }
        matched_launches++;

        uint64_t sampled_load = *rd_sampled_by_kind.lookup(RK_LOAD);
        uint64_t sampled_store = *rd_sampled_by_kind.lookup(RK_STORE);
        uint64_t hits_load = *rd_hits_by_kind.lookup(RK_LOAD);
        uint64_t hits_store = *rd_hits_by_kind.lookup(RK_STORE);
        uint64_t total_sampled = sampled_load + sampled_store;
        uint64_t total_hits = hits_load + hits_store;
        uint64_t gap_sum = *rd_gap_sum.lookup(0);
        uint64_t gap_max = *rd_gap_max.lookup(0);
        uint64_t hist[kReuseHistBuckets] = {};
        for (int i = 0; i < kReuseHistBuckets; i++) {
            uint64_t* value = rd_hist.lookup(i);
            hist[i] = value ? *value : 0;
        }
        double avg_gap = total_hits ? (double)gap_sum / (double)total_hits : 0.0;

        if (verbose) {
            printf("[NVBPF] reuse_distance kernel=%s\n", func_name);
            printf("        sampled: loads=%lu stores=%lu total=%lu sample_every=%u line_shift=%u\n",
                   sampled_load, sampled_store, total_sampled, sample_every, line_shift);
            printf("        reuse: hits=%lu hit_rate=%.2f%% avg_gap=%.2f max_gap=%lu\n",
                   total_hits, pct(total_hits, total_sampled), avg_gap, gap_max);
            printf("        hist:");
            for (int i = 0; i < kReuseHistBuckets; i++) {
                printf(" %s=%lu", bucket_name(i), hist[i]);
            }
            printf("\n");
        } else {
            ReuseSummary* summary = find_summary(func_name);
            if (summary == nullptr) {
                ReuseSummary fresh{};
                fresh.kernel_name = func_name;
                summaries.push_back(fresh);
                summary = &summaries.back();
            }
            summary->launches++;
            summary->sampled[RK_LOAD] += sampled_load;
            summary->sampled[RK_STORE] += sampled_store;
            summary->hits[RK_LOAD] += hits_load;
            summary->hits[RK_STORE] += hits_store;
            summary->gap_sum += gap_sum;
            if (gap_max > summary->gap_max) summary->gap_max = gap_max;
            for (int i = 0; i < kReuseHistBuckets; i++) {
                summary->hist[i] += hist[i];
            }
        }
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    if (!verbose) {
        printf("[NVBPF REUSE_DISTANCE_PROFILER] matched_launches=%lu unique_kernels=%zu sample_every=%u line_shift=%u\n",
               matched_launches, summaries.size(), sample_every, line_shift);
        for (const auto& summary : summaries) {
            uint64_t total_sampled = summary.sampled[RK_LOAD] + summary.sampled[RK_STORE];
            uint64_t total_hits = summary.hits[RK_LOAD] + summary.hits[RK_STORE];
            double avg_gap = total_hits ? (double)summary.gap_sum / (double)total_hits : 0.0;
            printf("  x%-3lu %-32s | sampled=%-6lu hit=%5.2f%% avg_gap=%6.2f max=%-5lu | miss=%-5lu h1=%-5lu h4=%-5lu h16=%-5lu h64=%-5lu hfar=%-5lu\n",
                   summary.launches,
                   compact_kernel_name(summary.kernel_name).c_str(),
                   total_sampled,
                   pct(total_hits, total_sampled),
                   avg_gap,
                   summary.gap_max,
                   summary.hist[0],
                   summary.hist[1],
                   summary.hist[2],
                   summary.hist[3],
                   summary.hist[4],
                   summary.hist[5]);
        }
    }
    printf("[NVBPF REUSE_DISTANCE_PROFILER] Tool terminated\n");
}
