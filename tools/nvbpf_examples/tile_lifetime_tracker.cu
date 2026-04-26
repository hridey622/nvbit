/*
 * NV-BPF Example: Tile Lifetime Tracker
 *
 * Approximate warp-local "tile lifetime" estimator. It measures the number of
 * relevant warp-leader instruction events between the first producer-like load
 * and the first output store, while also counting how many math/tensor
 * instructions happened during that interval.
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
static constexpr int kLifetimeBuckets = 5;

enum TileEventKind {
    TE_PRODUCER = 0,
    TE_MATH = 1,
    TE_STORE = 2,
};

struct TileState {
    uint64_t start_seq;
    uint64_t math_count;
    uint8_t active;
    uint8_t _pad[7];
};

struct TileSummary {
    std::string kernel_name;
    uint64_t launches = 0;
    uint64_t segments = 0;
    uint64_t lifetime_sum = 0;
    uint64_t lifetime_max = 0;
    uint64_t math_sum = 0;
    uint64_t hist[kLifetimeBuckets] = {};
};

BPF_ARRAY(tlt_segments, uint64_t, 1);
BPF_ARRAY(tlt_lifetime_sum, uint64_t, 1);
BPF_ARRAY(tlt_lifetime_max, uint64_t, 1);
BPF_ARRAY(tlt_math_sum, uint64_t, 1);
BPF_ARRAY(tlt_hist, uint64_t, kLifetimeBuckets);
BPF_ARRAY(tlt_warp_seq, uint64_t, kWarpStates);
BPF_ARRAY(tlt_states, TileState, kWarpStates);

extern "C" __device__ __noinline__ void tlt_trace_event(int pred,
                                                        uint64_t psegments,
                                                        uint64_t plifetime_sum,
                                                        uint64_t plifetime_max,
                                                        uint64_t pmath_sum,
                                                        uint64_t phist,
                                                        uint64_t pwarp_seq,
                                                        uint64_t pstates,
                                                        uint32_t kind);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static std::vector<TileSummary> summaries;
static std::string kernel_name_filter;
static bool verbose = false;
static bool full_names = false;
static uint64_t matched_launches = 0;

static bool opcode_starts_with(const char* opcode, const char* prefix) {
    return strncmp(opcode, prefix, strlen(prefix)) == 0;
}

static bool is_tensor_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "HMMA") ||
           opcode_starts_with(opcode, "MMA") ||
           opcode_starts_with(opcode, "WGMMA") ||
           opcode_starts_with(opcode, "BMMA");
}

static bool is_math_opcode(const char* opcode) {
    return is_tensor_opcode(opcode) ||
           opcode_starts_with(opcode, "FFMA") ||
           opcode_starts_with(opcode, "HFMA2") ||
           opcode_starts_with(opcode, "FADD") ||
           opcode_starts_with(opcode, "FMUL") ||
           opcode_starts_with(opcode, "IMAD") ||
           opcode_starts_with(opcode, "IADD3");
}

static bool is_producer_opcode(Instr* instr, const char* opcode) {
    return (instr->isLoad() &&
            instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) ||
           opcode_starts_with(opcode, "CPASYNC") ||
           opcode_starts_with(opcode, "LDGSTS") ||
           opcode_starts_with(opcode, "LDMATRIX");
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

static TileSummary* find_summary(const char* func_name) {
    for (auto& summary : summaries) {
        if (summary.kernel_name == func_name) return &summary;
    }
    return nullptr;
}

static void reset_state() {
    tlt_segments.reset();
    tlt_lifetime_sum.reset();
    tlt_lifetime_max.reset();
    tlt_math_sum.reset();
    tlt_hist.reset();
    tlt_warp_seq.reset();
    tlt_states.reset();
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            const char* opcode = instr->getOpcodeShort();
            if (is_producer_opcode(instr, opcode)) {
                nvbit_insert_call(instr, "tlt_trace_event", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_segments.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_lifetime_sum.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_lifetime_max.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_math_sum.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_hist.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_warp_seq.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_states.data[0]);
                nvbit_add_call_arg_const_val32(instr, TE_PRODUCER);
            } else if (is_math_opcode(opcode)) {
                nvbit_insert_call(instr, "tlt_trace_event", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_segments.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_lifetime_sum.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_lifetime_max.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_math_sum.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_hist.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_warp_seq.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_states.data[0]);
                nvbit_add_call_arg_const_val32(instr, TE_MATH);
            } else if (instr->isStore()) {
                nvbit_insert_call(instr, "tlt_trace_event", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_segments.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_lifetime_sum.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_lifetime_max.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_math_sum.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_hist.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_warp_seq.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tlt_states.data[0]);
                nvbit_add_call_arg_const_val32(instr, TE_STORE);
            }
        }
    }
}

static const char* bucket_name(int idx) {
    switch (idx) {
        case 0: return "t4";
        case 1: return "t16";
        case 2: return "t64";
        case 3: return "t256";
        default: return "tlong";
    }
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF TILE_LIFETIME_TRACKER] Tool loaded\n");
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

        uint64_t segments = *tlt_segments.lookup(0);
        uint64_t lifetime_sum = *tlt_lifetime_sum.lookup(0);
        uint64_t lifetime_max = *tlt_lifetime_max.lookup(0);
        uint64_t math_sum = *tlt_math_sum.lookup(0);
        uint64_t hist[kLifetimeBuckets] = {};
        for (int i = 0; i < kLifetimeBuckets; i++) {
            uint64_t* value = tlt_hist.lookup(i);
            hist[i] = value ? *value : 0;
        }
        double avg_lifetime = segments ? (double)lifetime_sum / (double)segments : 0.0;
        double avg_math = segments ? (double)math_sum / (double)segments : 0.0;

        if (verbose) {
            printf("[NVBPF] tile_lifetime kernel=%s\n", func_name);
            printf("        segments=%lu avg_lifetime=%.2f max_lifetime=%lu avg_math=%.2f\n",
                   segments, avg_lifetime, lifetime_max, avg_math);
            printf("        hist:");
            for (int i = 0; i < kLifetimeBuckets; i++) {
                printf(" %s=%lu", bucket_name(i), hist[i]);
            }
            printf("\n");
        } else {
            TileSummary* summary = find_summary(func_name);
            if (summary == nullptr) {
                TileSummary fresh{};
                fresh.kernel_name = func_name;
                summaries.push_back(fresh);
                summary = &summaries.back();
            }
            summary->launches++;
            summary->segments += segments;
            summary->lifetime_sum += lifetime_sum;
            if (lifetime_max > summary->lifetime_max) summary->lifetime_max = lifetime_max;
            summary->math_sum += math_sum;
            for (int i = 0; i < kLifetimeBuckets; i++) {
                summary->hist[i] += hist[i];
            }
        }
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    if (!verbose) {
        printf("[NVBPF TILE_LIFETIME_TRACKER] matched_launches=%lu unique_kernels=%zu\n",
               matched_launches, summaries.size());
        for (const auto& summary : summaries) {
            double avg_lifetime =
                summary.segments ? (double)summary.lifetime_sum / (double)summary.segments : 0.0;
            double avg_math =
                summary.segments ? (double)summary.math_sum / (double)summary.segments : 0.0;
            printf("  x%-3lu %-32s | seg=%-6lu avg_life=%6.2f max=%-5lu avg_math=%6.2f | t4=%-5lu t16=%-5lu t64=%-5lu t256=%-5lu tlong=%-5lu\n",
                   summary.launches,
                   compact_kernel_name(summary.kernel_name).c_str(),
                   summary.segments,
                   avg_lifetime,
                   summary.lifetime_max,
                   avg_math,
                   summary.hist[0],
                   summary.hist[1],
                   summary.hist[2],
                   summary.hist[3],
                   summary.hist[4]);
        }
    }
    printf("[NVBPF TILE_LIFETIME_TRACKER] Tool terminated\n");
}
