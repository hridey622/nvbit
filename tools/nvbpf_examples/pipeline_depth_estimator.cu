/*
 * NV-BPF Example: Pipeline Depth Estimator
 *
 * Static instruction-stream estimator for how deeply a kernel appears to stage
 * producer instructions ahead of tensor/FFMA consumers.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

struct PipelineSummary {
    std::string kernel_name;
    uint64_t launches = 0;
    uint64_t producer_count = 0;
    uint64_t consumer_count = 0;
    uint64_t tensor_count = 0;
    uint64_t cp_async_count = 0;
    uint64_t ldmatrix_count = 0;
    uint64_t producer_bursts = 0;
    uint64_t max_producer_run = 0;
    double avg_gap = 0.0;
    uint64_t max_gap = 0;
    uint32_t stage_est = 0;
    uint32_t overlap_score = 0;
};

static pthread_mutex_t launch_mutex;
static std::vector<PipelineSummary> summaries;
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

static bool is_cp_async_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "CPASYNC") ||
           opcode_starts_with(opcode, "LDGSTS");
}

static bool is_producer_instr(Instr* instr, const char* opcode) {
    return (instr->isLoad() &&
            instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) ||
           is_cp_async_opcode(opcode) ||
           opcode_starts_with(opcode, "LDMATRIX");
}

static bool is_consumer_instr(const char* opcode) {
    return is_tensor_opcode(opcode) ||
           opcode_starts_with(opcode, "FFMA") ||
           opcode_starts_with(opcode, "HFMA2");
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

static const char* stage_label(uint32_t stage_est) {
    switch (stage_est) {
        case 0: return "no_pipeline";
        case 1: return "shallow";
        case 2: return "moderate";
        case 3: return "deep";
        default: return "very_deep";
    }
}

static PipelineSummary* find_summary(const char* func_name) {
    for (auto& summary : summaries) {
        if (summary.kernel_name == func_name) return &summary;
    }
    return nullptr;
}

static PipelineSummary analyze_pipeline(CUcontext ctx, CUfunction func,
                                        const char* func_name) {
    PipelineSummary summary{};
    summary.kernel_name = func_name;

    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    uint64_t gap_sum = 0;
    uint64_t gap_count = 0;
    int64_t last_producer_idx = -1;
    uint64_t current_producer_run = 0;

    for (auto f : related) {
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            const char* opcode = instr->getOpcodeShort();
            bool producer = is_producer_instr(instr, opcode);
            bool consumer = is_consumer_instr(opcode);

            if (producer) {
                summary.producer_count++;
                current_producer_run++;
                if (current_producer_run == 1) summary.producer_bursts++;
                if (current_producer_run > summary.max_producer_run) {
                    summary.max_producer_run = current_producer_run;
                }
                last_producer_idx = (int64_t)instr->getIdx();
                if (is_cp_async_opcode(opcode)) summary.cp_async_count++;
                if (opcode_starts_with(opcode, "LDMATRIX")) summary.ldmatrix_count++;
            } else {
                current_producer_run = 0;
            }

            if (consumer) {
                summary.consumer_count++;
                if (is_tensor_opcode(opcode)) summary.tensor_count++;
                if (last_producer_idx >= 0) {
                    uint64_t gap = (uint64_t)((int64_t)instr->getIdx() - last_producer_idx);
                    gap_sum += gap;
                    gap_count++;
                    if (gap > summary.max_gap) summary.max_gap = gap;
                }
            }
        }
    }

    summary.avg_gap = gap_count ? (double)gap_sum / (double)gap_count : 0.0;
    if (summary.consumer_count == 0 || summary.producer_count == 0) {
        summary.stage_est = 0;
        summary.overlap_score = 0;
        return summary;
    }

    if (summary.cp_async_count > 0 && summary.ldmatrix_count > 0 &&
        summary.avg_gap >= 6.0) {
        summary.stage_est = 4;
    } else if ((summary.cp_async_count > 0 || summary.ldmatrix_count > 0) &&
               summary.avg_gap >= 3.0) {
        summary.stage_est = 3;
    } else if (summary.avg_gap >= 2.0 || summary.max_producer_run >= 2) {
        summary.stage_est = 2;
    } else {
        summary.stage_est = 1;
    }

    double score =
        summary.avg_gap * 12.0 +
        (summary.cp_async_count > 0 ? 18.0 : 0.0) +
        (summary.ldmatrix_count > 0 ? 14.0 : 0.0) +
        (summary.tensor_count > 0 ? 8.0 : 0.0) +
        (double)summary.max_producer_run * 4.0;
    if (score > 100.0) score = 100.0;
    summary.overlap_score = (uint32_t)(score + 0.5);
    return summary;
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF PIPELINE_DEPTH_ESTIMATOR] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!is_exit || !nvbpf_is_launch_event(cbid)) return;
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);
    if (!kernel_name_filter.empty() &&
        strstr(func_name, kernel_name_filter.c_str()) == nullptr) {
        return;
    }

    pthread_mutex_lock(&launch_mutex);
    matched_launches++;
    PipelineSummary* summary = find_summary(func_name);
    if (summary == nullptr) {
        PipelineSummary fresh = analyze_pipeline(ctx, func, func_name);
        summaries.push_back(fresh);
        summary = &summaries.back();
    }
    summary->launches++;

    if (verbose) {
        printf("[NVBPF] pipeline_depth kernel=%s\n", func_name);
        printf("        producers=%lu consumers=%lu tensor=%lu cp_async=%lu ldmatrix=%lu bursts=%lu max_run=%lu\n",
               summary->producer_count, summary->consumer_count, summary->tensor_count,
               summary->cp_async_count, summary->ldmatrix_count,
               summary->producer_bursts, summary->max_producer_run);
        printf("        avg_gap=%.2f max_gap=%lu stage_est=%u overlap_score=%u (%s)\n",
               summary->avg_gap, summary->max_gap, summary->stage_est,
               summary->overlap_score, stage_label(summary->stage_est));
    }
    pthread_mutex_unlock(&launch_mutex);
}

void nvbit_at_term() {
    if (!verbose) {
        printf("[NVBPF PIPELINE_DEPTH_ESTIMATOR] matched_launches=%lu unique_kernels=%zu\n",
               matched_launches, summaries.size());
        for (const auto& summary : summaries) {
            printf("  x%-3lu %-32s | prod=%-6lu cons=%-6lu avg_gap=%5.2f max=%-4lu burst=%-4lu stage=%u overlap=%-3u | %s\n",
                   summary.launches,
                   compact_kernel_name(summary.kernel_name).c_str(),
                   summary.producer_count,
                   summary.consumer_count,
                   summary.avg_gap,
                   summary.max_gap,
                   summary.max_producer_run,
                   summary.stage_est,
                   summary.overlap_score,
                   stage_label(summary.stage_est));
        }
    }
    printf("[NVBPF PIPELINE_DEPTH_ESTIMATOR] Tool terminated\n");
}
