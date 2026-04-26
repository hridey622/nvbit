/*
 * NV-BPF Example: Register Pressure Distortion Meter
 *
 * Host-side estimator for whether register count and local-memory traffic are
 * likely distorting occupancy or pushing a kernel into an unfavorable
 * residency/reuse tradeoff.
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

struct StaticKernelMetrics {
    std::string kernel_name;
    uint64_t total_instrs = 0;
    uint64_t local_ops = 0;
    uint64_t math_ops = 0;
};

struct PressureSummary {
    std::string kernel_name;
    uint32_t bx = 1, by = 1, bz = 1;
    uint32_t regs = 0;
    uint32_t dynamic_smem = 0;
    uint32_t resident_ctas = 0;
    uint64_t launches = 0;
    uint64_t total_instrs = 0;
    uint64_t local_ops = 0;
    uint64_t math_ops = 0;
    double distortion_score = 0.0;
    int label_code = 0;
};

static pthread_mutex_t launch_mutex;
static std::vector<StaticKernelMetrics> static_metrics;
static std::vector<PressureSummary> summaries;
static std::string kernel_name_filter;
static bool verbose = false;
static bool full_names = false;
static uint64_t matched_launches = 0;

static bool opcode_starts_with(const char* opcode, const char* prefix) {
    return strncmp(opcode, prefix, strlen(prefix)) == 0;
}

static bool is_math_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "HMMA") ||
           opcode_starts_with(opcode, "MMA") ||
           opcode_starts_with(opcode, "WGMMA") ||
           opcode_starts_with(opcode, "BMMA") ||
           opcode_starts_with(opcode, "FFMA") ||
           opcode_starts_with(opcode, "HFMA2") ||
           opcode_starts_with(opcode, "IMAD") ||
           opcode_starts_with(opcode, "IADD3") ||
           opcode_starts_with(opcode, "FADD") ||
           opcode_starts_with(opcode, "FMUL");
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

static std::string join_clauses(const std::vector<std::string>& clauses) {
    std::string out;
    for (size_t i = 0; i < clauses.size(); ++i) {
        if (i != 0) out += "; ";
        out += clauses[i];
    }
    return out;
}

static const char* label_name(int label_code) {
    switch (label_code) {
        case 0: return "low";
        case 1: return "moderate";
        default: return "high";
    }
}

static double local_ratio(const PressureSummary& summary) {
    if (summary.total_instrs == 0) return 0.0;
    return (double)summary.local_ops / (double)summary.total_instrs;
}

static double regs_per_occ(const PressureSummary& summary) {
    return (double)summary.regs / (summary.resident_ctas > 0 ? (double)summary.resident_ctas : 1.0);
}

static const char* evidence_label(const PressureSummary& summary) {
    if (summary.total_instrs < 64) return "weak";
    if (summary.local_ops == 0 && summary.regs < 64) return "weak";
    if (summary.resident_ctas <= 2 || summary.regs >= 64 || local_ratio(summary) >= 0.08) {
        return "strong";
    }
    return "medium";
}

static std::string explain_summary(const PressureSummary& summary) {
    std::vector<std::string> clauses;
    double local_ratio_value = local_ratio(summary);

    if (summary.regs >= 96) {
        clauses.push_back("register count is high");
    } else if (summary.regs >= 64) {
        clauses.push_back("register count is moderately high");
    } else {
        clauses.push_back("register count is still fairly modest");
    }

    if (summary.resident_ctas <= 1) {
        clauses.push_back("only 1 CTA/SM fits at this launch shape");
    } else if (summary.resident_ctas <= 2) {
        clauses.push_back("residency is modest");
    } else if (summary.resident_ctas >= 4) {
        clauses.push_back("residency still looks healthy");
    }

    if (local_ratio_value >= 0.10) {
        clauses.push_back("local-memory ops are a meaningful part of the static instruction mix");
    } else if (summary.local_ops > 0) {
        clauses.push_back("some local-memory ops are present");
    }

    std::string prefix;
    if (summary.label_code >= 2) prefix = "High pressure suspicion: ";
    else if (summary.label_code == 1) prefix = "Moderate pressure suspicion: ";
    else prefix = "Low pressure suspicion: ";

    std::string note = prefix + join_clauses(clauses) + ".";
    if (summary.label_code >= 1) {
        note += " This is a heuristic lead derived from register count, occupancy, and local-memory activity.";
    } else {
        note += " Registers do not look like the main limiter for this launch shape.";
    }
    return note;
}

static std::string next_step_summary(const PressureSummary& summary) {
    double local_ratio_value = local_ratio(summary);
    if (summary.label_code >= 2) {
        return "Try reducing tile size, accumulator footprint, or unroll depth, and look for spills or local-memory traffic that can be eliminated.";
    }
    if (summary.label_code == 1) {
        return "Check whether smaller tiles or lighter unrolling improve residency before rewriting memory layout.";
    }
    if (local_ratio_value >= 0.10) {
        return "Registers do not look like the first limiter, but the local-memory footprint is noticeable; inspect indexing patterns and compiler-generated local traffic next.";
    }
    return "Registers look healthy for this launch shape; inspect memory behavior, synchronization, or control-flow inefficiency next.";
}

static void launch_block_config(nvbit_api_cuda_t cbid, void* params,
                                uint32_t* bx, uint32_t* by, uint32_t* bz,
                                uint32_t* dynamic_smem) {
    *bx = *by = *bz = 1;
    *dynamic_smem = 0;
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
        cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
        *bx = p->config->blockDimX;
        *by = p->config->blockDimY;
        *bz = p->config->blockDimZ;
        *dynamic_smem = p->config->sharedMemBytes;
    } else if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
        *bx = p->blockDimX;
        *by = p->blockDimY;
        *bz = p->blockDimZ;
        *dynamic_smem = p->sharedMemBytes;
    }
}

static StaticKernelMetrics* find_static_metrics(const char* func_name) {
    for (auto& metrics : static_metrics) {
        if (metrics.kernel_name == func_name) return &metrics;
    }
    return nullptr;
}

static StaticKernelMetrics analyze_static_metrics(CUcontext ctx, CUfunction func,
                                                  const char* func_name) {
    StaticKernelMetrics metrics{};
    metrics.kernel_name = func_name;
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);
    for (auto f : related) {
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            metrics.total_instrs++;
            if ((instr->isLoad() || instr->isStore()) &&
                instr->getMemorySpace() == InstrType::MemorySpace::LOCAL) {
                metrics.local_ops++;
            }
            if (is_math_opcode(instr->getOpcodeShort())) {
                metrics.math_ops++;
            }
        }
    }
    return metrics;
}

static PressureSummary* find_summary(const char* func_name, uint32_t bx,
                                     uint32_t by, uint32_t bz,
                                     uint32_t dynamic_smem) {
    for (auto& summary : summaries) {
        if (summary.kernel_name == func_name &&
            summary.bx == bx && summary.by == by && summary.bz == bz &&
            summary.dynamic_smem == dynamic_smem) {
            return &summary;
        }
    }
    return nullptr;
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF REGISTER_PRESSURE_DISTORTION_METER] Tool loaded\n");
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

    StaticKernelMetrics* metrics = find_static_metrics(func_name);
    if (metrics == nullptr) {
        static_metrics.push_back(analyze_static_metrics(ctx, func, func_name));
        metrics = &static_metrics.back();
    }

    func_config_t cfg{};
    nvbit_get_func_config(ctx, func, &cfg);
    uint32_t bx, by, bz, dynamic_smem;
    launch_block_config(cbid, params, &bx, &by, &bz, &dynamic_smem);
    int resident_ctas = 0;
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&resident_ctas, func,
                                                (int)(bx * by * bz),
                                                dynamic_smem);

    PressureSummary* summary = find_summary(func_name, bx, by, bz, dynamic_smem);
    if (summary == nullptr) {
        PressureSummary fresh{};
        fresh.kernel_name = func_name;
        fresh.bx = bx;
        fresh.by = by;
        fresh.bz = bz;
        fresh.regs = cfg.num_registers;
        fresh.dynamic_smem = dynamic_smem;
        fresh.resident_ctas = (resident_ctas < 0) ? 0u : (uint32_t)resident_ctas;
        fresh.total_instrs = metrics->total_instrs;
        fresh.local_ops = metrics->local_ops;
        fresh.math_ops = metrics->math_ops;

        double local_ratio =
            fresh.total_instrs ? (double)fresh.local_ops / (double)fresh.total_instrs : 0.0;
        double score =
            ((double)fresh.regs / 32.0) *
            (1.0 + local_ratio * 8.0) /
            (fresh.resident_ctas > 0 ? (double)fresh.resident_ctas : 1.0);
        fresh.distortion_score = score;
        if (fresh.resident_ctas <= 1 || score >= 4.0) {
            fresh.label_code = 2;
        } else if (fresh.resident_ctas <= 2 || score >= 2.0) {
            fresh.label_code = 1;
        } else {
            fresh.label_code = 0;
        }
        summaries.push_back(fresh);
        summary = &summaries.back();
    }
    summary->launches++;

    if (verbose) {
        printf("[NVBPF] reg_pressure kernel=%s\n", func_name);
        printf("        block=(%u,%u,%u) regs=%u resident_ctas=%u dyn_smem=%u local_ops=%lu math_ops=%lu score=%.2f (%s)\n",
               summary->bx, summary->by, summary->bz, summary->regs,
               summary->resident_ctas, summary->dynamic_smem,
               summary->local_ops, summary->math_ops,
               summary->distortion_score, label_name(summary->label_code));
    }
    pthread_mutex_unlock(&launch_mutex);
}

void nvbit_at_term() {
    if (!verbose) {
        printf("[NVBPF REGISTER_PRESSURE_DISTORTION_METER] matched_launches=%lu unique_configs=%zu\n",
               matched_launches, summaries.size());
        for (const auto& summary : summaries) {
            printf("  x%-3lu %-28s blk=%ux%ux%u | regs=%-4u occ=%-2u local=%-5lu math=%-5lu score=%5.2f | %s\n",
                   summary.launches,
                   compact_kernel_name(summary.kernel_name).c_str(),
                   summary.bx, summary.by, summary.bz,
                   summary.regs,
                   summary.resident_ctas,
                   summary.local_ops,
                   summary.math_ops,
                   summary.distortion_score,
                   label_name(summary.label_code));
            printf("      derived: total=%lu local_ratio=%.3f regs_per_occ=%.2f evidence=%s\n",
                   summary.total_instrs,
                   local_ratio(summary),
                   regs_per_occ(summary),
                   evidence_label(summary));
            printf("      note: %s\n", explain_summary(summary).c_str());
            printf("      next: %s\n", next_step_summary(summary).c_str());
        }
    }
    printf("[NVBPF REGISTER_PRESSURE_DISTORTION_METER] Tool terminated\n");
}
