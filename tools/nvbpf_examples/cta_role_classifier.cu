/*
 * NV-BPF Example: CTA Role Classifier
 *
 * Samples CTA-local activity and clusters sampled CTAs into broad role
 * families: compute-heavy, memory-heavy, control-heavy, edge/tail, or
 * balanced.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <unordered_set>
#include <vector>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

static constexpr int kCtaProfileSlots = 4096;

enum CtaMetricKind {
    CK_MEM = 0,
    CK_MATH = 1,
    CK_BRANCH = 2,
};

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

struct CtaRoleSummary {
    std::string kernel_name;
    uint64_t launches = 0;
    uint64_t sampled_ctas = 0;
    uint64_t dropped = 0;
    uint64_t compute_heavy = 0;
    uint64_t memory_heavy = 0;
    uint64_t control_heavy = 0;
    uint64_t edge_tail = 0;
    uint64_t balanced = 0;
};

BPF_ARRAY(crc_profiles, CtaProfile, kCtaProfileSlots);
BPF_ARRAY(crc_dropped, uint64_t, 1);

extern "C" __device__ __noinline__ void crc_mark_cta(int pred,
                                                     uint64_t pprofiles,
                                                     uint64_t pdropped);
extern "C" __device__ __noinline__ void crc_trace_site(int pred,
                                                       uint64_t pprofiles,
                                                       uint64_t pdropped,
                                                       uint32_t kind);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static std::vector<CtaRoleSummary> summaries;
static std::string kernel_name_filter;
static bool verbose = false;
static bool full_names = false;
static uint64_t matched_launches = 0;

static bool opcode_starts_with(const char* opcode, const char* prefix) {
    return strncmp(opcode, prefix, strlen(prefix)) == 0;
}

static bool is_branch_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "BRA") ||
           opcode_starts_with(opcode, "JMP") ||
           opcode_starts_with(opcode, "JMX") ||
           opcode_starts_with(opcode, "BRX") ||
           opcode_starts_with(opcode, "CALL") ||
           opcode_starts_with(opcode, "RET") ||
           opcode_starts_with(opcode, "EXIT");
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

static CtaRoleSummary* find_summary(const char* func_name) {
    for (auto& summary : summaries) {
        if (summary.kernel_name == func_name) return &summary;
    }
    return nullptr;
}

static void reset_state() {
    crc_profiles.reset();
    crc_dropped.reset();
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        bool inserted_mark = false;
        for (auto* instr : instrs) {
            if (!inserted_mark) {
                nvbit_insert_call(instr, "crc_mark_cta", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&crc_profiles.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&crc_dropped.data[0]);
                inserted_mark = true;
            }

            const char* opcode = instr->getOpcodeShort();
            if ((instr->isLoad() &&
                 instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) ||
                instr->isStore()) {
                nvbit_insert_call(instr, "crc_trace_site", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&crc_profiles.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&crc_dropped.data[0]);
                nvbit_add_call_arg_const_val32(instr, CK_MEM);
            } else if (is_branch_opcode(opcode)) {
                nvbit_insert_call(instr, "crc_trace_site", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&crc_profiles.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&crc_dropped.data[0]);
                nvbit_add_call_arg_const_val32(instr, CK_BRANCH);
            } else if (is_math_opcode(opcode)) {
                nvbit_insert_call(instr, "crc_trace_site", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&crc_profiles.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&crc_dropped.data[0]);
                nvbit_add_call_arg_const_val32(instr, CK_MATH);
            }
        }
    }
}

static const char* classify_profile(const CtaProfile& profile) {
    if (profile.edge) return "edge_tail";
    if (profile.math_sites >= profile.mem_sites * 2 &&
        profile.math_sites >= profile.branch_sites * 2) {
        return "compute_heavy";
    }
    if (profile.mem_sites >= profile.math_sites * 2 &&
        profile.mem_sites >= profile.branch_sites) {
        return "memory_heavy";
    }
    if (profile.branch_sites > profile.math_sites &&
        profile.branch_sites > profile.mem_sites) {
        return "control_heavy";
    }
    return "balanced";
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF CTA_ROLE_CLASSIFIER] Tool loaded\n");
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

        uint64_t dropped = *crc_dropped.lookup(0);
        uint64_t sampled_ctas = 0;
        uint64_t compute_heavy = 0;
        uint64_t memory_heavy = 0;
        uint64_t control_heavy = 0;
        uint64_t edge_tail = 0;
        uint64_t balanced = 0;

        for (int i = 0; i < kCtaProfileSlots; i++) {
            CtaProfile* profile = crc_profiles.lookup(i);
            if (!profile || !profile->valid) continue;
            sampled_ctas++;
            const char* role = classify_profile(*profile);
            if (strcmp(role, "compute_heavy") == 0) compute_heavy++;
            else if (strcmp(role, "memory_heavy") == 0) memory_heavy++;
            else if (strcmp(role, "control_heavy") == 0) control_heavy++;
            else if (strcmp(role, "edge_tail") == 0) edge_tail++;
            else balanced++;
        }

        if (verbose) {
            printf("[NVBPF] cta_role kernel=%s\n", func_name);
            printf("        sampled_ctas=%lu dropped=%lu compute=%lu memory=%lu control=%lu edge=%lu balanced=%lu\n",
                   sampled_ctas, dropped, compute_heavy, memory_heavy,
                   control_heavy, edge_tail, balanced);
        } else {
            CtaRoleSummary* summary = find_summary(func_name);
            if (summary == nullptr) {
                CtaRoleSummary fresh{};
                fresh.kernel_name = func_name;
                summaries.push_back(fresh);
                summary = &summaries.back();
            }
            summary->launches++;
            summary->sampled_ctas += sampled_ctas;
            summary->dropped += dropped;
            summary->compute_heavy += compute_heavy;
            summary->memory_heavy += memory_heavy;
            summary->control_heavy += control_heavy;
            summary->edge_tail += edge_tail;
            summary->balanced += balanced;
        }
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    if (!verbose) {
        printf("[NVBPF CTA_ROLE_CLASSIFIER] matched_launches=%lu unique_kernels=%zu\n",
               matched_launches, summaries.size());
        for (const auto& summary : summaries) {
            printf("  x%-3lu %-32s | ctas=%-6lu drop=%-5lu comp=%-5lu mem=%-5lu ctrl=%-5lu edge=%-5lu bal=%-5lu\n",
                   summary.launches,
                   compact_kernel_name(summary.kernel_name).c_str(),
                   summary.sampled_ctas,
                   summary.dropped,
                   summary.compute_heavy,
                   summary.memory_heavy,
                   summary.control_heavy,
                   summary.edge_tail,
                   summary.balanced);
        }
    }
    printf("[NVBPF CTA_ROLE_CLASSIFIER] Tool terminated\n");
}
