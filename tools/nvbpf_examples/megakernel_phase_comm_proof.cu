/*
 * NV-BPF Example: Megakernel Phase/Communication Proof
 *
 * Direct NVBPF/NVBit proof tool for:
 *   1. device-side megakernel phase ordering and phase overlap proxies
 *   2. multi-GPU communication/launch ordering and stream-event overlap
 *
 * This intentionally does not use NCU, NSYS, or CUPTI.
 */

#include <algorithm>
#include <chrono>
#include <cctype>
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

static constexpr uint64_t kPhaseMarkerMagic = 0x4e564200ULL;
static constexpr uint64_t kPhaseMarkerMask = 0xffffff00ULL;
static constexpr uint32_t kBitmapWords = 4;
static constexpr uint32_t kMaxGpu = 128;

BPF_ARRAY(phase_counts, uint64_t, PHASE_COUNT);
BPF_ARRAY(phase_first_clock, uint64_t, PHASE_COUNT);
BPF_ARRAY(phase_last_clock, uint64_t, PHASE_COUNT);
BPF_ARRAY(phase_active_sum, uint64_t, PHASE_COUNT);
BPF_ARRAY(phase_low_active, uint64_t, PHASE_COUNT);
BPF_ARRAY(phase_sm_bitmap, uint64_t, PHASE_COUNT * kBitmapWords);

extern "C" __device__ __noinline__ void mpcp_record_phase(int pred,
                                                           uint64_t pcounts,
                                                           uint64_t pfirst_clock,
                                                           uint64_t plast_clock,
                                                           uint64_t pactive_sum,
                                                           uint64_t plow_active,
                                                           uint64_t psm_bitmap,
                                                           uint32_t phase,
                                                           uint32_t threshold);

enum class OpKind {
    Launch,
    PeerCopy,
    DeviceCopy,
};

struct PendingOp {
    bool active = false;
    OpKind kind = OpKind::Launch;
    uint64_t seq = 0;
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    CUevent base = nullptr;
    CUevent start = nullptr;
    CUevent end = nullptr;
    int gpu = -1;
    int src_gpu = -1;
    int dst_gpu = -1;
    uint64_t bytes = 0;
    uint64_t host_enter_ns = 0;
    std::string name;
};

struct EventOp {
    OpKind kind = OpKind::Launch;
    uint64_t seq = 0;
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    CUevent base = nullptr;
    CUevent start = nullptr;
    CUevent end = nullptr;
    int gpu = -1;
    int src_gpu = -1;
    int dst_gpu = -1;
    uint64_t bytes = 0;
    uint64_t host_enter_ns = 0;
    uint64_t host_exit_ns = 0;
    std::string name;
    bool timing_valid = false;
    float start_ms = 0.0f;
    float end_ms = 0.0f;
    float duration_ms = 0.0f;
};

struct ContextBase {
    CUcontext ctx = nullptr;
    CUevent event = nullptr;
    int gpu = -1;
};

static pthread_mutex_t proof_mutex;
static pthread_mutex_t event_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static std::vector<EventOp> event_ops;
static std::vector<ContextBase> context_bases;
static std::string kernel_name_filter;
static std::vector<int> expected_order;
static bool full_names = false;
static bool verbose = false;
static bool device_phases = true;
static bool sync_for_phases = true;
static bool event_proof = false;
static bool source_markers = true;
static uint32_t low_active_threshold = 16;
static double min_phase_overlap_ratio = 0.10;
static double min_comm_overlap_ratio = 0.25;
static uint64_t launch_seq = 0;
static uint64_t event_seq = 0;
static uint64_t matched_launches = 0;
static uint64_t peer_copy_seen = 0;
static uint64_t stream_wait_seen = 0;
static uint64_t event_record_seen = 0;
static uint64_t host_sync_seen = 0;
static uint64_t gpu_launches[kMaxGpu] = {};
static thread_local bool inside_tool_cuda = false;
static thread_local PendingOp pending_op;

static const char* phase_name(int phase) {
    switch (phase) {
        case PHASE_ENTRY: return "entry";
        case PHASE_TMA: return "tma";
        case PHASE_TMEM: return "tmem";
        case PHASE_TENSOR: return "tensor";
        case PHASE_SCALE: return "scale";
        case PHASE_MULTICAST: return "multicast";
        case PHASE_SYNC: return "sync";
        case PHASE_PRODUCER: return "producer";
        case PHASE_HANDOFF: return "handoff";
        case PHASE_CONSUMER: return "consumer";
        case PHASE_EPILOGUE: return "epilogue";
        case PHASE_BRANCH: return "branch";
        default: return "unknown";
    }
}

static uint64_t host_now_ns() {
    using clock = std::chrono::steady_clock;
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count();
}

static bool opcode_starts_with(const char* opcode, const char* prefix) {
    return strncmp(opcode, prefix, strlen(prefix)) == 0;
}

static bool csv_or_substring_match(const char* name, const std::string& filter) {
    if (filter.empty()) return true;
    size_t start = 0;
    while (start <= filter.size()) {
        size_t end = filter.find(',', start);
        if (end == std::string::npos) end = filter.size();
        std::string token = filter.substr(start, end - start);
        if (!token.empty() && strstr(name, token.c_str()) != nullptr) return true;
        if (end == filter.size()) break;
        start = end + 1;
    }
    return false;
}

static std::string compact_kernel_name(const char* raw) {
    if (raw == nullptr) return std::string("<unknown>");
    if (full_names) return std::string(raw);
    std::string name = raw;
    if (name.rfind("void ", 0) == 0) name = name.substr(5);
    size_t paren = name.find('(');
    if (paren != std::string::npos) name = name.substr(0, paren);
    if (name.size() <= 72) return name;
    return name.substr(0, 34) + "..." + name.substr(name.size() - 34);
}

static int current_device() {
    CUdevice dev = (CUdevice)-1;
    return cuCtxGetDevice(&dev) == CUDA_SUCCESS ? (int)dev : -1;
}

static int device_for_pointer(CUdeviceptr ptr) {
    if (ptr == 0) return -1;
    CUdevice dev = (CUdevice)-1;
    if (cuPointerGetAttribute(&dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr) == CUDA_SUCCESS) {
        return (int)dev;
    }
    return -1;
}

static int phase_from_token(const std::string& token) {
    for (int i = 0; i < PHASE_COUNT; ++i) {
        if (token == phase_name(i)) return i;
    }
    if (token == "produce") return PHASE_PRODUCER;
    if (token == "consume") return PHASE_CONSUMER;
    if (token == "scale_factor" || token == "scale-factor") return PHASE_SCALE;
    if (token == "cluster_multicast" || token == "cluster-multicast") return PHASE_MULTICAST;
    return -1;
}

static std::vector<int> parse_phase_order(const char* env) {
    std::string value = env ? env : "entry,tma,tensor,epilogue";
    std::vector<int> out;
    size_t start = 0;
    while (start <= value.size()) {
        size_t end = value.find(',', start);
        if (end == std::string::npos) end = value.size();
        std::string token = value.substr(start, end - start);
        token.erase(
            std::remove_if(token.begin(), token.end(),
                           [](unsigned char c) { return std::isspace(c); }),
            token.end());
        int phase = phase_from_token(token);
        if (phase >= 0) out.push_back(phase);
        if (end == value.size()) break;
        start = end + 1;
    }
    return out;
}

static bool is_tma_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "CPASYNC") ||
           opcode_starts_with(opcode, "LDGSTS") ||
           opcode_starts_with(opcode, "UTMA") ||
           opcode_starts_with(opcode, "TMA");
}

static bool is_tmem_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "LDT") ||
           opcode_starts_with(opcode, "STT");
}

static bool is_tensor_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "UTC") ||
           opcode_starts_with(opcode, "TCGEN05") ||
           opcode_starts_with(opcode, "MMA") ||
           opcode_starts_with(opcode, "WGMMA") ||
           opcode_starts_with(opcode, "HMMA") ||
           opcode_starts_with(opcode, "BMMA");
}

static bool is_sync_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "BAR") ||
           opcode_starts_with(opcode, "DEPBAR") ||
           opcode_starts_with(opcode, "MEMBAR") ||
           opcode_starts_with(opcode, "WARPSYNC") ||
           opcode_starts_with(opcode, "UTCBAR") ||
           opcode_starts_with(opcode, "FENCE");
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

static int source_marker_phase_from_instr(Instr* instr) {
    if (!source_markers) return -1;
    int operands = instr->getNumOperands();
    for (int i = 0; i < operands; ++i) {
        const InstrType::operand_t* operand = instr->getOperand(i);
        if (operand == nullptr) continue;
        if (operand->type != InstrType::OperandType::IMM_UINT64) continue;
        uint64_t value = operand->u.imm_uint64.value;
        if ((value & kPhaseMarkerMask) != kPhaseMarkerMagic) continue;
        uint32_t phase = (uint32_t)(value & 0xffu);
        if (phase < PHASE_COUNT) return (int)phase;
    }
    return -1;
}

static void reset_phase_state() {
    phase_counts.reset();
    phase_first_clock.reset();
    phase_last_clock.reset();
    phase_active_sum.reset();
    phase_low_active.reset();
    phase_sm_bitmap.reset();
}

static void inject_phase_counter(Instr* instr, ProofPhase phase) {
    nvbit_insert_call(instr, "mpcp_record_phase", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(instr);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&phase_counts.data[0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&phase_first_clock.data[0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&phase_last_clock.data[0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&phase_active_sum.data[0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&phase_low_active.data[0]);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&phase_sm_bitmap.data[0]);
    nvbit_add_call_arg_const_val32(instr, (uint32_t)phase);
    nvbit_add_call_arg_const_val32(instr, low_active_threshold);
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);
    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        bool entry_inserted = false;
        for (auto* instr : instrs) {
            const char* opcode = instr->getOpcodeShort();
            if (!entry_inserted) {
                inject_phase_counter(instr, PHASE_ENTRY);
                entry_inserted = true;
            }
            int marker_phase = source_marker_phase_from_instr(instr);
            if (marker_phase >= 0) {
                inject_phase_counter(instr, (ProofPhase)marker_phase);
                if (verbose) {
                    printf("[NVBPF PROOF] marker phase=%s sass=%s\n",
                           phase_name(marker_phase), instr->getSass());
                }
                continue;
            }
            if (is_tma_opcode(opcode)) inject_phase_counter(instr, PHASE_TMA);
            if (is_tmem_opcode(opcode)) inject_phase_counter(instr, PHASE_TMEM);
            if (is_tensor_opcode(opcode)) inject_phase_counter(instr, PHASE_TENSOR);
            if (is_sync_opcode(opcode)) inject_phase_counter(instr, PHASE_SYNC);
            if (instr->isStore() || opcode_starts_with(opcode, "ATOM") ||
                opcode_starts_with(opcode, "RED")) {
                inject_phase_counter(instr, PHASE_EPILOGUE);
            }
            if (is_branch_opcode(opcode)) inject_phase_counter(instr, PHASE_BRANCH);
        }
    }
}

static uint64_t read_phase_map(BpfArrayMap<uint64_t, PHASE_COUNT>& map, int phase) {
    uint64_t* val = map.lookup((uint32_t)phase);
    return val ? *val : 0ULL;
}

static uint64_t read_phase_bitmap_word(int phase, int word) {
    uint64_t* val = phase_sm_bitmap.lookup((uint32_t)(phase * kBitmapWords + word));
    return val ? *val : 0ULL;
}

static int phase_active_sms(int phase) {
    int total = 0;
    for (uint32_t word = 0; word < kBitmapWords; ++word) {
        total += __builtin_popcountll(read_phase_bitmap_word(phase, word));
    }
    return total;
}

static uint64_t interval_overlap(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1) {
    if (a0 == 0 || b0 == 0 || a1 <= a0 || b1 <= b0) return 0;
    uint64_t lo = std::max(a0, b0);
    uint64_t hi = std::min(a1, b1);
    return hi > lo ? hi - lo : 0;
}

static void print_phase_proof(uint64_t seq, int gpu, const char* kernel_name) {
    uint64_t counts[PHASE_COUNT] = {};
    uint64_t first[PHASE_COUNT] = {};
    uint64_t last[PHASE_COUNT] = {};
    uint64_t active_sum[PHASE_COUNT] = {};
    uint64_t low_active[PHASE_COUNT] = {};

    for (int phase = 0; phase < PHASE_COUNT; ++phase) {
        counts[phase] = read_phase_map(phase_counts, phase);
        first[phase] = read_phase_map(phase_first_clock, phase);
        last[phase] = read_phase_map(phase_last_clock, phase);
        active_sum[phase] = read_phase_map(phase_active_sum, phase);
        low_active[phase] = read_phase_map(phase_low_active, phase);
    }

    printf("[NVBPF PROOF] phase_launch #%lu GPU%d %s\n",
           seq, gpu, compact_kernel_name(kernel_name).c_str());

    for (int phase = 0; phase < PHASE_COUNT; ++phase) {
        if (counts[phase] == 0) continue;
        double avg_lanes = counts[phase] ? (double)active_sum[phase] / (double)counts[phase] : 0.0;
        printf("    phase %-8s count=%lu first=%lu last=%lu span=%lu avg_lanes=%.1f low_active=%lu active_sms=%d\n",
               phase_name(phase), counts[phase], first[phase], last[phase],
               last[phase] > first[phase] ? last[phase] - first[phase] : 0,
               avg_lanes, low_active[phase], phase_active_sms(phase));
    }

    bool order_pass = true;
    bool missing_phase = false;
    int previous = -1;
    printf("    expected_order=");
    for (size_t i = 0; i < expected_order.size(); ++i) {
        printf("%s%s", i ? "<" : "", phase_name(expected_order[i]));
    }
    for (int phase : expected_order) {
        if (counts[phase] == 0) {
            missing_phase = true;
            continue;
        }
        if (previous >= 0 && counts[previous] > 0 && first[phase] < first[previous]) {
            order_pass = false;
        }
        previous = phase;
    }
    printf(" verdict=%s%s\n",
           order_pass ? "PASS" : "FAIL",
           missing_phase ? " missing_expected_phase=1" : "");

    uint64_t tma_tensor = interval_overlap(first[PHASE_TMA], last[PHASE_TMA],
                                           first[PHASE_TENSOR], last[PHASE_TENSOR]);
    uint64_t tensor_epi = interval_overlap(first[PHASE_TENSOR], last[PHASE_TENSOR],
                                           first[PHASE_EPILOGUE], last[PHASE_EPILOGUE]);
    uint64_t tma_span = last[PHASE_TMA] > first[PHASE_TMA]
                            ? last[PHASE_TMA] - first[PHASE_TMA]
                            : 0;
    uint64_t tensor_span = last[PHASE_TENSOR] > first[PHASE_TENSOR]
                               ? last[PHASE_TENSOR] - first[PHASE_TENSOR]
                               : 0;
    double tma_tensor_ratio =
        std::min(tma_span, tensor_span) > 0
            ? (double)tma_tensor / (double)std::min(tma_span, tensor_span)
            : 0.0;
    printf("    phase_overlap tma_tensor_ticks=%lu ratio=%.3f verdict=%s\n",
           tma_tensor, tma_tensor_ratio,
           tma_tensor_ratio >= min_phase_overlap_ratio ? "PASS" : "WEAK");
    if (tensor_epi > 0) {
        printf("    phase_overlap tensor_epilogue_ticks=%lu note=pipelined_or_fused_epilogue\n",
               tensor_epi);
    }
}

static bool is_peer_copy_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuMemcpyPeer ||
           cbid == API_CUDA_cuMemcpyPeer_ptds ||
           cbid == API_CUDA_cuMemcpyPeerAsync ||
           cbid == API_CUDA_cuMemcpyPeerAsync_ptsz ||
           cbid == API_CUDA_cuMemcpy3DPeer ||
           cbid == API_CUDA_cuMemcpy3DPeer_ptds ||
           cbid == API_CUDA_cuMemcpy3DPeerAsync ||
           cbid == API_CUDA_cuMemcpy3DPeerAsync_ptsz;
}

static bool is_async_comm_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuMemcpyAsync ||
           cbid == API_CUDA_cuMemcpyAsync_ptsz ||
           cbid == API_CUDA_cuMemcpyDtoDAsync ||
           cbid == API_CUDA_cuMemcpyDtoDAsync_v2 ||
           cbid == API_CUDA_cuMemcpyDtoDAsync_v2_ptsz ||
           cbid == API_CUDA_cuMemcpyPeerAsync ||
           cbid == API_CUDA_cuMemcpyPeerAsync_ptsz ||
           cbid == API_CUDA_cuMemcpy3DPeerAsync ||
           cbid == API_CUDA_cuMemcpy3DPeerAsync_ptsz;
}

static bool is_host_sync_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuCtxSynchronize ||
           cbid == API_CUDA_cuCtxSynchronize_v2 ||
           cbid == API_CUDA_cuStreamSynchronize ||
           cbid == API_CUDA_cuStreamSynchronize_ptsz ||
           cbid == API_CUDA_cuEventSynchronize;
}

static bool is_event_record_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuEventRecord ||
           cbid == API_CUDA_cuEventRecord_ptsz ||
           cbid == API_CUDA_cuEventRecordWithFlags ||
           cbid == API_CUDA_cuEventRecordWithFlags_ptsz;
}

static bool is_stream_wait_cbid(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuStreamWaitEvent ||
           cbid == API_CUDA_cuStreamWaitEvent_ptsz;
}

static CUstream launch_stream(nvbit_api_cuda_t cbid, void* params) {
    if (cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
        return p->config ? p->config->hStream : nullptr;
    }
    if (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz) {
        return ((cuLaunchKernel_params*)params)->hStream;
    }
    if (cbid == API_CUDA_cuLaunchGridAsync) {
        return ((cuLaunchGridAsync_params*)params)->hStream;
    }
    return nullptr;
}

static CUstream copy_stream(nvbit_api_cuda_t cbid, void* params) {
    switch (cbid) {
        case API_CUDA_cuMemcpyAsync:
            return ((cuMemcpyAsync_params*)params)->hStream;
        case API_CUDA_cuMemcpyAsync_ptsz:
            return ((cuMemcpyAsync_ptsz_params*)params)->hStream;
        case API_CUDA_cuMemcpyDtoDAsync:
            return ((cuMemcpyDtoDAsync_params*)params)->hStream;
        case API_CUDA_cuMemcpyDtoDAsync_v2:
            return ((cuMemcpyDtoDAsync_v2_params*)params)->hStream;
        case API_CUDA_cuMemcpyDtoDAsync_v2_ptsz:
            return ((cuMemcpyDtoDAsync_v2_ptsz_params*)params)->hStream;
        case API_CUDA_cuMemcpyPeerAsync:
            return ((cuMemcpyPeerAsync_params*)params)->hStream;
        case API_CUDA_cuMemcpyPeerAsync_ptsz:
            return ((cuMemcpyPeerAsync_ptsz_params*)params)->hStream;
        case API_CUDA_cuMemcpy3DPeerAsync:
            return ((cuMemcpy3DPeerAsync_params*)params)->hStream;
        case API_CUDA_cuMemcpy3DPeerAsync_ptsz:
            return ((cuMemcpy3DPeerAsync_ptsz_params*)params)->hStream;
        default:
            return nullptr;
    }
}

static uint64_t copy_bytes(nvbit_api_cuda_t cbid, void* params) {
    switch (cbid) {
        case API_CUDA_cuMemcpyAsync:
            return ((cuMemcpyAsync_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyAsync_ptsz:
            return ((cuMemcpyAsync_ptsz_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoDAsync:
            return ((cuMemcpyDtoDAsync_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoDAsync_v2:
            return ((cuMemcpyDtoDAsync_v2_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyDtoDAsync_v2_ptsz:
            return ((cuMemcpyDtoDAsync_v2_ptsz_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyPeer:
            return ((cuMemcpyPeer_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyPeer_ptds:
            return ((cuMemcpyPeer_ptds_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyPeerAsync:
            return ((cuMemcpyPeerAsync_params*)params)->ByteCount;
        case API_CUDA_cuMemcpyPeerAsync_ptsz:
            return ((cuMemcpyPeerAsync_ptsz_params*)params)->ByteCount;
        case API_CUDA_cuMemcpy3DPeer:
            return ((cuMemcpy3DPeer_params*)params)->pCopy
                       ? ((cuMemcpy3DPeer_params*)params)->pCopy->WidthInBytes *
                             ((cuMemcpy3DPeer_params*)params)->pCopy->Height *
                             ((cuMemcpy3DPeer_params*)params)->pCopy->Depth
                       : 0ULL;
        case API_CUDA_cuMemcpy3DPeer_ptds:
            return ((cuMemcpy3DPeer_ptds_params*)params)->pCopy
                       ? ((cuMemcpy3DPeer_ptds_params*)params)->pCopy->WidthInBytes *
                             ((cuMemcpy3DPeer_ptds_params*)params)->pCopy->Height *
                             ((cuMemcpy3DPeer_ptds_params*)params)->pCopy->Depth
                       : 0ULL;
        case API_CUDA_cuMemcpy3DPeerAsync:
            return ((cuMemcpy3DPeerAsync_params*)params)->pCopy
                       ? ((cuMemcpy3DPeerAsync_params*)params)->pCopy->WidthInBytes *
                             ((cuMemcpy3DPeerAsync_params*)params)->pCopy->Height *
                             ((cuMemcpy3DPeerAsync_params*)params)->pCopy->Depth
                       : 0ULL;
        case API_CUDA_cuMemcpy3DPeerAsync_ptsz:
            return ((cuMemcpy3DPeerAsync_ptsz_params*)params)->pCopy
                       ? ((cuMemcpy3DPeerAsync_ptsz_params*)params)->pCopy->WidthInBytes *
                             ((cuMemcpy3DPeerAsync_ptsz_params*)params)->pCopy->Height *
                             ((cuMemcpy3DPeerAsync_ptsz_params*)params)->pCopy->Depth
                       : 0ULL;
        default:
            return 0ULL;
    }
}

static void copy_endpoints(nvbit_api_cuda_t cbid, void* params, int* src, int* dst) {
    *src = -1;
    *dst = -1;
    if (cbid == API_CUDA_cuMemcpyPeer) {
        cuMemcpyPeer_params* p = (cuMemcpyPeer_params*)params;
        *src = device_for_pointer(p->srcDevice);
        *dst = device_for_pointer(p->dstDevice);
    } else if (cbid == API_CUDA_cuMemcpyPeer_ptds) {
        cuMemcpyPeer_ptds_params* p = (cuMemcpyPeer_ptds_params*)params;
        *src = device_for_pointer(p->srcDevice);
        *dst = device_for_pointer(p->dstDevice);
    } else if (cbid == API_CUDA_cuMemcpyPeerAsync) {
        cuMemcpyPeerAsync_params* p = (cuMemcpyPeerAsync_params*)params;
        *src = device_for_pointer(p->srcDevice);
        *dst = device_for_pointer(p->dstDevice);
    } else if (cbid == API_CUDA_cuMemcpyPeerAsync_ptsz) {
        cuMemcpyPeerAsync_ptsz_params* p = (cuMemcpyPeerAsync_ptsz_params*)params;
        *src = device_for_pointer(p->srcDevice);
        *dst = device_for_pointer(p->dstDevice);
    } else if (cbid == API_CUDA_cuMemcpy3DPeer) {
        const CUDA_MEMCPY3D_PEER* p = ((cuMemcpy3DPeer_params*)params)->pCopy;
        if (p) {
            *src = device_for_pointer(p->srcDevice);
            *dst = device_for_pointer(p->dstDevice);
        }
    } else if (cbid == API_CUDA_cuMemcpy3DPeer_ptds) {
        const CUDA_MEMCPY3D_PEER* p = ((cuMemcpy3DPeer_ptds_params*)params)->pCopy;
        if (p) {
            *src = device_for_pointer(p->srcDevice);
            *dst = device_for_pointer(p->dstDevice);
        }
    } else if (cbid == API_CUDA_cuMemcpy3DPeerAsync) {
        const CUDA_MEMCPY3D_PEER* p = ((cuMemcpy3DPeerAsync_params*)params)->pCopy;
        if (p) {
            *src = device_for_pointer(p->srcDevice);
            *dst = device_for_pointer(p->dstDevice);
        }
    } else if (cbid == API_CUDA_cuMemcpy3DPeerAsync_ptsz) {
        const CUDA_MEMCPY3D_PEER* p = ((cuMemcpy3DPeerAsync_ptsz_params*)params)->pCopy;
        if (p) {
            *src = device_for_pointer(p->srcDevice);
            *dst = device_for_pointer(p->dstDevice);
        }
    }
}

static ContextBase* get_or_create_base(CUcontext ctx, CUstream stream) {
    for (auto& base : context_bases) {
        if (base.ctx == ctx) return &base;
    }

    ContextBase base;
    base.ctx = ctx;
    base.gpu = current_device();
    inside_tool_cuda = true;
    CUresult create_status = cuEventCreate(&base.event, CU_EVENT_DEFAULT);
    CUresult record_status = create_status == CUDA_SUCCESS
                                 ? cuEventRecord(base.event, stream)
                                 : create_status;
    inside_tool_cuda = false;
    if (create_status != CUDA_SUCCESS || record_status != CUDA_SUCCESS) {
        base.event = nullptr;
    }
    context_bases.push_back(base);
    return &context_bases.back();
}

static void begin_event_op(CUcontext ctx, nvbit_api_cuda_t cbid, void* params,
                           OpKind kind, const char* name, const char* label) {
    if (!event_proof || inside_tool_cuda || pending_op.active) return;

    CUstream stream = nullptr;
    if (kind == OpKind::Launch) {
        stream = launch_stream(cbid, params);
    } else {
        stream = copy_stream(cbid, params);
    }

    PendingOp pending;
    pending.active = true;
    pending.kind = kind;
    pending.seq = ++event_seq;
    pending.ctx = ctx;
    pending.stream = stream;
    pending.gpu = current_device();
    pending.host_enter_ns = host_now_ns();
    pending.name = label ? label : name;
    pending.bytes = kind == OpKind::Launch ? 0ULL : copy_bytes(cbid, params);
    if (kind != OpKind::Launch) {
        copy_endpoints(cbid, params, &pending.src_gpu, &pending.dst_gpu);
    }

    pthread_mutex_lock(&event_mutex);
    ContextBase* base = get_or_create_base(ctx, stream);
    pending.base = base ? base->event : nullptr;
    pthread_mutex_unlock(&event_mutex);

    inside_tool_cuda = true;
    CUresult start_create = cuEventCreate(&pending.start, CU_EVENT_DEFAULT);
    CUresult end_create = cuEventCreate(&pending.end, CU_EVENT_DEFAULT);
    CUresult start_record = CUDA_SUCCESS;
    if (start_create == CUDA_SUCCESS && end_create == CUDA_SUCCESS) {
        start_record = cuEventRecord(pending.start, stream);
    }
    inside_tool_cuda = false;

    if (start_create != CUDA_SUCCESS || end_create != CUDA_SUCCESS ||
        start_record != CUDA_SUCCESS || pending.base == nullptr) {
        if (pending.start) {
            inside_tool_cuda = true;
            cuEventDestroy(pending.start);
            inside_tool_cuda = false;
        }
        if (pending.end) {
            inside_tool_cuda = true;
            cuEventDestroy(pending.end);
            inside_tool_cuda = false;
        }
        return;
    }
    pending_op = pending;
}

static void finish_event_op() {
    if (!event_proof || inside_tool_cuda || !pending_op.active) return;

    PendingOp pending = pending_op;
    pending_op = PendingOp{};

    inside_tool_cuda = true;
    CUresult end_record = cuEventRecord(pending.end, pending.stream);
    inside_tool_cuda = false;
    if (end_record != CUDA_SUCCESS) return;

    EventOp op;
    op.kind = pending.kind;
    op.seq = pending.seq;
    op.ctx = pending.ctx;
    op.stream = pending.stream;
    op.base = pending.base;
    op.start = pending.start;
    op.end = pending.end;
    op.gpu = pending.gpu;
    op.src_gpu = pending.src_gpu;
    op.dst_gpu = pending.dst_gpu;
    op.bytes = pending.bytes;
    op.host_enter_ns = pending.host_enter_ns;
    op.host_exit_ns = host_now_ns();
    op.name = pending.name;

    pthread_mutex_lock(&event_mutex);
    event_ops.push_back(op);
    pthread_mutex_unlock(&event_mutex);
}

static const char* op_kind_name(OpKind kind) {
    switch (kind) {
        case OpKind::Launch: return "launch";
        case OpKind::PeerCopy: return "peer_copy";
        case OpKind::DeviceCopy: return "device_copy";
        default: return "unknown";
    }
}

static bool op_touches_launch_gpu(const EventOp& comm, const EventOp& launch) {
    if (comm.gpu == launch.gpu) return true;
    if (comm.src_gpu == launch.gpu || comm.dst_gpu == launch.gpu) return true;
    return false;
}

static float interval_overlap_ms(const EventOp& a, const EventOp& b) {
    if (!a.timing_valid || !b.timing_valid) return 0.0f;
    float lo = std::max(a.start_ms, b.start_ms);
    float hi = std::min(a.end_ms, b.end_ms);
    return hi > lo ? hi - lo : 0.0f;
}

static void finalize_event_proof() {
    if (!event_proof) return;

    inside_tool_cuda = true;
    for (auto& op : event_ops) {
        if (op.end == nullptr || op.start == nullptr || op.base == nullptr) continue;
        if (cuEventSynchronize(op.end) != CUDA_SUCCESS) continue;
        float start_ms = 0.0f;
        float end_ms = 0.0f;
        float duration_ms = 0.0f;
        if (cuEventElapsedTime(&start_ms, op.base, op.start) != CUDA_SUCCESS) continue;
        if (cuEventElapsedTime(&end_ms, op.base, op.end) != CUDA_SUCCESS) continue;
        if (cuEventElapsedTime(&duration_ms, op.start, op.end) != CUDA_SUCCESS) continue;
        op.start_ms = start_ms;
        op.end_ms = end_ms;
        op.duration_ms = duration_ms;
        op.timing_valid = true;
    }
    inside_tool_cuda = false;

    printf("[NVBPF PROOF] event_timeline ops=%zu\n", event_ops.size());
    for (const auto& op : event_ops) {
        if (!op.timing_valid) {
            printf("    op#%lu kind=%s gpu=%d stream=%p timing=invalid name=%s\n",
                   op.seq, op_kind_name(op.kind), op.gpu, op.stream, op.name.c_str());
            continue;
        }
        printf("    op#%lu kind=%s gpu=%d stream=%p start_ms=%.3f end_ms=%.3f dur_ms=%.3f",
               op.seq, op_kind_name(op.kind), op.gpu, op.stream,
               op.start_ms, op.end_ms, op.duration_ms);
        if (op.kind != OpKind::Launch) {
            printf(" bytes=%lu src_gpu=%d dst_gpu=%d", op.bytes, op.src_gpu, op.dst_gpu);
        }
        printf(" name=%s\n", op.name.c_str());
    }

    uint64_t candidates = 0;
    uint64_t overlaps = 0;
    uint64_t efficient = 0;
    for (const auto& launch : event_ops) {
        if (launch.kind != OpKind::Launch || !launch.timing_valid) continue;
        for (const auto& comm : event_ops) {
            if (comm.kind == OpKind::Launch || !comm.timing_valid) continue;
            if (!op_touches_launch_gpu(comm, launch)) continue;
            candidates++;
            float overlap = interval_overlap_ms(launch, comm);
            float denom = std::min(launch.duration_ms, comm.duration_ms);
            double ratio = denom > 0.0f ? (double)overlap / (double)denom : 0.0;
            const char* order = "overlap";
            if (comm.end_ms <= launch.start_ms) order = "comm_before_kernel";
            if (launch.end_ms <= comm.start_ms) order = "kernel_before_comm";
            if (overlap > 0.0f) overlaps++;
            if (ratio >= min_comm_overlap_ratio) efficient++;
            printf("[NVBPF PROOF] comm_kernel_pair launch#%lu comm#%lu gpu=%d order=%s overlap_ms=%.3f ratio=%.3f verdict=%s\n",
                   launch.seq, comm.seq, launch.gpu, order, overlap, ratio,
                   ratio >= min_comm_overlap_ratio ? "PASS" : "WEAK");
        }
    }
    printf("[NVBPF PROOF] overlap_summary candidates=%lu overlapped=%lu efficient=%lu min_ratio=%.3f\n",
           candidates, overlaps, efficient, min_comm_overlap_ratio);
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    if (getenv("NVBPF_FORCE_DEVICE_ALLOC") != nullptr) {
        setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    }
    pthread_mutex_init(&proof_mutex, nullptr);
    pthread_mutex_init(&event_mutex, nullptr);

    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    device_phases = getenv("NVBPF_PROOF_DEVICE_PHASES") == nullptr ||
                    strcmp(getenv("NVBPF_PROOF_DEVICE_PHASES"), "0") != 0;
    sync_for_phases = getenv("NVBPF_PROOF_SYNC_FOR_PHASES") == nullptr ||
                      strcmp(getenv("NVBPF_PROOF_SYNC_FOR_PHASES"), "0") != 0;
    event_proof = getenv("NVBPF_PROOF_EVENTS") != nullptr &&
                  strcmp(getenv("NVBPF_PROOF_EVENTS"), "0") != 0;
    source_markers = getenv("NVBPF_PROOF_SOURCE_MARKERS") == nullptr ||
                     strcmp(getenv("NVBPF_PROOF_SOURCE_MARKERS"), "0") != 0;
    if (const char* env = getenv("NVBPF_PROOF_LOW_LANES")) {
        low_active_threshold = (uint32_t)strtoul(env, nullptr, 0);
    }
    if (const char* env = getenv("NVBPF_PROOF_MIN_PHASE_OVERLAP")) {
        min_phase_overlap_ratio = strtod(env, nullptr);
    }
    if (const char* env = getenv("NVBPF_PROOF_MIN_COMM_OVERLAP")) {
        min_comm_overlap_ratio = strtod(env, nullptr);
    }
    expected_order = parse_phase_order(getenv("NVBPF_PHASE_ORDER"));

    printf("[NVBPF PROOF] loaded filter=%s device_phases=%d event_proof=%d sync_for_phases=%d source_markers=%d\n",
           kernel_name_filter.empty() ? "<all>" : kernel_name_filter.c_str(),
           device_phases ? 1 : 0, event_proof ? 1 : 0, sync_for_phases ? 1 : 0,
           source_markers ? 1 : 0);
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (inside_tool_cuda) return;

    bool is_launch = nvbpf_is_launch_event(cbid);
    bool async_comm = is_async_comm_cbid(cbid);

    if (!is_exit) {
        if (is_launch) {
            CUfunction func = nvbpf_get_launch_func(cbid, params);
            const char* func_name = nvbit_get_func_name(ctx, func);
            bool match = csv_or_substring_match(func_name, kernel_name_filter);
            if (match) {
                begin_event_op(ctx, cbid, params, OpKind::Launch, name, func_name);
            }
            if (device_phases) {
                pthread_mutex_lock(&proof_mutex);
                if (match) {
                    instrument_function_if_needed(ctx, func);
                    reset_phase_state();
                    nvbit_enable_instrumented(ctx, func, true);
                } else {
                    nvbit_enable_instrumented(ctx, func, false);
                    pthread_mutex_unlock(&proof_mutex);
                }
            }
        } else if (async_comm) {
            begin_event_op(ctx, cbid, params,
                           is_peer_copy_cbid(cbid) ? OpKind::PeerCopy : OpKind::DeviceCopy,
                           name, name);
        }
        return;
    }

    if (is_launch || async_comm) {
        finish_event_op();
    }
    if (is_peer_copy_cbid(cbid)) peer_copy_seen++;
    if (is_event_record_cbid(cbid)) event_record_seen++;
    if (is_stream_wait_cbid(cbid)) stream_wait_seen++;
    if (is_host_sync_cbid(cbid)) host_sync_seen++;

    if (!is_launch || !device_phases) return;

    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);
    bool match = csv_or_substring_match(func_name, kernel_name_filter);

    if (!is_exit) return;
    if (!match) return;

    /*
     * NVBit launch exit is the API enqueue exit. For phase proof, we
     * intentionally synchronize to make device maps readable. Use
     * NVBPF_PROOF_DEVICE_PHASES=0 with NVBPF_PROOF_EVENTS=1 for the
     * non-perturbing communication-overlap pass.
     */
    if (sync_for_phases) {
        inside_tool_cuda = true;
        cudaDeviceSynchronize();
        inside_tool_cuda = false;
    }
    int gpu = current_device();
    uint64_t seq = ++launch_seq;
    matched_launches++;
    if (gpu >= 0 && gpu < (int)kMaxGpu) gpu_launches[gpu]++;
    print_phase_proof(seq, gpu, func_name);
    pthread_mutex_unlock(&proof_mutex);
}

void nvbit_at_term() {
    finalize_event_proof();

    printf("[NVBPF PROOF] summary launches=%lu peer_copy=%lu event_record=%lu stream_wait=%lu host_sync=%lu\n",
           matched_launches, peer_copy_seen, event_record_seen, stream_wait_seen, host_sync_seen);
    for (int gpu = 0; gpu < (int)kMaxGpu; ++gpu) {
        if (gpu_launches[gpu] > 0) {
            printf("  gpu=%d phase_launches=%lu\n", gpu, gpu_launches[gpu]);
        }
    }

    inside_tool_cuda = true;
    for (auto& op : event_ops) {
        if (op.start) cuEventDestroy(op.start);
        if (op.end) cuEventDestroy(op.end);
    }
    for (auto& base : context_bases) {
        if (base.event) cuEventDestroy(base.event);
    }
    inside_tool_cuda = false;
    printf("[NVBPF PROOF] terminated\n");
}
