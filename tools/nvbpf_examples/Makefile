# NV-BPF Examples Makefile
# SPDX-License-Identifier: BSD-3-Clause

NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool!)
endif

PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc)

ifeq ($(PTXAS_VER_CHECK), 0)
MAXRREGCOUNT_FLAG=-maxrregcount=24
else
MAXRREGCOUNT_FLAG=
endif

NVBIT_PATH=../../core
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

ARCH?=sm_89

# Flags for host-side tool files (NVBit callbacks, maps, hook registration)
HOST_FLAGS=-dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no \
           -Xcompiler -Wall -arch=$(ARCH) -O3 -Xcompiler -fPIC

# Flags for device-side hook files (must keplesep device functions for NVBit injection)
HOOK_FLAGS=$(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch \
           --keep-device-functions -arch=$(ARCH) \
           -Xcompiler -Wall -Xcompiler -fPIC -c

all: instr_count.so mem_trace.so sm_profiler.so attention_debug.so \
     kernel_summary.so attention_trace.so sampling_mem_trace.so branch_divergence.so \
     nvlink_trace.so gemm_wavefit_trace.so gemm_orchestration_map.so \
     epilogue_fusion_trace.so tail_fragment_tracker.so reuse_distance_profiler.so \
     pipeline_depth_estimator.so tile_lifetime_tracker.so \
     bank_conflict_suspicion.so register_pressure_distortion_meter.so \
     cta_role_classifier.so \
     sampling_mem_events.so peer_copy_trace.so

# ── instr_count ──────────────────────────────────────────────────────────────
instr_count.so: instr_count.o instr_count_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 instr_count.o instr_count_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

instr_count.o: instr_count.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

instr_count_hooks.o: instr_count_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── mem_trace ────────────────────────────────────────────────────────────────
mem_trace.so: mem_trace.o mem_trace_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 mem_trace.o mem_trace_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

mem_trace.o: mem_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

mem_trace_hooks.o: mem_trace_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── sm_profiler ───────────────────────────────────────────────────────────────
sm_profiler.so: sm_profiler.o sm_profiler_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 sm_profiler.o sm_profiler_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

sm_profiler.o: sm_profiler.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

sm_profiler_hooks.o: sm_profiler_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── attention_debug ───────────────────────────────────────────────────────────
attention_debug.so: attention_debug.o attention_debug_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 attention_debug.o attention_debug_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@
attention_debug.o: attention_debug.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@
attention_debug_hooks.o: attention_debug_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── kernel_summary ────────────────────────────────────────────────────────────
kernel_summary.so: kernel_summary.o kernel_summary_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 kernel_summary.o kernel_summary_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

kernel_summary.o: kernel_summary.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

kernel_summary_hooks.o: kernel_summary_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── attention_trace ───────────────────────────────────────────────────────────
attention_trace.so: attention_trace.o attention_trace_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 attention_trace.o attention_trace_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

attention_trace.o: attention_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

attention_trace_hooks.o: attention_trace_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── sampling_mem_trace ────────────────────────────────────────────────────────
sampling_mem_trace.so: sampling_mem_trace.o sampling_mem_trace_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 sampling_mem_trace.o sampling_mem_trace_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

sampling_mem_trace.o: sampling_mem_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

sampling_mem_trace_hooks.o: sampling_mem_trace_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── branch_divergence ─────────────────────────────────────────────────────────
branch_divergence.so: branch_divergence.o branch_divergence_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 branch_divergence.o branch_divergence_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

branch_divergence.o: branch_divergence.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

branch_divergence_hooks.o: branch_divergence_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── nvlink_trace ──────────────────────────────────────────────────────────────
nvlink_trace.so: nvlink_trace.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 nvlink_trace.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

nvlink_trace.o: nvlink_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

# ── gemm_wavefit_trace ────────────────────────────────────────────────────────
gemm_wavefit_trace.so: gemm_wavefit_trace.o gemm_wavefit_trace_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 gemm_wavefit_trace.o gemm_wavefit_trace_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

gemm_wavefit_trace.o: gemm_wavefit_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

gemm_wavefit_trace_hooks.o: gemm_wavefit_trace_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── gemm_orchestration_map ────────────────────────────────────────────────────
gemm_orchestration_map.so: gemm_orchestration_map.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 gemm_orchestration_map.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

gemm_orchestration_map.o: gemm_orchestration_map.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

# ── epilogue_fusion_trace ─────────────────────────────────────────────────────
epilogue_fusion_trace.so: epilogue_fusion_trace.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 epilogue_fusion_trace.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

epilogue_fusion_trace.o: epilogue_fusion_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

# ── tail_fragment_tracker ─────────────────────────────────────────────────────
tail_fragment_tracker.so: tail_fragment_tracker.o tail_fragment_tracker_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 tail_fragment_tracker.o tail_fragment_tracker_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

tail_fragment_tracker.o: tail_fragment_tracker.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

tail_fragment_tracker_hooks.o: tail_fragment_tracker_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── reuse_distance_profiler ───────────────────────────────────────────────────
reuse_distance_profiler.so: reuse_distance_profiler.o reuse_distance_profiler_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 reuse_distance_profiler.o reuse_distance_profiler_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

reuse_distance_profiler.o: reuse_distance_profiler.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

reuse_distance_profiler_hooks.o: reuse_distance_profiler_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── pipeline_depth_estimator ──────────────────────────────────────────────────
pipeline_depth_estimator.so: pipeline_depth_estimator.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 pipeline_depth_estimator.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

pipeline_depth_estimator.o: pipeline_depth_estimator.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

# ── tile_lifetime_tracker ─────────────────────────────────────────────────────
tile_lifetime_tracker.so: tile_lifetime_tracker.o tile_lifetime_tracker_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 tile_lifetime_tracker.o tile_lifetime_tracker_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

tile_lifetime_tracker.o: tile_lifetime_tracker.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

tile_lifetime_tracker_hooks.o: tile_lifetime_tracker_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── bank_conflict_suspicion ───────────────────────────────────────────────────
bank_conflict_suspicion.so: bank_conflict_suspicion.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 bank_conflict_suspicion.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

bank_conflict_suspicion.o: bank_conflict_suspicion.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

# ── register_pressure_distortion_meter ────────────────────────────────────────
register_pressure_distortion_meter.so: register_pressure_distortion_meter.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 register_pressure_distortion_meter.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

register_pressure_distortion_meter.o: register_pressure_distortion_meter.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

# ── cta_role_classifier ───────────────────────────────────────────────────────
cta_role_classifier.so: cta_role_classifier.o cta_role_classifier_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 cta_role_classifier.o cta_role_classifier_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

cta_role_classifier.o: cta_role_classifier.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

cta_role_classifier_hooks.o: cta_role_classifier_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# -- BEGIN NVBPF_PY sampling_mem_events --
# ── sampling_mem_events (nvbpf_py) ─────────────────────────────────────────────
sampling_mem_events.so: sampling_mem_events.o sampling_mem_events_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 sampling_mem_events.o sampling_mem_events_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

sampling_mem_events.o: sampling_mem_events.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

sampling_mem_events_hooks.o: sampling_mem_events_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@
# -- END NVBPF_PY sampling_mem_events --

# -- BEGIN NVBPF_PY peer_copy_trace --
# ── peer_copy_trace (nvbpf_py) ─────────────────────────────────────────────
peer_copy_trace.so: peer_copy_trace.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 peer_copy_trace.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

peer_copy_trace.o: peer_copy_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@
# -- END NVBPF_PY peer_copy_trace --

clean:
	rm -f *.so *.o
