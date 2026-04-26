# Writing A New NV-BPF Tool

This directory is the fastest place to learn the NV-BPF authoring pattern.

If you want a Python-first starting point instead of handwritten `.cu` files,
see [`../../nvbpf_py/README.md`](../../nvbpf_py/README.md). The Python DSL can
also generate directly into this directory and patch the shared `Makefile`
automatically with:

```bash
python3 -m nvbpf_py.cli build --integrate-examples --force your_spec.py
```

Most tools here follow the same structure:

1. A host-side tool file, for example `kernel_summary.cu`
2. A device hook file, for example `kernel_summary_hooks.cu`
3. A `Makefile` target that builds `tool.so`

The workflow is:

1. declare maps in the host file
2. inject a small device helper into matching instructions
3. reset maps before launch
4. read maps after launch
5. print a human-readable summary

## Tool Types

There are two common styles in this directory.

### 1. Hook + map tools

Use this when you want to count instructions, sample memory, or track per-SM state.

Examples:
- `instr_count.cu` + `instr_count_hooks.cu`
- `kernel_summary.cu` + `kernel_summary_hooks.cu`
- `sampling_mem_trace.cu` + `sampling_mem_trace_hooks.cu`
- `branch_divergence.cu` + `branch_divergence_hooks.cu`
- `gemm_wavefit_trace.cu` + `gemm_wavefit_trace_hooks.cu`
- `tail_fragment_tracker.cu` + `tail_fragment_tracker_hooks.cu`
- `reuse_distance_profiler.cu` + `reuse_distance_profiler_hooks.cu`
- `tile_lifetime_tracker.cu` + `tile_lifetime_tracker_hooks.cu`
- `cta_role_classifier.cu` + `cta_role_classifier_hooks.cu`

These tools usually use:
- `BPF_ARRAY(...)`
- `BPF_PERCPU_ARRAY(...)`
- `BPF_RINGBUF(...)`

### 2. Host-only tools

Use this when you want to observe CUDA API events or launch metadata without injecting device code.

Example:
- `nvlink_trace.cu`
- `gemm_orchestration_map.cu`
- `epilogue_fusion_trace.cu`
- `pipeline_depth_estimator.cu`
- `bank_conflict_suspicion.cu`
- `register_pressure_distortion_meter.cu`

This style does not need a `_hooks.cu` file.

## The Smallest Useful Pattern

For most new tools, copy one of these first:

- copy `kernel_summary.*` if you want per-kernel counters
- copy `sampling_mem_trace.*` if you want memory events
- copy `branch_divergence.*` if you want branch-only logic
- copy `tail_fragment_tracker.*` if you want edge/tail inefficiency estimates
- copy `reuse_distance_profiler.*` if you want sampled memory-line reuse estimates
- copy `tile_lifetime_tracker.*` if you want producer-to-store lifetime estimates
- copy `cta_role_classifier.*` if you want sampled CTA behavior families
- copy `nvlink_trace.cu` if you want host-side API tracing
- copy `gemm_wavefit_trace.*` if you want CTA wave-fit or per-SM CTA spread
- copy `gemm_orchestration_map.cu` if you want host-side kernel neighborhood analysis
- copy `epilogue_fusion_trace.cu` if you want fused-vs-separate epilogue analysis
- copy `pipeline_depth_estimator.cu` if you want producer/consumer staging estimates
- copy `bank_conflict_suspicion.cu` if you want shared-memory conflict heuristics
- copy `register_pressure_distortion_meter.cu` if you want occupancy-vs-register tradeoff estimates

Then change only:

1. the map declarations
2. the opcode or instruction selection logic
3. the printed summary

## Step By Step

### 1. Add a new host file

Example skeleton:

```cpp
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

BPF_ARRAY(total_events, uint64_t, 1);

extern "C" __device__ __noinline__ void my_count(int pred, uint64_t pcounter);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;

static void reset_state() {
    total_events.reset();
}
```

The host file is where you:
- declare maps
- decide which instructions to instrument
- enable instrumentation at kernel launch
- read and print results after kernel completion

### 2. Add a hook file

Example skeleton:

```cpp
#include <stdint.h>
#include "utils/utils.h"

extern "C" __device__ __noinline__ void my_count(int pred, uint64_t pcounter) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    if (first_laneid == laneid && __popc(predicate_mask) > 0) {
        atomicAdd((unsigned long long*)pcounter, 1ULL);
    }
}
```

The hook should stay small. Good hooks usually:
- do one thing
- update one or two counters
- avoid complex logic unless really needed

### 3. Instrument only the instructions you care about

Typical pattern:

```cpp
for (auto* instr : instrs) {
    if (!matches(instr)) continue;
    nvbit_insert_call(instr, "my_count", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(instr);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&total_events.data[0]);
}
```

Common selectors:
- `instr->isLoad()`
- `instr->isStore()`
- `instr->getOpcodeShort()`
- `instr->getMemorySpace()`

### 4. Reset before launch, read after launch

Typical pattern:

```cpp
if (!is_exit) {
    instrument_function_if_needed(ctx, func);
    reset_state();
    nvbit_enable_instrumented(ctx, func, true);
} else {
    cudaDeviceSynchronize();
    printf("count=%lu\n", *total_events.lookup(0));
}
```

### 5. Add a Makefile target

Hook-based tool:

```make
my_tool.so: my_tool.o my_tool_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 my_tool.o my_tool_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

my_tool.o: my_tool.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

my_tool_hooks.o: my_tool_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@
```

Host-only tool:

```make
my_tool.so: my_tool.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 my_tool.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@
```

## Real Example: Atomic Counter

If you want a tool that counts only atomic instructions, the changes are small:

1. copy `kernel_summary.cu` and `kernel_summary_hooks.cu`
2. keep only one map:
   `BPF_ARRAY(total_atomics, uint64_t, 1);`
3. change the selector to:

```cpp
static bool is_atomic_opcode(const char* opcode) {
    return strncmp(opcode, "ATOM", 4) == 0 || strncmp(opcode, "RED", 3) == 0;
}
```

4. print:

```cpp
printf("[NVBPF] %s atomics=%lu\n", func_name, *total_atomics.lookup(0));
```

That is usually enough to get a first working version.

## Recommended Defaults

If you want tools that are easy for other people to use, prefer:

- one summary line per kernel
- `NVBPF_KERNEL_FILTER` support
- sampling instead of tracing every event
- `ACK_CTX_INIT_LIMITATION=1`
- `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` for map-heavy tools

## Build And Run

From this directory:

```bash
make clean && make kernel_summary.so
```

Then from the repo root:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_KERNEL_FILTER=sgemm \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/kernel_summary.so \
python3 test-apps/attention_pytorch/attention_pytorch.py --backend math
```

For GEMM wave fit:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_GEMM_FILTER=sgemm \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/gemm_wavefit_trace.so \
python3 test-apps/attention_pytorch/attention_pytorch.py --backend math
```

By default, `gemm_wavefit_trace` now prints one compact summary per unique
kernel shape. Set `NVBPF_VERBOSE=1` to restore the per-launch detail dump, and
`NVBPF_FULL_NAMES=1` if you also want the full kernel names.

For GEMM orchestration:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_GEMM_FILTER=sgemm \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/gemm_orchestration_map.so \
python3 test-apps/attention_pytorch/attention_pytorch.py --backend math
```

`gemm_orchestration_map` also defaults to compact grouped summaries. Use
`NVBPF_VERBOSE=1` for the raw neighborhood listing and `NVBPF_FULL_NAMES=1` for
unshortened kernel names.

For epilogue fusion:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_GEMM_FILTER=sgemm \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/epilogue_fusion_trace.so \
python3 test-apps/attention_pytorch/attention_pytorch.py --backend math
```

`epilogue_fusion_trace` prints grouped per-kernel fusion summaries by default.
Use `NVBPF_EPILOGUE_WINDOW=3` to change how many post-kernel launches are
considered before the analysis cuts off, `NVBPF_VERBOSE=1` for the raw
per-launch epilogue listing, and `NVBPF_FULL_NAMES=1` for unshortened kernel
names.

For tail-fragment inefficiency:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_KERNEL_FILTER=sgemm \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/tail_fragment_tracker.so \
python3 test-apps/attention_pytorch/attention_pytorch.py --backend math
```

`tail_fragment_tracker` prints grouped per-kernel summaries by default. Use
`NVBPF_TAIL_ACTIVE_LANES=16` to change the "low active lanes" threshold,
`NVBPF_VERBOSE=1` for the per-launch detail dump, and `NVBPF_FULL_NAMES=1` for
unshortened kernel names.

## Plotting Tool Output

You can turn NV-BPF logs into CSV summaries and SVG plots with
[`../nvbpf_analysis`](../nvbpf_analysis).

Example:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_GEMM_FILTER=sgemm \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/gemm_wavefit_trace.so \
python3 test-apps/attention_pytorch/attention_pytorch.py --backend math \
  | tee gemm_wavefit_math.log

python3 tools/nvbpf_analysis/plot_tool_output.py \
  --input gemm_wavefit_math.log \
  --output-dir tools/nvbpf_analysis/out
```

The current plotter understands:

- `kernel_summary`
- `sampling_mem_trace`
- `gemm_wavefit_trace`
- `gemm_orchestration_map`
- `epilogue_fusion_trace`
- `tail_fragment_tracker`

## Which Example Should I Copy?

- `instr_count`: simplest counting pattern
- `kernel_summary`: best general template
- `sampling_mem_trace`: best memory-event template
- `branch_divergence`: best branch-analysis template
- `tail_fragment_tracker`: best tail/edge inefficiency template
- `reuse_distance_profiler`: best sampled line-reuse template
- `tile_lifetime_tracker`: best producer-to-store lifetime template
- `cta_role_classifier`: best sampled CTA behavior template
- `attention_trace`: best workload-specific template
- `gemm_wavefit_trace`: best wave-quantization template
- `gemm_orchestration_map`: best kernel-neighborhood template
- `epilogue_fusion_trace`: best fused-vs-separate epilogue template
- `pipeline_depth_estimator`: best producer/consumer staging template
- `bank_conflict_suspicion`: best shared-memory conflict heuristic template
- `register_pressure_distortion_meter`: best occupancy-vs-register tradeoff template
- `nvlink_trace`: best host-only trace template

## Rule Of Thumb

If the question is:

- "count something in device code" -> use maps + hooks
- "sample or stream device events" -> use maps + hooks + ring buffer
- "watch CUDA API behavior" -> use a host-only tool

Start from the closest existing example, not from scratch.
