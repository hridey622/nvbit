# NV-BPF Project Report

## 1. What This Project Is

`nvbpf` is a GPU instrumentation project built on top of NVBit.

Its main idea is:

- keep NVBit as the low-level binary instrumentation engine
- wrap it in a more structured, eBPF-like model
- make GPU instrumentation easier to author, reuse, and explain

In this repo, that shows up in two main layers:

1. a C++/CUDA NV-BPF runtime layer in `core/`
2. a Python-first DSL in `nvbpf_py/` that can generate real NV-BPF tools

So the project is not just "a few example NVBit tools". It is now a small instrumentation framework with:

- reusable GPU-side map abstractions
- reusable host-side tool patterns
- handwritten analysis tools
- Python-generated analysis tools
- CUDA, PyTorch, and CuTe/CUTLASS test workloads

## 2. High-Level Repo Layout

### Core runtime

- `core/nvbpf.h`
- `core/nvbpf_maps.h`
- `core/nvbpf_helpers.h`
- `core/nvbpf_hooks.h`
- `core/nvbpf_loader.h`
- `core/nvbpf_types.h`
- `core/nvbpf_wrapper.h`
- `core/nvbit.h`
- `core/nvbit_tool.h`
- `core/libnvbit.a`

This is the base layer that provides the NV-BPF programming model.

### Handwritten NV-BPF tools

- `tools/nvbpf_examples/`

This is the main directory for authored tools built directly in CUDA/C++ on top of the NV-BPF layer.

### Original NVBit / reference-style tools

- `tools/instr_count/`
- `tools/instr_count_bb/`
- `tools/instr_count_cuda_graph/`
- `tools/mem_trace/`
- `tools/mem_printf2/`
- `tools/mov_replace/`
- `tools/opcode_hist/`
- `tools/record_reg_vals/`

These are useful as reference material and for comparing raw NVBit style vs NV-BPF style.

### Python DSL

- `nvbpf_py/`
- `tools/nvbpf_py_examples/`

This is the Python-first tool generator and its example specs.

### Analysis and visualization helpers

- `tools/nvbpf_analysis/`

This directory now contains log post-processing helpers that can turn NV-BPF
stdout logs into CSV summaries and SVG plots.

### Workloads and demos

- `test-apps/vectoradd/`
- `test-apps/conv_pytorch/`
- `test-apps/attention_pytorch/`
- `test-apps/attention_cute/`
- `test-apps/axpy_relu_cute/`
- `test-apps/elementwise_add_cute/`
- `elementwise_add.ipynb`

These are the programs you use to exercise and validate the instrumentation.

## 3. What The Core NV-BPF Layer Provides

The `core/` layer gives the project its main instrumentation abstractions.

### 3.1 Map abstractions

The project supports eBPF-like map patterns for GPU tooling:

- `BPF_ARRAY(...)`
- `BPF_HASH(...)`
- `BPF_PERCPU_ARRAY(...)`
- `BPF_RINGBUF(...)`

These let tools maintain:

- scalar counters
- keyed state
- per-SM counters
- GPU-to-CPU event streams

### 3.2 Hook / probe model

The project exposes familiar probe-style entry points such as:

- `SEC_KPROBE(...)`
- `SEC_KRETPROBE(...)`
- `SEC_TRACEPOINT_INSTR(...)`
- `SEC_TRACEPOINT_MEM_LOAD(...)`
- `SEC_TRACEPOINT_MEM_STORE(...)`
- `SEC_TRACEPOINT_OPCODE(...)`

This gives authors a structured way to attach instrumentation logic to:

- kernel entry / exit
- all instructions
- specific opcodes
- memory loads and stores

### 3.3 Helpers

The helper layer provides contextual data and convenience operations such as:

- current SM
- current warp
- timing / clock helpers
- safe memory reads
- debug printing

### 3.4 Runtime / loader integration

The project also includes the plumbing needed to:

- identify kernel launches
- fetch the active `CUfunction`
- insert instrumentation callbacks
- manage related functions
- read results after launch

That is what makes the higher-level tools possible.

## 4. What The Project Can Do Today

Today, this repo can instrument and analyze GPU programs across several levels.

### 4.1 Basic kernel instrumentation

It can:

- count instructions
- count loads and stores
- count branches
- measure per-SM activity
- sample memory events
- stream sampled events through ring buffers

### 4.2 Kernel-structure summaries

It can summarize kernels in terms of:

- launch geometry
- register count
- shared memory use
- instruction count
- load/store count
- opcode-family counts
- active SM spread

### 4.3 Branch and tail behavior

It can analyze:

- branch divergence
- partial-warp behavior
- low-active-lane sites
- wasted lanes in tail / ragged launches

### 4.4 GEMM / attention analysis

It can analyze:

- GEMM wave fit / wave quantization
- CTA distribution across SMs
- kernel orchestration around GEMMs
- likely fused vs separate epilogue behavior
- fused attention / FMHA neighborhood structure

### 4.5 CUDA API / topology-level analysis

It can trace:

- peer-copy APIs
- launch neighborhoods
- likely NVLink / multi-GPU traffic correlation

### 4.6 Multi-framework workloads

It is already set up to observe:

- CUDA C++ apps
- PyTorch CUDA workloads
- CuTe / CUTLASS DSL Python workloads

That is one of the strongest parts of the repo: the same instrumentation ideas can be reused across all three.

### 4.7 Post-processing and visualization

It can now also post-process tool output into:

- CSV summaries
- SVG dashboards / heatmaps for grouped tool logs

This is especially useful for:

- GEMM wave-fit comparisons
- orchestration and epilogue-fusion comparisons
- tail-fragment / edge-inefficiency comparisons
- compact per-kernel structure summaries

## 5. Handwritten NV-BPF Tools In `tools/nvbpf_examples`

This directory is the main curated tool collection.

### 5.1 Core general-purpose tools

| Tool | Files | What it does |
| --- | --- | --- |
| `instr_count` | `instr_count.cu`, `instr_count_hooks.cu` | Counts warp-level instruction activity |
| `mem_trace` | `mem_trace.cu`, `mem_trace_hooks.cu` | Streams memory access events |
| `sm_profiler` | `sm_profiler.cu`, `sm_profiler_hooks.cu` | Tracks per-SM instruction and entry activity |
| `kernel_summary` | `kernel_summary.cu`, `kernel_summary_hooks.cu` | Produces a compact structural summary for each kernel |
| `sampling_mem_trace` | `sampling_mem_trace.cu`, `sampling_mem_trace_hooks.cu` | Samples memory accesses instead of tracing all of them |
| `branch_divergence` | `branch_divergence.cu`, `branch_divergence_hooks.cu` | Estimates divergence and predicate behavior |
| `tail_fragment_tracker` | `tail_fragment_tracker.cu`, `tail_fragment_tracker_hooks.cu` | Estimates tail/edge inefficiency and wasted-lane behavior |
| `reuse_distance_profiler` | `reuse_distance_profiler.cu`, `reuse_distance_profiler_hooks.cu` | Estimates sampled memory-line reuse distance within warps |
| `tile_lifetime_tracker` | `tile_lifetime_tracker.cu`, `tile_lifetime_tracker_hooks.cu` | Estimates producer-to-store tile lifetimes and math per lifetime window |
| `cta_role_classifier` | `cta_role_classifier.cu`, `cta_role_classifier_hooks.cu` | Classifies sampled CTAs into compute/memory/control/edge roles |

### 5.2 Workload-specific or domain-specific tools

| Tool | Files | What it does |
| --- | --- | --- |
| `attention_debug` | `attention_debug.cu`, `attention_debug_hooks.cu` | Debug-oriented attention kernel instrumentation |
| `attention_trace` | `attention_trace.cu`, `attention_trace_hooks.cu` | Attention-focused tracing |
| `gemm_wavefit_trace` | `gemm_wavefit_trace.cu`, `gemm_wavefit_trace_hooks.cu` | Estimates resident-wave fit, `fill_fraction`, and CTA spread |
| `gemm_orchestration_map` | `gemm_orchestration_map.cu` | Host-only neighborhood analysis around GEMM/attention kernels |
| `epilogue_fusion_trace` | `epilogue_fusion_trace.cu` | Detects likely fused vs separate epilogue behavior |
| `pipeline_depth_estimator` | `pipeline_depth_estimator.cu` | Estimates producer/consumer staging depth and overlap score from the instruction stream |
| `bank_conflict_suspicion` | `bank_conflict_suspicion.cu` | Heuristically scores shared-memory bank-conflict risk |
| `register_pressure_distortion_meter` | `register_pressure_distortion_meter.cu` | Estimates whether registers and local-memory traffic are likely distorting residency |

### 5.3 Topology / multi-GPU / API correlation tools

| Tool | Files | What it does |
| --- | --- | --- |
| `nvlink_trace` | `nvlink_trace.cu` | Host-only peer-copy and topology correlation trace |
| `peer_copy_trace` | `peer_copy_trace.cu` | Python-generated host-only peer-copy trace |

### 5.4 Event and DSL-generated integrated examples

| Tool | Files | What it does |
| --- | --- | --- |
| `sampling_mem_events` | `sampling_mem_events.cu`, `sampling_mem_events_hooks.cu` | DSL-generated sampled memory event tool |
| `peer_copy_trace` | `peer_copy_trace.cu` | DSL-generated host-only API trace |

## 6. Python DSL: What It Can Do Now

The Python DSL in `nvbpf_py/` is one of the biggest upgrades in this repo.

It allows authors to define tools in Python and generate:

- host `.cu`
- hook `._hooks.cu`
- `Makefile`
- optional integrated example targets
- compiled `.so` outputs

### 6.1 Core Python DSL building blocks

The DSL supports:

- `@tool(...)`
- `counter(...)`
- `array(...)`
- `percpu_array(...)`
- `event(...)`
- `@hook(...)`
- `@on_launch_enter()`
- `@on_launch_exit()`
- `api_trace(...)`

### 6.2 What Python hooks can do

Restricted Python hook bodies currently support:

- `if` / `else`
- `for ... in range(...)`
- `break`
- `continue`
- `return`
- `pass`
- local assignments
- augmented assignments
- `ctx.count(...)`
- `ctx.atomic_add(...)`
- `ctx.map_get(...)`
- `ctx.map_set(...)`
- `ctx.emit(...)`

### 6.3 Hook context available in Python

The hook DSL can use:

- `ctx.pred`
- `ctx.addr`
- `ctx.is_load`
- `ctx.active_mask`
- `ctx.predicate_mask`
- `ctx.lane_id`
- `ctx.active_lanes`
- `ctx.sm_id`
- `ctx.warp_id`
- `ctx.cta_id_x`
- `ctx.cta_id_y`
- `ctx.cta_id_z`
- `ctx.grid_dim_x`
- `ctx.grid_dim_y`
- `ctx.grid_dim_z`
- `ctx.block_dim_x`
- `ctx.block_dim_y`
- `ctx.block_dim_z`

### 6.4 Host-side launch reporting available in Python

The launch callbacks support helpers like:

- `counter_value(...)`
- `map_value(...)`
- `kernel_name()`
- `short_kernel_name()`
- `grid_dim_x()`, `grid_dim_y()`, `grid_dim_z()`
- `block_dim_x()`, `block_dim_y()`, `block_dim_z()`
- `regs()`
- `smem_static()`
- `smem_dynamic()`

### 6.5 Built-in high-level analyzers exposed through Python

The DSL can directly generate specialized host analyzers for:

- `gemm_wavefit(...)`
- `gemm_orchestration_map(...)`
- `epilogue_fusion_trace(...)`
- `tail_fragment_tracker(...)`

That means Python is no longer limited to simple counters only.

### 6.6 CLI workflow

The CLI supports:

- `build`
- `build --compile`
- `build --integrate-examples`
- `run`

So you can go from Python spec to running `.so` with one command.

## 7. Python DSL Example Specs In The Repo

These files are good starting points for tool authors.

| File | What it demonstrates |
| --- | --- |
| `tools/nvbpf_py_examples/minimal_load_report.py` | Smallest practical Python-authored tool |
| `tools/nvbpf_py_examples/python_only_report.py` | Counter + map + launch-enter / launch-exit reporting |
| `tools/nvbpf_py_examples/loop_bucket_report.py` | Loops, map reads/writes, bucketed summaries |
| `tools/nvbpf_py_examples/atomic_count.py` | Opcode-based instruction counting |
| `tools/nvbpf_py_examples/memory_mix.py` | Load/store mix summaries |
| `tools/nvbpf_py_examples/sampling_mem_events.py` | Event schemas + sampled memory reporting |
| `tools/nvbpf_py_examples/peer_copy_trace.py` | Host-only CUDA API trace |
| `tools/nvbpf_py_examples/gemm_wavefit.py` | GEMM wave-fit analyzer |
| `tools/nvbpf_py_examples/gemm_orchestration.py` | GEMM neighborhood / orchestration analyzer |
| `tools/nvbpf_py_examples/epilogue_fusion.py` | Epilogue fusion analyzer |
| `tools/nvbpf_py_examples/tail_fragment.py` | Tail inefficiency analyzer |

## 8. Demo / Test Workloads Included In The Repo

The repo contains multiple workloads to drive the tools.

### 8.1 CUDA C++ workloads

| File | What it is useful for |
| --- | --- |
| `test-apps/vectoradd/vectoradd.cu` | Basic CUDA kernel instrumentation |
| `test-apps/vectoradd/matrix_add.cu` | Simple memory and instruction-count tests |

### 8.2 PyTorch workloads

| File | What it is useful for |
| --- | --- |
| `test-apps/conv_pytorch/conv2d_pytorch.py` | Convolution kernels and cuDNN/cuBLAS-style launches |
| `test-apps/attention_pytorch/attention_pytorch.py` | PyTorch SDPA backends: `math`, `flash`, `mem_efficient` |

### 8.3 CuTe / CUTLASS DSL workloads

| File | What it is useful for |
| --- | --- |
| `test-apps/attention_cute/attention_cute.py` | Naive CuTe attention |
| `test-apps/axpy_relu_cute/axpy_relu_cute.py` | CuTe elementwise AXPY+ReLU |
| `test-apps/elementwise_add_cute/naive_elementwise_add.py` | Naive CuTe elementwise add |
| `test-apps/elementwise_add_cute/vectorized_elementwise_add.py` | Vectorized CuTe elementwise add |
| `test-apps/elementwise_add_cute/tiled_elementwise_add.py` | Tiled CuTe elementwise add |
| `test-apps/elementwise_add_cute/generic_elementwise_add.py` | Generic-structure CuTe elementwise add |
| `elementwise_add.ipynb` | Notebook form of the CuTe elementwise exploration |

## 9. Example End-To-End Workflows

Here are realistic things the repo can already do.

### 9.1 Summarize any kernel structurally

Example:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_KERNEL_FILTER=kernel_cutlass \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/kernel_summary.so \
python3 test-apps/elementwise_add_cute/vectorized_elementwise_add.py
```

This can tell you:

- launch dimensions
- register count
- shared memory use
- instruction count
- memory counts
- active SM spread

### 9.2 Compare CuTe kernel variants

The elementwise add CuTe runners can be compared under the same tool:

- naive
- vectorized
- tiled
- generic-structure

This is useful for understanding:

- instruction count changes
- register pressure changes
- memory behavior changes

### 9.3 Compare PyTorch attention backends

Example:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_GEMM_FILTER=sgemm \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/gemm_wavefit_trace.so \
python3 test-apps/attention_pytorch/attention_pytorch.py --backend math
```

and:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_GEMM_FILTER=sgemm \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/gemm_orchestration_map.so \
python3 test-apps/attention_pytorch/attention_pytorch.py --backend mem_efficient
```

This is useful for explaining:

- whether launches are underfilled
- whether fused attention kernels are present
- whether a path is dominated by extra copy / reduction / elementwise kernels

### 9.4 Build a new tool entirely in Python

Example:

```bash
python3 -m nvbpf_py.cli run \
  tools/nvbpf_py_examples/minimal_load_report.py \
  -- python3 test-apps/conv_pytorch/conv2d_pytorch.py
```

This creates, builds, preloads, and runs a tool from a single Python spec.

### 9.5 Trace CUDA API behavior around peer copies

Example:

```bash
python3 -m nvbpf_py.cli build --force --compile tools/nvbpf_py_examples/peer_copy_trace.py
```

This is useful when you want API-level behavior instead of device-injected instruction counting.

## 10. What Makes This Repo Strong

The strongest parts of the repo today are:

- it has a real reusable instrumentation layer, not just ad hoc examples
- it spans C++/CUDA, PyTorch, and CuTe workloads
- it has both handwritten and Python-authored tool paths
- it already contains several higher-level analysis tools, not just raw counters
- it is strong at explaining GPU behavior, not only collecting low-level events

In practice, the repo is already useful for:

- GPU kernel education
- quick instrumentation prototyping
- PyTorch backend comparison
- CuTe/CUTLASS kernel inspection
- building custom NVBit tools with less boilerplate

## 11. Current Limitations

A few limitations are visible in the current checkout.

### Python DSL limitations

The DSL is still intentionally restricted:

- no arbitrary Python in device hooks
- no arbitrary Python-authored generic host analyzers yet
- no full free-form GPU-side runtime
- still easiest for counters, maps, launch summaries, and selected built-in analyzers

### Tooling / environment limitations

- building still depends on NVCC / PTXAS / NVBit-compatible setup
- some workflows depend on environment variables and `LD_PRELOAD`
- some example outputs can get noisy without filtering or compact modes

### Scope limitations

The repo now covers wave fit, orchestration, fusion, tail inefficiency, sampled
reuse distance, pipeline depth, tile lifetime, CTA roles, register-pressure
tradeoffs, and bank-conflict suspicion. The remaining limitations are more
about precision than missing categories:

- several of the more advanced analyzers are still heuristic estimators
- bank-conflict and register-pressure tools explain likely causes rather than
  exposing direct hardware counters
- reuse-distance and tile-lifetime tools are sampled approximations, not full
  dataflow reconstruction

## 12. Best Starting Points For Different Users

### If you want to understand the framework

Start with:

- `README.md`
- `core/nvbpf.h`
- `tools/nvbpf_examples/README.md`

### If you want to use ready-made tools

Start with:

- `tools/nvbpf_examples/kernel_summary.cu`
- `tools/nvbpf_examples/sampling_mem_trace.cu`
- `tools/nvbpf_examples/gemm_wavefit_trace.cu`
- `tools/nvbpf_examples/gemm_orchestration_map.cu`

### If you want to write new tools quickly

Start with:

- `nvbpf_py/README.md`
- `tools/nvbpf_py_examples/minimal_load_report.py`
- `tools/nvbpf_py_examples/python_only_report.py`
- `tools/nvbpf_py_examples/loop_bucket_report.py`

### If you want to validate against real workloads

Start with:

- `test-apps/conv_pytorch/conv2d_pytorch.py`
- `test-apps/attention_pytorch/attention_pytorch.py`
- `test-apps/elementwise_add_cute/`

## 13. Bottom Line

This project has grown into a small GPU instrumentation platform.

It currently provides:

- a reusable NV-BPF runtime on top of NVBit
- a curated set of handwritten GPU analysis tools
- a Python DSL that can generate and run real tools
- example workloads across CUDA, PyTorch, and CuTe
- higher-level analysis aimed at explaining why kernels or backends behave differently

If someone asks "what can this repo do today?", the short answer is:

- instrument GPU kernels
- summarize and sample their behavior
- analyze GEMM/attention structure
- compare backend implementations
- and let users author new tools in Python instead of always writing CUDA by hand
