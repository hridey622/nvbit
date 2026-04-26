# Agent Tool Authoring

This file is the shortest contract an LLM agent should follow when creating a
new NV-BPF tool in Python.

## Goal

Write one Python spec file and let `nvbpf_py` generate:

- host `.cu`
- hook `._hooks.cu`
- `Makefile`
- optional compiled `.so`

## Good Tool Shapes

The DSL is strongest for these patterns:

- per-kernel counters
- load/store/branch/opcode summaries
- sampled hook-side map updates
- per-launch reports
- cross-launch host aggregation
- end-of-run summaries
- simple CUDA API traces
- GEMM/attention host analyzers via built-in helpers

## Start From The Right Template

- smallest summary tool:
  [minimal_load_report.py](/home/hridey/nvbpf/tools/nvbpf_py_examples/minimal_load_report.py:1)
- custom hook plus launch report:
  [python_only_report.py](/home/hridey/nvbpf/tools/nvbpf_py_examples/python_only_report.py:1)
- hook loops plus map access:
  [loop_bucket_report.py](/home/hridey/nvbpf/tools/nvbpf_py_examples/loop_bucket_report.py:1)
- cross-launch host aggregation:
  [aggregated_load_summary.py](/home/hridey/nvbpf/tools/nvbpf_py_examples/aggregated_load_summary.py:1)
- generic scaffold:
  [agent_tool_template.py](/home/hridey/nvbpf/tools/nvbpf_py_examples/agent_tool_template.py:1)

## Minimal Agent Recipe

1. Pick a tool name and banner.
2. Prefer scaffolding first:

```bash
python3 -m nvbpf_py.cli scaffold my_tool --style minimal
```

3. Declare only the counters and maps you need.
4. Add a `@hook(...)` only if counters alone are not enough.
5. Add `@on_launch_exit()` for per-kernel output.
6. Add `host_scalar(...)` / `host_array(...)` and `@on_term()` if the user
   wants aggregated output.
7. Build with:

```bash
python3 -m nvbpf_py.cli build --force --compile path/to/tool.py
```

8. Or build and run in one step:

```bash
python3 -m nvbpf_py.cli run path/to/tool.py -- python3 your_app.py
```

## Allowed Building Blocks

Top-level declarations:

- `counter(...)`
- `array(...)`
- `percpu_array(...)`
- `event(...)`
- `host_scalar(...)`
- `host_array(...)`
- `api_trace(...)`
- `gemm_wavefit(...)`
- `gemm_orchestration_map(...)`
- `epilogue_fusion_trace(...)`
- `tail_fragment_tracker(...)`

Tool-level options:

- `@tool(..., kernel_filter_mode="substring" | "exact" | "csv")`
- `@tool(..., kernel_filter_default="...")`

Callbacks:

- `@hook(...)`
- `@on_tool_init()`
- `@on_launch_enter()`
- `@on_launch_exit()`
- `@on_term()`

Multiple lifecycle callbacks per stage are allowed and run in declaration
order.

Scaffold styles:

- `minimal`
- `hook-report`
- `aggregated`
- `api-trace`
- `gemm-wavefit`
- `gemm-orchestration`
- `epilogue-fusion`
- `tail-fragment`

## Hook DSL Rules

Inside `@hook(...)`, keep logic simple.

Allowed statements:

- `if` / `else`
- `for i in range(...)`
- `break`
- `continue`
- `return`
- `pass`
- assignments like `tmp = expr`
- augmented assignments like `tmp += 1`
- `count(...)`
- `atomic_add(...)`
- `map_set(...)`
- `emit(...)`

Preferred `ctx.*` interface:

- fields:
  - `ctx.pred`
  - `ctx.addr`
  - `ctx.is_load`
  - `ctx.active_lanes`
  - `ctx.sm_id`
  - `ctx.warp_id`
  - `ctx.cta_id_x`
  - `ctx.cta_id_y`
  - `ctx.cta_id_z`
- helpers:
  - `ctx.count(...)`
  - `ctx.atomic_add(...)`
  - `ctx.map_get(...)`
  - `ctx.map_set(...)`
  - `ctx.emit(...)`
  - `ctx.ballot(...)`
  - `ctx.popc(...)`
  - `ctx.ffs(...)`

## Host Callback Rules

Inside host callbacks, use:

- `counter_value(...)`
- `map_value(...)`
- `api_trace_value(...)`
- `api_trace_correlated(...)`
- `api_trace_delta(...)`
- `kernel_name()`
- `short_kernel_name()`
- `grid_dim_x/y/z()`
- `block_dim_x/y/z()`
- `regs()`
- `smem_static()`
- `smem_dynamic()`
- `state_get(...)`
- `state_set(...)`
- `state_add(...)`
- `env_int(...)`
- `env_flag(...)`

Host callbacks currently support:

- `if` / `else`
- `for ... in range(...)`
- `return`
- `break`
- `continue`
- assignments
- augmented assignments
- `print(...)`

Use `kernel_filter_mode="exact"` when you want one precise CuTeDSL kernel
name, or `kernel_filter_mode="csv"` when you want a few related CUTLASS/CuTeDSL
kernel families without writing C++.

Use `api_trace(..., correlate_launches=True)` plus the `api_trace_*` helpers
when the user asks about system behavior around kernels such as copies, syncs,
or allocation churn between launches.

## When Not To Use The Generic DSL

Do not force the generic Python DSL when the tool needs:

- arbitrary host-side C++ data structures
- custom CUDA API interposition beyond `api_trace(...)`
- complicated multi-kernel correlation logic
- unrestricted device-side control flow
- custom device helper libraries

For those, prefer:

- a specialized built-in analyzer if one exists
- or a handwritten tool in `tools/nvbpf_examples/`

## Good Agent Output Pattern

For a user request like:

"Count global-memory loads and print one summary per kernel."

an agent should produce a Python file that looks like:

```python
from nvbpf_py import counter, counter_value, on_launch_exit, short_kernel_name, tool


@tool("global_load_report_py", banner="GLOBAL_LOAD_REPORT_PY")
class GlobalLoadReportPy:
    loads = counter(loads=True)

    @on_launch_exit()
    def report():
        print("kernel=", short_kernel_name(), "loads=", counter_value("loads"))
```

For a user request like:

"Bucket kernels by load count and print a final summary."

the agent should add:

- `host_scalar(...)`
- `host_array(...)`
- `@on_term()`

instead of dropping into handwritten C++.

## Practical Advice For Agents

- prefer readable one-line summaries over huge dumps
- default to `short_kernel_name()`
- prefer counters first, hooks second, specialized analyzers third
- only add maps when a counter is not enough
- only add host state when the user wants aggregation across launches
- keep the first tool version small and composable
