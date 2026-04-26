# NV-BPF Log Plotting

This directory contains small analysis helpers that turn NV-BPF text logs into
CSV summaries and SVG plots.

The current entry point is:

- [plot_tool_output.py](/home/hridey/nvbpf/tools/nvbpf_analysis/plot_tool_output.py:1)

It is dependency-free and does not require `matplotlib`.

## Supported Tool Logs

- `kernel_summary`
- `sampling_mem_trace`
- `gemm_wavefit_trace`
- `gemm_orchestration_map`
- `epilogue_fusion_trace`
- `tail_fragment_tracker`
- `bank_conflict_suspicion`
- `register_pressure_distortion_meter`

## Basic Usage

Capture a tool run to a log file:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_GEMM_FILTER=sgemm \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/gemm_wavefit_trace.so \
python3 test-apps/attention_pytorch/attention_pytorch.py --backend math \
  | tee gemm_wavefit_math.log
```

Then render plots:

```bash
python3 tools/nvbpf_analysis/plot_tool_output.py \
  --input gemm_wavefit_math.log \
  --output-dir tools/nvbpf_analysis/out
```

This writes:

- `*.csv`
- `*.svg`
- `*.summary.md` for tools that have a plain-English markdown summary

## Examples

Wave-fit plot:

```bash
python3 tools/nvbpf_analysis/plot_tool_output.py \
  --input gemm_wavefit_math.log
```

Epilogue fusion plot:

```bash
python3 tools/nvbpf_analysis/plot_tool_output.py \
  --input epilogue_mem_efficient.log
```

Tail-fragment plot:

```bash
python3 tools/nvbpf_analysis/plot_tool_output.py \
  --input tail_fragment.log
```

If auto-detection is ambiguous, force the parser:

```bash
python3 tools/nvbpf_analysis/plot_tool_output.py \
  --tool gemm_orchestration \
  --input orchestration.log
```

## Notes

- The newer analysis tools are easiest to plot in their compact default mode.
- `NVBPF_VERBOSE=1` is still useful for deep inspection, but the plotter is
  designed around the grouped summary lines.
- For `kernel_summary` and `sampling_mem_trace`, the plotter aggregates the
  repeated per-launch lines into per-kernel rows before rendering.
- For `bank_conflict_suspicion` and `register_pressure_distortion_meter`, the
  renderer now produces a report-style SVG with overview cards, raw signals,
  derived signals, findings, and suggested next steps.
- Those two tools also emit structured `derived:`, `note:`, and `next:` lines
  in their compact logs so the text output itself is more understandable even
  before plotting.
