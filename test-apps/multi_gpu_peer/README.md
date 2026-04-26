# Multi-GPU Peer Workload

This directory contains a small Python workload for exercising:

- compute on one visible GPU
- explicit `cudaMemcpyPeerAsync` to another visible GPU
- compute on the destination GPU
- optional peer copy back

The main entry point is:

- [multi_gpu_peer.py](/home/hridey/nvbpf/test-apps/multi_gpu_peer/multi_gpu_peer.py)

It is intended to work well with the repo's multi-GPU tracing tools:

- `tools/nvbpf_examples/multi_gpu_kernel_trace.so`
- `tools/nvbpf_examples/nvlink_trace.so`
- `tools/nvbpf_examples/peer_copy_trace.so`
- `tools/nvbpf_generated/kernel_system_context_py/kernel_system_context_py.so`

## Why `CUDA_VISIBLE_DEVICES` Matters

The script uses visible device indices, not physical GPU IDs. That means:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
~/venv/bin/python test-apps/multi_gpu_peer/multi_gpu_peer.py
```

will use physical GPUs `2` and `3`, but inside the process they appear as
visible devices `0` and `1`.

That is the safest way to reserve the first two GPUs for other users while
testing multi-GPU behavior on the last two.

## Basic Run

```bash
CUDA_VISIBLE_DEVICES=2,3 \
~/venv/bin/python test-apps/multi_gpu_peer/multi_gpu_peer.py \
  --warmup 0 --iters 1
```

## With `nvlink_trace`

```bash
cd /home/hridey/nvbpf
CUDA_VISIBLE_DEVICES=2,3 \
ACK_CTX_INIT_LIMITATION=1 \
LD_PRELOAD=$PWD/tools/nvbpf_examples/nvlink_trace.so \
~/venv/bin/python test-apps/multi_gpu_peer/multi_gpu_peer.py \
  --warmup 0 --iters 1
```

## With `peer_copy_trace`

```bash
cd /home/hridey/nvbpf
CUDA_VISIBLE_DEVICES=2,3 \
ACK_CTX_INIT_LIMITATION=1 \
LD_PRELOAD=$PWD/tools/nvbpf_examples/peer_copy_trace.so \
~/venv/bin/python test-apps/multi_gpu_peer/multi_gpu_peer.py \
  --warmup 0 --iters 1
```

## With `multi_gpu_kernel_trace`

Use this when you want a compact per-kernel view of which visible GPU runs each
matched kernel, the launch shape, and the copy/sync/alloc activity since the
previous matched kernel launch.

```bash
cd /home/hridey/nvbpf
CUDA_VISIBLE_DEVICES=2,3 \
ACK_CTX_INIT_LIMITATION=1 \
NOBANNER=1 \
NVBPF_KERNEL_FILTER=cupy_multiply,gemm \
LD_PRELOAD=$PWD/tools/nvbpf_examples/multi_gpu_kernel_trace.so \
~/venv/bin/python test-apps/multi_gpu_peer/multi_gpu_peer.py \
  --warmup 0 --iters 1
```

Add `NVBPF_VERBOSE=1` for context/stream pointers and other low-level details.
Add `NVBPF_TRACE_API=1` for raw API lines such as `cuMemcpyPeerAsync`
endpoints around the kernel launches.

## Useful Flags

- `--src-device` and `--dst-device` choose the source/destination visible GPUs
- `--m`, `--n`, `--k` control matrix sizes
- `--dtype {fp16,fp32}` controls the tensor dtype
- `--no-copy-back` removes the return peer copy
- `--warmup` and `--iters` separate cold-start and timed iterations
