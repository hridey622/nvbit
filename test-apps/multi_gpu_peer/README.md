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
- `tools/nvbpf_examples/megakernel_sync_debugger.so`
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

## With `megakernel_sync_debugger`

Use this when you want both the multi-GPU launch/API context and in-kernel
signals that tend to matter while debugging synchronization-heavy megakernels:
barriers, memory barriers, atomics/reductions, async-copy pipeline ops, active
SM coverage, and low-active-lane warp events. Heavier memory, branch, and
tensor counters are opt-in.

```bash
cd /home/hridey/nvbpf
CUDA_VISIBLE_DEVICES=2,3 \
ACK_CTX_INIT_LIMITATION=1 \
NOBANNER=1 \
NVBPF_KERNEL_FILTER=cupy_multiply,gemm \
LD_PRELOAD=$PWD/tools/nvbpf_examples/megakernel_sync_debugger.so \
~/venv/bin/python test-apps/multi_gpu_peer/multi_gpu_peer.py \
  --warmup 0 --iters 1
```

Useful knobs:

- `NVBPF_MEGA_LOW_LANES=16` changes the low-active-lane threshold.
- `NVBPF_MEGA_MEMORY=1` enables load/store counters.
- `NVBPF_MEGA_BRANCH=1` enables branch counters.
- `NVBPF_MEGA_TENSOR=1` enables tensor-op counters.
- `NVBPF_VERBOSE=1` prints active-lane histograms and context/stream details.
- `NVBPF_TRACE_API=1` prints raw copy/sync/alloc API events.
- `NVBPF_FORCE_DEVICE_ALLOC=1` restores the older managed-allocation behavior
  if a system needs it, but the debugger leaves it off by default.

## With `megakernel_phase_comm_proof`

Use the proof tool in two passes. The phase pass synchronizes so it can read
device maps and check expected phase order:

```bash
cd /home/hridey/nvbpf
CUDA_VISIBLE_DEVICES=2,3 \
NVBPF_KERNEL_FILTER=cupy_multiply,gemm \
NVBPF_PROOF_DEVICE_PHASES=1 \
NVBPF_PROOF_EVENTS=0 \
NVBPF_PHASE_ORDER=entry,tma,tensor,epilogue \
LD_PRELOAD=$PWD/tools/nvbpf_examples/megakernel_phase_comm_proof.so \
~/venv/bin/python test-apps/multi_gpu_peer/multi_gpu_peer.py \
  --warmup 0 --iters 1
```

The overlap pass disables device map reads and brackets matched launches plus
async peer copies with CUDA events:

```bash
cd /home/hridey/nvbpf
CUDA_VISIBLE_DEVICES=2,3 \
NVBPF_KERNEL_FILTER=cupy_multiply,gemm \
NVBPF_PROOF_DEVICE_PHASES=0 \
NVBPF_PROOF_EVENTS=1 \
NVBPF_PROOF_MIN_COMM_OVERLAP=0.25 \
LD_PRELOAD=$PWD/tools/nvbpf_examples/megakernel_phase_comm_proof.so \
~/venv/bin/python test-apps/multi_gpu_peer/multi_gpu_peer.py \
  --warmup 0 --iters 1
```

Look for `expected_order=... verdict=PASS` in the first pass and
`comm_kernel_pair ... order=overlap ... verdict=PASS` in the second pass.
For semantic regions that opcode families cannot identify, include
`tools/nvbpf_examples/nvbpf_phase_markers.cuh` in your CUDA source and add
`NVBPF_PHASE_BEGIN(...)`, `NVBPF_PHASE_END(...)`, or `NVBPF_PHASE_MARK(...)`
around producer, handoff, consumer, scale-factor, multicast, or epilogue code.

## Useful Flags

- `--src-device` and `--dst-device` choose the source/destination visible GPUs
- `--m`, `--n`, `--k` control matrix sizes
- `--dtype {fp16,fp32}` controls the tensor dtype
- `--no-copy-back` removes the return peer copy
- `--warmup` and `--iters` separate cold-start and timed iterations
