# NV-BPF L40S Showcase Results

Generated on 2026-04-30 on the remote server at `/home/dev9/venv/nvbpf`.

This README captures a small, reproducible NV-BPF showcase on the 4x NVIDIA L40S machine. The goal is to show that NV-BPF can explain CUDA kernel behavior across single-GPU kernels, AI/attention kernels, and multi-GPU peer-copy workflows that are relevant to semiconductor research workloads such as EDA acceleration, computational lithography, wafer image analysis, and GPU-accelerated simulation.

## System Snapshot

- GPUs: 4x NVIDIA L40S, 46068 MiB each
- Driver: 570.211.01
- CUDA compiler: nvcc 12.0.140
- Runtime Python: `/home/dev9/venv/bin/python`
- Python packages used: PyTorch 2.10.0+cu128, CuPy 14.0.1, CUTLASS/CuTe 4.5.0.dev0, Triton 3.6.0
- Repository branch: `main`
- Repository commit: `0ed916b Fix header formatting in README.md`
- Worktree status: dirty before this run, with generated `.so`/`.o` files, test apps, `nvbpf_py/`, and existing README edits already present

Topology note from `nvidia-smi topo -m`: GPUs 0-1 share a NUMA-side `NODE` relationship, GPUs 2-3 share a `NODE` relationship, and cross-pair traffic is `SYS`. The multi-GPU run below used `CUDA_VISIBLE_DEVICES=2,3`, so physical GPUs 2 and 3 appeared as visible GPU0 and GPU1 inside the process.

Raw logs are in:

```text
reports/l40s_showcase_20260430/
```

## Why These Kernels Matter

The selected workloads map well onto semiconductor research patterns:

- CuTe elementwise add: regular layout/data-movement kernel, similar to simple stencil, mask, image, and field-update passes.
- PyTorch attention math backend: GEMM plus softmax stages, similar to AI models used for hotspot detection, defect classification, yield prediction, and surrogate simulation.
- CuPy multi-GPU peer workflow: explicit inter-GPU movement plus compute, similar to domain decomposition, multi-GPU simulation, and distributed image/model pipelines.

Instrumentation timings below include NVBit/NV-BPF overhead. Treat the counts, launch geometry, SM spread, and API correlations as the primary results, not as baseline application performance.

## Result Summary

| Tool | Workload | Key Result | Raw Log |
| --- | --- | --- | --- |
| `kernel_summary.so` | CuTe vectorized elementwise add, 1024x1024 fp16 | 118,784 warp instructions, 8,192 loads, 4,096 stores, all 142 SMs active | `reports/l40s_showcase_20260430/kernel_summary_cute_vectorized.log` |
| `sampling_mem_trace.so` | Same CuTe kernel | Same load/store counts with sampled memory stream, 39 sampled events, 0 dropped | `reports/l40s_showcase_20260430/sampling_mem_cute_vectorized.log` |
| `gemm_wavefit_trace.so` | PyTorch SDPA math backend | Small attention GEMMs launch only 4 CTAs and use 4/142 SMs, fill fraction 0.014, severe underfill | `reports/l40s_showcase_20260430/gemm_wavefit_attention_math.log` |
| `attention_trace.so` | PyTorch SDPA math backend | Separates QK matmul, softmax, and PV matmul instruction/memory behavior | `reports/l40s_showcase_20260430/attention_trace_math.log` |
| `epilogue_fusion_trace.so` | PyTorch SDPA math backend and CuPy 2-GPU GEMM | Attention GEMMs show separate post-kernel work; multi-GPU split-K tail shows separate scale/copy/reduction signals | `reports/l40s_showcase_20260430/epilogue_fusion_attention_math.log`, `reports/l40s_showcase_20260430/epilogue_fusion_multigpu_peer.log` |
| `multi_gpu_kernel_trace.so` | CuPy 2-GPU GEMM plus peer copies | Captures compute on visible GPU0, peer copy to GPU1, compute on GPU1, copy back, and CUDA API activity | `reports/l40s_showcase_20260430/multi_gpu_kernel_trace_peer_fp32_copyback.log` |

## 1. CuTe Elementwise Kernel Summary

Command:

```bash
CUDA_VISIBLE_DEVICES=2 \
ACK_CTX_INIT_LIMITATION=1 \
NOBANNER=1 \
NVBPF_KERNEL_FILTER=kernel_cutlass \
LD_PRELOAD=$PWD/tools/nvbpf_examples/kernel_summary.so \
~/venv/bin/python test-apps/elementwise_add_cute/vectorized_elementwise_add.py \
  --m 1024 --n 1024 --warmup 1 --iters 3
```

Key output:

```text
method=vectorized m=1024 n=1024 dtype=fp16
avg_kernel_time_ms=0.684
max_abs_err=3.879547e-03
device: NVIDIA L40S
cc: (8, 9)

[NVBPF] kernel_cutlass_vectorized_elementwise_add... (elementwise)
        launch: grid=(512,1,1) block=(256,1,1) regs=24 smem=0+0
        instrs=118784 tensor=0 ffma=0 ldmatrix=0 cp_async=0 branches=4096
        mem: loads=8192 stores=4096 active_sms=142
```

Interpretation:

- The vectorized CuTe kernel spreads across all 142 SMs on the L40S.
- The kernel is simple and memory-oriented: no tensor instructions, no FFMA, and a clean 2:1 load/store relationship.
- This is the kind of summary that helps compare layout transforms, stencil passes, mask kernels, and simple EDA/image-processing kernels.

## 2. Sampled Memory Trace On The Same Kernel

Command:

```bash
CUDA_VISIBLE_DEVICES=2 \
ACK_CTX_INIT_LIMITATION=1 \
NOBANNER=1 \
NVBPF_KERNEL_FILTER=kernel_cutlass \
NVBPF_SAMPLE_EVERY=512 \
LD_PRELOAD=$PWD/tools/nvbpf_examples/sampling_mem_trace.so \
~/venv/bin/python test-apps/elementwise_add_cute/vectorized_elementwise_add.py \
  --m 1024 --n 1024 --warmup 1 --iters 3
```

Key output:

```text
[NVBPF] kernel_cutlass_vectorized_elementwise_add...
        loads=8192 stores=4096 sampled=39 dropped=0 sample_every=512
        addr_window=[0x0, 0xffffffffffffffff]
```

Interpretation:

- The sampled memory tool confirms the same load/store structure as `kernel_summary`.
- Sampling gives a lower-overhead path for studying memory-heavy kernels where full tracing would be too noisy.
- This is useful for semiconductor pipelines with large grids, images, sparse tensors, or layout tiles.

## 3. Attention GEMM Wave-Fit

Command:

```bash
CUDA_VISIBLE_DEVICES=2 \
ACK_CTX_INIT_LIMITATION=1 \
NOBANNER=1 \
NVBPF_GEMM_FILTER=sgemm \
LD_PRELOAD=$PWD/tools/nvbpf_examples/gemm_wavefit_trace.so \
~/venv/bin/python test-apps/attention_pytorch/attention_pytorch.py \
  --backend math --batch 1 --heads 4 --seq-len 128 --head-dim 64 \
  --dtype fp16 --warmup 1 --iters 3
```

Key output:

```text
batch=1 heads=4 seq_len=128 head_dim=64 dtype=fp16 causal=False backend=math
avg_kernel_time_ms=0.861
max_abs_err=2.409816e-04

[NVBPF GEMM_WAVEFIT_TRACE] matched_launches=10 unique_kernels=2
  x5   ampere_sgemm_128x128_tn          | ctas=4    fill=0.014 sms=4/142 regs=118 smem=16896+0 | underfill
  x5   ampere_sgemm_128x128_nn          | ctas=4    fill=0.014 sms=4/142 regs=118 smem=16640+0 | underfill
```

Interpretation:

- The small attention shape generates GEMM launches with only 4 CTAs.
- On a 142-SM L40S, that means only 4 SMs are active for these GEMMs.
- This is a clear example of wave underfill: the GPU is powerful, but the launch shape is too small to occupy it.
- In semiconductor AI workloads, this can explain why small batch/patch inference may fail to scale on high-SM-count GPUs.

## 4. Attention Stage Trace

Command:

```bash
CUDA_VISIBLE_DEVICES=2 \
ACK_CTX_INIT_LIMITATION=1 \
NOBANNER=1 \
NVBPF_ATTENTION_FILTER=sgemm,softmax \
LD_PRELOAD=$PWD/tools/nvbpf_examples/attention_trace.so \
~/venv/bin/python test-apps/attention_pytorch/attention_pytorch.py \
  --backend math --batch 1 --heads 4 --seq-len 128 --head-dim 64 \
  --dtype fp16 --warmup 0 --iters 1
```

Key output:

```text
[NVBPF] stage=qk_matmul kernel=ampere_sgemm_128x128_tn
        instrs=174256 ffma=131072 tensor=0 loads=12800 stores=4640 branches=704
[NVBPF] stage=softmax kernel=void (anonymous namespace)::softmax_warp_forward<...>
        instrs=87808 ffma=18432 tensor=0 loads=2048 stores=7168 branches=2048
[NVBPF] stage=pv_matmul kernel=ampere_sgemm_128x128_nn
        instrs=319968 ffma=262144 tensor=0 loads=22016 stores=5664 branches=992

[NVBPF ATTENTION_TRACE] Stage totals:
  qk_matmul: launches=2 instrs=348512
  softmax: launches=2 instrs=175616
  pv_matmul: launches=2 instrs=639936
```

Interpretation:

- NV-BPF can separate attention into recognizable stages instead of only reporting a flat kernel list.
- PV matmul is the largest stage in this run by instruction count.
- Softmax has fewer total instructions but a higher branch count relative to its size.
- This style is useful for ML-driven semiconductor workflows where model behavior needs to be tied back to GPU execution stages.

## 5. Multi-GPU Peer-Copy Trace

Command:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
ACK_CTX_INIT_LIMITATION=1 \
NOBANNER=1 \
NVBPF_KERNEL_FILTER=cupy_multiply,gemm \
NVBPF_TRACE_API=1 \
LD_PRELOAD=$PWD/tools/nvbpf_examples/multi_gpu_kernel_trace.so \
~/venv/bin/python test-apps/multi_gpu_peer/multi_gpu_peer.py \
  --m 256 --n 256 --k 256 --dtype fp32 --warmup 0 --iters 1
```

Key output:

```text
visible_device_count=2
visible_devices= [(0, 'NVIDIA L40S'), (1, 'NVIDIA L40S')]
src_device=0 dst_device=1 src_to_dst_peer=1 dst_to_src_peer=1
shape=(256,256)x(256,256) dtype=fp32 warmup=0 iters=1 copy_back=1

timed_iter=0 peer_bytes=262144 iter_ms=387.649 checksum=10.385671
peer_bytes_per_direction=262144

[NVBPF MGPU] #1 GPU0 cupy_multiply__float32_float_float32
    launch: grid=512x1x1 block=128x1x1 regs=18 dyn_smem=0 B
    before: sync=2, alloc=5 (960.0 KiB)
[NVBPF MGPU] #2 GPU0 ampere_sgemm_64x32_sliced1x4_nn
    launch: grid=4x8x4 block=256x1x1 regs=82 dyn_smem=0 B
    before: alloc=5 (8.38 MiB)
[NVBPF MGPU] #3 GPU1 cupy_multiply__float32_float_float32
    launch: grid=512x1x1 block=128x1x1 regs=18 dyn_smem=0 B
    before: peer_copy=1 (256.0 KiB), sync=2, alloc=2 (512.0 KiB)

[NVBPF MGPU] summary: launches=3 gpu0=2 gpu1=1
[NVBPF MGPU] activity: peer_copy=2 (512.0 KiB), other_copy=1 (4 B), sync=11, alloc=22 (9.81 MiB)
```

Interpretation:

- The tool correlated CUDA API activity with nearby kernel launches.
- It observed one compute phase on visible GPU0, a GEMM on visible GPU0, a peer copy to visible GPU1, compute on visible GPU1, and a peer copy back.
- Peer access was available in both directions.
- This is directly relevant to multi-GPU semiconductor simulations, layout partitioning, large image pipelines, and distributed AI inference/training workflows.


## 6. Epilogue Fusion Trace

The epilogue-fusion tool was run on the GEMM/attention-heavy workloads from this showcase. It does not inject device hooks; instead, it observes launch neighborhoods after focus kernels and flags whether post-GEMM work appears fused or split into separate epilogue, copy, reduction, or elementwise kernels.

Attention command:

```bash
CUDA_VISIBLE_DEVICES=2 \
ACK_CTX_INIT_LIMITATION=1 \
NOBANNER=1 \
NVBPF_GEMM_FILTER=sgemm,softmax \
NVBPF_EPILOGUE_WINDOW=4 \
LD_PRELOAD=$PWD/tools/nvbpf_examples/epilogue_fusion_trace.so \
~/venv/bin/python test-apps/attention_pytorch/attention_pytorch.py \
  --backend math --batch 1 --heads 4 --seq-len 128 --head-dim 64 \
  --dtype fp16 --warmup 1 --iters 3
```

Attention key output:

```text
[NVBPF EPILOGUE_FUSION_TRACE] focus_kernels=15 fused_likely=4 separate_signals=11
[NVBPF EPILOGUE_FUSION_TRACE] unique_focus_kernels=3
  x5   gemm       ampere_sgemm_128x128_tn          | post=0-1 epi=0-1 bias=0 act=0 scale=0 copy=0 red=0 elem=0-1 | fused_likely,separate_epilogue
  x5   reduction                                   | post=1-4 epi=0-2 bias=0 act=0 scale=0 copy=1 red=0-1 elem=0-2 | separate_epilogue,copyout_after,reduction_tail
  x5   gemm       ampere_sgemm_128x128_nn          | post=3-4 epi=0-2 bias=0 act=0 scale=0 copy=1-4 red=0-1 elem=0-2 | separate_epilogue,copyout_after,reduction_tail
```

Multi-GPU GEMM command:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
ACK_CTX_INIT_LIMITATION=1 \
NOBANNER=1 \
NVBPF_GEMM_FILTER=sgemm,gemm \
NVBPF_EPILOGUE_WINDOW=5 \
LD_PRELOAD=$PWD/tools/nvbpf_examples/epilogue_fusion_trace.so \
~/venv/bin/python test-apps/multi_gpu_peer/multi_gpu_peer.py \
  --m 256 --n 256 --k 256 --dtype fp32 --warmup 0 --iters 1
```

Multi-GPU key output:

```text
shape=(256,256)x(256,256) dtype=fp32 warmup=0 iters=1 copy_back=1
timed_iter=0 peer_bytes=262144 iter_ms=384.411 checksum=10.385671

[NVBPF EPILOGUE_FUSION_TRACE] focus_kernels=2 fused_likely=1 separate_signals=1
[NVBPF EPILOGUE_FUSION_TRACE] unique_focus_kernels=2
  x1   gemm       ampere_sgemm_64x32_sliced1x4_nn  | post=0 epi=0 bias=0 act=0 scale=0 copy=0 red=0 elem=0 | fused_likely
  x1   gemm       cublasLt::splitKreduce_k...oat, true, false, false> | post=5 epi=1 bias=0 act=0 scale=1 copy=3 red=1 elem=1 | separate_scale,copyout_after,reduction_tail
```

Interpretation:

- The attention math backend has mixed epilogue behavior: some QK GEMM neighborhoods look clean/fused, while PV and softmax-adjacent neighborhoods show separate epilogue, copyout, and reduction-tail signals.
- The multi-GPU workflow shows one GEMM as `fused_likely`, then identifies a cuBLASLt split-K reduction tail with separate scale, copyout, and reduction signals.
- For semiconductor workloads, this is useful when comparing GEMM-based solvers, ML inference paths, or layout/image kernels where bias, scale, activation, format conversion, or reductions may either be fused into the main kernel or emitted as extra launches.

## Takeaways For Semiconductor Research

1. NV-BPF can explain whether kernels are instruction-heavy, memory-heavy, branch-heavy, or underfilled.
2. On high-SM-count GPUs like L40S, small GEMMs and small attention shapes can severely underfill the device.
3. Stage-aware tools make AI workloads easier to connect to model operations such as QK matmul, softmax, and PV matmul.
4. Epilogue-fusion tracing can distinguish likely fused GEMM paths from separate post-processing, copyout, and reduction-tail work.
5. Multi-GPU tools can expose peer-copy volume, copy direction, launch placement, allocation activity, and synchronization around distributed workloads.
6. The same instrumentation approach works across CUDA/C++, CuPy, PyTorch, and CuTe/CUTLASS DSL workloads.

## Suggested Next Runs

For a stronger semiconductor-specific demo, run these same tools on:

- a lithography or stencil-style CUDA kernel
- a sparse matrix or graph kernel from placement/routing research
- a wafer defect CNN or ViT inference script
- a multi-GPU domain-decomposition simulation
- a CUTLASS/CuTe GEMM sweep with larger shapes to contrast the underfilled attention result
