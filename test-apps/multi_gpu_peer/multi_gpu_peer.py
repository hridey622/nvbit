#!/usr/bin/env python3
"""
Small multi-GPU peer-copy workload for NV-BPF / NVBit experiments.

This script is designed to stay within the set of GPUs made visible to the
process through CUDA_VISIBLE_DEVICES. That makes it easy to reserve a subset
of devices, for example:

  CUDA_VISIBLE_DEVICES=2,3 python3 test-apps/multi_gpu_peer/multi_gpu_peer.py

Within the process, those physical GPUs appear as visible devices 0 and 1.

The workload intentionally does:
1. compute on the source GPU
2. an explicit cudaMemcpyPeerAsync to the destination GPU
3. compute on the destination GPU
4. an optional cudaMemcpyPeerAsync back to the source GPU

That pattern is useful for:
- nvlink_trace.so
- peer_copy_trace.so
- kernel_system_context_py
"""

from __future__ import annotations

import argparse
import sys
import time


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small explicit peer-copy workload across two visible GPUs."
    )
    parser.add_argument("--src-device", type=int, default=0)
    parser.add_argument("--dst-device", type=int, default=1)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-copy-back", action="store_true")
    return parser.parse_args(argv)


def import_cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "CuPy is required for test-apps/multi_gpu_peer/multi_gpu_peer.py. "
            "Activate the environment that has CuPy installed before running it."
        ) from exc
    return cp


def dtype_from_name(cp, name: str):
    if name == "fp16":
        return cp.float16
    if name == "fp32":
        return cp.float32
    raise ValueError(f"unsupported dtype: {name}")


def device_name(cp, device_index: int) -> str:
    props = cp.cuda.runtime.getDeviceProperties(device_index)
    raw = props["name"]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def enable_peer_access_if_possible(cp, src_device: int, dst_device: int) -> tuple[bool, bool]:
    src_to_dst = bool(cp.cuda.runtime.deviceCanAccessPeer(src_device, dst_device))
    dst_to_src = bool(cp.cuda.runtime.deviceCanAccessPeer(dst_device, src_device))

    if src_to_dst:
        with cp.cuda.Device(src_device):
            try:
                cp.cuda.runtime.deviceEnablePeerAccess(dst_device)
            except cp.cuda.runtime.CUDARuntimeError:
                pass
    if dst_to_src:
        with cp.cuda.Device(dst_device):
            try:
                cp.cuda.runtime.deviceEnablePeerAccess(src_device)
            except cp.cuda.runtime.CUDARuntimeError:
                pass
    return src_to_dst, dst_to_src


def random_matrix(cp, shape: tuple[int, int], dtype, seed: int):
    cp.random.seed(seed)
    # cupy.random only generates float32/64 directly, so cast for fp16 runs.
    base = cp.random.randn(*shape, dtype=cp.float32)
    return base.astype(dtype, copy=False)


def synchronize_device(cp, device_index: int) -> None:
    with cp.cuda.Device(device_index):
        cp.cuda.runtime.deviceSynchronize()


def run_once(cp, args: argparse.Namespace, dtype, iteration: int, timed: bool):
    src_device = args.src_device
    dst_device = args.dst_device
    shape_a = (args.m, args.k)
    shape_b = (args.k, args.n)
    scalar_two = dtype(2.0)
    scalar_three = dtype(3.0)

    start = time.perf_counter()

    with cp.cuda.Device(src_device):
        a = random_matrix(cp, shape_a, dtype, args.seed + iteration * 17)
        b = random_matrix(cp, shape_b, dtype, args.seed + iteration * 17 + 1)
        src_out = (a * scalar_two) @ b

    synchronize_device(cp, src_device)

    with cp.cuda.Device(dst_device):
        peer_in = cp.empty_like(src_out)

    cp.cuda.runtime.memcpyPeerAsync(
        int(peer_in.data.ptr),
        dst_device,
        int(src_out.data.ptr),
        src_device,
        src_out.nbytes,
        0,
    )
    synchronize_device(cp, dst_device)

    with cp.cuda.Device(dst_device):
        dst_out = peer_in * scalar_three

    synchronize_device(cp, dst_device)

    returned = None
    if not args.no_copy_back:
        with cp.cuda.Device(src_device):
            returned = cp.empty_like(dst_out)
        cp.cuda.runtime.memcpyPeerAsync(
            int(returned.data.ptr),
            src_device,
            int(dst_out.data.ptr),
            dst_device,
            dst_out.nbytes,
            0,
        )
        synchronize_device(cp, src_device)
    else:
        returned = dst_out

    if timed:
        elapsed_ms = (time.perf_counter() - start) * 1e3
    else:
        elapsed_ms = None

    checksum = float(returned[:8, :8].astype(cp.float32).sum().get())
    return checksum, src_out.nbytes, elapsed_ms


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cp = import_cupy()
    dtype = dtype_from_name(cp, args.dtype)

    visible_count = cp.cuda.runtime.getDeviceCount()
    if visible_count < 2:
        raise SystemExit("This workload requires at least two visible CUDA devices.")
    if args.src_device == args.dst_device:
        raise SystemExit("--src-device and --dst-device must be different.")
    if args.src_device < 0 or args.src_device >= visible_count:
        raise SystemExit(f"--src-device must be in [0, {visible_count - 1}]")
    if args.dst_device < 0 or args.dst_device >= visible_count:
        raise SystemExit(f"--dst-device must be in [0, {visible_count - 1}]")

    src_to_dst, dst_to_src = enable_peer_access_if_possible(cp, args.src_device, args.dst_device)

    print(f"visible_device_count={visible_count}")
    print(
        "visible_devices=",
        [(i, device_name(cp, i)) for i in range(visible_count)],
    )
    print(
        f"src_device={args.src_device} dst_device={args.dst_device} "
        f"src_to_dst_peer={int(src_to_dst)} dst_to_src_peer={int(dst_to_src)}"
    )
    print(
        f"shape=({args.m},{args.k})x({args.k},{args.n}) dtype={args.dtype} "
        f"warmup={args.warmup} iters={args.iters} copy_back={int(not args.no_copy_back)}"
    )

    for i in range(args.warmup):
        checksum, peer_bytes, _ = run_once(cp, args, dtype, i, timed=False)
        print(f"warmup_iter={i} peer_bytes={peer_bytes} checksum={checksum:.6f}")

    total_ms = 0.0
    last_checksum = 0.0
    last_peer_bytes = 0
    for i in range(args.iters):
        checksum, peer_bytes, elapsed_ms = run_once(cp, args, dtype, args.warmup + i, timed=True)
        total_ms += elapsed_ms or 0.0
        last_checksum = checksum
        last_peer_bytes = peer_bytes
        print(
            f"timed_iter={i} peer_bytes={peer_bytes} "
            f"iter_ms={(elapsed_ms or 0.0):.3f} checksum={checksum:.6f}"
        )

    avg_ms = total_ms / max(args.iters, 1)
    print(f"avg_iter_ms={avg_ms:.3f}")
    print(f"peer_bytes_per_direction={last_peer_bytes}")
    print(f"last_checksum={last_checksum:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
