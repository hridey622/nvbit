#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Probe:
    name: str
    spec: str
    purpose: str
    measures: tuple[str, ...]
    blind_spots: tuple[str, ...]
    kind: str = "dsl"


PROBES: dict[str, Probe] = {
    "proof": Probe(
        name="proof",
        spec="tools/nvbpf_examples/megakernel_phase_comm_proof.so",
        purpose="two-pass phase ordering and multi-GPU communication overlap proof",
        measures=(
            "device-side phase order from dynamic SASS phase intervals",
            "TMA/tensor and tensor/epilogue phase overlap proxies",
            "peer-copy and kernel stream-event intervals without profiler APIs",
            "copy/kernel overlap ratios and ordering verdicts",
        ),
        blind_spots=(
            "phase pass synchronizes to read maps, so run overlap mode separately",
            "actual copy/kernel overlap is bracketed with CUDA events inserted by this tool",
        ),
        kind="example",
    ),
    "proxies": Probe(
        name="proxies",
        spec="tools/nvbpf_py_examples/blackwell_megakernel_stack.py",
        purpose="single-pass B200 opcode and around-kernel context probe",
        measures=(
            "TMA-ish SASS families: CPASYNC, LDGSTS, UTMA, TMA",
            "TMEM-ish families: LDT, STT, FENCE",
            "tcgen05/tensor families: UTC, TCGEN05",
            "sync, epilogue load/store/atomic/reduction, and branch proxies",
            "host copy/sync/allocation activity between matched launches",
        ),
        blind_spots=(
            "does not read hardware performance counters",
            "does not know semantic phases unless kernel names or source markers expose them",
        ),
    ),
    "persistent": Probe(
        name="persistent",
        spec="tools/nvbpf_py_examples/blackwell_persistent_balance.py",
        purpose="persistent scheduling and active-SM spread probe",
        measures=(
            "SMs touched by each matched launch",
            "CTA-entry spread proxy across SMs",
            "wave-fit/fill-fraction style imbalance hints",
        ),
        blind_spots=(
            "does not read warp scheduler hardware counters",
            "multi-kernel frameworks may need a narrow NVBPF_KERNEL_FILTER",
        ),
    ),
    "tail": Probe(
        name="tail",
        spec="tools/nvbpf_py_examples/tail_fragment.py",
        purpose="low-active-lane and tail-fragment probe",
        measures=(
            "partial-warp events",
            "lane waste by memory/math/branch category",
            "epilogue/tail imbalance symptoms",
        ),
        blind_spots=(
            "higher overhead than host-only probes",
            "phase attribution needs source markers or narrow kernel filters",
        ),
    ),
}


def normalize_probes(raw: list[str] | None) -> list[str]:
    if not raw:
        return ["proxies"]
    out: list[str] = []
    for item in raw:
        for name in item.split(","):
            key = name.strip().lower()
            if not key:
                continue
            if key == "all":
                return list(PROBES)
            if key not in PROBES:
                raise SystemExit(f"unknown probe {key!r}; valid: {', '.join(PROBES)}")
            if key not in out:
                out.append(key)
    return out


def run_cmd(cmd: list[str], *, dry_run: bool) -> int:
    print(shlex.join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def command_plan(args: argparse.Namespace) -> int:
    probes = normalize_probes(args.probe)
    print("Blackwell B200 direct NVBPF stack")
    print()
    print("No NCU, NSYS, or CUPTI dependency. These probes use NVBPF/NVBit")
    print("launch callbacks, a handwritten proof tool, and Python-DSL SASS probes.")
    print()
    print("Run order:")
    print("  1. proof phase pass: check phase order and phase overlap")
    print("  2. proof event pass: check peer-copy/kernel ordering and overlap")
    print("  3. proxies: cheap first look at TMA/TMEM/tcgen05/sync/epilogue symptoms")
    print("  4. persistent/tail: deeper scheduling and low-active-lane probes")
    print()
    for name in probes:
        probe = PROBES[name]
        print(f"{name}: {probe.purpose}")
        for item in probe.measures:
            print(f"  measures: {item}")
        for item in probe.blind_spots:
            print(f"  boundary: {item}")
    return 0


def command_build(args: argparse.Namespace) -> int:
    rc = 0
    for name in normalize_probes(args.probe):
        probe = PROBES[name]
        if probe.kind == "example":
            if args.compile:
                cmd = ["make", "-C", "tools/nvbpf_examples", Path(probe.spec).name]
                rc = run_cmd(cmd, dry_run=args.dry_run) or rc
            else:
                print(f"{probe.name}: handwritten example; use --compile to build {probe.spec}")
        else:
            cmd = [
                sys.executable,
                "-m",
                "nvbpf_py.cli",
                "build",
                "--force",
            ]
            if args.compile:
                cmd.append("--compile")
            cmd.append(probe.spec)
            rc = run_cmd(cmd, dry_run=args.dry_run) or rc
    return rc


def command_run(args: argparse.Namespace) -> int:
    probes = normalize_probes(args.probe)
    if not args.cmd:
        raise SystemExit("run requires a command after --")
    rc = 0
    for name in probes:
        probe = PROBES[name]
        print(f"\n=== NVBPF B200 probe: {name} ===")
        if probe.kind == "example":
            so_path = REPO_ROOT / probe.spec
            cmd = ["env", f"LD_PRELOAD={so_path}", "ACK_CTX_INIT_LIMITATION=1"]
            cmd += args.cmd
        else:
            cmd = [
                sys.executable,
                "-m",
                "nvbpf_py.cli",
                "run",
            ]
            if args.no_build:
                cmd.append("--no-build")
            cmd += [probe.spec, "--"]
            cmd += args.cmd
        rc = run_cmd(cmd, dry_run=args.dry_run) or rc
    return rc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct NVBPF/NVBit B200 megakernel probes.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_probe_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--probe",
            action="append",
            default=None,
            help="Probe name or comma list: proof, proxies, persistent, tail, all. Default: proxies.",
        )

    plan = sub.add_parser("plan", help="Describe the direct NVBPF probe stack.")
    add_probe_arg(plan)
    plan.set_defaults(func=command_plan)

    build = sub.add_parser("build", help="Generate and optionally compile selected probes.")
    add_probe_arg(build)
    build.add_argument("--compile", action="store_true", help="Run make for each generated probe.")
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(func=command_build)

    run = sub.add_parser("run", help="Run selected probes over a target command.")
    add_probe_arg(run)
    run.add_argument("--no-build", action="store_true", help="Reuse existing generated .so files.")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("cmd", nargs=argparse.REMAINDER, help="Target command after --")
    run.set_defaults(func=command_run)

    args = parser.parse_args(argv)
    if getattr(args, "cmd", None) and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
