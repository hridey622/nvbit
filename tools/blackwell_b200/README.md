# Blackwell B200 Direct NVBPF Stack

This directory is intentionally **not** an NCU, NSYS, or CUPTI wrapper.

The stack uses NVBPF/NVBit launch callbacks, a handwritten proof tool, and
Python-DSL SASS probes. It gives custom debugging signals that are useful while
designing B200 megakernels:

- TMA-ish instruction activity
- TMEM-ish load/store/fence pressure
- `tcgen05`/tensor instruction activity
- persistent scheduling balance proxies
- cluster/multicast symptoms through TMA/barrier/launch context
- scale-factor overhead symptoms through TMEM/load/branch mix
- epilogue bottleneck symptoms through load/store/atomic/reduction/branch mix

Because this does not use hardware-counter APIs, it reports direct
instrumentation proxies rather than hardware utilization percentages. That is a
deliberate design choice here.

## Probes

- `proof`
  - Tool: `tools/nvbpf_examples/megakernel_phase_comm_proof.so`
  - Two-pass proof for phase order and peer-copy/kernel stream overlap.
- `proxies`
  - Spec: `tools/nvbpf_py_examples/blackwell_megakernel_stack.py`
  - One-pass SASS family counters plus host copy/sync/allocation context.
- `persistent`
  - Spec: `tools/nvbpf_py_examples/blackwell_persistent_balance.py`
  - Active-SM spread and CTA-entry balance proxy.
- `tail`
  - Spec: `tools/nvbpf_py_examples/tail_fragment.py`
  - Low-active-lane and tail-fragment symptoms.

## Plan

```bash
python3 tools/blackwell_b200/stack.py plan --probe all
```

## Build

```bash
python3 tools/blackwell_b200/stack.py build --probe proof --compile
python3 tools/blackwell_b200/stack.py build --probe proxies --compile
python3 tools/blackwell_b200/stack.py build --probe persistent --compile
```

## Prove Phase Ordering

This pass intentionally synchronizes after each matched kernel so NVBPF can read
the device maps and print phase intervals.

```bash
NVBPF_KERNEL_FILTER=cutlass,gemm,tcgen05,mma \
NVBPF_PROOF_DEVICE_PHASES=1 \
NVBPF_PROOF_EVENTS=0 \
NVBPF_PHASE_ORDER=entry,tma,tensor,epilogue \
python3 tools/blackwell_b200/stack.py run --probe proof -- \
  python your_app.py
```

Look for:

- `expected_order=... verdict=PASS`
- `phase_overlap tma_tensor_ticks=... verdict=PASS`
- missing phase warnings when a required phase was not observed

`NVBPF_PHASE_ORDER` accepts these phase tokens:
`entry`, `tma`, `tmem`, `tensor`, `scale`, `multicast`, `sync`,
`producer`, `handoff`, `consumer`, `epilogue`, `branch`.

## Add Source Phase Markers

For semantic phases that cannot be recovered reliably from SASS opcode families,
include the marker header in your CUDA/CuTe/CUTLASS source:

```cpp
#include "nvbpf_phase_markers.cuh"
```

Then place markers around producer, handoff, consumer, scale-factor, multicast,
or epilogue regions:

```cpp
__global__ void gpu0_producer_kernel(...) {
    NVBPF_PHASE_BEGIN(NVBPF_PHASE_PRODUCER);
    // produce tile or activation data
    NVBPF_PHASE_END(NVBPF_PHASE_PRODUCER);

    NVBPF_PHASE_BEGIN(NVBPF_PHASE_TMA);
    // async/TMA movement
    NVBPF_PHASE_END(NVBPF_PHASE_TMA);

    NVBPF_PHASE_MARK(NVBPF_PHASE_HANDOFF);
}

__global__ void gpu1_consumer_kernel(...) {
    NVBPF_PHASE_MARK(NVBPF_PHASE_CONSUMER);
    // consume data from GPU0

    NVBPF_PHASE_BEGIN(NVBPF_PHASE_EPILOGUE);
    // writeback / reduction / output transform
    NVBPF_PHASE_END(NVBPF_PHASE_EPILOGUE);
}
```

The marker emits a volatile `mov.u32` with a magic immediate. The proof tool
recognizes that instruction and records it as a phase event. Use
`NVBPF_PROOF_SOURCE_MARKERS=0` to ignore source markers and fall back to opcode
families only.

For different producer/consumer kernels, run the phase pass with a tight
`NVBPF_KERNEL_FILTER` per kernel family, for example:

```bash
NVBPF_KERNEL_FILTER=gpu0_producer \
NVBPF_PHASE_ORDER=entry,producer,tma,handoff \
...

NVBPF_KERNEL_FILTER=gpu1_consumer \
NVBPF_PHASE_ORDER=entry,consumer,tensor,epilogue \
...
```

## Prove Communication Overlap

This pass disables device map reads and brackets launches plus async peer/device
copies with CUDA events inserted by the NVBPF tool. It does not use profiler
APIs.

```bash
NVBPF_KERNEL_FILTER=cutlass,gemm,tcgen05,mma \
NVBPF_PROOF_DEVICE_PHASES=0 \
NVBPF_PROOF_EVENTS=1 \
NVBPF_PROOF_MIN_COMM_OVERLAP=0.25 \
python3 tools/blackwell_b200/stack.py run --probe proof -- \
  python your_app.py
```

Look for:

- `event_timeline` rows for matched launches and peer copies
- `comm_kernel_pair ... order=overlap ... verdict=PASS`
- `overlap_summary candidates=... overlapped=... efficient=...`

Use tight filters first. Device-side instrumentation can be expensive on large
megakernels.

## Direct Signals

- TMA efficiency proxy:
  `CPASYNC`, `LDGSTS`, `UTMA`, and `TMA` dynamic counts, plus copy activity
  before launches.
- TMEM pressure proxy:
  `LDT`, `STT`, and `FENCE` dynamic counts.
- `tcgen05` utilization proxy:
  `UTC` and `TCGEN05` dynamic counts.
- Persistent scheduling balance:
  active-SM spread and CTA-entry distribution from `blackwell_persistent_balance.py`.
- Cluster multicast effectiveness:
  no hardware multicast counter is used; inspect TMA/barrier mix and add
  source-level phase markers when possible.
- Scale-factor overhead:
  infer from TMEM/load/branch/epilogue mix. For exact attribution, compile
  separate kernels or add source markers around scale-factor code.
- Epilogue bottlenecks:
  infer from stores, atomics/reductions, branches, and low-active-lane tail
  fragments.
