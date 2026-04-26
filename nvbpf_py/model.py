from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CounterSpec:
    name: str
    opcodes: tuple[str, ...] = ()
    loads: bool = False
    stores: bool = False
    branches: bool = False
    exclude_constant_loads: bool = True
    description: str = ""


@dataclass(frozen=True)
class EventFieldSpec:
    name: str
    type_name: str


@dataclass(frozen=True)
class EventSpec:
    name: str
    fields: tuple[EventFieldSpec, ...] = ()
    capacity: int = 8192
    description: str = ""


@dataclass(frozen=True)
class ApiTraceSpec:
    name: str
    callbacks: tuple[str, ...] = ()
    on_exit: bool = True
    correlate_launches: bool = False
    correlate_window_events: int = 8
    description: str = ""


@dataclass(frozen=True)
class DeviceHookSpec:
    name: str
    args: tuple[str, ...]
    source: str
    opcodes: tuple[str, ...] = ()
    loads: bool = False
    stores: bool = False
    branches: bool = False
    exclude_constant_loads: bool = True
    description: str = ""


@dataclass(frozen=True)
class LaunchExitCallbackSpec:
    name: str
    source: str
    description: str = ""


@dataclass(frozen=True)
class LaunchEnterCallbackSpec:
    name: str
    source: str
    description: str = ""


@dataclass(frozen=True)
class ToolInitCallbackSpec:
    name: str
    source: str
    description: str = ""


@dataclass(frozen=True)
class TermCallbackSpec:
    name: str
    source: str
    description: str = ""


@dataclass(frozen=True)
class MapSpec:
    name: str
    kind: str
    type_name: str
    length: int
    description: str = ""


@dataclass(frozen=True)
class HostStateSpec:
    name: str
    kind: str
    type_name: str
    length: int
    initial: int = 0
    description: str = ""


@dataclass(frozen=True)
class GemmWavefitSpec:
    sm_cta_entries_map: str
    active_sm_bitmap_map: str
    filter_csv: str = (
        "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
    )
    filter_env: str = "NVBPF_GEMM_FILTER"


@dataclass(frozen=True)
class GemmOrchestrationSpec:
    filter_csv: str = (
        "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
    )
    filter_env: str = "NVBPF_GEMM_FILTER"
    window_env: str = "NVBPF_ORCH_WINDOW"
    default_window: int = 2


@dataclass(frozen=True)
class EpilogueFusionSpec:
    filter_csv: str = (
        "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
    )
    filter_env: str = "NVBPF_GEMM_FILTER"
    window_env: str = "NVBPF_EPILOGUE_WINDOW"
    default_window: int = 3


@dataclass(frozen=True)
class TailFragmentSpec:
    filter_env: str = "NVBPF_KERNEL_FILTER"
    threshold_env: str = "NVBPF_TAIL_ACTIVE_LANES"
    default_threshold: int = 16


@dataclass(frozen=True)
class ToolSpec:
    name: str
    host_states: tuple[HostStateSpec, ...] = field(default_factory=tuple)
    maps: tuple[MapSpec, ...] = field(default_factory=tuple)
    counters: tuple[CounterSpec, ...] = field(default_factory=tuple)
    events: tuple[EventSpec, ...] = field(default_factory=tuple)
    device_hooks: tuple[DeviceHookSpec, ...] = field(default_factory=tuple)
    tool_init_callbacks: tuple[ToolInitCallbackSpec, ...] = field(default_factory=tuple)
    launch_enter_callbacks: tuple[LaunchEnterCallbackSpec, ...] = field(default_factory=tuple)
    launch_exit_callbacks: tuple[LaunchExitCallbackSpec, ...] = field(default_factory=tuple)
    term_callbacks: tuple[TermCallbackSpec, ...] = field(default_factory=tuple)
    api_traces: tuple[ApiTraceSpec, ...] = field(default_factory=tuple)
    gemm_wavefit: GemmWavefitSpec | None = None
    gemm_orchestration: GemmOrchestrationSpec | None = None
    epilogue_fusion: EpilogueFusionSpec | None = None
    tail_fragment: TailFragmentSpec | None = None
    kernel_filter_env: str = "NVBPF_KERNEL_FILTER"
    kernel_filter_mode: str = "substring"
    kernel_filter_default: str = ""
    banner: str = ""
