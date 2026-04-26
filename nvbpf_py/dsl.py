from __future__ import annotations

import inspect
import textwrap
from typing import Any

from .model import (
    ApiTraceSpec,
    CounterSpec,
    DeviceHookSpec,
    EpilogueFusionSpec,
    EventFieldSpec,
    EventSpec,
    GemmOrchestrationSpec,
    GemmWavefitSpec,
    HostStateSpec,
    LaunchEnterCallbackSpec,
    LaunchExitCallbackSpec,
    MapSpec,
    TailFragmentSpec,
    TermCallbackSpec,
    ToolInitCallbackSpec,
    ToolSpec,
)


class _CounterField:
    def __init__(
        self,
        *,
        opcodes: list[str] | tuple[str, ...] | None = None,
        loads: bool = False,
        stores: bool = False,
        branches: bool = False,
        exclude_constant_loads: bool = True,
        description: str = "",
    ) -> None:
        self.opcodes = tuple(opcodes or ())
        self.loads = loads
        self.stores = stores
        self.branches = branches
        self.exclude_constant_loads = exclude_constant_loads
        self.description = description

    def to_spec(self, name: str) -> CounterSpec:
        return CounterSpec(
            name=name,
            opcodes=self.opcodes,
            loads=self.loads,
            stores=self.stores,
            branches=self.branches,
            exclude_constant_loads=self.exclude_constant_loads,
            description=self.description,
        )


class _EventField:
    def __init__(
        self,
        *,
        fields: dict[str, str],
        capacity: int = 8192,
        description: str = "",
    ) -> None:
        if not fields:
            raise ValueError("event() requires at least one field")
        self.fields = tuple(EventFieldSpec(name=k, type_name=v) for k, v in fields.items())
        self.capacity = capacity
        self.description = description

    def to_spec(self, name: str) -> EventSpec:
        return EventSpec(
            name=name,
            fields=self.fields,
            capacity=self.capacity,
            description=self.description,
        )


class _ApiTraceField:
    def __init__(
        self,
        *,
        callbacks: list[str] | tuple[str, ...],
        on_exit: bool = True,
        correlate_launches: bool = False,
        correlate_window_events: int = 8,
        description: str = "",
    ) -> None:
        if not callbacks:
            raise ValueError("api_trace() requires at least one callback")
        if correlate_window_events <= 0:
            raise ValueError("api_trace() correlate_window_events must be positive")
        self.callbacks = tuple(callbacks)
        self.on_exit = on_exit
        self.correlate_launches = correlate_launches
        self.correlate_window_events = correlate_window_events
        self.description = description

    def to_spec(self, name: str) -> ApiTraceSpec:
        return ApiTraceSpec(
            name=name,
            callbacks=self.callbacks,
            on_exit=self.on_exit,
            correlate_launches=self.correlate_launches,
            correlate_window_events=self.correlate_window_events,
            description=self.description,
        )


class _MapField:
    def __init__(
        self,
        *,
        kind: str,
        type_name: str,
        length: int,
        description: str = "",
    ) -> None:
        if length <= 0:
            raise ValueError("map length must be positive")
        self.kind = kind
        self.type_name = type_name
        self.length = length
        self.description = description

    def to_spec(self, name: str) -> MapSpec:
        return MapSpec(
            name=name,
            kind=self.kind,
            type_name=self.type_name,
            length=self.length,
            description=self.description,
        )


class _HostStateField:
    def __init__(
        self,
        *,
        kind: str,
        type_name: str,
        length: int,
        initial: int = 0,
        description: str = "",
    ) -> None:
        if length <= 0:
            raise ValueError("host state length must be positive")
        self.kind = kind
        self.type_name = type_name
        self.length = length
        self.initial = initial
        self.description = description

    def to_spec(self, name: str) -> HostStateSpec:
        return HostStateSpec(
            name=name,
            kind=self.kind,
            type_name=self.type_name,
            length=self.length,
            initial=int(self.initial),
            description=self.description,
        )


class _GemmWavefitField:
    def __init__(
        self,
        *,
        sm_cta_entries_map: str,
        active_sm_bitmap_map: str,
        filter_csv: str = (
            "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
        ),
        filter_env: str = "NVBPF_GEMM_FILTER",
    ) -> None:
        self.sm_cta_entries_map = sm_cta_entries_map
        self.active_sm_bitmap_map = active_sm_bitmap_map
        self.filter_csv = filter_csv
        self.filter_env = filter_env

    def to_spec(self) -> GemmWavefitSpec:
        return GemmWavefitSpec(
            sm_cta_entries_map=self.sm_cta_entries_map,
            active_sm_bitmap_map=self.active_sm_bitmap_map,
            filter_csv=self.filter_csv,
            filter_env=self.filter_env,
        )


class _GemmOrchestrationField:
    def __init__(
        self,
        *,
        filter_csv: str = (
            "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
        ),
        filter_env: str = "NVBPF_GEMM_FILTER",
        window_env: str = "NVBPF_ORCH_WINDOW",
        default_window: int = 2,
    ) -> None:
        self.filter_csv = filter_csv
        self.filter_env = filter_env
        self.window_env = window_env
        self.default_window = default_window

    def to_spec(self) -> GemmOrchestrationSpec:
        return GemmOrchestrationSpec(
            filter_csv=self.filter_csv,
            filter_env=self.filter_env,
            window_env=self.window_env,
            default_window=self.default_window,
        )


class _EpilogueFusionField:
    def __init__(
        self,
        *,
        filter_csv: str = (
            "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
        ),
        filter_env: str = "NVBPF_GEMM_FILTER",
        window_env: str = "NVBPF_EPILOGUE_WINDOW",
        default_window: int = 3,
    ) -> None:
        self.filter_csv = filter_csv
        self.filter_env = filter_env
        self.window_env = window_env
        self.default_window = default_window

    def to_spec(self) -> EpilogueFusionSpec:
        return EpilogueFusionSpec(
            filter_csv=self.filter_csv,
            filter_env=self.filter_env,
            window_env=self.window_env,
            default_window=self.default_window,
        )


class _TailFragmentField:
    def __init__(
        self,
        *,
        filter_env: str = "NVBPF_KERNEL_FILTER",
        threshold_env: str = "NVBPF_TAIL_ACTIVE_LANES",
        default_threshold: int = 16,
    ) -> None:
        self.filter_env = filter_env
        self.threshold_env = threshold_env
        self.default_threshold = default_threshold

    def to_spec(self) -> TailFragmentSpec:
        return TailFragmentSpec(
            filter_env=self.filter_env,
            threshold_env=self.threshold_env,
            default_threshold=self.default_threshold,
        )


class _DeviceHookDecorator:
    def __init__(
        self,
        *,
        opcodes: list[str] | tuple[str, ...] | None = None,
        loads: bool = False,
        stores: bool = False,
        branches: bool = False,
        exclude_constant_loads: bool = True,
        description: str = "",
    ) -> None:
        self.opcodes = tuple(opcodes or ())
        self.loads = loads
        self.stores = stores
        self.branches = branches
        self.exclude_constant_loads = exclude_constant_loads
        self.description = description

    def __call__(self, fn: Any) -> Any:
        if not self.opcodes and not self.loads and not self.stores and not self.branches:
            raise ValueError("device_hook() requires at least one trigger predicate")
        source = textwrap.dedent(inspect.getsource(fn))
        sig = inspect.signature(fn)
        fn._nvbpf_device_hook = DeviceHookSpec(
            name=fn.__name__,
            args=tuple(sig.parameters.keys()),
            source=source,
            opcodes=self.opcodes,
            loads=self.loads,
            stores=self.stores,
            branches=self.branches,
            exclude_constant_loads=self.exclude_constant_loads,
            description=self.description,
        )
        return fn


class _LaunchExitDecorator:
    def __init__(self, *, description: str = "") -> None:
        self.description = description

    def __call__(self, fn: Any) -> Any:
        source = textwrap.dedent(inspect.getsource(fn))
        sig = inspect.signature(fn)
        if len(sig.parameters) != 0:
            raise ValueError("@on_launch_exit callbacks must not take arguments in this DSL version")
        fn._nvbpf_launch_exit = LaunchExitCallbackSpec(
            name=fn.__name__,
            source=source,
            description=self.description,
        )
        return fn


class _LaunchEnterDecorator:
    def __init__(self, *, description: str = "") -> None:
        self.description = description

    def __call__(self, fn: Any) -> Any:
        source = textwrap.dedent(inspect.getsource(fn))
        sig = inspect.signature(fn)
        if len(sig.parameters) != 0:
            raise ValueError("@on_launch_enter callbacks must not take arguments in this DSL version")
        fn._nvbpf_launch_enter = LaunchEnterCallbackSpec(
            name=fn.__name__,
            source=source,
            description=self.description,
        )
        return fn


class _ToolInitDecorator:
    def __init__(self, *, description: str = "") -> None:
        self.description = description

    def __call__(self, fn: Any) -> Any:
        source = textwrap.dedent(inspect.getsource(fn))
        sig = inspect.signature(fn)
        if len(sig.parameters) != 0:
            raise ValueError("@on_tool_init callbacks must not take arguments in this DSL version")
        fn._nvbpf_tool_init = ToolInitCallbackSpec(
            name=fn.__name__,
            source=source,
            description=self.description,
        )
        return fn


class _TermDecorator:
    def __init__(self, *, description: str = "") -> None:
        self.description = description

    def __call__(self, fn: Any) -> Any:
        source = textwrap.dedent(inspect.getsource(fn))
        sig = inspect.signature(fn)
        if len(sig.parameters) != 0:
            raise ValueError("@on_term callbacks must not take arguments in this DSL version")
        fn._nvbpf_term = TermCallbackSpec(
            name=fn.__name__,
            source=source,
            description=self.description,
        )
        return fn


def counter(
    *,
    opcodes: list[str] | tuple[str, ...] | None = None,
    loads: bool = False,
    stores: bool = False,
    branches: bool = False,
    exclude_constant_loads: bool = True,
    description: str = "",
) -> _CounterField:
    if not opcodes and not loads and not stores and not branches:
        raise ValueError("counter() requires at least one of opcodes/loads/stores/branches")
    return _CounterField(
        opcodes=opcodes,
        loads=loads,
        stores=stores,
        branches=branches,
        exclude_constant_loads=exclude_constant_loads,
        description=description,
    )


def event(
    *,
    fields: dict[str, str],
    capacity: int = 8192,
    description: str = "",
) -> _EventField:
    return _EventField(fields=fields, capacity=capacity, description=description)


def api_trace(
    *,
    callbacks: list[str] | tuple[str, ...],
    on_exit: bool = True,
    correlate_launches: bool = False,
    correlate_window_events: int = 8,
    description: str = "",
) -> _ApiTraceField:
    return _ApiTraceField(
        callbacks=callbacks,
        on_exit=on_exit,
        correlate_launches=correlate_launches,
        correlate_window_events=correlate_window_events,
        description=description,
    )


def device_hook(
    *,
    opcodes: list[str] | tuple[str, ...] | None = None,
    loads: bool = False,
    stores: bool = False,
    branches: bool = False,
    exclude_constant_loads: bool = True,
    description: str = "",
) -> _DeviceHookDecorator:
    return _DeviceHookDecorator(
        opcodes=opcodes,
        loads=loads,
        stores=stores,
        branches=branches,
        exclude_constant_loads=exclude_constant_loads,
        description=description,
    )


def hook(
    *,
    opcodes: list[str] | tuple[str, ...] | None = None,
    loads: bool = False,
    stores: bool = False,
    branches: bool = False,
    exclude_constant_loads: bool = True,
    description: str = "",
) -> _DeviceHookDecorator:
    return device_hook(
        opcodes=opcodes,
        loads=loads,
        stores=stores,
        branches=branches,
        exclude_constant_loads=exclude_constant_loads,
        description=description,
    )


def array(
    *,
    type_name: str,
    length: int,
    description: str = "",
) -> _MapField:
    return _MapField(
        kind="array",
        type_name=type_name,
        length=length,
        description=description,
    )


def percpu_array(
    *,
    type_name: str,
    length: int,
    description: str = "",
) -> _MapField:
    return _MapField(
        kind="percpu_array",
        type_name=type_name,
        length=length,
        description=description,
    )


def host_scalar(
    *,
    type_name: str,
    initial: int = 0,
    description: str = "",
) -> _HostStateField:
    return _HostStateField(
        kind="scalar",
        type_name=type_name,
        length=1,
        initial=initial,
        description=description,
    )


def host_array(
    *,
    type_name: str,
    length: int,
    initial: int = 0,
    description: str = "",
) -> _HostStateField:
    return _HostStateField(
        kind="array",
        type_name=type_name,
        length=length,
        initial=initial,
        description=description,
    )


def gemm_wavefit(
    *,
    sm_cta_entries_map: str,
    active_sm_bitmap_map: str,
    filter_csv: str = (
        "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
    ),
    filter_env: str = "NVBPF_GEMM_FILTER",
) -> _GemmWavefitField:
    return _GemmWavefitField(
        sm_cta_entries_map=sm_cta_entries_map,
        active_sm_bitmap_map=active_sm_bitmap_map,
        filter_csv=filter_csv,
        filter_env=filter_env,
    )


def gemm_orchestration_map(
    *,
    filter_csv: str = (
        "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
    ),
    filter_env: str = "NVBPF_GEMM_FILTER",
    window_env: str = "NVBPF_ORCH_WINDOW",
    default_window: int = 2,
) -> _GemmOrchestrationField:
    return _GemmOrchestrationField(
        filter_csv=filter_csv,
        filter_env=filter_env,
        window_env=window_env,
        default_window=default_window,
    )


def epilogue_fusion_trace(
    *,
    filter_csv: str = (
        "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
    ),
    filter_env: str = "NVBPF_GEMM_FILTER",
    window_env: str = "NVBPF_EPILOGUE_WINDOW",
    default_window: int = 3,
) -> _EpilogueFusionField:
    return _EpilogueFusionField(
        filter_csv=filter_csv,
        filter_env=filter_env,
        window_env=window_env,
        default_window=default_window,
    )


def tail_fragment_tracker(
    *,
    filter_env: str = "NVBPF_KERNEL_FILTER",
    threshold_env: str = "NVBPF_TAIL_ACTIVE_LANES",
    default_threshold: int = 16,
) -> _TailFragmentField:
    return _TailFragmentField(
        filter_env=filter_env,
        threshold_env=threshold_env,
        default_threshold=default_threshold,
    )


def on_launch_exit(*, description: str = "") -> _LaunchExitDecorator:
    return _LaunchExitDecorator(description=description)


def on_launch_enter(*, description: str = "") -> _LaunchEnterDecorator:
    return _LaunchEnterDecorator(description=description)


def on_tool_init(*, description: str = "") -> _ToolInitDecorator:
    return _ToolInitDecorator(description=description)


def on_term(*, description: str = "") -> _TermDecorator:
    return _TermDecorator(description=description)


def count(name: str) -> None:
    raise RuntimeError("count() is only valid inside an NV-BPF Python hook body")


def emit(event_name: str, **kwargs: Any) -> None:
    raise RuntimeError("emit() is only valid inside an NV-BPF Python hook body")


def atomic_add(name: str, *args: Any) -> None:
    raise RuntimeError("atomic_add() is only valid inside an NV-BPF Python hook body")


def map_get(name: str, index: int) -> Any:
    raise RuntimeError("map_get() is only valid inside an NV-BPF Python hook body")


def map_set(name: str, index: int, value: Any) -> None:
    raise RuntimeError("map_set() is only valid inside an NV-BPF Python hook body")


def counter_value(name: str) -> Any:
    raise RuntimeError("counter_value() is only valid inside an NV-BPF Python launch-exit callback")


def map_value(name: str, index: int) -> Any:
    raise RuntimeError("map_value() is only valid inside an NV-BPF Python launch-exit callback")


def kernel_name() -> str:
    raise RuntimeError("kernel_name() is only valid inside an NV-BPF Python launch-exit callback")


def short_kernel_name() -> str:
    raise RuntimeError("short_kernel_name() is only valid inside an NV-BPF Python launch-exit callback")


def grid_dim_x() -> int:
    raise RuntimeError("grid_dim_x() is only valid inside an NV-BPF Python launch-exit callback")


def grid_dim_y() -> int:
    raise RuntimeError("grid_dim_y() is only valid inside an NV-BPF Python launch-exit callback")


def grid_dim_z() -> int:
    raise RuntimeError("grid_dim_z() is only valid inside an NV-BPF Python launch-exit callback")


def block_dim_x() -> int:
    raise RuntimeError("block_dim_x() is only valid inside an NV-BPF Python launch-exit callback")


def block_dim_y() -> int:
    raise RuntimeError("block_dim_y() is only valid inside an NV-BPF Python launch-exit callback")


def block_dim_z() -> int:
    raise RuntimeError("block_dim_z() is only valid inside an NV-BPF Python launch-exit callback")


def regs() -> int:
    raise RuntimeError("regs() is only valid inside an NV-BPF Python launch-exit callback")


def smem_static() -> int:
    raise RuntimeError("smem_static() is only valid inside an NV-BPF Python launch-exit callback")


def smem_dynamic() -> int:
    raise RuntimeError("smem_dynamic() is only valid inside an NV-BPF Python launch-exit callback")


def state_get(name: str, index: int) -> Any:
    raise RuntimeError("state_get() is only valid inside an NV-BPF Python host callback")


def state_set(name: str, index: int, value: Any) -> None:
    raise RuntimeError("state_set() is only valid inside an NV-BPF Python host callback")


def state_add(name: str, index: int, value: Any = 1) -> None:
    raise RuntimeError("state_add() is only valid inside an NV-BPF Python host callback")


def env_int(name: str, default: int = 0) -> int:
    raise RuntimeError("env_int() is only valid inside an NV-BPF Python host callback")


def env_flag(name: str) -> bool:
    raise RuntimeError("env_flag() is only valid inside an NV-BPF Python host callback")


def api_trace_value(name: str) -> int:
    raise RuntimeError("api_trace_value() is only valid inside an NV-BPF Python host callback")


def api_trace_bytes_value(name: str) -> int:
    raise RuntimeError("api_trace_bytes_value() is only valid inside an NV-BPF Python host callback")


def api_trace_correlated(name: str) -> bool:
    raise RuntimeError("api_trace_correlated() is only valid inside an NV-BPF Python launch callback")


def api_trace_delta(name: str) -> int:
    raise RuntimeError("api_trace_delta() is only valid inside an NV-BPF Python launch callback")


def tool(
    name: str,
    *,
    kernel_filter_env: str = "NVBPF_KERNEL_FILTER",
    kernel_filter_mode: str = "substring",
    kernel_filter_default: str = "",
    banner: str | None = None,
):
    def decorator(cls: type[Any]) -> type[Any]:
        host_states: list[HostStateSpec] = []
        maps: list[MapSpec] = []
        counters: list[CounterSpec] = []
        events: list[EventSpec] = []
        api_traces: list[ApiTraceSpec] = []
        device_hooks: list[DeviceHookSpec] = []
        tool_init_callbacks: list[ToolInitCallbackSpec] = []
        launch_enter_callbacks: list[LaunchEnterCallbackSpec] = []
        launch_exit_callbacks: list[LaunchExitCallbackSpec] = []
        term_callbacks: list[TermCallbackSpec] = []
        gemm_wavefit_spec: GemmWavefitSpec | None = None
        gemm_orchestration_spec: GemmOrchestrationSpec | None = None
        epilogue_fusion_spec: EpilogueFusionSpec | None = None
        tail_fragment_spec: TailFragmentSpec | None = None

        for attr_name, value in cls.__dict__.items():
            if isinstance(value, _HostStateField):
                host_states.append(value.to_spec(attr_name))
            elif isinstance(value, _MapField):
                maps.append(value.to_spec(attr_name))
            elif isinstance(value, _CounterField):
                counters.append(value.to_spec(attr_name))
            elif isinstance(value, _EventField):
                events.append(value.to_spec(attr_name))
            elif isinstance(value, _ApiTraceField):
                api_traces.append(value.to_spec(attr_name))
            elif isinstance(value, _GemmWavefitField):
                if gemm_wavefit_spec is not None:
                    raise ValueError(f"tool {name!r} declares multiple gemm_wavefit() analyses")
                gemm_wavefit_spec = value.to_spec()
            elif isinstance(value, _GemmOrchestrationField):
                if gemm_orchestration_spec is not None:
                    raise ValueError(
                        f"tool {name!r} declares multiple gemm_orchestration_map() analyses"
                    )
                gemm_orchestration_spec = value.to_spec()
            elif isinstance(value, _EpilogueFusionField):
                if epilogue_fusion_spec is not None:
                    raise ValueError(
                        f"tool {name!r} declares multiple epilogue_fusion_trace() analyses"
                    )
                epilogue_fusion_spec = value.to_spec()
            elif isinstance(value, _TailFragmentField):
                if tail_fragment_spec is not None:
                    raise ValueError(
                        f"tool {name!r} declares multiple tail_fragment_tracker() analyses"
                    )
                tail_fragment_spec = value.to_spec()

            hook_spec = getattr(value, "_nvbpf_device_hook", None)
            if isinstance(hook_spec, DeviceHookSpec):
                device_hooks.append(hook_spec)
            tool_init_spec = getattr(value, "_nvbpf_tool_init", None)
            if isinstance(tool_init_spec, ToolInitCallbackSpec):
                tool_init_callbacks.append(tool_init_spec)
            launch_enter_spec = getattr(value, "_nvbpf_launch_enter", None)
            if isinstance(launch_enter_spec, LaunchEnterCallbackSpec):
                launch_enter_callbacks.append(launch_enter_spec)
            launch_exit_spec = getattr(value, "_nvbpf_launch_exit", None)
            if isinstance(launch_exit_spec, LaunchExitCallbackSpec):
                launch_exit_callbacks.append(launch_exit_spec)
            term_spec = getattr(value, "_nvbpf_term", None)
            if isinstance(term_spec, TermCallbackSpec):
                term_callbacks.append(term_spec)

        specialized_count = sum(
            spec is not None
            for spec in (
                gemm_wavefit_spec,
                gemm_orchestration_spec,
                epilogue_fusion_spec,
                tail_fragment_spec,
            )
        )
        if specialized_count > 1:
            raise ValueError(
                f"tool {name!r} cannot combine multiple high-level analyses in this DSL version"
            )

        maps_by_name = {spec.name: spec for spec in maps}
        if gemm_wavefit_spec is not None:
            if host_states or counters or events or api_traces or device_hooks or tool_init_callbacks or launch_enter_callbacks or launch_exit_callbacks or term_callbacks:
                raise ValueError(
                    f"tool {name!r} uses gemm_wavefit(); combine it only with explicit device maps in this DSL version"
                )
            sm_entries = maps_by_name.get(gemm_wavefit_spec.sm_cta_entries_map)
            if sm_entries is None:
                raise ValueError(
                    f"gemm_wavefit() references unknown map {gemm_wavefit_spec.sm_cta_entries_map!r}"
                )
            if sm_entries.kind != "percpu_array" or sm_entries.type_name != "u64":
                raise ValueError(
                    "gemm_wavefit() requires sm_cta_entries_map to be a percpu_array(type_name='u64', ...)"
                )
            bitmap = maps_by_name.get(gemm_wavefit_spec.active_sm_bitmap_map)
            if bitmap is None:
                raise ValueError(
                    f"gemm_wavefit() references unknown map {gemm_wavefit_spec.active_sm_bitmap_map!r}"
                )
            if bitmap.kind != "array" or bitmap.type_name != "u64":
                raise ValueError(
                    "gemm_wavefit() requires active_sm_bitmap_map to be an array(type_name='u64', ...)"
                )
        if gemm_orchestration_spec is not None and (host_states or maps or counters or events or api_traces or device_hooks):
            raise ValueError(
                f"tool {name!r} uses gemm_orchestration_map(); keep it host-only in this DSL version"
            )
        if epilogue_fusion_spec is not None and (host_states or maps or counters or events or api_traces or device_hooks):
            raise ValueError(
                f"tool {name!r} uses epilogue_fusion_trace(); keep it host-only in this DSL version"
            )
        if tail_fragment_spec is not None and (host_states or maps or counters or events or api_traces or device_hooks):
            raise ValueError(
                f"tool {name!r} uses tail_fragment_tracker(); keep it standalone in this DSL version"
            )
        if (gemm_wavefit_spec or gemm_orchestration_spec or epilogue_fusion_spec or tail_fragment_spec) and (tool_init_callbacks or launch_enter_callbacks or launch_exit_callbacks or term_callbacks):
            raise ValueError(
                f"tool {name!r} cannot combine launch callbacks with specialized analyses in this DSL version"
            )
        if kernel_filter_mode not in {"substring", "exact", "csv"}:
            raise ValueError(
                f"tool {name!r} uses unsupported kernel_filter_mode={kernel_filter_mode!r}; "
                "expected one of: substring, exact, csv"
            )

        if (
            not host_states
            and
            not maps
            and not counters
            and not events
            and not api_traces
            and not device_hooks
            and not tool_init_callbacks
            and not launch_enter_callbacks
            and not launch_exit_callbacks
            and not term_callbacks
            and gemm_wavefit_spec is None
            and gemm_orchestration_spec is None
            and epilogue_fusion_spec is None
            and tail_fragment_spec is None
        ):
            raise ValueError(
                f"tool {name!r} has no maps, counters, events, hooks, api traces, or analyses"
            )

        cls.tool_spec = ToolSpec(
            name=name,
            host_states=tuple(host_states),
            maps=tuple(maps),
            counters=tuple(counters),
            events=tuple(events),
            device_hooks=tuple(device_hooks),
            tool_init_callbacks=tuple(tool_init_callbacks),
            launch_enter_callbacks=tuple(launch_enter_callbacks),
            launch_exit_callbacks=tuple(launch_exit_callbacks),
            term_callbacks=tuple(term_callbacks),
            api_traces=tuple(api_traces),
            gemm_wavefit=gemm_wavefit_spec,
            gemm_orchestration=gemm_orchestration_spec,
            epilogue_fusion=epilogue_fusion_spec,
            tail_fragment=tail_fragment_spec,
            kernel_filter_env=kernel_filter_env,
            kernel_filter_mode=kernel_filter_mode,
            kernel_filter_default=kernel_filter_default,
            banner=banner or name.upper(),
        )
        return cls

    return decorator
