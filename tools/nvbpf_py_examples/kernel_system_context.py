from nvbpf_py import (
    api_trace,
    api_trace_bytes_value,
    api_trace_correlated,
    api_trace_delta,
    api_trace_value,
    host_scalar,
    on_launch_enter,
    on_launch_exit,
    on_term,
    short_kernel_name,
    tool,
)


@tool(
    "kernel_system_context_py",
    banner="KERNEL_SYSTEM_CONTEXT_PY",
    kernel_filter_mode="csv",
    kernel_filter_default="cutlass::Kernel2,kernel_cutlass_kernel",
)
class KernelSystemContextPy:
    launches = host_scalar(type_name="u64")
    launches_with_copy = host_scalar(type_name="u64")
    launches_with_sync = host_scalar(type_name="u64")
    launches_with_alloc = host_scalar(type_name="u64")

    total_h2d_between = host_scalar(type_name="u64")
    total_d2h_between = host_scalar(type_name="u64")
    total_d2d_between = host_scalar(type_name="u64")
    total_sync_between = host_scalar(type_name="u64")
    total_alloc_between = host_scalar(type_name="u64")
    total_h2d_bytes_between = host_scalar(type_name="u64")
    total_d2h_bytes_between = host_scalar(type_name="u64")
    total_d2d_bytes_between = host_scalar(type_name="u64")
    total_alloc_bytes_between = host_scalar(type_name="u64")

    prev_h2d_hits = host_scalar(type_name="u64")
    prev_d2h_hits = host_scalar(type_name="u64")
    prev_d2d_hits = host_scalar(type_name="u64")
    prev_sync_hits = host_scalar(type_name="u64")
    prev_alloc_hits = host_scalar(type_name="u64")
    prev_h2d_bytes = host_scalar(type_name="u64")
    prev_d2h_bytes = host_scalar(type_name="u64")
    prev_d2d_bytes = host_scalar(type_name="u64")
    prev_alloc_bytes = host_scalar(type_name="u64")

    last_h2d_between = host_scalar(type_name="u64")
    last_d2h_between = host_scalar(type_name="u64")
    last_d2d_between = host_scalar(type_name="u64")
    last_sync_between = host_scalar(type_name="u64")
    last_alloc_between = host_scalar(type_name="u64")
    last_h2d_bytes_between = host_scalar(type_name="u64")
    last_d2h_bytes_between = host_scalar(type_name="u64")
    last_d2d_bytes_between = host_scalar(type_name="u64")
    last_alloc_bytes_between = host_scalar(type_name="u64")

    prelaunch_copy_near = host_scalar(type_name="u64")
    prelaunch_sync_near = host_scalar(type_name="u64")
    prelaunch_alloc_near = host_scalar(type_name="u64")
    prelaunch_copy_delta = host_scalar(type_name="i64", initial=-1)
    prelaunch_sync_delta = host_scalar(type_name="i64", initial=-1)
    prelaunch_alloc_delta = host_scalar(type_name="i64", initial=-1)

    h2d = api_trace(
        callbacks=[
            "API_CUDA_cuMemcpyHtoD",
            "API_CUDA_cuMemcpyHtoD_v2",
            "API_CUDA_cuMemcpyHtoD_v2_ptds",
            "API_CUDA_cuMemcpyHtoDAsync",
            "API_CUDA_cuMemcpyHtoDAsync_v2",
            "API_CUDA_cuMemcpyHtoDAsync_v2_ptsz",
        ],
        correlate_launches=True,
        description="Host-to-device transfer traffic near launches",
    )
    d2h = api_trace(
        callbacks=[
            "API_CUDA_cuMemcpyDtoH",
            "API_CUDA_cuMemcpyDtoH_v2",
            "API_CUDA_cuMemcpyDtoH_v2_ptds",
            "API_CUDA_cuMemcpyDtoHAsync",
            "API_CUDA_cuMemcpyDtoHAsync_v2",
            "API_CUDA_cuMemcpyDtoHAsync_v2_ptsz",
        ],
        correlate_launches=True,
        description="Device-to-host transfer traffic near launches",
    )
    d2d = api_trace(
        callbacks=[
            "API_CUDA_cuMemcpyDtoD",
            "API_CUDA_cuMemcpyDtoD_v2",
            "API_CUDA_cuMemcpyDtoD_v2_ptds",
            "API_CUDA_cuMemcpyDtoDAsync",
            "API_CUDA_cuMemcpyDtoDAsync_v2",
            "API_CUDA_cuMemcpyDtoDAsync_v2_ptsz",
        ],
        correlate_launches=True,
        description="Device-to-device transfer traffic near launches",
    )
    sync = api_trace(
        callbacks=[
            "API_CUDA_cuCtxSynchronize",
            "API_CUDA_cuCtxSynchronize_v2",
            "API_CUDA_cuEventSynchronize",
            "API_CUDA_cuStreamSynchronize",
            "API_CUDA_cuStreamSynchronize_ptsz",
        ],
        correlate_launches=True,
        description="Explicit synchronization activity near launches",
    )
    alloc = api_trace(
        callbacks=[
            "API_CUDA_cuMemAlloc",
            "API_CUDA_cuMemAlloc_v2",
            "API_CUDA_cuMemAllocAsync",
            "API_CUDA_cuMemAllocAsync_ptsz",
            "API_CUDA_cuMemAllocFromPoolAsync",
            "API_CUDA_cuMemAllocFromPoolAsync_ptsz",
            "API_CUDA_cuMemFree",
            "API_CUDA_cuMemFree_v2",
            "API_CUDA_cuMemFreeAsync",
            "API_CUDA_cuMemFreeAsync_ptsz",
        ],
        correlate_launches=True,
        description="Allocation and free activity near launches; byte totals reflect allocation calls",
    )

    @on_launch_enter()
    def snapshot_prelaunch_context():
        copy_near = int(
            api_trace_correlated("h2d")
            or api_trace_correlated("d2h")
            or api_trace_correlated("d2d")
        )
        copy_delta = -1
        if api_trace_correlated("h2d"):
            copy_delta = api_trace_delta("h2d")
        elif api_trace_correlated("d2h"):
            copy_delta = api_trace_delta("d2h")
        elif api_trace_correlated("d2d"):
            copy_delta = api_trace_delta("d2d")

        prelaunch_copy_near = copy_near
        prelaunch_copy_delta = copy_delta
        prelaunch_sync_near = int(api_trace_correlated("sync"))
        prelaunch_sync_delta = api_trace_delta("sync") if api_trace_correlated("sync") else -1
        prelaunch_alloc_near = int(api_trace_correlated("alloc"))
        prelaunch_alloc_delta = api_trace_delta("alloc") if api_trace_correlated("alloc") else -1

    @on_launch_exit()
    def accumulate():
        h2d_hits = api_trace_value("h2d")
        d2h_hits = api_trace_value("d2h")
        d2d_hits = api_trace_value("d2d")
        sync_hits = api_trace_value("sync")
        alloc_hits = api_trace_value("alloc")
        h2d_bytes = api_trace_bytes_value("h2d")
        d2h_bytes = api_trace_bytes_value("d2h")
        d2d_bytes = api_trace_bytes_value("d2d")
        alloc_bytes = api_trace_bytes_value("alloc")

        h2d_between = h2d_hits - prev_h2d_hits
        d2h_between = d2h_hits - prev_d2h_hits
        d2d_between = d2d_hits - prev_d2d_hits
        sync_between = sync_hits - prev_sync_hits
        alloc_between = alloc_hits - prev_alloc_hits
        h2d_bytes_between = h2d_bytes - prev_h2d_bytes
        d2h_bytes_between = d2h_bytes - prev_d2h_bytes
        d2d_bytes_between = d2d_bytes - prev_d2d_bytes
        alloc_bytes_between = alloc_bytes - prev_alloc_bytes

        last_h2d_between = h2d_between
        last_d2h_between = d2h_between
        last_d2d_between = d2d_between
        last_sync_between = sync_between
        last_alloc_between = alloc_between
        last_h2d_bytes_between = h2d_bytes_between
        last_d2h_bytes_between = d2h_bytes_between
        last_d2d_bytes_between = d2d_bytes_between
        last_alloc_bytes_between = alloc_bytes_between

        prev_h2d_hits = h2d_hits
        prev_d2h_hits = d2h_hits
        prev_d2d_hits = d2d_hits
        prev_sync_hits = sync_hits
        prev_alloc_hits = alloc_hits
        prev_h2d_bytes = h2d_bytes
        prev_d2h_bytes = d2h_bytes
        prev_d2d_bytes = d2d_bytes
        prev_alloc_bytes = alloc_bytes

        launches += 1
        total_h2d_between += h2d_between
        total_d2h_between += d2h_between
        total_d2d_between += d2d_between
        total_sync_between += sync_between
        total_alloc_between += alloc_between
        total_h2d_bytes_between += h2d_bytes_between
        total_d2h_bytes_between += d2h_bytes_between
        total_d2d_bytes_between += d2d_bytes_between
        total_alloc_bytes_between += alloc_bytes_between

        if h2d_between + d2h_between + d2d_between > 0:
            launches_with_copy += 1
        if sync_between > 0:
            launches_with_sync += 1
        if alloc_between > 0:
            launches_with_alloc += 1

    @on_launch_exit()
    def report():
        print(
            "kernel=", short_kernel_name(),
            "h2d_between=", last_h2d_between,
            "h2d_bytes_between=", last_h2d_bytes_between,
            "d2h_between=", last_d2h_between,
            "d2h_bytes_between=", last_d2h_bytes_between,
            "d2d_between=", last_d2d_between,
            "d2d_bytes_between=", last_d2d_bytes_between,
            "sync_between=", last_sync_between,
            "alloc_between=", last_alloc_between,
            "alloc_bytes_between=", last_alloc_bytes_between,
            "copy_before=", prelaunch_copy_near,
            "copy_before_delta=", prelaunch_copy_delta,
            "sync_before=", prelaunch_sync_near,
            "sync_before_delta=", prelaunch_sync_delta,
            "alloc_before=", prelaunch_alloc_near,
            "alloc_before_delta=", prelaunch_alloc_delta,
        )

    @on_term()
    def final_report():
        print(
            "launches=", launches,
            "launches_with_copy=", launches_with_copy,
            "launches_with_sync=", launches_with_sync,
            "launches_with_alloc=", launches_with_alloc,
            "total_h2d_between=", total_h2d_between,
            "total_h2d_bytes_between=", total_h2d_bytes_between,
            "total_d2h_between=", total_d2h_between,
            "total_d2h_bytes_between=", total_d2h_bytes_between,
            "total_d2d_between=", total_d2d_between,
            "total_d2d_bytes_between=", total_d2d_bytes_between,
            "total_sync_between=", total_sync_between,
            "total_alloc_between=", total_alloc_between,
            "total_alloc_bytes_between=", total_alloc_bytes_between,
        )
