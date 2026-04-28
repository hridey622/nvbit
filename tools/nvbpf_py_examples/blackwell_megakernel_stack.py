from nvbpf_py import (
    api_trace,
    api_trace_bytes_value,
    api_trace_value,
    block_dim_x,
    counter,
    counter_value,
    grid_dim_x,
    host_scalar,
    on_launch_exit,
    on_term,
    regs,
    short_kernel_name,
    smem_dynamic,
    smem_static,
    tool,
)


@tool(
    "blackwell_megakernel_stack_py",
    banner="BLACKWELL_MEGAKERNEL_STACK_PY",
    kernel_filter_mode="csv",
    kernel_filter_default="cutlass,gemm,tcgen05,mma,attention",
)
class BlackwellMegakernelStackPy:
    launches = host_scalar(type_name="u64")

    total_tma_ops = host_scalar(type_name="u64")
    total_tmem_ops = host_scalar(type_name="u64")
    total_tcgen05_ops = host_scalar(type_name="u64")
    total_sync_ops = host_scalar(type_name="u64")
    total_epilogue_ops = host_scalar(type_name="u64")

    prev_copy_hits = host_scalar(type_name="u64")
    prev_sync_hits = host_scalar(type_name="u64")
    prev_alloc_hits = host_scalar(type_name="u64")
    prev_copy_bytes = host_scalar(type_name="u64")
    prev_alloc_bytes = host_scalar(type_name="u64")

    total_copy_between = host_scalar(type_name="u64")
    total_sync_between = host_scalar(type_name="u64")
    total_alloc_between = host_scalar(type_name="u64")
    total_copy_bytes_between = host_scalar(type_name="u64")
    total_alloc_bytes_between = host_scalar(type_name="u64")

    tma_ops = counter(
        opcodes=["CPASYNC", "LDGSTS", "UTMA", "TMA"],
        description="TMA-ish SASS proxy from direct NVBPF/NVBit instruction instrumentation.",
    )
    tmem_ops = counter(
        opcodes=["LDT", "STT", "FENCE"],
        description="TMEM load/store/fence proxy. FENCE is broad and should be read as pressure.",
    )
    tcgen05_ops = counter(
        opcodes=["UTC", "TCGEN05"],
        description="Blackwell tensor-core/tcgen05 proxy based on SASS opcode prefixes.",
    )
    sync_ops = counter(
        opcodes=["BAR", "DEPBAR", "MEMBAR", "WARPSYNC", "UTCBAR"],
        description="CTA, warp, memory, and tensor-core barrier/fence proxy.",
    )
    epilogue_loads = counter(
        loads=True,
        description="Load pressure that often appears in epilogues and scale-factor paths.",
    )
    epilogue_stores = counter(
        stores=True,
        description="Store pressure that often exposes epilogue bottlenecks.",
    )
    epilogue_atomics = counter(
        opcodes=["ATOM", "RED"],
        description="Atomic/reduction work in reductions or epilogues.",
    )
    branch_ops = counter(
        branches=True,
        description="Control-flow proxy for scheduler imbalance and epilogue conditionals.",
    )

    copy = api_trace(
        callbacks=[
            "API_CUDA_cuMemcpy",
            "API_CUDA_cuMemcpyAsync",
            "API_CUDA_cuMemcpyAsync_ptsz",
            "API_CUDA_cuMemcpyHtoD",
            "API_CUDA_cuMemcpyHtoD_v2",
            "API_CUDA_cuMemcpyHtoD_v2_ptds",
            "API_CUDA_cuMemcpyHtoDAsync",
            "API_CUDA_cuMemcpyHtoDAsync_v2",
            "API_CUDA_cuMemcpyHtoDAsync_v2_ptsz",
            "API_CUDA_cuMemcpyDtoH",
            "API_CUDA_cuMemcpyDtoH_v2",
            "API_CUDA_cuMemcpyDtoH_v2_ptds",
            "API_CUDA_cuMemcpyDtoHAsync",
            "API_CUDA_cuMemcpyDtoHAsync_v2",
            "API_CUDA_cuMemcpyDtoHAsync_v2_ptsz",
            "API_CUDA_cuMemcpyDtoD",
            "API_CUDA_cuMemcpyDtoD_v2",
            "API_CUDA_cuMemcpyDtoD_v2_ptds",
            "API_CUDA_cuMemcpyDtoDAsync",
            "API_CUDA_cuMemcpyDtoDAsync_v2",
            "API_CUDA_cuMemcpyDtoDAsync_v2_ptsz",
            "API_CUDA_cuMemcpyPeer",
            "API_CUDA_cuMemcpyPeer_ptds",
            "API_CUDA_cuMemcpyPeerAsync",
            "API_CUDA_cuMemcpyPeerAsync_ptsz",
        ],
        correlate_launches=True,
        correlate_window_events=128,
        description="Transfer activity around matched kernels.",
    )
    host_sync = api_trace(
        callbacks=[
            "API_CUDA_cuCtxSynchronize",
            "API_CUDA_cuCtxSynchronize_v2",
            "API_CUDA_cuEventSynchronize",
            "API_CUDA_cuStreamSynchronize",
            "API_CUDA_cuStreamSynchronize_ptsz",
        ],
        correlate_launches=True,
        correlate_window_events=128,
        description="Host-visible sync activity around matched kernels.",
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
        correlate_window_events=128,
        description="Allocation/free churn around matched kernels.",
    )

    @on_launch_exit()
    def accumulate_and_report():
        copy_hits = api_trace_value("copy")
        sync_hits = api_trace_value("host_sync")
        alloc_hits = api_trace_value("alloc")
        copy_bytes = api_trace_bytes_value("copy")
        alloc_bytes = api_trace_bytes_value("alloc")

        copy_between = copy_hits - prev_copy_hits
        sync_between = sync_hits - prev_sync_hits
        alloc_between = alloc_hits - prev_alloc_hits
        copy_bytes_between = copy_bytes - prev_copy_bytes
        alloc_bytes_between = alloc_bytes - prev_alloc_bytes

        prev_copy_hits = copy_hits
        prev_sync_hits = sync_hits
        prev_alloc_hits = alloc_hits
        prev_copy_bytes = copy_bytes
        prev_alloc_bytes = alloc_bytes

        launch_tma = counter_value("tma_ops")
        launch_tmem = counter_value("tmem_ops")
        launch_tcgen05 = counter_value("tcgen05_ops")
        launch_sync = counter_value("sync_ops")
        launch_epilogue = (
            counter_value("epilogue_loads")
            + counter_value("epilogue_stores")
            + counter_value("epilogue_atomics")
        )

        launches += 1
        total_tma_ops += launch_tma
        total_tmem_ops += launch_tmem
        total_tcgen05_ops += launch_tcgen05
        total_sync_ops += launch_sync
        total_epilogue_ops += launch_epilogue
        total_copy_between += copy_between
        total_sync_between += sync_between
        total_alloc_between += alloc_between
        total_copy_bytes_between += copy_bytes_between
        total_alloc_bytes_between += alloc_bytes_between

        print(
            "[B200]",
            "kernel=", short_kernel_name(),
            "grid_x=", grid_dim_x(),
            "block_x=", block_dim_x(),
            "regs=", regs(),
            "smem=", smem_static() + smem_dynamic(),
            "tma=", launch_tma,
            "tmem=", launch_tmem,
            "tcgen05=", launch_tcgen05,
            "sync=", launch_sync,
            "epi_load=", counter_value("epilogue_loads"),
            "epi_store=", counter_value("epilogue_stores"),
            "epi_atomic=", counter_value("epilogue_atomics"),
            "branch=", counter_value("branch_ops"),
            "copy_between=", copy_between,
            "copy_bytes=", copy_bytes_between,
            "host_sync_between=", sync_between,
            "alloc_between=", alloc_between,
            "alloc_bytes=", alloc_bytes_between,
        )

        if launch_tcgen05 == 0:
            print("[B200]", "hint=", "no tcgen05/UTC tensor ops matched this kernel")
        if launch_tma == 0:
            print("[B200]", "hint=", "no TMA-ish SASS matched by the configured opcode prefixes")
        if launch_tmem > launch_tcgen05 * 4 and launch_tcgen05 > 0:
            print("[B200]", "hint=", "TMEM proxy is high relative to tcgen05 proxy")
        if counter_value("epilogue_stores") > launch_tcgen05 and launch_tcgen05 > 0:
            print("[B200]", "hint=", "store-heavy epilogue may be limiting tensor work")
        if sync_between > 0:
            print("[B200]", "hint=", "host synchronization occurred between matched kernels")

    @on_term()
    def final_report():
        print(
            "[B200]",
            "summary_launches=", launches,
            "total_tma=", total_tma_ops,
            "total_tmem=", total_tmem_ops,
            "total_tcgen05=", total_tcgen05_ops,
            "total_sync=", total_sync_ops,
            "total_epilogue=", total_epilogue_ops,
            "total_copy_between=", total_copy_between,
            "total_copy_bytes=", total_copy_bytes_between,
            "total_host_sync_between=", total_sync_between,
            "total_alloc_between=", total_alloc_between,
            "total_alloc_bytes=", total_alloc_bytes_between,
        )
