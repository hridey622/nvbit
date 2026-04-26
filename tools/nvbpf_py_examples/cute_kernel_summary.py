from nvbpf_py import (
    counter,
    counter_value,
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
    "cute_kernel_summary_py",
    banner="CUTE_KERNEL_SUMMARY_PY",
    kernel_filter_mode="csv",
    kernel_filter_default="cutlass::Kernel2,kernel_cutlass_kernel",
)
class CuTeKernelSummaryPy:
    launches = host_scalar(type_name="u64")
    total_loads = host_scalar(type_name="u64")
    total_stores = host_scalar(type_name="u64")
    total_branches = host_scalar(type_name="u64")

    loads = counter(loads=True)
    stores = counter(stores=True)
    branches = counter(branches=True)

    @on_launch_exit()
    def accumulate():
        launch_loads = counter_value("loads")
        launch_stores = counter_value("stores")
        launch_branches = counter_value("branches")

        launches += 1
        total_loads += launch_loads
        total_stores += launch_stores
        total_branches += launch_branches

    @on_launch_exit()
    def report():
        print(
            "kernel=", short_kernel_name(),
            "loads=", counter_value("loads"),
            "stores=", counter_value("stores"),
            "branches=", counter_value("branches"),
            "regs=", regs(),
            "smem_static=", smem_static(),
            "smem_dynamic=", smem_dynamic(),
        )

    @on_term()
    def final_report():
        print(
            "launches=", launches,
            "total_loads=", total_loads,
            "total_stores=", total_stores,
            "total_branches=", total_branches,
        )
