from nvbpf_py import (
    counter,
    counter_value,
    env_int,
    host_array,
    host_scalar,
    on_launch_exit,
    on_term,
    on_tool_init,
    short_kernel_name,
    state_add,
    state_get,
    tool,
)


@tool("aggregated_load_summary_py", banner="AGGREGATED_LOAD_SUMMARY_PY")
class AggregatedLoadSummaryPy:
    total_launches = host_scalar(type_name="u64")
    total_loads = host_scalar(type_name="u64")
    load_buckets = host_array(type_name="u64", length=4)

    loads = counter(loads=True)

    @on_tool_init()
    def init():
        print("bucket_base=", env_int("NVBPF_BUCKET_BASE", 1000))

    @on_launch_exit()
    def accumulate():
        base = env_int("NVBPF_BUCKET_BASE", 1000)
        launch_loads = counter_value("loads")

        total_launches += 1
        total_loads += launch_loads

        if launch_loads <= base:
            state_add("load_buckets", 0)
        elif launch_loads <= base * 10:
            state_add("load_buckets", 1)
        elif launch_loads <= base * 100:
            state_add("load_buckets", 2)
        else:
            state_add("load_buckets", 3)

        print(
            "kernel=", short_kernel_name(),
            "loads=", launch_loads,
            "launches=", total_launches,
        )

    @on_term()
    def final_report():
        print(
            "launches=", total_launches,
            "total_loads=", total_loads,
            "b0=", state_get("load_buckets", 0),
            "b1=", state_get("load_buckets", 1),
            "b2=", state_get("load_buckets", 2),
            "b3=", state_get("load_buckets", 3),
        )
