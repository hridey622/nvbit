from nvbpf_py import (
    counter,
    counter_value,
    host_array,
    host_scalar,
    hook,
    on_launch_exit,
    on_term,
    short_kernel_name,
    state_add,
    state_get,
    tool,
)


@tool("agent_tool_template_py", banner="AGENT_TOOL_TEMPLATE_PY")
class AgentToolTemplatePy:
    sampled = counter(loads=True)

    total_launches = host_scalar(type_name="u64")
    total_sampled = host_scalar(type_name="u64")
    buckets = host_array(type_name="u64", length=4)

    @hook(loads=True)
    def on_load(ctx):
        if ctx.active_lanes == 0:
            return
        ctx.count("sampled")

    @on_launch_exit()
    def report():
        launch_sampled = counter_value("sampled")
        total_launches += 1
        total_sampled += launch_sampled

        if launch_sampled <= 1000:
            state_add("buckets", 0)
        elif launch_sampled <= 10000:
            state_add("buckets", 1)
        elif launch_sampled <= 100000:
            state_add("buckets", 2)
        else:
            state_add("buckets", 3)

        print(
            "kernel=", short_kernel_name(),
            "sampled=", launch_sampled,
            "total_launches=", total_launches,
        )

    @on_term()
    def final_report():
        print(
            "launches=", total_launches,
            "total_sampled=", total_sampled,
            "b0=", state_get("buckets", 0),
            "b1=", state_get("buckets", 1),
            "b2=", state_get("buckets", 2),
            "b3=", state_get("buckets", 3),
        )
