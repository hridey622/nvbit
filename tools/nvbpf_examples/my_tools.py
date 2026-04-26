from nvbpf_py import (
    array,
    counter,
    counter_value,
    grid_dim_x,
    hook,
    map_value,
    on_launch_exit,
    short_kernel_name,
    tool,
)

@tool("my_tool", banner="MY_TOOL")
class MyTool:
    sampled_loads = counter(loads=True)
    active_lane_sum = array(type_name="u64", length=1)

    @hook(loads=True)
    def on_load(ctx):
        if ctx.active_lanes == 0:
            return
        ctx.atomic_add("active_lane_sum", 0, ctx.active_lanes)

    @on_launch_exit()
    def report():
        print(
            "kernel=", short_kernel_name(),
            "grid_x=", grid_dim_x(),
            "loads=", counter_value("sampled_loads"),
            "lanes=", map_value("active_lane_sum", 0),
        )
