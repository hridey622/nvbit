from nvbpf_py import counter, counter_value, on_launch_exit, short_kernel_name, tool


@tool("minimal_load_report_py", banner="MINIMAL_LOAD_REPORT_PY")
class MinimalLoadReportPy:
    loads = counter(loads=True)

    @on_launch_exit()
    def report():
        print(
            "kernel=", short_kernel_name(),
            "loads=", counter_value("loads"),
        )
