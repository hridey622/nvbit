from __future__ import annotations

import ast
from dataclasses import dataclass

from .model import (
    ApiTraceSpec,
    DeviceHookSpec,
    EventSpec,
    HostStateSpec,
    LaunchEnterCallbackSpec,
    LaunchExitCallbackSpec,
    MapSpec,
    TermCallbackSpec,
    ToolInitCallbackSpec,
)


class TranspileError(RuntimeError):
    pass


@dataclass(frozen=True)
class HookRender:
    signature_params: tuple[str, ...]
    body: str
    needs_addr: bool
    needs_is_load: bool


@dataclass(frozen=True)
class LaunchExitRender:
    body: str


_BIN_OPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}

_CMP_OPS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

_HOOK_CTX_ATTRS = {
    "pred": "pred",
    "addr": "addr",
    "is_load": "is_load",
    "active_mask": "active_mask",
    "predicate_mask": "predicate_mask",
    "lane_id": "laneid",
    "active_lanes": "active_lanes",
    "sm_id": "sm_id",
    "warp_id": "warp_id",
    "cta_id_x": "cta_id_x",
    "cta_id_y": "cta_id_y",
    "cta_id_z": "cta_id_z",
    "grid_dim_x": "gridDim.x",
    "grid_dim_y": "gridDim.y",
    "grid_dim_z": "gridDim.z",
    "block_dim_x": "blockDim.x",
    "block_dim_y": "blockDim.y",
    "block_dim_z": "blockDim.z",
}

_MAP_C_TYPES = {
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "bool": "uint8_t",
}


class _HookTranspiler:
    def __init__(
        self,
        hook: DeviceHookSpec,
        events: dict[str, EventSpec],
        counters: set[str],
        maps: dict[str, MapSpec],
    ) -> None:
        self.hook = hook
        self.events = events
        self.counters = counters
        self.maps = maps
        self.indent = 1
        self.lines: list[str] = []
        self.locals: set[str] = set()
        self.arg_names = set(hook.args)
        self.loop_depth = 0

    def render(self) -> HookRender:
        tree = ast.parse(self.hook.source)
        fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        if fn is None:
            raise TranspileError(f"could not find function body for hook {self.hook.name}")

        signature_params = ["int pred"]
        if "addr" in self.arg_names:
            signature_params.append("uint64_t addr")
        if "is_load" in self.arg_names:
            signature_params.append("int is_load")
        for counter in sorted(self.counters):
            signature_params.append(f"uint64_t p_counter_{counter}")
        for map_name in sorted(self.maps):
            signature_params.append(f"uint64_t p_map_{map_name}")
        for event_name in sorted(self.events):
            signature_params.append(f"uint64_t p_event_{event_name}")

        self._emit("const int active_mask = __ballot_sync(__activemask(), 1);")
        self._emit("const int predicate_mask = __ballot_sync(__activemask(), pred);")
        self._emit("const int laneid = get_laneid();")
        self._emit("const int first_laneid = __ffs(active_mask) - 1;")
        self._emit("const int active_lanes = __popc(predicate_mask);")
        self._emit("const uint32_t sm_id = bpf_get_current_sm_id();")
        self._emit("const uint32_t warp_id = bpf_get_current_warp_id();")
        self._emit("const uint32_t cta_id_x = blockIdx.x;")
        self._emit("const uint32_t cta_id_y = blockIdx.y;")
        self._emit("const uint32_t cta_id_z = blockIdx.z;")
        self._emit("if (first_laneid != laneid) return;")

        for counter in sorted(self.counters):
            self._emit(f"uint64_t* counter_{counter} = (uint64_t*)p_counter_{counter};")
        for map_name, map_spec in sorted(self.maps.items()):
            c_type = _MAP_C_TYPES.get(map_spec.type_name)
            if c_type is None:
                raise TranspileError(
                    f"unsupported hook map type for {map_name!r}: {map_spec.type_name!r}"
                )
            self._emit(f"{c_type}* map_{map_name} = ({c_type}*)p_map_{map_name};")
        for event_name, event in sorted(self.events.items()):
            self._emit(
                f"auto* ring_{event_name} = "
                f"(BpfRingBufMap<{event_name}_event_t, {event.capacity}>*)p_event_{event_name};"
            )

        for stmt in fn.body:
            self._stmt(stmt)

        body = "\n".join(self.lines)
        return HookRender(
            signature_params=tuple(signature_params),
            body=body,
            needs_addr="addr" in self.arg_names,
            needs_is_load="is_load" in self.arg_names,
        )

    def _emit(self, line: str) -> None:
        self.lines.append("    " * self.indent + line)

    def _stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.If):
            self._emit(f"if ({self._expr(stmt.test)}) {{")
            self.indent += 1
            for inner in stmt.body:
                self._stmt(inner)
            self.indent -= 1
            if stmt.orelse:
                self._emit("} else {")
                self.indent += 1
                for inner in stmt.orelse:
                    self._stmt(inner)
                self.indent -= 1
            self._emit("}")
            return

        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                raise TranspileError("return with a value is not supported in hook bodies")
            self._emit("return;")
            return

        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise TranspileError("only simple assignments are supported in hook bodies")
            name = stmt.targets[0].id
            expr = self._expr(stmt.value)
            if name in self.locals:
                self._emit(f"{name} = {expr};")
            else:
                self.locals.add(name)
                self._emit(f"auto {name} = {expr};")
            return

        if isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                raise TranspileError("only simple augmented assignments are supported in hook bodies")
            op = _BIN_OPS.get(type(stmt.op))
            if op is None:
                raise TranspileError(f"unsupported augmented assignment operator: {type(stmt.op).__name__}")
            self._emit(f"{stmt.target.id} = ({stmt.target.id} {op} {self._expr(stmt.value)});")
            return

        if isinstance(stmt, ast.For):
            self._for_stmt(stmt)
            return

        if isinstance(stmt, ast.Break):
            if self.loop_depth <= 0:
                raise TranspileError("break is only supported inside hook for-loops")
            self._emit("break;")
            return

        if isinstance(stmt, ast.Continue):
            if self.loop_depth <= 0:
                raise TranspileError("continue is only supported inside hook for-loops")
            self._emit("continue;")
            return

        if isinstance(stmt, ast.Pass):
            return

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if self._is_call_named(call, "count"):
                self._emit(self._count_call(call))
                return
            if self._is_call_named(call, "atomic_add"):
                self._emit(self._atomic_add_call(call))
                return
            if self._is_call_named(call, "map_set"):
                self._emit(self._map_set_call(call))
                return
            if self._is_call_named(call, "emit"):
                for line in self._emit_call(call):
                    self._emit(line)
                return

        raise TranspileError(
            f"unsupported statement in hook {self.hook.name}: {ast.dump(stmt, include_attributes=False)}"
        )

    def _count_call(self, call: ast.Call) -> str:
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise TranspileError("count() expects one string literal argument")
        counter_name = call.args[0].value
        if counter_name not in self.counters:
            raise TranspileError(f"count() references unknown counter {counter_name!r}")
        return f"atomicAdd((unsigned long long*)counter_{counter_name}, 1ULL);"

    def _atomic_add_call(self, call: ast.Call) -> str:
        if not call.args or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise TranspileError("atomic_add() expects a string literal as its first argument")
        target_name = call.args[0].value
        if target_name in self.counters:
            if len(call.args) == 1:
                value_expr = "1"
            elif len(call.args) == 2:
                value_expr = self._expr(call.args[1])
            else:
                raise TranspileError("atomic_add(counter) accepts one or two arguments")
            return (
                f"atomicAdd((unsigned long long*)counter_{target_name}, "
                f"(unsigned long long)({value_expr}));"
            )
        if target_name in self.maps:
            if len(call.args) == 2:
                index_expr = self._expr(call.args[1])
                value_expr = "1"
            elif len(call.args) == 3:
                index_expr = self._expr(call.args[1])
                value_expr = self._expr(call.args[2])
            else:
                raise TranspileError("atomic_add(map) accepts two or three arguments")
            return (
                f"atomicAdd((unsigned long long*)&map_{target_name}[{index_expr}], "
                f"(unsigned long long)({value_expr}));"
            )
        raise TranspileError(f"atomic_add() references unknown counter or map {target_name!r}")

    def _map_set_call(self, call: ast.Call) -> str:
        if len(call.args) != 3 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise TranspileError("map_set() expects a map name string literal, an index expression, and a value expression")
        map_name = call.args[0].value
        map_spec = self.maps.get(map_name)
        if map_spec is None:
            raise TranspileError(f"map_set() references unknown map {map_name!r}")
        c_type = _MAP_C_TYPES.get(map_spec.type_name)
        if c_type is None:
            raise TranspileError(f"unsupported map type for map_set({map_name!r}): {map_spec.type_name!r}")
        index_expr = self._expr(call.args[1])
        value_expr = self._expr(call.args[2])
        return f"{self._map_element_expr(map_name, index_expr)} = ({c_type})({value_expr});"

    def _for_stmt(self, stmt: ast.For) -> None:
        if not isinstance(stmt.target, ast.Name):
            raise TranspileError("for-loops in hook bodies require a simple loop variable")
        if not isinstance(stmt.iter, ast.Call) or not isinstance(stmt.iter.func, ast.Name) or stmt.iter.func.id != "range":
            raise TranspileError("only for ... in range(...) loops are supported in hook bodies")
        if stmt.iter.keywords:
            raise TranspileError("range() keyword arguments are not supported in hook bodies")
        args = stmt.iter.args
        if len(args) == 1:
            start_expr = "0"
            stop_expr = self._expr(args[0])
            step_expr = "1"
        elif len(args) == 2:
            start_expr = self._expr(args[0])
            stop_expr = self._expr(args[1])
            step_expr = "1"
        elif len(args) == 3:
            start_expr = self._expr(args[0])
            stop_expr = self._expr(args[1])
            step_arg = args[2]
            if not isinstance(step_arg, ast.Constant) or not isinstance(step_arg.value, int) or step_arg.value <= 0:
                raise TranspileError("range(..., ..., step) currently requires a positive integer literal step")
            step_expr = str(step_arg.value)
        else:
            raise TranspileError("range() in hook bodies supports 1 to 3 positional arguments")
        if stmt.orelse:
            raise TranspileError("for-else is not supported in hook bodies")
        loop_var = stmt.target.id
        self._emit(f"for (int {loop_var} = {start_expr}; {loop_var} < {stop_expr}; {loop_var} += {step_expr}) {{")
        self.indent += 1
        self.loop_depth += 1
        for inner in stmt.body:
            self._stmt(inner)
        self.loop_depth -= 1
        self.indent -= 1
        self._emit("}")

    def _emit_call(self, call: ast.Call) -> list[str]:
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise TranspileError("emit() expects an event name string literal as its first argument")
        event_name = call.args[0].value
        event = self.events.get(event_name)
        if event is None:
            raise TranspileError(f"emit() references unknown event {event_name!r}")

        kw = {item.arg: item.value for item in call.keywords}
        extra = sorted(name for name in kw if name not in {field.name for field in event.fields})
        if extra:
            raise TranspileError(f"emit({event_name!r}) got unexpected fields: {', '.join(extra)}")

        lines = [f"{event_name}_event_t evt{{}};"]
        for field in event.fields:
            if field.name in kw:
                value_expr = self._expr(kw[field.name])
            else:
                value_expr = "0"
            lines.append(f"evt.{field.name} = {value_expr};")
        lines.append(f"ring_{event_name}->output(&evt);")
        return lines

    def _is_call_named(self, call: ast.Call, name: str) -> bool:
        if isinstance(call.func, ast.Name):
            return call.func.id == name
        return (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "ctx"
            and call.func.attr == name
        )

    def _expr(self, expr: ast.expr) -> str:
        if isinstance(expr, ast.Name):
            return expr.id
        if (
            isinstance(expr, ast.Attribute)
            and isinstance(expr.value, ast.Name)
            and expr.value.id == "ctx"
        ):
            mapped = _HOOK_CTX_ATTRS.get(expr.attr)
            if mapped is None:
                raise TranspileError(f"unsupported ctx attribute in hook {self.hook.name}: ctx.{expr.attr}")
            if mapped == "addr" and "addr" not in self.arg_names:
                raise TranspileError(f"hook {self.hook.name} uses ctx.addr but is not a memory hook")
            if mapped == "is_load" and "is_load" not in self.arg_names:
                raise TranspileError(f"hook {self.hook.name} uses ctx.is_load but is not a memory hook")
            return mapped
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return "true" if expr.value else "false"
            if isinstance(expr.value, str):
                return f'"{expr.value}"'
            return repr(expr.value)
        if isinstance(expr, ast.BinOp):
            op = _BIN_OPS.get(type(expr.op))
            if op is None:
                raise TranspileError(f"unsupported binary operator: {type(expr.op).__name__}")
            return f"({self._expr(expr.left)} {op} {self._expr(expr.right)})"
        if isinstance(expr, ast.BoolOp):
            op = "&&" if isinstance(expr.op, ast.And) else "||"
            return "(" + f" {op} ".join(self._expr(v) for v in expr.values) + ")"
        if isinstance(expr, ast.UnaryOp):
            if isinstance(expr.op, ast.Not):
                return f"!({self._expr(expr.operand)})"
            if isinstance(expr.op, ast.USub):
                return f"-({self._expr(expr.operand)})"
            raise TranspileError(f"unsupported unary operator: {type(expr.op).__name__}")
        if isinstance(expr, ast.Compare):
            if len(expr.ops) != 1 or len(expr.comparators) != 1:
                raise TranspileError("chained comparisons are not supported")
            op = _CMP_OPS.get(type(expr.ops[0]))
            if op is None:
                raise TranspileError(f"unsupported comparison operator: {type(expr.ops[0]).__name__}")
            return f"({self._expr(expr.left)} {op} {self._expr(expr.comparators[0])})"
        if isinstance(expr, ast.IfExp):
            return f"({self._expr(expr.test)} ? {self._expr(expr.body)} : {self._expr(expr.orelse)})"
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name) and expr.func.id == "int":
                if len(expr.args) != 1:
                    raise TranspileError("int() expects one argument")
                return f"((int){self._expr(expr.args[0])})"
            if self._is_call_named(expr, "map_get"):
                if len(expr.args) != 2 or not isinstance(expr.args[0], ast.Constant) or not isinstance(expr.args[0].value, str):
                    raise TranspileError("map_get() expects a map name string literal and an index expression")
                map_name = expr.args[0].value
                if map_name not in self.maps:
                    raise TranspileError(f"map_get() references unknown map {map_name!r}")
                return self._map_element_expr(map_name, self._expr(expr.args[1]))
            if (
                isinstance(expr.func, ast.Attribute)
                and isinstance(expr.func.value, ast.Name)
                and expr.func.value.id == "ctx"
            ):
                if expr.func.attr == "ballot":
                    if len(expr.args) != 1:
                        raise TranspileError("ctx.ballot() expects one argument")
                    return f"__ballot_sync(__activemask(), {self._expr(expr.args[0])})"
                if expr.func.attr == "popc":
                    if len(expr.args) != 1:
                        raise TranspileError("ctx.popc() expects one argument")
                    return f"__popc({self._expr(expr.args[0])})"
                if expr.func.attr == "ffs":
                    if len(expr.args) != 1:
                        raise TranspileError("ctx.ffs() expects one argument")
                    return f"__ffs({self._expr(expr.args[0])})"

        raise TranspileError(
            f"unsupported expression in hook {self.hook.name}: {ast.dump(expr, include_attributes=False)}"
        )

    def _map_element_expr(self, map_name: str, index_expr: str) -> str:
        map_spec = self.maps[map_name]
        if map_spec.kind == "percpu_array":
            return f"map_{map_name}[((uint32_t)sm_id * {map_spec.length}) + (uint32_t)({index_expr})]"
        return f"map_{map_name}[{index_expr}]"


def render_custom_hook(
    hook: DeviceHookSpec,
    events: dict[str, EventSpec],
    counters: set[str],
    maps: dict[str, MapSpec],
) -> HookRender:
    return _HookTranspiler(hook, events, counters, maps).render()


class _HostCallbackTranspiler:
    def __init__(
        self,
        callback: LaunchEnterCallbackSpec | LaunchExitCallbackSpec | ToolInitCallbackSpec | TermCallbackSpec,
        maps: dict[str, MapSpec],
        counters: set[str],
        host_states: dict[str, HostStateSpec],
        api_traces: dict[str, ApiTraceSpec],
        *,
        metadata_enabled: bool,
    ) -> None:
        self.callback = callback
        self.maps = maps
        self.counters = counters
        self.host_states = host_states
        self.api_traces = api_traces
        self.metadata_enabled = metadata_enabled
        self.indent = 2
        self.lines: list[str] = []
        self.locals: set[str] = set()
        self.print_index = 0
        self.loop_depth = 0

    def render(self) -> LaunchExitRender:
        tree = ast.parse(self.callback.source)
        fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        if fn is None:
            raise TranspileError(f"could not find function body for callback {self.callback.name}")
        for stmt in fn.body:
            self._stmt(stmt)
        return LaunchExitRender(body="\n".join(self.lines))

    def _emit(self, line: str) -> None:
        self.lines.append("    " * self.indent + line)

    def _stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.If):
            self._emit(f"if ({self._expr(stmt.test)}) {{")
            self.indent += 1
            for inner in stmt.body:
                self._stmt(inner)
            self.indent -= 1
            if stmt.orelse:
                self._emit("} else {")
                self.indent += 1
                for inner in stmt.orelse:
                    self._stmt(inner)
                self.indent -= 1
            self._emit("}")
            return
        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                raise TranspileError("return with a value is not supported in host callbacks")
            self._emit("return;")
            return
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise TranspileError("only simple assignments are supported in host callbacks")
            name = stmt.targets[0].id
            expr = self._expr(stmt.value)
            if name in self.host_states:
                host_state = self.host_states[name]
                if host_state.kind != "scalar":
                    raise TranspileError(f"host array {name!r} must be accessed via state_get/state_set/state_add")
                self._emit(f"_nvbpf_state_{name} = {expr};")
            elif name in self.locals:
                self._emit(f"{name} = {expr};")
            else:
                self.locals.add(name)
                self._emit(f"auto {name} = {expr};")
            return
        if isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                raise TranspileError("only simple augmented assignments are supported in host callbacks")
            op = _BIN_OPS.get(type(stmt.op))
            if op is None:
                raise TranspileError(f"unsupported augmented assignment operator: {type(stmt.op).__name__}")
            target_name = stmt.target.id
            expr = self._expr(stmt.value)
            if target_name in self.host_states:
                host_state = self.host_states[target_name]
                if host_state.kind != "scalar":
                    raise TranspileError(f"host array {target_name!r} must be accessed via state_get/state_set/state_add")
                self._emit(f"_nvbpf_state_{target_name} = (_nvbpf_state_{target_name} {op} {expr});")
            else:
                self._emit(f"{target_name} = ({target_name} {op} {expr});")
            return
        if isinstance(stmt, ast.For):
            self._for_stmt(stmt)
            return
        if isinstance(stmt, ast.Break):
            if self.loop_depth <= 0:
                raise TranspileError("break is only supported inside host callback for-loops")
            self._emit("break;")
            return
        if isinstance(stmt, ast.Continue):
            if self.loop_depth <= 0:
                raise TranspileError("continue is only supported inside host callback for-loops")
            self._emit("continue;")
            return
        if isinstance(stmt, ast.Pass):
            return
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == "print":
                for line in self._print_call(call):
                    self._emit(line)
                return
            if isinstance(call.func, ast.Name) and call.func.id == "state_set":
                self._emit(self._state_set_call(call))
                return
            if isinstance(call.func, ast.Name) and call.func.id == "state_add":
                self._emit(self._state_add_call(call))
                return
        raise TranspileError(
            "unsupported statement in host callback "
            f"{self.callback.name}: {ast.dump(stmt, include_attributes=False)}"
        )

    def _for_stmt(self, stmt: ast.For) -> None:
        if not isinstance(stmt.target, ast.Name):
            raise TranspileError("for-loops in host callbacks require a simple loop variable")
        if not isinstance(stmt.iter, ast.Call) or not isinstance(stmt.iter.func, ast.Name) or stmt.iter.func.id != "range":
            raise TranspileError("only for ... in range(...) loops are supported in host callbacks")
        if stmt.iter.keywords:
            raise TranspileError("range() keyword arguments are not supported in host callbacks")
        args = stmt.iter.args
        if len(args) == 1:
            start_expr = "0"
            stop_expr = self._expr(args[0])
            step_expr = "1"
        elif len(args) == 2:
            start_expr = self._expr(args[0])
            stop_expr = self._expr(args[1])
            step_expr = "1"
        elif len(args) == 3:
            start_expr = self._expr(args[0])
            stop_expr = self._expr(args[1])
            step_arg = args[2]
            if not isinstance(step_arg, ast.Constant) or not isinstance(step_arg.value, int) or step_arg.value <= 0:
                raise TranspileError("range(..., ..., step) currently requires a positive integer literal step")
            step_expr = str(step_arg.value)
        else:
            raise TranspileError("range() in host callbacks supports 1 to 3 positional arguments")
        if stmt.orelse:
            raise TranspileError("for-else is not supported in host callbacks")
        loop_var = stmt.target.id
        self._emit(f"for (int {loop_var} = {start_expr}; {loop_var} < {stop_expr}; {loop_var} += {step_expr}) {{")
        self.indent += 1
        self.loop_depth += 1
        for inner in stmt.body:
            self._stmt(inner)
        self.loop_depth -= 1
        self.indent -= 1
        self._emit("}")

    def _print_call(self, call: ast.Call) -> list[str]:
        if call.keywords:
            raise TranspileError("print() keyword arguments are not supported in host callbacks")
        idx = self.print_index
        self.print_index += 1
        lines = [f"std::ostringstream _nvbpf_oss_{idx};"]
        if not call.args:
            lines.append('printf("\\n");')
            return lines
        first = True
        for arg in call.args:
            expr = self._expr(arg)
            if first:
                lines.append(f"_nvbpf_oss_{idx} << {expr};")
                first = False
            else:
                lines.append(f'_nvbpf_oss_{idx} << " " << {expr};')
        lines.append(f'printf("%s\\n", _nvbpf_oss_{idx}.str().c_str());')
        return lines

    def _state_set_call(self, call: ast.Call) -> str:
        if len(call.args) != 3 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise TranspileError("state_set() expects a host-array name string literal, an index expression, and a value expression")
        state_name = call.args[0].value
        state_spec = self.host_states.get(state_name)
        if state_spec is None:
            raise TranspileError(f"state_set() references unknown host state {state_name!r}")
        if state_spec.kind != "array":
            raise TranspileError(f"state_set() expects a host array, got scalar {state_name!r}")
        return f"_nvbpf_state_{state_name}[{self._expr(call.args[1])}] = {self._expr(call.args[2])};"

    def _state_add_call(self, call: ast.Call) -> str:
        if len(call.args) not in (2, 3) or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise TranspileError("state_add() expects a host-array name string literal, an index expression, and an optional value expression")
        state_name = call.args[0].value
        state_spec = self.host_states.get(state_name)
        if state_spec is None:
            raise TranspileError(f"state_add() references unknown host state {state_name!r}")
        if state_spec.kind != "array":
            raise TranspileError(f"state_add() expects a host array, got scalar {state_name!r}")
        value_expr = "1" if len(call.args) == 2 else self._expr(call.args[2])
        return f"_nvbpf_state_{state_name}[{self._expr(call.args[1])}] += {value_expr};"

    def _expr(self, expr: ast.expr) -> str:
        if isinstance(expr, ast.Name):
            if expr.id in self.host_states:
                state_spec = self.host_states[expr.id]
                if state_spec.kind != "scalar":
                    raise TranspileError(f"host array {expr.id!r} must be accessed via state_get/state_set/state_add")
                return f"_nvbpf_state_{expr.id}"
            return expr.id
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return "true" if expr.value else "false"
            if isinstance(expr.value, str):
                return f'"{expr.value}"'
            return repr(expr.value)
        if isinstance(expr, ast.BinOp):
            op = _BIN_OPS.get(type(expr.op))
            if op is None:
                raise TranspileError(f"unsupported binary operator: {type(expr.op).__name__}")
            return f"({self._expr(expr.left)} {op} {self._expr(expr.right)})"
        if isinstance(expr, ast.BoolOp):
            op = "&&" if isinstance(expr.op, ast.And) else "||"
            return "(" + f" {op} ".join(self._expr(v) for v in expr.values) + ")"
        if isinstance(expr, ast.UnaryOp):
            if isinstance(expr.op, ast.Not):
                return f"!({self._expr(expr.operand)})"
            if isinstance(expr.op, ast.USub):
                return f"-({self._expr(expr.operand)})"
            raise TranspileError(f"unsupported unary operator: {type(expr.op).__name__}")
        if isinstance(expr, ast.Compare):
            if len(expr.ops) != 1 or len(expr.comparators) != 1:
                raise TranspileError("chained comparisons are not supported")
            op = _CMP_OPS.get(type(expr.ops[0]))
            if op is None:
                raise TranspileError(f"unsupported comparison operator: {type(expr.ops[0]).__name__}")
            return f"({self._expr(expr.left)} {op} {self._expr(expr.comparators[0])})"
        if isinstance(expr, ast.IfExp):
            return f"({self._expr(expr.test)} ? {self._expr(expr.body)} : {self._expr(expr.orelse)})"
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            func_name = expr.func.id
            if func_name == "int":
                if len(expr.args) != 1:
                    raise TranspileError("int() expects one argument")
                return f"((int){self._expr(expr.args[0])})"
            if func_name == "counter_value":
                if len(expr.args) != 1 or not isinstance(expr.args[0], ast.Constant) or not isinstance(expr.args[0].value, str):
                    raise TranspileError("counter_value() expects one string literal argument")
                counter_name = expr.args[0].value
                if counter_name not in self.counters:
                    raise TranspileError(f"counter_value() references unknown counter {counter_name!r}")
                return f"_nvbpf_read_counter_{counter_name}()"
            if func_name == "map_value":
                if len(expr.args) != 2 or not isinstance(expr.args[0], ast.Constant) or not isinstance(expr.args[0].value, str):
                    raise TranspileError("map_value() expects a map name string literal and an index expression")
                map_name = expr.args[0].value
                if map_name not in self.maps:
                    raise TranspileError(f"map_value() references unknown map {map_name!r}")
                return f"_nvbpf_read_map_{map_name}({self._expr(expr.args[1])})"
            if func_name == "state_get":
                if len(expr.args) != 2 or not isinstance(expr.args[0], ast.Constant) or not isinstance(expr.args[0].value, str):
                    raise TranspileError("state_get() expects a host-array name string literal and an index expression")
                state_name = expr.args[0].value
                state_spec = self.host_states.get(state_name)
                if state_spec is None:
                    raise TranspileError(f"state_get() references unknown host state {state_name!r}")
                if state_spec.kind != "array":
                    raise TranspileError(f"state_get() expects a host array, got scalar {state_name!r}")
                return f"_nvbpf_state_{state_name}[{self._expr(expr.args[1])}]"
            if func_name == "env_int":
                if len(expr.args) not in (1, 2) or not isinstance(expr.args[0], ast.Constant) or not isinstance(expr.args[0].value, str):
                    raise TranspileError("env_int() expects an environment-variable name string literal and an optional default")
                default_expr = "0" if len(expr.args) == 1 else self._expr(expr.args[1])
                return f'_nvbpf_env_int("{expr.args[0].value}", {default_expr})'
            if func_name == "env_flag":
                if len(expr.args) != 1 or not isinstance(expr.args[0], ast.Constant) or not isinstance(expr.args[0].value, str):
                    raise TranspileError("env_flag() expects an environment-variable name string literal")
                return f'(getenv("{expr.args[0].value}") != nullptr)'
            if func_name == "api_trace_value":
                if len(expr.args) != 1 or not isinstance(expr.args[0], ast.Constant) or not isinstance(expr.args[0].value, str):
                    raise TranspileError("api_trace_value() expects one trace name string literal")
                trace_name = expr.args[0].value
                if trace_name not in self.api_traces:
                    raise TranspileError(f"api_trace_value() references unknown api trace {trace_name!r}")
                return f"_nvbpf_read_api_trace_{trace_name}()"
            if func_name == "api_trace_bytes_value":
                if len(expr.args) != 1 or not isinstance(expr.args[0], ast.Constant) or not isinstance(expr.args[0].value, str):
                    raise TranspileError("api_trace_bytes_value() expects one trace name string literal")
                trace_name = expr.args[0].value
                if trace_name not in self.api_traces:
                    raise TranspileError(f"api_trace_bytes_value() references unknown api trace {trace_name!r}")
                return f"_nvbpf_read_api_trace_bytes_{trace_name}()"
            if func_name == "api_trace_correlated":
                if len(expr.args) != 1 or not isinstance(expr.args[0], ast.Constant) or not isinstance(expr.args[0].value, str):
                    raise TranspileError("api_trace_correlated() expects one trace name string literal")
                trace_name = expr.args[0].value
                trace = self.api_traces.get(trace_name)
                if trace is None:
                    raise TranspileError(f"api_trace_correlated() references unknown api trace {trace_name!r}")
                if not self.metadata_enabled:
                    raise TranspileError("api_trace_correlated() is only available in launch callbacks")
                if not trace.correlate_launches:
                    raise TranspileError(
                        f"api_trace_correlated() requires api_trace({trace_name!r}, correlate_launches=True)"
                    )
                return f"_nvbpf_api_trace_correlated_{trace_name}"
            if func_name == "api_trace_delta":
                if len(expr.args) != 1 or not isinstance(expr.args[0], ast.Constant) or not isinstance(expr.args[0].value, str):
                    raise TranspileError("api_trace_delta() expects one trace name string literal")
                trace_name = expr.args[0].value
                trace = self.api_traces.get(trace_name)
                if trace is None:
                    raise TranspileError(f"api_trace_delta() references unknown api trace {trace_name!r}")
                if not self.metadata_enabled:
                    raise TranspileError("api_trace_delta() is only available in launch callbacks")
                if not trace.correlate_launches:
                    raise TranspileError(
                        f"api_trace_delta() requires api_trace({trace_name!r}, correlate_launches=True)"
                    )
                return f"_nvbpf_api_trace_delta_{trace_name}"
            metadata = {
                "kernel_name": "func_name",
                "short_kernel_name": "_nvbpf_compact_kernel_name(func_name)",
                "grid_dim_x": "func_cfg.gridDimX",
                "grid_dim_y": "func_cfg.gridDimY",
                "grid_dim_z": "func_cfg.gridDimZ",
                "block_dim_x": "func_cfg.blockDimX",
                "block_dim_y": "func_cfg.blockDimY",
                "block_dim_z": "func_cfg.blockDimZ",
                "regs": "func_cfg.num_registers",
                "smem_static": "func_cfg.shmem_static_nbytes",
                "smem_dynamic": "func_cfg.shmem_dynamic_nbytes",
            }
            if func_name in metadata:
                if not self.metadata_enabled:
                    raise TranspileError(f"{func_name}() is only available in launch callbacks")
                if expr.args:
                    raise TranspileError(f"{func_name}() does not take arguments")
                return metadata[func_name]

        raise TranspileError(
            "unsupported expression in host callback "
            f"{self.callback.name}: {ast.dump(expr, include_attributes=False)}"
        )


def render_launch_exit_callback(
    callback: LaunchExitCallbackSpec,
    maps: dict[str, MapSpec],
    counters: set[str],
    host_states: dict[str, HostStateSpec],
    api_traces: dict[str, ApiTraceSpec],
) -> LaunchExitRender:
    return _HostCallbackTranspiler(
        callback,
        maps,
        counters,
        host_states,
        api_traces,
        metadata_enabled=True,
    ).render()


def render_launch_enter_callback(
    callback: LaunchEnterCallbackSpec,
    maps: dict[str, MapSpec],
    counters: set[str],
    host_states: dict[str, HostStateSpec],
    api_traces: dict[str, ApiTraceSpec],
) -> LaunchExitRender:
    return _HostCallbackTranspiler(
        callback,
        maps,
        counters,
        host_states,
        api_traces,
        metadata_enabled=True,
    ).render()


def render_tool_init_callback(
    callback: ToolInitCallbackSpec,
    maps: dict[str, MapSpec],
    counters: set[str],
    host_states: dict[str, HostStateSpec],
    api_traces: dict[str, ApiTraceSpec],
) -> LaunchExitRender:
    return _HostCallbackTranspiler(
        callback,
        maps,
        counters,
        host_states,
        api_traces,
        metadata_enabled=False,
    ).render()


def render_term_callback(
    callback: TermCallbackSpec,
    maps: dict[str, MapSpec],
    counters: set[str],
    host_states: dict[str, HostStateSpec],
    api_traces: dict[str, ApiTraceSpec],
) -> LaunchExitRender:
    return _HostCallbackTranspiler(
        callback,
        maps,
        counters,
        host_states,
        api_traces,
        metadata_enabled=False,
    ).render()
