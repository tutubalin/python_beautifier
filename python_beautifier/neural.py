"""Best-effort, static architecture inference for ``torch.nn.Module``-like classes.

No PyTorch import, model execution, tracing or weights are involved.  The analyzer uses only
syntax already parsed by :class:`Source`: imported base aliases, local inheritance, assignments in
``__init__``, calls in ``forward`` and literal layer parameters.  It deliberately says ``Hout`` or
``?`` when a spatial dimension cannot be proved instead of presenting a guess as fact.
"""
from __future__ import annotations

import ast
import copy
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import analysis
from .source import Source

Shape = Tuple[str, ...]

# Well-known leaf modules and the shape rules we can apply without executing Python.
_PRESERVE = {
    "ReLU", "ReLU6", "GELU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "Dropout", "Dropout1d",
    "Dropout2d", "Dropout3d", "AlphaDropout", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "SyncBatchNorm", "LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
    "Identity", "ELU", "LeakyReLU", "PReLU", "SiLU", "Mish", "Hardtanh", "Hardswish", "Hardsigmoid",
    "AvgPool1d", "AvgPool2d", "AvgPool3d", "MaxPool1d", "MaxPool2d", "MaxPool3d", "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
    "Upsample", "Embedding", "EmbeddingBag", "MultiheadAttention", "TransformerEncoderLayer",
    "TransformerDecoderLayer", "RNN", "LSTM", "GRU", "RNNCell", "LSTMCell", "GRUCell",
    "RMSNorm", "QKNorm",
}
_LAYER_RE = re.compile(r"^(?:Linear|Bilinear|Conv(?:Transpose)?[1-3]d|LazyLinear|LazyConv[1-3]d)$")
_ARRAY_LIBRARIES = {"torch", "tensorflow", "keras", "jax"}


@dataclass
class Spec:
    """A module assigned in ``__init__`` (or a child of Sequential)."""

    kind: str
    name: str
    params: Dict[str, object] = field(default_factory=dict)
    shown_params: str = ""
    children: List["Spec"] = field(default_factory=list)
    custom: bool = False
    line: int = 0
    condition: Optional[str] = None


@dataclass
class Step:
    """One layer or merge observed in ``forward``."""

    ident: str
    name: str
    kind: str
    params: str
    incoming: Tuple[str, ...]
    in_shape: Shape
    out_shape: Shape
    line: int = 0
    merge: bool = False


@dataclass
class Route:
    """One statically traced method that executes layers."""

    method_name: str
    input_name: str
    input_shape: Shape
    steps: List[Step]
    output_shapes: List[Shape]
    notes: List[str]
    kind: str = "method"  # method | factory | attribute


@dataclass
class Schema:
    """Architecture summary attached to one class card."""

    name: str
    detection: str  # pytorch | inherited | lookalike
    reason: str
    input_name: str
    input_shape: Shape
    steps: List[Step]
    output_shapes: List[Shape]
    notes: List[str]
    declared: int = 0
    method_name: str = "forward"
    routes: List[Route] = field(default_factory=list)


def _dotted(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    return ""


def _tail(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _classes(tree: ast.AST) -> Dict[str, ast.ClassDef]:
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _imported_names(tree: ast.AST) -> Set[str]:
    """Names imported into this module, used to recognize opaque custom layer classes."""
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    return names


def _aliases(tree: ast.AST) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """(nn module aliases, Module class aliases, imported torch.nn layer aliases)."""
    nn: Set[str] = set()
    modules: Set[str] = set()
    layers: Dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "torch.nn":
                    nn.add(a.asname or "torch.nn")
                elif a.name == "torch":
                    nn.add(f"{a.asname or 'torch'}.nn")
                elif a.name in ("tensorflow.keras.layers", "keras.layers"):
                    nn.add(a.asname or a.name.rsplit(".", 1)[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module == "torch.nn" or n.module == "torch.nn.modules.module" or (n.module or "").startswith("torch.nn.modules."):
                for a in n.names:
                    if a.name == "Module":
                        modules.add(a.asname or a.name)
                    elif a.name in _PRESERVE or _LAYER_RE.match(a.name) or a.name in ("Sequential", "ModuleList", "ModuleDict", "Flatten"):
                        layers[a.asname or a.name] = a.name
            elif n.module == "torch":
                for a in n.names:
                    if a.name == "nn":
                        nn.add(a.asname or a.name)
            elif n.module in ("tensorflow.keras", "keras"):
                for a in n.names:
                    if a.name == "Model":
                        modules.add(a.asname or a.name)
                    elif a.name == "layers":
                        nn.add(a.asname or a.name)
            elif n.module in ("tensorflow.keras.layers", "keras.layers"):
                for a in n.names:
                    if a.name == "Layer":
                        modules.add(a.asname or a.name)
                    else:
                        layers[a.asname or a.name] = a.name
    return nn, modules, layers


def _is_module_base(expr: ast.expr, nn: Set[str], module_names: Set[str]) -> bool:
    name = _dotted(expr)
    if name in module_names:
        return True
    if name.endswith(("torch.nn.Module", "torch.nn.modules.module.Module", "tensorflow.keras.Model", "keras.Model", "keras.layers.Layer")):
        return True
    return any(name == f"{alias}.Module" or name == f"{alias}.Model" or name == f"{alias}.Layer" for alias in nn)


def _known_class_kind(name: str, nn: Set[str], layer_aliases: Dict[str, str]) -> Optional[str]:
    tail = _tail(name)
    if tail in layer_aliases:
        return layer_aliases[tail]
    if any(name.startswith(alias + ".") for alias in nn):
        return tail
    return None


def _class_method(node: ast.ClassDef, name: str) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    for s in node.body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == name:
            return s
    return None


def _inherited_method(
    node: ast.ClassDef,
    classes: Dict[str, ast.ClassDef],
    name: str,
    seen: Optional[Set[str]] = None,
) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    seen = set() if seen is None else seen
    for base in node.bases:
        key = _tail(_dotted(base))
        parent = classes.get(key)
        if parent is None or key in seen:
            continue
        seen.add(key)
        method = _class_method(parent, name)
        if method is not None:
            return method
        method = _inherited_method(parent, classes, name, seen)
        if method is not None:
            return method
    return None


def _inherited_specs(node: ast.ClassDef, classes: Dict[str, ast.ClassDef], nn: Set[str], layers: Set[str], called: Set[str], imported: Set[str], factories: Dict[str, ast.FunctionDef | ast.AsyncFunctionDef], seen: Optional[Set[str]] = None) -> Dict[str, Spec]:
    seen = set() if seen is None else seen
    out: Dict[str, Spec] = {}
    for base in node.bases:
        key = _tail(_dotted(base))
        parent = classes.get(key)
        if parent is None or key in seen:
            continue
        seen.add(key)
        out.update(_inherited_specs(parent, classes, nn, layers, called, imported, factories, seen))
        out.update(_declared(parent, nn, layers, classes, called, imported, factories))
    return out


def _self_attr(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
            return node.attr
        parent = _self_attr(node.value)
        return f"{parent}.{node.attr}" if parent else None
    if isinstance(node, ast.Subscript):
        parent = _self_attr(node.value)
        if parent:
            index = node.slice
            if isinstance(index, ast.Constant):
                return f"{parent}.{index.value}"
            return parent
    return None


def _literal(node: Optional[ast.AST]) -> object:
    """Safely evaluate a literal (or a small arithmetic expression over literals)."""
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        pass
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Pow)):
        left, right = _literal(node.left), _literal(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            try:
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
                if isinstance(node.op, ast.FloorDiv): return left // right
                if isinstance(node.op, ast.Div): return left / right
                return left ** right
            except (ArithmeticError, OverflowError):
                return None
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_literal(e) for e in node.elts]
        return values if all(v is not None for v in values) else None
    return None


def _literal_or_text(node: ast.AST) -> object:
    value = _literal(node)
    if value is not None:
        return value
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return None


def _args(call: ast.Call, kind: Optional[str] = None) -> Dict[str, object]:
    values: Dict[str, object] = {}
    names = {
        "Linear": ("in_features", "out_features", "bias"),
        "Conv1d": ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias", "padding_mode"),
        "Conv2d": ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias", "padding_mode"),
        "Conv3d": ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias", "padding_mode"),
        "ConvTranspose1d": ("in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation"),
        "ConvTranspose2d": ("in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation"),
        "ConvTranspose3d": ("in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation"),
        "BatchNorm1d": ("num_features",), "BatchNorm2d": ("num_features",), "BatchNorm3d": ("num_features",),
        "LayerNorm": ("normalized_shape",), "Embedding": ("num_embeddings", "embedding_dim"),
        "MaxPool1d": ("kernel_size", "stride", "padding", "dilation"),
        "MaxPool2d": ("kernel_size", "stride", "padding", "dilation"),
        "MaxPool3d": ("kernel_size", "stride", "padding", "dilation"),
        "AvgPool1d": ("kernel_size", "stride", "padding"),
        "AvgPool2d": ("kernel_size", "stride", "padding"),
        "AvgPool3d": ("kernel_size", "stride", "padding"),
        "AdaptiveAvgPool1d": ("output_size",), "AdaptiveAvgPool2d": ("output_size",), "AdaptiveAvgPool3d": ("output_size",),
        "AdaptiveMaxPool1d": ("output_size",), "AdaptiveMaxPool2d": ("output_size",), "AdaptiveMaxPool3d": ("output_size",),
        "Flatten": ("start_dim", "end_dim"), "Dropout": ("p",), "ReLU": ("inplace",),
        "GroupNorm": ("num_groups", "num_channels"), "MultiheadAttention": ("embed_dim", "num_heads"),
        "Upsample": ("size", "scale_factor", "mode", "align_corners"),
        "PixelShuffle": ("upscale_factor",), "PixelUnshuffle": ("downscale_factor",),
    }
    params = names.get(kind or _tail(_dotted(call.func)), ())
    for i, arg in enumerate(call.args):
        if i < len(params):
            values[params[i]] = _literal_or_text(arg)
    for kw in call.keywords:
        if kw.arg and (kw.arg in params or kw.arg in {"ceil_mode", "output_padding", "padding_mode", "eps", "momentum", "affine", "elementwise_affine", "num_heads", "batch_first"}):
            values[kw.arg] = _literal_or_text(kw.value)
    return values


def _show_args(values: Dict[str, object]) -> str:
    skip = {"bias": True, "inplace": False, "stride": None, "padding": 0, "dilation": 1, "groups": 1}
    out = []
    for k, v in values.items():
        if v is None or skip.get(k, object()) == v:
            continue
        out.append(f"{k}={v}")
    return ", ".join(out)


def _function_return(fn: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> Optional[ast.AST]:
    """Return expression of a simple factory; never execute the function."""
    if isinstance(fn, ast.Lambda):
        return fn.body
    for stmt in fn.body:
        if isinstance(stmt, ast.Return):
            return stmt.value
    return None


def _expand_factory(call: ast.Call, fn: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> Optional[ast.AST]:
    """Substitute the literal/source arguments into a simple factory's return expression."""
    returned = _function_return(fn)
    if returned is None:
        return None
    params = fn.args.posonlyargs + fn.args.args if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn.args.posonlyargs + fn.args.args
    defaults = [None] * (len(params) - len(fn.args.defaults)) + list(fn.args.defaults)
    bindings: Dict[str, ast.AST] = {}
    for i, param in enumerate(params):
        if param.arg in ("self", "cls"):
            continue
        if i < len(call.args):
            bindings[param.arg] = call.args[i]
        elif any(k.arg == param.arg for k in call.keywords):
            bindings[param.arg] = next(k.value for k in call.keywords if k.arg == param.arg)
        elif defaults[i] is not None:
            bindings[param.arg] = defaults[i]
    extra_keywords = [k for k in call.keywords if k.arg is not None and k.arg not in {p.arg for p in params}]
    packed: Dict[str, List[ast.keyword]] = {}
    if fn.args.kwarg:
        packed[fn.args.kwarg.arg] = extra_keywords
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for stmt in fn.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and _tail(_dotted(stmt.value.func)) == "dict":
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        packed[target.id] = [
                            ast.keyword(k.arg, copy.deepcopy(k.value))
                            for k in stmt.value.keywords if k.arg
                        ]

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            replacement = bindings.get(node.id)
            return ast.copy_location(copy.deepcopy(replacement), node) if replacement is not None else node

        def visit_Call(self, node: ast.Call):
            node.func = self.visit(node.func)
            node.args = [self.visit(a) for a in node.args]
            kws: List[ast.keyword] = []
            for kw in node.keywords:
                if kw.arg is None and isinstance(kw.value, ast.Name) and kw.value.id in packed:
                    kws.extend(ast.keyword(k.arg, self.visit(k.value)) for k in packed[kw.value.id])
                else:
                    kws.append(ast.keyword(kw.arg, self.visit(kw.value)))
            node.keywords = kws
            return node

    return ast.fix_missing_locations(Substitute().visit(copy.deepcopy(returned)))


def _make_spec(
    expr: ast.AST,
    name: str,
    nn: Set[str],
    layer_aliases: Dict[str, str],
    local: Dict[str, ast.ClassDef],
    factories: Optional[Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = None,
    seen_factories: Optional[Set[str]] = None,
) -> Optional[Spec]:
    if isinstance(expr, ast.IfExp):
        left = _make_spec(expr.body, name, nn, layer_aliases, local, factories, seen_factories)
        right = _make_spec(expr.orelse, name, nn, layer_aliases, local, factories, seen_factories)
        if left or right:
            out_channels = (left.params.get("out_channels") if left else None) or (right.params.get("out_channels") if right else None)
            params = {"condition": ast.unparse(expr.test)}
            if out_channels is not None:
                params["out_channels"] = out_channels
            alternatives = " or ".join(spec.kind if spec else "non-module" for spec in (left, right))
            children = [spec for spec in (left, right) if spec is not None]
            return Spec("Conditional layer", name, params, alternatives, children, custom=True, line=getattr(expr, "lineno", 0))
        return None
    if not isinstance(expr, ast.Call):
        return None
    called = _dotted(expr.func)
    tail = _tail(called)
    factories = factories or {}
    seen_factories = set() if seen_factories is None else seen_factories
    if tail in factories and tail not in seen_factories:
        expanded = _expand_factory(expr, factories[tail])
        if expanded is not None:
            return _make_spec(expanded, name, nn, layer_aliases, local, factories, seen_factories | {tail})
    kind = _known_class_kind(called, nn, layer_aliases)
    if kind is None and tail in local:
        kind = tail
        params: Dict[str, object] = {}
        init = _class_method(local[kind], "__init__")
        if init:
            formals = [a.arg for a in init.args.posonlyargs + init.args.args if a.arg not in ("self", "cls")]
            for formal, arg in zip(formals, expr.args):
                params[formal] = _literal_or_text(arg)
            for kw in expr.keywords:
                if kw.arg:
                    params[kw.arg] = _literal_or_text(kw.value)
            defaults = [None] * (len(formals) - len(init.args.defaults)) + list(init.args.defaults)
            for formal, default in zip(formals, defaults):
                if formal not in params and default is not None:
                    params[formal] = _literal_or_text(default)
            # Keep familiar channel names for common module signatures.
            if "n_in" in params:
                params["in_channels"] = params["n_in"]
            if "n_out" in params:
                params["out_channels"] = params["n_out"]
            if "in_channels" in params:
                params.setdefault("in_channels", params["in_channels"])
            if "out_channels" in params:
                params.setdefault("out_channels", params["out_channels"])
        shown = "custom module" + (f" (out_channels={params['out_channels']})" if "out_channels" in params else "")
        return Spec(kind, name, params, shown, custom=True, line=getattr(expr, "lineno", 0))
    if kind is None and (_LAYER_RE.match(_tail(called)) or _tail(called) in _PRESERVE or _tail(called) == "Flatten"):
        kind = _tail(called)
        return Spec(kind, name, _args(expr), _show_args(_args(expr)), custom=True, line=getattr(expr, "lineno", 0))
    if kind is None:
        return None
    params = _args(expr, kind)
    if kind in ("Sequential", "ModuleList", "ModuleDict"):
        children: List[Spec] = []
        for arg in expr.args:
            if kind == "ModuleDict" and isinstance(arg, ast.Dict):
                pairs = zip(arg.keys, arg.values)
                for key, child in pairs:
                    key_name = str(_literal(key) if _literal(key) is not None else len(children))
                    spec = _make_spec(child, f"{name}.{key_name}", nn, layer_aliases, local, factories, seen_factories)
                    if spec:
                        children.append(spec)
                continue
            items = list(arg.elts) if isinstance(arg, (ast.List, ast.Tuple)) else [arg.elt] if isinstance(arg, ast.ListComp) else [arg]
            for child in items:
                spec = _make_spec(child, f"{name}.{len(children)}", nn, layer_aliases, local, factories, seen_factories)
                if spec:
                    children.append(spec)
        return Spec(kind, name, params, _show_args(params), children, line=getattr(expr, "lineno", 0))
    return Spec(kind, name, params, _show_args(params), custom=False, line=getattr(expr, "lineno", 0))


def _called_self_attrs(method: Optional[ast.FunctionDef | ast.AsyncFunctionDef]) -> Set[str]:
    if method is None:
        return set()
    return {name for call in ast.walk(method) if isinstance(call, ast.Call) and (name := _self_attr(call.func))}


def _conditional_contexts(method: ast.FunctionDef | ast.AsyncFunctionDef) -> Dict[int, str]:
    """Map assignments nested in simple branches to the branch condition."""
    contexts: Dict[int, str] = {}

    def visit(stmts: Sequence[ast.stmt], conditions: Sequence[str]) -> None:
        for stmt in stmts:
            if conditions and isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                contexts[id(stmt)] = " and ".join(conditions)
            if isinstance(stmt, ast.If):
                test = ast.unparse(stmt.test)
                visit(stmt.body, (*conditions, test))
                visit(stmt.orelse, (*conditions, f"not ({test})"))
            else:
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child, ast.stmt):
                        visit([child], conditions)
                    elif isinstance(child, list):
                        visit([item for item in child if isinstance(item, ast.stmt)], conditions)
    visit(method.body, ())
    return contexts


def _safe_eval(node: ast.AST, values: Dict[str, object]) -> object:
    """Evaluate a small, side-effect-free expression subset for static parameter choices."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id, _UNKNOWN)
    if isinstance(node, (ast.Tuple, ast.List)):
        items = [_safe_eval(item, values) for item in node.elts]
        return _UNKNOWN if _UNKNOWN in items else tuple(items)
    if isinstance(node, ast.UnaryOp):
        value = _safe_eval(node.operand, values)
        if value is _UNKNOWN:
            return _UNKNOWN
        if isinstance(node.op, ast.Not): return not value
        if isinstance(node.op, ast.USub) and isinstance(value, (int, float)): return -value
        if isinstance(node.op, ast.UAdd) and isinstance(value, (int, float)): return value
    if isinstance(node, ast.BinOp):
        left, right = _safe_eval(node.left, values), _safe_eval(node.right, values)
        if left is _UNKNOWN or right is _UNKNOWN or not isinstance(left, (int, float)) or not isinstance(right, (int, float)): return _UNKNOWN
        try:
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.FloorDiv): return left // right
            if isinstance(node.op, ast.Div): return left / right
        except (TypeError, ArithmeticError):
            return _UNKNOWN
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left, right = _safe_eval(node.left, values), _safe_eval(node.comparators[0], values)
        if left is _UNKNOWN or right is _UNKNOWN: return _UNKNOWN
        op = node.ops[0]
        try:
            if isinstance(op, ast.Eq): return left == right
            if isinstance(op, ast.NotEq): return left != right
            if isinstance(op, ast.Lt): return left < right
            if isinstance(op, ast.LtE): return left <= right
            if isinstance(op, ast.Gt): return left > right
            if isinstance(op, ast.GtE): return left >= right
            if isinstance(op, ast.In): return left in right
            if isinstance(op, ast.NotIn): return left not in right
        except (TypeError, ValueError):
            return _UNKNOWN
    if isinstance(node, ast.BoolOp):
        vals = [_safe_eval(v, values) for v in node.values]
        if _UNKNOWN in vals: return _UNKNOWN
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    return _UNKNOWN


class _Unknown:
    pass


_UNKNOWN = _Unknown()


def _static_value(expression: str, values: Dict[str, object]) -> object:
    try:
        return _safe_eval(ast.parse(expression, mode="eval").body, values)
    except (SyntaxError, ValueError):
        return _UNKNOWN


def _substitute_expression(expression: str, values: Dict[str, object]) -> str:
    """Resolve constructor aliases inside symbolic expressions without evaluating user code."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return expression

    class ReplaceNames(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            if node.id not in values:
                return node
            value = values[node.id]
            if isinstance(value, bool):
                replacement: ast.AST = ast.Constant(value=value)
            elif isinstance(value, (int, float)):
                replacement = ast.Constant(value=value)
            elif isinstance(value, str):
                try:
                    replacement = ast.parse(value, mode="eval").body
                except SyntaxError:
                    return node
            else:
                return node
            return ast.copy_location(replacement, node)

    try:
        return ast.unparse(ast.fix_missing_locations(ReplaceNames().visit(tree)))
    except (ValueError, TypeError):
        return expression


def _specialize_spec(spec: Spec, values: Dict[str, object]) -> Optional[Spec]:
    """Resolve constructor parameters/conditional branches without executing model code."""
    if spec.condition:
        active = _static_value(spec.condition, values)
        if active is False:
            return None
    params: Dict[str, object] = {}
    for key, value in spec.params.items():
        if key == "condition":
            params[key] = value
        elif isinstance(value, str):
            resolved = _static_value(value, values)
            params[key] = _substitute_expression(value, values) if resolved is _UNKNOWN else resolved
        else:
            params[key] = value
    if spec.kind == "Conditional layer" and len(spec.children) >= 2:
        active = _static_value(str(params.get("condition", "")), values)
        if isinstance(active, bool):
            selected = spec.children[0 if active else 1]
            return _specialize_spec(selected, values)
    children = [resolved for child in spec.children if (resolved := _specialize_spec(child, values)) is not None]
    shown = spec.shown_params
    if spec.kind == "Conditional layer":
        shown = "conditional: " + shown
    elif spec.custom:
        shown = "custom module" + (f" (out_channels={params['out_channels']})" if params.get("out_channels") is not None else "")
    elif params:
        shown = _show_args(params)
    return Spec(spec.kind, spec.name, params, shown, children, spec.custom, spec.line, spec.condition)


def _declared(node: ast.ClassDef, nn: Set[str], layer_aliases: Dict[str, str], local: Dict[str, ast.ClassDef], called: Optional[Set[str]] = None, imported: Optional[Set[str]] = None, factories: Optional[Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = None) -> Dict[str, Spec]:
    init = _class_method(node, "__init__")
    specs: Dict[str, Spec] = {}
    called = called or set()
    imported = imported or set()
    if init is None:
        return specs
    conditional_contexts = _conditional_contexts(init)
    statements = sorted(analysis.walk_scope(init), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    factory_scope = dict(factories or {})
    formal_names = {arg.arg for arg in init.args.posonlyargs + init.args.args + init.args.kwonlyargs}
    local_values: Dict[str, ast.AST] = {}
    for stmt in statements:
        if not isinstance(stmt, ast.Assign):
            continue
        pairs: List[Tuple[ast.AST, ast.AST]] = []
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], (ast.Tuple, ast.List)) and isinstance(stmt.value, (ast.Tuple, ast.List)):
            pairs = list(zip(stmt.targets[0].elts, stmt.value.elts))
        else:
            pairs = [(target, stmt.value) for target in stmt.targets]
        for target, value in pairs:
            if isinstance(target, ast.Name):
                if isinstance(value, ast.Lambda):
                    factory_scope[target.id] = value
                elif target.id not in formal_names and not _self_attr(target):
                    local_values[target.id] = value

    class LocalSubstitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            if node.id in local_values:
                return ast.copy_location(copy.deepcopy(local_values[node.id]), node)
            return node

    for stmt in statements:
        value: Optional[ast.AST] = None
        target: Optional[ast.AST] = None
        if isinstance(stmt, ast.Assign):
            value = stmt.value
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign):
            value, targets = stmt.value, [stmt.target]
        else:
            continue
        for target in targets:
            name = _self_attr(target)
            expanded_value = ast.fix_missing_locations(LocalSubstitute().visit(copy.deepcopy(value))) if value is not None and local_values else value
            spec = _make_spec(expanded_value, name or "", nn, layer_aliases, local, factory_scope)
            if name and spec is None and name in called and isinstance(value, ast.Call):
                # Only treat an opaque constructor as a custom module when its class is defined
                # locally and has a forward body. Arbitrary callable attributes (e.g. object())
                # are not enough to classify an ordinary class as a neural model.
                kind = _tail(_dotted(value.func)) or "CustomModule"
                child_cls = local.get(kind)
                local_module = child_cls is not None and _class_method(child_cls, "forward") is not None
                imported_module = kind in imported and kind[:1].isupper() and kind not in {"Object", "Path"}
                if local_module or imported_module:
                    spec = Spec(kind, name, shown_params="custom module", custom=True)
            if name and spec:
                condition = conditional_contexts.get(id(stmt))
                # Keep an unconditional constructor assignment as the primary static path;
                # record a conditional reassignment only when no baseline exists.
                if condition and name in specs:
                    continue
                spec.condition = condition
                specs[name] = spec
    return specs


def _shape_hint(method: ast.FunctionDef | ast.AsyncFunctionDef, src: Source, input_name: str) -> Optional[Shape]:
    # A useful explicit Tensor[...] annotation, e.g. Tensor[B, 3, 224, 224].
    arg = next((a for a in method.args.posonlyargs + method.args.args if a.arg == input_name), None)
    if arg and isinstance(arg.annotation, ast.Subscript):
        base = _tail(_dotted(arg.annotation.value))
        if base in ("Tensor", "Size", "Shape"):
            sl = arg.annotation.slice
            dims = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
            shape = tuple(str(_literal(d)) if isinstance(_literal(d), (int, float)) else _tail(_dotted(d)) or "?" for d in dims)
            if shape:
                return shape
    doc = ast.get_docstring(method) or ""
    pattern = re.compile(rf"\b{re.escape(input_name)}\b[^\n]*?\bshape\b\s*(?:is\s*|=\s*)?`{{0,2}}[\(\[]([^()\[\]]{{1,100}})[\)\]]", re.I)
    match = pattern.search(doc)
    if match:
        dims = [p.strip() for p in match.group(1).split(",") if p.strip()]
        if 1 < len(dims) <= 8 and all(re.fullmatch(r"[A-Za-z_][\w]*|\d+|\.\.\.", p) for p in dims):
            return tuple(dims)
    return None


def _dim_tuple(value: object, rank: int, default: int = 1) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * rank
    if isinstance(value, (tuple, list)) and value and all(isinstance(x, int) for x in value):
        return tuple(value)
    return (default,) * rank


def _conv_out(dim: str, k: int, s: int, p: int, d: int, ceil: bool = False, padding: object = None) -> str:
    if dim.isdigit():
        x = int(dim)
        if padding == "same":
            return str(math.ceil(x / s))
        numerator = x + 2 * p - d * (k - 1) - 1
        return str((math.ceil if ceil else math.floor)(numerator / s) + 1)
    if padding == "same":
        return dim if s == 1 else f"ceil({dim}/{s})"
    if 2 * p == d * (k - 1):
        if s == 1:
            return dim
        symbolic = re.fullmatch(r"ceil\((.+)/(\d+)\)", dim)
        fractional = re.fullmatch(r"(.+)/(\d+)", dim)
        if symbolic or fractional:
            base, divisor = (symbolic or fractional).groups()
            return f"ceil({base}/{int(divisor) * s})"
        return f"ceil({dim}/{s})"
    match = re.fullmatch(r"(.*)out(\d*)", dim)
    stem, prior = (match.group(1), match.group(2)) if match else (dim, "")
    ordinal = int(prior or "1") + 1 if match else 1
    return f"{stem}out{ordinal if ordinal > 1 else ''}"


def _shape_after(kind: str, p: Dict[str, object], shape: Shape) -> Shape:
    if kind in ("Block", "Conditional layer"):
        channels = p.get("out_channels")
        if channels is not None and len(shape) >= 3:
            idx = len(shape) - 3
            dims = list(shape)
            # A conditional projection either maps to out_channels or is Identity only
            # when in_channels == out_channels, so both branches have this channel count.
            dims[idx] = str(channels)
            return tuple(dims)
        return shape
    if kind in ("Linear", "LazyLinear", "Bilinear"):
        out = p.get("out_features")
        if out is None:
            return shape or ("…", "features")
        return (shape[:-1] if shape and shape != ("?",) else ("…",)) + (str(out),)
    if kind == "Flatten":
        start = p.get("start_dim", 1)
        end = p.get("end_dim", -1)
        if isinstance(start, int) and isinstance(end, int) and shape:
            start = start + len(shape) if start < 0 else start
            end = end + len(shape) if end < 0 else end
            if 0 <= start <= end < len(shape):
                return shape[:start] + (_product(shape[start : end + 1]),) + shape[end + 1 :]
        return ("…", "features")
    if kind == "Embedding":
        dim = p.get("embedding_dim")
        if dim is not None:
            return (shape or ("…",)) + (str(dim),)
    m = re.match(r"^(Conv(?:Transpose)?|MaxPool|AvgPool)([1-3])d$", kind)
    if m:
        rank = int(m.group(2))
        if len(shape) < rank + 1 or (shape and shape[0] == "…" and len(shape) < rank + 2):
            shape = ("…", str(p.get("in_channels", "C")), *(f"S{i + 1}" for i in range(rank)))
        dims = list(shape)
        channel_idx = len(dims) - rank - 1
        if m.group(1) in ("Conv", "ConvTranspose"):
            dims[channel_idx] = str(p.get("out_channels", "Cout"))
        trans = m.group(1) == "ConvTranspose"
        pool = m.group(1) in ("MaxPool", "AvgPool")
        if not (pool and kind.startswith("Adaptive")):
            ks = _dim_tuple(p.get("kernel_size"), rank)
            default_stride = p.get("kernel_size") if pool else 1
            ss = _dim_tuple(p.get("stride") or default_stride, rank)
            ps = _dim_tuple(p.get("padding"), rank, 0)
            ds = _dim_tuple(p.get("dilation"), rank)
            for i in range(rank):
                dim_idx = len(dims) - rank + i
                if trans:
                    if dims[dim_idx].isdigit():
                        dims[dim_idx] = str((int(dims[dim_idx]) - 1) * ss[i] - 2 * ps[i] + ds[i] * (ks[i] - 1) + int(_dim_tuple(p.get("output_padding"), rank, 0)[i]) + 1)
                    else:
                        dims[dim_idx] += "out"
                else:
                    dims[dim_idx] = _conv_out(dims[dim_idx], ks[i], ss[i], ps[i], ds[i], bool(p.get("ceil_mode", False)), p.get("padding"))
        return tuple(dims)
    if kind == "Upsample":
        scale = p.get("scale_factor")
        size = p.get("size")
        rank = max(0, len(shape) - 2)
        if isinstance(size, int) and rank == 1:
            return shape[:-1] + (str(size),)
        if isinstance(size, (tuple, list)) and len(size) == rank:
            return shape[:-rank] + tuple(str(v) for v in size)
        scales = _dim_tuple(scale, rank, 1) if scale is not None else (1,) * rank
        if rank:
            dims = list(shape)
            for i, factor in enumerate(scales):
                index = len(dims) - rank + i
                if dims[index].isdigit():
                    dims[index] = str(int(dims[index]) * factor)
                elif factor != 1:
                    prior = re.fullmatch(r"(.+)×(\d+)", dims[index])
                    dims[index] = f"{prior.group(1)}×{int(prior.group(2)) * factor}" if prior else f"{dims[index]}×{factor}"
            return tuple(dims)
        return shape
    if kind in ("PixelShuffle", "PixelUnshuffle") and len(shape) >= 3:
        factor = p.get("upscale_factor", p.get("downscale_factor", 1))
        if isinstance(factor, int) and factor > 0:
            dims = list(shape)
            if kind == "PixelShuffle":
                channels = dims[-3]
                dims[-3] = str(int(channels) // (factor * factor)) if channels.isdigit() else f"{channels}/{factor * factor}"
                for i in (-2, -1):
                    if dims[i].isdigit():
                        dims[i] = str(int(dims[i]) * factor)
                    else:
                        prior = re.fullmatch(r"(.+)×(\d+)", dims[i])
                        dims[i] = f"{prior.group(1)}×{int(prior.group(2)) * factor}" if prior else f"{dims[i]}×{factor}"
            else:
                channels = dims[-3]
                dims[-3] = str(int(channels) * factor * factor) if channels.isdigit() else f"{channels}×{factor * factor}"
                for i in (-2, -1):
                    dims[i] = str(int(dims[i]) // factor) if dims[i].isdigit() else f"{dims[i]}/{factor}"
            return tuple(dims)
    if kind in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm", "LayerNorm", "GroupNorm", "RMSNorm", "QKNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "ReLU", "ReLU6", "GELU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "Identity", "ELU", "LeakyReLU", "SiLU", "Mish", "PReLU", "Hardtanh", "Hardswish", "Hardsigmoid"):
        return shape
    if kind.startswith("Adaptive") and "Pool" in kind:
        out_size = p.get("output_size")
        rank = int(kind[-2]) if kind[-2].isdigit() else 2
        sizes = _dim_tuple(out_size, rank, 1)
        if len(shape) >= rank:
            return shape[:-rank] + tuple(str(x) for x in sizes)
    return shape or ("…", "?")


def format_shape(shape: Shape) -> str:
    return "[" + ", ".join(shape) + "]" if shape else "[?]"


@dataclass
class _Value:
    producers: Tuple[str, ...]
    shape: Shape


class _Tracer:
    def __init__(
        self,
        src: Source,
        specs: Dict[str, Spec],
        input_name: str,
        initial: Shape,
        methods: Optional[Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = None,
        classes: Optional[Dict[str, ast.ClassDef]] = None,
        class_methods: Optional[Dict[str, Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]] = None,
        class_specs: Optional[Dict[str, Dict[str, Spec]]] = None,
    ):
        self.src = src
        self.specs = specs
        self.methods = methods or {}
        self.classes = classes or {}
        self.class_methods = class_methods or {}
        self.class_specs = class_specs or {}
        self.instance_params: Dict[str, object] = {}
        self.known_null_attrs: Set[str] = set()
        self.inline_stack: List[str] = []
        self.return_values: List[_Value] = []
        self.input_name = input_name
        self.initial = initial
        self.steps: List[Step] = []
        self.notes: List[str] = []
        self.serial = 0
        self.used: Set[str] = set()
        self.loop_specs: Dict[str, List[Spec]] = {}
        self.loop_modules: Dict[str, Spec] = {}

    def _id(self, base: str) -> str:
        self.serial += 1
        return f"n{self.serial}"

    def _spec_for(self, name: str) -> Optional[Spec]:
        if name in self.specs:
            return self.specs[name]
        parts = name.split(".")
        current = self.specs.get(parts[0])
        prefix = parts[0]
        if current is None:
            return None
        for part in parts[1:]:
            child: Optional[Spec] = None
            if part.isdigit() and int(part) < len(current.children):
                child = current.children[int(part)]
            elif current.children:
                child = next((item for item in current.children if item.name.rsplit(".", 1)[-1] == part), None)
            if child is None and current.custom:
                declared = self.class_specs.get(current.kind, {}).get(part)
                if declared is not None:
                    child = _specialize_spec(declared, current.params)
            if child is None:
                return None
            prefix += "." + part
            current = Spec(child.kind, prefix, dict(child.params), child.shown_params, list(child.children), child.custom, child.line, child.condition)
        return current

    def _add(self, spec: Spec, value: _Value, line: int = 0) -> _Value:
        ident = self._id(spec.name)
        shape = _shape_after(spec.kind, spec.params, value.shape)
        known_layer = bool(_LAYER_RE.match(spec.kind) or spec.kind in _PRESERVE or spec.kind in ("Flatten", "Sequential", "ModuleList", "ModuleDict"))
        if spec.custom and not known_layer and spec.kind not in ("Block", "Conditional layer"):
            shape = ("…", "?")  # an opaque custom block's output rank/size is unknown without opening its implementation
        expected = spec.params.get("in_features")
        if spec.kind in ("Linear", "LazyLinear") and isinstance(expected, int) and value.shape and value.shape[-1].isdigit() and int(value.shape[-1]) != expected:
            self.notes.append(f"{spec.name} expects {expected} input features, but static shape inference sees {value.shape[-1]} at line {line}.")
        conv = re.match(r"^Conv(?:Transpose)?([1-3])d$", spec.kind)
        expected_channels = spec.params.get("in_channels")
        if conv and isinstance(expected_channels, int):
            channel_idx = len(value.shape) - int(conv.group(1)) - 1
            if channel_idx >= 0 and value.shape[channel_idx].isdigit() and int(value.shape[channel_idx]) != expected_channels:
                self.notes.append(f"{spec.name} expects {expected_channels} input channels, but static shape inference sees {value.shape[channel_idx]} at line {line}.")
        self.steps.append(Step(ident, spec.name, spec.kind, spec.shown_params, value.producers or ("input",), value.shape, shape, line))
        self.used.add(spec.name.split(".")[0])
        return _Value((ident,), shape)

    def _eval(self, node: Optional[ast.AST], env: Dict[str, _Value]) -> Optional[_Value]:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id == self.input_name:
                return _Value(("input",), self.initial)
            return None
        if isinstance(node, ast.Call):
            attr = _self_attr(node.func)
            if attr:
                spec = self._spec_for(attr)
                arg = self._eval(node.args[0], env) if node.args else _Value(("input",), self.initial)
                if arg is None:
                    arg = _Value(("input",), self.initial)
                if spec:
                    return self._apply(spec, arg, getattr(node, "lineno", 0))
                if attr in self.methods and attr not in self.inline_stack and len(self.inline_stack) < 8:
                    return self._inline(self.methods[attr], node, env, arg)
                kind = attr.rsplit(".", 1)[-1]
                return self._add(Spec(kind, attr, custom=True, shown_params="custom/dynamic"), arg, getattr(node, "lineno", 0))
            if isinstance(node.func, ast.Name) and node.func.id in self.loop_specs:
                value = self._eval(node.args[0], env) if node.args else _Value(("input",), self.initial)
                for spec in self.loop_specs[node.func.id]:
                    value = self._apply(spec, value or _Value(("input",), self.initial), getattr(node, "lineno", 0))
                return value
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                loop_spec = self.loop_modules.get(node.func.value.id)
                if loop_spec is not None:
                    return self._apply_loop_method(loop_spec, node.func.attr, node, env)
            # Shape-changing tensor methods / torch functions between learned layers.
            fn = _dotted(node.func)
            tail = _tail(fn)
            base_value = None
            if isinstance(node.func, ast.Attribute):
                base_value = self._eval(node.func.value, env)
            shape_ops = ("flatten", "view", "reshape", "unflatten", "permute", "transpose", "movedim", "unsqueeze", "squeeze", "contiguous", "clone", "detach", "relu", "gelu", "sigmoid", "tanh", "softmax", "dropout", "to", "type_as", "expand", "repeat", "mean", "rsqrt", "max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d", "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d")
            if tail in shape_ops:
                value = base_value or (self._eval(node.args[0], env) if node.args else None)
                if value is None:
                    return None
                bound = isinstance(node.func, ast.Attribute) and not _dotted(node.func).startswith(("torch.", "nn.", "F."))
                normalized = tail
                if normalized.startswith(("max_pool", "avg_pool")):
                    normalized = "MaxPool" + normalized[-2] + "d" if normalized.startswith("max") else "AvgPool" + normalized[-2] + "d"
                elif normalized.startswith(("adaptive_avg_pool", "adaptive_max_pool")):
                    rank = normalized[-2]
                    normalized = ("AdaptiveAvgPool" if "avg" in normalized else "AdaptiveMaxPool") + rank + "d"
                transformed = self._transform(normalized, node, value.shape, bound=bound)
                shape_ops_that_matter = {
                    "flatten", "view", "reshape", "unflatten", "permute", "transpose", "movedim",
                    "unsqueeze", "squeeze", "expand", "repeat", "MaxPool1d", "MaxPool2d", "MaxPool3d",
                    "AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "mean",
                    "AdaptiveAvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
                }
                functional_layers = {"relu", "gelu", "sigmoid", "tanh", "softmax", "dropout", "rsqrt"}
                if (normalized in shape_ops_that_matter and transformed != value.shape) or normalized in functional_layers:
                    labels = {
                        "flatten": "Flatten", "view": "Reshape", "reshape": "Reshape", "permute": "Permute",
                        "transpose": "Transpose", "movedim": "MoveDims", "unsqueeze": "Unsqueeze", "squeeze": "Squeeze",
                        "expand": "Expand", "repeat": "Repeat", "relu": "ReLU", "gelu": "GELU", "sigmoid": "Sigmoid",
                        "tanh": "Tanh", "softmax": "Softmax", "dropout": "Dropout", "mean": "Mean", "rsqrt": "Rsqrt",
                    }
                    consumed = 1 if not bound else 0
                    shown = [ast.unparse(a) for a in node.args[consumed:]]
                    shown.extend(f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords if kw.arg)
                    ident = self._id(normalized)
                    label = labels.get(normalized, normalized)
                    self.steps.append(Step(ident, label, label, ", ".join(shown) or "shape-preserving operation", value.producers or ("input",), value.shape, transformed, getattr(node, "lineno", 0)))
                    return _Value((ident,), transformed)
                return _Value(value.producers, transformed)
            if tail in ("cat", "concat", "concatenate", "stack", "add"):
                tensors = node.args[0] if node.args and isinstance(node.args[0], (ast.Tuple, ast.List)) else None
                vals = [self._eval(x, env) for x in tensors.elts] if tensors else [self._eval(x, env) for x in node.args[:2]]
                vals = [v for v in vals if v is not None]
                if vals:
                    dim = next((_literal(k.value) for k in node.keywords if k.arg in ("dim", "axis")), 0)
                    dim = int(dim) if isinstance(dim, int) else 0
                    shape = self._combine_shape(tail, vals, dim)
                    names = {"cat": "Concat", "concat": "Concat", "concatenate": "Concat", "stack": "Stack", "add": "Add"}
                    label = names[tail]
                    ident = self._id(tail)
                    parents = tuple(dict.fromkeys(p for v in vals for p in v.producers))
                    params = f"dim={dim}"
                    if tail == "add" and any(v.shape != vals[0].shape for v in vals[1:]):
                        shape = ("?",)
                        self.notes.append(f"Elementwise merge at line {getattr(node, 'lineno', 0)} has different statically inferred shapes; broadcasting may apply.")
                    self.steps.append(Step(ident, label, label, params, parents or ("input",), vals[0].shape, shape, getattr(node, "lineno", 0), True))
                    return _Value((ident,), shape)
            if node.args:
                # Preserve lineage through common non-layer functions that wrap a tensor.
                return self._eval(node.args[0], env)
            return None
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, env)
            right = self._eval(node.right, env)
            if left and right:
                if isinstance(node.op, (ast.Add, ast.Sub)) and set(left.producers) != set(right.producers):
                    ident = self._id("add")
                    compatible = left.shape == right.shape
                    if not compatible:
                        self.notes.append(f"Elementwise merge at line {getattr(node, 'lineno', 0)} has different statically inferred shapes; broadcasting may apply.")
                    merged_shape = left.shape if compatible else ("?",)
                    self.steps.append(Step(ident, "+" if isinstance(node.op, ast.Add) else "−", "Add" if isinstance(node.op, ast.Add) else "Subtract", "elementwise merge", tuple(dict.fromkeys(left.producers + right.producers)), left.shape, merged_shape, getattr(node, "lineno", 0), True))
                    return _Value((ident,), merged_shape)
                return left
            return left or right
        if isinstance(node, ast.IfExp):
            condition = self._known_test(node.test)
            if condition is not _UNKNOWN and isinstance(condition, bool):
                return self._eval(node.body if condition else node.orelse, env)
            left_attr = _self_attr(node.body.func) if isinstance(node.body, ast.Call) else None
            right_attr = _self_attr(node.orelse.func) if isinstance(node.orelse, ast.Call) else None
            conditional_spec = self._spec_for(left_attr) if left_attr and left_attr == right_attr else None
            if conditional_spec and conditional_spec.kind == "Conditional layer":
                return self._eval(node.body, env)
            a = self._eval(node.body, env)
            b = self._eval(node.orelse, env)
            vals = [v for v in (a, b) if v]
            if not vals:
                return None
            return _Value(tuple(dict.fromkeys(p for v in vals for p in v.producers)), vals[0].shape)
        if isinstance(node, ast.Subscript):
            return self._eval(node.value, env)
        if isinstance(node, ast.Attribute):
            return self._eval(node.value, env)
        if isinstance(node, ast.UnaryOp):
            return self._eval(node.operand, env)
        if isinstance(node, (ast.Tuple, ast.List)):
            vals = [self._eval(x, env) for x in node.elts]
            vals = [v for v in vals if v]
            return _Value(tuple(dict.fromkeys(p for v in vals for p in v.producers)), vals[0].shape) if vals else None
        return None

    def _inline(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        call: ast.Call,
        caller_env: Dict[str, _Value],
        first_value: _Value,
    ) -> Optional[_Value]:
        """Inline a small local helper so an entrypoint can be traced through it."""
        params = [a for a in method.args.posonlyargs + method.args.args if a.arg not in ("self", "cls")]
        local: Dict[str, _Value] = {}
        if params:
            local[params[0].arg] = first_value
        for param, argument in zip(params[1:], call.args[1:]):
            value = self._eval(argument, caller_env)
            if value is not None:
                local[param.arg] = value
        by_name = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        for param in params[1:]:
            if param.arg in by_name:
                value = self._eval(by_name[param.arg], caller_env)
                if value is not None:
                    local[param.arg] = value
        saved_returns = self.return_values
        self.return_values = []
        self.inline_stack.append(method.name)
        try:
            self._block(method.body, local)
            returned = self.return_values
        finally:
            self.inline_stack.pop()
            self.return_values = saved_returns
        if not returned:
            return None
        shapes = [v.shape for v in returned]
        shape = shapes[0] if all(candidate == shapes[0] for candidate in shapes) else ("?",)
        producers = tuple(dict.fromkeys(p for value in returned for p in value.producers))
        return _Value(producers or ("input",), shape)

    def _apply_loop_method(self, spec: Spec, method: str, call: ast.Call, env: Dict[str, _Value]) -> _Value:
        """Keep a custom ModuleList method call visible when its tuple contract is opaque."""
        values = [value for arg in call.args if (value := self._eval(arg, env)) is not None]
        parents = tuple(dict.fromkeys(parent for value in values for parent in value.producers))
        shape = values[0].shape if values else ("…", "?")
        name = f"{spec.name.rsplit('.', 1)[0]}[*].{method}"
        dynamic = Spec(
            spec.kind, name, {}, f"custom block method {method}; return structure not expanded",
            custom=True, line=getattr(call, "lineno", 0),
        )
        self.notes.append(f"Custom ModuleList method {method} at line {getattr(call, 'lineno', 0)} is shown as one opaque step; its multiple outputs are not separated statically.")
        return self._add(dynamic, _Value(parents or ("input",), shape), getattr(call, "lineno", 0))

    def _apply_custom(self, spec: Spec, value: _Value, line: int) -> Optional[_Value]:
        kind = spec.kind
        forward = self.class_methods.get(kind, {}).get("forward")
        base_specs = self.class_specs.get(kind)
        if forward is None or base_specs is None or f"class:{kind}" in self.inline_stack:
            return None
        self.used.add(spec.name.split(".", 1)[0])
        params = dict(spec.params)
        specialized: Dict[str, Spec] = {}
        for name, child in base_specs.items():
            resolved = _specialize_spec(child, params)
            if resolved is not None:
                specialized[name] = resolved
                if resolved.condition:
                    note = f"{name} is conditional on {resolved.condition}; its candidate layer is shown, but runtime activation is unresolved."
                    if note not in self.notes:
                        self.notes.append(note)
        before_steps = len(self.steps)
        saved = (self.specs, self.methods, self.return_values, self.instance_params, self.known_null_attrs)
        self.specs = specialized
        self.methods = self.class_methods[kind]
        self.return_values = []
        self.instance_params = params
        self.known_null_attrs = set(base_specs) - set(specialized)
        self.inline_stack.append(f"class:{kind}")
        try:
            env = {next((arg.arg for arg in forward.args.args if arg.arg not in ("self", "cls")), "x"): value}
            self._block(forward.body, env)
            returned = self.return_values
        finally:
            self.inline_stack.pop()
            self.specs, self.methods, self.return_values, self.instance_params, self.known_null_attrs = saved
        if not returned:
            return None
        shape = returned[0].shape if all(v.shape == returned[0].shape for v in returned) else ("?",)
        producers = tuple(dict.fromkeys(p for v in returned for p in v.producers))
        if len(self.steps) == before_steps:
            placeholder = Spec(spec.kind, spec.name, dict(spec.params), spec.shown_params or "custom module", custom=True, line=line)
            return self._add(placeholder, value, line)
        return _Value(producers or value.producers, shape)

    def _known_test(self, node: ast.AST) -> object:
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            attr = _self_attr(node.left)
            right = node.comparators[0]
            if attr in self.known_null_attrs and isinstance(right, ast.Constant) and right.value is None:
                op = node.ops[0]
                if isinstance(op, ast.IsNot): return False
                if isinstance(op, ast.Is): return True
        return _safe_eval(node, self.instance_params)

    def _apply(self, spec: Spec, value: _Value, line: int) -> _Value:
        line = spec.line or line
        if spec.custom and spec.kind in self.classes:
            expanded = self._apply_custom(spec, value, line)
            if expanded is not None:
                return expanded
            if self.class_specs.get(spec.kind):
                self.notes.append(f"Custom module {spec.kind} at line {line} could not be expanded; its output shape is left unknown.")
        if spec.kind in ("Sequential", "ModuleList", "ModuleDict") and spec.children:
            for child in spec.children:
                value = self._apply(child, value, child.line or line)
            return value
        return self._add(spec, value, line)

    def _transform(self, kind: str, call: ast.Call, shape: Shape, *, bound: bool = False) -> Shape:
        args = [_literal(a) for a in call.args]
        kw = {k.arg: _literal(k.value) for k in call.keywords if k.arg}
        if kind.startswith(("MaxPool", "AvgPool")):
            rank = int(kind[-2])
            params = {
                "kernel_size": kw.get("kernel_size", args[1] if len(args) > 1 else 1),
                "stride": kw.get("stride", args[2] if len(args) > 2 else None),
                "padding": kw.get("padding", args[3] if len(args) > 3 else 0),
                "dilation": kw.get("dilation", args[4] if len(args) > 4 else 1),
                "ceil_mode": kw.get("ceil_mode", args[5] if len(args) > 5 else False),
            }
            return _shape_after(kind, params, shape)
        if kind.startswith(("AdaptiveAvgPool", "AdaptiveMaxPool")):
            rank = int(kind[-2])
            output_size = kw.get("output_size", args[1] if len(args) > 1 else None)
            sizes = _dim_tuple(output_size, rank, 1)
            return shape[:-rank] + tuple(str(x) for x in sizes) if len(shape) >= rank else ("…", "C", *(str(x) for x in sizes))
        if kind == "mean":
            dim = kw.get("dim", args[1] if len(args) > 1 else None)
            keepdim = kw.get("keepdim", args[2] if len(args) > 2 else False)
            dims = [dim] if isinstance(dim, int) else list(dim) if isinstance(dim, (tuple, list)) else None
            if dims is None:
                return (shape[0],) if keepdim and shape else ("…",)
            axes = sorted({axis % len(shape) for axis in dims})
            if keepdim:
                result = list(shape)
                for axis in axes:
                    result[axis] = "1"
                return tuple(result)
            return tuple(value for i, value in enumerate(shape) if i not in axes)
        if kind == "flatten":
            default_start = args[0] if bound and args else args[1] if len(args) > 1 else 0
            default_end = args[1] if bound and len(args) > 1 else args[2] if len(args) > 2 else -1
            start = kw.get("start_dim", default_start)
            end = kw.get("end_dim", default_end)
            if not isinstance(start, int) or not isinstance(end, int) or not shape:
                return ("…", "features")
            start = start + len(shape) if start < 0 else start
            end = end + len(shape) if end < 0 else end
            product = _product(shape[start : end + 1])
            return shape[:start] + (product,) + shape[end + 1 :]
        payload = args if bound else args[1:]
        if kind in ("view", "reshape"):
            dims = payload
            if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
                dims = list(dims[0])
            if dims and all(isinstance(x, int) for x in dims):
                return tuple("B" if x == -1 else str(x) for x in dims)
            return ("…", "features")
        if kind == "permute":
            dims = payload
            if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
                dims = list(dims[0])
            if dims and all(isinstance(x, int) and -len(shape) <= x < len(shape) for x in dims):
                return tuple(shape[x % len(shape)] for x in dims)
        if kind == "movedim" and len(payload) >= 2 and shape:
            source, destination = payload[:2]
            if isinstance(source, int) and isinstance(destination, int):
                dims = list(shape)
                item = dims.pop(source % len(dims))
                dims.insert(max(0, min(len(dims), destination % len(dims))), item)
                return tuple(dims)
        if kind == "unflatten" and len(payload) >= 2 and shape:
            dim, sizes = payload[:2]
            if isinstance(dim, int) and isinstance(sizes, (tuple, list)) and all(isinstance(x, int) for x in sizes):
                idx = dim % len(shape)
                return shape[:idx] + tuple(str(x) for x in sizes) + shape[idx + 1 :]
        if kind in ("expand", "repeat") and payload and shape:
            target = list(payload)
            if len(target) == 1 and isinstance(target[0], (tuple, list)):
                target = list(target[0])
            if target and all(isinstance(x, int) for x in target):
                if len(target) < len(shape):
                    target = [1] * (len(shape) - len(target)) + target
                current = ("1",) * max(0, len(target) - len(shape)) + shape
                if kind == "expand":
                    return tuple(current[i] if n == -1 else str(n) for i, n in enumerate(target))
                return tuple(str(int(current[i]) * n) if current[i].isdigit() else (current[i] if n == 1 else f"{current[i]}×{n}") for i, n in enumerate(target))
        if kind == "transpose" and len(payload) >= 2 and shape:
            a, b = payload[:2]
            if isinstance(a, int) and isinstance(b, int):
                dims = list(shape)
                a, b = a % len(dims), b % len(dims)
                dims[a], dims[b] = dims[b], dims[a]
                return tuple(dims)
        if kind == "unsqueeze" and payload and isinstance(payload[0], int):
            dims = list(shape)
            idx = max(0, min(len(dims), payload[0] % (len(dims) + 1)))
            dims.insert(idx, "1")
            return tuple(dims)
        if kind == "squeeze" and payload and isinstance(payload[0], int) and shape:
            dims = list(shape)
            idx = payload[0] % len(dims)
            if dims[idx] == "1":
                dims.pop(idx)
            else:
                dims[idx] = f"squeeze({dims[idx]})?"
            return tuple(dims)
        if kind in ("cat", "concat", "concatenate", "stack"):
            return shape
        return shape

    def _combine_shape(self, kind: str, vals: Sequence[_Value], dim: int) -> Shape:
        shape = list(vals[0].shape)
        if not shape:
            return ("…", "?")
        idx = dim % len(shape)
        sizes = [v.shape[idx] for v in vals if len(v.shape) == len(shape)]
        if kind == "stack":
            insert_at = dim + len(shape) + 1 if dim < 0 else dim
            shape.insert(max(0, min(len(shape), insert_at)), str(len(vals)))
        elif sizes and all(s.isdigit() for s in sizes):
            shape[idx] = str(sum(map(int, sizes)))
        else:
            shape[idx] = f"{shape[idx]}+…"
        return tuple(shape)

    def _stmt(self, stmt: ast.stmt, env: Dict[str, _Value]) -> Dict[str, _Value]:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value_node = stmt.value
            value = self._eval(value_node, env)
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if value:
                for target in targets:
                    self._bind(target, value, env)
            return env
        if isinstance(stmt, ast.Expr):
            self._eval(stmt.value, env)
            return env
        if isinstance(stmt, ast.Return):
            value = self._eval(stmt.value, env)
            if value:
                self.return_values.append(value)
            return env
        if isinstance(stmt, ast.If):
            condition = self._known_test(stmt.test)
            self._eval(stmt.test, env)
            before = dict(env)
            if isinstance(condition, bool):
                return self._block(stmt.body if condition else stmt.orelse, before)
            yes = self._block(stmt.body, dict(before))
            no = self._block(stmt.orelse, dict(before)) if stmt.orelse else dict(before)
            if stmt.orelse:
                self.notes.append(f"Conditional branches at line {stmt.lineno} are both shown; only one path runs at a time.")
            else:
                self.notes.append(f"Conditional branch at line {stmt.lineno} is shown as optional; it may be skipped at runtime.")
            return self._join(yes, no)
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            old_loop_specs = dict(self.loop_specs)
            old_loop_modules = dict(self.loop_modules)
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                self._eval(stmt.iter, env)
                self._bind(stmt.target, _Value(("input",), self.initial), env)
                iterable = stmt.iter
                if isinstance(iterable, ast.Call) and _tail(_dotted(iterable.func)) == "enumerate" and iterable.args:
                    iterable = iterable.args[0]
                attr = _self_attr(iterable)
                spec = self._spec_for(attr) if attr else None
                if spec and spec.children:
                    targets = [stmt.target] if isinstance(stmt.target, ast.Name) else list(stmt.target.elts) if isinstance(stmt.target, (ast.Tuple, ast.List)) else []
                    names = [target.id for target in targets if isinstance(target, ast.Name)]
                    if names:
                        module_var = names[-1] if isinstance(stmt.target, (ast.Tuple, ast.List)) else names[0]
                        self.loop_specs[module_var] = spec.children
                        self.loop_modules[module_var] = spec.children[0]
            else:
                self._eval(stmt.test, env)
            self.notes.append(f"Loop at line {stmt.lineno} is shown once; it may execute zero or more times.")
            before = dict(env)
            body = self._block(stmt.body, dict(before))
            self.loop_specs = old_loop_specs
            self.loop_modules = old_loop_modules
            return self._join(before, body)
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                self._eval(item.context_expr, env)
            return self._block(stmt.body, env)
        if isinstance(stmt, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            branches = [self._block(stmt.body, dict(env))]
            branches.extend(self._block(h.body, dict(env)) for h in stmt.handlers)
            if stmt.orelse:
                branches.append(self._block(stmt.orelse, dict(env)))
            if stmt.finalbody:
                branches = [self._block(stmt.finalbody, self._join(*branches))]
            if len(branches) > 1:
                self.notes.append(f"Try/except branches at line {stmt.lineno} are both shown; only one path runs at a time.")
            joined = branches[0]
            for b in branches[1:]:
                joined = self._join(joined, b)
            return joined
        if isinstance(stmt, ast.Match):
            branches = [self._block(c.body, dict(env)) for c in stmt.cases]
            if len(branches) > 1:
                self.notes.append(f"Match cases at line {stmt.lineno} are shown together; only one path runs at a time.")
            joined = dict(env)
            for b in branches:
                joined = self._join(joined, b)
            return joined
        return env

    def _bind(self, target: ast.AST, value: _Value, env: Dict[str, _Value]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                self._bind(t, value, env)

    def _join(self, *envs: Dict[str, _Value]) -> Dict[str, _Value]:
        keys = set().union(*(e.keys() for e in envs))
        out: Dict[str, _Value] = {}
        for key in keys:
            vals = [e[key] for e in envs if key in e]
            if vals:
                shape = vals[0].shape if all(v.shape == vals[0].shape for v in vals) else ("?",)
                out[key] = _Value(tuple(dict.fromkeys(p for v in vals for p in v.producers)), shape)
        return out

    def _block(self, stmts: Sequence[ast.stmt], env: Dict[str, _Value]) -> Dict[str, _Value]:
        for stmt in stmts:
            env = self._stmt(stmt, env)
        return env

    def trace(self, forward: ast.FunctionDef | ast.AsyncFunctionDef) -> Tuple[List[Step], List[Shape], List[str], Set[str]]:
        args = forward.args.posonlyargs + forward.args.args
        self.input_name = next((a.arg for a in args if a.arg not in ("self", "cls")), "input")
        self.initial = _shape_hint(forward, self.src, self.input_name) or self.initial
        self.return_values = []
        self._block(forward.body, {})
        self.outputs = [value.shape for value in self.return_values]
        return self.steps, self.outputs, self.notes, self.used


def _product(dims: Sequence[str]) -> str:
    if all(d.isdigit() for d in dims):
        return str(math.prod(map(int, dims)))
    clean = [d for d in dims if d != "1"]
    return "×".join(clean) or "1"


def _initial_shape(specs: Dict[str, Spec]) -> Shape:
    flat: List[Spec] = []

    def collect(items: Iterable[Spec]) -> None:
        for item in items:
            flat.append(item)
            if item.children:
                collect(item.children)

    collect(specs.values())
    first_index = next((i for i, item in enumerate(flat) if item.kind in ("Linear", "LazyLinear") or re.match(r"^(?:Conv(?:Transpose)?)[1-3]d$", item.kind)), None)
    if first_index is None:
        return ("…", "?")
    spec = flat[first_index]
    if spec.kind in ("Linear", "LazyLinear"):
        return ("…", str(spec.params.get("in_features") or "features"))
    rank = int(spec.kind[-2])
    channels = spec.params.get("in_channels") or "C"
    # Invert leading pixel pack/unpack operations to describe the factory's true input.
    for prefix in flat[:first_index]:
        factor = prefix.params.get("downscale_factor", prefix.params.get("upscale_factor", 1))
        if not isinstance(factor, int) or factor <= 0:
            continue
        if prefix.kind == "PixelUnshuffle":
            channels = str(int(channels) // (factor * factor)) if str(channels).isdigit() else f"{channels}/{factor * factor}"
        elif prefix.kind == "PixelShuffle":
            channels = str(int(channels) * factor * factor) if str(channels).isdigit() else f"{channels}×{factor * factor}"
    return ("…", str(channels), *("H" if i == 0 else "W" if i == 1 else "D" for i in range(rank)))


def _class_methods(node: ast.ClassDef, classes: Dict[str, ast.ClassDef], seen: Optional[Set[str]] = None) -> Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Methods visible on a class, including local bases, with child overrides winning."""
    seen = set() if seen is None else seen
    if node.name in seen:
        return {}
    seen.add(node.name)
    out: Dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for base in node.bases:
        parent = classes.get(_tail(_dotted(base)))
        if parent is not None:
            out.update(_class_methods(parent, classes, seen))
    out.update({
        s.name: s for s in node.body
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
    })
    return out


def _declared_ref(attr: str, specs: Dict[str, Spec], class_specs: Optional[Dict[str, Dict[str, Spec]]] = None) -> bool:
    if attr in specs:
        return True
    parts = attr.split(".")
    current = specs.get(parts[0])
    if current is None:
        return False
    for part in parts[1:]:
        child = None
        if part.isdigit() and int(part) < len(current.children):
            child = current.children[int(part)]
        elif current.children:
            child = next((item for item in current.children if item.name.rsplit(".", 1)[-1] == part), None)
        if child is None and current.custom:
            child = (class_specs or {}).get(current.kind, {}).get(part)
        if child is None:
            return False
        current = child
    return True


def _layer_calls(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    specs: Dict[str, Spec],
    class_specs: Optional[Dict[str, Dict[str, Spec]]] = None,
) -> Set[str]:
    """Calls to declared module attributes, recognized by expression content, not method name."""
    nodes = analysis.walk_scope(method)
    refs = {
        attr for call in nodes if isinstance(call, ast.Call)
        if (attr := _self_attr(call.func)) and _declared_ref(attr, specs, class_specs)
    }
    # A ModuleList/Sequential is often invoked through its loop variable, so the module
    # attribute occurs in the iterator expression rather than as a call target.
    for loop in analysis.walk_scope(method):
        if not isinstance(loop, (ast.For, ast.AsyncFor)):
            continue
        iterable = loop.iter
        if isinstance(iterable, ast.Call) and _tail(_dotted(iterable.func)) == "enumerate" and iterable.args:
            iterable = iterable.args[0]
        attr = _self_attr(iterable)
        if attr and _declared_ref(attr, specs, class_specs):
            refs.add(attr)
    return refs


def _method_calls(method: ast.FunctionDef | ast.AsyncFunctionDef) -> Set[str]:
    """Calls from this method to other methods on self/cls."""
    return {
        attr for call in analysis.walk_scope(method) if isinstance(call, ast.Call)
        if (attr := _self_attr(call.func)) and "." not in attr
    }


_FUNCTIONAL_OPS = {
    "relu", "gelu", "tanh", "sigmoid", "softmax", "log_softmax", "dropout",
    "flatten", "reshape", "view", "permute", "transpose", "movedim", "unsqueeze", "squeeze",
    "max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "mean", "rsqrt",
}


def _functional_calls(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Known tensor operations are content evidence for functional-only module methods."""
    return any(
        isinstance(node, ast.Call) and _tail(_dotted(node.func)).lower() in _FUNCTIONAL_OPS
        for node in analysis.walk_scope(method)
    )


def _execution_methods(
    methods: Dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    specs: Dict[str, Spec],
    class_specs: Optional[Dict[str, Dict[str, Spec]]] = None,
) -> List[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Find top-level methods whose bodies reach calls to declared layers.

    ``forward`` is preferred only when it is actually present in the content-derived call graph.
    Otherwise methods such as ``forward_kv_extract``, ``forward_kv_cached`` or ``run_encoder`` are
    found because they call layer attributes. Helpers called by another layer-using method are
    treated as implementation details rather than separate entry points.
    """
    candidates = {name: method for name, method in methods.items() if name not in ("__init__", "__new__")}
    relevant = {name for name, method in candidates.items() if _layer_calls(method, specs, class_specs) or _functional_calls(method)}
    # If an entrypoint delegates to an internal helper, include the caller in the content-derived
    # graph. _Tracer inlines these local helpers, so the path still ends at the actual layer nodes.
    changed = True
    while changed:
        changed = False
        for name, method in candidates.items():
            if name not in relevant and _method_calls(method) & relevant:
                relevant.add(name)
                changed = True
    called_by = {
        callee for name in relevant for callee in _method_calls(candidates[name])
        if callee in relevant
    }
    roots = [candidates[name] for name in candidates if name in relevant and name not in called_by]
    return roots or [candidates[name] for name in candidates if name in relevant]


def _factory_schema(
    src: Source,
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    nn: Set[str],
    layer_aliases: Dict[str, str],
    classes: Dict[str, ast.ClassDef],
    factories: Dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    class_methods: Dict[str, Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    class_specs: Dict[str, Dict[str, Spec]],
) -> Optional[Schema]:
    """A function returning a Sequential is itself a useful, callable model architecture."""
    call = ast.Call(func=ast.Name(id=fn.name, ctx=ast.Load()), args=[], keywords=[])
    spec = _make_spec(call, fn.name, nn, layer_aliases, classes, factories)
    if spec is None or spec.kind not in ("Sequential", "ModuleList", "ModuleDict") or not spec.children:
        return None
    initial = _initial_shape({fn.name: spec})
    input_name = "image" if "encoder" in fn.name.lower() else "latent"
    tracer = _Tracer(src, {fn.name: spec}, input_name, initial, classes=classes, class_methods=class_methods, class_specs=class_specs)
    value = tracer._apply(spec, _Value(("input",), initial), getattr(fn, "lineno", 0))
    route = Route(fn.name, input_name, initial, tracer.steps, [value.shape], tracer.notes, "factory")
    return Schema(fn.name, "factory", "returns a statically expandable Sequential of layer modules", input_name, initial, route.steps, route.output_shapes, route.notes, len(spec.children), fn.name, [route])


def _conditional_factory_notes(
    init: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    factories: Dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> Dict[str, List[str]]:
    """Explain conditional factory replacements without pretending one branch is the default."""
    if init is None:
        return {}
    contexts = _conditional_contexts(init)
    notes: Dict[str, List[str]] = {}
    for stmt in analysis.walk_scope(init):
        if id(stmt) not in contexts or not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        value = stmt.value
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        pairs: List[Tuple[ast.AST, ast.AST]] = []
        if len(targets) == 1 and isinstance(targets[0], (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            pairs = list(zip(targets[0].elts, value.elts))
        else:
            pairs = [(target, value) for target in targets]
        for target, expr in pairs:
            attr = _self_attr(target)
            if not attr or not isinstance(expr, ast.Call):
                continue
            factory = _tail(_dotted(expr.func))
            if factory not in factories:
                continue
            note = f"Conditional architecture: {attr} is replaced by {factory}(…) when {contexts[id(stmt)]}; the factory has its own schema path."
            notes.setdefault(attr, []).append(note)
    return notes


def _attribute_routes(
    src: Source,
    specs: Dict[str, Spec],
    classes: Dict[str, ast.ClassDef],
    class_methods: Dict[str, Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    class_specs: Dict[str, Dict[str, Spec]],
    conditional_notes: Optional[Dict[str, List[str]]] = None,
) -> List[Route]:
    """A wrapper with declared encoder/decoder stacks but no own forward still has useful paths."""
    routes: List[Route] = []
    for name, spec in specs.items():
        if spec.kind not in ("Sequential", "ModuleList", "ModuleDict") or not spec.children:
            continue
        initial = _initial_shape({name: spec})
        input_name = "image" if "encoder" in name.lower() else "latent"
        tracer = _Tracer(src, {name: spec}, input_name, initial, classes=classes, class_methods=class_methods, class_specs=class_specs)
        value = tracer._apply(spec, _Value(("input",), initial), spec.line)
        notes = list(dict.fromkeys([*tracer.notes, *((conditional_notes or {}).get(name, []))]))
        routes.append(Route(name, input_name, initial, tracer.steps, [value.shape], notes, "attribute"))
    return routes


def analyze(src: Source) -> Dict[int, Schema]:
    """Find module-like classes and trace methods whose contents invoke declared layers."""
    classes = _classes(src.tree)
    nn, module_names, layer_aliases = _aliases(src.tree)
    imported = _imported_names(src.tree)
    factories = {n.name: n for n in src.tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    class_methods = {name: _class_methods(node, classes) for name, node in classes.items()}
    class_specs: Dict[str, Dict[str, Spec]] = {}
    for class_name, class_node in classes.items():
        visible = class_methods[class_name]
        refs = set().union(*(_called_self_attrs(method) for method in visible.values())) if visible else set()
        declared = _inherited_specs(class_node, classes, nn, layer_aliases, refs, imported, factories)
        declared.update(_declared(class_node, nn, layer_aliases, classes, refs, imported, factories))
        class_specs[class_name] = declared
    module_like: Set[str] = set()
    detection: Dict[str, Tuple[str, str]] = {}
    # Resolve inheritance to a fixed point so local intermediate base classes work.
    for _ in range(len(classes) + 1):
        changed = False
        for name, node in classes.items():
            if name in module_like:
                continue
            direct = any(_is_module_base(base, nn, module_names) for base in node.bases)
            indirect = any(_tail(_dotted(base)) in module_like for base in node.bases)
            if direct or indirect:
                module_like.add(name)
                detection[name] = ("pytorch" if direct else "inherited", "inherits a PyTorch/Keras module base" if direct else "inherits a locally declared module base")
                changed = True
        if not changed:
            break
    result: Dict[int, Schema] = {}
    for name, node in classes.items():
        methods = class_methods[name]
        called = set().union(*(_called_self_attrs(m) for m in methods.values())) if methods else set()
        specs = _inherited_specs(node, classes, nn, layer_aliases, called, imported, factories)
        specs.update(_declared(node, nn, layer_aliases, classes, called, imported, factories))
        routes_methods = _execution_methods(methods, specs, class_specs)
        direct_layers = [s for s in specs.values() if s.kind not in ("Parameter", "Buffer") and (s.kind not in ("Sequential", "ModuleList", "ModuleDict") or s.children)]

        if name not in module_like:
            # A plain class only qualifies if its code actually calls a declared module-like child.
            custom_children = any(s.custom and s.kind in classes and _class_method(classes[s.kind], "forward") for s in specs.values())
            if not (routes_methods and (direct_layers or custom_children)):
                continue
            detection[name] = ("lookalike", "contains methods that call layer-like attributes")

        routes: List[Route] = []
        for method in routes_methods:
            input_name = next((a.arg for a in method.args.posonlyargs + method.args.args if a.arg not in ("self", "cls")), "input")
            initial = _shape_hint(method, src, input_name) or _initial_shape(specs)
            tracer = _Tracer(src, specs, input_name, initial, methods, classes, class_methods, class_specs)
            steps, outputs, notes, used = tracer.trace(method)
            notes[:] = list(dict.fromkeys(notes))
            unused = [k for k, spec in specs.items() if spec.kind not in ("Parameter", "Buffer") and k not in used]
            if unused:
                notes.append("Declared modules not observed in this method: " + ", ".join(unused[:8]) + (" …" if len(unused) > 8 else "."))
            if not steps:
                notes.append("No declared layer calls could be traced from this method.")
            routes.append(Route(method.name, input_name, tracer.initial, steps, outputs, notes))

        if not routes and name in module_like:
            conditional_notes = _conditional_factory_notes(_class_method(node, "__init__"), factories)
            routes = _attribute_routes(src, specs, classes, class_methods, class_specs, conditional_notes)
            if routes:
                routes[0].notes.append("This wrapper has no own forward method; the displayed paths are its declared module stacks.")
            else:
                child_names = [key for key, child in specs.items() if child.kind != "Parameter"]
                explanation = (
                    f"Declares child modules ({', '.join(child_names[:8])}) but has no standalone execution path; a parent method may call these components."
                    if child_names else
                    "No method body calling a declared layer was detected, and no sequential child stack was found."
                )
                result[id(node)] = Schema(
                    name, detection[name][0], detection[name][1], "input", ("…", "?"), [], [],
                    [explanation], len(specs), "", [],
                )
                continue
        if not routes:
            continue

        primary = routes[0]
        result[id(node)] = Schema(
            name, detection[name][0], detection[name][1], primary.input_name, primary.input_shape,
            primary.steps, primary.output_shapes, primary.notes, len(specs), primary.method_name, routes,
        )
    for fn in factories.values():
        schema = _factory_schema(src, fn, nn, layer_aliases, classes, factories, class_methods, class_specs)
        if schema is not None:
            result[id(fn)] = schema
    return result
