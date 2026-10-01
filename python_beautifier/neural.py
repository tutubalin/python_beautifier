"""Best-effort, static architecture inference for ``torch.nn.Module``-like classes.

No PyTorch import, model execution, tracing or weights are involved.  The analyzer uses only
syntax already parsed by :class:`Source`: imported base aliases, local inheritance, assignments in
``__init__``, calls in ``forward`` and literal layer parameters.  It deliberately says ``Hout`` or
``?`` when a spatial dimension cannot be proved instead of presenting a guess as fact.
"""
from __future__ import annotations

import ast
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


def _inherited_specs(node: ast.ClassDef, classes: Dict[str, ast.ClassDef], nn: Set[str], layers: Set[str], called: Set[str], seen: Optional[Set[str]] = None) -> Dict[str, Spec]:
    seen = set() if seen is None else seen
    out: Dict[str, Spec] = {}
    for base in node.bases:
        key = _tail(_dotted(base))
        parent = classes.get(key)
        if parent is None or key in seen:
            continue
        seen.add(key)
        out.update(_inherited_specs(parent, classes, nn, layers, called, seen))
        out.update(_declared(parent, nn, layers, classes, called))
    return out


def _self_attr(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
        return node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name) and node.value.value.id == "self":
        idx = node.slice
        if isinstance(idx, ast.Constant):
            return f"{node.value.attr}.{idx.value}"
        return node.value.attr
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


def _make_spec(expr: ast.AST, name: str, nn: Set[str], layer_aliases: Dict[str, str], local: Dict[str, ast.ClassDef]) -> Optional[Spec]:
    if not isinstance(expr, ast.Call):
        return None
    called = _dotted(expr.func)
    kind = _known_class_kind(called, nn, layer_aliases)
    if kind is None and _tail(called) in local:
        kind = _tail(called)
        return Spec(kind, name, shown_params="custom module", custom=True)
    if kind is None and (_LAYER_RE.match(_tail(called)) or _tail(called) in _PRESERVE or _tail(called) == "Flatten"):
        kind = _tail(called)
        return Spec(kind, name, _args(expr), _show_args(_args(expr)), custom=True)
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
                    spec = _make_spec(child, f"{name}.{key_name}", nn, layer_aliases, local)
                    if spec:
                        children.append(spec)
                continue
            items = list(arg.elts) if isinstance(arg, (ast.List, ast.Tuple)) else [arg]
            for child in items:
                spec = _make_spec(child, f"{name}.{len(children)}", nn, layer_aliases, local)
                if spec:
                    children.append(spec)
        return Spec(kind, name, params, _show_args(params), children)
    return Spec(kind, name, params, _show_args(params), custom=False)


def _called_self_attrs(method: Optional[ast.FunctionDef | ast.AsyncFunctionDef]) -> Set[str]:
    if method is None:
        return set()
    return {name for call in ast.walk(method) if isinstance(call, ast.Call) and (name := _self_attr(call.func))}


def _declared(node: ast.ClassDef, nn: Set[str], layer_aliases: Dict[str, str], local: Dict[str, ast.ClassDef], called: Optional[Set[str]] = None) -> Dict[str, Spec]:
    init = _class_method(node, "__init__")
    specs: Dict[str, Spec] = {}
    called = called or set()
    if init is None:
        return specs
    statements = sorted(analysis.walk_scope(init), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
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
            spec = _make_spec(value, name or "", nn, layer_aliases, local)
            if name and spec is None and name in called and isinstance(value, ast.Call):
                # Imported/custom child module whose constructor is out of this file's reach.
                spec = Spec(_tail(_dotted(value.func)) or "CustomModule", name, shown_params="custom module", custom=True)
            if name and spec:
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
    pattern = re.compile(rf"\b{re.escape(input_name)}\b[^\n]*?(?:shape\s*)?[\(\[]([^()\[\]]{{1,100}})[\)\]]", re.I)
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
    if s == 1 and 2 * p == d * (k - 1):
        return dim
    stem = re.sub(r"out\d*$", "", dim)
    return f"{stem}out"


def _shape_after(kind: str, p: Dict[str, object], shape: Shape) -> Shape:
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
    if kind in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm", "LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "ReLU", "ReLU6", "GELU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "Identity", "ELU", "LeakyReLU", "SiLU", "Mish", "PReLU", "Hardtanh", "Hardswish", "Hardsigmoid"):
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
    def __init__(self, src: Source, specs: Dict[str, Spec], input_name: str, initial: Shape):
        self.src = src
        self.specs = specs
        self.input_name = input_name
        self.initial = initial
        self.steps: List[Step] = []
        self.notes: List[str] = []
        self.serial = 0
        self.used: Set[str] = set()
        self.loop_specs: Dict[str, List[Spec]] = {}

    def _id(self, base: str) -> str:
        self.serial += 1
        return f"n{self.serial}"

    def _spec_for(self, name: str) -> Optional[Spec]:
        if name in self.specs:
            return self.specs[name]
        head, dot, tail = name.partition(".")
        parent = self.specs.get(head)
        if parent and dot and tail.isdigit() and int(tail) < len(parent.children):
            return parent.children[int(tail)]
        if parent and dot:
            return next((c for c in parent.children if c.name.rsplit(".", 1)[-1] == tail), None)
        return None

    def _add(self, spec: Spec, value: _Value, line: int = 0) -> _Value:
        ident = self._id(spec.name)
        shape = _shape_after(spec.kind, spec.params, value.shape)
        known_layer = bool(_LAYER_RE.match(spec.kind) or spec.kind in _PRESERVE or spec.kind in ("Flatten", "Sequential", "ModuleList", "ModuleDict"))
        if spec.custom and not known_layer:
            shape = ("…", "?")  # a custom block's output rank/size is unknown without opening its implementation
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
                kind = attr.rsplit(".", 1)[-1]
                return self._add(Spec(kind, attr, custom=True, shown_params="custom/dynamic"), arg, getattr(node, "lineno", 0))
            if isinstance(node.func, ast.Name) and node.func.id in self.loop_specs:
                value = self._eval(node.args[0], env) if node.args else _Value(("input",), self.initial)
                for spec in self.loop_specs[node.func.id]:
                    value = self._apply(spec, value or _Value(("input",), self.initial), getattr(node, "lineno", 0))
                return value
            # Shape-changing tensor methods / torch functions between learned layers.
            fn = _dotted(node.func)
            tail = _tail(fn)
            base_value = None
            if isinstance(node.func, ast.Attribute):
                base_value = self._eval(node.func.value, env)
            shape_ops = ("flatten", "view", "reshape", "unflatten", "permute", "transpose", "movedim", "unsqueeze", "squeeze", "contiguous", "clone", "detach", "relu", "gelu", "sigmoid", "tanh", "softmax", "dropout", "to", "type_as", "expand", "repeat", "max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d", "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d")
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
                    "AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d",
                    "AdaptiveAvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
                }
                functional_layers = {"relu", "gelu", "sigmoid", "tanh", "softmax", "dropout"}
                if (normalized in shape_ops_that_matter and transformed != value.shape) or normalized in functional_layers:
                    labels = {
                        "flatten": "Flatten", "view": "Reshape", "reshape": "Reshape", "permute": "Permute",
                        "transpose": "Transpose", "movedim": "MoveDims", "unsqueeze": "Unsqueeze", "squeeze": "Squeeze",
                        "expand": "Expand", "repeat": "Repeat", "relu": "ReLU", "gelu": "GELU", "sigmoid": "Sigmoid",
                        "tanh": "Tanh", "softmax": "Softmax", "dropout": "Dropout",
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

    def _apply(self, spec: Spec, value: _Value, line: int) -> _Value:
        if spec.kind in ("Sequential", "ModuleList", "ModuleDict") and spec.children:
            for child in spec.children:
                value = self._apply(child, value, line)
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
                self.outputs.append(value.shape)
            return env
        if isinstance(stmt, ast.If):
            self._eval(stmt.test, env)
            before = dict(env)
            yes = self._block(stmt.body, dict(before))
            no = self._block(stmt.orelse, dict(before)) if stmt.orelse else dict(before)
            if stmt.orelse:
                self.notes.append(f"Conditional branches at line {stmt.lineno} are both shown; only one path runs at a time.")
            return self._join(yes, no)
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            old_loop_specs = dict(self.loop_specs)
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                self._eval(stmt.iter, env)
                self._bind(stmt.target, _Value(("input",), self.initial), env)
                attr = _self_attr(stmt.iter)
                spec = self._spec_for(attr) if attr else None
                if isinstance(stmt.target, ast.Name) and spec and spec.children:
                    self.loop_specs[stmt.target.id] = spec.children
            else:
                self._eval(stmt.test, env)
            self.notes.append(f"Loop at line {stmt.lineno} is shown once; it may execute zero or more times.")
            before = dict(env)
            body = self._block(stmt.body, dict(before))
            self.loop_specs = old_loop_specs
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
        self.outputs: List[Shape] = []
        self._block(forward.body, {})
        return self.steps, self.outputs, self.notes, self.used


def _product(dims: Sequence[str]) -> str:
    if all(d.isdigit() for d in dims):
        return str(math.prod(map(int, dims)))
    clean = [d for d in dims if d != "1"]
    return "×".join(clean) or "1"


def _initial_shape(specs: Dict[str, Spec]) -> Shape:
    def first(items: Iterable[Spec]) -> Optional[Spec]:
        for item in items:
            if item.children:
                nested = first(item.children)
                if nested:
                    return nested
            if item.kind in ("Linear", "LazyLinear"):
                return item
            if re.match(r"^(?:Conv(?:Transpose)?)[1-3]d$", item.kind):
                return item
        return None
    spec = first(specs.values())
    if spec is None:
        return ("…", "?")
    if spec.kind in ("Linear", "LazyLinear"):
        return ("…", str(spec.params.get("in_features") or "features"))
    rank = int(spec.kind[-2])
    return ("…", str(spec.params.get("in_channels") or "C"), *("H" if i == 0 else "W" if i == 1 else "D" for i in range(rank)))


def analyze(src: Source) -> Dict[int, Schema]:
    """Find module-like classes, then trace their declared layers through ``forward``."""
    classes = _classes(src.tree)
    nn, module_names, layer_aliases = _aliases(src.tree)
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
        forward = _class_method(node, "forward") or _inherited_method(node, classes, "forward")
        if forward is None:
            if name in module_like:
                result[id(node)] = Schema(
                    name, detection[name][0], detection[name][1], "input", ("…", "?"), [], [],
                    ["No forward() method is defined in this class."], 0,
                )
            continue
        called = _called_self_attrs(forward)
        specs = _inherited_specs(node, classes, nn, layer_aliases, called)
        specs.update(_declared(node, nn, layer_aliases, classes, called))
        direct_layers = [s for s in specs.values() if s.kind not in ("Sequential", "ModuleList", "ModuleDict") or s.children]
        if name not in module_like:
            # Structural lookalike: a forward method and at least one declared module-like child,
            # either from a known layer constructor or from a custom class with its own forward.
            custom_children = any(s.custom and s.kind in classes and _class_method(classes[s.kind], "forward") for s in specs.values())
            if not (forward and (direct_layers or custom_children)):
                continue
            detection[name] = ("lookalike", "has a forward() method and layer-like attributes")
        input_name = next((a.arg for a in forward.args.posonlyargs + forward.args.args if a.arg not in ("self", "cls")), "input")
        initial = _shape_hint(forward, src, input_name) or _initial_shape(specs)
        tracer = _Tracer(src, specs, input_name, initial)
        steps, outputs, notes, used = tracer.trace(forward)
        unused = [k for k in specs if k not in used]
        if unused:
            notes.append("Declared modules not observed in forward(): " + ", ".join(unused[:8]) + (" …" if len(unused) > 8 else "."))
        if not steps:
            notes.append("No declared layer calls could be traced from forward().")
        result[id(node)] = Schema(name, detection[name][0], detection[name][1], input_name, tracer.initial, steps, outputs, notes, len(specs))
    return result
