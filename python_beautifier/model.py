"""Build a structured, analysed model of a Python module.

The renderer never walks raw AST for *documentation* purposes; it asks the
model.  The model merges three sources of truth for every definition:
the signature (AST), the docstring (:mod:`docstrings`) and the body
(:mod:`analysis`).
"""
from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import analysis
from .docstrings import Doc, parse as parse_doc
from .source import Comment, Source

FUNC_NODES = analysis.FUNC_NODES

# --------------------------------------------------------------------------- #
#  records
# --------------------------------------------------------------------------- #


@dataclass
class ParamInfo:
    name: str
    kind: str  # posonly | normal | kwonly | vararg | kwarg
    annotation: Optional[str] = None
    default: Optional[str] = None
    doc_default: str = ""  # default mentioned in the docstring only
    doc_type: str = ""
    desc: str = ""
    comment: str = ""
    inferred: str = ""
    documented: bool = False
    optional_doc: bool = False
    line: int = 0

    @property
    def required(self) -> bool:
        return self.default is None and self.kind in ("posonly", "normal", "kwonly")

    @property
    def shown_name(self) -> str:
        return {"vararg": "*", "kwarg": "**"}.get(self.kind, "") + self.name

    @property
    def type(self) -> str:
        return self.annotation or self.doc_type or ""


@dataclass
class AttrInfo:
    name: str
    type: str = ""
    type_origin: str = ""  # annotation | doc | inferred
    default: Optional[str] = None
    desc: str = ""
    scope: str = "class"  # class | instance | field | member | documented
    line: int = 0
    set_in: str = ""  # method that assigns it (instance attributes)


@dataclass
class Todo:
    tag: str
    text: str
    line: int


@dataclass
class ImportInfo:
    module: str
    names: List[Tuple[str, str]]  # (name, alias)
    alias: str
    level: int
    group: str  # future | stdlib | third | local
    line: int
    is_from: bool


@dataclass
class ConstInfo:
    name: str
    type: str
    value: str
    line: int
    comment: str = ""
    inferred: bool = False


@dataclass
class DefInfo:
    node: ast.AST
    kind: str  # class | function
    name: str
    qualname: str
    anchor: str
    parent: Optional["DefInfo"]
    depth: int
    doc: Doc
    first_line: int
    last_line: int
    loc: int
    cc: int
    grade: str
    nesting: int
    uses: analysis.Uses
    role: str = "function"  # function | method | classmethod | staticmethod | property | class
    is_async: bool = False
    is_generator: bool = False
    params: List[ParamInfo] = field(default_factory=list)
    returns: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    children: List["DefInfo"] = field(default_factory=list)
    callees: List["DefInfo"] = field(default_factory=list)
    callers: List["DefInfo"] = field(default_factory=list)
    external_calls: Dict[str, int] = field(default_factory=dict)
    attrs: List[AttrInfo] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    class_keywords: List[str] = field(default_factory=list)
    badges: List[str] = field(default_factory=list)
    recursive: bool = False
    has_docstring: bool = False
    docstring_node: Optional[ast.stmt] = None
    typed_params: int = 0
    total_params: int = 0

    @property
    def is_class(self) -> bool:
        return self.kind == "class"

    @property
    def is_method(self) -> bool:
        return self.kind == "function" and self.parent is not None and self.parent.is_class

    @property
    def methods(self) -> List["DefInfo"]:
        return [c for c in self.children if not c.is_class]

    @property
    def self_name(self) -> Optional[str]:
        if not self.is_method or self.role == "staticmethod":
            return None
        a = self.node.args  # type: ignore[attr-defined]
        first = (a.posonlyargs + a.args)[:1]
        return first[0].arg if first else None

    @property
    def total_cc(self) -> int:
        return self.cc + sum(c.total_cc for c in self.children)


@dataclass
class ModuleInfo:
    src: Source
    doc: Doc
    docstring_node: Optional[ast.stmt]
    defs: List[DefInfo]
    top: List[DefInfo]
    by_node: Dict[int, DefInfo]
    imports: List[ImportInfo]
    consts: List[ConstInfo]
    todos: List[Todo]
    shebang: str = ""
    encoding: str = ""
    all_names: Optional[List[str]] = None
    meta: Dict[str, str] = field(default_factory=dict)
    header_comments: List[Comment] = field(default_factory=list)
    features: List[str] = field(default_factory=list)

    @property
    def classes(self) -> List[DefInfo]:
        return [d for d in self.defs if d.is_class]

    @property
    def functions(self) -> List[DefInfo]:
        return [d for d in self.defs if not d.is_class and not d.is_method]

    @property
    def methods(self) -> List[DefInfo]:
        return [d for d in self.defs if d.is_method]


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

_TODO_RE = re.compile(r"\b(TODO|FIXME|HACK|XXX|BUG|OPTIMIZE|REVIEW|NOTE)\b\s*(?:\([^)]*\))?\s*[:\-]?\s*(.*)")


def compact(src: Source, node: ast.AST) -> str:
    """Source of *node* on one line; falls back to ``ast.unparse``."""
    try:
        if node.lineno == node.end_lineno:  # type: ignore[attr-defined]
            return src.text_of(node)
    except AttributeError:
        pass
    return src.oneline(node)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def docstring_of(node: ast.AST) -> Tuple[Optional[str], Optional[ast.stmt]]:
    body = getattr(node, "body", None)
    if not body or not isinstance(body, list):
        return None, None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.value.value, first
    return None, None


_STDLIB_CACHE: Optional[Set[str]] = None


def stdlib_modules() -> Set[str]:
    global _STDLIB_CACHE
    if _STDLIB_CACHE is None:
        names = getattr(sys, "stdlib_module_names", None)
        if names:
            _STDLIB_CACHE = set(names)
        else:  # Python 3.9
            import pkgutil
            import sysconfig

            found = set(sys.builtin_module_names)
            try:
                std = sysconfig.get_paths()["stdlib"]
                found |= {m.name for m in pkgutil.iter_modules([std])}
            except Exception:  # pragma: no cover
                pass
            _STDLIB_CACHE = found
    return _STDLIB_CACHE


def classify_import(module: str, level: int, here: Optional[Path]) -> str:
    if module == "__future__":
        return "future"
    if level > 0:
        return "local"
    top = module.split(".")[0]
    if top in stdlib_modules():
        return "stdlib"
    if here is not None:
        if (here / f"{top}.py").exists() or (here / top / "__init__.py").exists():
            return "local"
    return "third"


# --------------------------------------------------------------------------- #
#  the builder
# --------------------------------------------------------------------------- #


class _Builder:
    def __init__(self, src: Source):
        self.src = src
        self.defs: List[DefInfo] = []
        self.by_node: Dict[int, DefInfo] = {}
        self.used_anchors: Dict[str, int] = {}

    # -- anchors -------------------------------------------------------- #

    def anchor(self, qualname: str) -> str:
        base = "d-" + _slug(qualname)
        n = self.used_anchors.get(base, 0)
        self.used_anchors[base] = n + 1
        return base if n == 0 else f"{base}-{n + 1}"

    # -- walk ----------------------------------------------------------- #

    def visit_body(self, body: Sequence[ast.stmt], parent: Optional[DefInfo], depth: int) -> List[DefInfo]:
        out: List[DefInfo] = []
        for stmt in body:
            out.extend(self.visit_stmt(stmt, parent, depth))
        return out

    def visit_stmt(self, stmt: ast.stmt, parent: Optional[DefInfo], depth: int) -> List[DefInfo]:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            info = self.make_def(stmt, parent, depth)
            if parent is not None:
                parent.children.append(info)
            self.visit_body(stmt.body, info, depth + 1)  # appends the nested definitions to ``info``
            return [info]
        out: List[DefInfo] = []
        # definitions nested in compound statements (if/try/with/for...)
        for attr in ("body", "orelse", "finalbody"):
            sub = getattr(stmt, attr, None)
            if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                out.extend(self.visit_body(sub, parent, depth))
        for h in getattr(stmt, "handlers", []) or []:
            out.extend(self.visit_body(h.body, parent, depth))
        for c in getattr(stmt, "cases", []) or []:
            out.extend(self.visit_body(c.body, parent, depth))
        return out

    # -- definitions ---------------------------------------------------- #

    def make_def(self, node: ast.AST, parent: Optional[DefInfo], depth: int) -> DefInfo:
        src = self.src
        is_class = isinstance(node, ast.ClassDef)
        name = node.name  # type: ignore[attr-defined]
        qual = f"{parent.qualname}.{name}" if parent else name
        raw, doc_node = docstring_of(node)
        doc = parse_doc(raw)
        first = src.first_line(node)
        last = node.end_lineno  # type: ignore[attr-defined]
        loc, _ = src.line_span(first, last)
        cc = analysis.complexity(node)
        info = DefInfo(
            node=node,
            kind="class" if is_class else "function",
            name=name,
            qualname=qual,
            anchor=self.anchor(qual),
            parent=parent,
            depth=depth,
            doc=doc,
            first_line=first,
            last_line=last,
            loc=loc,
            cc=cc,
            grade=analysis.grade(cc),
            nesting=analysis.max_depth(node.body),  # type: ignore[attr-defined]
            uses=analysis.Uses(),
            has_docstring=bool(raw and raw.strip()),
            docstring_node=doc_node,
        )
        info.decorators = [compact(src, d) for d in node.decorator_list]  # type: ignore[attr-defined]
        self.defs.append(info)
        self.by_node[id(node)] = info
        if doc_node is not None:
            src.exempt(src.start(doc_node), src.end(doc_node))
        if is_class:
            self.fill_class(info)
        else:
            self.fill_function(info)
        return info

    # -- functions ------------------------------------------------------ #

    def fill_function(self, info: DefInfo) -> None:
        node = info.node
        assert isinstance(node, FUNC_NODES)
        info.is_async = isinstance(node, ast.AsyncFunctionDef)
        info.is_generator = analysis.is_generator(node)
        decos = {self._deco_base(d) for d in node.decorator_list}
        in_class = info.parent is not None and info.parent.is_class
        if in_class:
            if "staticmethod" in decos:
                info.role = "staticmethod"
            elif "classmethod" in decos or info.name in ("__init_subclass__", "__class_getitem__"):
                info.role = "classmethod"
            elif decos & {"property", "cached_property", "cache", "abstractproperty"}:
                info.role = "property"
            elif any(d.endswith((".setter", ".getter", ".deleter")) for d in decos):
                info.role = "property"
            else:
                info.role = "method"
        else:
            info.role = "function"
        info.uses = analysis.collect_uses(node, info.self_name)
        info.params = self.build_params(node, info)
        info.returns = compact(self.src, node.returns) if node.returns is not None else None
        if info.doc.returns and not info.returns and info.doc.returns[0].type:
            pass  # shown from the docstring by the renderer
        info.recursive = info.name in info.uses.calls or (
            info.self_name is not None and f"{info.self_name}.{info.name}" in info.uses.calls
        )
        info.badges = self._function_badges(info, decos)

    @staticmethod
    def _deco_base(d: ast.expr) -> str:
        if isinstance(d, ast.Call):
            d = d.func
        return analysis.dotted_name(d) or ""

    def _function_badges(self, info: DefInfo, decos: Set[str]) -> List[str]:
        b: List[str] = []
        if info.is_async:
            b.append("async")
        if info.is_generator:
            b.append("generator")
        if info.role in ("staticmethod", "classmethod", "property"):
            b.append(info.role)
        if any(d.endswith(".setter") for d in decos):
            b.append("setter")
        if any(d.split(".")[-1] in ("abstractmethod", "abstractproperty") for d in decos):
            b.append("abstract")
        if any(d.split(".")[-1] == "overload" for d in decos):
            b.append("overload")
        if any(d.split(".")[-1] in ("contextmanager", "asynccontextmanager") for d in decos):
            b.append("context manager")
        if any(d.split(".")[-1] in ("lru_cache", "cache", "cached_property") for d in decos):
            b.append("cached")
        if any(d.split(".")[-1] == "deprecated" for d in decos) or re.search(r"\bdeprecated\b", info.doc.raw[:200], re.I):
            b.append("deprecated")
        name = info.name
        if name.startswith("__") and name.endswith("__"):
            b.append("dunder")
        elif name.startswith("_"):
            b.append("private")
        if info.recursive:
            b.append("recursive")
        return b

    def build_params(self, node: ast.AST, info: DefInfo) -> List[ParamInfo]:
        src = self.src
        a = node.args  # type: ignore[attr-defined]
        doc = info.doc
        out: List[ParamInfo] = []
        pos = list(a.posonlyargs) + list(a.args)
        defaults: List[Optional[ast.expr]] = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
        n_posonly = len(a.posonlyargs)

        def make(arg: ast.arg, kind: str, default: Optional[ast.expr]) -> ParamInfo:
            p = ParamInfo(name=arg.arg, kind=kind, line=arg.lineno)
            if arg.annotation is not None:
                p.annotation = compact(src, arg.annotation)
            if default is not None:
                p.default = compact(src, default)
                p.inferred = analysis.infer_type(default) or ""
            f = doc.param(arg.arg)
            if f is not None:
                p.documented = True
                p.doc_type = f.type
                p.desc = f.desc
                p.optional_doc = f.optional
                if p.default is None and f.default:
                    p.doc_default = f.default
            return p

        for i, arg in enumerate(pos):
            out.append(make(arg, "posonly" if i < n_posonly else "normal", defaults[i]))
        if a.vararg:
            out.append(make(a.vararg, "vararg", None))
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            out.append(make(arg, "kwonly", d))
        if a.kwarg:
            out.append(make(a.kwarg, "kwarg", None))

        # inline comments in the signature serve as a fallback description
        self._signature_comments(node, out)
        # self / cls are implied by the role; keep them but flag them
        if info.is_method and info.role != "staticmethod" and out and out[0].kind in ("posonly", "normal"):
            out[0].kind = out[0].kind  # retained for completeness; renderer decides visibility
        info.total_params = sum(1 for p in out if not self._is_implicit(info, p))
        info.typed_params = sum(1 for p in out if p.annotation and not self._is_implicit(info, p))
        return out

    @staticmethod
    def _is_implicit(info: DefInfo, p: ParamInfo) -> bool:
        return (
            info.is_method
            and info.role != "staticmethod"
            and p.name == info.self_name
            and p.kind in ("posonly", "normal")
        )

    def _signature_comments(self, node: ast.AST, params: List[ParamInfo]) -> None:
        src = self.src
        arg_nodes = {}
        a = node.args  # type: ignore[attr-defined]
        for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs) + [a.vararg, a.kwarg]:
            if arg is not None:
                arg_nodes[arg.arg] = arg
        by_name = {p.name: p for p in params}
        header_end = node.body[0].lineno  # type: ignore[attr-defined]
        for name, arg in arg_nodes.items():
            p = by_name[name]
            if p.desc:
                continue
            end_row = arg.end_lineno or arg.lineno
            # extend through a default value that continues on the same logical line
            dflt = self._default_node(node, name)
            if dflt is not None:
                end_row = max(end_row, dflt.end_lineno)
            c = src.trailing_comment(end_row)
            if c is not None and not c.used and c.row < header_end:
                p.comment = c.body
                c.used = True
            else:
                above = src.comment_above(arg.lineno)
                # only a comment on the line directly above, inside the signature
                if above is not None and above.row > node.lineno and not above.used:
                    p.comment = above.body
                    above.used = True

    @staticmethod
    def _default_node(node: ast.AST, name: str) -> Optional[ast.expr]:
        a = node.args  # type: ignore[attr-defined]
        pos = list(a.posonlyargs) + list(a.args)
        defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
        for arg, d in zip(pos, defaults):
            if arg.arg == name:
                return d
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            if arg.arg == name:
                return d
        return None

    # -- classes -------------------------------------------------------- #

    def fill_class(self, info: DefInfo) -> None:
        node = info.node
        assert isinstance(node, ast.ClassDef)
        src = self.src
        info.role = "class"
        info.bases = [compact(src, b) for b in node.bases]
        info.class_keywords = [f"{k.arg}={compact(src, k.value)}" if k.arg else f"**{compact(src, k.value)}" for k in node.keywords]
        info.uses = analysis.Uses()
        info.badges = self._class_badges(info)

    def _class_badges(self, info: DefInfo) -> List[str]:
        node = info.node
        bases = [b.split(".")[-1].split("[")[0] for b in info.bases]
        decos = [d.split("(")[0].split(".")[-1] for d in info.decorators]
        badges: List[str] = []
        if "dataclass" in decos:
            badges.append("dataclass")
        if any(b in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "ReprEnum") for b in bases):
            badges.append("enum")
        if "Protocol" in bases:
            badges.append("protocol")
        if any(b in ("ABC",) for b in bases) or any("ABCMeta" in k for k in info.class_keywords):
            badges.append("abstract")
        if "NamedTuple" in bases:
            badges.append("namedtuple")
        if "TypedDict" in bases:
            badges.append("typeddict")
        if any(b in ("Exception", "BaseException") or b.endswith(("Error", "Exception", "Warning")) for b in bases):
            badges.append("exception")
        if any(b.endswith("Model") or b == "BaseModel" for b in bases) and "dataclass" not in decos:
            badges.append("model")
        if info.name.startswith("_") and not info.name.startswith("__"):
            badges.append("private")
        if any(isinstance(n, ast.ClassDef) for n in node.body):  # type: ignore[attr-defined]
            badges.append("nested classes")
        return badges

    def finish_class(self, info: DefInfo) -> None:
        """Extract attributes once all methods have been analysed."""
        node = info.node
        src = self.src
        is_enum = "enum" in info.badges
        is_field_class = bool(set(info.badges) & {"dataclass", "namedtuple", "typeddict", "model"}) or "Protocol" in info.bases
        attrs: Dict[str, AttrInfo] = {}

        def comment_for(row: int) -> str:
            c = src.trailing_comment(row) or src.comment_above(row)
            return c.body if c else ""

        for stmt in node.body:  # type: ignore[attr-defined]
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                a = AttrInfo(
                    name=name,
                    type=compact(src, stmt.annotation),
                    type_origin="annotation",
                    default=compact(src, stmt.value) if stmt.value is not None else None,
                    scope="field" if is_field_class else "class",
                    line=stmt.lineno,
                )
                a.desc = comment_for(stmt.lineno)
                attrs[name] = a
            elif isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name) and not (t.id.startswith("__") and t.id.endswith("__")):
                        a = AttrInfo(
                            name=t.id,
                            type=analysis.infer_type(stmt.value) or "",
                            type_origin="inferred",
                            default=compact(src, stmt.value),
                            scope="member" if is_enum else "class",
                            line=stmt.lineno,
                        )
                        a.desc = comment_for(stmt.lineno)
                        attrs[t.id] = a
        # instance attributes assigned through ``self.x = ...`` inside methods
        methods = [c for c in info.children if not c.is_class and c.self_name]
        methods.sort(key=lambda m: (m.name != "__init__", m.first_line))
        for m in methods:
            self_name = m.self_name
            ptypes = {p.name: p for p in m.params}
            for n in analysis.walk_scope(m.node):
                target = None
                value = None
                ann = None
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        self._instance_attr(t, self_name, n.value, None, m, ptypes, attrs, n.lineno)
                    continue
                if isinstance(n, ast.AnnAssign):
                    target, value, ann = n.target, n.value, n.annotation
                elif isinstance(n, ast.AugAssign):
                    continue
                if target is not None:
                    self._instance_attr(target, self_name, value, ann, m, ptypes, attrs, n.lineno)
        # merge docstring attribute documentation
        for f in info.doc.attributes:
            nm = f.name.lstrip("*")
            if nm in attrs:
                a = attrs[nm]
                if f.desc:
                    a.desc = f.desc
                if f.type and not a.type:
                    a.type, a.type_origin = f.type, "doc"
            elif nm:
                attrs[nm] = AttrInfo(nm, f.type, "doc", None, f.desc, "documented", info.first_line)
        info.attrs = sorted(attrs.values(), key=lambda a: (a.line if a.scope != "documented" else 10**9, a.name))

    def _instance_attr(
        self,
        target: ast.AST,
        self_name: Optional[str],
        value: Optional[ast.expr],
        ann: Optional[ast.expr],
        method: DefInfo,
        ptypes: Dict[str, ParamInfo],
        attrs: Dict[str, AttrInfo],
        line: int,
    ) -> None:
        src = self.src
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == self_name):
            if isinstance(target, (ast.Tuple, ast.List)):
                for t in target.elts:
                    self._instance_attr(t, self_name, None, None, method, ptypes, attrs, line)
            return
        name = target.attr
        if name in attrs and attrs[name].scope in ("class", "field", "member") and not attrs[name].set_in:
            attrs[name].set_in = method.name
            return
        if name in attrs:
            return
        typ, origin = "", ""
        if ann is not None:
            typ, origin = compact(src, ann), "annotation"
        elif isinstance(value, ast.Name) and value.id in ptypes:
            p = ptypes[value.id]
            if p.annotation:
                typ, origin = p.annotation, "annotation"
            elif p.doc_type:
                typ, origin = p.doc_type, "doc"
            elif p.inferred and p.inferred != "None":
                typ, origin = p.inferred, "inferred"
        elif value is not None:
            t = analysis.infer_type(value)
            if t:
                typ, origin = t, "inferred"
        desc = ""
        if isinstance(value, ast.Name) and value.id in ptypes:
            desc = ptypes[value.id].desc or ptypes[value.id].comment
        if isinstance(value, ast.Name) and value.id in ptypes:
            default = ptypes[value.id].default  # the parameter's own default, if any
        elif value is not None and len(compact(src, value)) <= 60:
            default = compact(src, value)
        else:
            default = None
        a = AttrInfo(
            name=name,
            type=typ,
            type_origin=origin,
            default=default,
            desc=desc,
            scope="instance",
            line=line,
            set_in=method.name,
        )
        attrs[name] = a


# --------------------------------------------------------------------------- #
#  call graph
# --------------------------------------------------------------------------- #


def _resolve_calls(defs: List[DefInfo]) -> None:
    top_funcs: Dict[str, DefInfo] = {}
    top_classes: Dict[str, DefInfo] = {}
    for d in defs:
        if d.parent is None:
            (top_classes if d.is_class else top_funcs).setdefault(d.name, d)

    def class_methods(cls: DefInfo, seen: Optional[Set[int]] = None) -> Dict[str, DefInfo]:
        """Methods of *cls* including those of locally defined bases."""
        seen = seen or set()
        if id(cls) in seen:
            return {}
        seen.add(id(cls))
        out: Dict[str, DefInfo] = {}
        for b in reversed(cls.bases):
            base = top_classes.get(b.split(".")[-1].split("[")[0])
            if base is not None:
                out.update(class_methods(base, seen))
        for c in cls.children:
            if not c.is_class:
                out[c.name] = c
        return out

    def enclosing_class(d: DefInfo) -> Optional[DefInfo]:
        p = d.parent
        while p is not None and not p.is_class:
            p = p.parent
        return p

    for d in defs:
        if d.is_class:
            continue
        cls = enclosing_class(d)
        seen_targets: Dict[int, DefInfo] = {}
        external: Dict[str, int] = {}
        for name, count in d.uses.calls.items():
            target: Optional[DefInfo] = None
            parts = name.split(".")
            if len(parts) == 1:
                target = top_funcs.get(name)
                if target is None and name in top_classes:
                    c = top_classes[name]
                    target = class_methods(c).get("__init__") or c
            elif cls is not None and parts[0] in ("self", "cls") and len(parts) == 2:
                target = class_methods(cls).get(parts[1])
            elif parts[0] == "super()" and cls is not None and len(parts) == 2:
                for b in cls.bases:
                    base = top_classes.get(b.split(".")[-1].split("[")[0])
                    if base is not None:
                        target = class_methods(base).get(parts[1])
                        if target is not None:
                            break
            elif len(parts) == 2 and parts[0] in top_classes:
                target = class_methods(top_classes[parts[0]]).get(parts[1])
            if target is not None:
                if target is d:
                    continue
                seen_targets[id(target)] = target
            else:
                external[name] = count
        d.callees = list(seen_targets.values())
        d.external_calls = external
        for t in d.callees:
            if d not in t.callers:
                t.callers.append(d)


# --------------------------------------------------------------------------- #
#  module level
# --------------------------------------------------------------------------- #


def build_module(src: Source) -> ModuleInfo:
    tree = src.tree
    raw, doc_node = docstring_of(tree)
    doc = parse_doc(raw)
    if doc_node is not None:
        src.exempt(src.start(doc_node), src.end(doc_node))

    b = _Builder(src)
    top = b.visit_body(tree.body, None, 0)
    # attributes need all methods to be analysed first
    for d in b.defs:
        if d.is_class:
            b.finish_class(d)
    _resolve_calls(b.defs)

    info = ModuleInfo(
        src=src,
        doc=doc,
        docstring_node=doc_node,
        defs=b.defs,
        top=top,
        by_node=b.by_node,
        imports=[],
        consts=[],
        todos=[],
    )
    _module_header(info)
    _imports(info)
    _constants(info)
    _todos(info)
    _features(info)
    return info


def _module_header(info: ModuleInfo) -> None:
    src = info.src
    for row, line in enumerate(src.lines[:3], 1):
        if line.startswith("#!"):
            info.shebang = line.strip()
            c = src.take_inline(row, 0) or next((c for c in src.comments if c.row == row), None)
            if c:
                c.used = True
        elif re.match(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)", line):
            m = re.match(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)", line)
            info.encoding = m.group(1) if m else ""
            c = next((c for c in src.comments if c.row == row), None)
            if c:
                c.used = True


def _imports(info: ModuleInfo) -> None:
    src = info.src
    here = Path(src.filename).resolve().parent if os.path.exists(src.filename) else None
    for stmt in info.src.tree.body:
        if isinstance(stmt, ast.Import):
            for a in stmt.names:
                info.imports.append(
                    ImportInfo(a.name, [], a.asname or "", 0, classify_import(a.name, 0, here), stmt.lineno, False)
                )
        elif isinstance(stmt, ast.ImportFrom):
            mod = stmt.module or ""
            info.imports.append(
                ImportInfo(
                    mod,
                    [(a.name, a.asname or "") for a in stmt.names],
                    "",
                    stmt.level,
                    classify_import(mod, stmt.level, here),
                    stmt.lineno,
                    True,
                )
            )


def _constants(info: ModuleInfo) -> None:
    src = info.src

    def comment_for(row: int) -> str:
        c = src.trailing_comment(row) or src.comment_above(row)
        return c.body if c else ""

    for stmt in src.tree.body:
        name = typ = ""
        value: Optional[ast.expr] = None
        inferred = False
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, value = stmt.targets[0].id, stmt.value
            typ = analysis.infer_type(value) or ""
            inferred = True
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name, value = stmt.target.id, stmt.value
            typ = compact(src, stmt.annotation)
        else:
            continue
        if name == "__all__" and value is not None:
            try:
                info.all_names = [e.value for e in value.elts if isinstance(e, ast.Constant)]  # type: ignore[attr-defined]
            except AttributeError:
                pass
            continue
        if name in ("__version__", "__author__", "__license__", "__copyright__", "__email__", "__maintainer__") and isinstance(
            value, ast.Constant
        ):
            info.meta[name.strip("_")] = str(value.value)
            continue
        if name.startswith("__") and name.endswith("__"):
            continue
        val = compact(src, value) if value is not None else ""
        info.consts.append(ConstInfo(name, typ, val, stmt.lineno, comment_for(stmt.lineno), inferred))


def _todos(info: ModuleInfo) -> None:
    for c in info.src.comments:
        m = _TODO_RE.search(c.text)
        if m and (m.group(1) != "NOTE" or c.text.lstrip("# ").startswith("NOTE")):
            info.todos.append(Todo(m.group(1), m.group(2).strip(), c.row))


def _features(info: ModuleInfo) -> None:
    """Note interesting language features used by the module."""
    tree = info.src.tree
    feats: List[str] = []
    kinds = {type(n) for n in ast.walk(tree)}
    if any(getattr(ast, n, None) in kinds for n in ("Match",)):
        feats.append("match")
    if ast.AsyncFunctionDef in kinds:
        feats.append("async")
    if ast.NamedExpr in kinds:
        feats.append("walrus")
    if ast.JoinedStr in kinds:
        feats.append("f-strings")
    if ast.Yield in kinds or ast.YieldFrom in kinds:
        feats.append("generators")
    if ast.Lambda in kinds:
        feats.append("lambdas")
    if ast.With in kinds:
        feats.append("context managers")
    info.features = feats
