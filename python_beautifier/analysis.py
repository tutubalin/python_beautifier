"""Static analysis helpers: complexity, control-flow facts and usage summaries.

Everything here works on the :mod:`ast` only; nothing is ever executed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Set

FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_MATCH = getattr(ast, "Match", None)
_TRY_TYPES = tuple(t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None)
_MATCH_CASE = getattr(ast, "match_case", None)

# --------------------------------------------------------------------------- #
#  scope-aware walking
# --------------------------------------------------------------------------- #


def walk_scope(node: ast.AST, *, nested: bool = False) -> Iterator[ast.AST]:
    """Walk *node*'s subtree.

    Nested function/class definitions are not entered (unless *nested*), so a
    function's metrics are not polluted by its inner functions.
    """
    stack = list(_body_children(node))
    while stack:
        n = stack.pop()
        yield n
        if not nested and isinstance(n, DEF_NODES):
            # still look at decorators/defaults, which run in the enclosing scope
            stack.extend(getattr(n, "decorator_list", []))
            if isinstance(n, FUNC_NODES):
                stack.extend(n.args.defaults)
                stack.extend(d for d in n.args.kw_defaults if d is not None)
            else:
                stack.extend(n.bases)
            continue
        stack.extend(ast.iter_child_nodes(n))


def _body_children(node: ast.AST) -> Iterator[ast.AST]:
    if isinstance(node, DEF_NODES):
        yield from node.body
        return
    yield from ast.iter_child_nodes(node)


# --------------------------------------------------------------------------- #
#  complexity
# --------------------------------------------------------------------------- #


def complexity(node: ast.AST) -> int:
    """McCabe cyclomatic complexity of a function (or class body)."""
    c = 1
    for n in walk_scope(node):
        if isinstance(n, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.Assert)):
            c += 1
        elif isinstance(n, ast.BoolOp):
            c += len(n.values) - 1
        elif isinstance(n, ast.comprehension):
            c += 1 + len(n.ifs)
        elif _MATCH_CASE is not None and isinstance(n, _MATCH_CASE):
            pat = n.pattern
            wildcard = type(pat).__name__ == "MatchAs" and getattr(pat, "pattern", None) is None and n.guard is None
            if not wildcard:
                c += 1
    return c


def grade(cc: int) -> str:
    """Letter grade for a complexity score (same bands as radon)."""
    if cc <= 5:
        return "A"
    if cc <= 10:
        return "B"
    if cc <= 20:
        return "C"
    if cc <= 30:
        return "D"
    if cc <= 40:
        return "E"
    return "F"


def max_depth(stmts: Sequence[ast.stmt]) -> int:
    """Deepest nesting of control structures inside *stmts*."""

    def depth(body: Sequence[ast.stmt]) -> int:
        best = 0
        for s in body:
            best = max(best, stmt_depth(s))
        return best

    def stmt_depth(s: ast.stmt) -> int:
        if isinstance(s, ast.If):
            # an ``elif`` is an If alone in orelse: same level as its parent
            d = depth(s.body)
            orelse = s.orelse
            while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                d = max(d, depth(orelse[0].body))
                orelse = orelse[0].orelse
            d = max(d, depth(orelse))
            return 1 + d
        if isinstance(s, (ast.For, ast.AsyncFor, ast.While)):
            return 1 + max(depth(s.body), depth(s.orelse))
        if isinstance(s, (ast.With, ast.AsyncWith)):
            return 1 + depth(s.body)
        if _TRY_TYPES and isinstance(s, _TRY_TYPES):
            inner = [depth(s.body), depth(s.orelse), depth(s.finalbody)]
            inner.extend(depth(h.body) for h in s.handlers)
            return 1 + max(inner)
        if _MATCH is not None and isinstance(s, _MATCH):
            return 1 + max((depth(c.body) for c in s.cases), default=0)
        return 0

    return depth(stmts)


# --------------------------------------------------------------------------- #
#  control-flow facts
# --------------------------------------------------------------------------- #

_EXIT_CALLS = {"sys.exit", "exit", "quit", "os._exit", "os.abort", "abort"}


def dotted_name(node: ast.AST) -> Optional[str]:
    """``a.b.c`` for a Name/Attribute chain, else ``None``."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):  # e.g. super().method
        inner = dotted_name(node.func)
        if inner:
            parts.append(inner + "()")
            return ".".join(reversed(parts))
    return None


def ends_flow(stmts: Sequence[ast.stmt]) -> Optional[str]:
    """If control can never fall off the end of *stmts*, say how it leaves.

    Returns ``'return'``, ``'raise'``, ``'break'``, ``'continue'``, ``'exit'`` or
    ``None`` when execution may continue past the block.
    """
    for s in stmts:
        if isinstance(s, ast.Return):
            return "return"
        if isinstance(s, ast.Raise):
            return "raise"
        if isinstance(s, ast.Break):
            return "break"
        if isinstance(s, ast.Continue):
            return "continue"
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            if dotted_name(s.value.func) in _EXIT_CALLS:
                return "exit"
        if isinstance(s, ast.If) and s.orelse:
            a, b = ends_flow(s.body), ends_flow(s.orelse)
            if a and b:
                return a if a == b else "return"
        if isinstance(s, (ast.With, ast.AsyncWith)):
            r = ends_flow(s.body)
            if r:
                return r
        if _TRY_TYPES and isinstance(s, _TRY_TYPES):
            if s.finalbody and ends_flow(s.finalbody):
                return ends_flow(s.finalbody)
            paths = [ends_flow(s.body + s.orelse)] + [ends_flow(h.body) for h in s.handlers]
            if all(paths):
                return paths[0]
    return None


def is_generator(func: ast.AST) -> bool:
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in walk_scope(func))


def has_await(func: ast.AST) -> bool:
    return any(isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith)) for n in walk_scope(func))


# --------------------------------------------------------------------------- #
#  usage summary
# --------------------------------------------------------------------------- #


@dataclass
class Uses:
    """What a function touches: attributes, calls, exceptions."""

    reads: Dict[str, int] = field(default_factory=dict)
    writes: Dict[str, int] = field(default_factory=dict)
    calls: Dict[str, int] = field(default_factory=dict)
    raises: Dict[str, int] = field(default_factory=dict)
    names: Set[str] = field(default_factory=set)
    returns: int = 0
    yields: int = 0
    awaits: int = 0
    reraise: bool = False


def _bump(d: Dict[str, int], k: str) -> None:
    d[k] = d.get(k, 0) + 1


def collect_uses(func: ast.AST, self_name: Optional[str]) -> Uses:
    """Summarise attribute access, calls and raises inside *func*.

    Nested functions are included: a closure's behaviour is part of its
    parent's behaviour.
    """
    u = Uses()
    call_funcs: Set[int] = set()
    body = func.body if isinstance(func, DEF_NODES) else [func]
    nodes: List[ast.AST] = []
    for b in body:
        nodes.extend(ast.walk(b))
    for n in nodes:
        if isinstance(n, ast.Call):
            call_funcs.add(id(n.func))
    for n in nodes:
        if isinstance(n, ast.Call):
            name = dotted_name(n.func)
            if name:
                _bump(u.calls, name)
        elif isinstance(n, ast.Attribute) and self_name and isinstance(n.value, ast.Name) and n.value.id == self_name:
            if id(n) in call_funcs:
                continue
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                _bump(u.writes, n.attr)
            else:
                _bump(u.reads, n.attr)
        elif isinstance(n, ast.AugAssign):
            t = n.target
            if self_name and isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == self_name:
                _bump(u.reads, t.attr)
        elif isinstance(n, ast.Raise):
            if n.exc is None:
                u.reraise = True
            else:
                exc = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
                name = dotted_name(exc)
                if name:
                    _bump(u.raises, name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            u.names.add(n.id)
        elif isinstance(n, ast.Return):
            u.returns += 1
        elif isinstance(n, (ast.Yield, ast.YieldFrom)):
            u.yields += 1
        elif isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
            u.awaits += 1
    return u


# --------------------------------------------------------------------------- #
#  misc
# --------------------------------------------------------------------------- #


def infer_type(node: Optional[ast.AST]) -> Optional[str]:
    """Best-effort type name for a default value / assigned expression."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        v = node.value
        if v is None:
            return "None"
        if v is Ellipsis:
            return "ellipsis"
        return type(v).__name__
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.ListComp):
        return "list"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.DictComp):
        return "dict"
    if isinstance(node, (ast.Set, ast.SetComp)):
        return "set"
    if isinstance(node, ast.GeneratorExp):
        return "Generator"
    if isinstance(node, ast.Lambda):
        return "Callable"
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        return infer_type(node.operand)
    if isinstance(node, ast.Call):
        name = dotted_name(node.func)
        if name:
            tail = name.split(".")[-1]
            if tail in {"dict", "list", "set", "tuple", "frozenset", "str", "int", "float", "bool", "bytes", "bytearray"}:
                return tail
            if tail in {"defaultdict", "OrderedDict", "Counter", "deque", "Path", "Lock", "RLock", "Event"}:
                return tail
            if tail[:1].isupper():
                return tail
    if isinstance(node, ast.Compare):
        return "bool"
    if isinstance(node, ast.BoolOp):
        return None
    return None
