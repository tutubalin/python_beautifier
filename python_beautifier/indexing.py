"""Taking NumPy / PyTorch style subscripts apart.

``img[:, num_txt_tokens:, ...]`` packs several different ideas into one pair
of brackets: whole-axis colons, slices, an ellipsis, ``None`` for a new axis,
plain picks, masks.  This module decodes such a subscript axis by axis and
says in words what every part does.

It is pure analysis.  ``render.index`` turns the result into HTML, and
:class:`~python_beautifier.source.Source` uses the positions to mark the terms
inside the highlighted code.

What is worth explaining: a subscript with several comma separated terms of
which at least one is a slice or ``np.newaxis`` (``x[:, 0]``, ``x[0, np.newaxis]``).
Ordinary Python indexing and slicing (``xs[0]``, ``s[1:]``, ``d["key"]``,
``grid[r, c]``) and type subscripts (``Dict[str, int]``, ``tuple[int, ...]``) are
left alone.

``...`` and ``None`` on their own are ambiguous: ``x[..., 0]`` and ``x[None]``
index arrays, but ``handlers[None]`` is a dictionary lookup and pyparsing's
``expr[1, ...]`` means "one or more".  They therefore only count in files that
import an array library (NumPy, PyTorch, JAX, pandas ...).
"""
from __future__ import annotations

import ast
import re
from bisect import bisect_left
from dataclasses import dataclass, field
from html import escape
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

Pos = Tuple[int, int]
Seg = Tuple[str, bool]  # (text, is_code): a piece of a sentence

MINUS = "\u2212"

# Subscripts of these are type expressions, never array indexing.
_TYPE_NAMES = frozenset(
    """Any Optional Union Callable Type Tuple List Dict Set FrozenSet Sequence MutableSequence Mapping
    MutableMapping Iterable Iterator Generator AsyncIterator AsyncIterable AsyncGenerator Awaitable Coroutine
    Collection Container Reversible Counter Deque DefaultDict OrderedDict ChainMap Literal Annotated ClassVar
    Final Generic Protocol TypeGuard TypeAlias Concatenate Unpack Required NotRequired Pattern Match
    list dict tuple set frozenset type""".split()
)

KINDS = ("all", "slice", "rest", "new", "pick", "mask", "take", "others")

# A file that imports one of these may index arrays with a bare ``...`` or ``None``.
_ARRAY_LIBRARIES = frozenset(
    """numpy torch jax tensorflow keras scipy pandas cupy einops mxnet paddle xarray dask sklearn skimage
    flax mlx numba cv2""".split()
)


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #


def plain(segs: Sequence[Seg]) -> str:
    """A sentence as text, code pieces in backticks."""
    return "".join(f"`{t}`" if is_code else t for t, is_code in segs)


@dataclass
class Axis:
    """One comma separated term of a subscript (or the implicit tail)."""

    kind: str  # one of KINDS
    label: str  # "axis 0", "axis \u22121", "rest", "new axis", "rows", "others"
    term: str  # the term as written ("" for the implicit tail)
    meaning: List[Seg]
    note: str = ""  # a longer explanation, for tooltips
    start: Optional[Pos] = None  # source range of the term (None: nothing to mark)
    end: Optional[Pos] = None
    cell: int = 0  # the strip cell this term is shown in (runs of trivial terms share one)

    @property
    def title(self) -> str:
        text = f"{self.label} \u00b7 {plain(self.meaning)}"
        return f"{text} \u2014 {self.note}" if self.note else text


@dataclass
class Guide:
    """Everything known about one decoded subscript."""

    base: str  # the indexed expression, e.g. ``img``
    mode: str  # read | write | delete
    axes: List[Axis]  # one per term, with its place in the source
    cells: List[Axis]  # what the strip shows: runs of trivial terms merged, plus the implicit tail
    key: str  # the whole subscript as written, whitespace collapsed
    start: Pos
    ids: List[str] = field(default_factory=list)  # one DOM id per cell
    shown: bool = True  # False for repeats within one statement


class Wrap(NamedTuple):
    """A term to mark inside highlighted code."""

    start: Pos
    end: Pos
    kind: str
    ident: str
    title: str

    @property
    def open_tag(self) -> str:
        return f'<span class="ax ik-{self.kind}" data-ix="{self.ident}" title="{escape(self.title, quote=True)}">'


class Index:
    """All decoded subscripts of a module."""

    def __init__(self) -> None:
        self.guides: List[Guide] = []
        self.wraps: List[Wrap] = []
        self._starts: List[Pos] = []
        self._by_stmt: Dict[int, List[Guide]] = {}

    def __bool__(self) -> bool:
        return bool(self.guides)

    def for_statement(self, stmt: ast.AST) -> List[Guide]:
        """The guides to show for *stmt* (its own expressions, repeats removed)."""
        return self._by_stmt.get(id(stmt), [])

    def wraps_in(self, start: Pos, end: Pos) -> List[Wrap]:
        """Terms lying completely inside the source range, outermost first."""
        i = bisect_left(self._starts, start)
        out = []
        while i < len(self.wraps) and self._starts[i] <= end:
            if self.wraps[i].end <= end:
                out.append(self.wraps[i])
            i += 1
        return out


# --------------------------------------------------------------------------- #
#  recognising the parts
# --------------------------------------------------------------------------- #


def _int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _int(node.operand)
        if inner is not None:
            return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _is_ellipsis(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and node.value is Ellipsis) or (
        isinstance(node, ast.Name) and node.id == "Ellipsis"
    )


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_explicit_newaxis(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "newaxis") or (
        isinstance(node, ast.Name) and node.id == "newaxis"
    )


def _is_newaxis(node: ast.AST) -> bool:
    return _is_none(node) or _is_explicit_newaxis(node)


def _is_slice_object(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "slice"


def _is_mask(node: ast.AST) -> bool:
    """Looks like a boolean mask: a comparison, ``~m``, or ``&`` / ``|`` of those."""
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return _is_mask(node.left) or _is_mask(node.right)
    return False


def _worth_explaining(elts: List[ast.expr], several: bool, arrays: bool) -> bool:
    """Is this subscript array indexing, as far as the syntax alone can tell?

    Slices and an explicit ``np.newaxis`` are unmistakable.  ``...`` and ``None`` also mean
    something else in ordinary code (a dict key, a DSL), so they need an array library in sight.
    """
    if several:
        if any(isinstance(e, ast.Slice) or _is_explicit_newaxis(e) for e in elts):
            return True
        return arrays and any(_is_ellipsis(e) or _is_none(e) for e in elts)
    e = elts[0]
    return _is_explicit_newaxis(e) or (arrays and (_is_ellipsis(e) or _is_none(e)))


def _uses_array_library(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] in _ARRAY_LIBRARIES for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in _ARRAY_LIBRARIES:
                return True
    return False


def _is_next(lo: ast.AST, hi: ast.AST) -> bool:
    """``i : i + 1`` - the idiom for a length-one slice that keeps the axis."""
    return (
        isinstance(hi, ast.BinOp)
        and isinstance(hi.op, ast.Add)
        and isinstance(hi.right, ast.Constant)
        and type(hi.right.value) is int
        and hi.right.value == 1
        and ast.dump(hi.left) == ast.dump(lo)
    )


def _type_like(value: ast.AST) -> bool:
    name = value.id if isinstance(value, ast.Name) else value.attr if isinstance(value, ast.Attribute) else ""
    return name in _TYPE_NAMES


def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th, 11th, 21st ..."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _short(text: str, limit: int = 30) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


# --------------------------------------------------------------------------- #
#  saying what a term does
# --------------------------------------------------------------------------- #

_NOTES = {
    "all": "`:` keeps the whole axis.",
    "slice": "start:stop:step. The stop is not included; a negative number counts from the end.",
    "rest": "`...` stands for as many whole-axis `:` as are needed to fill in the remaining axes.",
    "new": "`None` (np.newaxis) inserts a new axis of length 1 at this position.",
    "mask": "A boolean mask keeps the positions where it is true. A mask with several dimensions uses up "
    "that many axes, which shifts the numbers of the axes after it.",
    "take": "A list or index array gathers those positions (advanced indexing).",
    "pick": "A single integer picks one position and removes the axis; an index array or a boolean mask "
    "selects several positions instead (advanced indexing). A mask with several dimensions uses up "
    "that many axes.",
    "others": "Axes that are not mentioned are kept whole.",
}


def _code(text: str) -> Seg:
    return (_short(text), True)


def _say(*parts: object) -> List[Seg]:
    return [p if isinstance(p, tuple) else (str(p), False) for p in parts]


def _slice_meaning(src, s: ast.Slice, label_based: bool) -> Tuple[str, List[Seg]]:
    lo, hi, st = s.lower, s.upper, s.step

    def txt(node: ast.AST) -> str:
        return " ".join(src.text_of(node).split())

    c_lo = _int(lo) if lo is not None else None
    c_hi = _int(hi) if hi is not None else None
    c_st = _int(st) if st is not None else None

    if lo is None and hi is None and st is None:
        return "all", _say("all")

    if label_based:  # pandas .loc: labels, and the end is included
        if st is None:
            if hi is None:
                return "slice", _say("from ", _code(txt(lo)), " onward")
            if lo is None:
                return "slice", _say("up to and including ", _code(txt(hi)))
            return "slice", _say("from ", _code(txt(lo)), " through ", _code(txt(hi)), " (both included)")
        return "slice", _say("start ", _code(txt(lo)) if lo is not None else "\u2013", ", stop ", _code(txt(hi)) if hi is not None else "\u2013", ", step ", _code(txt(st)))

    # the step: a plain forward step, or (without bounds) a reversal or an unknown step
    step: Optional[List[Seg]] = None
    if st is not None and c_st != 1:
        backwards = c_st is None or c_st < 0  # start and stop swap roles: only the literal form is safe
        if backwards and (lo is not None or hi is not None):
            parts: List[object] = []
            if lo is not None:
                parts += ["start ", _code(txt(lo)), ", "]
            if hi is not None:
                parts += ["stop ", _code(txt(hi)), ", "]
            parts += ["step ", _code(txt(st))]
            return "slice", _say(*parts)
        if c_st == -1:
            step = _say("reversed")
        elif c_st is not None and c_st > 1:
            step = _say(f"every {ordinal(c_st)} item")
        elif c_st is not None and c_st < -1:
            step = _say(f"every {ordinal(-c_st)} item, reversed")
        else:
            step = _say("every ", _code(txt(st)), "-th item")
    if lo is None and hi is None:
        return ("all", _say("all")) if step is None else ("slice", step)

    # the range
    rng: List[Seg]
    if hi is None:
        if c_lo == 0:
            rng = []
        elif c_lo is not None and c_lo > 0:
            rng = _say("skip the first item" if c_lo == 1 else f"skip the first {c_lo}")
        elif c_lo is not None:
            rng = _say("only the last item \u00b7 axis kept" if c_lo == -1 else f"the last {-c_lo}")
        else:
            rng = _say("from ", _code(txt(lo)), " to the end")
    elif lo is None:
        if c_hi == 0:
            rng = _say("nothing (empty)")
        elif c_hi is not None and c_hi > 0:
            rng = _say("only the first item \u00b7 axis kept" if c_hi == 1 else f"the first {c_hi}")
        elif c_hi is not None:
            rng = _say("all but the last item" if c_hi == -1 else f"all but the last {-c_hi}")
        else:
            rng = _say("up to ", _code(txt(hi)), ", not included")
    else:
        if c_lo is not None and c_hi is not None and c_lo >= 0 and c_hi >= 0:
            if c_hi <= c_lo:
                rng = _say("nothing (empty)")
            elif c_lo == 0:
                rng = _say("only the first item \u00b7 axis kept" if c_hi == 1 else f"the first {c_hi}")
            elif c_hi == c_lo + 1:
                rng = _say(f"only item {c_lo} \u00b7 axis kept")
            else:
                rng = _say(f"items {c_lo} to {c_hi - 1}")
        elif c_lo is not None and c_hi is not None and c_lo >= 0 and c_hi < 0:
            tail = "the last item" if c_hi == -1 else f"the last {-c_hi}"
            rng = _say(f"all but {tail}" if c_lo == 0 else f"skip the first {c_lo} and {tail}")
        elif _is_next(lo, hi):
            rng = _say("only item ", _code(txt(lo)), " \u00b7 axis kept")
        else:
            rng = _say("from ", _code(txt(lo)), " up to ", _code(txt(hi)), ", not included")
    if step is not None:
        rng = rng + _say(" \u00b7 ") + step if rng else step
    if not rng:
        return "all", _say("all")
    return "slice", rng


def _pick_meaning(e: ast.AST, label_based: bool, text: str) -> Tuple[str, List[Seg]]:
    n = _int(e)
    if n is not None and not label_based:
        if n >= 0:
            return "pick", _say(f"pick item {n} \u00b7 axis removed")
        if n == -1:
            return "pick", _say("pick the last item \u00b7 axis removed")
        return "pick", _say(f"pick item {n} ({ordinal(-n)} from the end) \u00b7 axis removed")
    if _is_mask(e):
        return "mask", _say("keep where ", _code(text))
    if isinstance(e, ast.List):
        return "take", _say("select labels " if label_based else "gather positions ", _code(text))
    if isinstance(e, ast.Tuple):
        if label_based:  # df.loc[(level_0, level_1), :]
            return "pick", _say("MultiIndex key ", _code(text))
        return "take", _say("gather positions ", _code(text))
    if _is_slice_object(e):
        return "slice", _say("slice object ", _code(text))
    if label_based:
        word = "label " if isinstance(e, ast.Constant) else "label(s) "
        return "pick", _say(word, _code(text))
    return "pick", _say("index with ", _code(text))


def _describe(src, e: ast.AST, label_based: bool, position: str, text: str) -> Tuple[str, List[Seg]]:
    """(kind, meaning) of term *e*; *position* matters for ``...`` (first, middle, last, only)."""
    if isinstance(e, ast.Slice):
        return _slice_meaning(src, e, label_based)
    if _is_ellipsis(e):
        words = {
            "only": "every axis (the whole array)",
            "first": "all leading axes",
            "middle": "all axes in between",
            "last": "all remaining axes",
            "again": "invalid: only one `...` is allowed",
        }[position]
        return "rest", _say(words)
    if _is_newaxis(e):
        return "new", _say("new axis of length 1")
    return _pick_meaning(e, label_based, text)


# --------------------------------------------------------------------------- #
#  building the guide for one subscript
# --------------------------------------------------------------------------- #


def _span_label(first: str, last: str) -> str:
    """"axes 0\u20132" for the labels of a run of axes."""
    a, b = first.removeprefix("axis "), last.removeprefix("axis ")
    return f"axes {a}\u2013{b}" if not a.startswith(MINUS) and not b.startswith(MINUS) else f"axes {a} to {b}"


def _group(axes: List[Axis]) -> List[Axis]:
    """The cells of the strip: runs of ``None`` (2+) and of ``:`` (3+) become one cell each.

    Sets ``Axis.cell`` on every term.
    """
    cells: List[Axis] = []
    i = 0
    while i < len(axes):
        j = i
        if axes[i].kind in ("all", "new"):
            while j + 1 < len(axes) and axes[j + 1].kind == axes[i].kind:
                j += 1
        run = axes[i : j + 1]
        if len(run) >= (2 if axes[i].kind == "new" else 3):
            term = ", ".join(a.term for a in run)
            if axes[i].kind == "new":
                cell = Axis("new", "new axes", term, _say(f"{len(run)} new axes of length 1"), _NOTES["new"])
            else:
                cell = Axis("all", _span_label(run[0].label, run[-1].label), term, _say("all"), _NOTES["all"])
        else:
            run = axes[i : i + 1]
            cell = axes[i]
        for a in run:
            a.cell = len(cells)
        cells.append(cell)
        i += len(run)
    return cells


def _guide(src, sub: ast.Subscript, arrays: bool) -> Optional[Guide]:
    if _type_like(sub.value):
        return None
    idx = sub.slice
    several = isinstance(idx, ast.Tuple)
    elts = list(idx.elts) if isinstance(idx, ast.Tuple) else [idx]
    if (several and len(elts) < 2) or not _worth_explaining(elts, several, arrays):
        return None  # ordinary Python: plain indexing, a single slice, a dictionary lookup

    spans = []
    for e in elts:
        s, t = src.start(e), src.end(e)
        if not src.on_token_boundaries(s, t):
            return None  # e.g. inside an f-string: nothing we could mark reliably
        spans.append((s, t))

    value = sub.value
    accessor = value.func if isinstance(value, ast.Call) else value  # also ``df.loc(axis=0)[...]``
    pandas = isinstance(accessor, ast.Attribute) and accessor.attr in ("loc", "iloc")
    label_based = isinstance(accessor, ast.Attribute) and accessor.attr == "loc"

    ell = [i for i, e in enumerate(elts) if _is_ellipsis(e)]
    first_ell = ell[0] if ell else None
    consuming = [not (_is_newaxis(e) or _is_ellipsis(e)) for e in elts]
    back = sum(1 for i, c in enumerate(consuming) if c and first_ell is not None and i > first_ell)

    axes: List[Axis] = []
    front = 0
    for i, e in enumerate(elts):
        text = re.sub(r"\s*\n\s*", " ", src.slice(*spans[i]))
        if _is_ellipsis(e):
            if i != first_ell:
                position = "again"
            elif len(elts) == 1:
                position = "only"
            elif i == 0:
                position = "first"
            elif i == len(elts) - 1:
                position = "last"
            else:
                position = "middle"
            label = "rest"
        else:
            position = ""
            if _is_newaxis(e):
                label = "new axis"
            elif first_ell is not None and i > first_ell:
                label = f"axis {MINUS}{back}"
                back -= 1
            else:
                label = ("rows", "columns")[front] if pandas and front < 2 else f"axis {front}"
                front += 1
        kind, meaning = _describe(src, e, label_based, position, text)
        axes.append(Axis(kind, label, text, meaning, _NOTES.get(kind, ""), spans[i][0], spans[i][1]))

    cells = _group(axes)
    if first_ell is None and not pandas:
        cells.append(Axis("others", "others", "", _say("further axes stay whole"), _NOTES["others"], cell=len(cells)))

    mode = {ast.Store: "write", ast.Del: "delete"}.get(type(sub.ctx), "read")
    key = " ".join(src.text_of(sub).split())
    base = " ".join(src.text_of(value).split())
    return Guide(base, mode, axes, cells, key, src.start(sub))


# --------------------------------------------------------------------------- #
#  finding them
# --------------------------------------------------------------------------- #


def _children(node: ast.AST) -> Iterator[ast.AST]:
    """Child nodes, without type annotations, base classes and f-strings (no array indexing there)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield from node.decorator_list
        yield from node.args.defaults
        yield from (d for d in node.args.kw_defaults if d is not None)
        yield from node.body
    elif isinstance(node, ast.ClassDef):
        yield from node.decorator_list
        yield from node.body
    elif isinstance(node, ast.AnnAssign):
        yield node.target
        if node.value is not None:
            yield node.value
    elif isinstance(node, ast.JoinedStr) or type(node).__name__ in ("TypeAlias", "TemplateStr"):
        return
    else:
        yield from ast.iter_child_nodes(node)


def _subscripts(tree: ast.AST) -> Iterator[Tuple[ast.AST, ast.Subscript]]:
    """(nearest enclosing statement, subscript) for every subscript in the module."""
    stack: List[Tuple[ast.AST, Optional[ast.AST]]] = [(tree, None)]
    while stack:
        node, stmt = stack.pop()
        if isinstance(node, ast.stmt):
            stmt = node
        if isinstance(node, ast.Subscript) and stmt is not None:
            yield stmt, node
        for child in _children(node):
            stack.append((child, stmt))


def build(src) -> Index:
    """Decode every worthwhile subscript of *src* (a :class:`~python_beautifier.source.Source`)."""
    index = Index()
    found: List[Tuple[ast.AST, Guide]] = []
    arrays = _uses_array_library(src.tree)
    for stmt, sub in _subscripts(src.tree):
        try:
            guide = _guide(src, sub, arrays)
        except Exception:  # noqa: BLE001 - an annotation must never break the page
            guide = None
        if guide is not None:
            found.append((stmt, guide))
    found.sort(key=lambda pair: pair[1].start)

    counter = 0
    seen: Dict[int, Dict[str, Guide]] = {}
    for stmt, guide in found:
        first = seen.setdefault(id(stmt), {}).get(guide.key)
        if first is None:
            counter += 1
            guide.ids = [f"ix{counter}-{k}" for k in range(len(guide.cells))]
            seen[id(stmt)][guide.key] = guide
            index._by_stmt.setdefault(id(stmt), []).append(guide)
        else:  # the same subscript again in one statement: share the explanation
            guide.ids = first.ids
            guide.shown = False
        index.guides.append(guide)
        for axis in guide.axes:
            if axis.start is not None and axis.end is not None:
                index.wraps.append(Wrap(axis.start, axis.end, axis.kind, guide.ids[axis.cell], axis.title))

    index.wraps.sort(key=lambda w: (w.start, (-w.end[0], -w.end[1])))
    index._starts = [w.start for w in index.wraps]
    return index
