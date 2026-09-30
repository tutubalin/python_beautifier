"""Source file wrapper: text, tokens, comments and position arithmetic.

:class:`Source` is the single place that knows how to map AST nodes back to
the exact text they came from.  It also keeps track of which tokens and
comments have been rendered, which lets the renderer prove that no code was
lost on the way to HTML (see :meth:`Source.unrendered`).
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .highlight import Item, Pos, SpanSet, classify, render_items
from .indexing import Index
from .indexing import build as build_index

_SHEBANG = re.compile(r"^#!")
_CODING = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")


def decode_source(data: bytes) -> str:
    """Decode *data* honouring PEP 263 cookies and BOMs."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    except SyntaxError:
        encoding = "utf-8"
    try:
        return data.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return data.decode("utf-8", errors="replace")


@dataclass
class Comment:
    row: int
    col: int
    text: str  # includes the leading '#'
    own_line: bool
    used: bool = False

    @property
    def body(self) -> str:
        return self.text[1:].strip()


class Source:
    """A parsed Python file with token-accurate position helpers."""

    def __init__(self, text: str, filename: str = "<string>"):
        text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        self.text = text
        self.filename = filename
        self.lines: List[str] = text.split("\n")
        if self.lines and self.lines[-1] == "":
            # a trailing newline does not start a new (empty) line
            self._nlines = len(self.lines) - 1
        else:
            self._nlines = len(self.lines)
        self._ascii = [ln.isascii() for ln in self.lines]

        self.tree: ast.Module = ast.parse(text, filename=filename)
        self._soft_keywords = self._find_soft_keywords()
        self._type_spans = self._find_type_spans()

        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        self.items: List[Item] = classify(
            tokens, self.lines, soft_keywords=self._soft_keywords, type_spans=self._type_spans
        )
        self._starts: List[Pos] = [it.start for it in self.items]
        self._ends: Set[Pos] = {it.end for it in self.items}
        self._index: Optional[Index] = None
        self._rendered: Set[int] = set()
        self._exempt: Set[int] = set()
        self._count: Dict[int, int] = {}

        self.comments: List[Comment] = []
        self._comment_at: Dict[Pos, Comment] = {}
        self._by_row: Dict[int, Comment] = {}
        for it in self.items:
            if it.kind == "comment":
                own = not self.lines[it.srow - 1][: it.scol].strip()
                c = Comment(it.srow, it.scol, it.text.rstrip(), own)
                self.comments.append(c)
                self._comment_at[it.start] = c
                self._by_row[it.srow] = c

    # ------------------------------------------------------------------ #
    #  positions
    # ------------------------------------------------------------------ #

    @property
    def nlines(self) -> int:
        return self._nlines

    def char_col(self, row: int, byte_col: int) -> int:
        """Convert an AST (UTF-8 byte) column into a character column."""
        if row - 1 >= len(self.lines) or self._ascii[row - 1]:
            return byte_col
        return len(self.lines[row - 1].encode("utf-8")[:byte_col].decode("utf-8", "ignore"))

    def start(self, node: ast.AST) -> Pos:
        return (node.lineno, self.char_col(node.lineno, node.col_offset))  # type: ignore[attr-defined]

    def end(self, node: ast.AST) -> Pos:
        return (
            node.end_lineno,  # type: ignore[attr-defined]
            self.char_col(node.end_lineno, node.end_col_offset),  # type: ignore[attr-defined]
        )

    def first_line(self, node: ast.AST) -> int:
        """First physical line of *node*, including any decorators."""
        rows = [node.lineno]  # type: ignore[attr-defined]
        rows.extend(d.lineno for d in getattr(node, "decorator_list", ()))
        return min(rows)

    def end_line(self, node: ast.AST) -> int:
        return node.end_lineno  # type: ignore[attr-defined]

    def indent_of(self, row: int) -> int:
        line = self.lines[row - 1]
        return len(line) - len(line.lstrip())

    # ------------------------------------------------------------------ #
    #  text access
    # ------------------------------------------------------------------ #

    def slice(self, start: Pos, end: Pos) -> str:
        (sr, sc), (er, ec) = start, end
        if sr == er:
            return self.lines[sr - 1][sc:ec]
        parts = [self.lines[sr - 1][sc:]]
        parts.extend(self.lines[sr : er - 1])
        parts.append(self.lines[er - 1][:ec])
        return "\n".join(parts)

    def text_of(self, node: ast.AST) -> str:
        return self.slice(self.start(node), self.end(node))

    def oneline(self, node: ast.AST) -> str:
        """Source of *node* collapsed onto one line (comments dropped)."""
        try:
            return ast.unparse(node)
        except Exception:  # pragma: no cover - defensive
            return re.sub(r"\s+", " ", self.text_of(node))

    def line_span(self, first: int, last: int) -> Tuple[int, int]:
        """(number of non-blank lines, widest line) for rows *first*..*last*."""
        count = 0
        width = 0
        lines = self.lines
        for r in range(first, min(last, len(lines)) + 1):
            s = lines[r - 1].rstrip()
            if s.strip():
                count += 1
                if len(s) > width:
                    width = len(s)
        return count, width

    # ------------------------------------------------------------------ #
    #  rendering helpers
    # ------------------------------------------------------------------ #

    def _item_index(self, pos: Pos) -> int:
        return bisect_left(self._starts, pos)

    @property
    def index(self) -> Index:
        """The decoded NumPy / PyTorch style subscripts of this file (built on first use)."""
        if self._index is None:
            try:
                self._index = build_index(self)
            except Exception:  # noqa: BLE001 - decoration is optional; never let it break the page
                self._index = Index()
        return self._index

    def on_token_boundaries(self, start: Pos, end: Pos) -> bool:
        """Does a source range begin at a token start and finish at a token end?"""
        i = self._item_index(start)
        return i < len(self._starts) and self._starts[i] == start and end in self._ends

    def hl(self, start: Pos, end: Pos, dedent: Optional[int] = None) -> str:
        """Highlighted HTML for the source between *start* and *end*."""
        if dedent is None:
            dedent = start[1]
        i = self._item_index(start)
        j = i
        items = self.items
        n = len(items)
        while j < n and items[j].end <= end:
            j += 1
        for k in range(i, j):
            self._rendered.add(k)
            self._count[k] = self._count.get(k, 0) + 1
            if items[k].kind == "comment":
                c = self._comment_at.get(items[k].start)
                if c:
                    c.used = True
        wraps = [(w.start, w.end, w.open_tag) for w in self.index.wraps_in(start, end)]
        return render_items(items[i:j], self.lines, start, end, dedent, wraps)

    def hl_node(self, node: ast.AST, dedent: Optional[int] = None) -> str:
        return self.hl(self.start(node), self.end(node), dedent)

    def exempt(self, start: Pos, end: Pos) -> None:
        """Mark tokens as intentionally rendered elsewhere (e.g. docstrings)."""
        i = self._item_index(start)
        items = self.items
        while i < len(items) and items[i].end <= end:
            self._exempt.add(i)
            i += 1

    def tokens_between(self, start: Pos, end: Pos) -> List[Item]:
        i = self._item_index(start)
        out = []
        while i < len(self.items) and self.items[i].end <= end:
            out.append(self.items[i])
            i += 1
        return out

    def find_colon(self, start: Pos) -> Optional[Pos]:
        """Position of the first ``:`` at bracket depth 0 at/after *start*."""
        depth = 0
        for it in self.items[self._item_index(start) :]:
            if it.kind != "op":
                continue
            s = it.text
            if s in "([{":
                depth += 1
            elif s in ")]}":
                # a closing bracket at depth 0 belongs to an enclosing
                # construct that started before *start*, e.g. ``if (a and b):``
                depth = max(0, depth - 1)
            elif s == ":" and depth == 0:
                return it.start
        return None

    def find_keyword(self, word: str, start: Pos, end: Pos) -> Optional[Pos]:
        """Position of the first keyword *word* between *start* and *end*."""
        i = self._item_index(start)
        while i < len(self.items) and self.items[i].start < end:
            it = self.items[i]
            if it.kind == "kw" and it.text == word:
                return it.start
            i += 1
        return None

    # ------------------------------------------------------------------ #
    #  comments
    # ------------------------------------------------------------------ #

    def take_own_line(self, after_row: int, before_row: int, min_col: int = 0) -> List[Comment]:
        """Claim own-line comments with ``after_row < row < before_row``."""
        out: List[Comment] = []
        lo = _bisect_rows(self.comments, after_row + 1)
        for c in self.comments[lo:]:
            if c.row >= before_row:
                break
            if c.own_line and not c.used and c.col >= min_col:
                c.used = True
                out.append(c)
        return out

    def peek_own_line(self, after_row: int, before_row: int, min_col: int = 0) -> List[Comment]:
        lo = _bisect_rows(self.comments, after_row + 1)
        out = []
        for c in self.comments[lo:]:
            if c.row >= before_row:
                break
            if c.own_line and not c.used and c.col >= min_col:
                out.append(c)
        return out

    def take_inline(self, row: int, after_col: int = 0) -> Optional[Comment]:
        """Claim the trailing comment on *row* located right of *after_col*."""
        lo = _bisect_rows(self.comments, row)
        for c in self.comments[lo:]:
            if c.row > row:
                break
            if not c.own_line and not c.used and c.col >= after_col:
                c.used = True
                return c
        return None

    def trailing_comment(self, row: int) -> Optional[Comment]:
        """The trailing (same-line) comment on *row*, without claiming it."""
        c = self._by_row.get(row)
        return c if c is not None and not c.own_line else None

    def comment_above(self, row: int) -> Optional[Comment]:
        """An own-line comment on the line right above *row* (not claimed)."""
        c = self._by_row.get(row - 1)
        return c if c is not None and c.own_line else None

    # ------------------------------------------------------------------ #
    #  coverage
    # ------------------------------------------------------------------ #

    def duplicated(self) -> List[Item]:
        """Tokens that were rendered more than once (a bug: the output would repeat code)."""
        return [self.items[k] for k, n in sorted(self._count.items()) if n > 1 and self.items[k].kind in ("str", "fstr", "num", "name", "comment")]

    def unrendered(self) -> List[Item]:
        """Names, numbers, strings and comments that never reached the output."""
        missed = []
        for k, it in enumerate(self.items):
            if k in self._rendered or k in self._exempt:
                continue
            if it.kind in ("str", "fstr", "num", "comment") or (
                it.kind == "name"
            ):
                if it.kind == "comment":
                    c = self._comment_at.get(it.start)
                    if c is not None and c.used:
                        continue
                missed.append(it)
        return missed

    # ------------------------------------------------------------------ #
    #  ast-derived hints for the highlighter
    # ------------------------------------------------------------------ #

    def _find_soft_keywords(self) -> Set[Pos]:
        soft: Set[Pos] = set()
        match_cls = getattr(ast, "Match", None)
        alias_cls = getattr(ast, "TypeAlias", None)
        for node in ast.walk(self.tree):
            if match_cls is not None and isinstance(node, match_cls):
                soft.add((node.lineno, self.char_col(node.lineno, node.col_offset)))
                for case in node.cases:
                    pat = case.pattern
                    row = pat.lineno
                    line = self.lines[row - 1]
                    col = self.char_col(row, pat.col_offset)
                    # the ``case`` keyword precedes the pattern on the same line
                    k = line.rfind("case", 0, col)
                    if k >= 0:
                        soft.add((row, k))
            elif alias_cls is not None and isinstance(node, alias_cls):
                soft.add((node.lineno, self.char_col(node.lineno, node.col_offset)))
        return soft

    def _find_type_spans(self) -> SpanSet:
        spans = []

        def add(n: Optional[ast.AST]) -> None:
            if n is not None and hasattr(n, "end_lineno"):
                spans.append((self.start(n), self.end(n)))

        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
                    if arg is not None:
                        add(arg.annotation)
                add(node.returns)
            elif isinstance(node, ast.AnnAssign):
                add(node.annotation)
        return SpanSet(spans)


def _bisect_rows(comments: Sequence[Comment], row: int) -> int:
    lo, hi = 0, len(comments)
    while lo < hi:
        mid = (lo + hi) // 2
        if comments[mid].row < row:
            lo = mid + 1
        else:
            hi = mid
    return lo
