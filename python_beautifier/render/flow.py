"""The flow renderer: statements in, structured visual blocks out.

This module is where "syntax highlighting is not enough" happens.  Each kind
of statement gets a visual form that mirrors its *control flow*:

* ``if``/``else`` with short branches become **two columns** (yes | no);
* ``if``/``elif``/``else`` ladders become **decision tables** (when | then);
* ``if cond: return`` style guards become one slim **guard strip**;
* loops, ``with`` blocks and ``try`` statements become **framed blocks**,
  ``try`` with its **happy path and error path side by side**;
* ``return``/``raise``/``break``/``continue`` become **exit chips**;
* comments become **notes**, banner comments become **section rules**.

Layout decisions use simple, tunable heuristics (see the constants below);
the side-by-side / stacked switch happens in CSS (auto-fit grids and wrapping
flex rows), so a layout that does not fit simply stacks instead of overflowing.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from .. import analysis
from ..docstrings import inline_html
from ..highlight import fragment
from ..indexing import Guide
from ..source import Comment, Source
from .icons import icon
from .index import strip as index_strip
from .util import esc

TRY_TYPES = analysis._TRY_TYPES
MATCH = getattr(ast, "Match", None)

# --- layout tuning ---------------------------------------------------------
SPLIT_MAX_LINES = 10  # rows per column for an if/else split
SPLIT_MAX_WIDTH = (58, 42)  # widest line per column, by split depth
SPLIT_MAX_DEPTH = 2  # how many side-by-side layouts may nest
LADDER_MAX_LINES = 6  # rows per rung in a decision ladder
LADDER_MAX_WIDTH = 66
GUARD_MAX_LINES = 4
GUARD_MAX_WIDTH = 74
LANES_MAX_LINES = 10
COND_BREAKDOWN_WIDTH = 60  # and/or chains longer than this are split into rows
COLUMN_OVERHEAD = 8  # ch of gutter + padding consumed by each column
SOFT_WRAP = 0.85  # columns may be this fraction of the longest line (longer lines wrap)
FOLD_MIN_STATEMENTS = 3

_PRAGMA = re.compile(
    r"^\s*(noqa|type:\s*ignore|type:|pragma:|pylint:|fmt:|isort:|mypy:|ruff:|flake8:|nosec|pyright:|pytype:|coding[:=]|-\*-)",
    re.I,
)
_DECORATION = re.compile(r"^[\s#=\-_*~+.\u00b7\u2022\u2500\u2501\u2550\u25aa/\\|<>!:]*$")
_BANNER_TITLE = re.compile(r"^[\s#=\-_*~+.\u2500\u2501\u2550]{3,}\s*(?P<t>[^#=\-_*~+.\u2500\u2501\u2550].*?)\s*[\s#=\-_*~+.\u2500\u2501\u2550]{3,}$")
_TODO = re.compile(r"^\s*(TODO|FIXME|HACK|XXX|BUG|OPTIMIZE|REVIEW|NOTE)\b[\s:(\-]*(.*)$", re.S)
_REGION = re.compile(r"^\s*(?:region|#region)\b[:\s]*(.*)$", re.I)
_ENDREGION = re.compile(r"^\s*(?:end\s*region|#endregion)\b", re.I)
_LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log"}


@dataclass
class Ctx:
    """Where a body lives in the source, and how deeply it is nested."""

    indent: int  # column of the body's first statement
    start_row: int  # rows up to here are already accounted for
    end_row: int  # first row that belongs to whatever follows
    split_depth: int = 0  # enclosing side-by-side layouts
    depth: int = 0  # enclosing boxes
    fold_attrs: bool = False  # fold a leading block of class attributes
    fold_imports: bool = False  # fold import statements (module level)
    top: bool = False  # module-level flow


class _Out:
    """Accumulates flow children, grouping simple statements into runs."""

    def __init__(self) -> None:
        self.parts: List[str] = []
        self.run: List[str] = []
        self.run_widths: List[int] = []

    def row(self, html: str, align_width: Optional[int] = None) -> None:
        self.run.append(html)
        if align_width is not None:
            self.run_widths.append(align_width)

    def flush(self) -> None:
        if not self.run:
            return
        style = ""
        if self.run_widths:
            cw = min(max(self.run_widths), 60) + 2
            style = f' style="--cw:{cw}ch"'
        self.parts.append(f'<div class="run"{style}>{"".join(self.run)}</div>')
        self.run = []
        self.run_widths = []

    def block(self, html: str) -> None:
        self.flush()
        self.parts.append(html)

    def html(self) -> str:
        self.flush()
        return "".join(self.parts)


# --------------------------------------------------------------------------- #
#  comment classification
# --------------------------------------------------------------------------- #


def _looks_like_code(text: str) -> bool:
    t = text.strip()
    if len(t) < 3 or _DECORATION.match(t):
        return False
    try:
        tree = ast.parse(t)
    except SyntaxError:
        return bool(re.match(r"^(if|elif|else|for|while|try|except|finally|with|def|class)\b.*:\s*$", t)) or bool(
            re.match(r"^(return|raise|import|from|yield|break|continue|pass)\b", t)
            and not re.match(r"^(return|from|import)\s+\w+(\s+\w+){2,}", t)
        )
    if not tree.body:
        return False

    def trivial(n: ast.AST) -> bool:
        if isinstance(n, (ast.Name, ast.Constant, ast.Starred)):
            return True
        if isinstance(n, ast.Attribute):
            return trivial(n.value)
        if isinstance(n, (ast.Tuple, ast.List)):
            return all(trivial(e) for e in n.elts)
        return False

    for st in tree.body:
        if isinstance(st, ast.Expr) and trivial(st.value):
            continue
        if isinstance(st, ast.AnnAssign) and st.value is None:
            continue
        return True
    return False


def _is_trivial_string(s: ast.stmt) -> bool:
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


# --------------------------------------------------------------------------- #
#  the renderer
# --------------------------------------------------------------------------- #


class Flow:
    """Renders statement lists. Owned by the page renderer (``self.r``)."""

    def __init__(self, r) -> None:  # r: render.core.Renderer
        self.r = r
        self.src: Source = r.src
        self.used_ids: Set[str] = set()

    # ------------------------------------------------------------------ #
    #  public entry
    # ------------------------------------------------------------------ #

    def body(self, stmts: Sequence[ast.stmt], ctx: Ctx) -> str:
        """Render a statement list as flow HTML."""
        src = self.src
        out = _Out()
        prev_end = ctx.start_row
        n = len(stmts)
        fold: List[Tuple[str, Optional[int]]] = []  # rows destined for a fold block
        fold_count = 0
        folded_once = False

        def flush_fold() -> None:
            nonlocal fold, fold_count, folded_once
            if not fold:
                return
            if fold_count >= FOLD_MIN_STATEMENTS:
                label = "Imports" if ctx.fold_imports and ctx.top else "Class attributes"
                lines = sum(h.count('class="ln') for h, _ in fold)
                out.block(
                    f'<details class="fold"><summary>{icon("chev")}<span>{label}</span>'
                    f'<em>{lines} lines</em></summary><div class="run">{"".join(h for h, _ in fold)}</div></details>'
                )
                folded_once = True
            else:
                for h, w in fold:
                    out.row(h, w)
            fold = []
            fold_count = 0

        for idx, s in enumerate(stmts):
            first = src.first_line(s)
            next_row = src.first_line(stmts[idx + 1]) if idx + 1 < n else ctx.end_row
            notes = src.take_own_line(prev_end, first, ctx.indent)
            gap = self._blank_between(prev_end, notes[0].row if notes else first)
            note_items = self._notes(notes)

            foldable = self._foldable(s, ctx, folded_once)
            if not foldable:
                flush_fold()

            if foldable:
                for is_rule, h in note_items:
                    fold.append((h, None))
                same_row = idx + 1 < n and stmts[idx + 1].lineno == s.end_lineno
                h, w = self.simple(s, ctx, tail_ok=not same_row, gap=gap and not notes)
                fold.append((h, w))
                fold_count += 1
            else:
                for is_rule, h in note_items:
                    if is_rule:
                        out.block(h)
                    else:
                        out.row(h)
                same_row = idx + 1 < n and stmts[idx + 1].lineno == s.end_lineno
                self._statement(s, ctx, out, next_row, tail_ok=not same_row, gap=gap and not notes)
            prev_end = s.end_lineno

        flush_fold()
        trailing = src.take_own_line(prev_end, ctx.end_row, ctx.indent)
        for is_rule, h in self._notes(trailing):
            if is_rule:
                out.block(h)
            else:
                out.row(h)
        return out.html()

    # ------------------------------------------------------------------ #
    #  dispatch
    # ------------------------------------------------------------------ #

    def _foldable(self, s: ast.stmt, ctx: Ctx, folded_once: bool) -> bool:
        if ctx.fold_imports and isinstance(s, (ast.Import, ast.ImportFrom)):
            return True
        if ctx.fold_attrs and not folded_once:
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                return True
            if isinstance(s, ast.Assign) and all(isinstance(t, ast.Name) for t in s.targets):
                return True
        return False

    def _statement(self, s: ast.stmt, ctx: Ctx, out: _Out, next_row: int, tail_ok: bool, gap: bool) -> None:
        self._dispatch(s, ctx, out, next_row, tail_ok, gap)
        if not self._is_simple(s):
            # comments that no clause claimed (e.g. dedented between ``if`` and
            # ``else``) still deserve to be shown; they land after the block
            for is_rule, h in self._notes(self.src.take_own_line(self.src.first_line(s) - 1, s.end_lineno + 1, 0)):
                if is_rule:
                    out.block(h)
                else:
                    out.row(h)

    def _lead(self, prev_end_row: int, kw_row: int) -> str:
        """Notes written between two clauses (before ``else``/``except``...)."""
        items = self._notes(self.src.take_own_line(prev_end_row, kw_row, 0))
        if not items:
            return ""
        return '<div class="run">' + "".join(h for _, h in items) + "</div>"

    def _dispatch(self, s: ast.stmt, ctx: Ctx, out: _Out, next_row: int, tail_ok: bool, gap: bool) -> None:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.block(self.r.def_card(s, ctx, next_row))
            return
        if not ctx.top and not self._is_simple(s):
            # NumPy-style subscripts in a header (``if x[:, 0].any():``): explained right above it
            strip = index_strip(self._header_guides(s))
            if strip:
                out.row(strip)
        if isinstance(s, ast.If):
            out.block(self.if_stmt(s, ctx, next_row))
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            out.block(self.for_stmt(s, ctx, next_row))
        elif isinstance(s, ast.While):
            out.block(self.while_stmt(s, ctx, next_row))
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            out.block(self.with_stmt(s, ctx, next_row))
        elif TRY_TYPES and isinstance(s, TRY_TYPES):
            out.block(self.try_stmt(s, ctx, next_row))
        elif MATCH is not None and isinstance(s, MATCH):
            out.block(self.match_stmt(s, ctx, next_row))
        else:
            h, w = self.simple(s, ctx, tail_ok=tail_ok, gap=gap)
            out.row(h, w)

    def _is_elif(self, node: ast.If) -> bool:
        """Was this nested ``If`` written as ``elif`` (and not as ``else:`` followed by an ``if``)?"""
        return self.src.lines[node.lineno - 1][node.col_offset : node.col_offset + 4] == "elif"

    def _header_guides(self, s: ast.stmt) -> List[Guide]:
        """Decoded subscripts in the header of *s*; for ``if`` that includes every ``elif`` of its chain."""
        guides = list(self.src.index.for_statement(s))
        node = s
        while isinstance(node, ast.If) and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            if not self._is_elif(node.orelse[0]):
                break
            node = node.orelse[0]
            guides += self.src.index.for_statement(node)
        return guides

    # ------------------------------------------------------------------ #
    #  measuring
    # ------------------------------------------------------------------ #

    def measure(self, stmts: Sequence[ast.stmt]) -> Tuple[int, int]:
        """(lines, widest line relative to the block indent) of *stmts*."""
        if not stmts:
            return 0, 0
        src = self.src
        first = src.first_line(stmts[0])
        last = stmts[-1].end_lineno
        base = src.start(stmts[0])[1]
        lines = 0
        width = 0
        for r in range(first, min(last, len(src.lines)) + 1):
            text = src.lines[r - 1].rstrip()
            if text.strip():
                lines += 1
                width = max(width, len(text) - base)
        return lines, width

    @staticmethod
    def _is_simple(s: ast.stmt) -> bool:
        return not isinstance(
            s,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            + ((MATCH,) if MATCH else ())
            + (tuple(TRY_TYPES) if TRY_TYPES else ()),
        )

    def _blank_between(self, prev_end: int, upto: int) -> bool:
        lines = self.src.lines
        for r in range(prev_end + 1, upto):
            if 0 < r <= len(lines) and not lines[r - 1].strip():
                return True
        return False

    # ------------------------------------------------------------------ #
    #  notes (comments)
    # ------------------------------------------------------------------ #

    def _notes(self, comments: List[Comment]) -> List[Tuple[bool, str]]:
        """Turn own-line comments into (is_rule, html) items."""
        if not comments:
            return []
        groups: List[List[Comment]] = [[comments[0]]]
        for c in comments[1:]:
            if c.row == groups[-1][-1].row + 1:
                groups[-1].append(c)
            else:
                groups.append([c])
        items: List[Tuple[bool, str]] = []
        for g in groups:
            items.extend(self._note_group(g))
        return items

    def _note_group(self, g: List[Comment]) -> List[Tuple[bool, str]]:
        lines = [c.text[1:].lstrip("#") if c.text.startswith("#") else c.text for c in g]
        lines = [ln[1:] if ln.startswith(" ") else ln for ln in lines]
        first_row = g[0].row
        # pragma-only lines (``# type: ignore`` alone on a line)
        if all(_PRAGMA.match(ln) for ln in lines):
            return [(False, f'<div class="note pragma"><span class="nt">{esc(" ".join(l.strip() for l in lines))}</span></div>')]
        # regions
        if len(lines) == 1:
            if _ENDREGION.match(lines[0]):
                return []
            m = _REGION.match(lines[0])
            if m:
                return [(True, self._rule(m.group(1).strip() or "Region", first_row))]
        # banners:  # ------ / # Title / # ------
        deco = [ln for ln in lines if ln.strip() and _DECORATION.match(ln) and len(ln.strip()) >= 3]
        if deco:
            titles = [ln.strip() for ln in lines if ln.strip() and not (_DECORATION.match(ln) and len(ln.strip()) >= 3)]
            title = " \u00b7 ".join(re.sub(r"^[\s#=\-_*~+.]+|[\s#=\-_*~+.]+$", "", t) for t in titles)
            return [(True, self._rule(title, first_row))]
        if len(lines) == 1:
            m = _BANNER_TITLE.match(lines[0])
            if m:
                return [(True, self._rule(m.group("t"), first_row))]
        text = "\n".join(lines)
        m = _TODO.match(text)
        if m:
            tag, rest = m.group(1), m.group(2).strip()
            body = inline_html(rest) if rest else ""
            return [
                (
                    False,
                    f'<div class="note todo"><span class="todo-tag">{tag}</span>'
                    f'<div class="nt">{body}</div></div>',
                )
            ]
        if _looks_like_code(text):
            code = "\n".join(l.rstrip() for l in lines)
            return [(False, f'<div class="note ccode"><span class="cc-tag">commented out</span><code class="c">{fragment(code)}</code></div>')]
        body = "".join(f'<span class="nl">{inline_html(ln.rstrip()) or "&nbsp;"}</span>' for ln in lines)
        return [(False, f'<div class="note">{icon("comment")}<div class="nt">{body}</div></div>')]

    def _rule(self, title: str, row: int) -> str:
        label = f"<span>{inline_html(title)}</span>" if title else ""
        return f'<div class="rule">{label}</div>'

    # ------------------------------------------------------------------ #
    #  simple statements
    # ------------------------------------------------------------------ #

    def _simple_kind(self, s: ast.stmt) -> Tuple[str, str]:
        if isinstance(s, ast.Return):
            return "x x-ret", "return"
        if isinstance(s, ast.Raise):
            return "x x-raise", "alert"
        if isinstance(s, ast.Break):
            return "x x-brk", "stop"
        if isinstance(s, ast.Continue):
            return "x x-cont", "skip"
        if isinstance(s, ast.Pass):
            return "dim", ""
        if isinstance(s, ast.Assert):
            return "st st-assert", "circle-check"
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            return "st st-imp", "package"
        if isinstance(s, (ast.Global, ast.Nonlocal)):
            return "st st-glob", "globe"
        if isinstance(s, ast.Delete):
            return "st st-del", "trash"
        value = getattr(s, "value", None)
        if isinstance(s, ast.Expr):
            if isinstance(value, ast.Call):
                name = analysis.dotted_name(value.func) or ""
                if name in analysis._EXIT_CALLS:
                    return "x x-exit", "stop"
                head, _, method = name.rpartition(".")
                if name == "print" or (method in _LOG_METHODS and ("log" in head.lower())):
                    return "st st-log", "terminal"
            if _is_trivial_string(s):
                return "st st-str", ""
        if isinstance(value, ast.Await):
            return "st st-await", "clock"
        if isinstance(value, (ast.Yield, ast.YieldFrom)):
            return "st st-yield", "up-right"
        return "", ""

    def _anchor(self, row: int) -> str:
        key = f"L{row}"
        if key in self.used_ids:
            return ""
        self.used_ids.add(key)
        return f' id="{key}"'

    def tail_html(self, c: Optional[Comment]) -> str:
        if c is None:
            return ""
        body = c.body
        if not body:
            return ""
        cls = "tc pr" if _PRAGMA.match(body) else "tc"
        m = _TODO.match(body)
        if m:
            cls += " td"
        return f'<span class="{cls}">{esc(body)}</span>'

    def simple(self, s: ast.stmt, ctx: Ctx, *, tail_ok: bool, gap: bool = False) -> Tuple[str, Optional[int]]:
        src = self.src
        code = src.hl_node(s)
        kind, ic = self._simple_kind(s)
        end_row, end_col = src.end(s)
        tail = src.take_inline(end_row, end_col) if tail_ok else None
        single = s.lineno == s.end_lineno
        align = None
        classes = "ln"
        if kind:
            classes += " " + kind
        if gap:
            classes += " gap"
        if tail is not None:
            classes += " hc" if single else ""
            if single:
                align = end_col - src.start(s)[1]
        glyph = icon(ic) if ic else ""
        html = (
            f'<div class="{classes}"{self._anchor(s.lineno)}>'
            f'<i class="g" data-n="{s.lineno}">{glyph}</i>'
            f'<code class="c">{code}</code>{self.tail_html(tail)}</div>'
        )
        return html + index_strip(src.index.for_statement(s)), align

    # ------------------------------------------------------------------ #
    #  headers and boxes
    # ------------------------------------------------------------------ #

    def tag(self, kind: str, label: str, ic: str) -> str:
        return f'<span class="tag t-{kind}">{icon(ic)}<b>{label}</b></span>'

    def _header_end(self, expr_end: Tuple[int, int], body: Sequence[ast.stmt]) -> Tuple[int, int, Optional[Comment]]:
        """(colon row, colon col, trailing header comment)."""
        src = self.src
        colon = src.find_colon(expr_end) or expr_end
        row, col = colon
        tail = None
        if not (body and body[0].lineno == row):
            tail = src.take_inline(row, col + 1)
        return row, col, tail

    def _child_ctx(self, parent: Ctx, stmts: Sequence[ast.stmt], start_row: int, end_row: int, *, split: bool = False) -> Ctx:
        indent = self.src.start(stmts[0])[1] if stmts else parent.indent + 4
        return Ctx(
            indent=indent,
            start_row=start_row,
            end_row=end_row,
            split_depth=parent.split_depth + (1 if split else 0),
            depth=parent.depth + 1,
        )

    def box(
        self,
        kind: str,
        tag: str,
        head: str,
        body: str,
        *,
        row: int = 0,
        tail: Optional[Comment] = None,
        cls: str = "",
        open_: bool = True,
        aside: str = "",
    ) -> str:
        ident = self._anchor(row) if row else ""
        op = " open" if open_ else ""
        return (
            f'<details class="box b-{kind}{(" " + cls) if cls else ""}"{ident}{op}>'
            f'<summary class="bh">{icon("chev", "chv")}{tag}{head}{aside}{self.tail_html(tail)}</summary>'
            f'<div class="bb">{body}</div></details>'
        )

    def code(self, html: str, cls: str = "") -> str:
        return f'<code class="c{(" " + cls) if cls else ""}">{html}</code>'

    # ------------------------------------------------------------------ #
    #  if / elif / else
    # ------------------------------------------------------------------ #

    @dataclass
    class _Branch:
        kw: str  # if | elif | else
        test: Optional[ast.expr]
        body: Sequence[ast.stmt]
        row: int  # row of the keyword
        colon: Tuple[int, int]
        tail: Optional[Comment]
        end_row: int = 0
        lines: int = 0
        width: int = 0
        prev_end: int = 0  # last row of the previous clause

    def _flatten_if(self, s: ast.If, next_row: int) -> List["Flow._Branch"]:
        src = self.src
        branches: List[Flow._Branch] = []
        node = s
        while True:
            kw = "if" if node is s else "elif"
            colon_row, colon_col, tail = self._header_end(src.end(node.test), node.body)
            branches.append(Flow._Branch(kw, node.test, node.body, node.lineno, (colon_row, colon_col), tail))
            orelse = node.orelse
            if len(orelse) == 1 and isinstance(orelse[0], ast.If) and self._is_elif(orelse[0]):
                node = orelse[0]
                continue
            break
        if orelse:
            last_body_end = src.end(node.body[-1])
            kw_pos = src.find_keyword("else", last_body_end, src.start(orelse[0])) or (orelse[0].lineno - 1, 0)
            colon = src.find_colon(kw_pos) or kw_pos
            tail = None if orelse[0].lineno == colon[0] else src.take_inline(colon[0], colon[1] + 1)
            branches.append(Flow._Branch("else", None, orelse, kw_pos[0], colon, tail))
        for i, b in enumerate(branches):
            b.end_row = branches[i + 1].row if i + 1 < len(branches) else next_row
            b.lines, b.width = self.measure(b.body)
            if i:
                b.prev_end = branches[i - 1].body[-1].end_lineno
        return branches

    def _branch_body(self, b: "Flow._Branch", ctx: Ctx, *, split: bool = False) -> str:
        lead = self._lead(b.prev_end, b.row) if b.prev_end else ""
        c = self._child_ctx(ctx, b.body, b.colon[0], b.end_row, split=split)
        return lead + self.body(b.body, c)

    def if_stmt(self, s: ast.If, ctx: Ctx, next_row: int) -> str:
        if self._is_main_guard(s.test):
            return self._main_block(s, ctx, next_row)
        branches = self._flatten_if(s, next_row)
        n = len(branches)
        has_else = branches[-1].kw == "else"
        if n == 1:
            b = branches[0]
            if self._guard_ok(b, ctx):
                return self._guard(b, ctx)
            return self._gate(b, ctx, s.lineno)
        if n == 2 and has_else:
            if self._split_ok(branches, ctx):
                return self._split(branches, ctx, s.lineno)
            return self._stack(branches, ctx, s.lineno)
        if self._ladder_ok(branches):
            return self._ladder(branches, ctx, s.lineno)
        return self._stack(branches, ctx, s.lineno)

    # -- conditions ------------------------------------------------------- #

    def cond(self, test: ast.expr, *, allow_rows: bool = True) -> str:
        """HTML for a condition; long and/or chains are broken into rows."""
        src = self.src
        multi = test.lineno != test.end_lineno
        text_w = len(src.oneline(test))
        commented = any(test.lineno <= c.row <= test.end_lineno for c in src.comments)
        if (
            allow_rows
            and isinstance(test, ast.BoolOp)
            and not commented
            and (text_w > COND_BREAKDOWN_WIDTH or (multi and len(test.values) > 2))
        ):
            return self._cond_rows(test)
        return self.code(src.hl_node(test), "cond")

    def _cond_rows(self, test: ast.BoolOp) -> str:
        src = self.src
        word = "AND" if isinstance(test.op, ast.And) else "OR"
        rows = []
        for i, v in enumerate(test.values):
            html = src.hl_node(v)
            if isinstance(v, (ast.BoolOp, ast.IfExp, ast.Lambda, ast.NamedExpr)):
                html = f"({html})"
            op = f'<b class="cop">{word}</b>' if i else '<b class="cop first"></b>'
            rows.append(f'<span class="cr">{op}<code class="c">{html}</code></span>')
        return f'<span class="cond-rows">{"".join(rows)}</span>'

    def _cond_width(self, test: Optional[ast.expr]) -> int:
        if test is None:
            return 0
        if isinstance(test, ast.BoolOp) and len(self.src.oneline(test)) > COND_BREAKDOWN_WIDTH:
            return max(len(self.src.oneline(v)) for v in test.values) + 6
        return len(self.src.oneline(test))

    @staticmethod
    def _is_main_guard(test: ast.expr) -> bool:
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        )

    def _main_block(self, s: ast.If, ctx: Ctx, next_row: int) -> str:
        branches = self._flatten_if(s, next_row)
        b = branches[0]
        body = self._branch_body(b, ctx)
        head = self.code(self.src.hl_node(s.test), "cond")
        tag = self.tag("main", "ENTRY POINT", "play")
        html = self.box("main", tag, head, body, row=s.lineno, tail=b.tail)
        rest = ""
        if len(branches) > 1:
            rest = self._stack(branches[1:], ctx, branches[1].row)
        return html + rest

    # -- layout predicates ------------------------------------------------ #

    def _exit_only(self, body: Sequence[ast.stmt]) -> Optional[str]:
        return analysis.ends_flow(body)

    def _guard_ok(self, b: "Flow._Branch", ctx: Ctx) -> bool:
        if not all(self._is_simple(s) for s in b.body):
            return False
        if b.lines > GUARD_MAX_LINES or b.width > GUARD_MAX_WIDTH:
            return False
        if self._cond_width(b.test) > GUARD_MAX_WIDTH or (b.test is not None and b.test.lineno != b.test.end_lineno):
            return False
        # a guard reads best when the body leaves the flow or is a single action
        return len(b.body) <= 2 or bool(self._exit_only(b.body))

    def _split_ok(self, branches: List["Flow._Branch"], ctx: Ctx) -> bool:
        if ctx.split_depth >= SPLIT_MAX_DEPTH:
            return False
        a, b = branches
        limit = SPLIT_MAX_WIDTH[min(ctx.split_depth, len(SPLIT_MAX_WIDTH) - 1)]
        if max(a.lines, b.lines) > SPLIT_MAX_LINES or max(a.width, b.width) > limit:
            return False
        if a.lines == 0 or b.lines == 0:
            return False
        lo, hi = sorted((a.lines, b.lines))
        if hi > 4 and hi > 3 * lo + 1:  # very unbalanced: stacking is tighter
            return False
        return True

    def _ladder_ok(self, branches: List["Flow._Branch"]) -> bool:
        if len(branches) < 2:
            return False
        for b in branches:
            if b.lines > LADDER_MAX_LINES or b.width > LADDER_MAX_WIDTH:
                return False
            if b.test is not None and self._cond_width(b.test) > 52:
                return False
        return True

    # -- layouts ---------------------------------------------------------- #

    def _guard(self, b: "Flow._Branch", ctx: Ctx) -> str:
        leaves = self._exit_only(b.body)
        kind = f" g-{leaves}" if leaves else ""
        child = self._child_ctx(ctx, b.body, b.colon[0], b.end_row)
        body = self.body(b.body, child)
        tag = self.tag("if", "IF", "diamond")
        ident = self._anchor(b.row)
        return (
            f'<div class="guard{kind}"{ident}><div class="gc">{tag}{self.cond(b.test, allow_rows=False)}{self.tail_html(b.tail)}</div>'
            f'<span class="garrow">{icon("arrow-r")}</span><div class="gt">{body}</div></div>'
        )

    def _gate(self, b: "Flow._Branch", ctx: Ctx, row: int) -> str:
        body = self._branch_body(b, ctx)
        tag = self.tag("if", "IF", "diamond")
        return self.box("if", tag, self.cond(b.test), body, row=row, tail=b.tail)

    def _if_header(self, b: "Flow._Branch", kw_label: str, ic: str = "diamond") -> str:
        return f'<div class="if-h">{self.tag("if", kw_label, ic)}{self.cond(b.test)}{self.tail_html(b.tail)}</div>'

    def _split(self, branches: List["Flow._Branch"], ctx: Ctx, row: int) -> str:
        yes, no = branches
        col = int(max(yes.width, no.width, 20) * SOFT_WRAP) + COLUMN_OVERHEAD
        left = self._branch_body(yes, ctx, split=True)
        right = self._branch_body(no, ctx, split=True)
        yes_exit = self._exit_only(yes.body)
        no_exit = self._exit_only(no.body)
        ident = self._anchor(row)
        return (
            f'<div class="if split"{ident}>'
            f"{self._if_header(yes, 'IF')}"
            f'<div class="cols" style="--col:{col}ch">'
            f'<div class="col yes{" leaves" if yes_exit else ""}"><div class="col-h">{icon("check")}<b>yes</b><span>condition holds</span></div><div class="col-b">{left}</div></div>'
            f'<div class="col no{" leaves" if no_exit else ""}"><div class="col-h">{icon("x")}<b>no</b><span>otherwise</span>{self.tail_html(no.tail)}</div><div class="col-b">{right}</div></div>'
            f"</div></div>"
        )

    @staticmethod
    def _ladder_style(cond_w: int, then_w: int) -> str:
        """Column sizes of a decision ladder: the rungs wrap when the width runs out."""
        when = max(26, min(cond_w + 13, 48))
        then = max(30, min(then_w + COLUMN_OVERHEAD, 84))
        return f"--when-w:{when}ch;--then-w:{then}ch"

    def _ladder(self, branches: List["Flow._Branch"], ctx: Ctx, row: int) -> str:
        width = max(b.width for b in branches)
        cond_w = max((self._cond_width(b.test) for b in branches), default=0)
        style = self._ladder_style(cond_w, width)
        rungs = []
        for b in branches:
            body = self._branch_body(b, ctx, split=True)
            if b.kw == "else":
                when = f'{self.tag("else", "ELSE", "git-merge")}<span class="otherwise">otherwise</span>{self.tail_html(b.tail)}'
                cls = "rung else"
            else:
                label = "IF" if b.kw == "if" else "ELIF"
                when = f'{self.tag("if", label, "diamond" if b.kw == "if" else "branch")}{self.cond(b.test)}{self.tail_html(b.tail)}'
                cls = "rung"
            ex = self._exit_only(b.body)
            rungs.append(f'<div class="{cls}{" leaves" if ex else ""}"><div class="when">{when}</div><div class="then">{body}</div></div>')
        ident = self._anchor(row)
        return f'<div class="if ladder" style="{style}"{ident}>{"".join(rungs)}</div>'

    def _stack(self, branches: List["Flow._Branch"], ctx: Ctx, row: int) -> str:
        parts = []
        for i, b in enumerate(branches):
            body = self._branch_body(b, ctx)
            if b.kw == "else":
                head = f'{self.tag("else", "ELSE", "git-merge")}<span class="otherwise">otherwise</span>'
                sec = "no"
            else:
                label = "IF" if b.kw == "if" else "ELIF"
                head = f'{self.tag("if", label, "diamond" if b.kw == "if" else "branch")}{self.cond(b.test)}'
                sec = "yes"
            parts.append(
                f'<details class="arm {sec}" open><summary class="bh">{icon("chev", "chv")}{head}{self.tail_html(b.tail)}</summary>'
                f'<div class="bb">{body}</div></details>'
            )
        return f'<div class="if stack"{self._anchor(row)}>{"".join(parts)}</div>'

    # ------------------------------------------------------------------ #
    #  loops
    # ------------------------------------------------------------------ #

    def _else_clause(self, stmts: Sequence[ast.stmt], last_end: Tuple[int, int], keyword: str = "else") -> Tuple[int, Tuple[int, int], Optional[Comment]]:
        src = self.src
        kw_pos = src.find_keyword(keyword, last_end, src.start(stmts[0])) or (stmts[0].lineno - 1, 0)
        colon = src.find_colon(kw_pos) or kw_pos
        tail = None if stmts[0].lineno == colon[0] else src.take_inline(colon[0], colon[1] + 1)
        return kw_pos[0], colon, tail

    def for_stmt(self, s, ctx: Ctx, next_row: int) -> str:
        src = self.src
        is_async = isinstance(s, ast.AsyncFor)
        head = self.code(src.hl(src.start(s.target), src.end(s.iter)))
        row, col, tail = self._header_end(src.end(s.iter), s.body)
        else_row = None
        if s.orelse:
            else_row, else_colon, else_tail = self._else_clause(s.orelse, src.end(s.body[-1]))
        body_ctx = self._child_ctx(ctx, s.body, row, else_row or next_row)
        body = self.body(s.body, body_ctx)
        if s.orelse:
            else_ctx = self._child_ctx(ctx, s.orelse, else_colon[0], next_row)
            body += (
                f'<div class="sub"><div class="sub-h">{self.tag("else", "ELSE", "git-merge")}'
                f'<span class="otherwise">loop finished without <code>break</code></span>{self.tail_html(else_tail)}</div>'
                f'<div class="sub-b">{self.body(s.orelse, else_ctx)}</div></div>'
            )
        label = "ASYNC FOR" if is_async else "FOR EACH"
        return self.box("loop", self.tag("loop", label, "loop"), head, body, row=s.lineno, tail=tail)

    def while_stmt(self, s: ast.While, ctx: Ctx, next_row: int) -> str:
        src = self.src
        row, col, tail = self._header_end(src.end(s.test), s.body)
        forever = isinstance(s.test, ast.Constant) and bool(s.test.value) is True
        else_row = None
        if s.orelse:
            else_row, else_colon, else_tail = self._else_clause(s.orelse, src.end(s.body[-1]))
        body_ctx = self._child_ctx(ctx, s.body, row, else_row or next_row)
        body = self.body(s.body, body_ctx)
        if s.orelse:
            else_ctx = self._child_ctx(ctx, s.orelse, else_colon[0], next_row)
            body += (
                f'<div class="sub"><div class="sub-h">{self.tag("else", "ELSE", "git-merge")}'
                f'<span class="otherwise">loop ended without <code>break</code></span>{self.tail_html(else_tail)}</div>'
                f'<div class="sub-b">{self.body(s.orelse, else_ctx)}</div></div>'
            )
        if forever:
            src.exempt(src.start(s.test), src.end(s.test))  # ``while 1:`` / ``while True:``
            tag = self.tag("loop", "LOOP FOREVER", "infinity")
            head = '<span class="hint">until <code>break</code> or <code>return</code></span>'
        else:
            tag = self.tag("loop", "WHILE", "rotate")
            head = self.cond(s.test)
        return self.box("loop", tag, head, body, row=s.lineno, tail=tail)

    # ------------------------------------------------------------------ #
    #  with
    # ------------------------------------------------------------------ #

    def with_stmt(self, s, ctx: Ctx, next_row: int) -> str:
        src = self.src
        is_async = isinstance(s, ast.AsyncWith)
        items = []
        for it in s.items:
            end = src.end(it.optional_vars) if it.optional_vars is not None else src.end(it.context_expr)
            items.append(self.code(src.hl(src.start(it.context_expr), end), "wi"))
        last = s.items[-1]
        last_end = src.end(last.optional_vars) if last.optional_vars is not None else src.end(last.context_expr)
        row, col, tail = self._header_end(last_end, s.body)
        body_ctx = self._child_ctx(ctx, s.body, row, next_row)
        body = self.body(s.body, body_ctx)
        head = f'<span class="wis">{"".join(items)}</span>'
        label = "ASYNC WITH" if is_async else "WITH"
        return self.box("with", self.tag("with", label, "layers"), head, body, row=s.lineno, tail=tail)

    # ------------------------------------------------------------------ #
    #  try / except
    # ------------------------------------------------------------------ #

    def try_stmt(self, s, ctx: Ctx, next_row: int) -> str:
        src = self.src
        star = TRY_TYPES and type(s).__name__ == "TryStar"
        clauses = []  # (kind, stmts, header_row, colon, tail, node)
        try_colon = src.find_colon((s.lineno, s.col_offset)) or (s.lineno, 0)
        try_tail = None if s.body[0].lineno == try_colon[0] else src.take_inline(try_colon[0], try_colon[1] + 1)
        clauses.append(("try", s.body, s.lineno, try_colon, try_tail, None))
        prev_end = src.end(s.body[-1])
        for h in s.handlers:
            colon = src.find_colon(src.end(h.type) if h.type is not None else (h.lineno, h.col_offset)) or (h.lineno, 0)
            tail = None if h.body[0].lineno == colon[0] else src.take_inline(colon[0], colon[1] + 1)
            clauses.append(("except", h.body, h.lineno, colon, tail, h))
            prev_end = src.end(h.body[-1])
        if s.orelse:
            row, colon, tail = self._else_clause(s.orelse, prev_end)
            clauses.append(("else", s.orelse, row, colon, tail, None))
            prev_end = src.end(s.orelse[-1])
        if s.finalbody:
            row, colon, tail = self._else_clause(s.finalbody, prev_end, "finally")
            clauses.append(("finally", s.finalbody, row, colon, tail, None))

        rendered = []
        metrics = []
        for i, (kind, stmts, hrow, colon, tail, node) in enumerate(clauses):
            end_row = clauses[i + 1][2] if i + 1 < len(clauses) else next_row
            lanes_mode = kind in ("try", "except")
            c = self._child_ctx(ctx, stmts, colon[0], end_row, split=lanes_mode)
            rendered.append(self.body(stmts, c))
            metrics.append(self.measure(stmts))

        handlers = [i for i, c in enumerate(clauses) if c[0] == "except"]
        use_lanes = False
        if handlers and ctx.split_depth < SPLIT_MAX_DEPTH:
            limit = SPLIT_MAX_WIDTH[min(ctx.split_depth, len(SPLIT_MAX_WIDTH) - 1)]
            lane_idx = [0] + handlers
            use_lanes = (
                len(handlers) <= 3
                and all(metrics[i][0] <= LANES_MAX_LINES and metrics[i][1] <= limit for i in lane_idx)
                and sum(metrics[i][0] for i in handlers) <= 14
            )

        def handler_head(node: ast.ExceptHandler, colon: Tuple[int, int]) -> str:
            label = "EXCEPT*" if star else "EXCEPT"
            if node.type is None:
                what = '<span class="hint">anything else</span>'
            else:
                what = self.code(src.hl(src.start(node.type), colon), "exc")
                # ``hl`` stops at the colon; drop a dangling space
            return f'{self.tag("except", label, "alert")}{what}'

        def section(kind: str, idx: int, label_html: str, cls: str) -> str:
            kind_, stmts, hrow, colon, tail, node = clauses[idx]
            return (
                f'<div class="handler {cls}"><div class="bh">{label_html}{self.tail_html(tail)}</div>'
                f'<div class="bb">{rendered[idx]}</div></div>'
            )

        parts: List[str] = []
        ident = self._anchor(s.lineno)
        try_tag = self.tag("try", "TRY", "shield")
        if use_lanes:
            col = int(max(max(metrics[i][1] for i in [0] + handlers), 20) * SOFT_WRAP) + COLUMN_OVERHEAD
            handler_html = "".join(
                section("except", i, handler_head(clauses[i][5], clauses[i][3]), "err") for i in handlers
            )
            parts.append(
                f'<div class="lanes" style="--col:{col}ch">'
                f'<div class="lane ok"><div class="lane-h">{icon("check")}<b>try</b><span>happy path</span>{self.tail_html(clauses[0][4])}</div>'
                f'<div class="lane-b">{rendered[0]}</div></div>'
                f'<div class="lane err"><div class="lane-h">{icon("alert")}<b>if it fails</b><span>error path</span></div>'
                f'<div class="lane-b">{handler_html}</div></div></div>'
            )
        else:
            parts.append(
                f'<div class="handler ok"><div class="bh">{try_tag}{self.tail_html(clauses[0][4])}</div><div class="bb">{rendered[0]}</div></div>'
            )
            for i in handlers:
                parts.append(section("except", i, handler_head(clauses[i][5], clauses[i][3]), "err"))
        for i, c in enumerate(clauses):
            if c[0] == "else":
                label = f'{self.tag("else", "ELSE", "git-merge")}<span class="otherwise">no exception was raised</span>'
                parts.append(section("else", i, label, "okelse"))
            elif c[0] == "finally":
                label = f'{self.tag("finally", "FINALLY", "flag")}<span class="otherwise">always runs</span>'
                parts.append(section("finally", i, label, "fin"))
        head = f'<span class="hint">{"error handling" if handlers else "cleanup guaranteed"}</span>'
        header = f'<div class="try-h">{self.tag("try", "TRY", "shield")}{head}</div>' if use_lanes else ""
        return f'<div class="try {"lanes-mode" if use_lanes else "stack-mode"}"{ident}>{header}{"".join(parts)}</div>'

    # ------------------------------------------------------------------ #
    #  match
    # ------------------------------------------------------------------ #

    def match_stmt(self, s, ctx: Ctx, next_row: int) -> str:
        src = self.src
        subject = self.code(src.hl_node(s.subject), "cond")
        colon = src.find_colon(src.end(s.subject)) or src.end(s.subject)
        tail = src.take_inline(colon[0], colon[1] + 1) if not s.cases[0].body[0].lineno == colon[0] else None
        rungs = []
        widths = []
        for i, case in enumerate(s.cases):
            pat_end = src.end(case.guard) if case.guard is not None else src.end(case.pattern)
            ccolon = src.find_colon(pat_end) or pat_end
            ctail = None if case.body[0].lineno == ccolon[0] else src.take_inline(ccolon[0], ccolon[1] + 1)
            end_row = s.cases[i + 1].pattern.lineno if i + 1 < len(s.cases) else next_row
            c = self._child_ctx(ctx, case.body, ccolon[0], end_row, split=True)
            body = self.body(case.body, c)
            lines, width = self.measure(case.body)
            widths.append((lines, width, pat_end[1] - src.start(case.pattern)[1]))
            pattern = self.code(src.hl(src.start(case.pattern), ccolon), "cond")
            is_wild = type(case.pattern).__name__ == "MatchAs" and getattr(case.pattern, "pattern", None) is None and case.guard is None
            tag = self.tag("case", "DEFAULT" if is_wild else "CASE", "shuffle" if not is_wild else "git-merge")
            shown = pattern if not is_wild else '<span class="otherwise">anything else</span>'
            when = f"{tag}{shown}{self.tail_html(ctail)}"
            ex = self._exit_only(case.body)
            rungs.append(f'<div class="rung{" else" if is_wild else ""}{" leaves" if ex else ""}"><div class="when">{when}</div><div class="then">{body}</div></div>')
        compact = all(l <= LADDER_MAX_LINES and w <= LADDER_MAX_WIDTH for l, w, _ in widths)
        style = self._ladder_style(max((p for _, _, p in widths), default=20), max((w for _, w, _ in widths), default=30))
        inner = f'<div class="if ladder" style="{style}">{"".join(rungs)}</div>'
        if not compact:
            inner = f'<div class="if ladder stackme">{"".join(rungs)}</div>'
        return self.box("match", self.tag("match", "MATCH", "shuffle"), subject, inner, row=s.lineno, tail=tail)
