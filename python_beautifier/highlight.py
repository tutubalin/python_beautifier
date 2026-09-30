"""Token classification and HTML generation for Python source code.

The highlighter goes a little further than a plain lexer: it knows about
calls, attribute access, type annotations, exception names, ``self`` and
keyword arguments, so that code can be coloured *semantically*.

Everything here works on :mod:`tokenize` output, so the exact source text is
always preserved - the HTML is a lossless rendering of the input.
"""
from __future__ import annotations

import builtins
import io
import keyword
import re
import tokenize
from bisect import bisect_right
from html import escape
from typing import Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

Pos = Tuple[int, int]

# --------------------------------------------------------------------------- #
#  vocabulary
# --------------------------------------------------------------------------- #

_SKIP_TYPES = {
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
}

BUILTIN_NAMES: Set[str] = {n for n in dir(builtins) if not n.startswith("_")}
BUILTIN_EXCEPTIONS: Set[str] = {
    n
    for n in BUILTIN_NAMES
    if isinstance(getattr(builtins, n), type) and issubclass(getattr(builtins, n), BaseException)
}
BUILTIN_NAMES -= BUILTIN_EXCEPTIONS
# names that read like keywords in context but are really just names
BUILTIN_NAMES -= {"copyright", "credits", "license", "exit", "quit"}

_EXC_SUFFIX = re.compile(r"(Error|Exception|Warning|Exit|Interrupt|Timeout)$")
_SELF_NAMES = {"self", "cls", "mcs", "mcls", "metacls"}
_CAMEL = re.compile(r"^_*[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*$")
_CONST = re.compile(r"^_*[A-Z][A-Z0-9_]+$")
_ESCAPE = re.compile(
    r"\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|N\{[^}\n]*\}|[0-7]{1,3}|\r?\n|.)",
    re.S,
)
_STR_PREFIX = re.compile(r"^([A-Za-z]*)(\"\"\"|'''|\"|')")

_FSTRING_STARTS = {"FSTRING_START", "TSTRING_START"}
_FSTRING_ENDS = {"FSTRING_END", "TSTRING_END"}


class Item(NamedTuple):
    """One highlighted token: a source range, a kind and its rendered HTML."""

    srow: int
    scol: int
    erow: int
    ecol: int
    kind: str  # name | kw | str | num | op | comment | other
    text: str  # raw source text of the token
    html: str  # rendered HTML (already escaped)

    @property
    def start(self) -> Pos:
        return (self.srow, self.scol)

    @property
    def end(self) -> Pos:
        return (self.erow, self.ecol)


class SpanSet:
    """A set of (start, end) source ranges supporting fast membership tests."""

    def __init__(self, spans: Iterable[Tuple[Pos, Pos]] = ()):
        merged: List[Tuple[Pos, Pos]] = []
        for s, e in sorted(spans):
            if merged and s <= merged[-1][1]:
                if e > merged[-1][1]:
                    merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        self._starts = [s for s, _ in merged]
        self._ends = [e for _, e in merged]

    def __contains__(self, pos: Pos) -> bool:
        i = bisect_right(self._starts, pos) - 1
        return i >= 0 and pos < self._ends[i]


# --------------------------------------------------------------------------- #
#  small html helpers
# --------------------------------------------------------------------------- #


def span(cls: str, text: str) -> str:
    """Wrap *text* (raw) in an ``<i class=cls>`` element."""
    t = escape(text, quote=False)
    return f"<i class={cls}>{t}</i>" if cls else t


def _string_body(text: str, raw: bool) -> str:
    """Escape string *text*, highlighting backslash escape sequences."""
    if raw or "\\" not in text:
        return escape(text, quote=False)
    out: List[str] = []
    last = 0
    for m in _ESCAPE.finditer(text):
        out.append(escape(text[last : m.start()], quote=False))
        out.append(f"<i class=es>{escape(m.group(), quote=False)}</i>")
        last = m.end()
    out.append(escape(text[last:], quote=False))
    return "".join(out)


# --------------------------------------------------------------------------- #
#  f-strings (handled textually so 3.8 - 3.14 behave identically)
# --------------------------------------------------------------------------- #


def _scan_braces(text: str, i: int, end: int) -> int:
    """Return the index of the ``}`` matching the ``{`` just before *i*."""
    depth = 0
    n = end
    while i < n:
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "}":
            if depth == 0:
                return i
            depth -= 1
        elif ch in "'\"":
            # skip a nested string literal
            q = ch * 3 if text.startswith(ch * 3, i) else ch
            i += len(q)
            while i < n and not text.startswith(q, i):
                i += 2 if text[i] == "\\" else 1
            i += len(q) - 1
        i += 1
    return -1


def _split_spec(inner: str) -> int:
    """Index where the conversion/format spec of an f-string field begins."""
    depth = 0
    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch in "'\"":
            q = ch * 3 if inner.startswith(ch * 3, i) else ch
            i += len(q)
            while i < n and not inner.startswith(q, i):
                i += 2 if inner[i] == "\\" else 1
            i += len(q) - 1
        elif depth == 0 and ch == "!" and inner[i + 1 : i + 2] != "=":
            return i
        elif depth == 0 and ch == ":" and inner[i + 1 : i + 2] != "=":
            return i
        i += 1
    return n


def fstring_html(text: str) -> str:
    """Render an f-string (or t-string) token with highlighted replacement fields."""
    m = _STR_PREFIX.match(text)
    if not m:
        return span("s", text)
    prefix, quote = m.groups()
    low = prefix.lower()
    raw = "r" in low
    start = m.end()
    end = len(text) - len(quote)
    if end < start:
        return span("s", text)
    out = [f"<i class=s>{escape(text[:start], quote=False)}"]
    i = lit = start
    while i < end:
        ch = text[i]
        if ch == "{":
            if text[i + 1 : i + 2] == "{":
                i += 2
                continue
            j = _scan_braces(text, i + 1, end)
            if j < 0:
                break
            out.append(_string_body(text[lit:i], raw))
            inner = text[i + 1 : j]
            k = _split_spec(inner)
            out.append("</i><i class=sf>{</i>")
            out.append(fragment(inner[:k]))
            if k < len(inner):
                out.append(f"<i class=sf>{escape(inner[k:], quote=False)}</i>")
            out.append("<i class=sf>}</i><i class=s>")
            i = lit = j + 1
        elif ch == "}" and text[i + 1 : i + 2] == "}":
            i += 2
        elif ch == "\\" and not raw:
            i += 2
        else:
            i += 1
    out.append(_string_body(text[lit:end], raw))
    out.append(escape(text[end:], quote=False))
    out.append("</i>")
    return "".join(out)


# --------------------------------------------------------------------------- #
#  classification
# --------------------------------------------------------------------------- #


class _Tok(NamedTuple):
    kind: str
    string: str
    start: Pos
    end: Pos


def _merge_tokens(tokens: Sequence[tokenize.TokenInfo], lines: Sequence[str]) -> List[_Tok]:
    """Drop layout tokens, glue 3.12+ f-string token runs back into one token."""
    out: List[_Tok] = []
    n = len(tokens)
    i = 0
    tok_name = tokenize.tok_name
    while i < n:
        t = tokens[i]
        if t.type in _SKIP_TYPES:
            i += 1
            continue
        name = tok_name.get(t.type, "")
        if name in _FSTRING_STARTS:
            depth = 1
            j = i + 1
            while j < n and depth:
                nm = tok_name.get(tokens[j].type, "")
                if nm in _FSTRING_STARTS:
                    depth += 1
                elif nm in _FSTRING_ENDS:
                    depth -= 1
                j += 1
            first, last = tokens[i], tokens[j - 1]
            text = _slice(lines, first.start, last.end)
            out.append(_Tok("fstr", text, first.start, last.end))
            i = j
            continue
        if t.type == tokenize.NAME:
            kind = "name"
        elif t.type == tokenize.OP:
            kind = "op"
        elif t.type == tokenize.NUMBER:
            kind = "num"
        elif t.type == tokenize.STRING:
            kind = "fstr" if _is_fstring_token(t.string) else "str"
        elif t.type == tokenize.COMMENT:
            kind = "comment"
        else:
            kind = "other"
        out.append(_Tok(kind, t.string, t.start, t.end))
        i += 1
    return out


def _is_fstring_token(s: str) -> bool:
    m = _STR_PREFIX.match(s)
    return bool(m and set(m.group(1).lower()) & {"f", "t"})


def _slice(lines: Sequence[str], start: Pos, end: Pos) -> str:
    (sr, sc), (er, ec) = start, end
    if sr == er:
        return lines[sr - 1][sc:ec]
    parts = [lines[sr - 1][sc:]]
    parts.extend(lines[sr : er - 1])
    parts.append(lines[er - 1][:ec] if er - 1 < len(lines) else "")
    return "\n".join(parts)


_OPEN = {"(": ")", "[": "]", "{": "}"}
_PLAIN_OPS = set("()[]{},:;.")


def classify(
    tokens: Sequence[tokenize.TokenInfo],
    lines: Sequence[str],
    *,
    soft_keywords: Optional[Set[Pos]] = None,
    type_spans: Optional[SpanSet] = None,
) -> List[Item]:
    """Turn raw tokens into highlighted :class:`Item` objects."""
    toks = _merge_tokens(tokens, lines)
    soft = soft_keywords or set()
    tspans = type_spans
    items: List[Item] = []
    stack: List[str] = []
    prev: Optional[_Tok] = None  # previous significant, non-comment token
    n = len(toks)

    def nxt(i: int) -> Optional[_Tok]:
        j = i + 1
        while j < n and toks[j].kind == "comment":
            j += 1
        return toks[j] if j < n else None

    for i, t in enumerate(toks):
        kind, s = t.kind, t.string
        cls = ""
        html = ""
        text = s
        if kind == "comment":
            cls = "cm"
        elif kind == "str":
            cls = "s"
            m = _STR_PREFIX.match(s)
            raw = bool(m and "r" in m.group(1).lower())
            html = f"<i class=s>{_string_body(s, raw)}</i>"
        elif kind == "fstr":
            html = fstring_html(s)
        elif kind == "num":
            cls = "n"
        elif kind == "op":
            if s in _OPEN:
                stack.append(s)
            elif s in (")", "]", "}") and stack:
                stack.pop()
            if s not in _PLAIN_OPS:
                cls = "o"
        elif kind == "name":
            cls = _name_class(i, t, prev, nxt(i), stack, toks, soft, tspans)
            if keyword.iskeyword(s) or t.start in soft:
                kind = "kw"
        if kind != "comment":
            prev = t
        if not html:
            html = span(cls, text)
        # for multi-line tokens the slice is the authoritative text
        if t.start[0] != t.end[0]:
            text = _slice(lines, t.start, t.end)
        items.append(Item(t.start[0], t.start[1], t.end[0], t.end[1], kind, text, html))
    return items


def _name_class(
    i: int,
    t: _Tok,
    prev: Optional[_Tok],
    nx: Optional[_Tok],
    stack: List[str],
    toks: Sequence[_Tok],
    soft: Set[Pos],
    tspans: Optional[SpanSet],
) -> str:
    s = t.string
    if keyword.iskeyword(s):
        return "kc" if s in ("True", "False", "None") else "k"
    if t.start in soft:
        return "k"
    pv = prev.string if prev is not None else ""
    pk = prev.kind if prev is not None else ""
    is_attr = pv == "." and pk == "op"
    if tspans is not None and t.start in tspans:
        if is_attr and nx is not None and nx.string == "(":
            return "mc"
        return "ty"
    if pk == "name" and pv == "def":
        return "fd"
    if pk == "name" and pv == "class":
        return "cd"
    call = nx is not None and nx.kind == "op" and nx.string == "("
    if is_attr:
        if call:
            return "cn" if _CAMEL.match(s) else "mc"
        return "at"
    if s in _SELF_NAMES:
        return "sl"
    if s in BUILTIN_EXCEPTIONS or (_EXC_SUFFIX.search(s) and s[:1].isupper()):
        return "ex"
    if (
        nx is not None
        and nx.kind == "op"
        and nx.string == "="
        and stack
        and stack[-1] == "("
        and pv in ("(", ",")
    ):
        return "kw"
    if s in BUILTIN_NAMES and not (nx is not None and nx.string == ":" and pv in ("(", ",")):
        return "b"
    if call:
        return "cn" if _CAMEL.match(s) else "fc"
    if _CONST.match(s):
        return "cs"
    if _CAMEL.match(s):
        return "cn"
    if s.startswith("__") and s.endswith("__") and len(s) > 4:
        return "dn"
    return ""


# --------------------------------------------------------------------------- #
#  fragments (types, defaults, expressions rendered outside the main flow)
# --------------------------------------------------------------------------- #


def fragment(text: str, *, as_type: bool = False) -> str:
    """Highlight a standalone snippet of Python (single expression or statement)."""
    text = text.strip("\n")
    if not text.strip():
        return escape(text, quote=False)
    lines = text.split("\n")
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text + "\n").readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return escape(text, quote=False)
    spans = SpanSet([((1, 0), (len(lines) + 1, 0))]) if as_type else None
    items = classify(toks, lines, type_spans=spans)
    return render_items(items, lines, (1, 0), (len(lines), len(lines[-1])))


def render_items(
    items: Sequence[Item],
    lines: Sequence[str],
    start: Pos,
    end: Pos,
    dedent: int = 0,
) -> str:
    """Render *items* between *start* and *end*, keeping inter-token whitespace."""
    out: List[str] = []
    pr, pc = start
    for it in items:
        if it.start < start:
            continue
        if it.end > end:
            break
        gap = _slice(lines, (pr, pc), it.start)
        if gap:
            out.append(_dedent_gap(escape(gap, quote=False), dedent))
        out.append(it.html)
        pr, pc = it.end
    gap = _slice(lines, (pr, pc), end)
    if gap.strip():
        out.append(escape(gap, quote=False))
    return "".join(out)


_TRAILING_WS = re.compile(r"[ \t]+\n")


def _dedent_gap(text: str, dedent: int) -> str:
    if "\n" not in text:
        return text
    text = _TRAILING_WS.sub("\n", text)
    if not dedent:
        return text
    head, *rest = text.split("\n")
    out = [head]
    for seg in rest:
        k = 0
        while k < dedent and k < len(seg) and seg[k] in " \t":
            k += 1
        out.append(seg[k:])
    return "\n".join(out)
