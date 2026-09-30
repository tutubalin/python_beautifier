"""Docstring parsing.

Understands the four docstring dialects found in the wild - **Google**,
**NumPy**, **Sphinx/reST** field lists and **Epytext** - and turns them into
one structured :class:`Doc`: a summary, free-text blocks (paragraphs, lists,
code, admonitions) and typed fields (parameters, returns, raises, ...).

The parser is deliberately forgiving.  Real-world docstrings are messy, so
anything it does not recognise simply stays in the free-text description.
"""
from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  data model
# --------------------------------------------------------------------------- #


@dataclass
class Field:
    """A named, typed, described thing: a parameter, attribute, exception..."""

    name: str = ""
    type: str = ""
    desc: str = ""
    optional: bool = False
    default: str = ""


@dataclass
class Block:
    """A piece of free text.

    ``kind`` is one of ``p`` (paragraph), ``ul``/``ol`` (lists), ``code``,
    ``doctest`` or an admonition name such as ``note`` or ``warning``.
    """

    kind: str
    text: str = ""
    items: List[str] = field(default_factory=list)
    title: str = ""
    children: List["Block"] = field(default_factory=list)


@dataclass
class Section:
    """A named free-text section (Examples, Notes, See Also, ...)."""

    kind: str  # example | note | warning | seealso | todo | refs | other | plain (untitled text)
    title: str
    blocks: List[Block]


@dataclass
class Doc:
    raw: str = ""
    style: str = "plain"
    summary: str = ""
    blocks: List[Block] = field(default_factory=list)  # description, summary included
    params: List[Field] = field(default_factory=list)
    kwparams: List[Field] = field(default_factory=list)
    returns: List[Field] = field(default_factory=list)
    yields: List[Field] = field(default_factory=list)
    raises: List[Field] = field(default_factory=list)
    warns: List[Field] = field(default_factory=list)
    attributes: List[Field] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.raw.strip())

    @property
    def description_blocks(self) -> List[Block]:
        """Free text *after* the summary paragraph."""
        return self.blocks[1:] if self.blocks and self.blocks[0].kind == "p" else self.blocks

    def param(self, name: str) -> Optional[Field]:
        bare = name.lstrip("*")
        for f in self.params + self.kwparams:
            if f.name.lstrip("*") == bare:
                return f
        return None

    @property
    def has_details(self) -> bool:
        return bool(
            self.description_blocks
            or self.params
            or self.kwparams
            or self.returns
            or self.yields
            or self.raises
            or self.attributes
            or self.sections
        )


# --------------------------------------------------------------------------- #
#  vocabulary
# --------------------------------------------------------------------------- #

_PARAM_TITLES = {"args", "arguments", "parameters", "params", "parameter", "arg"}
_KWPARAM_TITLES = {
    "keyword args",
    "keyword arguments",
    "keyword parameters",
    "kwargs",
    "other parameters",
    "other params",
    "options",
}
_RETURN_TITLES = {"returns", "return"}
_YIELD_TITLES = {"yields", "yield", "receives"}
_RAISE_TITLES = {"raises", "raise", "exceptions", "except", "exception"}
_WARN_TITLES = {"warns", "warn"}
_ATTR_TITLES = {"attributes", "attribute", "members", "fields", "variables"}
_EXAMPLE_TITLES = {"example", "examples", "usage", "doctest", "doctests"}
_NOTE_TITLES = {"note", "notes", "remark", "remarks", "hint", "tip", "important", "caution"}
_WARNING_TITLES = {"warning", "warnings", "danger", "attention", "error", "deprecated"}
_SEEALSO_TITLES = {"see also", "seealso", "see"}
_TODO_TITLES = {"todo", "todos", "to do"}
_REF_TITLES = {"references", "reference", "bibliography", "sources"}
_OTHER_TITLES = {"methods", "returns", "version", "versionadded", "since", "license", "author", "authors"}

_ALL_TITLES = (
    _PARAM_TITLES
    | _KWPARAM_TITLES
    | _RETURN_TITLES
    | _YIELD_TITLES
    | _RAISE_TITLES
    | _WARN_TITLES
    | _ATTR_TITLES
    | _EXAMPLE_TITLES
    | _NOTE_TITLES
    | _WARNING_TITLES
    | _SEEALSO_TITLES
    | _TODO_TITLES
    | _REF_TITLES
    | _OTHER_TITLES
)

_GOOGLE_HEADER = re.compile(r"^(?P<indent>\s*)(?P<title>[A-Za-z][A-Za-z ]{1,24}?)\s*:\s*$")
_GOOGLE_INLINE = re.compile(r"^(?P<indent>\s*)(?P<title>Returns?|Yields?|Raises?)\s*:\s+(?P<rest>\S.*)$")
_NUMPY_UNDERLINE = re.compile(r"^\s*(-{3,}|={3,})\s*$")
_SPHINX_FIELD = re.compile(r"^(?P<indent>\s*):(?P<tag>\w+)(?:\s+(?P<arg>[^:]*?))?\s*:(?:\s+(?P<rest>.*))?$")
_EPY_FIELD = re.compile(r"^(?P<indent>\s*)@(?P<tag>\w+)(?:\s+(?P<arg>[^:]*?))?\s*:(?:\s+(?P<rest>.*))?$")
_BULLET = re.compile(r"^(?P<indent>\s*)(?P<mark>[-*+\u2022\u25cf\u25aa])\s+(?P<text>.*)$")
_NUMBERED = re.compile(r"^(?P<indent>\s*)(?P<mark>\d{1,3}[.)])\s+(?P<text>.*)$")
_DIRECTIVE = re.compile(r"^(?P<indent>\s*)\.\.\s+(?P<name>[\w:-]+)::\s*(?P<arg>.*)$")

_ADMONITIONS = {
    "note": "note",
    "warning": "warning",
    "danger": "warning",
    "caution": "warning",
    "attention": "warning",
    "important": "note",
    "tip": "note",
    "hint": "note",
    "seealso": "seealso",
    "todo": "todo",
    "deprecated": "warning",
    "versionadded": "note",
    "versionchanged": "note",
    "versionremoved": "warning",
}


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _dedent(lines: Sequence[str]) -> List[str]:
    body = [ln for ln in lines if ln.strip()]
    if not body:
        return [""] * len(lines)
    k = min(_indent(ln) for ln in body)
    return [ln[k:] if ln.strip() else "" for ln in lines]


def _trim_blank(lines: List[str]) -> List[str]:
    a, b = 0, len(lines)
    while a < b and not lines[a].strip():
        a += 1
    while b > a and not lines[b - 1].strip():
        b -= 1
    return lines[a:b]


def _join_lines(lines: Sequence[str]) -> str:
    """Join wrapped prose lines, keeping paragraph breaks as blank lines."""
    paras: List[List[str]] = [[]]
    for ln in lines:
        if ln.strip():
            paras[-1].append(ln.strip())
        elif paras[-1]:
            paras.append([])
    return "\n\n".join(" ".join(p) for p in paras if p)


def _split_balanced(text: str) -> Tuple[str, str]:
    """If *text* starts with ``(...)`` return (inner, rest) honouring nesting."""
    if not text.startswith("("):
        return "", text
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return text[1:i].strip(), text[i + 1 :]
    return "", text


_NAME_RE = re.compile(r"^(\*{0,2}[A-Za-z_][\w.]*)")


def _parse_item_header(line: str) -> Optional[Tuple[str, str, str]]:
    """Parse ``name (type): description`` -> (name, type, description)."""
    m = _NAME_RE.match(line)
    if not m:
        return None
    name = m.group(1)
    rest = line[m.end() :].lstrip()
    typ = ""
    if rest.startswith("("):
        typ, rest = _split_balanced(rest)
        rest = rest.lstrip()
    if rest.startswith(":"):
        return name, typ, rest[1:].strip()
    if not rest and typ:
        return name, typ, ""
    return None


def _split_optional(typ: str) -> Tuple[str, bool, str]:
    """Split a type like ``int, optional`` into (type, optional, default)."""
    optional = False
    default = ""
    parts = _split_top(typ, ",")
    keep: List[str] = []
    for p in parts:
        ps = p.strip()
        low = ps.lower()
        if low == "optional":
            optional = True
        elif low.startswith("default"):
            default = re.sub(r"^default\s*[:=]?\s*", "", ps, flags=re.I)
            optional = True
        else:
            keep.append(ps)
    return ", ".join(k for k in keep if k), optional, default


def _split_top(text: str, sep: str) -> List[str]:
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


# --------------------------------------------------------------------------- #
#  free-text blocks
# --------------------------------------------------------------------------- #


def _is_code_paragraph(lines: Sequence[str]) -> bool:
    """A multi-line paragraph that is valid Python is code (think ``Usage:`` blocks), not prose."""
    if len(lines) < 2:
        return False
    try:
        tree = ast.parse("\n".join(_dedent(lines)))
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    trivial = (ast.Name, ast.Constant, ast.Attribute)
    return any(not (isinstance(st, ast.Expr) and isinstance(st.value, trivial)) for st in tree.body)


def parse_blocks(lines: Sequence[str]) -> List[Block]:
    """Parse prose into paragraphs, lists, code/doctest blocks and admonitions."""
    lines = list(lines)
    blocks: List[Block] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # fenced code (markdown)
        if line.strip().startswith("```"):
            j = i + 1
            while j < n and not lines[j].strip().startswith("```"):
                j += 1
            blocks.append(Block("code", "\n".join(_dedent(lines[i + 1 : j]))))
            i = j + 1
            continue

        # doctest
        if line.lstrip().startswith(">>>"):
            j = i
            while j < n and lines[j].strip():
                j += 1
            blocks.append(Block("doctest", "\n".join(_dedent(lines[i:j]))))
            i = j
            continue

        # reST directive
        dm = _DIRECTIVE.match(line)
        if dm:
            name, arg, ind = dm.group("name"), dm.group("arg"), len(dm.group("indent"))
            j = i + 1
            body: List[str] = []
            while j < n and (not lines[j].strip() or _indent(lines[j]) > ind):
                body.append(lines[j])
                j += 1
            body = _trim_blank(_dedent(body))
            short = name.split(":")[-1].lower()
            if short in ("code-block", "code", "sourcecode", "highlight"):
                # options such as ":linenos:" precede the code
                code = [b for b in body if not re.match(r"^\s*:\w[\w-]*:", b)]
                blocks.append(Block("code", "\n".join(_trim_blank(code))))
            elif short in _ADMONITIONS:
                inner = parse_blocks(body)
                title = arg.strip()
                if short.startswith("version"):
                    title = f"{short.replace('version', 'version ')} {arg}".strip()
                    inner = inner or []
                    blocks.append(Block(_ADMONITIONS[short], title=title, children=inner))
                else:
                    if arg.strip() and not inner:
                        inner = [Block("p", arg.strip())]
                    elif arg.strip():
                        inner.insert(0, Block("p", arg.strip()))
                    blocks.append(Block(_ADMONITIONS[short], title=short.capitalize(), children=inner))
            else:
                text = " ".join(x.strip() for x in body if x.strip())
                blocks.append(Block("p", f"{arg} {text}".strip()))
            i = j
            continue

        # lists
        lm = _BULLET.match(line) or _NUMBERED.match(line)
        if lm:
            ordered = bool(_NUMBERED.match(line)) and not _BULLET.match(line)
            base = len(lm.group("indent"))
            items: List[List[str]] = []
            j = i
            while j < n:
                ln = lines[j]
                mm = (_NUMBERED if ordered else _BULLET).match(ln)
                if mm and len(mm.group("indent")) == base:
                    items.append([mm.group("text")])
                    j += 1
                elif ln.strip() and _indent(ln) > base and items:
                    items[-1].append(ln.strip())
                    j += 1
                elif not ln.strip() and j + 1 < n:
                    nxt = lines[j + 1]
                    mm2 = (_NUMBERED if ordered else _BULLET).match(nxt)
                    if nxt.strip() and ((mm2 and len(mm2.group("indent")) == base) or _indent(nxt) > base):
                        j += 1
                    else:
                        break
                else:
                    break
            blocks.append(Block("ol" if ordered else "ul", items=[" ".join(it) for it in items]))
            i = j
            continue

        # paragraph (possibly introducing a literal block with "::")
        j = i
        para: List[str] = []
        while j < n and lines[j].strip():
            if para and (
                lines[j].lstrip().startswith(">>>")
                or _BULLET.match(lines[j])
                or _DIRECTIVE.match(lines[j])
            ) and not _NUMBERED.match(lines[j]):
                break
            para.append(lines[j].strip())
            j += 1
        if _is_code_paragraph(lines[i:j]):
            blocks.append(Block("code", "\n".join(_dedent(lines[i:j]))))
            i = j
            continue
        text = " ".join(para)
        if text.endswith("::"):
            k = j
            while k < n and not lines[k].strip():
                k += 1
            if k < n and _indent(lines[k]) > _indent(lines[i]):
                ind = _indent(lines[i])
                lit: List[str] = []
                while k < n and (not lines[k].strip() or _indent(lines[k]) > ind):
                    lit.append(lines[k])
                    k += 1
                text = text[:-2].rstrip() + (":" if text[:-2].strip() else "")
                if text:
                    blocks.append(Block("p", text))
                blocks.append(Block("code", "\n".join(_trim_blank(_dedent(lit)))))
                i = k
                continue
        blocks.append(Block("p", text))
        i = j
    return blocks


# --------------------------------------------------------------------------- #
#  field parsing
# --------------------------------------------------------------------------- #


def _collect_items(lines: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """Group *lines* into (header, continuation-lines) using indentation."""
    lines = _trim_blank(list(lines))
    if not lines:
        return []
    base = min(_indent(ln) for ln in lines if ln.strip())
    items: List[Tuple[str, List[str]]] = []
    for ln in lines:
        if ln.strip() and _indent(ln) == base:
            items.append((ln.strip(), []))
        elif items:
            items[-1][1].append(ln)
        # stray leading continuation lines are dropped
    return items


def _item_desc(first: str, cont: Sequence[str]) -> str:
    body = _dedent(list(cont))
    text = _join_lines([first] + body) if first else _join_lines(body)
    return text


def _google_fields(lines: Sequence[str], *, typed_names: bool = True) -> List[Field]:
    out: List[Field] = []
    for head, cont in _collect_items(lines):
        parsed = _parse_item_header(head)
        if parsed is None:
            if out and not cont:
                # a wrapped line at item level: glue it to the previous entry
                out[-1].desc = (out[-1].desc + " " + head).strip()
                continue
            out.append(Field(name="", desc=_item_desc(head, cont)))
            continue
        name, typ, rest = parsed
        typ, optional, default = _split_optional(typ)
        out.append(Field(name, typ, _item_desc(rest, cont), optional, default))
    return out


def _google_returns(lines: Sequence[str]) -> List[Field]:
    lines = _trim_blank(list(lines))
    if not lines:
        return []
    text = _join_lines(_dedent(lines))
    first = lines[0].strip()
    m = re.match(r"^(?P<type>[^:\s][^:]*?)\s*:\s+(?P<rest>.*)$", first)
    if m and _looks_like_type(m.group("type")):
        desc = _item_desc(m.group("rest"), _dedent(lines[1:]))
        return [Field("", m.group("type").strip(), desc)]
    if first.endswith(":") and _looks_like_type(first[:-1]):
        return [Field("", first[:-1].strip(), _item_desc("", _dedent(lines[1:])))]
    return [Field("", "", text)]


def _looks_like_type(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > 60:
        return False
    if re.search(r"[\[\]|,()]", text):
        return bool(re.match(r"^[\w.\[\], |()'\"~*:]+$", text)) and " " not in text.split("[")[0]
    return bool(re.match(r"^[A-Za-z_][\w.]*(\s+(or|of)\s+[A-Za-z_][\w.]*)*$", text)) and len(text.split()) <= 4


def _numpy_fields(lines: Sequence[str]) -> List[Field]:
    out: List[Field] = []
    for head, cont in _collect_items(lines):
        desc = _item_desc("", cont)
        if " : " in head or re.match(r"^[\w*,\s.]+:\s", head) or head.endswith(" :"):
            names, _, typ = head.partition(":")
            typ = typ.strip()
        else:
            names, typ = head, ""
        typ, optional, default = _split_optional(typ)
        for nm in [x.strip() for x in names.split(",") if x.strip()]:
            out.append(Field(nm, typ, desc, optional, default))
    return out


def _numpy_returns(lines: Sequence[str]) -> List[Field]:
    out: List[Field] = []
    for head, cont in _collect_items(lines):
        desc = _item_desc("", cont)
        m = re.match(r"^(?P<name>[A-Za-z_]\w*)\s+:\s+(?P<type>.+)$", head)
        if m:
            out.append(Field(m.group("name"), m.group("type").strip(), desc))
        else:
            out.append(Field("", head, desc))
    return out


def _raise_fields(lines: Sequence[str], numpy: bool = False) -> List[Field]:
    out: List[Field] = []
    for head, cont in _collect_items(lines):
        if numpy:
            out.append(Field(head.rstrip(":"), "", _item_desc("", cont)))
            continue
        m = re.match(r"^(?P<exc>[\w.]+(?:\s*(?:,|or)\s*[\w.]+)*)\s*:\s*(?P<rest>.*)$", head)
        if m:
            out.append(Field(m.group("exc"), "", _item_desc(m.group("rest"), cont)))
        else:
            out.append(Field("", "", _item_desc(head, cont)))
    return out


# --------------------------------------------------------------------------- #
#  the parser
# --------------------------------------------------------------------------- #


def parse(raw: Optional[str]) -> Doc:
    """Parse a docstring into a :class:`Doc`."""
    if not raw or not raw.strip():
        return Doc(raw="")
    text = inspect.cleandoc(raw)
    lines = text.split("\n")
    doc = Doc(raw=text)

    desc_lines, sections, style = _split_sections(lines)
    doc.style = style

    # free-standing field lists (Sphinx / Epytext) inside the description
    desc_lines = _extract_field_lists(desc_lines, doc)

    doc.blocks = parse_blocks(desc_lines)
    if doc.blocks and doc.blocks[0].kind == "p":
        doc.summary = doc.blocks[0].text
    elif doc.blocks:
        doc.summary = doc.blocks[0].text or ""

    for title, body, numpy in sections:
        _apply_section(doc, title, body, numpy)
    if style == "plain" and (doc.params or doc.returns or doc.raises or doc.attributes):
        doc.style = "sphinx"
    return doc


def _split_sections(lines: List[str]) -> Tuple[List[str], List[Tuple[str, List[str], bool]], str]:
    """Split *lines* into (description, [(title, body, is_numpy)], style)."""
    n = len(lines)
    marks: List[Tuple[int, int, str, bool, str]] = []  # (start, body_start, title, numpy, inline)
    i = 0
    in_fence = False
    while i < n:
        ln = lines[i]
        if ln.strip().startswith("```"):
            in_fence = not in_fence
        if in_fence:
            i += 1
            continue
        # NumPy: Title / -----
        if (
            i + 1 < n
            and ln.strip()
            and _NUMPY_UNDERLINE.match(lines[i + 1])
            and ln.strip().lower() in _ALL_TITLES
            and len(_NUMPY_UNDERLINE.match(lines[i + 1]).group(1)) >= 3  # type: ignore[union-attr]
        ):
            marks.append((i, i + 2, ln.strip(), True, ""))
            i += 2
            continue
        m = _GOOGLE_HEADER.match(ln)
        if m and m.group("title").strip().lower() in _ALL_TITLES and _indent(ln) == 0:
            marks.append((i, i + 1, m.group("title").strip(), False, ""))
        else:
            m2 = _GOOGLE_INLINE.match(ln)
            if m2 and _indent(ln) == 0:
                marks.append((i, i + 1, m2.group("title"), False, m2.group("rest")))
        i += 1

    if not marks:
        return lines, [], "plain"

    style = "numpy" if any(mk[3] for mk in marks) else "google"
    desc = lines[: marks[0][0]]
    sections = []
    for k, (start, body_start, title, numpy, inline) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else n
        body = lines[body_start:end]
        trailing: List[str] = []
        if not numpy:
            body, trailing = _google_body(body)
        if inline:
            body = [inline] + body
        sections.append((title, body, numpy))
        if any(t.strip() for t in trailing):
            sections.append(("", trailing, False))  # prose that follows the section
    return desc, sections, style


def _google_body(body: List[str]) -> Tuple[List[str], List[str]]:
    """Split a Google section body from the text after it.

    A body is indented; it ends where the text returns to the header's column.
    An unindented body (``Usage:`` followed by flush-left code) ends at the first blank line.
    """
    first = next((k for k, ln in enumerate(body) if ln.strip()), None)
    if first is None:
        return body, []
    flush = _indent(body[first]) == 0
    for k in range(first + 1, len(body)):
        ln = body[k]
        if flush and not ln.strip():
            return body[:k], body[k:]
        if not flush and ln.strip() and _indent(ln) == 0:
            return body[:k], body[k:]
    return body, []


def _extract_field_lists(lines: List[str], doc: Doc) -> List[str]:
    """Pull Sphinx (``:param x:``) and Epytext (``@param x:``) fields out of *lines*."""
    out: List[str] = []
    i, n = 0, len(lines)
    types: Dict[str, str] = {}
    rtype = ""
    ytype = ""
    found = False
    while i < n:
        ln = lines[i]
        m = _SPHINX_FIELD.match(ln) or _EPY_FIELD.match(ln)
        if not m or m.group("tag").lower() not in _FIELD_TAGS:
            out.append(ln)
            i += 1
            continue
        found = True
        tag = m.group("tag").lower()
        arg = (m.group("arg") or "").strip()
        rest = (m.group("rest") or "").strip()
        ind = len(m.group("indent"))
        j = i + 1
        cont: List[str] = []
        while j < n and (not lines[j].strip() or _indent(lines[j]) > ind):
            if not lines[j].strip() and (j + 1 >= n or _indent(lines[j + 1]) <= ind):
                break
            cont.append(lines[j])
            j += 1
        desc = _item_desc(rest, _dedent(cont))
        kind = _FIELD_TAGS[tag]
        if kind in ("param", "kwparam"):
            parts = arg.rsplit(None, 1)
            name = parts[-1] if parts else ""
            typ = parts[0] if len(parts) == 2 else ""
            (doc.kwparams if kind == "kwparam" else doc.params).append(Field(name, typ, desc))
        elif kind == "type":
            types[arg] = desc
        elif kind == "returns":
            doc.returns.append(Field("", "", desc))
        elif kind == "rtype":
            rtype = desc
        elif kind == "yields":
            doc.yields.append(Field("", "", desc))
        elif kind == "ytype":
            ytype = desc
        elif kind == "raises":
            doc.raises.append(Field(arg, "", desc))
        elif kind == "attr":
            doc.attributes.append(Field(arg.split()[-1] if arg else "", "", desc))
        elif kind == "vartype":
            types[arg] = desc
        i = j
    if found:
        for f in doc.params + doc.kwparams + doc.attributes:
            if not f.type and f.name in types:
                f.type = types[f.name]
        if rtype:
            if doc.returns:
                doc.returns[0].type = rtype
            else:
                doc.returns.append(Field("", rtype, ""))
        if ytype:
            if doc.yields:
                doc.yields[0].type = ytype
            else:
                doc.yields.append(Field("", ytype, ""))
        doc.style = "sphinx"
    return out


_FIELD_TAGS = {
    "param": "param",
    "parameter": "param",
    "arg": "param",
    "argument": "param",
    "key": "kwparam",
    "keyword": "kwparam",
    "type": "type",
    "returns": "returns",
    "return": "returns",
    "rtype": "rtype",
    "yields": "yields",
    "yield": "yields",
    "ytype": "ytype",
    "raises": "raises",
    "raise": "raises",
    "except": "raises",
    "exception": "raises",
    "ivar": "attr",
    "cvar": "attr",
    "var": "attr",
    "vartype": "vartype",
}


def _apply_section(doc: Doc, title: str, body: List[str], numpy: bool) -> None:
    if not title:  # free text that follows a section
        doc.sections.append(Section("plain", "", parse_blocks(_dedent(_trim_blank(body)))))
        return
    key = re.sub(r"\s+", " ", title.strip().lower())
    body = _trim_blank(_dedent(body)) if numpy else _trim_blank(body)
    if key in _PARAM_TITLES:
        doc.params.extend(_numpy_fields(body) if numpy else _google_fields(_dedent(body)))
    elif key in _KWPARAM_TITLES:
        doc.kwparams.extend(_numpy_fields(body) if numpy else _google_fields(_dedent(body)))
    elif key in _RETURN_TITLES:
        doc.returns.extend(_numpy_returns(body) if numpy else _google_returns(body))
    elif key in _YIELD_TITLES:
        doc.yields.extend(_numpy_returns(body) if numpy else _google_returns(body))
    elif key in _RAISE_TITLES:
        doc.raises.extend(_raise_fields(_dedent(body), numpy))
    elif key in _WARN_TITLES:
        doc.warns.extend(_raise_fields(_dedent(body), numpy))
    elif key in _ATTR_TITLES:
        doc.attributes.extend(_numpy_fields(body) if numpy else _google_fields(_dedent(body)))
    else:
        if key in _EXAMPLE_TITLES:
            kind = "example"
        elif key in _NOTE_TITLES:
            kind = "note"
        elif key in _WARNING_TITLES:
            kind = "warning"
        elif key in _SEEALSO_TITLES:
            kind = "seealso"
        elif key in _TODO_TITLES:
            kind = "todo"
        elif key in _REF_TITLES:
            kind = "refs"
        else:
            kind = "other"
        dedented = _dedent(body)
        if kind == "example":
            blocks = _example_blocks(dedented)
        else:
            blocks = parse_blocks(dedented)
        doc.sections.append(Section(kind, title.strip().title() if title.islower() else title.strip(), blocks))


def _example_blocks(lines: List[str]) -> List[Block]:
    """Examples: doctest runs stay doctests, other indented text is code."""
    text = "\n".join(lines)
    if ">>>" in text:
        return parse_blocks(lines)
    blocks = parse_blocks(lines)
    # prose-only examples that look like code are better shown as code
    if len(blocks) == 1 and blocks[0].kind == "p" and ("\n" in text or "(" in text):
        return [Block("code", text)]
    return blocks


# --------------------------------------------------------------------------- #
#  inline markup
# --------------------------------------------------------------------------- #

_ROLE = re.compile(
    r":(?:py:|c:|cpp:)?(?:class|func|meth|mod|attr|exc|data|const|obj|ref|term|doc|method|property|type|envvar|option|program|pep|rfc|file|command|samp|kbd)"
    r":`(?P<t>[^`]+)`"
)
_CODE2 = re.compile(r"``(?P<t>[^`]+)``")
_CODE1 = re.compile(r"(?<![\w`])`(?P<t>[^`\n]+)`(?![\w`])")
_BOLD = re.compile(r"(?<![\w*])\*\*(?P<t>[^*\s][^*\n]*?)\*\*(?![\w*])")
_EMPH = re.compile(r"(?<![\w*\\])\*(?P<t>[^*\s][^*\n]*?)\*(?![\w*])")
_URL = re.compile(r"(?P<u>https?://[^\s<>\"')\]]+[^\s<>\"'.,;:)\]])")
_TITLE_REF = re.compile(r"(?<![\w`])`(?P<t>[^`<>\n]+?)(?:\s*<(?P<href>[^>]+)>)?`_{1,2}")


def inline_html(text: str, link: Optional[Callable[[str], Optional[str]]] = None) -> str:
    """Convert docstring inline markup to safe HTML.

    *link* may map a dotted name to an in-page anchor (``#d-foo``); when it
    returns one, the name becomes a hyperlink.
    """
    from html import escape

    tokens: List[str] = []

    def stash(html: str) -> str:
        tokens.append(html)
        return f"\x00{len(tokens) - 1}\x00"

    def code(target: str, title: Optional[str] = None) -> str:
        shown = title if title else target.lstrip("~!")
        if not title and target.startswith("~"):
            shown = shown.split(".")[-1]
        lookup = target.lstrip("~!").rstrip("()")
        href = link(lookup) if link else None
        body = escape(shown, quote=False)
        if href:
            return stash(f'<a class="xref" href="{escape(href)}"><code class="ic">{body}</code></a>')
        return stash(f'<code class="ic">{body}</code>')

    def role(m: "re.Match[str]") -> str:
        t = m.group("t")
        mm = re.match(r"^(?P<title>.+?)\s*<(?P<target>[^<>]+)>$", t)
        if mm:
            return code(mm.group("target"), mm.group("title"))
        return code(t)

    s = text
    s = _ROLE.sub(role, s)
    s = _CODE2.sub(lambda m: stash(f'<code class="ic">{escape(m.group("t"), quote=False)}</code>'), s)
    s = _TITLE_REF.sub(
        lambda m: stash(
            f'<a href="{escape(m.group("href") or "#")}" rel="noopener">{escape(m.group("t"), quote=False)}</a>'
        ),
        s,
    )
    s = _CODE1.sub(lambda m: code(m.group("t")), s)
    s = _URL.sub(
        lambda m: stash(
            f'<a href="{escape(m.group("u"))}" rel="noopener">{escape(m.group("u"), quote=False)}</a>'
        ),
        s,
    )
    s = escape(s, quote=False)
    s = _BOLD.sub(lambda m: f"<strong>{m.group('t')}</strong>", s)
    s = _EMPH.sub(lambda m: f"<em>{m.group('t')}</em>", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m.group(1))], s)


def first_sentence(text: str, limit: int = 180) -> str:
    """The first sentence of *text*, capped at *limit* characters."""
    text = " ".join(text.split())
    m = re.search(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(])", text)
    if m and m.start() <= limit:
        return text[: m.start()]
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "\u2026"
