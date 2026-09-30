"""Docstring prose -> HTML."""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from ..docstrings import Block, Section, inline_html
from ..highlight import fragment
from .icons import icon
from .util import esc

Linker = Optional[Callable[[str], Optional[str]]]

_SECTION_ICONS = {
    "example": "terminal",
    "note": "info",
    "warning": "alert",
    "seealso": "link",
    "todo": "flag",
    "refs": "book",
    "other": "file",
}


def highlight_code(text: str) -> str:
    """Highlight a code snippet line by line (robust against partial code)."""
    try:
        return fragment(text)
    except Exception:
        return esc(text)


def doctest_html(text: str) -> str:
    """Render a doctest session: prompts dimmed, code highlighted, output plain."""
    out: List[str] = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        pad = line[: len(line) - len(stripped)]
        if stripped.startswith(">>>") or stripped == "..." or stripped.startswith("... "):
            prompt, rest = (">>>", stripped[3:]) if stripped.startswith(">>>") else ("...", stripped[3:])
            code = rest[1:] if rest.startswith(" ") else rest
            out.append(f'<span class="dt-l">{esc(pad)}<b class="dt-p">{prompt}</b> {highlight_code(code)}</span>')
        else:
            out.append(f'<span class="dt-l dt-o">{esc(line) or "&nbsp;"}</span>')
    return "".join(out)


def blocks_html(blocks: Sequence[Block], link: Linker = None) -> str:
    """Render docstring blocks."""
    out: List[str] = []
    for b in blocks:
        if b.kind == "p":
            out.append(f"<p>{inline_html(b.text, link)}</p>")
        elif b.kind in ("ul", "ol"):
            items = "".join(f"<li>{inline_html(i, link)}</li>" for i in b.items)
            out.append(f"<{b.kind}>{items}</{b.kind}>")
        elif b.kind == "code":
            out.append(f'<pre class="snip"><code>{highlight_code(b.text)}</code></pre>')
        elif b.kind == "doctest":
            out.append(f'<pre class="snip doctest"><code>{doctest_html(b.text)}</code></pre>')
        else:  # admonition
            title = esc(b.title or b.kind.title())
            inner = blocks_html(b.children, link)
            ic = _SECTION_ICONS.get(b.kind, "info")
            out.append(f'<div class="adm adm-{b.kind}"><div class="adm-t">{icon(ic)}<b>{title}</b></div>{inner}</div>')
    return "".join(out)


def section_html(sec: Section, link: Linker = None) -> str:
    if sec.kind == "plain":
        return f'<div class="prose">{blocks_html(sec.blocks, link)}</div>'
    ic = _SECTION_ICONS.get(sec.kind, "file")
    return (
        f'<div class="adm adm-{sec.kind}"><div class="adm-t">{icon(ic)}<b>{esc(sec.title)}</b></div>'
        f"{blocks_html(sec.blocks, link)}</div>"
    )
