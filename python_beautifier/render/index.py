"""The indexing guide: one cell per axis under a statement that slices an array."""
from __future__ import annotations

from typing import Sequence

from ..highlight import fragment
from ..indexing import Axis, Guide, Seg
from .icons import icon
from .util import attr, esc

MAX_ROWS = 6  # subscripts explained per statement; the rest are still marked inline

_VERB = {"read": "index", "write": "assign", "delete": "delete"}


def _sentence(segs: Sequence[Seg]) -> str:
    return "".join(f"<code>{esc(text)}</code>" if is_code else esc(text) for text, is_code in segs)


def _cell(axis: Axis, ident: str) -> str:
    classes = f"ix-c ik-{axis.kind}"
    term = f'<code class="ix-t">{fragment(axis.term)}</code>' if axis.term else ""
    return (
        f'<span class="{classes}" data-ix="{ident}" title="{attr(axis.title)}">'
        f'<span class="ix-k"><small>{esc(axis.label)}</small>{term}</span>'
        f'<span class="ix-m">{_sentence(axis.meaning)}</span></span>'
    )


def _base(text: str, limit: int = 26) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def strip(guides: Sequence[Guide]) -> str:
    """HTML for the guides of one statement ('' when there is nothing to explain)."""
    if not guides:
        return ""
    rows = []
    for g in guides[:MAX_ROWS]:
        head = (
            f'<span class="ix-h">{icon("brackets")}<b>{_VERB[g.mode]}</b>'
            f'<code title="{attr(g.key)}">{esc(_base(g.base))}[\u2026]</code></span>'
        )
        rows.append(f'<div class="ix-r">{head}{"".join(_cell(a, i) for a, i in zip(g.cells, g.ids))}</div>')
    if len(guides) > MAX_ROWS:
        rows.append(f'<div class="ix-more">+ {len(guides) - MAX_ROWS} more subscripts, marked in the code above</div>')
    return f'<div class="ix" role="note" aria-label="How this statement indexes its arrays">{"".join(rows)}</div>'
