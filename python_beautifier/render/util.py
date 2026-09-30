"""Small shared helpers for the renderer."""
from __future__ import annotations

import os
from html import escape
from typing import Iterable


def esc(text: str) -> str:
    """Escape *text* for use as HTML element content."""
    return escape(text, quote=False)


def attr(text: str) -> str:
    """Escape *text* for use inside a double-quoted attribute."""
    return escape(text, quote=True)


def join_classes(*names: str) -> str:
    return " ".join(n for n in names if n)


def plural(n: int, one: str, many: str = "") -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def chips(items: Iterable[str], cls: str = "chip") -> str:
    return "".join(f'<span class="{cls}">{i}</span>' for i in items)


def file_label(filename: str) -> str:
    """The name to show for *filename*; pseudo names such as ``<string>`` get a friendly label."""
    if filename in ("", "<string>"):
        return "snippet"
    if filename == "<stdin>":
        return "stdin"
    return os.path.basename(filename.replace("\\", "/"))
