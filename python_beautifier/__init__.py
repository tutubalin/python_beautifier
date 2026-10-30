"""python_beautifier - turn Python source into a readable, structured HTML page.

>>> from python_beautifier import beautify
>>> html = beautify("def f(x):\\n    return x + 1\\n")
>>> "<article" in html
True
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

__version__ = "0.1.0"
__all__ = ["beautify", "beautify_file", "beautify_graph", "__version__"]


def beautify(
    source: Union[str, bytes],
    *,
    filename: str = "<string>",
    title: Optional[str] = None,
    theme: str = "auto",
) -> str:
    """Render *source* (Python code) as one self-contained HTML document.

    Args:
        source: The Python source text (or raw bytes; PEP 263 encodings are honoured).
        filename: Name shown in the page header and used to classify local imports.
        title: Overrides the page title (defaults to the file name).
        theme: ``"auto"`` (follow the OS), ``"light"`` or ``"dark"``.

    Returns:
        The complete HTML page. Files that do not parse yield a friendly error
        page instead of raising.
    """
    from ._runtime import run_deep
    from .render.core import error_page
    from .source import decode_source

    text = decode_source(source) if isinstance(source, (bytes, bytearray)) else source
    text = _scrub(text)
    try:
        return run_deep(_render, text, filename, title, theme)
    except SyntaxError as err:
        return error_page(text, filename, err)
    except (ValueError, RecursionError, MemoryError) as err:
        # ValueError: e.g. null bytes (Python < 3.12); the others: absurdly deep nesting
        return error_page(text, filename, err)


def beautify_graph(
    source: Union[str, bytes],
    *,
    filename: str = "<string>",
    title: Optional[str] = None,
    theme: str = "auto",
) -> str:
    """Render a focused, standalone HTML page containing only model graphs."""
    from .graph_tool import beautify_graph as _beautify_graph

    return _beautify_graph(source, filename=filename, title=title, theme=theme)


def _render(text: str, filename: str, title: Optional[str], theme: str) -> str:
    from .model import build_module
    from .render.core import Renderer
    from .source import Source

    src = Source(text, filename)
    return Renderer(build_module(src), title=title, theme=theme).render()


def _scrub(text: str) -> str:
    """Lone surrogates cannot be written as UTF-8; show them as U+FFFD instead."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = "".join("\ufffd" if "\ud800" <= ch <= "\udfff" else ch for ch in text)
    return text


def beautify_file(
    path: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    *,
    title: Optional[str] = None,
    theme: str = "auto",
) -> Path:
    """Beautify the Python file at *path* and write the HTML next to it.

    Args:
        path: Python source file.
        output: Destination ``.html`` file (default: *path* with ``.html`` appended).
        title: Overrides the page title.
        theme: ``"auto"``, ``"light"`` or ``"dark"``.

    Returns:
        The path of the written HTML file.
    """
    src_path = Path(path)
    out_path = Path(output) if output else src_path.with_suffix(".html")
    html = beautify(src_path.read_bytes(), filename=str(src_path), title=title, theme=theme)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
