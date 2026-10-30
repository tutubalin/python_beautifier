"""The public API: ``beautify`` and ``beautify_file``."""
from __future__ import annotations

import re

import pytest

import python_beautifier
from python_beautifier import beautify, beautify_file

from conftest import visible_text

CODE = '''"""Module doc."""
import os


def f(a: int, b="x"):
    """Do it.

    Args:
        a: the a.
        b: the b.
    """
    if a > 1:
        return 1
    else:
        return 2
'''


def test_returns_a_complete_html_document():
    html = beautify(CODE, filename="pkg/demo.py")
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "<title>demo.py" in html
    assert html.count("<style>") == 1 and html.count("<script>") >= 1


def test_output_is_self_contained():
    html = beautify(CODE)
    assert not re.search(r"<link\b", html)
    assert not re.search(r"<script[^>]+\bsrc=", html)
    assert "@import" not in html
    assert not re.search(r"url\(\s*['\"]?https?:", html)
    assert not re.search(r"<(?:img|iframe|source)[^>]+\bsrc=['\"]https?:", html)


def test_is_deterministic():
    assert beautify(CODE, filename="x.py") == beautify(CODE, filename="x.py")


def test_title_override_and_filename():
    html = beautify(CODE, filename="pkg/demo.py", title="My tour")
    assert "<title>My tour" in html
    assert "demo.py" in visible_text(html)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_forced_theme(theme):
    assert f'data-theme="{theme}"' in beautify(CODE, theme=theme)


def test_auto_theme_follows_the_system():
    html = beautify(CODE, theme="auto")
    assert 'data-theme="auto"' in html
    assert "prefers-color-scheme: dark" in html


def test_bytes_input_honours_the_encoding_cookie():
    raw = "# -*- coding: latin-1 -*-\nNAME = 'caf\xe9'\n".encode("latin-1")
    assert "café" in visible_text(beautify(raw))


def test_handles_bom_and_crlf():
    raw = b"\xef\xbb\xbfdef f(x):\r\n    return x\r\n"
    html = beautify(raw)
    assert 'id="d-f"' in html
    assert "\r" not in visible_text(html)


def test_syntax_errors_give_a_friendly_page_instead_of_raising():
    html = beautify("def (:\n    pass\n", filename="bad.py")
    assert "<title>Syntax error in bad.py</title>" in html
    assert "def (:" in visible_text(html)


@pytest.mark.parametrize("code", ["", "\n\n\n", "# just a comment\n", '"""only a docstring"""\n', "x = 1\n"])
def test_degenerate_modules(code):
    html = beautify(code)
    assert html.startswith("<!doctype html>")


def test_beautify_file_writes_next_to_the_source(tmp_path):
    src = tmp_path / "demo.py"
    src.write_text(CODE, encoding="utf-8")
    out = beautify_file(src)
    assert out == tmp_path / "demo.html"
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_beautify_file_creates_output_directories(tmp_path):
    src = tmp_path / "demo.py"
    src.write_text(CODE, encoding="utf-8")
    out = beautify_file(src, tmp_path / "deep" / "er" / "page.html", theme="dark")
    assert out.exists() and 'data-theme="dark"' in out.read_text(encoding="utf-8")


def test_version_is_exposed():
    assert re.fullmatch(r"\d+\.\d+\.\d+", python_beautifier.__version__)


def test_doctests_in_the_package_docstring():
    import doctest

    assert doctest.testmod(python_beautifier).failed == 0


def test_hero_shows_only_the_last_folders_of_a_path():
    html = beautify(CODE, filename="/home/someone/private/project/pkg/demo.py")
    assert "someone" not in html and "private" not in html
    assert "project/pkg/" in visible_text(html)


def test_custom_title_keeps_the_file_name_visible():
    html = beautify(CODE, filename="pkg/demo.py", title="My tour")
    assert "pkg/demo.py" in visible_text(html)


def test_pseudo_file_names_are_friendly():
    text = visible_text(beautify(CODE))
    assert "<string>" not in text and "snippet" in text
