"""The bundled CSS and JavaScript."""
from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from python_beautifier.render.core import read_asset, stylesheet

CSS_FILES = ["base.css", "cards.css", "flow.css"]


@pytest.mark.parametrize("name", CSS_FILES)
def test_css_braces_are_balanced(name):
    css = re.sub(r"/\*.*?\*/", "", read_asset(name), flags=re.S)
    css = re.sub(r'"(?:\\.|[^"\\])*"', '""', css)
    depth = 0
    for ch in css:
        depth += ch == "{"
        depth -= ch == "}"
        assert depth >= 0, f"{name}: stray closing brace"
    assert depth == 0, f"{name}: unbalanced braces"


def test_dark_tokens_are_emitted_for_explicit_and_automatic_dark_mode():
    css = stylesheet()
    assert 'data-theme="dark"' in css
    assert "prefers-color-scheme: dark" in css
    assert css.count("--bg:") >= 3  # light, dark, automatic dark


def test_icons_named_in_the_source_exist():
    from pathlib import Path

    from python_beautifier.render import icons

    used = set()
    for path in Path(icons.__file__).parent.glob("*.py"):
        used |= set(re.findall(r'icon\(\s*"([\w-]+)"', path.read_text(encoding="utf-8")))
    assert used - set(icons.ICONS) == set(), f"unknown icons: {sorted(used - set(icons.ICONS))}"


def test_every_icon_in_the_rendered_pages_is_defined(showcase_html):
    """Undefined ``<use>`` targets render as empty boxes, so make sure there are none."""
    from conftest import tricky_snippets
    from python_beautifier import beautify

    pages = [showcase_html] + [beautify(code) for code in tricky_snippets() if code.strip()]
    for html in pages:
        used = set(re.findall(r'href="#i-([\w-]+)"', html))
        defined = set(re.findall(r'id="i-([\w-]+)"', html))
        assert used <= defined, sorted(used - defined)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_javascript_is_syntactically_valid(tmp_path):
    js = tmp_path / "app.js"
    js.write_text(read_asset("app.js"), encoding="utf-8")
    run = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
