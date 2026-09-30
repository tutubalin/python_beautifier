"""Nothing may be lost or repeated on the way from source to HTML."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import render_checked, tricky_snippets, visible_text


def assert_complete(code: str, name: str = "t.py") -> str:
    html, src = render_checked(code, name)
    missing = [(it.srow, it.text) for it in src.unrendered()]
    repeated = [(it.srow, it.text) for it in src.duplicated()]
    assert not missing, f"{name}: tokens never rendered: {missing[:5]}"
    assert not repeated, f"{name}: tokens rendered twice: {repeated[:5]}"
    return html


@pytest.mark.parametrize("code", tricky_snippets())
def test_tricky_snippets_are_rendered_completely(code):
    assert_complete(code)


def test_showcase_is_rendered_completely(showcase_text):
    assert_complete(showcase_text, "showcase.py")


def test_every_string_and_comment_of_the_showcase_is_visible(showcase_text):
    import ast
    import re

    html = assert_complete(showcase_text, "showcase.py")
    text = visible_text(html)
    squashed = re.sub(r"\s+", " ", text)
    # short string constants of the code (skip docstrings, which are re-flowed by the docstring parser)
    tree = ast.parse(showcase_text)
    docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str)}
    missing = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            s = n.value.strip()
            if 3 <= len(s) <= 60 and "\n" not in s and "\\" not in s and s not in squashed:
                missing.append(s)
    assert not missing, missing[:10]
    import tokenize
    import io

    for tok in tokenize.generate_tokens(io.StringIO(showcase_text).readline):
        if tok.type != tokenize.COMMENT or tok.string.startswith("#!"):  # the shebang becomes a badge
            continue
        body = re.sub(r"^#+\s*", "", tok.string)
        body = re.sub(r"^(TODO|FIXME|XXX|HACK|NOTE)\b:?\s*", "", body).strip(" -=~*#")
        words = body.split()[:3]
        if len(words) >= 2:  # banner-only comments such as "# ----" carry no words
            assert " ".join(words) in squashed, tok.string


def stdlib_sample(limit: int = 45):
    root = Path(os.__file__).parent
    files = sorted(p for p in root.glob("*.py") if p.stat().st_size < 150_000)
    return files[:limit]


@pytest.mark.parametrize("path", stdlib_sample(), ids=lambda p: p.name)
def test_standard_library_modules_render_completely(path):
    from python_beautifier.source import decode_source

    try:
        code = decode_source(path.read_bytes())
        compile(code, str(path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        pytest.skip("not parseable by this interpreter")
    assert_complete(code, path.name)
