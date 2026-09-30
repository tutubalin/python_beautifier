"""The highlighter must be lossless: its output, stripped of markup, is the input."""
from __future__ import annotations

import html as htmllib
import re

import pytest

from python_beautifier.highlight import fragment

from conftest import tricky_snippets


def plain(markup: str) -> str:
    return htmllib.unescape(re.sub(r"<[^>]+>", "", markup))


@pytest.mark.parametrize("code", [c for c in tricky_snippets() if c.strip()])
def test_round_trip(code):
    assert plain(fragment(code)) == code.rstrip("\n") or plain(fragment(code)) == code


def test_round_trip_of_the_showcase(showcase_text):
    assert plain(fragment(showcase_text)).rstrip("\n") == showcase_text.rstrip("\n")


def test_token_classes():
    out = fragment("def f(x: int) -> str:\n    return 'a' + f'{x}'  # c\n")
    assert '<i class=k>def</i>' in out
    assert '<i class=k>return</i>' in out
    assert "class=cm" in out and "class=s" in out


def test_html_is_escaped():
    out = fragment("x = '<b>&</b>'")
    assert "<b>" not in out and "&lt;b&gt;" in out
