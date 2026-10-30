"""Hostile and pathological input must never crash the tool or produce an unusable page."""
from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from python_beautifier import _runtime, beautify
from python_beautifier.cli import main

from conftest import ROOT


def is_page(html: str) -> bool:
    return html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")


def is_error_page(html: str) -> bool:
    return '<meta name="pb-error"' in html[:700]


CASES = {
    "null byte": "x = 1\0\n",
    "invalid utf-8": b"x = '\xff\xfe'\n",
    "unknown coding cookie": b"# -*- coding: nosuchcodec -*-\nx = 1\n",
    "lone surrogate": "x = '\ud800'\n",
    "long sum chain": "x = " + " + ".join(["a"] * 3000) + "\n",
    "long elif chain": "if x == 0:\n    a = 0\n" + "".join(f"elif x == {i}:\n    a = {i}\n" for i in range(1, 400)),
    "deep if nesting": "".join("    " * i + "if x:\n" for i in range(60)) + "    " * 60 + "pass\n",
    "deep class nesting": "".join("    " * i + f"class C{i}:\n" for i in range(40)) + "    " * 40 + "pass\n",
    "very long line": "x = '" + "a" * 100_000 + "'\n",
    "form feed": "\x0c\n",
    "bom only": b"\xef\xbb\xbf",
    "bare carriage returns": "x = 1\ry = 2\r",
    "mixed tabs and spaces": "if x:\n\ty = 1\n        z = 2\n",
    "unterminated string": "x = 'abc\n",
    "unicode": "π = '🎉'\ndef ƒ(ñ): return ñ\n",
}


@pytest.mark.parametrize("name", list(CASES))
def test_never_raises_and_always_yields_a_page(name):
    html = beautify(CASES[name], filename="adv.py")
    assert is_page(html)
    html.encode("utf-8")  # must be writable to disk


def test_unparseable_input_gives_an_error_page_with_a_marker():
    assert is_error_page(beautify("def (:\n", filename="bad.py"))
    assert is_error_page(beautify("x = 1\0\n"))


def test_absurdly_deep_code_gives_an_error_page_not_a_crash():
    """Run in a subprocess: a hard crash of the interpreter must fail this test, not the whole session."""
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from python_beautifier import beautify\n"
        "html = beautify('x = ' + ' + '.join(['a'] * 400000) + '\\n')\n"
        "print(html[:700])\n"
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=240)
    assert run.returncode == 0, run.stderr[-500:]
    assert '<meta name="pb-error" content="limit">' in run.stdout


def test_cli_reports_unprocessable_files_with_status_1(tmp_path, capsys):
    deep = tmp_path / "deep.py"
    deep.write_text("x = " + " + ".join(["a"] * 400_000) + "\n", encoding="utf-8")
    assert main([str(deep)]) == 1
    assert "could not be processed" in capsys.readouterr().err
    assert is_error_page((tmp_path / "deep.html").read_text(encoding="utf-8"))


def test_run_deep_restores_the_interpreter_settings():
    limit, stack = sys.getrecursionlimit(), threading.stack_size()
    assert _runtime.run_deep(lambda: sys.getrecursionlimit()) >= limit
    assert sys.getrecursionlimit() == limit
    assert threading.stack_size() == stack


def test_run_deep_propagates_exceptions():
    def boom():
        raise KeyError("x")

    with pytest.raises(KeyError):
        _runtime.run_deep(boom)
    assert sys.getrecursionlimit() < _runtime.DEEP_RECURSION_LIMIT


def test_run_deep_falls_back_when_no_big_stack_is_available(monkeypatch):
    def refuse(size=0):
        if size:
            raise ValueError("no big stacks here")
        return 0

    monkeypatch.setattr(threading, "stack_size", refuse)
    assert _runtime.run_deep(lambda: 42) == 42
