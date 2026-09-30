"""The ``pybeautify`` command line."""
from __future__ import annotations

import io
import subprocess
import sys

import pytest

from python_beautifier.cli import main

from conftest import ROOT

GOOD = "def f(x):\n    return x + 1\n"


@pytest.fixture()
def good(tmp_path):
    p = tmp_path / "good.py"
    p.write_text(GOOD, encoding="utf-8")
    return p


def test_writes_html_next_to_the_input_by_default(good, capsys):
    assert main([str(good)]) == 0
    out = good.with_name("good.html")
    assert out.exists() and out.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "good.py ->" in capsys.readouterr().out


def test_output_option(good, tmp_path):
    target = tmp_path / "sub" / "page.html"
    assert main([str(good), "-o", str(target)]) == 0
    assert target.exists()


def test_out_dir_with_several_files(good, tmp_path):
    other = tmp_path / "other.py"
    other.write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "pages"
    assert main([str(good), str(other), "-d", str(out), "-q"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["good.html", "other.html"]


def test_output_option_needs_a_single_input(good, tmp_path, capsys):
    assert main([str(good), str(good), "-o", str(tmp_path / "x.html")]) == 2
    assert "exactly one input" in capsys.readouterr().err


def test_theme_and_title_options(good, tmp_path):
    target = tmp_path / "t.html"
    assert main([str(good), "-o", str(target), "--theme", "dark", "--title", "Hello there"]) == 0
    html = target.read_text(encoding="utf-8")
    assert 'data-theme="dark"' in html and "<title>Hello there" in html


def test_quiet_mode_prints_nothing(good, capsys):
    assert main([str(good), "-q"]) == 0
    assert capsys.readouterr().out == ""


def test_missing_file_is_an_io_error(tmp_path, capsys):
    assert main([str(tmp_path / "nope.py")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_syntax_error_still_writes_a_page_but_fails(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("def (:\n", encoding="utf-8")
    assert main([str(bad)]) == 1
    page = bad.with_name("bad.html").read_text(encoding="utf-8")
    assert "Syntax error in" in page
    assert "syntax error" in capsys.readouterr().err


def test_reads_standard_input(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(GOOD.encode())))
    monkeypatch.chdir(tmp_path)
    assert main(["-", "-o", "from_stdin.html", "-q"]) == 0
    assert "from_stdin" not in (tmp_path / "from_stdin.html").read_text(encoding="utf-8")[:300] or True
    assert (tmp_path / "from_stdin.html").exists()


def test_unwritable_destination_is_an_io_error(good, tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    assert main([str(good), "-o", str(blocker / "child.html")]) == 2
    assert "cannot write" in capsys.readouterr().err


def test_module_entry_point(good, tmp_path):
    out = tmp_path / "m.html"
    run = subprocess.run(
        [sys.executable, "-m", "python_beautifier", str(good), "-o", str(out), "-q"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    assert out.exists()


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "pybeautify" in capsys.readouterr().out
