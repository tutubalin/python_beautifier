"""Command line interface: ``pybeautify FILE -o OUT.html``."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

from . import __version__, beautify


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pybeautify",
        description="Turn Python source files into readable, structured, self-contained HTML pages.",
        epilog="Example: pybeautify examples/showcase.py -o showcase.html --open",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help="Python file(s) to beautify ('-' reads standard input)")
    out = p.add_mutually_exclusive_group()
    out.add_argument("-o", "--output", metavar="OUT.html", help="output file (only with a single input file)")
    out.add_argument("-d", "--out-dir", metavar="DIR", help="write <name>.html for every input into DIR")
    p.add_argument("--theme", choices=("auto", "light", "dark"), default="auto", help="initial colour theme (default: follow the OS)")
    p.add_argument("--title", help="page title (default: the file name)")
    p.add_argument("--open", action="store_true", help="open the result in your web browser")
    p.add_argument("-q", "--quiet", action="store_true", help="do not print progress messages")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _destination(src: str, args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    name = "stdin" if src == "-" else Path(src).name
    stem = name[:-3] if name.endswith(".py") else name
    if args.out_dir:
        return Path(args.out_dir) / f"{stem}.html"
    if src == "-":
        return Path("stdin.html")
    return Path(src).with_name(f"{stem}.html")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output and len(args.files) > 1:
        print("pybeautify: -o/--output needs exactly one input file (use -d for several)", file=sys.stderr)
        return 2
    status = 0
    last: Optional[Path] = None
    for src in args.files:
        try:
            data = sys.stdin.buffer.read() if src == "-" else Path(src).read_bytes()
        except OSError as exc:
            print(f"pybeautify: cannot read {src}: {exc.strerror or exc}", file=sys.stderr)
            status = 2
            continue
        name = "<stdin>" if src == "-" else src
        html = beautify(data, filename=name, title=args.title, theme=args.theme)
        dest = _destination(src, args)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(html, encoding="utf-8")
        except OSError as exc:
            print(f"pybeautify: cannot write {dest}: {exc.strerror or exc}", file=sys.stderr)
            status = 2
            continue
        last = dest
        if '<meta name="pb-error"' in html[:700]:
            kind = "has a syntax error" if 'content="syntax"' in html[:700] else "could not be processed"
            print(f"pybeautify: {name} {kind}; wrote an error page to {dest}", file=sys.stderr)
            status = max(status, 1)
        elif not args.quiet:
            print(f"{name} -> {dest}  ({len(html) / 1024:.0f} kB)")
    if args.open and last is not None:
        webbrowser.open(last.resolve().as_uri())
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
