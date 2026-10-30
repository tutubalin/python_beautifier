"""Command line interface for the compact, graph-only HTML explorer."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

from . import __version__, beautify_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pybeautify-graph",
        description="Create a focused HTML page containing only statically inferred model graphs.",
        epilog="Example: pybeautify-graph model.py -o model-graphs.html --open",
    )
    parser.add_argument("file", metavar="FILE", help="Python source file ('-' reads standard input)")
    parser.add_argument("-o", "--output", metavar="OUT.html", help="output HTML (default: <name>.graph.html)")
    parser.add_argument("--theme", choices=("auto", "light", "dark"), default="auto", help="initial color theme")
    parser.add_argument("--title", help="page title (default: input file name)")
    parser.add_argument("--open", action="store_true", help="open the graph page in a browser")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress the success message")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = sys.stdin.buffer.read() if args.file == "-" else Path(args.file).read_bytes()
    except OSError as exc:
        print(f"pybeautify-graph: cannot read {args.file}: {exc.strerror or exc}", file=sys.stderr)
        return 2

    filename = "<stdin>" if args.file == "-" else args.file
    html = beautify_graph(data, filename=filename, title=args.title, theme=args.theme)
    if args.output:
        destination = Path(args.output)
    elif args.file == "-":
        destination = Path("stdin.graph.html")
    else:
        source_path = Path(args.file)
        destination = source_path.with_name(source_path.stem + ".graph.html")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(html, encoding="utf-8")
    except OSError as exc:
        print(f"pybeautify-graph: cannot write {destination}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    if args.open:
        webbrowser.open(destination.resolve().as_uri())
    if not args.quiet:
        print(f"{filename} -> {destination}  ({len(html) / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
