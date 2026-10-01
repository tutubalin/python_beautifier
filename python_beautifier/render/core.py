"""Page assembly: sidebar, hero, overview panels and the source walkthrough."""
from __future__ import annotations

import ast
import json
import os
import re
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

from .. import __version__
from ..docstrings import first_sentence, inline_html
from ..highlight import fragment
from ..model import DefInfo, ModuleInfo
from ..neural import analyze as analyze_neural
from ..source import Source
from . import widgets
from .cards import Cards
from .flow import Ctx, Flow
from .icons import icon, sprite
from .prose import blocks_html, section_html
from .util import attr, esc, file_label, plural

NONE = '<span class="none">\u2014</span>'
_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


def read_asset(name: str) -> str:
    return (_ASSET_DIR / name).read_text(encoding="utf-8")


def stylesheet() -> str:
    """All CSS, plus the dark tokens again for ``theme=auto`` on dark systems."""
    base = read_asset("base.css")
    css = "\n".join([base, read_asset("cards.css"), read_asset("flow.css"), read_asset("neural.css")])
    m = re.search(r"/\*DARK\{\*/(.*?)/\*\}DARK\*/", base, re.S)
    if m:
        auto = m.group(1).replace('html[data-theme="dark"]', 'html[data-theme="auto"]')
        css += "\n@media (prefers-color-scheme: dark) {" + auto + "}\n"
    return css


# --------------------------------------------------------------------------- #
#  the renderer
# --------------------------------------------------------------------------- #


class Renderer:
    def __init__(self, mod: ModuleInfo, title: Optional[str] = None, theme: str = "auto") -> None:
        self.mod = mod
        self.src: Source = mod.src
        self.theme = theme
        self.title = title
        self.flow = Flow(self)
        self.cards = Cards(self)
        self.neural_schemas = analyze_neural(self.src)
        self._classes: Dict[str, DefInfo] = {}
        for d in mod.defs:
            if d.is_class and d.parent is None:
                self._classes.setdefault(d.name, d)
        self._by_qual = {d.qualname: d for d in mod.defs}
        self._short: Dict[str, List[DefInfo]] = {}
        for d in mod.defs:
            self._short.setdefault(d.name, []).append(d)
        self._subs: Dict[int, List[DefInfo]] = {}
        for d in mod.defs:
            if not d.is_class:
                continue
            for b in d.bases:
                base = self._classes.get(b.split("[")[0].split(".")[-1])
                if base is not None and base is not d:
                    self._subs.setdefault(id(base), []).append(d)

    # ------------------------------------------------------------------ #
    #  lookups used by docstrings and cards
    # ------------------------------------------------------------------ #

    def link(self, name: str) -> Optional[str]:
        name = name.strip().rstrip("()")
        if name.startswith(("self.", "cls.")):
            name = name.split(".", 1)[1]
        d = self._by_qual.get(name)
        if d is None:
            tail = name.split(".")[-1]
            cands = self._short.get(tail, [])
            if len(cands) == 1 and (name == tail or self._by_qual.get(".".join(name.split(".")[-2:])) is cands[0]):
                d = cands[0]
            elif "." in name:
                d = self._by_qual.get(".".join(name.split(".")[-2:]))
        return f"#{d.anchor}" if d is not None else None

    def mod_class(self, name: str) -> Optional[DefInfo]:
        return self._classes.get(name)

    def subclasses(self, d: DefInfo) -> List[DefInfo]:
        return self._subs.get(id(d), [])

    def def_card(self, node: ast.AST, ctx: Ctx, next_row: int) -> str:
        return self.cards.render(self.mod.by_node[id(node)], ctx, next_row)

    # ------------------------------------------------------------------ #
    #  statistics
    # ------------------------------------------------------------------ #

    def line_kinds(self) -> Dict[str, int]:
        src = self.src
        n = src.nlines
        kinds = ["code"] * n
        for i in range(n):
            if not src.lines[i].strip():
                kinds[i] = "blank"
        doc_nodes = [d.docstring_node for d in self.mod.defs if d.docstring_node is not None]
        if self.mod.docstring_node is not None:
            doc_nodes.append(self.mod.docstring_node)
        for node in doc_nodes:
            for r in range(node.lineno, node.end_lineno + 1):
                if r - 1 < n:
                    kinds[r - 1] = "docs"
        for c in src.comments:
            if c.own_line and c.row - 1 < n and kinds[c.row - 1] == "code":
                kinds[c.row - 1] = "comment"
        self._kinds = kinds
        counts = {"code": 0, "docs": 0, "comment": 0, "blank": 0}
        for k in kinds:
            counts[k] += 1
        return counts

    def coverage(self) -> Dict[str, float]:
        defs = self.mod.defs
        public = [d for d in defs if not d.name.startswith("_")] or defs
        docs = 100.0 * sum(1 for d in public if d.has_docstring) / len(public) if public else 100.0
        total = typed = 0
        for d in defs:
            if d.is_class:
                continue
            total += d.total_params + (0 if d.name == "__init__" else 1)
            typed += d.typed_params + (1 if (d.returns or d.name == "__init__") else 0)
        types = 100.0 * typed / total if total else 100.0
        return {"docs": docs, "types": types}

    # ------------------------------------------------------------------ #
    #  page
    # ------------------------------------------------------------------ #

    def file_label(self) -> str:
        return file_label(self.src.filename)

    def render(self) -> str:
        name = self.title or self.file_label()
        counts = self.line_kinds()
        cov = self.coverage()
        css = stylesheet()
        js = read_asset("app.js")
        source_json = json.dumps(self.src.text).replace("</", "<\\/")

        gutter = int(34 + 6.6 * max(3, len(str(self.src.nlines))))
        body_html = self._source_section()
        # any comment never claimed by the flow (defensive): show them rather than lose them
        page = (
            "<!doctype html>\n"
            f'<html lang="en" data-theme="{escape(self.theme)}" style="--gutter:{gutter}px">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{esc(name)} \u00b7 beautified</title>\n"
            '<meta name="generator" content="python_beautifier">\n'
            f"<style>\n{css}\n</style>\n</head>\n<body>\n{sprite()}\n"
            '<div class="app">\n'
            f"{self._sidebar(name)}\n"
            '<main class="main" id="main">\n'
            f"{self._topbar()}\n{self._hero(name, counts, cov)}\n{self._overview(counts, cov)}\n{body_html}\n{self._footer(name)}\n"
            "</main>\n</div>\n"
            f'<script type="text/plain" id="pb-source">{source_json}</script>\n'
            f"<script>\n{js}\n</script>\n</body>\n</html>\n"
        )
        return page

    # ------------------------------------------------------------------ #
    #  sidebar & topbar
    # ------------------------------------------------------------------ #

    def _outline_item(self, d: DefInfo, nested: bool = False) -> str:
        role = d.role if d.role in ("class", "property") else ("function" if not d.is_method else "method")
        ic = {"class": "box", "property": "eye", "function": "function", "method": "function"}[role]
        kids = ""
        if d.is_class:
            sub = [c for c in d.children]
            if sub:
                kids = '<ul class="ol-sub">' + "".join(self._outline_item(c, True) for c in sub) + "</ul>"
        chev = f'<button class="ol-t" type="button" aria-label="Toggle">{icon("chev")}</button>' if kids else '<span class="ol-t ph"></span>'
        return (
            f'<li class="ol-{role}{" has-kids" if kids else ""}" data-name="{attr(d.qualname.lower())}">'
            f'{chev}<a href="#{d.anchor}" data-target="{d.anchor}">{icon(ic)}<span>{esc(d.name)}</span><i class="dot g-{d.grade}"></i></a>{kids}</li>'
        )

    def _sidebar(self, name: str) -> str:
        mod = self.mod
        classes = [d for d in mod.top if d.is_class]
        funcs = [d for d in mod.top if not d.is_class]
        groups = []
        if classes:
            groups.append(f'<div class="ol-h">Classes<small>{len(classes)}</small></div><ul>{"".join(self._outline_item(d) for d in classes)}</ul>')
        if funcs:
            groups.append(f'<div class="ol-h">Functions<small>{len(funcs)}</small></div><ul>{"".join(self._outline_item(d) for d in funcs)}</ul>')
        if mod.todos:
            groups.append(f'<div class="ol-h">Notes</div><ul><li class="ol-todo"><span class="ol-t ph"></span><a href="#todos">{icon("flag")}<span>To-do board</span><i class="cnt">{len(mod.todos)}</i></a></li></ul>')
        return (
            '<aside class="side" id="side"><div class="side-in">'
            f'<a class="brand" href="#top"><span class="logo">{icon("sparkle")}</span><span class="bt"><b>{esc(name)}</b><small>beautified &middot; {plural(len(mod.defs), "definition")}</small></span></a>'
            f'<label class="search">{icon("search")}<input id="q" type="search" placeholder="Jump to\u2026  ( / )" autocomplete="off" spellcheck="false"></label>'
            f'<nav class="outline" aria-label="Outline"><a class="ol-top" href="#top">{icon("map")}<span>Overview</span></a>'
            f'<a class="ol-top" href="#source">{icon("terminal")}<span>Source walkthrough</span></a>'
            f'{"".join(groups)}<div class="ol-empty" hidden>No matches</div></nav>'
            "</div></aside>"
        )

    def _topbar(self) -> str:
        # only pages that slice arrays get the button that explains it
        indexing = (
            '<button class="btn" id="b-index" type="button" aria-pressed="true" '
            f'title="Explain NumPy / PyTorch style indexing such as x[:, 1:, ...]">{icon("brackets")}<span>Indexing</span></button>'
            if self.src.index
            else ""
        )
        return (
            '<div class="topbar">'
            f'<button class="btn icon-btn" id="menu" type="button" aria-label="Menu">{icon("menu")}</button>'
            '<div class="sp"></div>'
            f'<button class="btn" id="b-expand" type="button" title="Expand every card">{icon("expand")}<span>Expand all</span></button>'
            f'<button class="btn" id="b-collapse" type="button" title="Collapse every card">{icon("collapse")}<span>Collapse all</span></button>'
            f'<button class="btn" id="b-lines" type="button" aria-pressed="true" title="Toggle line numbers">{icon("hash")}<span>Lines</span></button>'
            f'<button class="btn" id="b-comments" type="button" aria-pressed="true" title="Show or hide comments">{icon("comment")}<span>Comments</span></button>'
            f"{indexing}"
            f'<button class="btn icon-btn" id="b-theme" type="button" aria-label="Toggle theme" title="Light / dark">{icon("sun", "t-sun")}{icon("moon", "t-moon")}</button>'
            f'<button class="btn icon-btn" id="b-print" type="button" aria-label="Print" title="Print">{icon("file")}</button>'
            "</div>"
        )

    # ------------------------------------------------------------------ #
    #  hero
    # ------------------------------------------------------------------ #

    def _minimap(self) -> str:
        """The file drawn as a skyline: one bar per line (indent + length), in columns."""
        src = self.src
        kinds = getattr(self, "_kinds", None) or []
        n = src.nlines
        if n == 0:
            return ""
        def_rows = {d.node.lineno for d in self.mod.defs}  # type: ignore[attr-defined]
        group = max(1, -(-n // 210))
        bars = []
        for start in range(0, n, group):
            best = 0
            indent = 10**6
            seen = set()
            is_def = False
            for i in range(start, min(n, start + group)):
                line = src.lines[i]
                if not line.strip():
                    continue
                best = max(best, len(line.rstrip()))
                indent = min(indent, len(line) - len(line.lstrip()))
                seen.add(kinds[i] if i < len(kinds) else "code")
                if (i + 1) in def_rows:
                    is_def = True
            if not seen:
                bars.append(None)
                continue
            kind = "def" if is_def else next(k for k in ("code", "docs", "comment") if k in seen)
            bars.append((indent, best, kind))
        rows = max(10, round(3.15 * (len(bars) ** 0.5)))
        cols = max(1, -(-len(bars) // rows))
        col_w, col_gap, bar_h, bar_gap = 54.0, 12.0, 3.2, 1.7
        unit = col_w / 64.0
        out = []
        for idx, bar in enumerate(bars):
            if bar is None:
                continue
            indent, length, kind = bar
            col, row = divmod(idx, rows)
            x = col * (col_w + col_gap) + min(indent, 16) * unit
            w = max(2.5, min(col_w - min(indent, 16) * unit, (length - indent) * unit))
            y = row * (bar_h + bar_gap)
            out.append(f'<rect class="mm-{kind}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h}" rx="1.4"/>')
        width = cols * col_w + (cols - 1) * col_gap
        height = rows * (bar_h + bar_gap)
        return f'<svg class="minimap" viewBox="0 0 {width:.0f} {height:.0f}" aria-hidden="true">{"".join(out)}</svg>'

    def _stat_tile(self, value: str, label: str, ic: str, tone: str = "") -> str:
        return f'<div class="tile {tone}"><span class="ti">{icon(ic)}</span><b>{value}</b><small>{label}</small></div>'

    def _hero(self, name: str, counts: Dict[str, int], cov: Dict[str, float]) -> str:
        mod = self.mod
        stem = name[:-3] if name.endswith(".py") else name
        ext = ".py" if name.endswith(".py") else ""
        folder = _short_dir(os.path.dirname(self.src.filename))
        if self.title and self.src.filename not in ("<stdin>", "<string>", ""):
            eyebrow = esc(folder + Path(self.src.filename).name)  # a custom title must not hide which file this is
        else:
            eyebrow = esc(folder) if folder else "Python module"
        lead = ""
        if mod.doc and mod.doc.summary:
            lead = f'<p class="lead">{inline_html(first_sentence(mod.doc.summary, 260), self.link)}</p>'
        chips = []
        if "version" in mod.meta:
            chips.append(widgets.badge("v" + mod.meta["version"], "accent", "tag"))
        if "author" in mod.meta:
            chips.append(widgets.badge(mod.meta["author"], "muted", "at"))
        if "license" in mod.meta:
            chips.append(widgets.badge(mod.meta["license"], "muted", "file"))
        if mod.shebang:
            chips.append(widgets.badge("executable script", "muted", "terminal", title=mod.shebang))
        if any(d.is_async for d in mod.defs if not d.is_class):
            chips.append(widgets.badge("async", "info", "zap"))
        if mod.all_names is not None:
            chips.append(widgets.badge(f"{len(mod.all_names)} public names", "muted", "globe"))
        if self._has_main():
            chips.append(widgets.badge("runnable", "good", "play"))
        fns = [d for d in mod.defs if not d.is_class]
        avg = sum(d.cc for d in fns) / len(fns) if fns else 1.0
        total = max(1, sum(counts.values()))
        tiles = [
            self._stat_tile(str(self.src.nlines), "lines", "file"),
            self._stat_tile(str(len(mod.classes)), "classes", "box"),
            self._stat_tile(str(len(mod.functions)), "functions", "function"),
            self._stat_tile(str(len(mod.methods)), "methods", "layers"),
            self._stat_tile(f"{avg:.1f}", "complexity", "cpu"),
            self._stat_tile(f"{cov['docs']:.0f}%", "documented", "book", "good" if cov["docs"] >= 80 else "mid" if cov["docs"] >= 50 else "low"),
            self._stat_tile(f"{cov['types']:.0f}%", "type hints", "braces", "good" if cov["types"] >= 80 else "mid" if cov["types"] >= 50 else "low"),
        ]
        bar = "".join(
            f'<span class="lb lb-{k}" style="flex:{max(v, 0.0001)}" title="{v} {k} lines ({100 * v / total:.0f}%)"></span>'
            for k, v in counts.items()
            if v
        )
        legend = "".join(
            f'<span class="lk"><i class="lb-{k}"></i>{k} <b>{counts[k]}</b></span>' for k in ("code", "docs", "comment", "blank")
        )
        return (
            '<header class="hero" id="top"><div class="hero-bg" aria-hidden="true"></div><div class="hero-in">'
            '<div class="hero-text">'
            f'<div class="eyebrow">{icon("file")}<span>{eyebrow}</span></div>'
            f'<h1 class="title"><span class="stem">{esc(stem)}</span><span class="suffix">{ext}</span></h1>'
            f'{lead}<div class="hero-chips">{"".join(chips)}</div></div>'
            f'<div class="hero-art">{self._minimap()}<small>the file, line by line</small></div>'
            f'</div><div class="hero-foot"><div class="tiles">{"".join(tiles)}</div>'
            f'<div class="linebar"><div class="lb-bar">{bar}</div><div class="lb-legend">{legend}</div></div></div></header>'
        )

    def _has_main(self) -> bool:
        for s in self.src.tree.body:
            if isinstance(s, ast.If) and self.flow._is_main_guard(s.test):
                return True
        return False

    # ------------------------------------------------------------------ #
    #  overview panels
    # ------------------------------------------------------------------ #

    def _panel(self, cls: str, ic: str, title: str, body: str, note: str = "", pid: str = "") -> str:
        if not body:
            return ""
        ident = f' id="{pid}"' if pid else ""
        n = f"<small>{note}</small>" if note else ""
        return f'<section class="panel {cls}"{ident}><h2 class="ph">{icon(ic)}<span>{title}</span>{n}</h2><div class="pb">{body}</div></section>'

    def _overview(self, counts: Dict[str, int], cov: Dict[str, float]) -> str:
        mod = self.mod
        panels: List[str] = []
        doc_html = ""
        if mod.doc and (mod.doc.description_blocks or mod.doc.sections):
            doc_html = (
                f'<div class="prose">{blocks_html(mod.doc.blocks[1:] if mod.doc.blocks and mod.doc.blocks[0].kind == "p" else mod.doc.blocks, self.link)}'
                f'{"".join(section_html(s, self.link) for s in mod.doc.sections)}</div>'
            )
        panels.append(self._panel("p-doc wide", "book", "About this module", doc_html))
        panels.append(self._panel("p-map wide", "map", "Code map", widgets.treemap(mod), "every box is a definition"))
        panels.append(self._panel("p-health", "cpu", "Health", self._health(cov)))
        panels.append(self._panel("p-chords", "git-merge", "Who calls whom", widgets.call_chords(mod), "local calls only"))
        panels.append(self._panel("p-api wide", "list", "API at a glance", self._api_index()))
        panels.append(self._panel("p-imports", "package", "Imports", self._imports()))
        panels.append(self._panel("p-consts", "hash", "Constants", self._constants(), pid="constants"))
        panels.append(self._panel("p-todos", "flag", "To-do board", self._todos(), pid="todos"))
        return f'<section class="overview"><div class="grid">{"".join(p for p in panels if p)}</div></section>'

    def _health(self, cov: Dict[str, float]) -> str:
        mod = self.mod
        fns = [d for d in mod.defs if not d.is_class]
        worst = sorted(fns, key=lambda d: -d.cc)[:3]
        worst_html = "".join(
            f'<a class="wl g-{d.grade}" href="#{d.anchor}"><b>{esc(d.name)}</b><span>complexity {d.cc}</span>{widgets.gauge(d.cc, "sm")}</a>'
            for d in worst
            if d.cc > 3
        )
        rings = (
            f'<div class="rings"><div class="rg">{widgets.ring(cov["docs"], "Docstring coverage")}<span>docstrings</span></div>'
            f'<div class="rg">{widgets.ring(cov["types"], "Type hint coverage")}<span>type hints</span></div></div>'
        )
        nested = max((d.nesting for d in fns), default=0)
        deep = max(fns, key=lambda d: d.nesting, default=None)
        facts = []
        if deep is not None and nested >= 3:
            facts.append(f'<li>{icon("layers")}deepest nesting: <b>{nested}</b> levels in <a href="#{deep.anchor}"><code>{esc(deep.name)}</code></a></li>')
        longest = max(fns, key=lambda d: d.loc, default=None)
        if longest is not None:
            facts.append(f'<li>{icon("file")}longest function: <a href="#{longest.anchor}"><code>{esc(longest.name)}</code></a> <b>{longest.loc}</b> lines</li>')
        if mod.todos:
            facts.append(f'<li>{icon("flag")}<a href="#todos"><b>{len(mod.todos)}</b> open to-do note{"s" if len(mod.todos) != 1 else ""}</a></li>')
        rec = [d for d in fns if d.recursive]
        if rec:
            facts.append(f'<li>{icon("rotate")}recursive: {", ".join(f"<code>{esc(d.name)}</code>" for d in rec[:4])}</li>')
        return (
            f"{rings}{widgets.grade_bar(mod.defs)}"
            f'{"<div class=worst><h3>Hot spots</h3>" + worst_html + "</div>" if worst_html else ""}'
            f'{"<ul class=facts>" + "".join(facts) + "</ul>" if facts else ""}'
        )

    def _api_index(self) -> str:
        mod = self.mod
        if not mod.top:
            return ""
        rows = []
        for d in mod.top:
            summary = first_sentence(d.doc.summary, 150) if d.has_docstring else ""
            sum_html = inline_html(summary, self.link) if summary else '<span class="none">\u2014</span>'
            if d.is_class:
                members = [c for c in d.children if not c.is_class]
                extra = f'<span class="ai-x">{plural(len(members), "method")}</span>'
                cc = max((m.cc for m in members), default=d.cc)
            else:
                params = [p.shown_name for p in d.params]
                extra = f'<span class="ai-x">({esc(", ".join(params))})</span>'
                cc = d.cc
            badges = "".join(
                widgets.badge(b, *_badge_style(b)) for b in d.badges if b in ("async", "generator", "dataclass", "enum", "protocol", "abstract", "exception", "private")
            )
            rows.append(
                f'<tr><td class="ai-n"><a href="#{d.anchor}"><span class="kind {"k-cls" if d.is_class else "k-fn"} sm">{icon("box" if d.is_class else "function")}</span>'
                f'<b>{esc(d.name)}</b></a>{extra}{badges}</td>'
                f'<td class="pc" data-label="Summary">{sum_html}</td><td class="mc" data-label="Complexity">{widgets.gauge(cc, "sm")}</td>'
                f'<td class="ml" data-label="Lines">{d.loc}</td></tr>'
            )
        return (
            '<div class="tbl"><table class="ptable mtable"><thead><tr><th>Definition</th><th>Summary</th><th>CC</th><th>Lines</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
        )

    def _imports(self) -> str:
        imps = self.mod.imports
        if not imps:
            return ""
        order = [("future", "__future__"), ("stdlib", "Standard library"), ("third", "Third party"), ("local", "This project")]
        out = []
        for key, label in order:
            items = [i for i in imps if i.group == key]
            if not items:
                continue
            chips = []
            seen = set()
            for i in items:
                name = ("." * i.level) + i.module if i.is_from else i.module
                names = ", ".join(n + (f" as {a}" if a else "") for n, a in i.names)
                if not i.is_from and i.alias:
                    names = f"as {i.alias}"
                ident = (name, names)
                if ident in seen:
                    continue
                seen.add(ident)
                short = names if len(names) <= 34 else names[:33] + "\u2026"
                chips.append(
                    f'<span class="imp imp-{key}" title="{esc(("from " if i.is_from else "import ") + name + (" import " + names if i.is_from else ""))}">'
                    f"<b>{esc(name)}</b>{f'<i>{esc(short)}</i>' if short else ''}</span>"
                )
            out.append(f'<div class="imp-g"><h3>{label}<small>{len(items)}</small></h3><div class="imp-c">{"".join(chips)}</div></div>')
        return "".join(out)

    def _constants(self) -> str:
        consts = self.mod.consts
        if not consts:
            return ""
        rows = []
        shown = consts[:16]
        for c in shown:
            value = c.value if len(c.value) <= 46 else c.value[:45] + "\u2026"
            ty = (
                f'<code class="ty{" inferred" if c.inferred else ""}">{esc(c.type)}</code>' if c.type else '<span class="none">\u2014</span>'
            )
            cmt = f'<span class="cmt">{esc(c.comment)}</span>' if c.comment else ""
            rows.append(
                f'<tr><td class="pn"><a href="#L{c.line}"><code>{esc(c.name)}</code></a></td><td class="pt" data-label="Type">{ty}</td>'
                f'<td class="pd" data-label="Value"><code class="dv">{fragment(value)}</code>{cmt}</td></tr>'
            )
        more = f'<p class="more-note">+ {len(consts) - len(shown)} more</p>' if len(consts) > len(shown) else ""
        return (
            '<div class="tbl"><table class="ptable ctable"><colgroup><col style="width:36%"><col style="width:16%"><col></colgroup>'
            '<thead><tr><th>Name</th><th>Type</th><th>Value</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>{more}'
        )

    def _todos(self) -> str:
        todos = self.mod.todos
        if not todos:
            return ""
        items = []
        for t in todos:
            items.append(
                f'<a class="todo-i td-{t.tag.lower()}" href="#L{t.line}"><span class="todo-tag">{t.tag}</span>'
                f'<span class="todo-t">{esc(t.text) or "<i>(no text)</i>"}</span><small>line {t.line}</small></a>'
            )
        return f'<div class="todo-list">{"".join(items)}</div>'

    # ------------------------------------------------------------------ #
    #  source
    # ------------------------------------------------------------------ #

    def _legend(self) -> str:
        items = [
            ("if", "diamond", "decision"),
            ("yes", "check", "yes"),
            ("no", "x", "no"),
            ("loop", "loop", "loop"),
            ("try", "shield", "try"),
            ("except", "alert", "except"),
            ("with", "layers", "with"),
            ("match", "shuffle", "match"),
            ("ret", "return", "return"),
            ("raise", "alert", "raise"),
        ]
        chips = "".join(f'<span class="lg lg-{k}">{icon(ic)}{label}</span>' for k, ic, label in items)
        return f'<div class="legend"><span class="legend-l">How to read</span>{chips}</div>'

    def _source_section(self) -> str:
        src = self.src
        tree = src.tree
        stmts = list(tree.body)
        start_row = 0
        head_notes = ""
        if self.mod.docstring_node is not None and stmts and stmts[0] is self.mod.docstring_node:
            doc_first = src.first_line(stmts[0])
            lead = src.take_own_line(0, doc_first, 0)
            items = self.flow._notes(lead)
            if items:
                head_notes = '<div class="run">' + "".join(h for _, h in items) + "</div>"
            stmts = stmts[1:]
            start_row = self.mod.docstring_node.end_lineno
        ctx = Ctx(indent=0, start_row=start_row, end_row=src.nlines + 1, top=True, fold_imports=True)
        flow_html = self.flow.body(stmts, ctx) if stmts else ""
        if not flow_html and not head_notes:
            flow_html = '<p class="more-note">This file has no code besides its docstring.</p>'
        # anything left over (defensive): comments in odd places
        leftovers = [c for c in src.comments if not c.used]
        extra = ""
        if leftovers:
            items = self.flow._notes([c for c in leftovers if c.own_line])
            for c in leftovers:
                c.used = True
            if items:
                extra = '<div class="run">' + "".join(h for _, h in items) + "</div>"
        return (
            '<section class="source" id="source"><h2 class="sec-title">'
            f'{icon("terminal")}<span>Source walkthrough</span><small>top to bottom, structured by control flow</small></h2>'
            f'{self._legend()}<div class="flow mod-flow">{head_notes}{flow_html}{extra}</div></section>'
        )

    def _footer(self, name: str) -> str:
        return (
            f'<footer class="foot"><span>{icon("sparkle")}Generated by <b>python_beautifier</b> {esc(__version__)}</span>'
            f"<span>{esc(name)} &middot; {self.src.nlines} lines &middot; fully offline, no external resources</span></footer>"
        )


def _short_dir(folder: str) -> str:
    """The last two folders of a path, for display (never leak a whole local path)."""
    parts = [p for p in re.split(r"[\\/]+", folder) if p and p != "." and not p.endswith(":")]
    if not parts:
        return ""
    return ("\u2026/" if len(parts) > 2 else "") + "/".join(parts[-2:]) + "/"


def _badge_style(b: str):
    from .cards import _BADGE_KIND

    return _BADGE_KIND.get(b, ("muted", ""))


# --------------------------------------------------------------------------- #
#  error page
# --------------------------------------------------------------------------- #


def error_page(text: str, filename: str, err: BaseException) -> str:
    """A friendly page for files that cannot be processed (syntax errors and absurd nesting)."""
    syntax = isinstance(err, SyntaxError)
    lines = text.splitlines()
    row = getattr(err, "lineno", None) or 0
    msg = getattr(err, "msg", None) or str(err) or "invalid syntax"
    if isinstance(err, RecursionError):
        msg = "the code is nested too deeply (Python's recursion limit was reached)"
    elif isinstance(err, MemoryError):
        msg = "the file is too large or too deeply nested to analyse"
    rows = []
    if row and lines:
        row = min(row, len(lines))
        lo, hi = max(1, row - 4), min(len(lines), row + 3)
        for r in range(lo, hi + 1):
            cls = "bad" if r == row else ""
            rows.append(f'<div class="el {cls}"><i>{r}</i><code>{escape(lines[r - 1]) or "&nbsp;"}</code></div>')
        offset = getattr(err, "offset", None)
        if offset and row <= len(lines):
            caret = f'<div class="el caret"><i></i><code>{" " * max(0, offset - 1)}^</code></div>'
            rows.insert(row - lo + 1, caret)
    if syntax and row:
        kind, lead, headline = "syntax", f"Python reports a syntax error on line {row}:", "Couldn\u2019t parse"
        title = f"Syntax error in {file_label(filename)}"
        hint = "Fix the error and run the beautifier again. Only the Python grammar of the interpreter running the tool is understood."
    elif syntax:
        kind, lead, headline = "syntax", "Python could not read this file:", "Couldn\u2019t parse"
        title = f"Syntax error in {file_label(filename)}"
        hint = "Fix the problem and run the beautifier again."
    else:
        kind, lead, headline = "limit", "This file cannot be processed:", "Couldn\u2019t process"
        title = f"Could not process {file_label(filename)}"
        hint = "Split very long chains (for example thousands of <code>elif</code> branches or <code>+</code> operands) and try again."
    css = read_asset("base.css") + read_asset("cards.css")
    code_block = f'<div class="errcode">{"".join(rows)}</div>' if rows else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta name="pb-error" content="{kind}"><title>{escape(title)}</title><style>{css}</style></head><body class=\'err-page\'>{sprite()}'
        '<main class="errbox"><div class="err-ic">' + icon("alert") + "</div>"
        f"<h1>{headline} <span>{escape(file_label(filename))}</span></h1>"
        f'<p class="lead">{lead}</p>'
        f'<p class="msg">{escape(msg)}</p>'
        f"{code_block}"
        f'<p class="hint">{hint}</p></main></body></html>'
    )
