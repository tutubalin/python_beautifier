"""Definition cards: headers, parameter tables, class overviews and bodies."""
from __future__ import annotations

import ast
from typing import List, Optional, Sequence

from ..docstrings import first_sentence, inline_html
from ..highlight import fragment
from ..model import DefInfo, ParamInfo
from . import widgets
from .flow import Ctx
from .icons import icon
from .neural import render as render_neural
from .prose import blocks_html, section_html
from .util import attr, esc, plural

_KIND_ICON = {
    "class": "box",
    "function": "function",
    "method": "function",
    "classmethod": "function",
    "staticmethod": "function",
    "property": "eye",
}
_KIND_CLASS = {
    "class": "k-cls",
    "function": "k-fn",
    "method": "k-m",
    "classmethod": "k-m",
    "staticmethod": "k-m",
    "property": "k-p",
}
_BADGE_KIND = {
    "async": ("info", "zap"),
    "generator": ("info", "shuffle"),
    "property": ("muted", ""),
    "staticmethod": ("muted", "pin"),
    "classmethod": ("muted", "layers"),
    "abstract": ("warn", ""),
    "dataclass": ("accent", "braces"),
    "enum": ("accent", "list"),
    "protocol": ("accent", "link"),
    "exception": ("danger", "alert"),
    "private": ("muted", "lock"),
    "dunder": ("muted", ""),
    "recursive": ("info", "rotate"),
    "deprecated": ("danger", ""),
    "cached": ("info", "clock"),
    "overload": ("muted", ""),
    "setter": ("muted", "edit"),
    "context manager": ("info", "layers"),
    "namedtuple": ("accent", "braces"),
    "typeddict": ("accent", "braces"),
    "model": ("accent", "braces"),
    "nested classes": ("muted", "layers"),
}
_BADGED_DECORATORS = {
    "property", "classmethod", "staticmethod", "abstractmethod", "abc.abstractmethod", "dataclass",
    "dataclasses.dataclass", "overload", "typing.overload", "cached_property",
    "functools.cached_property", "contextmanager", "contextlib.contextmanager", "asynccontextmanager",
}
_BADGE_LABEL = {"staticmethod": "static", "classmethod": "classmethod"}


class Cards:
    """Renders class and function definitions."""

    def __init__(self, r) -> None:  # r: render.core.Renderer
        self.r = r
        self.src = r.src
        self.mod = r.mod

    # ------------------------------------------------------------------ #
    #  shared bits
    # ------------------------------------------------------------------ #

    def link(self, name: str) -> Optional[str]:
        return self.r.link(name)

    def prose(self, text: str) -> str:
        return inline_html(text, self.link)

    def badges(self, d: DefInfo) -> str:
        out = []
        for b in d.badges:
            kind, ic = _BADGE_KIND.get(b, ("muted", ""))
            out.append(widgets.badge(_BADGE_LABEL.get(b, b), kind, ic))
        if not d.has_docstring and not ("private" in d.badges or "dunder" in d.badges):
            out.append(widgets.badge("undocumented", "ghost"))
        return "".join(out)

    def decorators(self, d: DefInfo) -> str:
        src = self.src
        node = d.node
        chips = []
        for dec in node.decorator_list:  # type: ignore[attr-defined]
            src.exempt(src.start(dec), src.end(dec))
            if isinstance(dec, (ast.Name, ast.Attribute)) and self._compact(dec) in _BADGED_DECORATORS:
                tail = src.take_inline(dec.end_lineno, src.end(dec)[1])
                continue
            tail = src.take_inline(dec.end_lineno, src.end(dec)[1])
            title = f' title="{attr(tail.body)}"' if tail else ""
            chips.append(f'<code class="deco"{title}>{fragment("@" + self._compact(dec))}</code>')
        return f'<div class="decos">{"".join(chips)}</div>' if chips else ""

    def _compact(self, node: ast.AST) -> str:
        from ..model import compact

        return compact(self.src, node)

    # ------------------------------------------------------------------ #
    #  signature
    # ------------------------------------------------------------------ #

    def signature_text(self, d: DefInfo) -> str:
        if d.is_class:
            args = list(d.bases) + list(d.class_keywords)
            return f"class {d.name}" + (f"({', '.join(args)})" if args else "")
        items: List[str] = []
        params = d.params
        last_pos = max((i for i, p in enumerate(params) if p.kind == "posonly"), default=-1)
        has_vararg = any(p.kind == "vararg" for p in params)
        star_done = has_vararg
        for i, p in enumerate(params):
            if p.kind == "kwonly" and not star_done:
                items.append("*")
                star_done = True
            text = p.shown_name
            if p.annotation:
                text += f": {p.annotation}"
            if p.default is not None:
                text += f" = {p.default}" if p.annotation else f"={p.default}"
            items.append(text)
            if i == last_pos:
                items.append("/")
        prefix = "async def" if d.is_async else "def"
        ret = f" -> {d.returns}" if d.returns else ""
        one = f"{prefix} {d.name}({', '.join(items)}){ret}"
        if len(one) <= 96 or not items:
            return one
        body = "".join(f"    {it},\n" for it in items)
        return f"{prefix} {d.name}(\n{body}){ret}"

    # ------------------------------------------------------------------ #
    #  tables
    # ------------------------------------------------------------------ #

    def type_cell(self, p: ParamInfo) -> str:
        if p.annotation:
            return f'<code class="ty">{fragment(p.annotation, as_type=True)}</code>'
        if p.doc_type:
            return f'<code class="ty from-doc" title="type taken from the docstring">{fragment(p.doc_type, as_type=True)}</code>'
        if p.inferred and p.inferred != "None":
            return f'<code class="ty inferred" title="inferred from the default value">{esc(p.inferred)}<sup>inferred</sup></code>'
        return '<span class="none">untyped</span>'

    def params_table(self, d: DefInfo) -> str:
        rows: List[str] = []
        shown = [
            p
            for p in d.params
            if not (d.is_method and d.role != "staticmethod" and p is d.params[0] and p.kind in ("posonly", "normal"))
        ]
        if not shown:
            return ""
        for p in shown:
            tags = []
            if p.kind == "posonly":
                tags.append("positional-only")
            if p.kind == "kwonly":
                tags.append("keyword-only")
            chips = "".join(f'<span class="pk">{t}</span>' for t in tags)
            if p.kind == "vararg":
                default = '<span class="dim">variadic</span>'
            elif p.kind == "kwarg":
                default = '<span class="dim">keywords</span>'
            elif p.default is not None:
                default = f'<code class="dv">{fragment(p.default)}</code>'
            elif p.doc_default:
                default = f'<code class="dv from-doc" title="default mentioned in the docstring">{esc(p.doc_default)}</code>'
            else:
                default = '<span class="req">required</span>'
            if p.desc:
                desc = self.prose(p.desc)
            elif p.comment:
                desc = f'<span class="cmt" title="from a source comment">{esc(p.comment)}</span>'
            else:
                desc = '<span class="none">\u2014</span>'
            opt = ""
            if p.optional_doc and p.default is None:
                opt = '<span class="pk">optional</span>'
            rows.append(
                f'<tr class="{"req-row" if p.required else ""}">'
                f'<td class="pn"><code>{esc(p.shown_name)}</code>{chips}{opt}</td>'
                f'<td class="pt" data-label="Type">{self.type_cell(p)}</td>'
                f'<td class="pd" data-label="Default">{default}</td>'
                f'<td class="pc" data-label="Comments">{desc}</td></tr>'
            )
        count = len(shown)
        return (
            f'<section class="sec sec-params"><h4 class="sh">{icon("columns")}Parameters<small>{count}</small></h4>'
            f'<div class="tbl"><table class="ptable"><thead><tr><th>Name</th><th>Type</th><th>Default</th><th>Comments</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></section>'
        )

    def attrs_table(self, d: DefInfo) -> str:
        attrs = [a for a in d.attrs if not (a.name.startswith("__") and a.name.endswith("__"))]
        if not attrs:
            return ""
        is_enum = "enum" in d.badges
        rows = []
        scope_label = {
            "class": "class",
            "instance": "instance",
            "field": "field",
            "member": "member",
            "documented": "documented",
        }
        for a in attrs:
            if a.type:
                cls = "ty" if a.type_origin == "annotation" else ("ty from-doc" if a.type_origin == "doc" else "ty inferred")
                tip = {
                    "annotation": "",
                    "doc": ' title="type taken from the docstring"',
                    "inferred": ' title="inferred from the assigned value"',
                }.get(a.type_origin, "")
                sup = "<sup>inferred</sup>" if a.type_origin == "inferred" else ""
                ty = f'<code class="{cls}"{tip}>{fragment(a.type, as_type=True) if a.type_origin != "inferred" else esc(a.type)}{sup}</code>'
            else:
                ty = '<span class="none">\u2014</span>'
            default = f'<code class="dv">{fragment(a.default)}</code>' if a.default is not None else '<span class="none">\u2014</span>'
            desc = self.prose(a.desc) if a.desc else '<span class="none">\u2014</span>'
            where = f' <span class="pk" title="assigned in {esc(a.set_in)}()">in {esc(a.set_in)}</span>' if a.set_in and a.scope == "instance" else ""
            rows.append(
                f'<tr><td class="pn"><code>{esc(a.name)}</code><span class="pk sc-{a.scope}">{scope_label[a.scope]}</span>{where}</td>'
                f'<td class="pt" data-label="Type">{ty}</td><td class="pd" data-label="Default">{default}</td>'
                f'<td class="pc" data-label="Comments">{desc}</td></tr>'
            )
        title = "Members" if is_enum else ("Fields" if "dataclass" in d.badges else "Attributes")
        return (
            f'<section class="sec sec-attrs"><h4 class="sh">{icon("braces")}{title}<small>{len(attrs)}</small></h4>'
            f'<div class="tbl"><table class="ptable"><thead><tr><th>Name</th><th>Type</th><th>{"Value" if is_enum else "Default"}</th><th>Comments</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></section>'
        )

    def methods_table(self, d: DefInfo) -> str:
        methods = [c for c in d.children if not c.is_class]
        if not methods:
            return ""
        rows = []
        for m in methods:
            summary = first_sentence(m.doc.summary) if m.has_docstring else ""
            badges = "".join(
                widgets.badge(_BADGE_LABEL.get(b, b), *_BADGE_KIND.get(b, ("muted", ""))) for b in m.badges if b in _BADGE_KIND and b not in ("recursive",)
            )
            params = [p for p in m.params if not (p is m.params[0] and m.role != "staticmethod" and p.kind in ("posonly", "normal"))]
            sig = ", ".join(p.shown_name for p in params)
            sum_html = self.prose(summary) if summary else '<span class="none">\u2014</span>'
            rows.append(
                f'<tr><td class="mn"><a href="#{m.anchor}"><code>{esc(m.name)}</code><span class="ms">({esc(sig)})</span></a>{badges}</td>'
                f'<td class="pc" data-label="Summary">{sum_html}</td>'
                f'<td class="mc" data-label="Complexity">{widgets.gauge(m.cc, "sm")}</td>'
                f'<td class="ml" data-label="Lines">{m.loc}</td></tr>'
            )
        return (
            f'<section class="sec sec-methods"><h4 class="sh">{icon("function")}Methods<small>{len(methods)}</small></h4>'
            f'{widgets.class_map(d)}'
            f'<div class="tbl"><table class="ptable mtable"><thead><tr><th>Method</th><th>Summary</th><th>CC</th><th>Lines</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></section>'
        )

    # ------------------------------------------------------------------ #
    #  inputs / outputs
    # ------------------------------------------------------------------ #

    def io_strip(self, d: DefInfo) -> str:
        cards: List[str] = []
        doc = d.doc
        # returns -----------------------------------------------------
        ret_type = d.returns or (doc.returns[0].type if doc.returns else "")
        ret_desc = doc.returns[0].desc if doc.returns else ""
        ret_name = doc.returns[0].name if doc.returns else ""
        inferred = False
        if not ret_type and not ret_desc and d.name != "__init__" and not d.is_generator:
            if d.uses.returns == 0 and "abstract" not in d.badges and not d.is_class:
                ret_type, inferred = "None", True
        if d.is_generator:
            y = doc.yields[0] if doc.yields else None
            y_type = (y.type if y else "") or ""
            y_desc = y.desc if y else ""
            ann = d.returns
            inner = ""
            if ann:
                inner = f'<code class="ty">{fragment(ann, as_type=True)}</code>'
            elif y_type:
                inner = f'<code class="ty from-doc">{fragment(y_type, as_type=True)}</code>'
            body = inner + (f"<p>{self.prose(y_desc)}</p>" if y_desc else "")
            if not body:
                body = '<span class="none">generator</span>'
            cards.append(
                f'<div class="io-c io-yield"><div class="io-h">{icon("up-right")}<b>Yields</b></div><div class="io-b">{body}</div></div>'
            )
            if doc.returns and (doc.returns[0].desc or doc.returns[0].type):
                ret_type, ret_desc = doc.returns[0].type, doc.returns[0].desc
            else:
                ret_type = ret_desc = ""
        if ret_type or ret_desc:
            tcls = "ty inferred" if inferred else ("ty" if d.returns else "ty from-doc")
            tip = ' title="no return statement: returns None"' if inferred else ""
            tcode = f'<code class="{tcls}"{tip}>{fragment(ret_type, as_type=True) if not inferred else esc(ret_type)}</code>' if ret_type else ""
            name = f'<code class="rn">{esc(ret_name)}</code> ' if ret_name else ""
            desc = f"<p>{self.prose(ret_desc)}</p>" if ret_desc else ""
            cards.append(
                f'<div class="io-c io-ret{" is-inferred" if inferred else ""}"><div class="io-h">{icon("return")}<b>Returns</b></div>'
                f'<div class="io-b">{name}{tcode}{desc}</div></div>'
            )
        # raises ------------------------------------------------------
        documented = {f.name.split(".")[-1]: f for f in doc.raises if f.name}
        rows = []
        for f in doc.raises:
            names = f.name.replace(" or ", ",").split(",") if f.name else [""]
            for nm in [n.strip() for n in names if n.strip()] or [""]:
                desc = self.prose(f.desc) if f.desc else ""
                rows.append(
                    f'<li><code class="ex">{esc(nm)}</code>{f"<span>{desc}</span>" if desc else ""}</li>'
                )
        for exc, count in d.uses.raises.items():
            if exc.split(".")[-1] in documented or any(exc.split(".")[-1] in (f.name or "") for f in doc.raises):
                continue
            rows.append(f'<li class="from-code"><code class="ex">{esc(exc)}</code><span class="dim">raised in the code</span></li>')
        if rows:
            cards.append(
                f'<div class="io-c io-raise"><div class="io-h">{icon("alert")}<b>Raises</b></div><ul class="io-list">{"".join(rows)}</ul></div>'
            )
        if not cards:
            return ""
        if len(cards) == 1 and "io-ret" in cards[0] and "<p>" not in cards[0]:
            # only a type: one slim line instead of a whole card
            inner = cards[0].split('<div class="io-b">', 1)[1].rsplit("</div></div>", 1)[0]
            return f'<div class="io-inline">{icon("return")}<b>Returns</b>{inner}</div>'
        return f'<div class="io">{"".join(cards)}</div>'

    def touches(self, d: DefInfo) -> str:
        if not d.is_method:
            return ""
        u = d.uses
        writes = [k for k in u.writes if not k.startswith("__")]
        reads = [k for k in u.reads if k not in u.writes and not k.startswith("__")]
        if not writes and not reads:
            return ""
        parts = []
        if writes:
            chips = "".join(f'<code class="tch w">{esc(k)}</code>' for k in writes[:10])
            more = f'<span class="dim">+{len(writes) - 10}</span>' if len(writes) > 10 else ""
            parts.append(f'<div class="tc-row">{icon("edit")}<span class="tc-l">writes</span>{chips}{more}</div>')
        if reads:
            chips = "".join(f'<code class="tch r">{esc(k)}</code>' for k in reads[:10])
            more = f'<span class="dim">+{len(reads) - 10}</span>' if len(reads) > 10 else ""
            parts.append(f'<div class="tc-row">{icon("eye")}<span class="tc-l">reads</span>{chips}{more}</div>')
        return f'<div class="touches"><div class="tc-h">{esc(d.self_name or "self")}.</div>{"".join(parts)}</div>'

    def connections_inner(self, d: DefInfo) -> str:
        svg = widgets.connections(d)
        ext = d.external_calls
        chips = ""
        if ext:
            items = sorted(ext.items(), key=lambda kv: (-kv[1], kv[0]))
            shown = items[:10]
            more = len(items) - len(shown)
            body = "".join(
                '<code class="ext">' + esc(name) + ("<i>\u00d7%d</i>" % n if n > 1 else "") + "</code>" for name, n in shown
            )
            if more > 0:
                body += f'<span class="dim">+{more} more</span>'
            chips = f'<div class="ext-row">{icon("up-right")}<span class="tc-l">also calls</span>{body}</div>'
        return f"{svg}{chips}"

    def insights(self, d: DefInfo, body_stmts: Sequence[ast.stmt]) -> str:
        conn = self.connections_inner(d)
        full = widgets.fingerprint(d, body_stmts) if not d.is_class else ""
        if not conn and not full:
            return ""
        mini = widgets.fingerprint(d, body_stmts, mini=True) if not d.is_class else ""
        callers = len([c for c in d.callers if c is not d])
        facts = []
        if callers:
            facts.append(plural(callers, "caller"))
        if d.callees:
            facts.append(plural(len(d.callees), "local call"))
        if d.external_calls:
            facts.append(plural(len(d.external_calls), "other call"))
        if d.uses.returns > 1:
            facts.append(plural(d.uses.returns, "return"))
        sep = " \u00b7 "
        summary = "<small>" + sep.join(facts) + "</small>" if facts else ""
        cols = []
        if conn:
            cols.append(f'<div class="ins-col"><h5>{icon("git-merge")}Connections</h5>{conn}</div>')
        if full:
            cols.append(f'<div class="ins-col"><h5>{icon("hash")}Shape<small>top-level statements</small></h5>{full}</div>')
        opened = " open" if (d.cc >= 6 or d.loc >= 32) else ""
        return (
            f'<details class="insights"{opened}><summary>{icon("chev", "chv")}{icon("sparkle")}<b>Insights</b>{mini}{summary}</summary>'
            f'<div class="ins-body">{"".join(cols)}</div></details>'
        )

    # ------------------------------------------------------------------ #
    #  the card
    # ------------------------------------------------------------------ #

    def render(self, d: DefInfo, ctx: Ctx, next_row: int) -> str:
        src = self.src
        node = d.node
        is_cls = d.is_class
        stmts: Sequence[ast.stmt] = node.body  # type: ignore[attr-defined]
        colon = src.find_colon((node.lineno, node.col_offset)) or (node.lineno, 0)  # type: ignore[attr-defined]
        header_tail = src.take_inline(colon[0], colon[1] + 1) if not (stmts and stmts[0].lineno == colon[0]) else None
        # header tokens (signature) are rendered from the model, not from source
        src.exempt((node.lineno, node.col_offset), (colon[0], colon[1] + 1))  # type: ignore[attr-defined]

        body_stmts = list(stmts)
        start_row = colon[0]
        if d.docstring_node is not None:
            body_stmts = body_stmts[1:]
            start_row = d.docstring_node.end_lineno

        # ---- header ----
        role = d.role if d.role in _KIND_ICON else ("class" if is_cls else "function")
        kind_cls = _KIND_CLASS.get(role, "k-fn")
        qual_prefix = ""
        if d.parent is not None:
            qual_prefix = f'<span class="qp">{esc(d.parent.qualname)}.</span>'
        summary = ""
        if d.has_docstring and d.doc.summary:
            summary = f'<p class="sum">{self.prose(first_sentence(d.doc.summary, 220))}</p>'
        stats = self._stats(d)
        bases = self._bases(d) if is_cls else ""
        sig = fragment(self.signature_text(d))
        tail = f'<span class="tc hdr-note">{esc(header_tail.body)}</span>' if header_tail is not None and header_tail.body else ""
        header = (
            f'<header class="dh"><div class="dh-top"><span class="kind {kind_cls}" title="{esc(role)}">{icon(_KIND_ICON.get(role, "function"))}</span>'
            f'<div class="dh-title">{self.decorators(d)}<h3 class="dname">{qual_prefix}<b class="nm">{esc(d.name)}</b></h3>'
            f'<div class="badges">{self.badges(d)}</div></div>'
            f'<div class="dstats">{stats}</div>'
            f'<button class="cpy" type="button" aria-label="Copy source" title="Copy source">{icon("copy")}</button>'
            f'<button class="tog" type="button" aria-label="Collapse or expand" title="Collapse / expand">{icon("chev")}</button></div>'
            f'{bases}<pre class="sig"><code>{sig}</code>{tail}</pre>{summary}</header>'
        )

        # ---- body ----
        parts: List[str] = []
        doc = d.doc
        if doc and (doc.description_blocks or doc.sections):
            desc_html = blocks_html(doc.description_blocks, self.link)
            sect_html = "".join(section_html(s, self.link) for s in doc.sections)
            long = len(doc.raw) > 900
            inner = f'<div class="prose">{desc_html}{sect_html}</div>'
            if long:
                parts.append(
                    f'<details class="more"><summary>{icon("chev", "chv")}Full description<small>{len(doc.raw.splitlines())} lines</small></summary>{inner}</details>'
                )
            else:
                parts.append(f'<section class="sec sec-doc">{inner}</section>')
        schema = self.r.neural_schemas.get(id(node))
        if is_cls:
            if schema is not None:
                parts.append(render_neural(schema))
            parts.append(self.attrs_table(d))
            parts.append(self.methods_table(d))
        else:
            parts.append(self.params_table(d))
            if schema is not None:
                parts.append(render_neural(schema))
            parts.append(self.io_strip(d))
            parts.append(self.touches(d))
        parts.append(self.insights(d, body_stmts))

        # ---- implementation ----
        impl = ""
        if body_stmts:
            indent = src.start(body_stmts[0])[1]
            c = Ctx(
                indent=indent,
                start_row=start_row,
                end_row=next_row,
                split_depth=0,
                depth=ctx.depth + 1,
                fold_attrs=is_cls,
            )
            flow_html = self.r.flow.body(body_stmts, c)
            label = "Class body" if is_cls else "Implementation"
            ic = "layers" if is_cls else "terminal"
            count = f"{d.loc} lines"
            if not is_cls and len(body_stmts) <= 3 and d.loc <= 5:
                impl = f'<div class="impl tiny"><div class="flow">{flow_html}</div></div>'
            else:
                impl = (
                    f'<details class="impl" open><summary class="ih">{icon("chev", "chv")}{icon(ic)}<b>{label}</b><small>{count}</small></summary>'
                    f'<div class="flow{" cls-flow" if is_cls else ""}">{flow_html}</div></details>'
                )
        depth_cls = f" nested n{min(ctx.depth, 4)}" if d.parent is not None else ""
        return (
            f'<article class="card def {"cls" if is_cls else "fn"} g-{d.grade}{depth_cls}" id="{d.anchor}" '
            f'data-name="{attr(d.qualname)}" data-l1="{d.first_line}" data-l2="{d.last_line}">'
            f'{header}<div class="cb">{"".join(p for p in parts if p)}{impl}</div></article>'
        )

    def _bases(self, d: DefInfo) -> str:
        chips = []
        for b in d.bases:
            base = b.split("[")[0].split(".")[-1]
            target = self.r.mod_class(base)
            if target is not None:
                chips.append(f'<a class="base local" href="#{target.anchor}"><code>{esc(b)}</code></a>')
            else:
                chips.append(f'<span class="base"><code>{esc(b)}</code></span>')
        for k in d.class_keywords:
            chips.append(f'<span class="base kw"><code>{esc(k)}</code></span>')
        subs = self.r.subclasses(d)
        parts = []
        if chips:
            parts.append(f'<span class="rel-l">{icon("up-right")}extends</span>{"".join(chips)}')
        if subs:
            links = "".join(f'<a class="base local" href="#{s.anchor}"><code>{esc(s.name)}</code></a>' for s in subs)
            parts.append(f'<span class="rel-l">{icon("git-merge")}subclasses</span>{links}')
        return f'<div class="rel">{"".join(parts)}</div>' if parts else ""

    def _stats(self, d: DefInfo) -> str:
        if d.is_class:
            methods = [c for c in d.children if not c.is_class]
            worst = max((m.cc for m in methods), default=1)
            bits = [
                f'<span class="stat" title="methods"><b>{len(methods)}</b><small>methods</small></span>',
                f'<span class="stat" title="attributes"><b>{len([a for a in d.attrs if not a.name.startswith("__")])}</b><small>attrs</small></span>',
                f'<span class="stat" title="lines of code"><b>{d.loc}</b><small>lines</small></span>',
            ]
            if methods:
                bits.append(widgets.gauge(worst))
            return "".join(bits)
        n_params = d.total_params
        return (
            f'<span class="stat" title="parameters"><b>{n_params}</b><small>params</small></span>'
            f'<span class="stat" title="lines of code"><b>{d.loc}</b><small>lines</small></span>'
            f"{widgets.gauge(d.cc)}"
        )
