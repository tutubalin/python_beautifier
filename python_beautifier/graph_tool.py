"""A compact, standalone HTML graph explorer for Python model schemas."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

from .neural import Route, analyze
from .render.icons import sprite
from .render.neural import render as render_schema
from .render.util import attr, esc
from .source import Source, decode_source

_ASSETS = Path(__file__).with_name("assets")


def _asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def _styles() -> str:
    base = _asset("base.css")
    neural = _asset("neural.css")
    dark = re.search(r"/\*DARK\{\*/(.*?)/\*\}DARK\*/", base, re.S)
    auto_dark = ""
    if dark:
        tokens = dark.group(1).replace('html[data-theme="dark"]', 'html[data-theme="auto"]')
        auto_dark = "@media (prefers-color-scheme: dark) {" + tokens + "}"
    shell = r"""
:root { --nn: #0d9488; }
.graph-topbar { position: sticky; z-index: 20; top: 0; display: flex; align-items: center; gap: 14px; min-height: 58px; padding: 8px 20px; border-bottom: 1px solid var(--line); background: color-mix(in oklab, var(--surface) 92%, transparent); backdrop-filter: blur(12px); }
.graph-brand { color: var(--ink); font: 800 13px var(--font-ui); text-decoration: none; }
.graph-brand span { color: var(--nn); }
.graph-file { min-width: 0; overflow: hidden; color: var(--ink-3); font: 11px var(--font-mono); text-overflow: ellipsis; white-space: nowrap; }
.graph-theme { margin-left: auto; padding: 6px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); color: var(--ink-2); cursor: pointer; font: 600 11px var(--font-ui); }
.graph-layout { display: grid; grid-template-columns: 230px minmax(0, 1fr); align-items: start; gap: 16px; }
.graph-index { position: sticky; top: 72px; max-height: calc(100vh - 88px); overflow: auto; margin: 14px 0 14px 14px; padding: 12px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
.graph-index h2 { margin: 0 0 8px; color: var(--ink-2); font: 800 10px var(--font-ui); letter-spacing: .08em; text-transform: uppercase; }
.graph-index a { display: block; overflow: hidden; padding: 6px 8px; border-radius: 7px; color: var(--ink-2); font: 11px/1.4 var(--font-mono); text-decoration: none; text-overflow: ellipsis; white-space: nowrap; }
.graph-index a:hover, .graph-index a:focus-visible { background: var(--surface-3); color: var(--accent); outline: none; }
.graph-index .graph-nav-class { margin: 8px 0 2px; padding: 4px 8px; color: var(--nn); font: 800 10px var(--font-ui); }
.graph-main { min-width: 0; padding: 8px 0 56px; }
.graph-page-heading { width: min(1480px, calc(100% - 38px)); margin: 18px auto 8px; color: var(--ink); font: 800 23px var(--font-ui); }
.graph-page-intro { width: min(1480px, calc(100% - 38px)); margin: 0 auto 16px; color: var(--ink-3); font: 12px/1.55 var(--font-ui); }
.graph-model { width: min(1480px, calc(100% - 38px)); margin: 16px auto 24px; scroll-margin-top: 74px; }
.graph-schema-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin: 0 0 8px; padding: 0 3px; }
.graph-schema-head h2 { margin: 0; color: var(--ink); font: 800 18px var(--font-ui); }
.graph-schema-head h2 code { font: inherit; }
.graph-schema-head > span { padding: 3px 8px; border-radius: 99px; background: color-mix(in oklab, var(--nn) 11%, var(--surface)); color: var(--nn); font: 700 10px var(--font-ui); }
.graph-model > .nn-schema { width: 100%; }
.graph-empty { width: min(900px, calc(100% - 38px)); margin: 24px auto; padding: 18px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); color: var(--ink-2); font: 13px/1.6 var(--font-ui); }
.graph-empty code { font: 11px var(--font-mono); }
@media (max-width: 820px) {
  .graph-layout { grid-template-columns: minmax(0, 1fr); }
  .graph-index { position: static; max-height: 190px; margin: 10px 14px 0; }
  .graph-main { padding-top: 0; }
  .graph-page-heading, .graph-page-intro, .graph-model { width: calc(100% - 24px); }
}
"""
    return "\n".join((base, neural, shell, auto_dark))


def _graph_script() -> str:
    return r"""(function(){
  "use strict";
  var root=document.documentElement, doc=document;
  var theme=doc.getElementById("graph-theme");
  function setTheme(value){root.setAttribute("data-theme",value);try{localStorage.setItem("pb-theme",value);}catch(e){}}
  if(theme){try{var saved=localStorage.getItem("pb-theme");if(saved==="light"||saved==="dark")root.setAttribute("data-theme",saved);}catch(e){}theme.addEventListener("click",function(){setTheme(root.getAttribute("data-theme")==="dark"?"light":"dark");});}
  function reveal(id){var el=id&&doc.getElementById(id),p=el;while(p){if(p.tagName==="DETAILS")p.open=true;p=p.parentElement;}}
  function revealHash(){if(location.hash)reveal(decodeURIComponent(location.hash.slice(1)));}
  window.addEventListener("hashchange",revealHash);revealHash();
  doc.addEventListener("click",function(e){var a=e.target.closest&&e.target.closest("a[href^='#']");if(a&&a.getAttribute("href").length>1)setTimeout(function(){reveal(decodeURIComponent(a.getAttribute("href").slice(1)));},0);});
  doc.querySelectorAll(".nn-svg-wrap-2d").forEach(function(wrap){
    var drag=null;
    function stop(e){if(!drag)return;wrap.classList.remove("is-panning");if(wrap.hasPointerCapture&&wrap.hasPointerCapture(e.pointerId))wrap.releasePointerCapture(e.pointerId);drag=null;}
    wrap.addEventListener("pointerdown",function(e){if(e.button!==0||(e.target.closest&&e.target.closest("a")))return;drag={x:e.clientX,y:e.clientY,left:wrap.scrollLeft,top:wrap.scrollTop};wrap.classList.add("is-panning");if(wrap.setPointerCapture)wrap.setPointerCapture(e.pointerId);e.preventDefault();});
    wrap.addEventListener("pointermove",function(e){if(!drag)return;wrap.scrollLeft=drag.left-(e.clientX-drag.x);wrap.scrollTop=drag.top-(e.clientY-drag.y);});
    wrap.addEventListener("pointerup",stop);wrap.addEventListener("pointercancel",stop);
  });
})();"""


def beautify_graph(
    source: Union[str, bytes],
    *,
    filename: str = "<string>",
    title: Optional[str] = None,
    theme: str = "auto",
) -> str:
    """Render only statically inferred model graphs as a small, self-contained HTML page."""
    text = decode_source(source) if isinstance(source, bytes) else source
    try:
        schemas = list(analyze(Source(text, filename)).values())
        problem = ""
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        schemas = []
        problem = f"Could not parse this Python source: {exc}"

    page_title = title or (Path(filename).name if filename not in ("<string>", "<stdin>") else "Model graphs")
    theme_value = theme if theme in ("auto", "light", "dark") else "auto"
    content: list[str] = []
    index_links: list[str] = []
    if schemas:
        for schema_index, schema in enumerate(schemas):
            slug = re.sub(r"[^a-zA-Z0-9_-]", "-", schema.name)
            routes = schema.routes or ([Route(schema.method_name, schema.input_name, schema.input_shape, schema.steps, schema.output_shapes, schema.notes)] if schema.steps else [])
            if routes:
                for route_index, route in enumerate(routes):
                    route_id = f"nn-{route_index}-{slug}"
                    index_links.append(f'<a href="#{attr(route_id)}">{esc(schema.name)} · {esc(route.method_name)}</a>')
            else:
                index_links.append(f'<a href="#graph-model-{schema_index}">{esc(schema.name)}</a>')
            detection = {
                "pytorch": "PyTorch module",
                "inherited": "local module inheritance",
                "lookalike": "module-like class",
                "factory": "sequential factory",
            }.get(schema.detection, "static model schema")
            schema_header = (
                f'<header class="graph-schema-head"><h2><code>{esc(schema.name)}</code></h2>'
                f'<span>{esc(detection)}</span></header>'
            )
            content.append(
                f'<div class="graph-model" id="graph-model-{schema_index}">{schema_header}{render_schema(schema)}</div>'
            )
    else:
        text_message = problem or "No statically traceable neural model schemas were found in this file."
        content.append(
            f'<div class="graph-empty"><b>{esc(text_message)}</b><br>'
            'The graph tool looks for <code>torch.nn.Module</code> classes, local module inheritance, '
            'and classes with module-like layer calls. Model code is analyzed, never executed.</div>'
        )

    index_html = (
        '<aside class="graph-index"><h2>Graphs &amp; routes</h2>' + "".join(index_links) + "</aside>"
        if index_links else '<aside class="graph-index"><h2>Graphs</h2><a href="#main">No graphs detected</a></aside>'
    )
    count = f"{len(schemas)} model {'schema' if len(schemas) == 1 else 'schemas'}"
    return (
        '<!doctype html>\n<html lang="en" data-theme="' + attr(theme_value) + '">\n<head>\n'
        '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="generator" content="python_beautifier graph view">\n<title>{esc(page_title)} · graphs</title>\n'
        f'<style>{_styles()}</style>\n</head>\n<body>\n{sprite()}'
        '<header class="graph-topbar"><a class="graph-brand" href="#main">Python Beautifier <span>· Graphs</span></a>'
        f'<span class="graph-file" title="{attr(filename)}">{esc(filename)} · {esc(count)}</span>'
        '<button class="graph-theme" id="graph-theme" type="button">Toggle theme</button></header>'
        f'<div class="graph-layout">{index_html}<main class="main graph-main" id="main">'
        f'<h1 class="graph-page-heading">{esc(page_title)}</h1>'
        '<p class="graph-page-intro">A focused view of statically inferred model execution graphs. '
        'Drag a layered 2D graph to pan; expand custom blocks to inspect their internals.</p>'
        f'{"".join(content)}</main></div>'
        f'<script>{_graph_script()}</script>\n</body>\n</html>\n'
    )
