"""Render statically inferred model execution paths as inline SVG diagrams and readable details."""
from __future__ import annotations

import re
from typing import Dict, List

from ..neural import Route, Schema, Step, format_shape
from .icons import icon
from .util import attr, esc

_DETECTION = {
    "pytorch": "PyTorch module",
    "inherited": "inherits local module",
    "lookalike": "module-like pattern",
    "factory": "sequential factory",
}

_SVG_W = 1040
_NODE_X = 155
_NODE_W = 780
_NODE_H = 62
_TOP = 102
_GAP = 32


def _source_name(ident: str, names: Dict[str, str]) -> str:
    if ident == "input":
        return "input"
    return names.get(ident, "unknown")


def _step(step: Step, names: Dict[str, str]) -> str:
    sources = step.incoming or ("input",)
    if len(sources) == 1:
        source = f'<code>{esc(_source_name(sources[0], names))}</code>'
    else:
        source = " <span class=\"nn-join\">+</span> ".join(f'<code>{esc(_source_name(s, names))}</code>' for s in sources)
    title = "Elementwise merge" if step.merge else ("Custom module (details are not statically expanded)" if step.kind not in ("Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d") and step.params == "custom module" else step.kind)
    tone = " nn-merge" if step.merge else ""
    params = f'<small class="nn-params">{esc(step.params)}</small>' if step.params else ""
    line = f'<a class="nn-line" href="#L{step.line}" title="Jump to source line {step.line}">L{step.line}</a>' if step.line else ""
    return (
        f'<div class="nn-stage{tone}" title="{attr(title)}">'
        f'<div class="nn-from"><span>from</span>{source}<i class="nn-arrow" aria-hidden="true">{icon("arrow-r")}</i></div>'
        f'<div class="nn-node"><div class="nn-node-top"><b>{esc(step.name)}</b>{line}</div>'
        f'<strong>{esc(step.kind)}</strong>{params}'
        f'<div class="nn-shapes"><span>{esc(format_shape(step.in_shape))}</span><i aria-hidden="true">{icon("arrow-r")}</i><b>{esc(format_shape(step.out_shape))}</b></div></div></div>'
    )


def _clip(text: str, size: int) -> str:
    return text if len(text) <= size else text[: size - 1] + "…"


def _svg(route: Route, uid: str) -> str:
    """A responsive, offline SVG of a route's DAG, including skip/merge connections."""
    steps = route.steps
    if not steps:
        return ""
    names = {step.ident: step.name for step in steps}
    index = {step.ident: i for i, step in enumerate(steps)}
    y = [_TOP + i * (_NODE_H + _GAP) for i in range(len(steps))]
    output_y = _TOP + len(steps) * (_NODE_H + _GAP) + 2
    height = output_y + 58
    cx = _NODE_X + _NODE_W / 2
    edge_bits: List[str] = []
    skip_edge_count = 0
    skip_color_count = 8
    skip_colors: set[int] = set()
    farthest_rail = 0
    for target_i, step in enumerate(steps):
        incoming = step.incoming or ("input",)
        for branch_i, producer in enumerate(incoming):
            if producer == "input" or producer not in index:
                if target_i == 0:
                    edge_bits.append(f'<path class="nn-edge" d="M {cx:g} 67 V {y[target_i]}" marker-end="url(#{uid}-arrow)"/>')
                else:
                    rail = _NODE_X - 18 - branch_i * 12
                    edge_bits.append(f'<path class="nn-edge" d="M {cx:g} 67 H {rail} V {y[target_i] + _NODE_H / 2} H {_NODE_X}" marker-end="url(#{uid}-arrow)"/>')
                continue
            parent_i = index[producer]
            if parent_i == target_i - 1 and len(incoming) == 1:
                edge_bits.append(f'<path class="nn-edge" d="M {cx:g} {y[parent_i] + _NODE_H} V {y[target_i]}" marker-end="url(#{uid}-arrow)"/>')
            else:
                rail = _NODE_X + _NODE_W + 18 + skip_edge_count * 12
                farthest_rail = max(farthest_rail, rail)
                source_y = y[parent_i] + _NODE_H / 2
                target_y = y[target_i] + _NODE_H / 2
                color_index = skip_edge_count % skip_color_count
                skip_edge_count += 1
                skip_colors.add(color_index)
                edge_bits.append(
                    f'<path class="nn-edge nn-skip nn-flow-{color_index}" d="M {_NODE_X + _NODE_W} {source_y} H {rail} V {target_y} H {_NODE_X + _NODE_W}" marker-end="url(#{uid}-arrow-skip-{color_index})"/>'
                )
    skip_markers = "".join(
        f'<marker class="nn-arrowhead-{i}" id="{uid}-arrow-skip-{i}" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z"/></marker>'
        for i in sorted(skip_colors)
    )
    nodes: List[str] = []
    for i, step in enumerate(steps):
        top = y[i]
        source_names = [_source_name(p, names) for p in step.incoming if p in names or p == "input"]
        source_text = " + ".join(source_names) or "input"
        node_class = "nn-svg-node nn-svg-merge" if step.merge else "nn-svg-node"
        line = f'<text class="nn-svg-line" x="{_NODE_X + _NODE_W - 16}" y="{top + 23}" text-anchor="end">L{step.line}</text>' if step.line else ""
        nodes.append(
            f'<g class="{node_class}"><title>{esc(step.name)} · {esc(step.kind)} · {esc(format_shape(step.in_shape))} to {esc(format_shape(step.out_shape))}</title>'
            f'<rect x="{_NODE_X}" y="{top}" width="{_NODE_W}" height="{_NODE_H}" rx="12"/>'
            f'<text class="nn-svg-name" x="{_NODE_X + 18}" y="{top + 25}">{esc(_clip(step.name, 30))}</text>{line}'
            f'<text class="nn-svg-kind" x="{_NODE_X + 230}" y="{top + 25}">{esc(_clip(step.kind, 24))}</text>'
            f'<text class="nn-svg-params" x="{_NODE_X + 400}" y="{top + 25}">{esc(_clip(step.params or "", 36))}</text>'
            f'<text class="nn-svg-shape" x="{_NODE_X + 18}" y="{top + 48}">{esc(_clip(format_shape(step.in_shape), 38))}</text>'
            f'<text class="nn-svg-arrow" x="{_NODE_X + 370}" y="{top + 48}">→</text>'
            f'<text class="nn-svg-shape nn-svg-out" x="{_NODE_X + 410}" y="{top + 48}">{esc(_clip(format_shape(step.out_shape), 38))}</text>'
            f'<text class="nn-svg-from" x="{_NODE_X + _NODE_W - 16}" y="{top + 48}" text-anchor="end">from {esc(_clip(source_text, 25))}</text>'
            '</g>'
        )
    output_label = "OUTPUT · " + "  |  ".join(_clip(format_shape(shape), 42) for shape in route.output_shapes)
    if not route.output_shapes:
        output_label = "OUTPUT · shape not statically determined"
    outs = f'<text class="nn-svg-output" x="{cx:g}" y="{output_y + 26}" text-anchor="middle">{esc(output_label)}</text>'
    view_width = max(_SVG_W, farthest_rail + 20)
    return (
        f'<div class="nn-svg-wrap"><svg class="nn-svg" viewBox="0 0 {view_width} {height}" role="img" '
        f'aria-label="{attr(f"{route.method_name} model path: {len(steps)} layers, input {format_shape(route.input_shape)}")}">'
        f'<title>{esc(route.method_name)} model path</title><desc>Layer graph. Each node shows input and output tensor dimensions; arrows show data flow.</desc>'
        f'<defs><marker id="{uid}-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z"/></marker>{skip_markers}</defs>'
        f'<rect class="nn-svg-port" x="{cx - 260:g}" y="12" width="520" height="54" rx="13"/>'
        f'<text class="nn-svg-port-label" x="{cx:g}" y="33" text-anchor="middle">INPUT · {esc(route.input_name)}</text>'
        f'<text class="nn-svg-port-shape" x="{cx:g}" y="53" text-anchor="middle">{esc(format_shape(route.input_shape))}</text>'
        f'<g class="nn-svg-edges">{"".join(edge_bits)}</g>{"".join(nodes)}'
        f'<rect class="nn-svg-port nn-svg-outport" x="{cx - 260:g}" y="{output_y}" width="520" height="42" rx="13"/>{outs}'
        '</svg></div>'
    )


def _route(route: Route, uid: str) -> str:
    names = {s.ident: s.name for s in route.steps}
    count = f'{len(route.steps)} layer step' if len(route.steps) == 1 else f'{len(route.steps)} layer steps'
    suffix = "()" if route.kind == "method" else ""
    heading = f'<h5 class="nn-route-title"><code>{esc(route.method_name)}{suffix}</code><span>{count}</span></h5>'
    svg = _svg(route, uid)
    detailed = "".join(_step(step, names) for step in route.steps)
    outputs = "".join(
        f'<span class="nn-output"><b>output</b><code>{esc(format_shape(shape))}</code></span>'
        for shape in route.output_shapes
    )
    detail = (
        f'<details class="nn-detail"><summary>Detailed connections and parameters</summary>'
        f'<div class="nn-flow"><div class="nn-input"><span class="nn-input-label">INPUT</span><b>{esc(route.input_name)}</b>'
        f'<code>{esc(format_shape(route.input_shape))}</code></div>{detailed}{outputs}</div></details>'
        if route.steps
        else ""
    )
    notes = "".join(f'<li>{esc(note)}</li>' for note in route.notes)
    note_html = f'<ul class="nn-notes">{notes}</ul>' if notes else ""
    return f'<div class="nn-route">{heading}{svg}{detail}{note_html}</div>'


def render(schema: Schema) -> str:
    """HTML block with an inline SVG and an optional text-first trace for each execution method."""
    detection = _DETECTION.get(schema.detection, "static inference")
    status = f'<span class="nn-detect" title="{attr(schema.reason)}">{icon("layers")}<b>{esc(detection)}</b></span>'
    routes = schema.routes or ([Route(schema.method_name, schema.input_name, schema.input_shape, schema.steps, schema.output_shapes, schema.notes)] if schema.steps else [])
    total_steps = sum(len(r.steps) for r in routes)
    summary = f'<span class="nn-count">{len(routes)} execution {"paths" if len(routes) != 1 else "path"} · {total_steps} layer steps</span>' if total_steps else '<span class="nn-count">no traceable path</span>'
    path_html = "".join(_route(route, f"nn-{i}-{re.sub(r'[^a-zA-Z0-9_-]', '-', schema.name)}") for i, route in enumerate(routes))
    empty_notes = "".join(f'<li>{esc(note)}</li>' for note in schema.notes) if not routes else ""
    note_html = f'<ul class="nn-notes">{empty_notes}</ul>' if empty_notes else ""
    return (
        f'<section class="sec nn-schema" aria-label="Static neural network schema for {attr(schema.name)}">'
        f'<div class="nn-head"><h4 class="sh">{icon("network")}Model schema</h4>{status}{summary}</div>'
        f'<p class="nn-intro">Inferred statically from <code>__init__</code> and method contents; model code is never run. Shapes are symbolic where dimensions are unknown.</p>'
        f'{path_html}{note_html}'
        f'<p class="nn-foot">Best-effort schema; dynamic control flow, data-dependent shapes and custom modules may not be fully expanded.</p>'
        f'</section>'
    )
