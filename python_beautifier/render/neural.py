"""Render a statically inferred neural-network schema inside a class card."""
from __future__ import annotations

from typing import Dict

from ..neural import Schema, Step, format_shape
from .icons import icon
from .util import attr, esc


_DETECTION = {
    "pytorch": "PyTorch module",
    "inherited": "inherits local module",
    "lookalike": "module-like pattern",
}


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


def render(schema: Schema) -> str:
    """HTML block for a detected model, even when no layer flow can be inferred."""
    names = {s.ident: s.name for s in schema.steps}
    detection = _DETECTION.get(schema.detection, "static inference")
    input_html = (
        f'<div class="nn-input"><span class="nn-input-label">INPUT</span><b>{esc(schema.input_name)}</b>'
        f'<code>{esc(format_shape(schema.input_shape))}</code></div>'
    )
    steps = "".join(_step(s, names) for s in schema.steps)
    outs = "".join(f'<span class="nn-output"><b>output</b><code>{esc(format_shape(shape))}</code></span>' for shape in schema.output_shapes)
    notes = "".join(f'<li>{esc(note)}</li>' for note in schema.notes)
    note_html = f'<ul class="nn-notes">{notes}</ul>' if notes else ""
    status = f'<span class="nn-detect" title="{attr(schema.reason)}">{icon("layers")}<b>{esc(detection)}</b></span>'
    summary = f'<span class="nn-count">{len(schema.steps)} traced steps</span>' if schema.steps else '<span class="nn-count">no traced layer calls</span>'
    return (
        f'<section class="sec nn-schema" aria-label="Static neural network schema for {attr(schema.name)}">'
        f'<div class="nn-head"><h4 class="sh">{icon("network")}Model schema</h4>{status}{summary}</div>'
        f'<p class="nn-intro">Inferred statically from <code>__init__</code> and <code>forward()</code>; no model code is run. Shapes are symbolic where dimensions are unknown.</p>'
        f'<div class="nn-flow" role="img" aria-label="Input flows through {len(schema.steps)} statically inferred layer steps">'
        f'{input_html}{steps}{outs}</div>{note_html}'
        f'<p class="nn-foot">Best-effort schema; dynamic control flow, data-dependent shapes and custom modules may not be fully expanded.</p>'
        f'</section>'
    )
