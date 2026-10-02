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


def _module_id(uid: str, step: Step) -> str:
    return f"{uid}-module-{step.ident}"


def _step(step: Step, names: Dict[str, str], uid: str) -> str:
    sources = step.incoming or ("input",)
    if len(sources) == 1:
        source = f'<code>{esc(_source_name(sources[0], names))}</code>'
    else:
        source = " <span class=\"nn-join\">+</span> ".join(f'<code>{esc(_source_name(s, names))}</code>' for s in sources)
    title = "Elementwise merge" if step.merge else step.kind
    tone = " nn-merge" if step.merge else ""
    params = f'<small class="nn-params">{esc(step.params)}</small>' if step.params else ""
    line = f'<a class="nn-line" href="#L{step.line}" title="Jump to source line {step.line}">L{step.line}</a>' if step.line else ""
    expanded = ""
    if step.children:
        child_names = {child.ident: child.name for child in step.children}
        child_uid = f"{uid}-{step.ident}"
        child_flow = "".join(_step(child, child_names, child_uid) for child in step.children)
        child_route = Route(step.kind, step.name, step.in_shape, step.children, [step.out_shape], [])
        child_svg = _svg(child_route, child_uid) if _needs_2d(child_route) else ""
        expanded = (
            f'<details class="nn-module-expand" id="{attr(_module_id(uid, step))}">'
            f'<summary><b>{esc(step.kind)}</b><span>Expand internals · {len(step.children)} layer steps</span></summary>'
            f'<div class="nn-module-body">{child_svg}<div class="nn-input"><span class="nn-input-label">MODULE INPUT</span>'
            f'<b>{esc(step.name)}</b><code>{esc(format_shape(step.in_shape))}</code></div>'
            f'<div class="nn-flow">{child_flow}</div>'
            f'<div class="nn-output"><b>module output</b><code>{esc(format_shape(step.out_shape))}</code></div></div>'
            f'</details>'
        )
    return (
        f'<div class="nn-stage{tone}" title="{attr(title)}">'
        f'<div class="nn-from"><span>from</span>{source}<i class="nn-arrow" aria-hidden="true">{icon("arrow-r")}</i></div>'
        f'<div class="nn-node"><div class="nn-node-top"><b>{esc(step.name)}</b>{line}</div>'
        f'<strong>{esc(step.kind)}</strong>{params}'
        f'<div class="nn-shapes"><span>{esc(format_shape(step.in_shape))}</span><i aria-hidden="true">{icon("arrow-r")}</i><b>{esc(format_shape(step.out_shape))}</b></div></div>'
        f'{expanded}</div>'
    )


def _clip(text: str, size: int) -> str:
    return text if len(text) <= size else text[: size - 1] + "…"


def _rounded_path(points: List[tuple[float, float]], radius: float = 7) -> str:
    """Build an orthogonal SVG route with visibly rounded elbows."""
    if not points:
        return ""
    fmt = lambda value: f"{value:g}"
    commands = [f"M {fmt(points[0][0])} {fmt(points[0][1])}"]
    for i in range(1, len(points) - 1):
        px, py = points[i - 1]
        x, y = points[i]
        nx, ny = points[i + 1]
        in_dx, in_dy = (0 if px == x else (1 if px > x else -1)), (0 if py == y else (1 if py > y else -1))
        out_dx, out_dy = (0 if nx == x else (1 if nx > x else -1)), (0 if ny == y else (1 if ny > y else -1))
        in_len, out_len = abs(px - x) + abs(py - y), abs(nx - x) + abs(ny - y)
        if in_len == 0 or out_len == 0 or in_dx * out_dy == in_dy * out_dx:
            commands.append(f"L {fmt(x)} {fmt(y)}")
            continue
        corner_radius = min(radius, in_len / 2, out_len / 2)
        before_x, before_y = x + in_dx * corner_radius, y + in_dy * corner_radius
        after_x, after_y = x + out_dx * corner_radius, y + out_dy * corner_radius
        commands.append(
            f"L {fmt(before_x)} {fmt(before_y)} Q {fmt(x)} {fmt(y)} {fmt(after_x)} {fmt(after_y)}"
        )
    commands.append(f"L {fmt(points[-1][0])} {fmt(points[-1][1])}")
    return " ".join(commands)


def _svg_linear(route: Route, uid: str) -> str:
    """Compact, vertical SVG for a simple model path."""
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
    skip_paths: List[tuple[int, int, str]] = []
    skip_color_count = 8
    skip_colors: set[int] = set()
    skip_color_by_source: Dict[str, int] = {}
    skip_lane_by_source: Dict[str, int] = {}
    farthest_rail = 0
    for target_i, step in enumerate(steps):
        incoming = step.incoming or ("input",)
        skip_producers = [
            producer for producer in incoming
            if producer in index and index[producer] != target_i - 1
        ]
        endpoint_offsets = {
            producer: (rank - (len(skip_producers) - 1) / 2) * 5
            for rank, producer in enumerate(skip_producers)
        }
        for producer in incoming:
            if producer == "input" or producer not in index:
                if target_i == 0:
                    edge_bits.append(f'<path class="nn-edge" d="M {cx:g} 67 V {y[target_i]}" marker-end="url(#{uid}-arrow)"/>')
                else:
                    # Give input branches a short downward tail before they turn toward
                    # a later node; without it the route appears to sprout sideways
                    # directly from the input port.
                    rail = _NODE_X - 18  # all branches from the shared input port use one trunk
                    stub_y = 77
                    target_y = y[target_i] + _NODE_H / 2
                    d = _rounded_path([(cx, 67), (cx, stub_y), (rail, stub_y), (rail, target_y), (_NODE_X, target_y)])
                    edge_bits.append(f'<path class="nn-edge" d="{d}" marker-end="url(#{uid}-arrow)"/>')
                continue
            parent_i = index[producer]
            if parent_i == target_i - 1:
                edge_bits.append(f'<path class="nn-edge" d="M {cx:g} {y[parent_i] + _NODE_H} V {y[target_i]}" marker-end="url(#{uid}-arrow)"/>')
            else:
                source_y = y[parent_i] + _NODE_H / 2
                target_y = y[target_i] + _NODE_H / 2 + endpoint_offsets.get(producer, 0)
                if producer not in skip_lane_by_source:
                    skip_lane_by_source[producer] = len(skip_lane_by_source)
                lane_index = skip_lane_by_source[producer]
                rail = _NODE_X + _NODE_W + 18 + lane_index * 12
                farthest_rail = max(farthest_rail, rail)
                if producer not in skip_color_by_source:
                    skip_color_by_source[producer] = len(skip_color_by_source) % skip_color_count
                color_index = skip_color_by_source[producer]
                skip_colors.add(color_index)
                d = _rounded_path([
                    (_NODE_X + _NODE_W, source_y),
                    (rail, source_y),
                    (rail, target_y),
                    (_NODE_X + _NODE_W, target_y),
                ])
                path = f'<path class="nn-edge nn-skip nn-flow-{color_index}" d="{d}" data-source="{attr(producer)}" data-target="{attr(step.ident)}" data-lane="{lane_index}" marker-end="url(#{uid}-arrow-skip-{color_index})"/>'
                path_length = abs(target_i - parent_i)
                skip_paths.append((target_i, path_length, path))
    # SVG paints later paths on top: group by destination, drawing the longer
    # vertical routes first and short paths last so converging arrowheads remain visible.
    edge_bits.extend(path for _, _, path in sorted(skip_paths, key=lambda item: (item[0], -item[1])))
    skip_markers = "".join(
        f'<marker class="nn-arrowhead-{i}" id="{uid}-arrow-skip-{i}" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z"/></marker>'
        for i in sorted(skip_colors)
    )
    nodes: List[str] = []
    for i, step in enumerate(steps):
        top = y[i]
        source_names = [_source_name(p, names) for p in step.incoming if p in names or p == "input"]
        source_text = " + ".join(source_names) or "input"
        node_classes = ["nn-svg-node"]
        if step.merge:
            node_classes.append("nn-svg-merge")
        if step.children:
            node_classes.append("nn-svg-expandable")
        node_class = " ".join(node_classes)
        line = f'<text class="nn-svg-line" x="{_NODE_X + _NODE_W - 16}" y="{top + 23}" text-anchor="end">L{step.line}</text>' if step.line else ""
        link_open = (
            f'<a class="nn-svg-module-link" href="#{attr(_module_id(uid, step))}" '
            f'aria-label="Expand {attr(step.kind)} module internals">'
            if step.children else ""
        )
        link_close = "</a>" if step.children else ""
        expand_hint = (
            f'<circle class="nn-svg-expand-badge" cx="{_NODE_X + _NODE_W - 15}" cy="{top + 13}" r="8"/>'
            f'<text class="nn-svg-expand-mark" x="{_NODE_X + _NODE_W - 15}" y="{top + 17}" text-anchor="middle">+</text>'
            if step.children else ""
        )
        nodes.append(
            f'{link_open}<g class="{node_class}"><title>{esc(step.name)} · {esc(step.kind)} · {esc(format_shape(step.in_shape))} to {esc(format_shape(step.out_shape))}{" · click to expand" if step.children else ""}</title>'
            f'<rect x="{_NODE_X}" y="{top}" width="{_NODE_W}" height="{_NODE_H}" rx="12"/>'
            f'<text class="nn-svg-name" x="{_NODE_X + 18}" y="{top + 25}">{esc(_clip(step.name, 30))}</text>{line}'
            f'<text class="nn-svg-kind" x="{_NODE_X + 230}" y="{top + 25}">{esc(_clip(step.kind, 24))}</text>'
            f'<text class="nn-svg-params" x="{_NODE_X + 400}" y="{top + 25}">{esc(_clip(step.params or "", 36))}</text>'
            f'<text class="nn-svg-shape" x="{_NODE_X + 18}" y="{top + 48}">{esc(_clip(format_shape(step.in_shape), 38))}</text>'
            f'<text class="nn-svg-arrow" x="{_NODE_X + 370}" y="{top + 48}">→</text>'
            f'<text class="nn-svg-shape nn-svg-out" x="{_NODE_X + 410}" y="{top + 48}">{esc(_clip(format_shape(step.out_shape), 38))}</text>'
            f'<text class="nn-svg-from" x="{_NODE_X + _NODE_W - 16}" y="{top + 48}" text-anchor="end">from {esc(_clip(source_text, 25))}</text>'
            f'{expand_hint}</g>{link_close}'
        )
    output_label = "OUTPUT · " + "  |  ".join(_clip(format_shape(shape), 42) for shape in route.output_shapes)
    if not route.output_shapes:
        output_label = "OUTPUT · shape not statically determined"
    outs = f'<text class="nn-svg-output" x="{cx:g}" y="{output_y + 26}" text-anchor="middle">{esc(output_label)}</text>'
    view_width = max(_SVG_W, farthest_rail + 20)
    return (
        f'<div class="nn-svg-wrap"><svg class="nn-svg" viewBox="0 0 {view_width} {height}" role="group" '
        f'aria-label="{attr(f"{route.method_name} model path: {len(steps)} layers, input {format_shape(route.input_shape)}")}">'
        f'<title>{esc(route.method_name)} model path</title><desc>Layer graph. Each node shows input and output tensor dimensions; arrows show data flow.</desc>'
        f'<defs><marker id="{uid}-arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z"/></marker>{skip_markers}</defs>'
        f'<rect class="nn-svg-port" x="{cx - 260:g}" y="12" width="520" height="54" rx="13"/>'
        f'<text class="nn-svg-port-label" x="{cx:g}" y="33" text-anchor="middle">INPUT · {esc(route.input_name)}</text>'
        f'<text class="nn-svg-port-shape" x="{cx:g}" y="53" text-anchor="middle">{esc(format_shape(route.input_shape))}</text>'
        f'<g class="nn-svg-edges">{"".join(edge_bits)}</g>{"".join(nodes)}'
        f'<rect class="nn-svg-port nn-svg-outport" x="{cx - 260:g}" y="{output_y}" width="520" height="42" rx="13"/>{outs}'
        '</svg></div>'
    )


_D2_NODE_W = 205
_D2_NODE_H = 88
_D2_ROW_STEP = 112
_D2_BASE_GAP = 125
_D2_LEFT = 190


def _graph_data(steps: List[Step]) -> tuple[Dict[str, int], List[int], List[List[int]], List[List[int]]]:
    """Return node indexes, longest-path ranks, parent lists and barycentrically ordered columns."""
    index = {step.ident: i for i, step in enumerate(steps)}
    parents = [[index[parent] for parent in step.incoming if parent in index] for step in steps]
    ranks: List[int] = []
    for parent_nodes in parents:
        ranks.append(max((ranks[parent] + 1 for parent in parent_nodes), default=0))
    groups: List[List[int]] = [[] for _ in range(max(ranks, default=0) + 1)]
    for i, rank in enumerate(ranks):
        groups[rank].append(i)
    children: List[List[int]] = [[] for _ in steps]
    for target, parent_nodes in enumerate(parents):
        for parent in parent_nodes:
            children[parent].append(target)

    # Reorder siblings using repeated barycenter sweeps. This tends to line up
    # branch/merge endpoints and reduce crossings while preserving stable ties.
    for _ in range(5):
        positions = {
            node: order - (len(group) - 1) / 2
            for group in groups for order, node in enumerate(group)
        }
        for rank in range(1, len(groups)):
            groups[rank].sort(key=lambda node: (
                sum(positions[parent] for parent in parents[node]) / len(parents[node]) if parents[node] else positions[node],
                node,
            ))
        positions = {
            node: order - (len(group) - 1) / 2
            for group in groups for order, node in enumerate(group)
        }
        for rank in range(len(groups) - 2, -1, -1):
            groups[rank].sort(key=lambda node: (
                sum(positions[child] for child in children[node]) / len(children[node]) if children[node] else positions[node],
                node,
            ))
    return index, ranks, parents, groups


def _needs_2d(route: Route) -> bool:
    """Choose a layered view only when a linear path would be dominated by skip edges."""
    if len(route.steps) < 6:
        return False
    _, ranks, parents, _ = _graph_data(route.steps)
    edge_count = sum(len(items) for items in parents)
    skip_count = sum(
        1 for target, parent_nodes in enumerate(parents)
        for parent in parent_nodes if ranks[target] - ranks[parent] > 1
    )
    return skip_count >= 6 or (len(route.steps) >= 10 and skip_count >= 3) or (
        len(route.steps) >= 22 and edge_count >= len(route.steps) + 8
    )


def _svg_layered(route: Route, uid: str) -> str:
    """Left-to-right DAG layout for dense routes, with cross-reduced columns and skip rails."""
    steps = route.steps
    if not steps:
        return ""
    index, ranks, parents, groups = _graph_data(steps)
    max_rank = max(ranks, default=0)
    max_rows = max((len(group) for group in groups), default=1)
    content_height = max_rows * _D2_ROW_STEP

    # Give every source one stable y rail, and reserve a unique x track for both
    # exits and entries in each inter-column gap. All arrows from one source reuse
    # its exit track and rail; unrelated sources never share either coordinate.
    lane_by_source: Dict[int, int] = {}
    skip_lanes: Dict[tuple[int, int], int] = {}
    skip_edges = [
        (parent, target)
        for target, parent_nodes in enumerate(parents)
        for parent in parent_nodes
        if ranks[target] - ranks[parent] > 1
    ]
    outgoing = [set() for _ in steps]
    for target, parent_nodes in enumerate(parents):
        for parent in parent_nodes:
            outgoing[parent].add(target)
    sinks = [i for i in range(len(steps)) if not outgoing[i]] or [len(steps) - 1]

    source_tracks_by_gap: Dict[int, set[int]] = {}
    target_tracks_by_gap: Dict[int, List[tuple[int, int]]] = {}
    for parent, target in skip_edges:
        if parent not in lane_by_source:
            lane_by_source[parent] = len(lane_by_source)
        skip_lanes[(parent, target)] = lane_by_source[parent]
        source_tracks_by_gap.setdefault(ranks[parent], set()).add(parent)
        target_tracks_by_gap.setdefault(ranks[target] - 1, []).append((parent, target))
    if len(sinks) > 1:
        for sink in sinks:
            source_tracks_by_gap.setdefault(ranks[sink], set()).add(sink)

    column_x = [_D2_LEFT]
    for gap in range(max_rank):
        track_count = len(source_tracks_by_gap.get(gap, set())) + len(target_tracks_by_gap.get(gap, []))
        gap_width = max(_D2_BASE_GAP, 18 + 12 * (track_count + 1))
        column_x.append(column_x[-1] + _D2_NODE_W + gap_width)
    output_gap_tracks = len(source_tracks_by_gap.get(max_rank, set())) + (len(sinks) if len(sinks) > 1 else 0)
    output_gap_width = max(70, 18 + 12 * (output_gap_tracks + 1)) if output_gap_tracks else 70
    output_x = column_x[-1] + _D2_NODE_W + output_gap_width
    node_x = [column_x[rank] for rank in ranks]

    group_order = {node: order for group in groups for order, node in enumerate(group)}
    source_track_x: Dict[int, float] = {}
    target_track_x: Dict[tuple[int, int], float] = {}
    output_track_x: Dict[int, float] = {}
    for gap in range(max_rank + 1):
        sources = sorted(source_tracks_by_gap.get(gap, set()), key=lambda node: group_order[node])
        targets = sorted(
            target_tracks_by_gap.get(gap, []),
            key=lambda edge: (group_order[edge[1]], group_order[edge[0]]),
        )
        output_targets = sorted(sinks, key=lambda node: group_order[node]) if gap == max_rank and len(sinks) > 1 else []
        tracks = (
            [("source", node) for node in sources]
            + [("target", edge) for edge in targets]
            + [("output", node) for node in output_targets]
        )
        left = column_x[gap] + _D2_NODE_W + 8
        right = (column_x[gap + 1] - 8) if gap < max_rank else output_x - 8
        for order, (track_kind, key) in enumerate(tracks):
            track_x = left + (order + 1) * (right - left) / (len(tracks) + 1)
            if track_kind == "source":
                source_track_x[key] = track_x  # type: ignore[index]
            elif track_kind == "target":
                target_track_x[key] = track_x  # type: ignore[index]
            else:
                output_track_x[key] = track_x  # type: ignore[index]

    top_pad = 50 + 15 * len(lane_by_source)
    node_y = [0.0] * len(steps)
    for group in groups:
        offset = (max_rows - len(group)) * _D2_ROW_STEP / 2 + (_D2_ROW_STEP - _D2_NODE_H) / 2
        for order, node in enumerate(group):
            node_y[node] = top_pad + offset + order * _D2_ROW_STEP
    node_cy = [value + _D2_NODE_H / 2 for value in node_y]

    incoming_offsets: Dict[tuple[int, int], float] = {}
    for target, parent_nodes in enumerate(parents):
        ordered = sorted(parent_nodes, key=lambda parent: (node_cy[parent], parent))
        spacing = min(8.0, 24.0 / max(1, len(ordered) - 1))
        for order, parent in enumerate(ordered):
            incoming_offsets[(parent, target)] = (order - (len(ordered) - 1) / 2) * spacing

    color_by_source: Dict[int, int] = {}
    colors: set[int] = set()
    for parent, _ in skip_edges:
        if parent not in color_by_source:
            color_by_source[parent] = len(color_by_source) % 8
        colors.add(color_by_source[parent])

    output_shapes = route.output_shapes
    output_text = "OUTPUT"
    if output_shapes:
        output_text += " · " + " | ".join(_clip(format_shape(shape), 22) for shape in output_shapes)
    output_h = max(58, 18 + 16 * len(sinks))
    graph_center_y = node_cy[sinks[0]] if len(sinks) == 1 else top_pad + content_height / 2
    output_y = graph_center_y - output_h / 2
    bottom_rail_base = top_pad + content_height + 26
    height = (
        max(top_pad + content_height + 36, output_y + output_h + 20)
        if len(sinks) == 1 else bottom_rail_base + len(sinks) * 14 + 36
    )
    input_x, input_w, port_w = 20, 120, 154
    input_cy = graph_center_y
    view_width = output_x + port_w + 24

    edge_bits: List[str] = []
    # External tensor input fans out to graph roots.
    for target, step in enumerate(steps):
        if not any(parent == "input" or parent not in index for parent in step.incoming) and parents[target]:
            continue
        tx = node_x[target]
        ty = node_cy[target]
        input_bus_x = input_x + input_w + 20
        d = _rounded_path([
            (input_x + input_w, input_cy),
            (input_bus_x, input_cy),
            (input_bus_x, ty),
            (tx, ty),
        ], radius=6)
        edge_bits.append(
            f'<path class="nn-2d-edge nn-2d-input" d="{d}" marker-end="url(#{uid}-2d-arrow)"/>'
        )

    for target, parent_nodes in enumerate(parents):
        for parent in parent_nodes:
            sx = node_x[parent] + _D2_NODE_W
            tx = node_x[target]
            sy = node_cy[parent]
            ty = node_cy[target] + incoming_offsets[(parent, target)]
            if ranks[target] - ranks[parent] == 1:
                dx = tx - sx
                c1, c2 = sx + dx * 0.42, tx - dx * 0.42
                edge_bits.append(
                    f'<path class="nn-2d-edge nn-2d-normal" d="M {sx} {sy:g} C {c1:g} {sy:g} {c2:g} {ty:g} {tx} {ty:g}" marker-end="url(#{uid}-2d-arrow)"/>'
                )
            else:
                lane = skip_lanes[(parent, target)]
                rail_y = 28 + lane * 14
                source_track = source_track_x[parent]
                target_track = target_track_x[(parent, target)]
                d = _rounded_path([
                    (sx, sy),
                    (source_track, sy),
                    (source_track, rail_y),
                    (target_track, rail_y),
                    (target_track, ty),
                    (tx, ty),
                ], radius=6)
                color = color_by_source[parent]
                edge_bits.append(
                    f'<path class="nn-2d-edge nn-2d-skip nn-flow-{color}" d="{d}" data-source="{attr(steps[parent].ident)}" data-target="{attr(steps[target].ident)}" data-lane="{lane}" data-source-x="{source_track:g}" data-target-x="{target_track:g}" marker-end="url(#{uid}-2d-arrow-skip-{color})"/>'
                )

    if len(sinks) == 1:
        sink = sinks[0]
        source_x = node_x[sink] + _D2_NODE_W
        source_y = node_cy[sink]
        edge_bits.append(
            f'<path class="nn-2d-edge nn-2d-output-edge" d="M {source_x} {source_y:g} H {output_x}" marker-end="url(#{uid}-2d-arrow)"/>'
        )
    else:
        # Multiple terminal branches use bottom rails to stay clear of intermediate nodes.
        ordered_sinks = sorted(sinks, key=lambda sink: (node_cy[sink], sink))
        for order, sink in enumerate(ordered_sinks):
            source_x = node_x[sink] + _D2_NODE_W
            source_y = node_cy[sink]
            lane_y = bottom_rail_base + order * 14
            output_target_y = graph_center_y + (order - (len(ordered_sinks) - 1) / 2) * min(14, output_h / len(ordered_sinks))
            source_track = source_track_x[sink]
            output_track = output_track_x[sink]
            d = _rounded_path([
                (source_x, source_y), (source_track, source_y), (source_track, lane_y),
                (output_track, lane_y), (output_track, output_target_y), (output_x, output_target_y),
            ], radius=6)
            edge_bits.append(
                f'<path class="nn-2d-edge nn-2d-output-edge" d="{d}" data-source="{attr(steps[sink].ident)}" data-source-x="{source_track:g}" data-output-x="{output_track:g}" marker-end="url(#{uid}-2d-arrow)"/>'
            )

    nodes: List[str] = []
    for i, step in enumerate(steps):
        x, y = node_x[i], node_y[i]
        kind = step.kind
        if step.merge:
            node_type = "merge"
        elif step.children:
            node_type = "custom"
        elif kind.startswith("Conv"):
            node_type = "conv"
        elif kind in ("Linear", "LazyLinear", "Bilinear"):
            node_type = "linear"
        elif "Norm" in kind or kind in ("ReLU", "GELU", "SiLU", "Softmax", "Sigmoid", "Tanh"):
            node_type = "transform"
        else:
            node_type = "other"
        title = f"{step.name} · {step.kind} · {format_shape(step.in_shape)} to {format_shape(step.out_shape)}"
        link_open = (
            f'<a class="nn-svg-module-link" href="#{attr(_module_id(uid, step))}" aria-label="Expand {attr(step.kind)} module internals">'
            if step.children else ""
        )
        link_close = "</a>" if step.children else ""
        icon_markup = ""
        text_x = x + 14
        if kind == "Conv2d" and re.search(r"kernel_size\s*=\s*(?:3(?:\b|$)|\(\s*3\s*,\s*3\s*\))", step.params):
            cells = "".join(
                f'<rect x="{x + 12 + col * 7}" y="{y + 14 + row * 7}" width="5" height="5" rx="1"/>'
                for row in range(3) for col in range(3)
            )
            icon_markup = f'<g class="nn-2d-kernel">{cells}<text x="{x + 11}" y="{y + 51}">3×3</text></g>'
            text_x = x + 48
        elif node_type == "merge":
            icon_markup = f'<circle class="nn-2d-type-icon" cx="{x + 25}" cy="{y + 30}" r="13"/><text class="nn-2d-icon-label" x="{x + 25}" y="{y + 34}" text-anchor="middle">+</text>'
            text_x = x + 48
        elif node_type == "linear":
            icon_markup = f'<circle class="nn-2d-type-icon" cx="{x + 25}" cy="{y + 30}" r="13"/><text class="nn-2d-icon-label" x="{x + 25}" y="{y + 34}" text-anchor="middle">FC</text>'
            text_x = x + 48
        elif node_type == "custom":
            icon_markup = f'<rect class="nn-2d-type-icon" x="{x + 13}" y="{y + 18}" width="24" height="24" rx="5"/><text class="nn-2d-icon-label" x="{x + 25}" y="{y + 34}" text-anchor="middle">M</text>'
            text_x = x + 48
        name_y = y + 24
        kind_y = y + 43
        params_y = y + 61
        shapes_y = y + 80
        expand = (
            f'<circle class="nn-svg-expand-badge" cx="{x + _D2_NODE_W - 14}" cy="{y + 14}" r="8"/>'
            f'<text class="nn-svg-expand-mark" x="{x + _D2_NODE_W - 14}" y="{y + 18}" text-anchor="middle">+</text>'
            if step.children else ""
        )
        node_class = f"nn-2d-node nn-2d-{node_type}"
        node_text = (
            f'<text class="nn-2d-name" x="{text_x}" y="{name_y}">{esc(_clip(step.name, 22))}</text>'
            f'<text class="nn-2d-kind" x="{text_x}" y="{kind_y}">{esc(_clip(kind, 20))}</text>'
            f'<text class="nn-2d-params" x="{text_x}" y="{params_y}">{esc(_clip(step.params or "", 25))}</text>'
            f'<text class="nn-2d-shape" x="{x + 11}" y="{shapes_y}">{esc(_clip(format_shape(step.in_shape) + " → " + format_shape(step.out_shape), 30))}</text>'
        )
        nodes.append(
            f'{link_open}<g class="{node_class}"><title>{esc(title)}{" · click to expand" if step.children else ""}</title>'
            f'<rect class="nn-2d-card" x="{x}" y="{y:g}" width="{_D2_NODE_W}" height="{_D2_NODE_H}" rx="12"/>'
            f'{icon_markup}{node_text}{expand}</g>{link_close}'
        )

    output_port = f'<rect class="nn-svg-port nn-2d-port" x="{output_x}" y="{output_y:g}" width="{port_w}" height="{output_h}" rx="13"/>'
    output_label = f'<text class="nn-svg-port-label" x="{output_x + port_w / 2}" y="{graph_center_y - 2:g}" text-anchor="middle">{esc(_clip(output_text, 20))}</text>'
    input_port = (
        f'<rect class="nn-svg-port nn-2d-port" x="{input_x}" y="{input_cy - 28:g}" width="{input_w}" height="56" rx="13"/>'
        f'<text class="nn-svg-port-label" x="{input_x + input_w / 2}" y="{input_cy - 5:g}" text-anchor="middle">INPUT</text>'
        f'<text class="nn-svg-port-shape" x="{input_x + input_w / 2}" y="{input_cy + 15:g}" text-anchor="middle">{esc(_clip(format_shape(route.input_shape), 18))}</text>'
    )
    skip_markers = "".join(
        f'<marker class="nn-arrowhead-{color}" id="{uid}-2d-arrow-skip-{color}" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z"/></marker>'
        for color in sorted(colors)
    )
    return (
        f'<div class="nn-svg-wrap nn-svg-wrap-2d"><svg class="nn-svg nn-svg-2d" style="width:{view_width}px;max-width:none" viewBox="0 0 {view_width} {height:g}" role="group" '
        f'aria-label="{attr(f"{route.method_name} layered 2D model graph with {len(steps)} nodes")}">'
        f'<title>{esc(route.method_name)} · layered 2D model graph</title><desc>Left-to-right dependency graph. Each block shows its layer type and input/output dimensions; curved links distinguish sequential edges from long skip routes.</desc>'
        f'<defs><marker id="{uid}-2d-arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z"/></marker>{skip_markers}</defs>'
        f'{input_port}{output_port}{"".join(nodes)}<g class="nn-svg-2d-edges">{"".join(edge_bits)}</g>{output_label}'
        '</svg></div>'
    )


def _svg(route: Route, uid: str) -> str:
    return _svg_layered(route, uid) if _needs_2d(route) else _svg_linear(route, uid)


def _route(route: Route, uid: str) -> str:
    names = {s.ident: s.name for s in route.steps}
    count = f'{len(route.steps)} layer step' if len(route.steps) == 1 else f'{len(route.steps)} layer steps'
    suffix = "()" if route.kind == "method" else ""
    mode_badge = '<em class="nn-layout-mode" title="Drag the graph to pan">layered 2D · drag to pan</em>' if _needs_2d(route) else ""
    heading = f'<h5 class="nn-route-title"><code>{esc(route.method_name)}{suffix}</code><span>{count}</span>{mode_badge}</h5>'
    svg = _svg(route, uid)
    detailed = "".join(_step(step, names, uid) for step in route.steps)
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
    return f'<div class="nn-route" id="{attr(uid)}">{heading}{svg}{detail}{note_html}</div>'


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
