"""Infographic widgets: gauges, fingerprints, maps and charts (inline SVG/HTML)."""
from __future__ import annotations

import ast
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from .. import analysis
from ..analysis import grade as grade_of
from ..model import DefInfo, ModuleInfo
from .icons import icon
from .util import attr, clamp, esc

GRADES = "ABCDEF"
GRADE_WORDS = {
    "A": "simple",
    "B": "moderate",
    "C": "complex",
    "D": "very complex",
    "E": "alarming",
    "F": "unmaintainable",
}


# --------------------------------------------------------------------------- #
#  small pieces
# --------------------------------------------------------------------------- #


def gauge(cc: int, size: str = "") -> str:
    """Donut gauge showing McCabe complexity and its letter grade."""
    g = grade_of(cc)
    frac = clamp(cc / 25.0, 0.1, 1.0)
    return (
        f'<span class="gauge g-{g} {size}" title="Cyclomatic complexity {cc} &middot; grade {g} ({GRADE_WORDS[g]})">'
        f'<svg viewBox="0 0 36 36" aria-hidden="true"><circle class="trk" cx="18" cy="18" r="15" pathLength="100"/>'
        f'<circle class="val" cx="18" cy="18" r="15" pathLength="100" stroke-dasharray="{frac * 100:.1f} 100" '
        f'transform="rotate(-90 18 18)"/></svg><b>{cc}</b><small>{g}</small></span>'
    )


def ring(pct: float, label: str, tone: str = "") -> str:
    """Small percentage ring with a caption."""
    pct = clamp(pct, 0, 100)
    if not tone:
        tone = "good" if pct >= 80 else "mid" if pct >= 50 else "low"
    return (
        f'<span class="ring r-{tone}" title="{esc(label)}: {pct:.0f}%">'
        f'<svg viewBox="0 0 36 36" aria-hidden="true"><circle class="trk" cx="18" cy="18" r="15" pathLength="100"/>'
        f'<circle class="val" cx="18" cy="18" r="15" pathLength="100" stroke-dasharray="{pct:.1f} 100" '
        f'transform="rotate(-90 18 18)"/></svg><b>{pct:.0f}<i>%</i></b></span>'
    )


def badge(text: str, kind: str = "", ic: str = "", title: str = "") -> str:
    k = f" bd-{kind}" if kind else ""
    i = icon(ic) if ic else ""
    t = f' title="{attr(title)}"' if title else ""
    return f'<span class="bd{k}"{t}>{i}{esc(text)}</span>'


# --------------------------------------------------------------------------- #
#  function fingerprint
# --------------------------------------------------------------------------- #


def _category(s: ast.stmt) -> str:
    if isinstance(s, ast.If):
        return "if"
    if isinstance(s, (ast.For, ast.AsyncFor, ast.While)):
        return "loop"
    if isinstance(s, (ast.With, ast.AsyncWith)):
        return "with"
    if isinstance(s, analysis._TRY_TYPES):
        return "try"
    if getattr(ast, "Match", None) is not None and isinstance(s, ast.Match):  # type: ignore[attr-defined]
        return "match"
    if isinstance(s, ast.Return):
        return "ret"
    if isinstance(s, ast.Raise):
        return "raise"
    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "def"
    if isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return "assign"
    if isinstance(s, ast.Expr):
        return "call"
    return "other"


_CAT_LABEL = {
    "if": "decision",
    "loop": "loop",
    "with": "context",
    "try": "error handling",
    "match": "match",
    "ret": "return",
    "raise": "raise",
    "def": "nested definition",
    "assign": "assignments",
    "call": "calls",
    "other": "other",
}


def fingerprint(d: DefInfo, body: Sequence[ast.stmt], mini: bool = False) -> str:
    """A barcode of the function's top-level structure, one colour per kind."""
    segs: List[Tuple[str, int, int, int]] = []  # (category, weight, first row, last row)
    for s in body:
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
            continue  # docstring / stray string
        cat = _category(s)
        lines = max(1, (s.end_lineno or s.lineno) - s.lineno + 1)
        if segs and segs[-1][0] == cat and cat in ("assign", "call", "other"):
            c, w, a, _ = segs[-1]
            segs[-1] = (c, w + lines, a, s.end_lineno or s.lineno)
        else:
            segs.append((cat, lines, s.lineno, s.end_lineno or s.lineno))
    if len(segs) < 2:
        return ""
    if mini:
        bits = "".join(f'<i class="fp-s fp-{cat}" style="flex:{max(1.0, math.sqrt(w)):.2f}"></i>' for cat, w, _a, _b in segs)
        return f'<span class="fp-mini" aria-hidden="true">{bits}</span>'
    parts = []
    for cat, w, a, b in segs:
        flex = max(1.0, math.sqrt(w) * 2.2)
        rows = f"line {a}" if a == b else f"lines {a}\u2013{b}"
        parts.append(
            f'<a class="fp-s fp-{cat}" href="#L{a}" style="flex:{flex:.2f}" title="{_CAT_LABEL[cat]} &middot; {rows}"></a>'
        )
    used = []
    for cat, *_ in segs:
        if cat not in used:
            used.append(cat)
    legend = "".join(f'<span class="fp-k"><i class="fp-{c}"></i>{_CAT_LABEL[c]}</span>' for c in used)
    return f'<div class="fp"><div class="fp-bar">{"".join(parts)}</div><div class="fp-legend">{legend}</div></div>'


# --------------------------------------------------------------------------- #
#  class map
# --------------------------------------------------------------------------- #


def class_map(cls: DefInfo) -> str:
    """Every method as a block: width ~ lines of code, colour ~ complexity."""
    methods = [m for m in cls.children if not m.is_class]
    if len(methods) < 3:
        return ""
    blocks = []
    for m in methods:
        flex = max(3, m.loc)
        tags = []
        if "private" in m.badges:
            tags.append("private")
        if m.role not in ("method", "function"):
            tags.append(m.role)
        extra = f" &middot; {', '.join(tags)}" if tags else ""
        label = esc(m.name)
        blocks.append(
            f'<a class="cm-b g-{m.grade}" href="#{m.anchor}" style="flex:{flex}" '
            f'title="{esc(m.name)} &middot; {m.loc} lines &middot; complexity {m.cc} ({m.grade}){extra}">'
            f'<span>{label}</span></a>'
        )
    legend = "".join(f'<span class="cm-k g-{g}"><i></i>{g}</span>' for g in "ABCDEF" if any(m.grade == g for m in methods))
    return (
        f'<div class="cmap"><div class="cm-bar">{"".join(blocks)}</div>'
        f'<div class="cm-legend"><span>width = lines of code</span><span class="sp"></span>complexity {legend}</div></div>'
    )


# --------------------------------------------------------------------------- #
#  connections mini-map
# --------------------------------------------------------------------------- #

_NODE_H = 26
_NODE_GAP = 8


def _short(name: str, limit: int = 22) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "\u2026"


def connections(d: DefInfo, max_nodes: int = 7) -> str:
    """Callers -> this function -> callees, drawn as a tiny network diagram."""
    callers = [c for c in d.callers if c is not d]
    callees = d.callees
    if not callers and not callees:
        return ""

    def trim(nodes: List[DefInfo]) -> Tuple[List[DefInfo], int]:
        return nodes[:max_nodes], max(0, len(nodes) - max_nodes)

    left, left_more = trim(callers)
    right, right_more = trim(callees)
    rows = max(len(left) + (1 if left_more else 0), len(right) + (1 if right_more else 0), 1)
    height = rows * _NODE_H + (rows - 1) * _NODE_GAP + 16
    char_w = 6.6

    def width_of(nodes: Sequence[DefInfo], fallback: int = 128) -> int:
        if not nodes:
            return fallback
        return int(clamp(max(len(_short(n.qualname if not n.is_method else f"{n.parent.name}.{n.name}")) for n in nodes) * char_w + 28, 88, 190))

    center_label = _short(d.name, 20)
    wl, wr = width_of(left), width_of(right)
    wc = int(clamp(len(center_label) * char_w + 34, 100, 170))
    gap = 54
    total_w = wl + gap + wc + gap + wr
    cy = height / 2
    out: List[str] = []

    def y_of(i: int, n: int) -> float:
        block = n * _NODE_H + (n - 1) * _NODE_GAP
        return (height - block) / 2 + i * (_NODE_H + _NODE_GAP)

    def label(n: DefInfo) -> str:
        return n.qualname if not n.is_method else f"{n.parent.name}.{n.name}"

    # edges first so nodes sit on top
    xl_end, xc_start, xc_end, xr_start = wl, wl + gap, wl + gap + wc, wl + gap + wc + gap
    nl = len(left) + (1 if left_more else 0)
    nr = len(right) + (1 if right_more else 0)
    for i, n in enumerate(left):
        y = y_of(i, nl) + _NODE_H / 2
        out.append(
            f'<path class="edge e-in" d="M{xl_end} {y:.1f} C{xl_end + gap * .55:.1f} {y:.1f} {xc_start - gap * .55:.1f} {cy:.1f} {xc_start} {cy:.1f}" marker-end="url(#ar)"/>'
        )
    for i, n in enumerate(right):
        y = y_of(i, nr) + _NODE_H / 2
        out.append(
            f'<path class="edge e-out" d="M{xc_end} {cy:.1f} C{xc_end + gap * .55:.1f} {cy:.1f} {xr_start - gap * .55:.1f} {y:.1f} {xr_start} {y:.1f}" marker-end="url(#ar)"/>'
        )
    # nodes
    def node(x: float, y: float, w: float, n: DefInfo, cls: str) -> str:
        text = esc(_short(label(n)))
        return (
            f'<a href="#{n.anchor}" class="node {cls} g-{n.grade}"><title>{esc(label(n))} &middot; complexity {n.cc}</title>'
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{_NODE_H}" rx="8"/>'
            f'<text x="{x + w / 2:.1f}" y="{y + _NODE_H / 2 + 4:.1f}" text-anchor="middle">{text}</text></a>'
        )

    for i, n in enumerate(left):
        out.append(node(0, y_of(i, nl), wl, n, "n-in"))
    if left_more:
        y = y_of(len(left), nl)
        out.append(f'<g class="node n-more"><text x="{wl / 2}" y="{y + _NODE_H / 2 + 4:.1f}" text-anchor="middle">+{left_more} more</text></g>')
    if not left:
        y = y_of(0, nl)
        out.append(
            f'<g class="node n-none"><rect x="0" y="{y:.1f}" width="{wl}" height="{_NODE_H}" rx="8"/>'
            f'<text x="{wl / 2}" y="{y + _NODE_H / 2 + 4:.1f}" text-anchor="middle">no callers here</text></g>'
        )
    cyy = cy - _NODE_H / 2 - 3
    out.append(
        f'<g class="node n-this g-{d.grade}"><rect x="{xc_start}" y="{cyy:.1f}" width="{wc}" height="{_NODE_H + 6}" rx="10"/>'
        f'<text x="{xc_start + wc / 2:.1f}" y="{cy + 4.5:.1f}" text-anchor="middle">{esc(center_label)}</text></g>'
    )
    for i, n in enumerate(right):
        out.append(node(xr_start, y_of(i, nr), wr, n, "n-out"))
    if right_more:
        y = y_of(len(right), nr)
        out.append(f'<g class="node n-more"><text x="{xr_start + wr / 2}" y="{y + _NODE_H / 2 + 4:.1f}" text-anchor="middle">+{right_more} more</text></g>')
    if not right:
        y = y_of(0, nr)
        out.append(
            f'<g class="node n-none"><rect x="{xr_start}" y="{y:.1f}" width="{wr}" height="{_NODE_H}" rx="8"/>'
            f'<text x="{xr_start + wr / 2}" y="{y + _NODE_H / 2 + 4:.1f}" text-anchor="middle">calls nothing local</text></g>'
        )
    defs = (
        '<defs><marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M1 1 9 5 1 9z" class="arrow"/></marker></defs>'
    )
    return (
        f'<svg class="conn" viewBox="0 0 {total_w} {height}" style="max-width:{total_w}px" role="img" '
        f'aria-label="Calls into and out of {esc(d.name)}">{defs}{"".join(out)}</svg>'
    )


# --------------------------------------------------------------------------- #
#  treemap
# --------------------------------------------------------------------------- #

Rect = Tuple[float, float, float, float]


def squarify(items: Sequence[Tuple[float, object]], x: float, y: float, w: float, h: float) -> List[Tuple[object, float, float, float, float]]:
    """Squarified treemap layout (Bruls, Huizing, van Wijk)."""
    items = [(v, o) for v, o in items if v > 0]
    if not items or w <= 0 or h <= 0:
        return []
    items.sort(key=lambda t: -t[0])
    total = sum(v for v, _ in items)
    scale = w * h / total
    areas = [(v * scale, o) for v, o in items]
    result: List[Tuple[object, float, float, float, float]] = []
    rx, ry, rw, rh = x, y, w, h

    def worst(row: List[Tuple[float, object]], side: float) -> float:
        s = sum(a for a, _ in row)
        mx = max(a for a, _ in row)
        mn = min(a for a, _ in row)
        return max(side * side * mx / (s * s), (s * s) / (side * side * mn))

    while areas:
        side = min(rw, rh)
        row = [areas[0]]
        i = 1
        while i < len(areas) and worst(row + [areas[i]], side) <= worst(row, side):
            row.append(areas[i])
            i += 1
        s = sum(a for a, _ in row)
        if rw >= rh:
            col_w = s / rh
            yy = ry
            for a, o in row:
                hh = a / col_w
                result.append((o, rx, yy, col_w, hh))
                yy += hh
            rx += col_w
            rw -= col_w
        else:
            row_h = s / rw
            xx = rx
            for a, o in row:
                ww = a / row_h
                result.append((o, xx, ry, ww, row_h))
                xx += ww
            ry += row_h
            rh -= row_h
        areas = areas[i:]
    return result


def _fit(text: str, width: float, font: float = 11.0) -> str:
    n = int(width / (font * 0.6))
    if n < 3:
        return ""
    return text if len(text) <= n else text[: max(1, n - 1)] + "\u2026"


def treemap(mod: ModuleInfo) -> str:
    """Code map: area = lines of code, colour = complexity grade."""
    W, H = 1000.0, 360.0
    entries: List[Tuple[float, DefInfo]] = []
    for d in mod.top:
        if d.is_class:
            inner = sum(max(c.loc, 2) for c in d.children if not c.is_class) or d.loc
            entries.append((max(float(d.loc), float(inner)), d))
        else:
            entries.append((float(max(d.loc, 2)), d))
    if not entries:
        return ""
    rects = squarify(entries, 0, 0, W, H)
    out: List[str] = []
    pad = 3.0
    for o, x, y, w, h in rects:
        d: DefInfo = o  # type: ignore[assignment]
        gx, gy, gw, gh = x + pad / 2, y + pad / 2, w - pad, h - pad
        if gw <= 0 or gh <= 0:
            continue
        if d.is_class and any(not c.is_class for c in d.children):
            head = 19.0 if gh > 50 and gw > 70 else 0.0
            out.append(
                f'<g class="tm-class"><a href="#{d.anchor}"><title>{esc(d.qualname)} &middot; class &middot; {d.loc} lines</title>'
                f'<rect class="tm-cbg" x="{gx:.1f}" y="{gy:.1f}" width="{gw:.1f}" height="{gh:.1f}" rx="9"/>'
                + (f'<text class="tm-ct" x="{gx + 9:.1f}" y="{gy + 13.5:.1f}">{esc(_fit(d.name, gw - 16, 12))}</text>' if head else "")
                + "</a>"
            )
            ms = [(float(max(c.loc, 2)), c) for c in d.children if not c.is_class]
            inner = squarify(ms, gx + 4, gy + head + (2 if head else 4), gw - 8, gh - head - (6 if head else 8))
            for mo, mx, my, mw, mh in inner:
                m: DefInfo = mo  # type: ignore[assignment]
                out.append(_tm_leaf(m, mx, my, mw, mh, 2.0, label=m.name))
            out.append("</g>")
        else:
            out.append(_tm_leaf(d, gx, gy, gw, gh, 0.0, label=d.name, cls="tm-top"))
    legend = "".join(f'<span class="cm-k g-{g}"><i></i>{g}</span>' for g in "ABCDEF")
    return (
        f'<div class="treemap"><svg viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="Code map">{"".join(out)}</svg>'
        f'<div class="cm-legend"><span>area = lines of code</span><span class="sp"></span>complexity {legend}</div></div>'
    )


def _tm_leaf(d: DefInfo, x: float, y: float, w: float, h: float, pad: float, label: str, cls: str = "") -> str:
    gx, gy, gw, gh = x + pad / 2, y + pad / 2, w - pad, h - pad
    if gw <= 1 or gh <= 1:
        return ""
    kind = "class" if d.is_class else d.role
    text = _fit(label, gw - 10) if gh >= 18 else ""
    sub = ""
    if text and gh >= 34 and gw >= 70:
        sub = f'<text class="tm-sub" x="{gx + 6:.1f}" y="{gy + 28:.1f}">{d.loc} lines &middot; {d.grade}{d.cc}</text>'
    return (
        f'<a href="#{d.anchor}" class="tm-leaf {cls} g-{d.grade}"><title>{esc(d.qualname)} &middot; {kind} &middot; {d.loc} lines &middot; complexity {d.cc} ({d.grade})</title>'
        f'<rect x="{gx:.1f}" y="{gy:.1f}" width="{gw:.1f}" height="{gh:.1f}" rx="6"/>'
        + (f'<text x="{gx + 6:.1f}" y="{gy + 14:.1f}">{esc(text)}</text>' if text else "")
        + sub
        + "</a>"
    )


# --------------------------------------------------------------------------- #
#  grade distribution
# --------------------------------------------------------------------------- #


def grade_bar(defs: Iterable[DefInfo]) -> str:
    """Stacked bar: how many functions fall in each complexity grade."""
    fns = [d for d in defs if not d.is_class]
    if not fns:
        return ""
    counts = {g: 0 for g in GRADES}
    for d in fns:
        counts[d.grade] += 1
    segs = "".join(
        f'<span class="gb-s g-{g}" style="flex:{c}" title="{c} function{"s" if c != 1 else ""} with grade {g} ({GRADE_WORDS[g]})"><b>{g}</b><i>{c}</i></span>'
        for g, c in counts.items()
        if c
    )
    return f'<div class="gbar" role="img" aria-label="complexity grades">{segs}</div>'


# --------------------------------------------------------------------------- #
#  chord diagram of the call graph
# --------------------------------------------------------------------------- #


def call_chords(mod: ModuleInfo, max_nodes: int = 44) -> str:
    """Circular diagram: nodes on a ring, one chord per local call."""
    edges: List[Tuple[DefInfo, DefInfo]] = []
    for d in mod.defs:
        if d.is_class:
            continue
        for t in d.callees:
            edges.append((d, t))
    if len(edges) < 2:
        return ""
    nodes: List[DefInfo] = []
    seen = set()
    for a, b in edges:
        for n in (a, b):
            if id(n) not in seen:
                seen.add(id(n))
                nodes.append(n)
    if len(nodes) > max_nodes:
        return ""

    def group(n: DefInfo) -> str:
        p = n
        while p.parent is not None:
            p = p.parent
        return p.name if p.is_class else "\u0000module"

    order: Dict[str, List[DefInfo]] = {}
    for n in sorted(nodes, key=lambda n: n.first_line):
        order.setdefault(group(n), []).append(n)
    groups = list(order.items())
    name_count: Dict[str, int] = {}
    for n in nodes:
        name_count[n.name] = name_count.get(n.name, 0) + 1
    size = 560.0
    cx = cy = size / 2
    radius = size / 2 - 120
    n_total = len(nodes)
    gap_angle = 0.08
    usable = 2 * math.pi - gap_angle * len(groups)
    angle = -math.pi / 2
    pos: Dict[int, Tuple[float, float, float]] = {}
    arcs: List[str] = []
    dots: List[str] = []
    for gi, (gname, members) in enumerate(groups):
        span = usable * len(members) / n_total
        step = span / len(members)
        start_a = angle
        for i, m in enumerate(members):
            a = angle + step * (i + 0.5)
            pos[id(m)] = (cx + radius * math.cos(a), cy + radius * math.sin(a), a)
        end_a = angle + span
        # group arc
        r_arc = radius + 12
        x1, y1 = cx + r_arc * math.cos(start_a + 0.01), cy + r_arc * math.sin(start_a + 0.01)
        x2, y2 = cx + r_arc * math.cos(end_a - 0.01), cy + r_arc * math.sin(end_a - 0.01)
        large = 1 if span > math.pi else 0
        arcs.append(f'<path class="cd-arc cd-{gi % 8}" d="M{x1:.1f} {y1:.1f} A{r_arc} {r_arc} 0 {large} 1 {x2:.1f} {y2:.1f}"/>')
        angle = end_a + gap_angle
    # chords
    chords: List[str] = []
    for a, b in edges:
        ax, ay, _ = pos[id(a)]
        bx, by, _ = pos[id(b)]
        gi = [g for g, _ in groups].index(group(a))
        chords.append(
            f'<path class="cd-chord cd-{gi % 8}" d="M{ax:.1f} {ay:.1f} Q{cx:.1f} {cy:.1f} {bx:.1f} {by:.1f}" data-a="{a.anchor}" data-b="{b.anchor}">'
            f"<title>{esc(a.qualname)} \u2192 {esc(b.qualname)}</title></path>"
        )
    for gi, (gname, members) in enumerate(groups):
        for m in members:
            x, y, a = pos[id(m)]
            deg = math.degrees(a)
            flip = math.cos(a) < 0
            lx, ly = cx + (radius + 22) * math.cos(a), cy + (radius + 22) * math.sin(a)
            rot = deg + 180 if flip else deg
            anchor = "end" if flip else "start"
            if name_count[m.name] > 1 and m.parent is not None:  # several ``__init__`` - say whose
                label = f"{_short(m.parent.name, 9)}.{_short(m.name, 9)}"
            else:
                label = _short(m.name, 18)
            dots.append(
                f'<g class="cd-node"><a href="#{m.anchor}"><title>{esc(m.qualname)}</title>'
                f'<circle class="cd-dot g-{m.grade}" cx="{x:.1f}" cy="{y:.1f}" r="5.2"/>'
                f'<text transform="translate({lx:.1f} {ly:.1f}) rotate({rot:.1f})" text-anchor="{anchor}" dy="0.32em">{esc(label)}</text></a></g>'
            )
    leg = []
    for gi, (gname, members) in enumerate(groups):
        name = "module functions" if gname == "\u0000module" else gname
        leg.append(f'<span class="cd-k"><i class="cd-{gi % 8}"></i>{esc(name)}</span>')
    return (
        f'<div class="chords"><svg viewBox="0 0 {size:.0f} {size:.0f}" role="img" aria-label="Call graph">'
        f'{"".join(arcs)}{"".join(chords)}{"".join(dots)}</svg><div class="cd-legend">{"".join(leg)}</div></div>'
    )
