"""
hipgraph.plot — matplotlib rendering, for environments without internet.

`draw_mermaid_png()` posts the diagram to mermaid.ink, so it needs network
access. A Kaggle notebook usually has none, and a fault tree that cannot be
drawn where the analysis runs is not much use. Everything here is matplotlib
only.

Fault trees are drawn with the conventional symbols: a rectangle for an event,
the **AND** gate (flat base, domed top) and **OR** gate (curved sides rising to
a point, concave base) beneath the event they resolve, a circle for a basic
event and a diamond for an undeveloped one. Gate symbols also carry their name,
so the distinction never rests on shape alone.

Colours are the validated design-system tokens rather than the source notebook's
palette, whose teal/apricot pair fails colour-vision separation (ΔE 5.4 protan,
against a floor of 8).
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Tuple

from ..architecture.acyclic import find_back_edges
from ..faulttree.analysis import TreeAnalysis
from ..architecture.model import Role, SystemModel
from ..faulttree.synthesis import FaultTree, expand_to_tree

__all__ = [
    "TOKENS",
    "plot_fault_tree",
    "plot_architecture",
    "plot_importance",
    "plot_cutset_orders",
]

#: Design tokens (light surface). Status colours are reserved and always ship
#: with a label, never as the only signal.
TOKENS = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "grid": "#e1e0d9",
    "series": "#2a78d6",        # sequential hue, step 450
    "series_light": "#cde2fb",  # step 100
    "critical": "#d03b3b",
    "warning": "#fab219",
    "serious": "#ec835a",
    "good": "#0ca30c",
}

_NODE_STYLE = {
    "top": dict(face=TOKENS["critical"], edge="#8f2020", text="#ffffff", shape="rect", lw=2.0),
    "intermediate": dict(face=TOKENS["series_light"], edge=TOKENS["series"], text=TOKENS["text"],
                         shape="rect", lw=1.2),
    "basic": dict(face=TOKENS["surface"], edge=TOKENS["text_secondary"], text=TOKENS["text"],
                  shape="ellipse", lw=1.2),
    "undeveloped": dict(face=TOKENS["warning"], edge="#a8730d", text=TOKENS["text"],
                        shape="diamond", lw=1.2),
    "house": dict(face=TOKENS["good"], edge="#076f07", text="#ffffff", shape="rect", lw=1.2),
    "transfer": dict(face="#e6e5e1", edge=TOKENS["text_secondary"], text=TOKENS["text"],
                     shape="triangle", lw=1.2),
}

_ROLE_STYLE = {
    Role.LLM_AGENT: dict(face="#cde2fb", edge="#2a78d6", shape="rect"),
    Role.TOOL: dict(face="#fde3d3", edge="#eb6834", shape="rect"),
    Role.AGGREGATOR: dict(face="#cdf0e2", edge="#1baf7a", shape="rect"),
    Role.ROUTER: dict(face="#fcfcfb", edge="#52514e", shape="diamond"),
    Role.FEEDBACK: dict(face="#f3d2d2", edge="#d03b3b", shape="rect"),
    Role.TRANSFORM: dict(face="#ececea", edge="#52514e", shape="rect"),
    Role.SOURCE: dict(face="#e6e5e1", edge="#52514e", shape="pill"),
    Role.SINK: dict(face="#e6e5e1", edge="#52514e", shape="pill"),
}


# --------------------------------------------------------------------------- #
# shared layout helpers
# --------------------------------------------------------------------------- #
def _short(text: str, width: int = 22, max_lines: int = 5) -> str:
    lines: List[str] = []
    for part in (text or "").split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [""])
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][: width - 1] + "…"]
    return "\n".join(lines)


def _spread(x: Dict[str, float], by_layer: Dict[int, List[str]],
            order: Dict[str, int], min_gap: float) -> None:
    """Push apart nodes that share a layer, then centre each layer."""
    for members in by_layer.values():
        members.sort(key=lambda n: (x[n], order.get(n, 0)))
        for i in range(1, len(members)):
            if x[members[i]] - x[members[i - 1]] < min_gap:
                x[members[i]] = x[members[i - 1]] + min_gap
        span = x[members[-1]] - x[members[0]]
        offset = -x[members[0]] - span / 2.0
        for nid in members:
            x[nid] += offset


# --------------------------------------------------------------------------- #
# fault tree layout
# --------------------------------------------------------------------------- #
def _topological(tree: FaultTree) -> List[str]:
    indeg: Dict[str, int] = {nid: 0 for nid in tree.nodes}
    for node in tree.nodes.values():
        for child in node.children:
            indeg[child] += 1
    queue = [nid for nid, d in indeg.items() if d == 0]
    out: List[str] = []
    while queue:
        nid = queue.pop(0)
        out.append(nid)
        for child in tree.nodes[nid].children:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    return out


def _layers(tree: FaultTree) -> Dict[str, int]:
    """Longest path from the root, so an edge never points upwards."""
    depth: Dict[str, int] = {tree.root: 0}
    for nid in _topological(tree):
        for child in tree.nodes[nid].children:
            depth[child] = max(depth.get(child, 0), depth.get(nid, 0) + 1)
    return depth


def _dfs_order(tree: FaultTree) -> List[str]:
    seen, out = set(), []
    stack = [tree.root]
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        out.append(nid)
        stack.extend(reversed(tree.nodes[nid].children))
    return out


def _positions(
    tree: FaultTree, min_gap: float = 1.06, y_step: float = 1.30
) -> Tuple[Dict[str, Tuple[float, float]], int, int]:
    """Tidy layout: leaves take successive slots left to right, parents centre
    over their own children. On a strict tree that produces subtrees which never
    interleave, so no connector ever crosses another."""
    depth = _layers(tree)
    order = {nid: i for i, nid in enumerate(_dfs_order(tree))}
    by_layer: Dict[int, List[str]] = {}
    for nid, d in depth.items():
        by_layer.setdefault(d, []).append(nid)

    x: Dict[str, float] = {}
    slot = [0.0]

    def place(nid: str, seen: Tuple[str, ...] = ()) -> float:
        if nid in x:
            return x[nid]
        node = tree.nodes[nid]
        kids = [c for c in node.children if c not in seen]
        if kids:
            centres = [place(c, seen + (nid,)) for c in kids]
            x[nid] = (min(centres) + max(centres)) / 2.0
        else:
            x[nid] = slot[0]
            slot[0] += min_gap
        return x[nid]

    place(tree.root)
    for nid in sorted(depth, key=lambda n: (-depth[n], order.get(n, 0))):
        if nid not in x:                       # unreachable safety net
            x[nid] = slot[0]
            slot[0] += min_gap

    _spread(x, by_layer, order, min_gap)
    positions = {nid: (x[nid], -float(depth[nid]) * y_step) for nid in tree.nodes}
    return positions, max(len(m) for m in by_layer.values()), max(by_layer) + 1


# --------------------------------------------------------------------------- #
# gate symbols
# --------------------------------------------------------------------------- #
def _and_gate_path(cx: float, cy: float, w: float, h: float):
    """Flat base, straight sides, semicircular top — the conventional AND gate."""
    from matplotlib.path import Path

    r = w / 2.0
    straight = max(h - r, h * 0.22)
    y0 = cy - h / 2.0            # base
    y1 = y0 + straight           # where the dome starts
    k = 0.5523 * r
    verts = [
        (cx - r, y0), (cx - r, y1),
        (cx - r, y1 + k), (cx - k, y1 + r), (cx, y1 + r),
        (cx + k, y1 + r), (cx + r, y1 + k), (cx + r, y1),
        (cx + r, y0), (cx - r, y0),
    ]
    codes = [
        Path.MOVETO, Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO, Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


def _or_gate_path(cx: float, cy: float, w: float, h: float):
    """Curved sides rising to a point, concave base — the conventional OR gate."""
    from matplotlib.path import Path

    r = w / 2.0
    y0 = cy - h / 2.0
    top = cy + h / 2.0
    verts = [
        (cx - r, y0),
        (cx - r, y0 + h * 0.70), (cx - r * 0.10, top), (cx, top),          # left side
        (cx + r * 0.10, top), (cx + r, y0 + h * 0.70), (cx + r, y0),       # right side
        (cx + r * 0.45, y0 + h * 0.34), (cx - r * 0.45, y0 + h * 0.34), (cx - r, y0),
        (cx - r, y0),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


class _GateProxy:
    """Legend stand-in so the key shows the real gate symbol, not a swatch."""

    def __init__(self, gate: str, label: str):
        self.gate = gate
        self._label = label

    def get_label(self) -> str:
        return self._label


class _GateHandler:
    """Legend handler that draws the gate outline at legend-entry size."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        import matplotlib.patches as mpatches

        w = handlebox.width * 0.62
        h = handlebox.height * 1.25
        cx = handlebox.xdescent + handlebox.width / 2.0
        cy = handlebox.ydescent + handlebox.height / 2.0
        builder = _and_gate_path if orig_handle.gate == "AND" else _or_gate_path
        patch = mpatches.PathPatch(builder(cx, cy, w, h), facecolor="#ffffff",
                                   edgecolor=TOKENS["text"], lw=1.0)
        patch.set_transform(handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


# --------------------------------------------------------------------------- #
# fault tree
# --------------------------------------------------------------------------- #
def plot_fault_tree(
    tree: FaultTree,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    scale: float = 1.0,
    label_width: int = 22,
    show_legend: bool = True,
    as_tree: bool = True,
) -> Any:
    """Draw a synthesised fault tree with the conventional FTA symbols.

    ``as_tree`` (default) repeats shared causes so the structure is a strict
    tree: every subtree is then local and the connectors are plain right angles
    — an event, its gate beneath it, a bus, and a vertical drop to each input,
    exactly as a fault tree is normally drawn. Set it to ``False`` to draw the
    synthesised DAG instead, where a shared cause appears once and its edges run
    across the diagram.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if as_tree:
        try:
            tree = expand_to_tree(tree)
        except ValueError:      # too much sharing to repeat; draw the DAG
            as_tree = False

    y_step = 1.30
    positions, _, depth = _positions(tree, y_step=y_step)
    box_w, box_h = 0.94, 0.62
    gate_w, gate_h, gate_gap = 0.46, 0.34, 0.05

    xs_all = [p[0] for p in positions.values()]
    x_extent = (max(xs_all) - min(xs_all)) + 2 * box_w

    # A wide tree gets compact leaf labels: the event id alone. The id names its
    # component and failure mode, and the full text is in the basic event
    # register, so nothing is lost that the reader cannot look up.
    n_leaves = sum(1 for n in tree.nodes.values() if not n.children)
    compact = n_leaves > 10

    if ax is None:
        # Uniform physical scale: one data unit is always the same size on
        # paper, so a 6-layer tree and a 25-layer tree are equally legible and
        # the labels never shrink out of their boxes.
        fig_w = max(6.5, min(200.0, x_extent * 1.18 * scale))
        fig_h = max(4.0, min(200.0, (depth * y_step + 1.6) * 0.92 * scale))
        _, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig = ax.figure
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_facecolor(TOKENS["surface"])

    def outlet(nid: str) -> float:
        """y at which this node's inputs leave it (below its gate, if any)."""
        _, y = positions[nid]
        node = tree.nodes[nid]
        if node.gate and node.children:
            return y - box_h / 2 - gate_gap - gate_h
        return y - box_h / 2

    # ---- connectors first, so the symbols sit on top of them --------------- #
    # The conventional form: a stub down from the gate, one horizontal bus, and
    # a vertical drop into each input. Only when a child sits further down than
    # the next layer (possible in the DAG form) does a per-edge elbow get used.
    line = dict(color=TOKENS["text_secondary"], lw=1.0, zorder=1,
                solid_joinstyle="miter", solid_capstyle="butt")
    for nid, node in tree.nodes.items():
        if not node.children:
            continue
        x0, _ = positions[nid]
        y_out = outlet(nid)
        tops = {c: positions[c][1] + box_h / 2 for c in node.children}
        level = max(tops.values())
        bus_y = (y_out + level) / 2

        same_level = [c for c in node.children if abs(tops[c] - level) < 1e-9]
        deeper = [c for c in node.children if c not in same_level]

        ax.plot([x0, x0], [y_out, bus_y], **line)
        xs_children = [positions[c][0] for c in same_level]
        if xs_children:
            left, right = min(xs_children + [x0]), max(xs_children + [x0])
            if right - left > 1e-9:
                ax.plot([left, right], [bus_y, bus_y], **line)
            for child in same_level:
                cx = positions[child][0]
                ax.plot([cx, cx], [bus_y, tops[child]], **line)
        for child in deeper:                      # DAG form only
            cx = positions[child][0]
            ax.plot([x0, cx, cx], [bus_y, bus_y, tops[child]], **line)

    # ---- events and gates -------------------------------------------------- #
    for nid, node in tree.nodes.items():
        x, y = positions[nid]
        style = _NODE_STYLE.get(node.ntype, _NODE_STYLE["intermediate"])

        if style["shape"] == "ellipse":
            patch: Any = mpatches.Ellipse((x, y), box_w, box_h)
        elif style["shape"] == "diamond":
            patch = mpatches.Polygon(
                [(x, y + box_h / 2), (x + box_w / 2, y), (x, y - box_h / 2), (x - box_w / 2, y)],
                closed=True,
            )
        elif style["shape"] == "triangle":     # transfer symbol, point downwards
            patch = mpatches.Polygon(
                [(x - box_w / 2, y + box_h / 2), (x + box_w / 2, y + box_h / 2),
                 (x, y - box_h / 2)],
                closed=True,
            )
        else:
            patch = mpatches.FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.012,rounding_size=0.05",
            )
        patch.set_facecolor(style["face"])
        patch.set_edgecolor(style["edge"])
        patch.set_linewidth(style["lw"])
        patch.set_zorder(2)
        ax.add_patch(patch)

        # an ellipse loses its corners, so its text must wrap narrower
        wrap = label_width - 5 if style["shape"] == "ellipse" else label_width
        raw_label = node.label
        is_leaf = node.ntype in ("basic", "undeveloped")
        if compact and is_leaf:
            raw_label = raw_label.split("\n")[0]
        text = _short(raw_label, width=wrap, max_lines=3 if (compact and is_leaf) else 5)
        if node.ntype == "transfer":
            # a triangle narrows downwards, so only the tag and a pointer fit
            text = f"▽ {node.transfer_ref}\nsee {node.component or 'above'}"
        elif node.transfer_ref:
            text = f"△ {node.transfer_ref}   {text}"
        elif node.repeat_of:
            text = "↺ " + text        # the same event, repeated for readability
        text_y = y + box_h * 0.16 if style["shape"] == "triangle" else y
        ax.text(x, text_y, text, ha="center", va="center",
                fontsize=5.9, color=style["text"], zorder=3, linespacing=1.15)

        if node.gate and node.children:
            gate_cy = y - box_h / 2 - gate_gap - gate_h / 2
            path = (_and_gate_path if node.gate == "AND" else _or_gate_path)(
                x, gate_cy, gate_w, gate_h
            )
            gate_patch = mpatches.PathPatch(
                path, facecolor="#ffffff", edgecolor=TOKENS["text"],
                lw=1.1, zorder=2, joinstyle="round",
            )
            ax.add_patch(gate_patch)
            ax.plot([x, x], [y - box_h / 2, gate_cy + gate_h / 2],
                    color=TOKENS["text"], lw=1.1, zorder=1)
            ax.text(x, gate_cy + gate_h * 0.06, node.gate, ha="center", va="center",
                    fontsize=5.0, color=TOKENS["text"], zorder=3)

    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs_all) - box_w, max(xs_all) + box_w)
    ax.set_ylim(min(ys) - box_h, max(ys) + box_h * 1.2)
    ax.set_axis_off()

    hazard = tree.hazard
    fig.suptitle(title or f"{tree.id} — {tree.name}", fontsize=12,
                 color=TOKENS["text"], x=0.012, y=0.998, ha="left", va="top")
    if hazard is not None:
        ax.set_title(
            _short(f"severity: {hazard.severity}   ·   top event: "
                   f"{', '.join(str(d) for d in hazard.deviations)}",
                   width=150, max_lines=2),
            fontsize=7.5, color=TOKENS["text_secondary"], loc="left", pad=6,
        )

    if show_legend:
        handles = [
            mpatches.Patch(facecolor=_NODE_STYLE["top"]["face"],
                           edgecolor=_NODE_STYLE["top"]["edge"], label="top event"),
            mpatches.Patch(facecolor=_NODE_STYLE["intermediate"]["face"],
                           edgecolor=_NODE_STYLE["intermediate"]["edge"],
                           label="intermediate event (rectangle)"),
            mpatches.Patch(facecolor=_NODE_STYLE["basic"]["face"],
                           edgecolor=_NODE_STYLE["basic"]["edge"],
                           label="basic event (circle)"),
            mpatches.Patch(facecolor=_NODE_STYLE["undeveloped"]["face"],
                           edgecolor=_NODE_STYLE["undeveloped"]["edge"],
                           label="undeveloped (diamond)"),
            mpatches.Patch(facecolor=_NODE_STYLE["transfer"]["face"],
                           edgecolor=_NODE_STYLE["transfer"]["edge"],
                           label="transfer ▽A → subtree △A"),
            _GateProxy("AND", label="AND gate"),
            _GateProxy("OR", label="OR gate"),
        ]
        # a narrow figure cannot fit seven entries on one row
        ncol = 7 if fig.get_size_inches()[0] >= 15 else 4
        fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
                   fontsize=7.5, labelcolor=TOKENS["text_secondary"],
                   bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
                   handler_map={_GateProxy: _GateHandler()})
        legend_rows = -(-len(handles) // ncol)
    else:
        legend_rows = 0
    fig.tight_layout(rect=(0.0, 0.012 + 0.030 * legend_rows, 1.0, 0.965))
    return ax


# --------------------------------------------------------------------------- #
# architecture
# --------------------------------------------------------------------------- #
def plot_architecture(
    system: SystemModel,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    show_ports: bool = False,
) -> Any:
    """Draw the LangGraph architecture — the offline equivalent of
    ``display(Image(graph.get_graph().draw_mermaid_png()))``.

    Feedback edges are drawn dashed in the reserved critical colour and labelled,
    because they are the reason the graph cannot be read as a fault tree until
    they are removed.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    ids = list(system.components)
    edges = [(c.src, c.dst, c.label, c.conditional) for c in system.connections]
    back = {(e[0], e[1]) for e in find_back_edges(ids, edges)}
    forward = [e for e in edges if (e[0], e[1]) not in back]

    # layer by longest path from the sources over the forward edges
    depth: Dict[str, int] = {nid: 0 for nid in ids}
    for _ in range(len(ids)):
        changed = False
        for src, dst, _lbl, _cond in forward:
            if depth[dst] < depth[src] + 1:
                depth[dst] = depth[src] + 1
                changed = True
        if not changed:
            break

    by_layer: Dict[int, List[str]] = {}
    for nid in ids:
        by_layer.setdefault(depth[nid], []).append(nid)
    order = {nid: i for i, nid in enumerate(ids)}

    x: Dict[str, float] = {}
    for layer in sorted(by_layer):
        for i, nid in enumerate(sorted(by_layer[layer], key=lambda n: order[n])):
            parents = [x[s] for s, d, _l, _c in forward if d == nid and s in x]
            x[nid] = sum(parents) / len(parents) if parents else float(i)
    _spread(x, by_layer, order, min_gap=1.25)

    y_step = 1.25
    pos = {nid: (x[nid], -depth[nid] * y_step) for nid in ids}
    box_w, box_h = 1.05, 0.52

    # Two kinds of edge cannot be drawn straight. Feedback edges run backwards,
    # and an edge that skips a layer would otherwise be a vertical line hidden
    # behind every component it passes, with its label landing on top of one of
    # them. Both are routed around the right-hand side instead, each in its own
    # lane so that two of them never overlap. An early exit to `__end__` is
    # exactly this case, and it is the edge a reader most needs to see.
    right = max(p[0] for p in pos.values())
    skips = [
        (src, dst, label, conditional)
        for src, dst, label, conditional in edges
        if (src, dst) not in back and depth[dst] - depth[src] > 1
    ]
    skip_lane = {(e[0], e[1]): right + box_w * (0.72 + 0.30 * i)
                 for i, e in enumerate(skips)}
    lane = (max(skip_lane.values()) if skip_lane else right) + box_w * 0.55

    if ax is None:
        left = min(p[0] for p in pos.values())
        span = (lane if back else max([right] + list(skip_lane.values()))) - left
        fig_w = max(6.0, min(40.0, (span + 2 * box_w) * 1.6))
        fig_h = max(3.5, min(40.0, (max(depth.values()) * y_step + 1.8) * 1.15))
        _, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig = ax.figure
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_facecolor(TOKENS["surface"])

    for src, dst, label, conditional in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        if (src, dst) in back:
            dip = y0 - box_h * 0.95          # run below the row, never through a node
            ax.plot([x0, x0, lane, lane], [y0 - box_h / 2, dip, dip, y1],
                    color=TOKENS["critical"], lw=1.3, linestyle="dashed", zorder=1,
                    solid_joinstyle="miter")
            ax.annotate("", xy=(x1 + box_w / 2, y1), xytext=(lane, y1),
                        arrowprops=dict(arrowstyle="-|>", color=TOKENS["critical"],
                                        lw=1.3, linestyle="dashed", shrinkA=0, shrinkB=1),
                        zorder=1)
            ax.text(lane + 0.06, (y0 + y1) / 2, "feedback", fontsize=7,
                    color=TOKENS["critical"], ha="left", va="center", rotation=90)
            continue
        if (src, dst) in skip_lane:
            side = skip_lane[(src, dst)]
            drop = y0 - box_h * 0.62
            ax.plot([x0, x0, side, side], [y0 - box_h / 2, drop, drop, y1],
                    color=TOKENS["text_secondary"], lw=1.1, zorder=1, alpha=0.75,
                    linestyle="dashed" if conditional else "solid",
                    solid_joinstyle="miter")
            ax.annotate("", xy=(x1 + box_w / 2, y1), xytext=(side, y1),
                        arrowprops=dict(arrowstyle="-|>", color=TOKENS["text_secondary"],
                                        lw=1.1, shrinkA=0, shrinkB=1), zorder=1)
            if label:
                ax.text(side + 0.05, (drop + y1) / 2, label, fontsize=7,
                        color=TOKENS["text_secondary"], ha="left", va="center",
                        rotation=90)
            continue
        mid = (y0 - box_h / 2 + y1 + box_h / 2) / 2
        ax.plot([x0, x0, x1, x1], [y0 - box_h / 2, mid, mid, y1 + box_h / 2],
                color=TOKENS["text_secondary"], lw=1.1, zorder=1,
                linestyle="dashed" if conditional else "solid", alpha=0.75)
        ax.annotate("", xy=(x1, y1 + box_h / 2), xytext=(x1, mid),
                    arrowprops=dict(arrowstyle="-|>", color=TOKENS["text_secondary"],
                                    lw=1.1, shrinkA=0, shrinkB=0), zorder=1)
        if label:
            ax.text(x1 + 0.06, mid + 0.05, label, fontsize=7,
                    color=TOKENS["text_secondary"], ha="left", va="bottom")

    for nid in ids:
        comp = system.components[nid]
        cx, cy = pos[nid]
        style = _ROLE_STYLE.get(comp.role, _ROLE_STYLE[Role.TRANSFORM])
        if style["shape"] == "diamond":
            patch: Any = mpatches.Polygon(
                [(cx, cy + box_h / 2), (cx + box_w / 2, cy),
                 (cx, cy - box_h / 2), (cx - box_w / 2, cy)], closed=True)
        elif style["shape"] == "pill":
            patch = mpatches.FancyBboxPatch(
                (cx - box_w / 2 + box_h / 2, cy - box_h / 2), box_w - box_h, box_h,
                boxstyle=f"round,pad=0.0,rounding_size={box_h / 2}")
        else:
            patch = mpatches.FancyBboxPatch(
                (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.012,rounding_size=0.06")
        patch.set_facecolor(style["face"])
        patch.set_edgecolor(style["edge"])
        patch.set_linewidth(1.4)
        patch.set_zorder(2)
        ax.add_patch(patch)

        # Materialised routers are named `<node>::router`; breaking that name on
        # the separator keeps the source node readable instead of splitting it
        # mid-word to fit the wrap width, and the role caption underneath is then
        # the same word twice, so it is dropped rather than crowding the shape.
        label = _short(comp.id.replace("::", "\n::"), width=20, max_lines=2)
        caption = comp.role.value.replace("_", " ")
        if comp.id.rsplit("::", 1)[-1] == comp.role.value:
            caption = ""
        if show_ports and comp.ports_in:
            caption = (caption + f"  ({len(comp.ports_in)} in)").strip()
        ax.text(cx, cy + (0.055 if caption else 0.0), label, ha="center",
                va="center", fontsize=7.6, color=TOKENS["text"], zorder=3,
                linespacing=1.1)
        if caption:
            ax.text(cx, cy - box_h * 0.30, caption, ha="center", va="center",
                    fontsize=6, color=TOKENS["text_secondary"], zorder=3)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    outer = max([max(xs)] + ([lane] if back else []) + list(skip_lane.values()))
    ax.set_xlim(min(xs) - box_w * 0.9, outer + box_w * 0.9)
    ax.set_ylim(min(ys) - box_h * 1.9, max(ys) + box_h * 1.4)
    ax.set_axis_off()
    fig.suptitle(title or f"Architecture — {system.name}", fontsize=12,
                 color=TOKENS["text"], x=0.012, y=0.998, ha="left", va="top")
    subtitle = (f"{len(system.components)} components · {len(system.connections)} connections"
                + ("  ·  dashed grey = conditional edge" if any(e[3] for e in edges) else "")
                + ("  ·  dashed red = feedback, cut before analysis" if back else ""))
    ax.set_title(_short(subtitle, width=96, max_lines=2), fontsize=7.5,
                 color=TOKENS["text_secondary"], loc="left", pad=6)
    fig.tight_layout(rect=(0.0, 0.01, 1.0, 0.965))
    return ax


# --------------------------------------------------------------------------- #
# contribution ranking
# --------------------------------------------------------------------------- #
def plot_importance(
    analysis: TreeAnalysis,
    top_n: int = 12,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
) -> Any:
    """Fussell-Vesely contribution per basic event, ranked.

    One measure, one hue. Single points of failure are marked with the reserved
    critical colour *and* a "single point" tag, so the distinction never rests
    on colour alone.
    """
    import matplotlib.pyplot as plt

    rows = analysis.importance[:top_n]
    if not rows:
        raise ValueError("no importance rows to plot")
    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, max(2.6, 0.42 * len(rows) + 1.4)))
    fig = ax.figure
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_facecolor(TOKENS["surface"])

    labels = [r.event_id for r in rows][::-1]
    values = [r.fussell_vesely for r in rows][::-1]
    spof = [r.single_point for r in rows][::-1]
    colours = [TOKENS["critical"] if s else TOKENS["series"] for s in spof]

    ax.barh(range(len(values)), values, color=colours, height=0.62, zorder=2, linewidth=0)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5, color=TOKENS["text"])
    ax.set_xlabel("Fussell-Vesely contribution to the top event", fontsize=8.5,
                  color=TOKENS["text_secondary"])
    headroom = 1.42 if any(spof) else 1.16
    ax.set_xlim(0, max(values) * headroom if max(values) > 0 else 1)
    ax.tick_params(axis="x", labelsize=7.5, colors=TOKENS["text_secondary"], length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=TOKENS["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    for i, (value, is_spof) in enumerate(zip(values, spof)):
        tag = f"{value:.3f}" + ("   ◆ single point" if is_spof else "")
        ax.text(value + max(values) * 0.015, i, tag, va="center", fontsize=7.5,
                color=TOKENS["text_secondary"])

    ax.set_title(title or f"{analysis.tree.id} — what drives «{analysis.tree.name}»",
                 fontsize=11, color=TOKENS["text"], loc="left", pad=10)
    fig.tight_layout()
    return ax


def plot_cutset_orders(
    analyses: Dict[str, TreeAnalysis],
    ax: Optional[Any] = None,
    title: str = "Minimal cut sets by order",
) -> Any:
    """Cut-set count per order, per hazard — how much defence in depth exists.

    Order 1 is a single point of failure, so a tall order-1 bar is the finding.
    """
    import matplotlib.pyplot as plt

    hazards = sorted(analyses)
    orders = sorted({len(cs) for a in analyses.values() for cs in a.cuts.sets})
    if not orders:
        raise ValueError("no cut sets to plot")
    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, 4.2))
    fig = ax.figure
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_facecolor(TOKENS["surface"])

    # one hue, stepped by order: magnitude, not identity
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#184f95"]
    width = 0.8 / len(orders)
    for j, order in enumerate(orders):
        counts = [sum(1 for cs in analyses[h].cuts.sets if len(cs) == order) for h in hazards]
        offsets = [i - 0.4 + width * (j + 0.5) for i in range(len(hazards))]
        colour = TOKENS["critical"] if order == 1 else ramp[min(j, len(ramp) - 1)]
        ax.bar(offsets, counts, width=width * 0.88, color=colour, zorder=2,
               label=f"order {order}" + (" (single point)" if order == 1 else ""))
        for x_pos, count in zip(offsets, counts):
            if count:
                ax.text(x_pos, count + 0.12, str(count), ha="center", fontsize=7,
                        color=TOKENS["text_secondary"])

    ax.set_xticks(range(len(hazards)))
    ax.set_xticklabels(
        [f"{h}\n" + "\n".join(textwrap.wrap(analyses[h].tree.name, width=22)[:2])
         for h in hazards],
        fontsize=7.5, color=TOKENS["text"],
    )
    ax.set_ylabel("number of minimal cut sets", fontsize=8.5, color=TOKENS["text_secondary"])
    ax.tick_params(axis="y", labelsize=7.5, colors=TOKENS["text_secondary"], length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=TOKENS["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, ncol=len(orders),
              labelcolor=TOKENS["text_secondary"], loc="upper right")
    ax.set_title(title, fontsize=11, color=TOKENS["text"], loc="left", pad=10)
    fig.tight_layout()
    return ax
