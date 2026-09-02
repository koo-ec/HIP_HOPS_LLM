"""Drawing Bayesian networks, with pyAgrum where it works and without it where it does not.

pyAgrum renders through Graphviz, and Graphviz is a separate native binary, not a
Python package --- ``pip install pydot`` does not provide it.  Whether it is
present varies by environment and by the day: it ships on a current Colab image
and not on many Windows installs, and pyAgrum's own import prints a warning when
it is missing.  A visualisation layer that only works when ``dot`` is on the PATH
is a visualisation layer that silently produces nothing exactly when a reader
most needs the picture, so this one does not assume either way ---
:func:`graphviz_available` runs ``dot -V`` and the backend follows the answer.

:class:`BayesNetView` therefore has two backends and one behaviour:

* ``"pyagrum"`` --- ``pyagrum.lib.notebook``, the richest output: node shading by
  posterior, inference histograms, side-by-side prior/posterior views;
* ``"matplotlib"`` --- a layered DAG drawn directly, which needs nothing beyond
  matplotlib.

``backend="auto"`` (the default) picks pyAgrum when Graphviz is genuinely
callable and matplotlib otherwise, so ``bn.show()`` always draws something.

The matplotlib backend draws the same *kind* of picture pyAgrum's
``showInference`` does --- each node is a titled box holding one labelled bar per
state --- rather than a reduced one.  A fallback that says less than the thing it
replaces trains a reader to distrust it, and the two views appearing side by side
across environments should be comparable at a glance.  Every bar carries its
percentage as text, so the reading never depends on judging a bar length or on
seeing colour.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .cpt import FAIL, LABELS, OK

if TYPE_CHECKING:  # pragma: no cover
    from .network import BayesianNetwork

__all__ = ["BayesNetView", "graphviz_available", "PALETTE"]

#: Node shading, low failure probability to high.  Status colour is never the
#: only signal --- every node also carries its probability as text.
PALETTE = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "muted": "#6b6b6b",
    "edge": "#9a9a9a",
    "basic": "#e8eef7",
    "basic_line": "#5b7fa6",
    "gate": "#f3efe6",
    "gate_line": "#a08a5f",
    "top": "#f7e7e4",
    "top_line": "#b26a5c",
    "evidence": "#2f6f4f",
    "low": "#eef4ee",
    "high": "#f6dcd6",
    # The inference box: a white interior with a thin grey border, a title strip
    # tinted by node kind, and pyAgrum's own bar colour so the two backends read
    # as one picture rather than two.
    "node_face": "#ffffff",
    "node_line": "#b0b0b0",
    "header_line": "#c9c9c9",
    "bar": "#8fbc8f",
    "caption": "#4d7098",
    "note": "#2f7d4f",
    "note_muted": "#7e9a89",
}

#: Where each column of a state row sits, as a fraction of the box width: where
#: the state label ends, where a full-length bar starts and ends, and the right
#: margin the percentage may not cross.  The percentage trails its own bar, and
#: falls back to sitting inside the bar's right end when there is no room after
#: it --- which is the only way a row at 100% stays readable.
_COLUMNS = {"label_end": 0.20, "bar_start": 0.23, "bar_end": 0.72, "value_end": 0.97}

#: Corner radius of a node box, in data units.  Shared with the title strip so
#: that the two round together instead of a square corner poking out of a round
#: one.
_ROUNDING = 0.03


def graphviz_available() -> bool:
    """Is the Graphviz ``dot`` binary actually callable?

    ``import pydot`` succeeding proves nothing: the Python binding is not the
    renderer.  This runs ``dot -V``.
    """
    exe = shutil.which("dot")
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "-V"], capture_output=True, timeout=10, check=True
        )
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


@dataclass(frozen=True)
class _Geometry:
    """Node box geometry in data units, with the font sizes that suit it.

    Nothing here is a constant, because the layout stretches: a two-column
    network and a ten-column one are drawn on axes with very different
    inches-per-data-unit, so a box fixed in data units comes out three times
    wider on one than on the other.  Sizing from the figure keeps a node the
    same physical size --- and its text the same physical size --- whatever the
    shape of the graph.
    """

    box_w: float
    box_h: float
    header_h: float
    row_h: float
    header_fs: float
    row_fs: float
    width_in: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _geometry(
    figsize: Tuple[float, float], x_span: float, y_span: float, detailed: bool
) -> _Geometry:
    """Box and font sizes for a figure of ``figsize`` showing ``x_span`` units.

    The 0.94/0.86 factors are the fraction of the figure the axes actually get
    once ``tight_layout`` has taken its margins; they only need to be close,
    since every size derived from them is clamped.
    """
    in_per_x = figsize[0] * 0.94 / max(x_span, 1e-6)
    in_per_y = figsize[1] * 0.86 / max(y_span, 1e-6)
    box_w = min(0.90, 1.65 / in_per_x)
    if not detailed:
        box_h = min(0.36, 0.50 / in_per_y)
        return _Geometry(
            box_w=box_w,
            box_h=box_h,
            header_h=box_h,
            row_h=0.0,
            header_fs=_clamp(72.0 * box_h * in_per_y * 0.26, 5.0, 8.5),
            row_fs=0.0,
            width_in=box_w * in_per_x,
        )
    box_h = min(0.60, 0.92 / in_per_y)
    header_h, row_h = 0.30 * box_h, 0.35 * box_h
    header_fs = _clamp(72.0 * header_h * in_per_y * 0.38, 4.5, 9.0)
    return _Geometry(
        box_w=box_w,
        box_h=box_h,
        header_h=header_h,
        row_h=row_h,
        header_fs=header_fs,
        row_fs=_clamp(72.0 * row_h * in_per_y * 0.30, 4.0, header_fs - 0.4),
        width_in=box_w * in_per_x,
    )


def _rounded_top(
    x: float, y: float, width: float, height: float, radius: float, **kwargs: Any
) -> Any:
    """A rectangle rounded at the top corners and square at the bottom ones.

    The title strip has to curve exactly where the node box curves and sit flush
    against the rule beneath it, and no stock ``boxstyle`` does both.  The
    quadratic corners are built the same way :class:`matplotlib.patches.BoxStyle`
    builds ``round``, so the two arcs coincide.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    r = max(0.0, min(radius, width / 2, height))
    right, ceiling = x + width, y + height
    return PathPatch(
        Path(
            [
                (x, y),
                (x, ceiling - r),
                (x, ceiling),
                (x + r, ceiling),
                (right - r, ceiling),
                (right, ceiling),
                (right, ceiling - r),
                (right, y),
                (x, y),
            ],
            [
                Path.MOVETO,
                Path.LINETO,
                Path.CURVE3,
                Path.CURVE3,
                Path.LINETO,
                Path.CURVE3,
                Path.CURVE3,
                Path.LINETO,
                Path.CLOSEPOLY,
            ],
        ),
        **kwargs,
    )


def _fits(text: str, width_in: float, fontsize: float) -> float:
    """``fontsize``, shrunk if ``text`` would overrun ``width_in`` inches.

    Matplotlib will happily draw a label wider than the box it belongs to, and a
    header that overruns its node reads as a bug rather than as a long name.
    0.58 em is a serviceable mean advance for DejaVu Sans, matplotlib's default
    face; the estimate only has to be good enough to keep the text inside.
    """
    longest = max((len(line) for line in text.split("\n")), default=0)
    if longest == 0:
        return fontsize
    needed = longest * 0.58 * fontsize / 72.0
    return fontsize if needed <= width_in else max(3.2, fontsize * width_in / needed)


@dataclass
class BayesNetView:
    """A drawable view of a :class:`~hiphopsllm.bayes.network.BayesianNetwork`.

    Parameters
    ----------
    network
        The network to draw.
    backend
        ``"auto"``, ``"pyagrum"`` or ``"matplotlib"``.
    evidence
        Observations to condition on; nodes are then shaded by their posterior
        and the evidence nodes are outlined.  Keys may be variable names, fault
        tree node ids or basic event ids.
    show_probabilities
        Draw the per-state bars inside each node (matplotlib backend).  With
        ``False`` the nodes collapse to compact labelled boxes.
    max_label
        Truncate node labels to this many characters.
    annotations
        Lines of explanatory text drawn above the graph, the first in green and
        the rest in a lighter grey-green.  Use them to say what the reader is
        looking at --- which hazard, which evidence, which bound.
    """

    network: "BayesianNetwork"
    backend: str = "auto"
    evidence: Optional[Mapping[str, Any]] = None
    show_probabilities: bool = True
    max_label: int = 28
    figsize: Optional[Tuple[float, float]] = None
    annotations: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        if self.backend not in ("auto", "pyagrum", "matplotlib"):
            raise ValueError("backend must be 'auto', 'pyagrum' or 'matplotlib'")

    # -- backend selection --------------------------------------------------- #
    @property
    def resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if not graphviz_available():
            return "matplotlib"
        try:
            self.network.net  # forces the pyAgrum build
        except Exception:
            return "matplotlib"
        return "pyagrum"

    # -- public drawing ------------------------------------------------------ #
    def show(self, evidence: Optional[Mapping[str, Any]] = None) -> Any:
        """Draw the network.  With evidence, draw the posterior view."""
        ev = evidence if evidence is not None else self.evidence
        if self.resolved_backend == "pyagrum":
            return self._show_pyagrum(ev)
        return self._show_matplotlib(ev)

    def show_inference(self, evidence: Optional[Mapping[str, Any]] = None) -> Any:
        """Draw the network with each node's posterior shown."""
        return self.show(evidence if evidence is not None else self.evidence or {})

    def figure(self, evidence: Optional[Mapping[str, Any]] = None) -> Any:
        """Return the matplotlib ``Figure`` without displaying it.

        :meth:`show` is for notebooks: inside one it calls ``display()`` and
        returns ``None``, because returning the figure as well would render it
        twice.  That leaves no handle for a caller who wants to adjust the
        figure, embed it in a larger layout, or assert something about it, which
        is what this is for::

            fig = bn.view().figure()
            fig.set_size_inches(14, 9)
            fig.savefig("bn.pdf")

        Always the matplotlib backend, so it never needs Graphviz.
        """
        return self._figure(evidence if evidence is not None else self.evidence)

    def side_by_side(self, evidence: Optional[Mapping[str, Any]] = None) -> Any:
        """Structure and inference next to each other (pyAgrum only).

        Falls back to a single annotated matplotlib figure when Graphviz is
        unavailable, rather than raising.
        """
        if self.resolved_backend != "pyagrum":
            return self._show_matplotlib(
                evidence if evidence is not None else self.evidence
            )
        import pyagrum.lib.notebook as gnb  # type: ignore

        net = self.network.net
        ev = self.network._resolve_evidence(
            evidence if evidence is not None else self.evidence
        )
        return gnb.sideBySide(
            net,
            gnb.getInference(net, evs=ev) if ev else gnb.getInference(net),
            captions=["structure", "inference"],
        )

    # -- files --------------------------------------------------------------- #
    def to_dot(self) -> str:
        """Graphviz source for the network.

        Written by hand rather than through pyAgrum so that it works without the
        ``dot`` binary --- the text is useful on its own, and can be rendered
        elsewhere.
        """
        lines = [
            f'digraph "{self.network.name}" {{',
            "  rankdir=BT;",
            f'  bgcolor="{PALETTE["surface"]}";',
            '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
            'fontsize=10, penwidth=1.2];',
            f'  edge [color="{PALETTE["edge"]}", arrowsize=0.7];',
        ]
        probs = self._probabilities(self.evidence)
        for var in self.network.cpts.order:
            cpt = self.network.cpts[var]
            fill, line = self._colours(var, cpt.is_root)
            label = self._label(var)
            if self.show_probabilities:
                label += f"\\n{probs.get(var, float('nan')):.3f}"
            lines.append(
                f'  "{var}" [label="{label}", fillcolor="{fill}", color="{line}"];'
            )
        for var in self.network.cpts.order:
            for parent in self.network.cpts[var].parents:
                lines.append(f'  "{parent}" -> "{var}";')
        lines.append("}")
        return "\n".join(lines)

    def to_png(self, path: str, dpi: int = 150) -> str:
        """Write a PNG.  Uses matplotlib, so no Graphviz is required."""
        import matplotlib

        fig = self._figure(self.evidence)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        matplotlib.pyplot.close(fig)
        return path

    def to_svg(self, path: Optional[str] = None) -> str:
        """SVG source (and optionally a file).  Rendered by matplotlib."""
        import io

        import matplotlib

        fig = self._figure(self.evidence)
        buffer = io.StringIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        matplotlib.pyplot.close(fig)
        svg = buffer.getvalue()
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(svg)
        return svg

    # -- pyAgrum backend ----------------------------------------------------- #
    def _show_pyagrum(self, evidence: Optional[Mapping[str, Any]]) -> Any:
        import pyagrum.lib.notebook as gnb  # type: ignore

        net = self.network.net
        if evidence:
            return gnb.showInference(
                net, evs=self.network._resolve_evidence(evidence)
            )
        return gnb.showInference(net)

    # -- matplotlib backend -------------------------------------------------- #
    def _show_matplotlib(self, evidence: Optional[Mapping[str, Any]]) -> Any:
        import matplotlib.pyplot as plt

        fig = self._figure(evidence)
        try:  # inside a notebook, returning the axes displays it
            from IPython.display import display  # type: ignore

            display(fig)
            plt.close(fig)
            return None
        except Exception:
            return fig.axes[0]

    def _figure(self, evidence: Optional[Mapping[str, Any]]) -> Any:
        import matplotlib.pyplot as plt

        layers = self._layers()
        posteriors, elapsed_ms = self._posteriors_timed(evidence)
        probs = {var: states[FAIL] for var, states in posteriors.items()}
        observed = set(self.network._resolve_evidence(evidence)) if evidence else set()
        # Without posteriors there is nothing to put in the bars, so the compact
        # box is the honest rendering rather than a row of empty tracks.
        detailed = self.show_probabilities and bool(posteriors)

        positions: Dict[str, Tuple[float, float]] = {}
        width = max((len(v) for v in layers.values()), default=1)
        for depth, members in layers.items():
            for i, var in enumerate(members):
                x = (i + 0.5) * width / max(len(members), 1)
                positions[var] = (x, float(depth))

        n_layers = max(layers) + 1 if layers else 1
        # Every line given is drawn, and the headroom grows to match: quietly
        # dropping the third one would lose whatever it said.
        notes = [str(note) for note in (self.annotations or [])]
        figsize = self.figsize or (
            max(7.0, 1.9 * width),
            max(3.8, (1.75 if detailed else 1.5) * n_layers + 0.35 * len(notes)),
        )
        x_span = width + 0.7
        y_span = n_layers + 0.9 + 0.3 * len(notes)
        geom = _geometry(figsize, x_span, y_span, detailed)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(PALETTE["surface"])
        ax.set_facecolor(PALETTE["surface"])

        # Arrows first, so that the boxes cover the ends rather than the reverse.
        for var in self.network.cpts.order:
            for parent in self.network.cpts[var].parents:
                x0, y0 = positions[parent]
                x1, y1 = positions[var]
                ax.annotate(
                    "",
                    xy=(x1, y1 - geom.box_h / 2),
                    xytext=(x0, y0 + geom.box_h / 2),
                    arrowprops=dict(
                        arrowstyle="-|>", color=PALETTE["edge"], lw=1.0,
                        shrinkA=1, shrinkB=1,
                    ),
                    zorder=1,
                )

        self._draw_nodes(ax, positions, geom, posteriors, probs, observed, detailed)

        ax.set_xlim(-0.6, width + 0.1)
        top_limit = (n_layers - 1) + geom.box_h / 2 + 0.28 + 0.22 * len(notes)
        ax.set_ylim(-geom.box_h / 2 - 0.66, top_limit)
        ax.axis("off")

        title = self.network.name
        if observed:
            title += "  ·  evidence: " + ", ".join(sorted(observed))
        ax.set_title(title, fontsize=10, color=PALETTE["text"], pad=12)

        for i, note in enumerate(notes):
            ax.text(
                -0.55, top_limit - 0.22 * (i + 0.6), note, fontsize=8.5,
                color=PALETTE["note"] if i == 0 else PALETTE["note_muted"],
                va="center", zorder=6,
            )

        # pyAgrum stamps the inference cost under its own graphs; the number is
        # worth keeping, because "diagnosis is cheap on this network" is a claim
        # a reader should be able to check rather than take on trust.
        ax.text(
            -0.55, -geom.box_h / 2 - 0.24, f"Inference in {elapsed_ms:.2f}ms",
            fontsize=8, color=PALETTE["caption"], style="italic", va="center",
        )
        caption = (
            "bars show P(state); every bar is labelled with its percentage"
            if detailed
            else "shaded by P(Fail)"
        )
        if observed:
            caption += "  ·  green outline = observed"
        if not graphviz_available():
            caption += "  ·  drawn with matplotlib (Graphviz not on PATH)"
        ax.text(
            -0.55, -geom.box_h / 2 - 0.46, caption, fontsize=7,
            color=PALETTE["muted"], va="center",
        )
        fig.tight_layout()
        return fig

    def _draw_nodes(
        self,
        ax: Any,
        positions: Mapping[str, Tuple[float, float]],
        geom: _Geometry,
        posteriors: Mapping[str, Tuple[float, float]],
        probs: Mapping[str, float],
        observed: Set[str],
        detailed: bool,
    ) -> None:
        """Draw one titled box per variable, with a labelled bar per state.

        The interiors --- title strips, separators, bars --- go into collections
        instead of being added to the axes one at a time.  ``ax.patches`` then
        still holds exactly one artist per variable, so "is every node drawn?"
        stays a question that can be answered by counting; the alternative buries
        fifteen boxes among ninety rectangles.
        """
        import matplotlib.patches as mpatches
        from matplotlib.collections import LineCollection, PatchCollection

        headers: List[Any] = []
        bars: List[Any] = []
        separators: List[List[Tuple[float, float]]] = []

        for var, (x, y) in positions.items():
            cpt = self.network.cpts[var]
            fill, line = self._colours(var, cpt.is_root, probs.get(var))
            states = posteriors.get(var)
            rich = detailed and states is not None
            left, right = x - geom.box_w / 2, x + geom.box_w / 2
            top, bottom = y + geom.box_h / 2, y - geom.box_h / 2
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (left, bottom),
                    geom.box_w,
                    geom.box_h,
                    boxstyle=f"round,pad=0,rounding_size={_ROUNDING}",
                    facecolor=PALETTE["node_face"] if rich else fill,
                    edgecolor=(
                        PALETTE["evidence"]
                        if var in observed
                        else (PALETTE["node_line"] if rich else line)
                    ),
                    linewidth=2.0 if var in observed else (0.9 if rich else 1.1),
                    zorder=3,
                )
            )
            if not rich:
                text = self._label(var, wrap=True)
                if self.show_probabilities and var in probs:
                    text += f"\n{probs[var]:.3f}"
                ax.text(
                    x, y, text, ha="center", va="center", zorder=5,
                    fontsize=_fits(text, geom.width_in * 0.94, geom.header_fs),
                    color=PALETTE["text"], linespacing=1.25,
                )
                continue

            # Inset by a hair so the strip does not paint over the box's own
            # border --- which, on an observed node, is the green outline.
            inset = 0.008
            headers.append(
                _rounded_top(
                    left + inset,
                    top - geom.header_h,
                    geom.box_w - 2 * inset,
                    geom.header_h - inset,
                    _ROUNDING - inset,
                    facecolor=fill,
                    edgecolor="none",
                )
            )
            separators.append(
                [(left, top - geom.header_h), (right, top - geom.header_h)]
            )
            header = self._label(var)
            ax.text(
                x, top - geom.header_h / 2, header,
                ha="center", va="center", zorder=5,
                fontsize=_fits(header, geom.width_in * 0.92, geom.header_fs),
                color=PALETTE["text"],
            )

            bar_h = geom.row_h * 0.58
            span = _COLUMNS["bar_end"] - _COLUMNS["bar_start"]
            for state in (OK, FAIL):
                probability = _clamp(states[state], 0.0, 1.0)
                centre = top - geom.header_h - (state + 0.5) * geom.row_h
                ax.text(
                    left + _COLUMNS["label_end"] * geom.box_w, centre, LABELS[state],
                    ha="right", va="center", fontsize=geom.row_fs,
                    color=PALETTE["muted"], zorder=5,
                )
                bar_end = _COLUMNS["bar_start"] + span * probability
                bars.append(
                    mpatches.Rectangle(
                        (
                            left + _COLUMNS["bar_start"] * geom.box_w,
                            centre - bar_h / 2,
                        ),
                        span * geom.box_w * probability,
                        bar_h,
                        facecolor=PALETTE["bar"],
                        edgecolor="none",
                    )
                )
                # Printed for every state, a state at 0% included: an empty bar
                # and an absent bar look identical, and only one of them is true.
                value = f"{probability * 100:.2f}%"
                needed = len(value) * 0.58 * geom.row_fs / 72.0 / geom.width_in
                after = bar_end + 0.025 + needed <= _COLUMNS["value_end"]
                ax.text(
                    left + (bar_end + 0.025 if after else _COLUMNS["value_end"])
                    * geom.box_w,
                    centre, value, ha="left" if after else "right", va="center",
                    fontsize=geom.row_fs, color=PALETTE["text"], zorder=5,
                )

        if headers:
            strips = PatchCollection(headers, match_original=True, zorder=4)
            strips.set_gid("bayesnet-headers")
            ax.add_collection(strips)
            rules = LineCollection(
                separators, colors=PALETTE["header_line"], linewidths=0.8, zorder=4
            )
            rules.set_gid("bayesnet-separators")
            ax.add_collection(rules)
        if bars:
            drawn = PatchCollection(bars, match_original=True, zorder=4)
            drawn.set_gid("bayesnet-bars")
            ax.add_collection(drawn)

    # -- helpers ------------------------------------------------------------- #
    def _layers(self, sweeps: int = 4) -> Dict[int, List[str]]:
        """Longest-path layering, then barycentre ordering within each layer.

        Fault-tree networks share sub-trees heavily --- that sharing is the whole
        reason the network gives an exact answer where cut sets give a bound ---
        so a naive layout produces a thicket of crossing edges.  Sweeping the
        barycentre heuristic up and down a few times is cheap and makes the
        picture readable.
        """
        depth: Dict[str, int] = {}
        for var in self.network.cpts.order:
            parents = self.network.cpts[var].parents
            depth[var] = 0 if not parents else 1 + max(depth[p] for p in parents)
        layers: Dict[int, List[str]] = {}
        for var, d in depth.items():
            layers.setdefault(d, []).append(var)
        layers = dict(sorted(layers.items()))

        children: Dict[str, List[str]] = {v: [] for v in depth}
        for var in self.network.cpts.order:
            for parent in self.network.cpts[var].parents:
                children[parent].append(var)

        index = {v: i for members in layers.values() for i, v in enumerate(members)}
        for sweep in range(sweeps):
            downward = sweep % 2 == 0
            keys = sorted(layers) if downward else sorted(layers, reverse=True)
            for d in keys:
                neighbours = (
                    (lambda v: list(self.network.cpts[v].parents))
                    if downward
                    else (lambda v: children[v])
                )

                def barycentre(v: str) -> float:
                    ns = neighbours(v)
                    return (
                        sum(index[n] for n in ns) / len(ns) if ns else float(index[v])
                    )

                layers[d] = sorted(layers[d], key=barycentre)
                for i, v in enumerate(layers[d]):
                    index[v] = i
        return layers

    def _posteriors_timed(
        self, evidence: Optional[Mapping[str, Any]]
    ) -> Tuple[Dict[str, Tuple[float, float]], float]:
        """Every variable's ``[P(OK), P(Fail)]``, and the milliseconds it took.

        The clock is started around the inference itself and not around the
        drawing, because the caption claims the cost of the *answer*; reporting
        the cost of the picture under that wording would be a small lie that
        happens to be a hundred times larger.
        """
        started = time.perf_counter()
        try:
            out = {
                var: (
                    float(states[OK]),
                    float(states[FAIL]),
                )
                for var, states in (
                    (v, self.network.posterior(v, evidence=evidence))
                    for v in self.network.cpts.order
                )
            }
        except Exception:  # pragma: no cover - inference should not fail here
            out = {}
        return out, (time.perf_counter() - started) * 1000.0

    def _probabilities(
        self, evidence: Optional[Mapping[str, Any]]
    ) -> Dict[str, float]:
        posteriors, _ = self._posteriors_timed(evidence)
        return {var: states[FAIL] for var, states in posteriors.items()}

    def _label(self, var: str, wrap: bool = False) -> str:
        node_id = self.network.cpts.node_of.get(var, var)
        cpt = self.network.cpts[var]
        text = node_id
        for event_id, variable in self.network.cpts.event_variable.items():
            if variable == var:
                text = event_id
                break
        else:
            if cpt.gate:
                text = f"{node_id}  [{cpt.gate}]"
        if len(text) > self.max_label:
            text = text[: self.max_label - 1] + "…"
        if wrap and len(text) > 16:
            cut = text.rfind("-", 0, 18)
            cut = cut if cut > 6 else 16
            text = text[:cut] + "\n" + text[cut:]
        return text

    def _colours(
        self, var: str, is_root: bool, probability: Optional[float] = None
    ) -> Tuple[str, str]:
        if var == self.network.top:
            return PALETTE["top"], PALETTE["top_line"]
        if is_root:
            base, line = PALETTE["basic"], PALETTE["basic_line"]
        else:
            base, line = PALETTE["gate"], PALETTE["gate_line"]
        if probability is None:
            return base, line
        return _blend(base, PALETTE["high"], min(max(probability, 0.0), 1.0)), line


def _blend(colour_a: str, colour_b: str, t: float) -> str:
    a = tuple(int(colour_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(colour_b[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(x + (y - x) * t) for x, y in zip(a, b))
    return "#{:02x}{:02x}{:02x}".format(*mixed)
