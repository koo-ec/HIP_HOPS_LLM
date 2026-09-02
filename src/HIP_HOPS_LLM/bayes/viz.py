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
* ``"matplotlib"`` --- a layered DAG drawn directly, shaded by the same
  posteriors, which needs nothing beyond matplotlib.

``backend="auto"`` (the default) picks pyAgrum when Graphviz is genuinely
callable and matplotlib otherwise, so ``bn.show()`` always draws something.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from .cpt import FAIL

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
}


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


@dataclass
class BayesNetView:
    """A drawable view of a :class:`~HIP_HOPS_LLM.bayes.network.BayesianNetwork`.

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
        Print ``P(Fail)`` inside each node (matplotlib backend).
    max_label
        Truncate node labels to this many characters.
    """

    network: "BayesianNetwork"
    backend: str = "auto"
    evidence: Optional[Mapping[str, Any]] = None
    show_probabilities: bool = True
    max_label: int = 28
    figsize: Optional[Tuple[float, float]] = None

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
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        layers = self._layers()
        probs = self._probabilities(evidence)
        observed = set(self.network._resolve_evidence(evidence)) if evidence else set()

        positions: Dict[str, Tuple[float, float]] = {}
        width = max((len(v) for v in layers.values()), default=1)
        for depth, members in layers.items():
            for i, var in enumerate(members):
                x = (i + 0.5) * width / max(len(members), 1)
                positions[var] = (x, float(depth))

        n_layers = max(layers) + 1 if layers else 1
        figsize = self.figsize or (
            max(7.0, 1.9 * width),
            max(3.5, 1.5 * n_layers),
        )
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(PALETTE["surface"])
        ax.set_facecolor(PALETTE["surface"])

        box_w, box_h = 0.78, 0.36
        for var in self.network.cpts.order:
            for parent in self.network.cpts[var].parents:
                x0, y0 = positions[parent]
                x1, y1 = positions[var]
                ax.annotate(
                    "",
                    xy=(x1, y1 - box_h / 2),
                    xytext=(x0, y0 + box_h / 2),
                    arrowprops=dict(
                        arrowstyle="-|>", color=PALETTE["edge"], lw=1.0,
                        shrinkA=1, shrinkB=1,
                    ),
                    zorder=1,
                )

        for var, (x, y) in positions.items():
            cpt = self.network.cpts[var]
            fill, line = self._colours(var, cpt.is_root, probs.get(var))
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x - box_w / 2, y - box_h / 2),
                    box_w,
                    box_h,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=fill,
                    edgecolor=PALETTE["evidence"] if var in observed else line,
                    linewidth=2.0 if var in observed else 1.1,
                    zorder=2,
                )
            )
            text = self._label(var, wrap=True)
            if self.show_probabilities and var in probs:
                text += f"\n{probs[var]:.3f}"
            ax.text(
                x, y, text, ha="center", va="center", fontsize=7.2,
                color=PALETTE["text"], zorder=3, linespacing=1.25,
            )

        ax.set_xlim(-0.6, width + 0.1)
        ax.set_ylim(-0.7, n_layers - 0.3)
        ax.axis("off")
        title = self.network.name
        if observed:
            title += "  ·  evidence: " + ", ".join(sorted(observed))
        ax.set_title(title, fontsize=10, color=PALETTE["text"], pad=12)
        caption = (
            "shaded by P(Fail); green outline = observed"
            if observed
            else "shaded by P(Fail)"
        )
        if not graphviz_available():
            caption += "  ·  drawn with matplotlib (Graphviz not on PATH)"
        ax.text(
            0.0, -0.6, caption, fontsize=7, color=PALETTE["muted"],
            transform=ax.transData,
        )
        fig.tight_layout()
        return fig

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

    def _probabilities(
        self, evidence: Optional[Mapping[str, Any]]
    ) -> Dict[str, float]:
        try:
            return {
                var: float(
                    self.network.posterior(var, evidence=evidence)[FAIL]
                )
                for var in self.network.cpts.order
            }
        except Exception:  # pragma: no cover - inference should not fail here
            return {}

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
