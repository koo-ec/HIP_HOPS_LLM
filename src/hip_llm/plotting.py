"""Publication-quality plotting for the HIP-LLM replication.

Every figure produced here is drawn *only* from arrays computed by this package.
No curve is digitised, adjusted or moved after the fact.  Each figure is written
as 300-dpi PNG, vector PDF and vector SVG, and is accompanied by a sidecar
``<name>.meta.json`` recording the configuration hash, data-source label, seeds,
generation time and git commit.

Envelope shading follows the paper's convention: the band between the lower and
upper CDF curves is filled, with per-model hatch/colour kept identical across
Figs. 3-5 and 8-10 so that models are comparable across panels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

# Only force the headless backend when no interactive/inline one is already in
# place.  Unconditionally calling matplotlib.use("Agg") on import would break
# inline figures in Jupyter and Google Colab, where the notebook front-end has
# already selected a backend.
if matplotlib.get_backend().lower() in {"agg", ""} or not hasattr(
    __import__("sys").modules.get("matplotlib.pyplot", None), "show"
):
    try:
        _current = matplotlib.get_backend().lower()
        if not any(k in _current for k in ("inline", "ipympl", "widget", "nbagg",
                                           "qt", "tk", "macosx", "gtk", "wx")):
            matplotlib.use("Agg", force=False)
    except Exception:  # pragma: no cover - backend selection is best effort
        pass

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .schemas import CDFEnvelope, ReliabilityEnvelope  # noqa: E402

__all__ = [
    "MODEL_STYLE",
    "FigureMetadata",
    "save_figure",
    "plot_cdf_envelopes",
    "plot_envelope_grid",
    "plot_expected_reliability",
    "plot_envelope_width",
    "plot_memory_growth",
    "plot_scalability",
    "plot_timing_breakdown",
    "draw_hierarchy_concept",
    "draw_hierarchy_detail",
    "apply_house_style",
]

#: Consistent per-model styling across every panel (mirrors the paper's legend).
MODEL_STYLE: dict[str, dict[str, Any]] = {
    "GPT-4o": {"color": "#ff7f0e", "hatch": "xxx", "facecolor": "#ff7f0e", "alpha": 0.35},
    "GPT-4o-mini": {"color": "#000000", "hatch": "...", "facecolor": "#ffffff", "alpha": 1.0},
    "Sonnet 4.5": {"color": "#2ca02c", "hatch": None, "facecolor": "#2ca02c", "alpha": 0.55},
    "Haiku 3.5": {"color": "#9467bd", "hatch": "...", "facecolor": "#9467bd", "alpha": 0.40},
}

_FALLBACK_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def apply_house_style() -> None:
    """Set rcParams once, so every figure in the notebook looks identical."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linewidth": 0.5,
            "legend.fontsize": 8.5,
            "legend.framealpha": 0.95,
            "lines.linewidth": 1.4,
            "hatch.linewidth": 0.6,
            "savefig.bbox": "tight",
            "figure.autolayout": False,
        }
    )


def _style_for(label: str, index: int) -> dict[str, Any]:
    if label in MODEL_STYLE:
        return dict(MODEL_STYLE[label])
    colour = _FALLBACK_CYCLE[index % len(_FALLBACK_CYCLE)]
    return {"color": colour, "hatch": None, "facecolor": colour, "alpha": 0.35}


@dataclass(frozen=True)
class FigureMetadata:
    """Provenance stamped onto every saved figure."""

    figure_id: str
    caption: str
    config_hash: str
    data_source: str
    seeds: Mapping[str, int]
    git_commit: str | None = None
    extra: Mapping[str, Any] = None  # type: ignore[assignment]

    def as_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "caption": self.caption,
            "config_hash": self.config_hash,
            "data_source": self.data_source,
            "seeds": dict(self.seeds),
            "git_commit": self.git_commit,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "extra": dict(self.extra or {}),
        }


def save_figure(
    fig: plt.Figure, outdir: str | Path, name: str, meta: FigureMetadata
) -> dict[str, Path]:
    """Write PNG (300 dpi), PDF and SVG plus a sidecar metadata JSON."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = meta.as_dict()

    paths: dict[str, Path] = {}
    for ext in ("png", "pdf", "svg"):
        p = outdir / f"{name}.{ext}"
        kwargs: dict[str, Any] = {}
        if ext == "png":
            kwargs["metadata"] = {
                "Title": meta.figure_id,
                "Description": json.dumps(payload, sort_keys=True),
                "Software": "hip-llm-replication",
            }
        elif ext == "pdf":
            kwargs["metadata"] = {
                "Title": meta.figure_id,
                "Subject": meta.caption[:400],
                "Keywords": f"config={meta.config_hash};source={meta.data_source}",
                "Creator": "hip-llm-replication",
            }
        else:
            kwargs["metadata"] = {"Title": meta.figure_id, "Description": meta.caption[:400]}
        fig.savefig(p, **kwargs)
        paths[ext] = p

    meta_path = outdir / f"{name}.meta.json"
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    paths["meta"] = meta_path
    return paths


# --------------------------------------------------------------------------- #
# CDF envelope panels (Figs. 3, 4, 5, 6, 7, 9, 10b-d)
# --------------------------------------------------------------------------- #
def plot_cdf_envelopes(
    ax: plt.Axes,
    envelopes: Mapping[str, CDFEnvelope],
    title: str,
    xlim: tuple[float, float] | None = None,
    xlabel: str = "",
    ylabel: str = "CDF",
    legend_loc: str = "upper left",
    legend_title: str | None = None,
) -> plt.Axes:
    """Draw one panel of lower/upper CDF envelopes.

    The shaded band spans ``[lower(t), upper(t)]``.  Because a lower CDF is
    stochastically larger, the *left* boundary of a band is its conservative
    (upper-CDF) edge and the *right* boundary is its optimistic (lower-CDF) edge.
    """
    for i, (label, env) in enumerate(envelopes.items()):
        st = _style_for(label, i)
        ax.fill_between(
            env.t_grid,
            env.lower,
            env.upper,
            facecolor=st["facecolor"],
            edgecolor=st["color"],
            hatch=st["hatch"],
            alpha=st["alpha"],
            linewidth=0.8,
            label=label,
            zorder=2,
        )
        ax.plot(env.t_grid, env.lower, color=st["color"], linewidth=0.9, zorder=3)
        ax.plot(env.t_grid, env.upper, color=st["color"], linewidth=0.9, zorder=3)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylim(-0.02, 1.02)
    if xlim is not None:
        ax.set_xlim(*xlim)
    leg = ax.legend(loc=legend_loc, title=legend_title)
    leg.set_zorder(10)
    return ax


def auto_xlim(
    envelopes: Mapping[str, CDFEnvelope], pad: float = 0.01, mass: float = 2e-3
) -> tuple[float, float]:
    """Zoom to the support actually occupied by the plotted envelopes.

    Purely a display choice -- the envelopes themselves are always computed on
    the paper's fixed ``linspace(0, 1, 201)`` grid, and no curve is altered.
    ``mass`` trims the far tails (default: ignore the outer 0.2% of probability)
    so that panels zoom to the same kind of window the published figures use
    instead of stretching to catch a single stray sample.
    """
    lo, hi = 1.0, 0.0
    for env in envelopes.values():
        active = env.t_grid[(env.upper > mass) & (env.lower < 1.0 - mass)]
        if active.size:
            lo = min(lo, float(active.min()))
            hi = max(hi, float(active.max()))
    if lo >= hi:
        return (0.0, 1.0)
    return (max(0.0, lo - pad), min(1.0, hi + pad))


def plot_envelope_grid(
    panels: Sequence[tuple[str, Mapping[str, CDFEnvelope]]],
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (5.2, 3.6),
    suptitle: str | None = None,
    xlabel: str = "non-failure probability",
    zoom: bool = True,
) -> plt.Figure:
    """Lay out several CDF-envelope panels in a grid (paper Figs. 3, 6, 7, 10)."""
    n = len(panels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for idx, (title, envs) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        plot_cdf_envelopes(
            ax, envs, title, xlim=auto_xlim(envs) if zoom else (0.0, 1.0), xlabel=xlabel
        )
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# RQ4: expected reliability (Fig. 8)
# --------------------------------------------------------------------------- #
def plot_expected_reliability(
    ax: plt.Axes, envelopes: Mapping[str, ReliabilityEnvelope], title: str
) -> plt.Axes:
    """Paper Fig. 8a: ``E[R_L(n_F)]`` envelopes on a logarithmic horizon axis."""
    for i, (label, env) in enumerate(envelopes.items()):
        st = _style_for(label, i)
        ax.fill_between(
            env.horizons, env.lower, env.upper, color=st["color"], alpha=0.30, linewidth=0
        )
        mid = 0.5 * (env.lower + env.upper)
        ax.plot(env.horizons, mid, color=st["color"], marker="s", markersize=3.0, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"$n_F$ (number of future operations)")
    ax.set_ylabel(r"$\mathbb{E}[R_L(n_F)]$ (expected reliability)")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 8, 10, 20, 40, 60])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(loc="upper right")
    return ax


def plot_envelope_width(
    ax: plt.Axes, envelopes: Mapping[str, ReliabilityEnvelope], title: str
) -> plt.Axes:
    """Paper Fig. 8b: envelope width (upper - lower) versus horizon."""
    for i, (label, env) in enumerate(envelopes.items()):
        st = _style_for(label, i)
        ax.plot(env.horizons, env.width, color=st["color"], marker="o", markersize=3.0, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"$n_F$ (number of future operations)")
    ax.set_ylabel("Envelope width (upper - lower)")
    ax.set_title(title)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 8, 10, 20, 40, 60])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(loc="upper right", title="Model")
    return ax


# --------------------------------------------------------------------------- #
# RQ7 / RQ8
# --------------------------------------------------------------------------- #
def plot_memory_growth(
    ax: plt.Axes, question_index: np.ndarray, retained_bytes: np.ndarray, title: str = "Memory used"
) -> plt.Axes:
    """Paper Fig. 10a: retained conversational context versus task number."""
    ax.plot(question_index, retained_bytes, color="#1f77b4")
    ax.set_xlabel("Question #")
    ax.set_ylabel("Bytes (approx)")
    ax.set_title(title)
    return ax


def plot_scalability(
    ax: plt.Axes,
    x: np.ndarray,
    times: np.ndarray,
    ci_low: np.ndarray | None,
    ci_high: np.ndarray | None,
    exponent: float,
    coefficient: float,
    xlabel: str,
    title: str,
) -> plt.Axes:
    """One panel of paper Fig. 11: measurements plus the fitted power law."""
    ax.plot(x, times, marker="o", color="#1f77b4", label="Measured")
    if ci_low is not None and ci_high is not None:
        ax.fill_between(x, ci_low, ci_high, color="#1f77b4", alpha=0.20, linewidth=0)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ax.plot(
        xs,
        coefficient * xs**exponent,
        linestyle="--",
        color="#ff7f0e",
        label=rf"Fit: $t \propto x^{{{exponent:.2f}}}$",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(title)
    ax.legend(title="Legend")
    return ax


def plot_timing_breakdown(ax: plt.Axes, stages: Mapping[str, float], title: str) -> plt.Axes:
    """Paper Fig. 11f: per-stage runtime with percentage annotations."""
    names = list(stages)
    values = np.array([stages[n] for n in names], dtype=float)
    total = values.sum()
    bars = ax.bar(names, values, color="#1f77b4", label="Stage time")
    for bar, v in zip(bars, values):
        pct = 100.0 * v / total if total > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.2f}s\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Pipeline stage")
    ax.set_title(title)
    ax.set_ylim(0, max(values.max() * 1.18, 1e-6))
    ax.legend(title="Legend")
    return ax


# --------------------------------------------------------------------------- #
# Figs. 1 and 2 -- redrawn from the model graph
# --------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, *, shape="rect", fc="#eaf2fb", ec="#1f4e79", fontsize=8):
    import matplotlib.patches as mpatches

    x, y = xy
    if shape == "oval":
        patch = mpatches.Ellipse((x + w / 2, y + h / 2), w, h, facecolor=fc, edgecolor=ec, lw=1.2)
    else:
        patch = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012", facecolor=fc, edgecolor=ec, lw=1.2
        )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=5)
    return (x + w / 2, y + h / 2)


def _arrow(ax, src, dst, color="#444444", style="-|>"):
    import matplotlib.patches as mpatches

    ax.add_patch(
        mpatches.FancyArrowPatch(
            src, dst, arrowstyle=style, mutation_scale=9, color=color, lw=1.0, shrinkA=3, shrinkB=3
        )
    )


def draw_hierarchy_concept(
    n_models: int = 3, domains: Sequence[str] = ("Domain 1", "Domain 2"),
    subdomains: Sequence[Sequence[str]] = (("Sub 11", "Sub 12"), ("Sub 21", "Sub 22")),
) -> plt.Figure:
    """Paper Fig. 1, redrawn programmatically from the model graph.

    Encodes exactly: multiple LLM instances; *independent* domains drawn as
    rectangles; *dependent* subdomains drawn as ovals; observed ``(C, N)`` at the
    leaves; OP-weighted aggregation on every edge.
    """
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.set_title(
        "Fig. 1 (redrawn): hierarchy across LLM instances, domains, subdomains and tasks\n"
        "rectangle = independent component,   oval = dependent component",
        fontsize=10,
        fontweight="bold",
    )

    for k in range(n_models):
        x0 = 0.25 + k * 3.25
        label = f"LLM$^{{({k + 1})}}$" if k < n_models - 1 else "LLM$^{(M)}$"
        top = _box(ax, (x0 + 0.75, 4.55), 1.6, 0.6, f"{label}\n$p_L=\\sum_i W_i p_i$", fc="#fdf1dc")
        if k > 0:
            continue  # detail only the first instance, as in the paper
        for i, dom in enumerate(domains):
            dx = x0 + i * 1.55
            dcentre = _box(
                ax, (dx, 3.30), 1.35, 0.62, f"{dom}\n$p_i=\\sum_j \\Omega_{{ij}}\\theta_{{ij}}$",
                fc="#e8f4ea",
            )
            _arrow(ax, dcentre, (top[0], 4.55), color="#1f4e79")
            ax.text(
                (dcentre[0] + top[0]) / 2 + 0.05,
                3.98,
                f"$W_{i + 1}$",
                fontsize=8,
                color="#1f4e79",
            )
            for j, sub in enumerate(subdomains[i]):
                sx = dx - 0.10 + j * 0.78
                scentre = _box(
                    ax, (sx, 2.05), 0.72, 0.55, f"{sub}\n$\\theta_{{{i + 1}{j + 1}}}$",
                    shape="oval", fc="#e6e2f5",
                )
                _arrow(ax, scentre, (dcentre[0], 3.30), color="#2e7d32")
                _box(
                    ax, (sx, 1.10), 0.72, 0.45,
                    f"$(C_{{{i + 1}{j + 1}}}, N_{{{i + 1}{j + 1}}})$",
                    fc="#ffffff", fontsize=7,
                )
                _arrow(ax, (scentre[0], 1.55), (scentre[0], 2.05), color="#777777")
            # dependence link between subdomains inside a domain
            _arrow(
                ax,
                (dx + 0.62, 2.33),
                (dx + 0.68, 2.33),
                color="#b71c1c",
                style="<|-|>",
            )
    ax.text(
        0.25, 0.55,
        "Domains are statistically independent.  Subdomains inside a domain are dependent through the\n"
        "shared latent pair $(\\mu_i,\\nu_i)$; conditional on $(\\mu_i,\\nu_i)$ they are independent.",
        fontsize=8.5, va="center",
    )
    fig.tight_layout()
    return fig


def draw_hierarchy_detail() -> plt.Figure:
    """Paper Fig. 2, redrawn: the detailed probabilistic structure for one domain."""
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title(
        "Fig. 2 (redrawn): detailed hierarchical probabilistic structure\n"
        "imprecise hyper-hyper-parameters -> hyperpriors -> shared latents -> subdomain reliabilities -> data",
        fontsize=10,
        fontweight="bold",
    )

    h = _box(ax, (3.4, 5.35), 3.3, 0.62,
             r"$h_i=(a_i,b_i,c_i,d_i)\in\mathcal{H}_i$" "\n" r"(intervals: imprecise prior)",
             fc="#fde9e9", ec="#b71c1c")
    mu = _box(ax, (2.15, 4.15), 1.9, 0.58, r"$\mu_i\sim\mathrm{Beta}(a_i,b_i)$", shape="oval", fc="#fdf1dc")
    nu = _box(ax, (5.95, 4.15), 1.9, 0.58, r"$\nu_i\sim\mathrm{Gamma}(c_i,\mathrm{rate}{=}d_i)$",
              shape="oval", fc="#fdf1dc")
    _arrow(ax, (h[0] - 0.7, 5.35), mu, color="#b71c1c")
    _arrow(ax, (h[0] + 0.7, 5.35), nu, color="#b71c1c")

    shared = _box(ax, (3.55, 3.05), 3.0, 0.55,
                  r"shared latents $(\mu_i,\nu_i)$ $\Rightarrow$ partial pooling",
                  fc="#e6e2f5", ec="#4a148c")
    _arrow(ax, mu, shared)
    _arrow(ax, nu, shared)

    centres = []
    for j, name in enumerate(("$S_{i1}$", "$S_{i2}$", r"$\cdots$", "$S_{in_i}$")):
        sx = 0.65 + j * 2.3
        c = _box(ax, (sx, 1.85), 1.55, 0.60,
                 f"{name}\n" r"$\theta_{ij}\mid\mu_i,\nu_i\sim\mathrm{Beta}(\mu_i\nu_i,(1-\mu_i)\nu_i)$",
                 shape="oval", fc="#e8f4ea", fontsize=7)
        _arrow(ax, shared, c, color="#4a148c")
        _box(ax, (sx + 0.15, 0.85), 1.25, 0.45,
             r"$C_{ij}\sim\mathrm{Bin}(N_{ij},\theta_{ij})$", fc="#ffffff", fontsize=7)
        _arrow(ax, (c[0], 1.30), (c[0], 1.85), color="#777777")
        centres.append(c)

    ax.annotate(
        "", xy=(centres[1][0] - 0.55, 2.55), xytext=(centres[0][0] + 0.55, 2.55),
        arrowprops=dict(arrowstyle="<|-|>", color="#b71c1c", lw=1.0),
    )
    ax.text(
        (centres[0][0] + centres[1][0]) / 2, 2.68,
        "dependent after marginalising $(\\mu_i,\\nu_i)$",
        ha="center", fontsize=7.5, color="#b71c1c",
    )
    ax.text(
        0.35, 0.30,
        r"Aggregation: $p_i=\sum_j\Omega_{ij}\theta_{ij}$,   $p_L=\sum_i W_i p_i$,   "
        r"$R(n_F)=p^{\,n_F}$.   Domains $D_1,\dots,D_m$ are independent.",
        fontsize=8.5,
    )
    fig.tight_layout()
    return fig
