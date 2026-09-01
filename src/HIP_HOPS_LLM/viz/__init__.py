"""Matplotlib rendering: fault trees, architectures, importance and cut sets.

Mermaid is the natural format for these diagrams, but a mermaid block needs a
renderer that Kaggle and offline Colab sessions do not have.  These functions
draw the same objects with matplotlib alone, so a notebook always produces a
picture.  Bayesian networks are drawn separately by
:mod:`HIP_HOPS_LLM.bayes.viz`, which prefers pyAgrum when Graphviz is present.
"""

from __future__ import annotations

from .plots import (
    TOKENS,
    plot_architecture,
    plot_cutset_orders,
    plot_fault_tree,
    plot_importance,
)

__all__ = [
    "TOKENS",
    "plot_architecture",
    "plot_cutset_orders",
    "plot_fault_tree",
    "plot_importance",
]
