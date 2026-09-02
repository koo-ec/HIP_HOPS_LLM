"""Alias so ``import HIP_HOPS_LLM`` resolves to :mod:`hiphopsllm`.

The canonical import name is ``hiphopsllm``. The repository and the distribution
are called ``HIP-HOPS-LLM``, so this makes the obvious guess work — and makes it
the *same module object*, not a second copy::

    import HIP_HOPS_LLM, hiphopsllm
    HIP_HOPS_LLM.AgenticReliabilityStudy is hiphopsllm.AgenticReliabilityStudy

Submodules resolve through the alias too, so ``HIP_HOPS_LLM.bayes.network``
works.

Only this one alias ships. A second, lower-case ``hip_hops_llm`` would be a
different file on Linux and the *same* file on Windows and macOS, whose
filesystems are case-insensitive — so the pair cannot be checked out reliably.
"""

from __future__ import annotations

import sys as _sys

import hiphopsllm as _pkg
from hiphopsllm import *  # noqa: F401,F403
from hiphopsllm import __version__  # noqa: F401

__all__ = list(_pkg.__all__)

for _dotted in (
    "architecture", "bayes", "faulttree", "io", "reliability", "viz",
    "architecture.model", "architecture.acyclic", "architecture.extract",
    "bayes.cpt", "bayes.network", "bayes.learn", "bayes.viz",
    "faulttree.failure", "faulttree.synthesis", "faulttree.analysis",
    "faulttree.export",
    "reliability.profile", "reliability.calibration", "reliability.hipllm",
    "io.examples", "viz.plots", "pipeline", "report",
):
    try:
        _sys.modules[f"{__name__}.{_dotted}"] = __import__(
            f"hiphopsllm.{_dotted}", fromlist=["_"]
        )
    except ImportError:  # pragma: no cover - optional dependency paths
        continue
del _dotted
