"""PEP 8 alias for :mod:`HIP_HOPS_LLM`.

The canonical import name matches the repository and the distribution
(``HIP_HOPS_LLM``), which is what appears throughout the documentation.  Some
projects prefer lower-case module names, so this alias exists::

    import hip_hops_llm as hh
    study = hh.AgenticReliabilityStudy(graph)

It is the same module object, not a copy: ``hip_hops_llm.X is HIP_HOPS_LLM.X``.
"""

from __future__ import annotations

import sys as _sys

import HIP_HOPS_LLM as _pkg
from HIP_HOPS_LLM import *  # noqa: F401,F403
from HIP_HOPS_LLM import __version__  # noqa: F401

__all__ = list(_pkg.__all__)

# Make submodules reachable as hip_hops_llm.bayes, hip_hops_llm.faulttree, ...
for _name in ("architecture", "bayes", "faulttree", "io", "reliability", "viz"):
    _sys.modules[f"{__name__}.{_name}"] = getattr(_pkg, _name, None) or __import__(
        f"HIP_HOPS_LLM.{_name}", fromlist=["_"]
    )
