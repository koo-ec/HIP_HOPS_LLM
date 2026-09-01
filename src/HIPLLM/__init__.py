"""High-level public API for HIP-LLM."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .failure_probability import FailureProb, LogprobsUnavailableError
from .operational_failure import (
    OperationalFailureProb,
    OperationalFailureResult,
    paper_inference_settings,
    quick_inference_settings,
)
from .results import FailureProbResult
from .strategyqa import (
    StrategyQALoadError,
    decomposition_stratum,
    load_strategyqa,
    parse_strategyqa_answer,
)

try:
    __version__ = version("HIPLLM")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "unknown"

__all__ = [
    "FailureProb",
    "FailureProbResult",
    "LogprobsUnavailableError",
    "OperationalFailureProb",
    "OperationalFailureResult",
    "StrategyQALoadError",
    "decomposition_stratum",
    "load_strategyqa",
    "paper_inference_settings",
    "parse_strategyqa_answer",
    "quick_inference_settings",
    "__version__",
]
