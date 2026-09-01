"""HIP-LLM replication package.

A transparent, tested re-implementation of

    R. Aghazadeh-Chakherlou, Q. Guo, S. Khastgir, P. Popov, X. Zhang, X. Zhao,
    "A hierarchical imprecise probability approach to reliability assessment of
    large language models", Reliability Engineering & System Safety 272 (2026)
    112615.

Nothing in this package hard-codes a published result.  Every posterior,
envelope and figure is computed from either (a) the authors' published
measurements, (b) real provider API calls, or (c) synthetic draws in the
experiments the paper itself defines as synthetic.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("HIPLLM")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "unknown"

PAPER_DOI = "10.1016/j.ress.2026.112615"
OFFICIAL_REPOSITORY = "https://github.com/aghazadehchakherlou-web/llm-imprecise-bayes"

from .schemas import (  # noqa: E402
    BenchmarkResult,
    CDFEnvelope,
    DomainData,
    GlobalSettings,
    HyperparameterConfiguration,
    HyperparameterInterval,
    HyperposteriorGrid,
    ModelResult,
    OperationalProfile,
    PosteriorSamples,
    ReliabilityEnvelope,
    ReproductionRecord,
    ReproductionStatus,
    RunMode,
    SourceRecord,
    SubdomainData,
)

__all__ = [
    "__version__",
    "PAPER_DOI",
    "OFFICIAL_REPOSITORY",
    "BenchmarkResult",
    "CDFEnvelope",
    "DomainData",
    "GlobalSettings",
    "HyperparameterConfiguration",
    "HyperparameterInterval",
    "HyperposteriorGrid",
    "ModelResult",
    "OperationalProfile",
    "PosteriorSamples",
    "ReliabilityEnvelope",
    "ReproductionRecord",
    "ReproductionStatus",
    "RunMode",
    "SourceRecord",
    "SubdomainData",
]
