"""Every HIP-LLM function, re-exported under this package's namespace.

HIP-LLM ships in two layers and both are vendored here in full (see
``docs/source/vendoring.md`` for the provenance and the commit they were taken
from):

``HIPLLM``
    The high-level API --- :class:`FailureProb`, :class:`OperationalFailureProb`,
    the StrategyQA loader --- which is what most users want.

``hip_llm``
    The replication engine underneath it: hyperposteriors, imprecise envelopes,
    reliability transforms, baselines, benchmark evaluation, plotting, and the
    schemas the two layers share.

Both are importable directly (``from HIPLLM import OperationalFailureProb`` keeps
working exactly as before).  This module makes them reachable through one import
as well, so an analysis that spans the fault tree and the reliability model does
not need three import lines::

    from hiphopsllm import OperationalFailureProb, quick_inference_settings

One name is deliberately *not* passed through unchanged.  HIP-LLM's
``OperationalProfile`` is a schema with parallel ``labels``/``weights`` arrays;
this package's :class:`hiphopsllm.reliability.profile.OperationalProfile` is a
mapping-shaped class that converts to it.  The HIP-LLM one is available here as
:data:`HIPLLMOperationalProfile`.
"""

from __future__ import annotations

# -- the high-level API ------------------------------------------------------ #
from HIPLLM import (
    FailureProb,
    FailureProbResult,
    LogprobsUnavailableError,
    OperationalFailureProb,
    OperationalFailureResult,
    StrategyQALoadError,
    decomposition_stratum,
    load_strategyqa,
    paper_inference_settings,
    parse_strategyqa_answer,
    quick_inference_settings,
)
from HIPLLM import __version__ as HIPLLM_VERSION

# -- the replication engine -------------------------------------------------- #
from hip_llm import (
    OFFICIAL_REPOSITORY,
    PAPER_DOI,
    BenchmarkResult,
    CDFEnvelope,
    DomainData,
    GlobalSettings,
    HyperparameterConfiguration,
    HyperparameterInterval,
    HyperposteriorGrid,
    ModelResult,
    PosteriorSamples,
    ReliabilityEnvelope,
    ReproductionRecord,
    ReproductionStatus,
    RunMode,
    SourceRecord,
    SubdomainData,
)
from hip_llm.schemas import OperationalProfile as HIPLLMOperationalProfile

# The engine's modules, so ``hiphopsllm.reliability.hipllm.envelopes`` works.
from hip_llm import (  # noqa: F401
    api_clients,
    baselines,
    benchmark_eval,
    envelopes,
    grids,
    hyperposterior,
    numerics,
    operational_profile,
    plotting,
    posterior,
    reliability,
    scalability,
    schemas,
    validation,
)

__all__ = [
    # high-level API
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
    "HIPLLM_VERSION",
    # schemas
    "BenchmarkResult",
    "CDFEnvelope",
    "DomainData",
    "GlobalSettings",
    "HIPLLMOperationalProfile",
    "HyperparameterConfiguration",
    "HyperparameterInterval",
    "HyperposteriorGrid",
    "ModelResult",
    "PosteriorSamples",
    "ReliabilityEnvelope",
    "ReproductionRecord",
    "ReproductionStatus",
    "RunMode",
    "SourceRecord",
    "SubdomainData",
    # provenance
    "OFFICIAL_REPOSITORY",
    "PAPER_DOI",
    # engine modules
    "api_clients",
    "baselines",
    "benchmark_eval",
    "envelopes",
    "grids",
    "hyperposterior",
    "numerics",
    "operational_profile",
    "plotting",
    "posterior",
    "reliability",
    "scalability",
    "schemas",
    "validation",
]
