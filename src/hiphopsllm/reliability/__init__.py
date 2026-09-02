"""Reliability: operational profiles, measured evidence, and HIP-LLM.

Three things live here.

:mod:`.profile` --- the operational profile, the mix of work the system will
actually meet.  Every estimate downstream is conditional on it.

:mod:`.hipllm` --- the whole of HIP-LLM, re-exported.  ``OperationalFailureProb``
turns observed correctness outcomes plus a profile into a hierarchical imprecise
posterior for the probability that the *next* item fails.

:mod:`.calibration` --- the bridge.  It runs that inference per component and
writes the resulting intervals onto the fault tree's basic events, replacing
engineering-judgement placeholders with measurement, and recording exactly what
it did and did not touch.
"""

from __future__ import annotations

from .calibration import (
    CalibrationReport,
    ComponentEvidence,
    EvidenceCalibrator,
    calibrate_failure_model,
    distribute_union,
)
from .hipllm import (  # noqa: F401
    FailureProb,
    FailureProbResult,
    GlobalSettings,
    HIPLLMOperationalProfile,
    HyperparameterInterval,
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
from .profile import (
    OperationalProfile,
    dataset_proportional_profile,
    empirical_profile,
    stratify,
    uniform_profile,
)

__all__ = [
    # profiles
    "OperationalProfile",
    "dataset_proportional_profile",
    "empirical_profile",
    "stratify",
    "uniform_profile",
    # calibration
    "CalibrationReport",
    "ComponentEvidence",
    "EvidenceCalibrator",
    "calibrate_failure_model",
    "distribute_union",
    # HIP-LLM
    "FailureProb",
    "FailureProbResult",
    "GlobalSettings",
    "HIPLLMOperationalProfile",
    "HyperparameterInterval",
    "LogprobsUnavailableError",
    "OperationalFailureProb",
    "OperationalFailureResult",
    "StrategyQALoadError",
    "decomposition_stratum",
    "load_strategyqa",
    "paper_inference_settings",
    "parse_strategyqa_answer",
    "quick_inference_settings",
]
