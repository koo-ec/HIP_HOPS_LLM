"""HIP-HOPS-LLM — hierarchical imprecise reliability for agentic AI systems.

Two established methods, joined at the place where each is weakest.

**HiP-HOPS** (Hierarchically Performed Hazard Origin and Propagation Studies)
gives compositional structure: annotate each component with local failure logic,
then synthesise system-level fault trees, minimal cut sets and an FMEA by
traversing the architecture.  Its basic-event probabilities, though, arrive as
engineering judgement.

**HIP-LLM** gives the numbers: a hierarchical imprecise-Bayesian posterior for
the probability that a language model fails on the *next* item drawn from an
explicit operational profile.  It returns an interval rather than a point,
because a few hundred benchmark items do not identify one.  But it describes a
single model, not a workflow of several agents, tools and routers.

This package reads a LangGraph application, synthesises its fault trees,
measures each component under an operational profile, writes those intervals onto
the trees' leaves, and converts the result into a Bayesian network for exact
inference, diagnosis and drawing.

The whole pipeline, from graph to picture::

    from HIP_HOPS_LLM import AgenticReliabilityStudy, load_example, load_outcomes

    study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
    study.observe(load_outcomes(), profile={"short": 0.3, "medium": 0.5, "long": 0.2})
    study.run()
    print(study.summary())
    study.bayesnet("H2").show()

HIP-LLM's own API is re-exported unchanged, so its usage is one import line::

    from HIP_HOPS_LLM import OperationalFailureProb, quick_inference_settings

Layers, if you want them separately:

============================================  ==========================================
:mod:`HIP_HOPS_LLM.architecture`              LangGraph → components, ports, connections
:mod:`HIP_HOPS_LLM.faulttree`                 failure logic → fault trees → cut sets, FMEA
:mod:`HIP_HOPS_LLM.reliability`               operational profiles, HIP-LLM, calibration
:mod:`HIP_HOPS_LLM.bayes`                     fault tree → CPTs → Bayesian network
:mod:`HIP_HOPS_LLM.viz`                       matplotlib rendering
============================================  ==========================================
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
from .architecture import (
    Component,
    Connection,
    CycleReport,
    LangGraphExtractor,
    Role,
    SystemModel,
    extract_architecture,
    find_cycles,
    is_acyclic,
    make_acyclic,
    parse_mermaid,
)

# --------------------------------------------------------------------------- #
# Fault trees
# --------------------------------------------------------------------------- #
from .faulttree import (
    AND,
    OR,
    BasicEvent,
    ComponentFailureLogic,
    CutSetResult,
    Deviation,
    FailureModel,
    FaultTree,
    FClass,
    FMEARow,
    FTNode,
    Hazard,
    Quantification,
    TreeAnalysis,
    analyse_tree,
    annotate_system,
    cut_sets,
    default_hazards,
    entropy_to_fail_prob,
    fmea_table,
    importance,
    markdown_report,
    quantify,
    simplify_tree,
    single_points_of_failure,
    synthesise_all,
    synthesise_fault_tree,
    to_dot,
    to_json,
    to_mermaid,
    to_openpsa_xml,
)

# --------------------------------------------------------------------------- #
# Reliability: operational profiles, HIP-LLM, calibration
# --------------------------------------------------------------------------- #
from .reliability import (
    CalibrationReport,
    ComponentEvidence,
    EvidenceCalibrator,
    FailureProb,
    FailureProbResult,
    GlobalSettings,
    HyperparameterInterval,
    LogprobsUnavailableError,
    OperationalFailureProb,
    OperationalFailureResult,
    OperationalProfile,
    StrategyQALoadError,
    calibrate_failure_model,
    decomposition_stratum,
    distribute_union,
    empirical_profile,
    load_strategyqa,
    paper_inference_settings,
    parse_strategyqa_answer,
    quick_inference_settings,
    stratify,
    uniform_profile,
)

# --------------------------------------------------------------------------- #
# Bayesian networks
# --------------------------------------------------------------------------- #
from .bayes import (
    CPT,
    BayesianNetwork,
    BayesNetView,
    CPTBuilder,
    CPTSet,
    Envelope,
    GateType,
    ImpreciseBayesianNetwork,
    LearnedCPT,
    compare_with_cutsets,
    deterministic_gate_cpt,
    exact_top_probability,
    fault_tree_to_bayesnet,
    fault_tree_to_cpts,
    fit_cpts,
    graphviz_available,
    k_of_n_cpt,
    learn_cpt,
    learn_gate,
    noisy_or_cpt,
)

# --------------------------------------------------------------------------- #
# Rendering, examples and the top-level pipeline
# --------------------------------------------------------------------------- #
from .io import EXAMPLES, describe_examples, load_example, load_outcomes
from .pipeline import AgenticReliabilityStudy, StudyNotReady
from .report import SafetyReport, analyse_langgraph, display_fault_tree, map_uncertainty
from .viz import (
    plot_architecture,
    plot_cutset_orders,
    plot_fault_tree,
    plot_importance,
)

try:
    __version__ = version("HIP-HOPS-LLM")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "0.1.0"

#: The HIP-LLM paper this package's reliability layer implements.
HIP_LLM_DOI = "10.1016/j.ress.2026.112615"

__all__ = [
    "__version__",
    "HIP_LLM_DOI",
    # top-level pipeline
    "AgenticReliabilityStudy",
    "StudyNotReady",
    "SafetyReport",
    "analyse_langgraph",
    "display_fault_tree",
    "map_uncertainty",
    # examples
    "EXAMPLES",
    "describe_examples",
    "load_example",
    "load_outcomes",
    # architecture
    "Component",
    "Connection",
    "CycleReport",
    "LangGraphExtractor",
    "Role",
    "SystemModel",
    "extract_architecture",
    "find_cycles",
    "is_acyclic",
    "make_acyclic",
    "parse_mermaid",
    # fault trees
    "AND",
    "OR",
    "BasicEvent",
    "ComponentFailureLogic",
    "CutSetResult",
    "Deviation",
    "FClass",
    "FMEARow",
    "FTNode",
    "FailureModel",
    "FaultTree",
    "Hazard",
    "Quantification",
    "TreeAnalysis",
    "analyse_tree",
    "annotate_system",
    "cut_sets",
    "default_hazards",
    "entropy_to_fail_prob",
    "fmea_table",
    "importance",
    "markdown_report",
    "quantify",
    "simplify_tree",
    "single_points_of_failure",
    "synthesise_all",
    "synthesise_fault_tree",
    "to_dot",
    "to_json",
    "to_mermaid",
    "to_openpsa_xml",
    # reliability
    "CalibrationReport",
    "ComponentEvidence",
    "EvidenceCalibrator",
    "FailureProb",
    "FailureProbResult",
    "GlobalSettings",
    "HyperparameterInterval",
    "LogprobsUnavailableError",
    "OperationalFailureProb",
    "OperationalFailureResult",
    "OperationalProfile",
    "StrategyQALoadError",
    "calibrate_failure_model",
    "decomposition_stratum",
    "distribute_union",
    "empirical_profile",
    "load_strategyqa",
    "paper_inference_settings",
    "parse_strategyqa_answer",
    "quick_inference_settings",
    "stratify",
    "uniform_profile",
    # Bayesian networks
    "CPT",
    "BayesNetView",
    "BayesianNetwork",
    "CPTBuilder",
    "CPTSet",
    "Envelope",
    "GateType",
    "ImpreciseBayesianNetwork",
    "LearnedCPT",
    "compare_with_cutsets",
    "deterministic_gate_cpt",
    "exact_top_probability",
    "fault_tree_to_bayesnet",
    "fault_tree_to_cpts",
    "fit_cpts",
    "graphviz_available",
    "k_of_n_cpt",
    "learn_cpt",
    "learn_gate",
    "noisy_or_cpt",
    # plotting
    "plot_architecture",
    "plot_cutset_orders",
    "plot_fault_tree",
    "plot_importance",
]
