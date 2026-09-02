"""Fault trees: local failure logic, compositional synthesis, and analysis.

Three HiP-HOPS phases live here.

**Annotate** (:mod:`.failure`) --- every component is given an IF-FMEA table:
each output deviation is a Boolean expression over the component's input
deviations and its own internal basic events.  The library covers the agentic
archetypes: LLM agent, tool/executor, router, aggregator, transform, boundary
and feedback cut.

**Synthesise** (:mod:`.synthesis`) --- a tree is produced by traversing
connections *backwards* from a system-level hazard and substituting each
component's local logic, memoised so shared sub-trees stay shared (HiP-HOPS
transfer gates).  Trees are never drawn by hand.

**Analyse** (:mod:`.analysis`) --- minimal cut sets by MOCUS with absorption,
quantification by the minimal cut upper bound, Birnbaum and Fussell-Vesely
importance, single points of failure, and a generated FMEA table.

The deviation notation is HiP-HOPS' own, ``class-component.port``:
``O-coder.out``, ``VS-aggregator.out``.  The six failure classes are in
:class:`FClass`; keeping ``VC`` (coarse, detectable) apart from ``VS`` (subtle,
plausible --- the hallucination case) is the single most important modelling
decision in the library, because they propagate identically but one is caught at
the system boundary and the other is delivered to the user.
"""

from __future__ import annotations

from .analysis import (
    CutSetResult,
    FMEARow,
    ImportanceRow,
    Quantification,
    TreeAnalysis,
    analyse_tree,
    cut_sets,
    fmea_table,
    importance,
    quantify,
    single_points_of_failure,
)
from .export import (
    markdown_report,
    to_dot,
    to_json,
    to_mermaid,
    to_openpsa_xml,
)
from .failure import (
    AND,
    OR,
    BasicEvent,
    ComponentFailureLogic,
    Deviation,
    Expr,
    FailureModel,
    FClass,
    annotate_system,
    entropy_to_fail_prob,
)
from .synthesis import (
    FaultTree,
    FTNode,
    Hazard,
    default_hazards,
    expand_to_tree,
    simplify_tree,
    synthesise_all,
    synthesise_fault_tree,
)

__all__ = [
    # failure logic
    "AND",
    "OR",
    "BasicEvent",
    "ComponentFailureLogic",
    "Deviation",
    "Expr",
    "FClass",
    "FailureModel",
    "annotate_system",
    "entropy_to_fail_prob",
    # synthesis
    "FTNode",
    "FaultTree",
    "Hazard",
    "default_hazards",
    "expand_to_tree",
    "simplify_tree",
    "synthesise_all",
    "synthesise_fault_tree",
    # analysis
    "CutSetResult",
    "FMEARow",
    "ImportanceRow",
    "Quantification",
    "TreeAnalysis",
    "analyse_tree",
    "cut_sets",
    "fmea_table",
    "importance",
    "quantify",
    "single_points_of_failure",
    # export
    "markdown_report",
    "to_dot",
    "to_json",
    "to_mermaid",
    "to_openpsa_xml",
]
