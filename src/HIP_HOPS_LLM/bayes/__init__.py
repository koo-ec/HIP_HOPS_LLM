"""Fault trees as Bayesian networks: CPTs, exact inference, diagnosis, drawing.

A fault tree and a Bayesian network describe the same object.  Building the
second *from* the first --- instead of authoring both --- buys three things:

* an **exact** top-event probability, where cut sets give only an upper bound;
* **diagnosis**: condition on what a run showed and read the posterior over
  causes;
* **learned gates**: where per-node outcomes were logged, a gate's table can be
  fitted from data rather than assumed to be AND or OR.

Start with :func:`fault_tree_to_cpts` if you want the tables, and
:class:`BayesianNetwork` if you want answers::

    from HIP_HOPS_LLM.bayes import BayesianNetwork

    bn = BayesianNetwork.from_fault_tree(report.tree("H2"), report.failure_model)
    bn.p_fail()                                   # exact P(top event)
    bn.posteriors({"BE-coder-EXECERR": "Fail"})   # posterior over causes
    bn.show()                                     # pyAgrum, or matplotlib

State indices are ``0 = OK`` and ``1 = Fail`` throughout.
"""

from __future__ import annotations

from .cpt import (
    FAIL,
    LABELS,
    MAX_GATE_INPUTS,
    OK,
    CPT,
    CPTBuilder,
    CPTSet,
    GateType,
    deterministic_gate_cpt,
    fault_tree_to_cpts,
    k_of_n_cpt,
    noisy_or_cpt,
    prior_cpt,
)
from .learn import (
    CPTLearningError,
    LearnedCPT,
    fit_cpts,
    learn_cpt,
    learn_gate,
)
from .network import (
    BayesianNetwork,
    Envelope,
    ImpreciseBayesianNetwork,
    PyAgrumUnavailable,
    compare_with_cutsets,
    exact_top_probability,
    fault_tree_to_bayesnet,
)
from .viz import BayesNetView, graphviz_available

__all__ = [
    # states
    "OK",
    "FAIL",
    "LABELS",
    "MAX_GATE_INPUTS",
    # conditional probability tables
    "CPT",
    "CPTBuilder",
    "CPTSet",
    "GateType",
    "deterministic_gate_cpt",
    "fault_tree_to_cpts",
    "k_of_n_cpt",
    "noisy_or_cpt",
    "prior_cpt",
    # learning
    "CPTLearningError",
    "LearnedCPT",
    "fit_cpts",
    "learn_cpt",
    "learn_gate",
    # networks
    "BayesianNetwork",
    "Envelope",
    "ImpreciseBayesianNetwork",
    "PyAgrumUnavailable",
    "compare_with_cutsets",
    "exact_top_probability",
    "fault_tree_to_bayesnet",
    # drawing
    "BayesNetView",
    "graphviz_available",
]
