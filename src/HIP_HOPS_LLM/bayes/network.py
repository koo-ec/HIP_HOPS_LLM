"""Bayesian networks over agentic fault trees.

:class:`BayesianNetwork` wraps a :class:`~HIP_HOPS_LLM.bayes.cpt.CPTSet` and does
four things a fault tree on its own cannot.

* **Exact top-event probability.**  The minimal cut upper bound over-estimates
  whenever cut sets share basic events.  Inference over the network is exact, so
  :meth:`compare_with_cutsets` brackets the answer instead of asserting one end
  of it.
* **Diagnosis.**  Condition on what a run actually showed --- the router took the
  error branch, the tool raised --- and read the posterior over causes.  That is
  the step run-time monitoring of an agentic system is reaching for.
* **Imprecision.**  Basic-event probabilities estimated from a few hundred
  benchmark items are intervals, not numbers.  A coherent fault tree's top-event
  probability is monotone in each of them, so evaluating at both ends gives a
  guaranteed envelope --- see :meth:`ImpreciseBayesianNetwork.envelope`.
* **Independence from pyAgrum.**  Every quantity is also computable by exact
  enumeration in NumPy alone, which is what :meth:`cross_check` uses to verify
  the pyAgrum result.  pyAgrum stays an optional dependency.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..faulttree.analysis import TreeAnalysis
from ..faulttree.failure import FailureModel
from ..faulttree.synthesis import FaultTree
from .cpt import FAIL, LABELS, OK, CPTSet, fault_tree_to_cpts

__all__ = [
    "BayesianNetwork",
    "ImpreciseBayesianNetwork",
    "Envelope",
    "fault_tree_to_bayesnet",
    "exact_top_probability",
    "compare_with_cutsets",
    "PyAgrumUnavailable",
]


class PyAgrumUnavailable(ImportError):
    """pyAgrum is not installed, and the requested operation needs it."""


def _require_pyagrum():
    try:
        import pyagrum as gum  # type: ignore

        return gum
    except ImportError:  # pragma: no cover - exercised only without pyagrum
        try:
            import pyAgrum as gum  # type: ignore  # noqa: N813

            return gum
        except ImportError as exc:
            raise PyAgrumUnavailable(
                "pyAgrum is required here. Install it with `pip install pyagrum`, "
                "or use the exact-enumeration path (BayesianNetwork.p_fail(engine="
                "'exact')), which needs only NumPy."
            ) from exc


def _state_index(value: Any) -> int:
    """Accept ``'Fail'``/``'OK'``, ``1``/``0``, ``True``/``False``."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("fail", "failed", "f", "1", "true", "yes"):
            return FAIL
        if v in ("ok", "o", "0", "false", "no", "working"):
            return OK
        raise ValueError(f"cannot read {value!r} as a state; use 'OK' or 'Fail'")
    return FAIL if bool(value) else OK


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
@dataclass
class BayesianNetwork:
    """A discrete two-state Bayesian network built from a fault tree.

    Construct it with :meth:`from_fault_tree`, or directly from a
    :class:`~HIP_HOPS_LLM.bayes.cpt.CPTSet` that was fitted from data.
    """

    cpts: CPTSet
    #: the fault tree it came from, when it came from one
    tree: Optional[FaultTree] = None
    name: str = ""
    _net: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.cpts.name

    # -- construction -------------------------------------------------------- #
    @classmethod
    def from_fault_tree(
        cls,
        tree: FaultTree,
        failure_model: Optional[FailureModel] = None,
        *,
        name: Optional[str] = None,
        bound: str = "point",
        soft_gates: bool = False,
        gate_overrides: Optional[Mapping[str, Any]] = None,
        **builder_kwargs: Any,
    ) -> "BayesianNetwork":
        """Convert a synthesised fault tree straight into a network."""
        cpts = fault_tree_to_cpts(
            tree,
            failure_model,
            name=name,
            bound=bound,
            soft_gates=soft_gates,
            gate_overrides=gate_overrides,
            **builder_kwargs,
        )
        return cls(cpts=cpts, tree=tree, name=cpts.name)

    @classmethod
    def imprecise_from_fault_tree(
        cls,
        tree: FaultTree,
        failure_model: Optional[FailureModel] = None,
        **kwargs: Any,
    ) -> "ImpreciseBayesianNetwork":
        """Build the lower and upper networks of an imprecise fault tree."""
        lower = cls.from_fault_tree(tree, failure_model, bound="lower", **kwargs)
        upper = cls.from_fault_tree(tree, failure_model, bound="upper", **kwargs)
        return ImpreciseBayesianNetwork(lower=lower, upper=upper, tree=tree)

    # -- names --------------------------------------------------------------- #
    @property
    def top(self) -> str:
        return self.cpts.top

    @property
    def variables(self) -> List[str]:
        return list(self.cpts.order)

    @property
    def basic_events(self) -> List[str]:
        """Basic event ids, i.e. the things that can be causes."""
        return sorted(self.cpts.event_variable)

    def resolve(self, key: str) -> str:
        return self.cpts.resolve(key)

    # -- pyAgrum ------------------------------------------------------------- #
    @property
    def net(self) -> Any:
        """The lazily built ``pyagrum.BayesNet``."""
        if self._net is None:
            self._net = self.to_pyagrum()
        return self._net

    def to_pyagrum(self) -> Any:
        """Materialise the CPT set as a ``pyagrum.BayesNet``."""
        gum = _require_pyagrum()
        net = gum.BayesNet(self.name or "faulttree")
        for var_name in self.cpts.order:
            cpt = self.cpts[var_name]
            variable = gum.LabelizedVariable(var_name, (cpt.label or var_name)[:120], 2)
            variable.changeLabel(OK, LABELS[OK])
            variable.changeLabel(FAIL, LABELS[FAIL])
            net.add(variable)
        for var_name in self.cpts.order:
            for parent in self.cpts[var_name].parents:
                net.addArc(parent, var_name)
        for var_name in self.cpts.order:
            cpt = self.cpts[var_name]
            if cpt.is_root:
                net.cpt(var_name)[:] = list(cpt.table)
                continue
            target = net.cpt(var_name)
            for combo in itertools.product((0, 1), repeat=len(cpt.parents)):
                assignment = dict(zip(cpt.parents, combo))
                target[assignment] = list(cpt.table[combo])
        return net

    # -- inference ----------------------------------------------------------- #
    def p_fail(
        self,
        target: Optional[str] = None,
        evidence: Optional[Mapping[str, Any]] = None,
        engine: str = "auto",
    ) -> float:
        """``P(target = Fail | evidence)``; ``target`` defaults to the top event.

        ``engine`` is ``"pyagrum"``, ``"exact"`` (NumPy enumeration) or
        ``"auto"``, which prefers pyAgrum and falls back to enumeration.
        """
        name = self.resolve(target) if target else self.top
        return float(self.posterior(name, evidence=evidence, engine=engine)[FAIL])

    def posterior(
        self,
        target: str,
        evidence: Optional[Mapping[str, Any]] = None,
        engine: str = "auto",
    ) -> np.ndarray:
        """Posterior ``[P(OK), P(Fail)]`` over one variable."""
        name = self.resolve(target)
        ev = self._resolve_evidence(evidence)
        if engine == "exact":
            return self._exact_posterior(name, ev)
        if engine == "pyagrum":
            return self._pyagrum_posterior(name, ev)
        if engine != "auto":
            raise ValueError("engine must be 'auto', 'pyagrum' or 'exact'")
        try:
            return self._pyagrum_posterior(name, ev)
        except PyAgrumUnavailable:
            return self._exact_posterior(name, ev)

    def posteriors(
        self,
        evidence: Optional[Mapping[str, Any]] = None,
        engine: str = "auto",
        basic_events_only: bool = True,
    ) -> Dict[str, float]:
        """``P(x = Fail | evidence)`` for every variable, ranked most likely first.

        With ``basic_events_only`` (the default) the result is a posterior over
        *causes* --- the diagnostic view.  Passing ``False`` returns every node,
        including the intermediate deviations, which is useful for locating where
        in the architecture a failure most likely entered.
        """
        ev = self._resolve_evidence(evidence)
        if basic_events_only:
            targets = {eid: var for eid, var in self.cpts.event_variable.items()}
        else:
            targets = {self._name_of(v): v for v in self.cpts.order}
        out: Dict[str, float] = {}
        for label, var in targets.items():
            out[label] = float(self.posterior(var, evidence=ev, engine=engine)[FAIL])
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    #: Older name kept because the source notebook used it.
    def evidence_posterior(
        self, evidence: Mapping[str, Any], engine: str = "auto"
    ) -> Dict[str, float]:
        """``P(each basic event = Fail | evidence)`` --- run-time diagnosis."""
        return self.posteriors(evidence=evidence, engine=engine)

    def most_probable_explanation(
        self, evidence: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, str]:
        """The single most probable joint assignment to the basic events.

        Exhaustive over basic events, so it is exact but exponential; it refuses
        above 22 basic events rather than running for hours.  For larger trees
        rank causes with :meth:`posteriors` instead.
        """
        ev = self._resolve_evidence(evidence)
        roots = self.cpts.roots
        if len(roots) > 22:
            raise ValueError(
                f"{len(roots)} basic events would need 2**{len(roots)} joint "
                "evaluations; use posteriors() to rank causes instead"
            )
        best_assignment: Optional[Tuple[int, ...]] = None
        best_p = -1.0
        for combo in itertools.product((0, 1), repeat=len(roots)):
            fixed = dict(zip(roots, combo))
            if any(fixed.get(k, v) != v for k, v in ev.items() if k in fixed):
                continue
            p = self._joint_marginal({**fixed, **ev})
            if p > best_p:
                best_p, best_assignment = p, combo
        assert best_assignment is not None
        return {
            self._name_of(v): LABELS[s] for v, s in zip(roots, best_assignment)
        }

    def _name_of(self, variable: str) -> str:
        """Report a variable by its basic event id where it has one.

        The fault tree node id of a leaf is ``BE::<event id>``; the event id is
        what appears in cut sets, in the FMEA and in evidence dictionaries, so it
        is the name a caller can act on.
        """
        for event_id, var in self.cpts.event_variable.items():
            if var == variable:
                return event_id
        return self.cpts.node_of.get(variable, variable)

    # -- engines ------------------------------------------------------------- #
    def _pyagrum_posterior(
        self, variable: str, evidence: Mapping[str, int]
    ) -> np.ndarray:
        gum = _require_pyagrum()
        engine = gum.LazyPropagation(self.net)
        if evidence:
            engine.setEvidence(dict(evidence))
        engine.makeInference()
        return np.asarray(engine.posterior(variable).toarray(), dtype=float).ravel()

    def _exact_posterior(
        self, variable: str, evidence: Mapping[str, int]
    ) -> np.ndarray:
        """Exact marginal by enumerating the roots only.

        Every non-root variable in a fault-tree network is a deterministic or
        noisy function of its parents, so the joint factorises and only the
        basic events need to be summed over --- that is what keeps this
        tractable enough to be a genuine cross-check rather than a toy.
        """
        roots = self.cpts.roots
        if len(roots) > 24:
            raise ValueError(
                f"exact enumeration over {len(roots)} basic events is not "
                "tractable; use engine='pyagrum'"
            )
        weights = np.zeros(2, dtype=float)
        total = 0.0
        for combo in itertools.product((0, 1), repeat=len(roots)):
            assignment = dict(zip(roots, combo))
            p = 1.0
            for var, state in zip(roots, combo):
                p *= float(self.cpts[var].table[state])
                if p == 0.0:
                    break
            if p == 0.0:
                continue
            marg = self._propagate(assignment)
            for var, state in evidence.items():
                p *= float(marg[var][state])
                if p == 0.0:
                    break
            if p == 0.0:
                continue
            total += p
            weights += p * marg[variable]
        if total <= 0.0:
            raise ValueError("the evidence has probability zero under this network")
        return weights / total

    def _propagate(self, root_states: Mapping[str, int]) -> Dict[str, np.ndarray]:
        """Distribution of every variable given a full assignment to the roots."""
        marg: Dict[str, np.ndarray] = {}
        for var in self.cpts.order:
            cpt = self.cpts[var]
            if cpt.is_root:
                state = root_states.get(var)
                if state is None:
                    marg[var] = np.asarray(cpt.table, dtype=float)
                else:
                    row = np.zeros(2)
                    row[state] = 1.0
                    marg[var] = row
                continue
            acc = np.zeros(2, dtype=float)
            for combo in itertools.product((0, 1), repeat=len(cpt.parents)):
                w = 1.0
                for parent, state in zip(cpt.parents, combo):
                    w *= float(marg[parent][state])
                    if w == 0.0:
                        break
                if w == 0.0:
                    continue
                acc += w * cpt.table[combo]
            marg[var] = acc
        return marg

    def _joint_marginal(self, states: Mapping[str, int]) -> float:
        marg = self._propagate(
            {v: s for v, s in states.items() if self.cpts[v].is_root}
        )
        p = 1.0
        for var, state in states.items():
            p *= float(marg[var][state])
        return p

    def _resolve_evidence(
        self, evidence: Optional[Mapping[str, Any]]
    ) -> Dict[str, int]:
        if not evidence:
            return {}
        out: Dict[str, int] = {}
        for key, value in evidence.items():
            out[self.resolve(key)] = _state_index(value)
        return out

    # -- verification -------------------------------------------------------- #
    def cross_check(
        self, rel_tol: float = 1e-9, abs_tol: float = 1e-12
    ) -> Dict[str, float]:
        """Compare the pyAgrum and exact-enumeration top-event probabilities.

        The two paths share no code, so agreement is real evidence that the
        conversion is right --- and cheap to obtain, before a number reaches a
        paper.

        The comparison is relative rather than absolute.  Exact enumeration sums
        ``2 ** (basic events)`` terms, so on a twenty-leaf tree its rounding error
        is around ``1e-12`` in absolute terms while still being correct to nine
        significant figures. An absolute threshold would flag that as a
        disagreement and teach the reader to ignore this check, which is worse
        than not having it.
        """
        exact = self.p_fail(engine="exact")
        try:
            agrum = self.p_fail(engine="pyagrum")
        except PyAgrumUnavailable:
            # The shape stays the same so callers never have to branch on it.
            # ``compared`` is 0 and ``agree`` is NaN, not 0: "the check did not
            # run" and "the two engines disagree" are different claims, and
            # reporting the second for the first would be a lie.
            return {
                "exact": exact,
                "pyagrum": float("nan"),
                "difference": float("nan"),
                "relative_difference": float("nan"),
                "agree": float("nan"),
                "compared": 0.0,
            }
        diff = abs(exact - agrum)
        scale = max(abs(exact), abs(agrum), 1e-300)
        return {
            "exact": exact,
            "pyagrum": agrum,
            "difference": diff,
            "relative_difference": diff / scale,
            "agree": float(diff <= max(abs_tol, rel_tol * scale)),
            "compared": 1.0,
        }

    def compare_with_cutsets(self, analysis: TreeAnalysis) -> Dict[str, float]:
        """Exact network probability against the cut-set bounds.

        The minimal cut upper bound is an *upper* bound on a coherent tree, so
        ``bound_overestimate`` should never be negative.  A negative value means
        the tree and the network have drifted apart.
        """
        exact = self.p_fail()
        mcub = float(analysis.quant.top_probability)
        return {
            "exact_bayesnet": exact,
            "minimal_cut_upper_bound": mcub,
            "rare_event_sum": float(analysis.quant.rare_event_sum),
            "bound_overestimate": mcub - exact,
        }

    # -- display ------------------------------------------------------------- #
    def view(self, **kwargs: Any):
        """A :class:`~HIP_HOPS_LLM.bayes.viz.BayesNetView` over this network."""
        from .viz import BayesNetView

        return BayesNetView(self, **kwargs)

    def show(self, **kwargs: Any):
        """Draw the network (and its inference) inline in a notebook."""
        return self.view(**kwargs).show()

    def to_frame(self, variable: Optional[str] = None):
        return self.cpts.to_frame(variable)

    def summary(self) -> str:
        lines = [
            f"Bayesian network — {self.name}",
            self.cpts.summary().split("\n", 1)[1],
            f"  P(top = Fail)  {self.p_fail():.6f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BayesianNetwork(name={self.name!r}, variables={len(self.cpts)}, "
            f"basic_events={len(self.cpts.roots)})"
        )


# --------------------------------------------------------------------------- #
# Imprecise pair
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Envelope:
    """A ``[lower, upper]`` probability interval with its own arithmetic."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.upper + 1e-12:
            raise ValueError(f"lower {self.lower} exceeds upper {self.upper}")

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    def contains(self, value: float) -> bool:
        return self.lower - 1e-12 <= value <= self.upper + 1e-12

    def as_tuple(self) -> Tuple[float, float]:
        return (self.lower, self.upper)

    def __str__(self) -> str:
        return f"[{self.lower:.6f}, {self.upper:.6f}]"


@dataclass
class ImpreciseBayesianNetwork:
    """Lower and upper networks of a fault tree with interval-valued events.

    A coherent (monotone) fault tree's top-event probability is non-decreasing in
    every basic-event probability.  Evaluating at the lower ends of all intervals
    therefore gives a genuine lower bound and the upper ends a genuine upper
    bound --- no optimisation over the interval box is needed, and no sampling.

    This is where HIP-LLM's imprecise posterior meets the fault tree: the
    interval on each basic event comes from
    :class:`~HIP_HOPS_LLM.reliability.calibration.EvidenceCalibrator`, which
    derives it from observed outcomes under an operational profile rather than
    from engineering judgement.
    """

    lower: BayesianNetwork
    upper: BayesianNetwork
    tree: Optional[FaultTree] = None

    @property
    def name(self) -> str:
        return self.lower.name

    def envelope(
        self,
        target: Optional[str] = None,
        evidence: Optional[Mapping[str, Any]] = None,
        engine: str = "auto",
    ) -> Envelope:
        """``[P_lower, P_upper]`` for a target, by default the top event."""
        lo = self.lower.p_fail(target, evidence=evidence, engine=engine)
        hi = self.upper.p_fail(target, evidence=evidence, engine=engine)
        return Envelope(min(lo, hi), max(lo, hi))

    def posterior_envelopes(
        self,
        evidence: Optional[Mapping[str, Any]] = None,
        engine: str = "auto",
    ) -> Dict[str, Envelope]:
        """Per-basic-event posterior envelopes, ranked by upper bound."""
        lo = self.lower.posteriors(evidence=evidence, engine=engine)
        hi = self.upper.posteriors(evidence=evidence, engine=engine)
        out = {
            k: Envelope(min(lo[k], hi[k]), max(lo[k], hi[k]))
            for k in lo
            if k in hi
        }
        return dict(sorted(out.items(), key=lambda kv: -kv[1].upper))

    def show(self, **kwargs: Any):
        """Draw the upper network; the envelope is reported alongside it."""
        return self.upper.view(**kwargs).show()

    def summary(self) -> str:
        env = self.envelope()
        return "\n".join(
            [
                f"Imprecise Bayesian network — {self.name}",
                f"  variables      {len(self.lower.cpts)}",
                f"  basic events   {len(self.lower.cpts.roots)}",
                f"  P(top = Fail)  {env}  (width {env.width:.6f})",
            ]
        )


# --------------------------------------------------------------------------- #
# Functional aliases
# --------------------------------------------------------------------------- #
def fault_tree_to_bayesnet(
    tree: FaultTree,
    failure_model: Optional[FailureModel] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> BayesianNetwork:
    """Build a :class:`BayesianNetwork` from a fault tree.

    The functional form of :meth:`BayesianNetwork.from_fault_tree`::

        bn = fault_tree_to_bayesnet(report.tree("H2"), report.failure_model)
        bn.p_fail()
    """
    return BayesianNetwork.from_fault_tree(
        tree, failure_model, name=name, **kwargs
    )


def exact_top_probability(network: BayesianNetwork) -> float:
    """Exact ``P(top event)`` --- the reference for the cut-set estimate."""
    return network.p_fail()


def compare_with_cutsets(
    network: BayesianNetwork, analysis: TreeAnalysis
) -> Dict[str, float]:
    """Exact network probability against the minimal cut upper bound."""
    return network.compare_with_cutsets(analysis)
