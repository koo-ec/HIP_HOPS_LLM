"""Fault tree to conditional probability tables.

A fault tree and a Bayesian network are two readings of the same object.  Every
node of the tree becomes a binary variable; every gate becomes a conditional
probability table over its inputs; every basic event becomes a root with its
failure probability as prior.  Converting rather than re-authoring matters for a
practical reason: the source notebook hand-wired a pyAgrum network *beside* its
fault tree, and nothing kept the two consistent when the graph changed.

Two things become available once the tree is a set of CPTs.

**An exact top-event probability.**  Cut-set quantification uses the minimal cut
upper bound, which over-estimates whenever cut sets share basic events --- and in
agentic architectures they always do, because every agent depends on the same
request and often on the same model snapshot.  Exact inference over the CPTs
gives the true value, so the pair brackets the answer.

**Soft gates.**  A deterministic OR says an input deviation *always* propagates.
A noisy-OR says each input propagates with its own probability and adds a leak
term for causes outside the model.  That is usually the more honest statement
about an LLM component, and it is a CPT away.

Convention
----------
Index ``0`` is ``OK`` and index ``1`` is ``Fail``, throughout, matching
:mod:`HIP_HOPS_LLM.bayes.network`.  (The HIP-MAS study code uses the opposite
order; :func:`CPTSet.to_hipmas_order` converts.)
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..faulttree.failure import BasicEvent, FailureModel
from ..faulttree.synthesis import FaultTree

__all__ = [
    "OK",
    "FAIL",
    "LABELS",
    "MAX_GATE_INPUTS",
    "GateType",
    "CPT",
    "CPTSet",
    "CPTBuilder",
    "fault_tree_to_cpts",
    "deterministic_gate_cpt",
    "noisy_or_cpt",
    "k_of_n_cpt",
    "prior_cpt",
]

#: State indices.  ``0 = OK``, ``1 = Fail``.
OK, FAIL = 0, 1
LABELS: Tuple[str, str] = ("OK", "Fail")

#: A deterministic gate table has ``2 ** inputs`` rows.  Past this the table
#: stops being a usable representation and the cut sets should be used instead.
MAX_GATE_INPUTS = 18


class GateType(str, Enum):
    """How a gate's conditional table is filled in."""

    OR = "OR"
    AND = "AND"
    KOFN = "KOFN"
    NOISY_OR = "NOISY_OR"
    NOISY_AND = "NOISY_AND"
    PASS = "PASS"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


def sanitise(node_id: str) -> str:
    """Turn a fault tree node id into a legal Bayesian-network variable name.

    A leaf's node id is ``BE::<event id>``, and the event id already begins with
    ``BE-``; naively substituting the illegal characters gives
    ``BE__BE_react_agent_HALLUC``, which is correct and unreadable.  The
    namespace prefix is dropped first, so the variable reads as the event does.
    Uniqueness is still enforced by the caller.
    """
    text = re.sub(r"^(BE|E|G)::", "", node_id)
    name = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if not name:
        return "v"
    return name if (name[0].isalpha() or name[0] == "_") else f"n_{name}"


# --------------------------------------------------------------------------- #
# Table constructors
# --------------------------------------------------------------------------- #
def prior_cpt(p_fail: float) -> np.ndarray:
    """Root table for a basic event: ``[P(OK), P(Fail)]``."""
    p = float(p_fail)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"a failure probability must lie in [0, 1], got {p!r}")
    return np.array([1.0 - p, p], dtype=float)


def deterministic_gate_cpt(n_parents: int, gate: str = "OR") -> np.ndarray:
    """Deterministic AND/OR table of shape ``(2,) * n_parents + (2,)``.

    The child fails with probability one exactly on the parent configurations
    the Boolean gate makes true, and with probability zero elsewhere.  This is
    the classical fault tree reading, and it is what makes exact inference over
    the network agree with an exhaustive evaluation of the Boolean function.
    """
    if n_parents < 1:
        raise ValueError("a gate needs at least one input")
    _check_fan_in(n_parents)
    table = np.zeros((2,) * n_parents + (2,), dtype=float)
    want_all = str(gate).upper() == "AND"
    for combo in itertools.product((0, 1), repeat=n_parents):
        fails = all(combo) if want_all else any(combo)
        table[combo] = (0.0, 1.0) if fails else (1.0, 0.0)
    return table


def k_of_n_cpt(n_parents: int, k: int) -> np.ndarray:
    """Voting gate: the child fails when at least ``k`` of ``n`` inputs fail.

    ``k = 1`` reproduces OR and ``k = n`` reproduces AND, so this generalises
    both.  It is the right shape for a majority-vote aggregator, where two wrong
    answers out of three carry the error through and one does not.
    """
    if not 1 <= k <= n_parents:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n_parents}")
    _check_fan_in(n_parents)
    table = np.zeros((2,) * n_parents + (2,), dtype=float)
    for combo in itertools.product((0, 1), repeat=n_parents):
        fails = sum(combo) >= k
        table[combo] = (0.0, 1.0) if fails else (1.0, 0.0)
    return table


def noisy_or_cpt(
    link_probabilities: Sequence[float],
    leak: float = 0.0,
) -> np.ndarray:
    """Noisy-OR table: each failed parent independently *tries* to cause failure.

    ``P(child = Fail | parents) = 1 - (1 - leak) * prod_{i failed} (1 - p_i)``

    The deterministic OR is the special case where every ``p_i`` is one and the
    leak is zero.  Softening it is the honest option when an input deviation only
    *sometimes* propagates --- a malformed tool observation that the agent
    occasionally recovers from, say --- and the leak covers causes the tree does
    not model, which for an LLM component is never an empty set.
    """
    ps = np.asarray(list(link_probabilities), dtype=float)
    if ps.ndim != 1 or ps.size < 1:
        raise ValueError("link_probabilities must be a non-empty 1-D sequence")
    if np.any((ps < 0.0) | (ps > 1.0)):
        raise ValueError("every link probability must lie in [0, 1]")
    if not 0.0 <= float(leak) <= 1.0:
        raise ValueError("leak must lie in [0, 1]")
    n = ps.size
    _check_fan_in(n)
    table = np.zeros((2,) * n + (2,), dtype=float)
    for combo in itertools.product((0, 1), repeat=n):
        survive = (1.0 - float(leak)) * float(np.prod(np.where(np.asarray(combo) == 1, 1.0 - ps, 1.0)))
        p_fail = 1.0 - survive
        table[combo] = (1.0 - p_fail, p_fail)
    return table


def _check_fan_in(n_parents: int) -> None:
    if n_parents > MAX_GATE_INPUTS:
        raise ValueError(
            f"a gate with {n_parents} inputs would need a 2**{n_parents}-row conditional "
            f"table. Tables are built up to {MAX_GATE_INPUTS} inputs; for wider gates use "
            "the minimal cut sets, or raise HIP_HOPS_LLM.bayes.cpt.MAX_GATE_INPUTS "
            "deliberately."
        )


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CPT:
    """One conditional probability table, with the provenance of its numbers."""

    variable: str
    parents: Tuple[str, ...]
    table: np.ndarray = field(repr=False)
    #: how the table was filled: deterministic gate, noisy-OR, prior, house
    kind: str = "deterministic"
    gate: Optional[GateType] = None
    #: fault tree node this came from, and a human-readable justification
    node_id: Optional[str] = None
    label: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        expected = (2,) * len(self.parents) + (2,)
        arr = np.asarray(self.table, dtype=float)
        if arr.shape != expected:
            raise ValueError(
                f"CPT for {self.variable!r} has shape {arr.shape}, expected {expected}"
            )
        sums = arr.sum(axis=-1)
        if not np.allclose(sums, 1.0, atol=1e-9):
            raise ValueError(
                f"CPT for {self.variable!r} is not normalised; row sums {np.unique(sums)}"
            )
        if np.any(arr < -1e-12):
            raise ValueError(f"CPT for {self.variable!r} has negative entries")
        object.__setattr__(self, "table", np.clip(arr, 0.0, 1.0))

    @property
    def n_rows(self) -> int:
        return int(np.prod(self.table.shape[:-1])) if self.parents else 1

    @property
    def is_root(self) -> bool:
        return not self.parents

    @property
    def prior_fail(self) -> float:
        """``P(Fail)`` for a root variable."""
        if not self.is_root:
            raise ValueError(f"{self.variable!r} has parents; use inference instead")
        return float(self.table[FAIL])

    def rows(self) -> List[Dict[str, Any]]:
        """The table as one dict per parent configuration, for display."""
        if self.is_root:
            return [{"P(Fail)": float(self.table[FAIL])}]
        out: List[Dict[str, Any]] = []
        for combo in itertools.product((0, 1), repeat=len(self.parents)):
            row: Dict[str, Any] = {
                p: LABELS[c] for p, c in zip(self.parents, combo)
            }
            row["P(Fail)"] = float(self.table[combo][FAIL])
            out.append(row)
        return out


@dataclass
class CPTSet:
    """Every CPT of one fault tree, in a valid topological order.

    ``order`` lists variables parents-first, so the set can be handed to any
    Bayesian-network library --- or to the exact enumeration reference in
    :mod:`HIP_HOPS_LLM.bayes.network` --- without a further sort.
    """

    name: str
    cpts: Dict[str, CPT] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    #: fault tree node id -> variable name, and the inverse
    variable_of: Dict[str, str] = field(default_factory=dict)
    node_of: Dict[str, str] = field(default_factory=dict)
    #: variable name of the top event
    top: str = ""
    #: basic event id -> variable name, for evidence and diagnosis by event name
    event_variable: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cpts)

    def __iter__(self):
        return iter(self.order)

    def __getitem__(self, key: str) -> CPT:
        if key in self.cpts:
            return self.cpts[key]
        if key in self.variable_of:
            return self.cpts[self.variable_of[key]]
        if key in self.event_variable:
            return self.cpts[self.event_variable[key]]
        raise KeyError(
            f"{key!r} is not a variable, node id or basic event of {self.name!r}"
        )

    def add(self, cpt: CPT) -> None:
        if cpt.variable in self.cpts:
            raise ValueError(f"variable {cpt.variable!r} declared twice")
        for parent in cpt.parents:
            if parent not in self.cpts:
                raise ValueError(
                    f"parent {parent!r} of {cpt.variable!r} must be declared first; "
                    "CPTSet keeps a topological order"
                )
        self.cpts[cpt.variable] = cpt
        self.order.append(cpt.variable)
        if cpt.node_id:
            self.variable_of[cpt.node_id] = cpt.variable
            self.node_of[cpt.variable] = cpt.node_id

    @property
    def roots(self) -> List[str]:
        return [v for v in self.order if self.cpts[v].is_root]

    @property
    def priors(self) -> Dict[str, float]:
        """``{variable: P(Fail)}`` for every root, i.e. every basic event."""
        return {v: self.cpts[v].prior_fail for v in self.roots}

    def parents_of(self, variable: str) -> Tuple[str, ...]:
        return self[variable].parents

    def resolve(self, key: str) -> str:
        """Variable name for a variable, fault tree node id, or basic event id."""
        if key in self.cpts:
            return key
        if key in self.variable_of:
            return self.variable_of[key]
        if key in self.event_variable:
            return self.event_variable[key]
        raise KeyError(
            f"{key!r} is unknown in {self.name!r}. Basic events here are: "
            f"{sorted(self.event_variable)}"
        )

    def to_frame(self, variable: Optional[str] = None):
        """CPTs as a pandas DataFrame --- one block per variable, for inspection."""
        import pandas as pd

        names = [self.resolve(variable)] if variable else self.order
        rows: List[Dict[str, Any]] = []
        for name in names:
            cpt = self.cpts[name]
            for row in cpt.rows():
                parents = " , ".join(
                    f"{k}={v}" for k, v in row.items() if k != "P(Fail)"
                )
                rows.append(
                    {
                        "variable": name,
                        "kind": cpt.kind,
                        "gate": str(cpt.gate) if cpt.gate else "",
                        "parents": parents or "(root)",
                        "P(Fail)": row["P(Fail)"],
                    }
                )
        return pd.DataFrame(rows)

    def to_hipmas_order(self) -> Dict[str, np.ndarray]:
        """Tables re-indexed to the HIP-MAS convention, where index 0 is ``Fail``.

        The HIP-MAS study code (``hipmas.bn``) puts ``Fail`` first.  This package
        puts ``OK`` first so that ``table[..., 1]`` reads as "probability of
        failure" everywhere.  Use this when handing tables to that code.
        """
        return {v: np.flip(c.table, axis=-1).copy() for v, c in self.cpts.items()}

    def summary(self) -> str:
        gates = [c for c in self.cpts.values() if not c.is_root]
        soft = [c for c in gates if c.kind != "deterministic"]
        lines = [
            f"CPT set — {self.name}",
            f"  variables      {len(self.cpts)}",
            f"  basic events   {len(self.roots)}",
            f"  gates          {len(gates)} ({len(soft)} soft, "
            f"{len(gates) - len(soft)} deterministic)",
            f"  top event      {self.top}",
            f"  table rows     {sum(c.n_rows for c in self.cpts.values())}",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The conversion
# --------------------------------------------------------------------------- #
@dataclass
class CPTBuilder:
    """Convert fault trees into :class:`CPTSet` objects, under fixed conventions.

    Parameters
    ----------
    bound
        Which end of an imprecise basic-event probability to use.  ``"point"``
        takes :attr:`BasicEvent.prob`; ``"lower"`` and ``"upper"`` take the ends
        of :attr:`BasicEvent.prob_interval` when one is present.  Because a
        coherent (monotone) fault tree's top-event probability is non-decreasing
        in every basic-event probability, building at both ends brackets the true
        value --- which is how :class:`HIP_HOPS_LLM.bayes.network.BayesianNetwork`
        reports an imprecise result.
    soft_gates
        Replace deterministic OR gates with noisy-OR.  ``False`` (the default)
        keeps the classical fault tree reading, in which the network and the cut
        sets describe exactly the same Boolean function.
    link_probability
        Per-input propagation probability used when ``soft_gates`` is on.
    leak
        Noisy-OR leak: the chance the child fails with every modelled input OK.
    default_prob
        Prior for a basic event that carries no probability at all.  It is
        deliberately visible in the report rather than silent, because a whole
        network quietly running on default priors is a failure mode this package
        exists to prevent.
    """

    bound: str = "point"
    soft_gates: bool = False
    link_probability: float = 0.9
    leak: float = 0.01
    default_prob: float = 0.05
    max_gate_inputs: int = MAX_GATE_INPUTS

    def __post_init__(self) -> None:
        if self.bound not in ("point", "lower", "upper"):
            raise ValueError("bound must be 'point', 'lower' or 'upper'")

    # -- probability lookup -------------------------------------------------- #
    def probability(self, event: Optional[BasicEvent]) -> float:
        if event is None:
            return float(self.default_prob)
        if self.bound == "point":
            return float(event.prob)
        lo, hi = event.interval
        return float(lo if self.bound == "lower" else hi)

    # -- the conversion ------------------------------------------------------ #
    def build(
        self,
        tree: FaultTree,
        failure_model: Optional[FailureModel] = None,
        name: Optional[str] = None,
        gate_overrides: Optional[Mapping[str, str | GateType | Tuple[str, int]]] = None,
    ) -> CPTSet:
        """Convert one fault tree into a :class:`CPTSet`.

        ``gate_overrides`` maps a fault tree node id to a replacement gate:
        ``"AND"``, ``"OR"``, ``"NOISY_OR"``, or ``("KOFN", k)`` for a voting
        gate.  Use it where the synthesised structure is right but the *logic*
        is not --- a two-of-three aggregator being the usual case.
        """
        overrides = dict(gate_overrides or {})
        cs = CPTSet(name=name or f"FT_{tree.id}")
        used_names: Dict[str, str] = {}
        missing: List[str] = []

        for node_id in _topological(tree):
            node = tree.nodes[node_id]
            var = _unique(sanitise(node_id), used_names)
            used_names[var] = node_id

            if node.ntype in ("basic", "undeveloped"):
                event = _lookup_event(node.event_id, tree, failure_model)
                if event is None and node.event_id:
                    missing.append(node.event_id)
                p = self.probability(event)
                cs.add(
                    CPT(
                        variable=var,
                        parents=(),
                        table=prior_cpt(p),
                        kind="prior",
                        node_id=node_id,
                        label=node.label,
                        evidence=(event.evidence if event else "no basic event found"),
                    )
                )
                if node.event_id:
                    cs.event_variable[node.event_id] = var
                continue

            if node.ntype == "house":
                cs.add(
                    CPT(
                        variable=var,
                        parents=(),
                        table=np.array([0.0, 1.0]),
                        kind="house",
                        node_id=node_id,
                        label=node.label,
                        evidence="house event: assumed to occur",
                    )
                )
                continue

            parents = tuple(cs.variable_of[c] for c in node.children)
            if not parents:
                # A gate with no inputs cannot fail.
                cs.add(
                    CPT(
                        variable=var,
                        parents=(),
                        table=np.array([1.0, 0.0]),
                        kind="prior",
                        node_id=node_id,
                        label=node.label,
                        evidence="gate with no inputs: cannot occur",
                    )
                )
                continue

            table, kind, gate_type = self._gate_table(
                node_id, node.gate, len(parents), overrides
            )
            cs.add(
                CPT(
                    variable=var,
                    parents=parents,
                    table=table,
                    kind=kind,
                    gate=gate_type,
                    node_id=node_id,
                    label=node.label,
                    evidence=(
                        "deterministic fault tree gate"
                        if kind == "deterministic"
                        else f"noisy gate, link={self.link_probability}, leak={self.leak}"
                    ),
                )
            )

        cs.top = cs.variable_of[tree.root]
        if missing:
            cs.notes.append(
                "no probability was found for "
                + ", ".join(sorted(set(missing)))
                + f"; the default prior {self.default_prob} was used"
            )
        if self.bound != "point":
            cs.notes.append(f"basic events were taken at their {self.bound} bound")
        if self.soft_gates:
            cs.notes.append(
                "OR gates were softened to noisy-OR: the network no longer encodes "
                "the same Boolean function as the minimal cut sets"
            )
        return cs

    def _gate_table(
        self,
        node_id: str,
        gate: Optional[str],
        n_parents: int,
        overrides: Mapping[str, Any],
    ) -> Tuple[np.ndarray, str, GateType]:
        spec = overrides.get(node_id, gate or "OR")
        k: Optional[int] = None
        if isinstance(spec, tuple):
            spec, k = spec[0], int(spec[1])
        name = str(spec).upper()

        if name == "KOFN":
            if k is None:
                raise ValueError(
                    f"gate override for {node_id!r} is KOFN but no k was given; "
                    "pass ('KOFN', k)"
                )
            return k_of_n_cpt(n_parents, k), "deterministic", GateType.KOFN
        if name == "NOISY_OR" or (self.soft_gates and name == "OR"):
            return (
                noisy_or_cpt([self.link_probability] * n_parents, leak=self.leak),
                "noisy_or",
                GateType.NOISY_OR,
            )
        if name == "AND":
            return deterministic_gate_cpt(n_parents, "AND"), "deterministic", GateType.AND
        return deterministic_gate_cpt(n_parents, "OR"), "deterministic", GateType.OR


def fault_tree_to_cpts(
    tree: FaultTree,
    failure_model: Optional[FailureModel] = None,
    *,
    name: Optional[str] = None,
    bound: str = "point",
    soft_gates: bool = False,
    link_probability: float = 0.9,
    leak: float = 0.01,
    default_prob: float = 0.05,
    gate_overrides: Optional[Mapping[str, Any]] = None,
) -> CPTSet:
    """Convert a fault tree into conditional probability tables.

    The functional form of :class:`CPTBuilder`, for the common case::

        cpts = fault_tree_to_cpts(report.tree("H2"), report.failure_model)
        print(cpts.summary())
        cpts.to_frame("BE-react_agent-HALLUC")

    See :class:`CPTBuilder` for what each option means.
    """
    builder = CPTBuilder(
        bound=bound,
        soft_gates=soft_gates,
        link_probability=link_probability,
        leak=leak,
        default_prob=default_prob,
    )
    return builder.build(
        tree, failure_model, name=name, gate_overrides=gate_overrides
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unique(base: str, taken: Mapping[str, str]) -> str:
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def _lookup_event(
    event_id: Optional[str],
    tree: FaultTree,
    failure_model: Optional[FailureModel],
) -> Optional[BasicEvent]:
    if not event_id:
        return None
    if failure_model is not None and event_id in failure_model.events:
        return failure_model.events[event_id]
    if event_id in tree.events:
        return tree.events[event_id]
    return None


def _topological(tree: FaultTree) -> List[str]:
    """Node ids parents-first (children before the gates that consume them)."""
    order: List[str] = []
    seen: set = set()
    stack: List[Tuple[str, bool]] = [(tree.root, False)]
    while stack:
        nid, expanded = stack.pop()
        if expanded:
            if nid not in seen:
                seen.add(nid)
                order.append(nid)
            continue
        if nid in seen:
            continue
        stack.append((nid, True))
        for child in tree.nodes[nid].children:
            if child not in seen:
                stack.append((child, False))
    # Anything unreachable from the root (there should be nothing) is appended.
    for nid in tree.nodes:
        if nid not in seen:
            order.append(nid)
    return order
