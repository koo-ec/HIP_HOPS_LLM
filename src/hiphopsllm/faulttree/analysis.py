"""
hipgraph.analysis — minimal cut sets, quantification and FMEA
(HiP-HOPS *Phase 3*: "analyse the synthesised trees").

Three products are derived from each synthesised tree:

**Minimal cut sets.**  The smallest combinations of basic events that are
together sufficient to cause the top event.  Order-1 cut sets are *single points
of failure*: one fault, one hazard, no redundancy in between.  For agentic
systems this is the number that matters — an architecture drawn with two
"independent" agents and a judge looks redundant, and the cut sets say whether
it actually is.

**Quantification.**  Point probabilities per cut set, and a top-event estimate by
the minimal-cut upper bound.  Where a basic event carries an imprecise
``prob_interval`` the bound is evaluated at both ends, so the result is reported
as an interval rather than false precision.  All defaults are placeholders and
are labelled as such; the value of the analysis is in the *structure* and the
ranking, which are unaffected by the absolute numbers.

**FMEA.**  The inverse view: for each component failure mode, which hazards it
causes, whether it causes them alone, and how much it contributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .failure import BasicEvent, FailureModel, FClass
from .synthesis import FaultTree, FTNode

__all__ = [
    "CutSetResult",
    "Quantification",
    "cut_sets",
    "quantify",
    "importance",
    "fmea_table",
    "single_points_of_failure",
    "analyse_tree",
    "TreeAnalysis",
]

CutSet = FrozenSet[str]


# --------------------------------------------------------------------------- #
# Minimal cut sets
# --------------------------------------------------------------------------- #
@dataclass
class CutSetResult:
    sets: List[CutSet] = field(default_factory=list)
    symbols: Dict[str, BasicEvent] = field(default_factory=dict)
    truncated: bool = False
    max_order: int = 0
    notes: List[str] = field(default_factory=list)

    def by_order(self) -> Dict[int, List[CutSet]]:
        out: Dict[int, List[CutSet]] = {}
        for cs in self.sets:
            out.setdefault(len(cs), []).append(cs)
        return dict(sorted(out.items()))

    def order_1(self) -> List[str]:
        return sorted(next(iter(cs)) for cs in self.sets if len(cs) == 1)

    def containing(self, event_id: str) -> List[CutSet]:
        return [cs for cs in self.sets if event_id in cs]


def _minimise(sets: Iterable[CutSet]) -> List[CutSet]:
    """Remove non-minimal cut sets (absorption law)."""
    unique = sorted(set(sets), key=lambda s: (len(s), sorted(s)))
    minimal: List[CutSet] = []
    for cs in unique:
        if not any(m <= cs for m in minimal):
            minimal.append(cs)
    return minimal


def cut_sets(
    tree: FaultTree,
    fmodel: Optional[FailureModel] = None,
    max_order: int = 6,
    max_sets: int = 20000,
) -> CutSetResult:
    """Compute the minimal cut sets of a synthesised tree (MOCUS, bottom-up).

    ``max_order`` and ``max_sets`` bound the combinatorial expansion; truncation
    is always reported rather than silently applied, because an unreported
    truncation reads as "there is nothing else", which would be false.
    """
    result = CutSetResult(max_order=max_order)
    memo: Dict[str, List[CutSet]] = {}

    def symbol_for(node: FTNode) -> str:
        if node.ntype == "basic" and node.event_id:
            eid = node.event_id
            if fmodel and eid in fmodel.events:
                result.symbols[eid] = fmodel.events[eid]
            elif eid in tree.events:
                result.symbols[eid] = tree.events[eid]
            return eid
        eid = f"UND-{node.id}"
        result.symbols.setdefault(eid, BasicEvent(
            id=eid, component=node.component or "?", label=node.label,
            fclass=FClass.OMISSION, prob=0.05, kind="undeveloped",
            rationale=node.detail or "Undeveloped event: analysis stops here.",
            mitigation="Develop this branch before relying on the quantitative result.",
            evidence="undeveloped — probability is a placeholder",
        ))
        return eid

    def compute(nid: str) -> List[CutSet]:
        if nid in memo:
            return memo[nid]
        node = tree.nodes[nid]
        if node.ntype == "transfer":
            # expand_to_tree() produces reference stubs for drawing only; they
            # carry no children, so quantifying them would silently drop whole
            # branches. Refuse rather than under-report.
            raise ValueError(
                "this tree contains transfer references (it came from "
                "expand_to_tree(), which is for drawing). Analyse the synthesised "
                "tree instead — report.trees[...] or report.raw_trees[...]."
            )
        if node.ntype in ("basic", "undeveloped"):
            out = [frozenset({symbol_for(node)})]
        elif node.ntype == "house":
            out = [frozenset()]
        elif not node.children:
            out = []
        elif node.gate == "AND" and len(node.children) > 1:
            out = [frozenset()]
            for child in node.children:
                merged: List[CutSet] = []
                child_sets = compute(child)
                if not child_sets:
                    out = []
                    break
                for a in out:
                    for b in child_sets:
                        union = a | b
                        if len(union) <= max_order:
                            merged.append(union)
                        else:
                            result.truncated = True
                out = _minimise(merged)
                if len(out) > max_sets:
                    result.truncated = True
                    out = out[:max_sets]
        else:  # OR gate, or a single-child pass-through
            collected: List[CutSet] = []
            for child in node.children:
                collected.extend(compute(child))
            out = _minimise(collected)
            if len(out) > max_sets:
                result.truncated = True
                out = out[:max_sets]
        memo[nid] = out
        return out

    result.sets = _minimise(compute(tree.root))
    if result.truncated:
        result.notes.append(
            f"Expansion truncated at order {max_order} / {max_sets} sets. Cut sets of "
            "higher order exist but are not listed; the quantitative result is therefore "
            "a lower bound on the number of contributors."
        )
    return result


# --------------------------------------------------------------------------- #
# Quantification
# --------------------------------------------------------------------------- #
@dataclass
class Quantification:
    top_probability: float = 0.0
    top_interval: Tuple[float, float] = (0.0, 0.0)
    rare_event_sum: float = 0.0
    cut_set_probability: Dict[CutSet, float] = field(default_factory=dict)
    method: str = "minimal-cut upper bound (MCUB)"
    imprecise: bool = False
    notes: List[str] = field(default_factory=list)


def _p(event: BasicEvent, bound: str = "point") -> float:
    if bound == "point":
        return float(event.prob)
    lo, hi = event.interval
    return float(lo if bound == "lower" else hi)


def _mcub(sets: Sequence[CutSet], symbols: Dict[str, BasicEvent], bound: str = "point") -> float:
    q = 1.0
    for cs in sets:
        p = 1.0
        for eid in cs:
            ev = symbols.get(eid)
            p *= _p(ev, bound) if ev else 0.05
        q *= (1.0 - p)
    return 1.0 - q


def quantify(result: CutSetResult) -> Quantification:
    """Top-event probability by the minimal-cut upper bound, with intervals."""
    q = Quantification()
    if not result.sets:
        q.notes.append("No cut sets: the hazard has no modelled cause.")
        return q

    for cs in result.sets:
        p = 1.0
        for eid in cs:
            ev = result.symbols.get(eid)
            p *= _p(ev) if ev else 0.05
        q.cut_set_probability[cs] = p

    q.rare_event_sum = sum(q.cut_set_probability.values())
    q.top_probability = _mcub(result.sets, result.symbols)
    q.imprecise = any(ev.prob_interval for ev in result.symbols.values())
    if q.imprecise:
        q.top_interval = (
            _mcub(result.sets, result.symbols, "lower"),
            _mcub(result.sets, result.symbols, "upper"),
        )
    else:
        q.top_interval = (q.top_probability, q.top_probability)
    if result.truncated:
        q.notes.append("Cut set expansion was truncated; the estimate is not complete.")
    q.notes.append(
        "Probabilities are per-invocation and, unless marked 'measured', are engineering "
        "placeholders. Use the ranking, not the absolute value."
    )
    return q


# --------------------------------------------------------------------------- #
# Importance measures
# --------------------------------------------------------------------------- #
@dataclass
class ImportanceRow:
    event_id: str
    component: str
    label: str
    probability: float
    n_cut_sets: int
    min_order: int
    fussell_vesely: float
    birnbaum: float
    risk_reduction_worth: float
    single_point: bool
    kind: str


def importance(result: CutSetResult, quant: Quantification) -> List[ImportanceRow]:
    """Rank basic events by contribution to the top event.

    Fussell-Vesely is the share of the (rare-event) top probability carried by
    cut sets containing the event; Birnbaum is the sensitivity of the top event
    to that event; risk reduction worth is the factor by which the top event
    would fall if the event were eliminated.
    """
    rows: List[ImportanceRow] = []
    total = quant.rare_event_sum or 1e-12
    for eid, event in result.symbols.items():
        containing = result.containing(eid)
        if not containing:
            continue
        fv = sum(quant.cut_set_probability.get(cs, 0.0) for cs in containing) / total

        original = event.prob
        try:
            event.prob = 1.0
            q_hi = _mcub(result.sets, result.symbols)
            event.prob = 0.0
            q_lo = _mcub(result.sets, result.symbols)
        finally:
            event.prob = original
        birnbaum = q_hi - q_lo
        rrw = (quant.top_probability / q_lo) if q_lo > 1e-12 else float("inf")

        rows.append(ImportanceRow(
            event_id=eid,
            component=event.component,
            label=event.label,
            probability=event.prob,
            n_cut_sets=len(containing),
            min_order=min(len(cs) for cs in containing),
            fussell_vesely=fv,
            birnbaum=birnbaum,
            risk_reduction_worth=rrw,
            single_point=any(len(cs) == 1 for cs in containing),
            kind=event.kind,
        ))
    rows.sort(key=lambda r: (-r.fussell_vesely, r.min_order, r.event_id))
    return rows


# --------------------------------------------------------------------------- #
# Per-tree analysis bundle
# --------------------------------------------------------------------------- #
@dataclass
class TreeAnalysis:
    tree: FaultTree
    cuts: CutSetResult
    quant: Quantification
    importance: List[ImportanceRow]

    @property
    def single_points(self) -> List[str]:
        return self.cuts.order_1()


def analyse_tree(
    tree: FaultTree,
    fmodel: Optional[FailureModel] = None,
    max_order: int = 6,
    max_sets: int = 20000,
) -> TreeAnalysis:
    cuts = cut_sets(tree, fmodel, max_order=max_order, max_sets=max_sets)
    quant = quantify(cuts)
    return TreeAnalysis(tree=tree, cuts=cuts, quant=quant, importance=importance(cuts, quant))


def single_points_of_failure(analyses: Dict[str, TreeAnalysis]) -> List[Dict[str, str]]:
    """Every (hazard, basic event) pair where one failure alone causes the hazard."""
    rows: List[Dict[str, str]] = []
    for hid, analysis in analyses.items():
        for eid in analysis.single_points:
            event = analysis.cuts.symbols.get(eid)
            rows.append({
                "hazard": hid,
                "hazard_name": analysis.tree.name,
                "severity": analysis.tree.hazard.severity if analysis.tree.hazard else "",
                "event": eid,
                "component": event.component if event else "?",
                "failure_mode": event.label if event else "",
                "kind": event.kind if event else "",
                "mitigation": event.mitigation if event else "",
            })
    order = {"catastrophic": 0, "critical": 1, "major": 2, "minor": 3}
    rows.sort(key=lambda r: (order.get(r["severity"], 9), r["component"], r["event"]))
    return rows


# --------------------------------------------------------------------------- #
# FMEA
# --------------------------------------------------------------------------- #
@dataclass
class FMEARow:
    component: str
    failure_mode: str
    event_id: str
    failure_class: str
    kind: str
    direct_effects: List[str] = field(default_factory=list)    # hazards caused alone
    further_effects: List[str] = field(default_factory=list)   # hazards caused with others
    max_severity: str = ""
    max_fv: float = 0.0
    probability: float = 0.0
    evidence: str = ""
    mitigation: str = ""


def fmea_table(analyses: Dict[str, TreeAnalysis]) -> List[FMEARow]:
    """Derive the FMEA from the synthesised trees (the HiP-HOPS inversion step).

    'Direct effect' means the failure mode causes the hazard on its own (an
    order-1 cut set); 'further effect' means it does so in combination with other
    failures.  This is the classical HiP-HOPS FMEA, which is generated from the
    trees rather than elicited separately, so the two views cannot disagree.
    """
    severity_rank = {"catastrophic": 4, "critical": 3, "major": 2, "minor": 1}
    rows: Dict[str, FMEARow] = {}

    for hid, analysis in analyses.items():
        severity = analysis.tree.hazard.severity if analysis.tree.hazard else "major"
        fv_by_event = {r.event_id: r.fussell_vesely for r in analysis.importance}
        for eid, event in analysis.cuts.symbols.items():
            containing = analysis.cuts.containing(eid)
            if not containing:
                continue
            row = rows.get(eid)
            if row is None:
                row = FMEARow(
                    component=event.component,
                    failure_mode=event.label,
                    event_id=eid,
                    failure_class=event.fclass.title,
                    kind=event.kind,
                    probability=event.prob,
                    evidence=event.evidence,
                    mitigation=event.mitigation,
                )
                rows[eid] = row
            label = f"{hid} ({analysis.tree.name})"
            if any(len(cs) == 1 for cs in containing):
                row.direct_effects.append(label)
            else:
                row.further_effects.append(label)
            if severity_rank.get(severity, 0) > severity_rank.get(row.max_severity, 0):
                row.max_severity = severity
            row.max_fv = max(row.max_fv, fv_by_event.get(eid, 0.0))

    out = list(rows.values())
    out.sort(key=lambda r: (
        -severity_rank.get(r.max_severity, 0),
        0 if r.direct_effects else 1,
        -r.max_fv,
        r.component,
    ))
    return out
