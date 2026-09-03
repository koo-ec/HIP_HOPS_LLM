"""
hiphopsllm.report — the notebook-facing entry point.

One call takes the object the notebook already renders and returns a complete
safety analysis::

    from hiphopsllm import analyse_langgraph

    report = analyse_langgraph(
        graph,                       # the compiled LangGraph
        name="Approach 1 — ReAct + calculator",
        globals_ns=globals(),        # lets shared model snapshots be detected
        run_state=final_state_1,     # optional: calibrates events from measured entropy
        unroll=1,
    )

    report.display("H2")             # fault tree for 'wrong answer delivered'
    print(report.summary())
    report.save("artifacts/approach1")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .architecture.acyclic import CycleReport, make_acyclic
from .faulttree.analysis import (
    FMEARow,
    TreeAnalysis,
    analyse_tree,
    fmea_table,
    single_points_of_failure,
)
from .architecture.model import Role, SystemModel, extract_architecture
from .faulttree.failure import ComponentFailureLogic, FailureModel, annotate_system
from .faulttree.export import markdown_report, to_dot, to_json, to_mermaid, to_openpsa_xml
from .faulttree.synthesis import FaultTree, Hazard, default_hazards, simplify_tree, synthesise_all

__all__ = ["SafetyReport", "analyse_langgraph", "display_fault_tree", "map_uncertainty"]


# --------------------------------------------------------------------------- #
# Uncertainty -> component mapping
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and not t.isdigit()}


_GENERIC_TOKENS = {"agent", "node", "approach", "step", "llm", "model", "graph"}


def map_uncertainty(
    system: SystemModel,
    uncertainty_summary: Optional[Dict[str, Any]],
    metric: str = "mean_cluster_entropy",
) -> Dict[str, float]:
    """Match the notebook's per-agent uncertainty records to graph components.

    The uncertainty log is keyed by agent labels chosen by the author
    (``approach2_react``), which are not the LangGraph node ids (``react_agent``).
    They are matched on shared, non-generic name tokens; unmatched records are
    ignored rather than guessed at.
    """
    if not uncertainty_summary:
        return {}
    out: Dict[str, float] = {}
    entries = {
        k: v for k, v in uncertainty_summary.items()
        if isinstance(v, dict) and not k.startswith("_")
    }
    for cid in system.components:
        base = cid.split("#")[0].replace("::router", "")
        ctoks = _tokens(base) - _GENERIC_TOKENS
        if not ctoks:
            continue
        best, best_score = None, 0
        for key, stats in entries.items():
            score = len(ctoks & (_tokens(key) - _GENERIC_TOKENS))
            if score > best_score:
                best, best_score = stats, score
        if best is None or best_score == 0:
            continue
        value = best.get(metric)
        if isinstance(value, (int, float)):
            out[cid] = float(value)
    return out


# --------------------------------------------------------------------------- #
# Report bundle
# --------------------------------------------------------------------------- #
@dataclass
class SafetyReport:
    """Everything the analysis produced, plus the exports."""

    name: str
    source_system: SystemModel          # architecture as extracted (may be cyclic)
    system: SystemModel                 # acyclic analysis model
    failure_model: FailureModel
    cycle_report: CycleReport
    hazards: List[Hazard]
    trees: Dict[str, FaultTree] = field(default_factory=dict)
    #: the unreduced trees, one intermediate event per deviation
    raw_trees: Dict[str, FaultTree] = field(default_factory=dict)
    analyses: Dict[str, TreeAnalysis] = field(default_factory=dict)

    # -- access ------------------------------------------------------------- #
    def tree(self, hazard_id: str) -> FaultTree:
        if hazard_id in self.trees:
            return self.trees[hazard_id]
        matches = [k for k in self.trees if k.startswith(hazard_id)]
        if not matches:
            raise KeyError(f"unknown hazard {hazard_id!r}; available: {sorted(self.trees)}")
        return self.trees[matches[0]]

    def analysis(self, hazard_id: str) -> TreeAnalysis:
        return self.analyses[self.tree(hazard_id).id]

    def cut_sets(self, hazard_id: str) -> List[List[str]]:
        return [sorted(cs) for cs in self.analysis(hazard_id).cuts.sets]

    def fmea(self) -> List[FMEARow]:
        return fmea_table(self.analyses)

    def single_points(self) -> List[Dict[str, str]]:
        return single_points_of_failure(self.analyses)

    # -- rendering ---------------------------------------------------------- #
    def mermaid(self, hazard_id: str) -> str:
        return to_mermaid(self.tree(hazard_id))

    def bayesnet(self, hazard_id: str = "H2") -> Any:
        """Equivalent pyAgrum network for one hazard (exact inference, evidence).

        Requires pyagrum. See :mod:`hiphopsllm.bayes`.
        """
        from .bayes import fault_tree_to_bayesnet

        tree = self.tree(hazard_id)
        return fault_tree_to_bayesnet(tree, self.failure_model,
                                      name=f"{self.name}_{tree.id}")

    def markdown(self, include_trees: bool = True) -> str:
        return markdown_report(
            self.system, self.failure_model, self.analyses,
            cycle_report=self.cycle_report, title=self.name,
            include_trees=include_trees,
        )

    def display(self, hazard_id: str = "H2") -> Any:
        """Render one fault tree in the notebook."""
        return display_fault_tree(self.tree(hazard_id))

    def display_architecture(self) -> Any:
        return _display_mermaid(self.system.to_mermaid())

    def summary(self) -> str:
        lines = [f"HiP-HOPS analysis — {self.name}", "=" * (len(self.name) + 22)]
        lines.append(
            f"components: {len(self.system.components)}  "
            f"connections: {len(self.system.connections)}  "
            f"basic events: {len(self.failure_model.events)}"
        )
        lines.append(self.cycle_report.summary())
        groups = self.system.common_cause_groups()
        if groups:
            lines.append("common-cause groups:")
            for (kind, value), members in sorted(groups.items()):
                lines.append(f"  {kind}={value}: {', '.join(members)}")
        lines.append("")
        header = f"{'hazard':<10} {'sev':<13} {'P(top)':>8}  {'MCS':>5} {'SPOF':>5}  name"
        lines.append(header)
        lines.append("-" * len(header))
        for hid in sorted(self.analyses):
            a = self.analyses[hid]
            sev = a.tree.hazard.severity if a.tree.hazard else ""
            lines.append(
                f"{hid:<10} {sev:<13} {a.quant.top_probability:>8.4f}  "
                f"{len(a.cuts.sets):>5} {len(a.single_points):>5}  {a.tree.name}"
            )
        spof = self.single_points()
        if spof:
            lines.append("")
            lines.append(f"single points of failure ({len(spof)}):")
            for row in spof[:12]:
                lines.append(
                    f"  [{row['severity']:<12}] {row['hazard']:<8} {row['event']} "
                    f"({row['component']})"
                )
            if len(spof) > 12:
                lines.append(f"  ... and {len(spof) - 12} more (see the report)")
        return "\n".join(lines)

    # -- persistence -------------------------------------------------------- #
    def save(self, directory: str, prefix: Optional[str] = None) -> List[str]:
        """Write the report and every per-hazard export. Returns the file paths."""
        os.makedirs(directory, exist_ok=True)
        stem = prefix or re.sub(r"[^0-9A-Za-z]+", "_", self.name).strip("_").lower()
        written: List[str] = []

        def _write(filename: str, content: str) -> None:
            path = os.path.join(directory, filename)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            written.append(path)

        _write(f"{stem}_report.md", self.markdown())
        _write(f"{stem}_architecture.mmd", self.system.to_mermaid())
        for hid, tree in self.trees.items():
            safe = re.sub(r"[^0-9A-Za-z]+", "_", hid)
            _write(f"{stem}_{safe}.mmd", to_mermaid(tree))
            _write(f"{stem}_{safe}.dot", to_dot(tree))
            _write(f"{stem}_{safe}.json", to_json(tree, self.analyses.get(hid)))
            _write(f"{stem}_{safe}.opsa.xml", to_openpsa_xml(tree, name=f"{stem}_{safe}"))
        _write(f"{stem}_cutsets.csv", self._cutsets_csv())
        _write(f"{stem}_fmea.csv", self._fmea_csv())
        return written

    def _cutsets_csv(self) -> str:
        rows = ["hazard,order,probability,events"]
        for hid in sorted(self.analyses):
            a = self.analyses[hid]
            for cs in sorted(a.cuts.sets, key=lambda s: (len(s), sorted(s))):
                p = a.quant.cut_set_probability.get(cs, 0.0)
                events = " + ".join(sorted(cs))
                rows.append('{0},{1},{2:.6g},"{3}"'.format(hid, len(cs), p, events))
        return "\n".join(rows) + "\n"

    def _fmea_csv(self) -> str:
        rows = ["component,event,failure_mode,class,probability,direct_effects,"
                "further_effects,max_severity,mitigation"]
        for r in self.fmea():
            def q(text: object) -> str:
                return '"' + str(text).replace('"', "'") + '"'
            rows.append(",".join([
                q(r.component), q(r.event_id), q(r.failure_mode), q(r.failure_class),
                f"{r.probability:.6g}", q("; ".join(r.direct_effects)),
                q("; ".join(r.further_effects)), q(r.max_severity), q(r.mitigation),
            ]))
        return "\n".join(rows) + "\n"


# --------------------------------------------------------------------------- #
# Notebook display
# --------------------------------------------------------------------------- #
def _display_mermaid(mermaid_text: str) -> Any:
    """Render mermaid in a notebook: PNG if possible, otherwise a fenced block."""
    try:  # the same renderer LangGraph uses for draw_mermaid_png()
        from IPython.display import Image, display  # type: ignore
        from langchain_core.runnables.graph_mermaid import draw_mermaid_png  # type: ignore

        return display(Image(draw_mermaid_png(mermaid_text)))
    except Exception:
        pass
    try:
        from IPython.display import Markdown, display  # type: ignore

        return display(Markdown(f"```mermaid\n{mermaid_text}\n```"))
    except Exception:
        print(mermaid_text)
        return None


def display_fault_tree(tree: FaultTree) -> Any:
    """Render a synthesised fault tree inline in the notebook."""
    return _display_mermaid(to_mermaid(tree))


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def analyse_langgraph(
    graph: Any,
    name: str = "LangGraph workflow",
    *,
    globals_ns: Optional[Dict[str, Any]] = None,
    node_functions: Optional[Dict[str, Callable[..., Any]]] = None,
    role_overrides: Optional[Dict[str, Role | str]] = None,
    resource_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    unroll: int = 1,
    simplify: bool = True,
    hazards: Optional[Sequence[Hazard]] = None,
    run_state: Optional[Dict[str, Any]] = None,
    uncertainty_summary: Optional[Dict[str, Any]] = None,
    entropy_by_component: Optional[Dict[str, float]] = None,
    probability_overrides: Optional[Dict[str, float]] = None,
    extra_logic: Optional[Dict[str, ComponentFailureLogic]] = None,
    max_order: int = 6,
    max_sets: int = 20000,
) -> SafetyReport:
    """Run the whole pipeline on a LangGraph application.

    Parameters
    ----------
    graph
        A compiled LangGraph, the drawable graph from ``graph.get_graph()``, the
        mermaid text from ``draw_mermaid()``, a dict specification, or an
        already-built :class:`SystemModel`.
    globals_ns
        Pass ``globals()`` from the notebook. Node functions are then found by
        name and the *actual* model objects are interrogated, which is what makes
        shared-snapshot (common-cause) detection reliable.
    unroll
        Iterations of each feedback loop represented explicitly (default 1).
    simplify
        Reduce each tree to its informative structure — a one-input gate is
        replaced by its input, nested combination gates are flattened. The
        Boolean function and the cut sets are unchanged; the unreduced trees
        stay available as ``report.raw_trees``.
    run_state
        The state returned by ``graph.invoke(...)``. Its ``uncertainty_summary``
        calibrates the hallucination / non-determinism events from measured
        semantic entropy instead of placeholders.
    """
    system = extract_architecture(
        graph, name=name, role_overrides=role_overrides,
        resource_overrides=resource_overrides, globals_ns=globals_ns,
        node_functions=node_functions,
    )
    acyclic, cycle_report = make_acyclic(system, unroll=unroll)

    if uncertainty_summary is None and run_state:
        uncertainty_summary = run_state.get("uncertainty_summary")
    entropies = dict(entropy_by_component or {})
    if uncertainty_summary:
        derived = map_uncertainty(acyclic, uncertainty_summary)
        for cid, value in derived.items():
            entropies.setdefault(cid, value)

    fmodel = annotate_system(
        acyclic,
        probability_overrides=probability_overrides,
        entropy_by_component=entropies or None,
        extra_logic=extra_logic,
    )

    hazard_list = list(hazards) if hazards is not None else default_hazards(acyclic)
    raw_trees = synthesise_all(fmodel, hazard_list, simplify=False)
    trees = (
        {hid: simplify_tree(tree) for hid, tree in raw_trees.items()}
        if simplify else raw_trees
    )
    analyses = {
        hid: analyse_tree(tree, fmodel, max_order=max_order, max_sets=max_sets)
        for hid, tree in trees.items()
    }

    if entropies:
        fmodel.notes.append(
            "Hallucination and non-determinism events for "
            + ", ".join(sorted(entropies))
            + " were calibrated from measured semantic-cluster entropy."
        )

    return SafetyReport(
        name=name,
        source_system=system,
        system=acyclic,
        failure_model=fmodel,
        cycle_report=cycle_report,
        hazards=hazard_list,
        trees=trees,
        raw_trees=raw_trees,
        analyses=analyses,
    )
