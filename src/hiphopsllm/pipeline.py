"""The whole analysis behind one object.

:class:`AgenticReliabilityStudy` runs the five steps this package exists to
connect, in order, and keeps every intermediate result addressable:

1. **Extract** the architecture from a LangGraph application.
2. **Synthesise** static fault trees for each system-level hazard, unrolling
   feedback loops so the trees are genuinely acyclic.
3. **Measure** each component's failure probability from observed outcomes under
   an explicit operational profile, using HIP-LLM's hierarchical imprecise
   posterior --- which returns an interval, because a few hundred items do not
   identify a point.
4. **Calibrate** the trees' basic events with those intervals, replacing
   engineering-judgement placeholders and recording what could not be matched.
5. **Convert** a tree to conditional probability tables and a Bayesian network,
   for an exact top-event probability, a posterior over causes, and a picture.

End to end::

    from hiphopsllm import AgenticReliabilityStudy, load_example, load_outcomes

    study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
    study.observe(load_outcomes(), stratum_column="stratum",
                  profile={"short": 0.3, "medium": 0.5, "long": 0.2})
    study.run()
    print(study.summary())
    study.bayesnet("H2").show()

Each step can be run on its own, and nothing is done implicitly: a study that has
not been given outcomes reports placeholder probabilities and says so, rather
than quietly presenting judgement as measurement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .architecture.model import Role, SystemModel
from .bayes.network import BayesianNetwork, Envelope, ImpreciseBayesianNetwork
from .faulttree.synthesis import Hazard
from .reliability.calibration import (
    CalibrationReport,
    ComponentEvidence,
    EvidenceCalibrator,
)
from .reliability.profile import OperationalProfile, empirical_profile
from .report import SafetyReport, analyse_langgraph

__all__ = ["AgenticReliabilityStudy", "StudyNotReady"]


class StudyNotReady(RuntimeError):
    """A step was asked for before the step it depends on had been run."""


@dataclass
class AgenticReliabilityStudy:
    """A reliability study of one agentic workflow, from graph to Bayesian network.

    Parameters
    ----------
    graph
        A compiled LangGraph, the drawable from ``graph.get_graph()``, mermaid
        text, a graph specification dict (see
        :func:`~hiphopsllm.io.examples.load_example`), or a
        :class:`~hiphopsllm.architecture.model.SystemModel`.
    name
        What the report calls this system.
    profile
        The operational profile.  It can also be supplied later, to
        :meth:`observe`.
    globals_ns
        Pass ``globals()`` in a notebook so the node functions and the real model
        objects can be found; that is what makes shared-snapshot detection --- and
        therefore common-cause analysis --- reliable.
    node_functions
        Explicit ``{node_id: function}`` mapping when the functions are not in
        ``globals_ns``.  Use ``"<node>::router"`` for a conditional-edge function.
    unroll
        Iterations of each feedback loop represented explicitly.
    hazards
        System-level hazards to synthesise trees for.  The default set is derived
        from the architecture.
    settings
        HIP-LLM inference settings; defaults to the interactive-speed ones.
    exact_inference
        Run HIP-LLM's full hierarchical inference (default).  ``False`` uses a
        Jeffreys approximation --- much faster, and labelled as an approximation
        in every evidence string it writes.
    """

    graph: Any
    name: str = "agentic workflow"
    profile: Optional[OperationalProfile | Mapping[str, float]] = None
    globals_ns: Optional[Dict[str, Any]] = None
    node_functions: Optional[Dict[str, Callable[..., Any]]] = None
    role_overrides: Dict[str, Role | str] = field(default_factory=dict)
    resource_overrides: Dict[str, Dict[str, str]] = field(default_factory=dict)
    unroll: int = 1
    hazards: Optional[Sequence[Hazard]] = None
    settings: Any = None
    exact_inference: bool = True
    credible_level: float = 0.95
    bound: str = "credible"

    # -- populated by the steps --------------------------------------------- #
    report: Optional[SafetyReport] = field(default=None, init=False)
    evidence: Dict[str, ComponentEvidence] = field(default_factory=dict, init=False)
    calibration: Optional[CalibrationReport] = field(default=None, init=False)
    #: the system-level HIP-LLM result, when whole-system outcomes were given
    operational_failure: Any = field(default=None, init=False)
    _observations: Dict[str, Tuple[Sequence[Any], Sequence[str]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _component_map: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.profile is not None:
            self.profile = OperationalProfile.coerce(self.profile)

    # ---------------------------------------------------------------- step 1-2
    def analyse(self, **kwargs: Any) -> SafetyReport:
        """Extract the architecture and synthesise the fault trees.

        Called automatically by :meth:`run` and by anything that needs a tree, so
        it rarely has to be called by hand.  It *is* worth calling on its own
        when you want the structural result --- cut sets, single points of
        failure, the FMEA --- before any measurement exists.
        """
        self.report = analyse_langgraph(
            self.graph,
            name=self.name,
            globals_ns=self.globals_ns,
            node_functions=self.node_functions,
            role_overrides=self.role_overrides or None,
            resource_overrides=self.resource_overrides or None,
            unroll=self.unroll,
            hazards=self.hazards,
            **kwargs,
        )
        return self.report

    # ------------------------------------------------------------------ step 3
    def observe(
        self,
        outcomes: Any = None,
        strata: Optional[Sequence[str]] = None,
        *,
        profile: Optional[OperationalProfile | Mapping[str, float]] = None,
        component: Optional[str] = None,
        stratum_column: str = "stratum",
        component_columns: Optional[Sequence[str]] = None,
        split_column: Optional[str] = "split",
        calibration_split: str = "calibration",
        component_map: Optional[Mapping[str, str]] = None,
    ) -> "AgenticReliabilityStudy":
        """Record measured outcomes.  Accepts three shapes.

        **A table** --- one row per item, a stratum column, and one ``1``/``0``
        correctness column per component::

            study.observe(load_outcomes(), profile={"short": .3, "medium": .5, "long": .2})

        **One component** --- outcomes and strata as sequences::

            study.observe([1, 1, 0, 1], ["short"] * 4, component="react_agent")

        **The whole system** --- outcomes and strata with no component named.
        This runs HIP-LLM over the system's end-to-end correctness and stores it
        as :attr:`operational_failure`, the direct analogue of calling
        ``OperationalFailureProb`` yourself::

            study.observe(outcomes, strata, profile={"short": .3, "long": .7})

        ``1`` means the item was answered **correctly**, matching HIP-LLM.

        When a ``split`` column is present, only the calibration rows are used;
        fitting basic-event probabilities on the evaluation set would make every
        downstream number optimistic and untestable.
        """
        if profile is not None:
            self.profile = OperationalProfile.coerce(profile)
        if component_map:
            self._component_map.update(component_map)

        if outcomes is None:
            raise ValueError("observe() needs outcomes")

        if hasattr(outcomes, "columns"):
            return self._observe_frame(
                outcomes,
                stratum_column=stratum_column,
                component_columns=component_columns,
                split_column=split_column,
                calibration_split=calibration_split,
            )

        if strata is None:
            raise ValueError(
                "observe() needs stratum labels alongside the outcomes; pass "
                "strata=[...] or a table with a stratum column"
            )
        if self.profile is None:
            self.profile = empirical_profile(strata)
        if component is None:
            self._fit_system(outcomes, strata)
        else:
            self._observations[component] = (list(outcomes), [str(s) for s in strata])
        return self

    def _observe_frame(
        self,
        frame: Any,
        *,
        stratum_column: str,
        component_columns: Optional[Sequence[str]],
        split_column: Optional[str],
        calibration_split: str,
    ) -> "AgenticReliabilityStudy":
        if stratum_column not in frame.columns:
            raise KeyError(
                f"the table has no column {stratum_column!r}; pass "
                "stratum_column=... to say which column holds the stratum labels"
            )
        working = frame
        if split_column and split_column in frame.columns:
            values = {str(v).strip().lower() for v in frame[split_column].dropna()}
            if calibration_split in values:
                working = frame[
                    frame[split_column].astype(str).str.strip().str.lower()
                    == calibration_split
                ]
        columns = list(
            component_columns
            or [
                c
                for c in frame.columns
                if c not in {stratum_column, split_column, "item_id", "id"}
                and _looks_binary(frame[c])
            ]
        )
        if not columns:
            raise ValueError(
                "no per-component outcome columns were found. Pass "
                "component_columns=[...] naming the 1/0 correctness columns."
            )
        strata = [str(s) for s in working[stratum_column]]
        if self.profile is None:
            self.profile = empirical_profile(strata)
        for column in columns:
            self._observations[column] = (
                [int(v) for v in working[column]],
                strata,
            )
        return self

    def _fit_system(self, outcomes: Sequence[Any], strata: Sequence[str]) -> None:
        from HIPLLM import OperationalFailureProb, quick_inference_settings

        estimator = OperationalFailureProb(
            profile=dict(self.profile.items()),  # type: ignore[union-attr]
            settings=self.settings or quick_inference_settings(),
            credible_level=self.credible_level,
        )
        self.operational_failure = estimator.fit(
            outcomes=list(outcomes), strata=[str(s) for s in strata]
        )

    # ------------------------------------------------------------------ step 4
    def calibrate(self, **kwargs: Any) -> CalibrationReport:
        """Fit each observed component and write the intervals into the trees.

        Re-synthesises the trees afterwards, so cut-set quantification and the
        Bayesian network both see the calibrated numbers.
        """
        if not self._observations:
            raise StudyNotReady(
                "no outcomes have been recorded; call observe() first. Without "
                "measurement every probability in this study is a placeholder."
            )
        if self.profile is None:
            raise StudyNotReady(
                "no operational profile has been set; pass profile=... to the "
                "study or to observe(). Every failure probability is conditional "
                "on it, so it is never inferred silently."
            )
        if self.report is None:
            self.analyse()
        assert self.report is not None

        calibrator = EvidenceCalibrator(
            profile=self.profile,
            settings=self.settings,
            bound=self.bound,
            credible_level=self.credible_level,
            exact=self.exact_inference,
            **kwargs,
        )
        self.evidence = calibrator.fit_many(self._observations)
        self.calibration = calibrator.apply(
            self.report.failure_model,
            self.evidence,
            component_map=self._component_map or None,
        )
        self._resynthesise()
        return self.calibration

    def _resynthesise(self) -> None:
        """Rebuild trees and analyses from the (now calibrated) failure model."""
        from .faulttree.analysis import analyse_tree
        from .faulttree.synthesis import simplify_tree, synthesise_all

        assert self.report is not None
        model = self.report.failure_model
        raw = synthesise_all(model, self.report.hazards, simplify=False)
        trees = {hid: simplify_tree(tree) for hid, tree in raw.items()}
        self.report.raw_trees = raw
        self.report.trees = trees
        self.report.analyses = {
            hid: analyse_tree(tree, model) for hid, tree in trees.items()
        }

    # ------------------------------------------------------------------- run
    def run(self) -> "AgenticReliabilityStudy":
        """Analyse, and calibrate if outcomes were given.  Returns ``self``.

        This is the one call that makes the whole pipeline a chain::

            study = AgenticReliabilityStudy(graph).observe(table).run()
        """
        if self.report is None:
            self.analyse()
        if self._observations and self.calibration is None:
            self.calibrate()
        return self

    # ------------------------------------------------------------------ step 5
    def bayesnet(self, hazard: str = "H2", **kwargs: Any) -> BayesianNetwork:
        """The Bayesian network for one hazard, built from its fault tree."""
        report = self._require_report()
        tree = report.tree(hazard)
        return BayesianNetwork.from_fault_tree(
            tree,
            report.failure_model,
            name=f"{self.name} — {tree.id}",
            **kwargs,
        )

    def imprecise_bayesnet(
        self, hazard: str = "H2", **kwargs: Any
    ) -> ImpreciseBayesianNetwork:
        """The lower/upper network pair, when basic events carry intervals.

        Only meaningful after :meth:`calibrate`: without measurement the
        intervals are degenerate and both networks are the same.
        """
        report = self._require_report()
        tree = report.tree(hazard)
        return BayesianNetwork.imprecise_from_fault_tree(
            tree,
            report.failure_model,
            name=f"{self.name} — {tree.id}",
            **kwargs,
        )

    def cpts(self, hazard: str = "H2", **kwargs: Any):
        """The conditional probability tables for one hazard's fault tree."""
        from .bayes.cpt import fault_tree_to_cpts

        report = self._require_report()
        return fault_tree_to_cpts(
            report.tree(hazard), report.failure_model, **kwargs
        )

    def hazard_probability(self, hazard: str = "H2") -> Envelope:
        """``P(hazard)`` as an interval, exactly, by inference over the network."""
        return self.imprecise_bayesnet(hazard).envelope()

    # ----------------------------------------------------------------- access
    @property
    def system(self) -> SystemModel:
        return self._require_report().system

    @property
    def failure_model(self):
        return self._require_report().failure_model

    def hazards_found(self) -> List[str]:
        return sorted(self._require_report().trees)

    def cut_sets(self, hazard: str = "H2") -> List[List[str]]:
        return self._require_report().cut_sets(hazard)

    def single_points(self) -> List[Dict[str, str]]:
        return self._require_report().single_points()

    def fmea(self):
        import pandas as pd

        rows = [
            {
                "component": r.component,
                "event": r.event_id,
                "failure mode": r.failure_mode,
                "class": r.failure_class,
                "P": r.probability,
                "direct effects": "; ".join(r.direct_effects),
                "further effects": "; ".join(r.further_effects),
                "severity": r.max_severity,
                "mitigation": r.mitigation,
            }
            for r in self._require_report().fmea()
        ]
        return pd.DataFrame(rows)

    def _require_report(self) -> SafetyReport:
        if self.report is None:
            self.analyse()
        assert self.report is not None
        return self.report

    # ---------------------------------------------------------------- display
    def plot(self, hazard: str = "H2"):
        """Draw one fault tree with matplotlib."""
        from .viz.plots import plot_fault_tree

        return plot_fault_tree(self._require_report().tree(hazard))

    def plot_architecture(self):
        from .viz.plots import plot_architecture

        return plot_architecture(self.system, title=self.name)

    def plot_importance(self, hazard: str = "H2", top_n: int = 12):
        """Fussell-Vesely contribution per basic event, ranked — what to fix first."""
        from .viz.plots import plot_importance

        report = self._require_report()
        return plot_importance(report.analysis(hazard), top_n=top_n)

    def plot_cutset_orders(self):
        """Cut sets per order, per hazard — how much defence in depth exists.

        A tall order-1 bar is the finding: those are single points of failure.
        """
        from .viz.plots import plot_cutset_orders

        return plot_cutset_orders(self._require_report().analyses)

    def show(self, hazard: str = "H2", **kwargs: Any):
        """Draw the Bayesian network for a hazard."""
        return self.bayesnet(hazard).show(**kwargs)

    def summary(self) -> str:
        """Everything the study currently knows, in one printable block."""
        report = self._require_report()
        lines = [report.summary()]
        if self.profile is not None:
            lines.append("")
            lines.append(self.profile.summary())
        if self.calibration is not None:
            lines.append("")
            lines.append(self.calibration.summary())
        else:
            lines.append("")
            lines.append(
                "NOT CALIBRATED — every probability above is engineering "
                "judgement. Call observe() then calibrate() to replace them with "
                "measurement under an operational profile."
            )
        if self.operational_failure is not None:
            summary = self.operational_failure.summary()
            lines.append("")
            lines.append("System-level operational failure probability")
            for key, value in summary.items():
                if isinstance(value, float):
                    lines.append(f"  {key:<48} {value:.6f}")
        return "\n".join(lines)

    def save(self, directory: str, prefix: Optional[str] = None) -> List[str]:
        """Write the report, every tree export, the cut sets and the FMEA."""
        written = self._require_report().save(directory, prefix=prefix)
        if self.calibration is not None:
            path = os.path.join(directory, "calibration.csv")
            self.calibration.to_frame().to_csv(path, index=False)
            written.append(path)
            path = os.path.join(directory, "evidence.csv")
            self.calibration.evidence_frame().to_csv(path, index=False)
            written.append(path)
        return written

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "analysed" if self.report is not None else "not analysed"
        if self.calibration is not None:
            state = "calibrated"
        return f"AgenticReliabilityStudy(name={self.name!r}, {state})"


def _looks_binary(column: Any) -> bool:
    try:
        values = set(int(v) for v in column.dropna().unique())
    except (TypeError, ValueError):
        return False
    return values <= {0, 1}
