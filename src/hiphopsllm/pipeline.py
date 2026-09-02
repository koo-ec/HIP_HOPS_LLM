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
from .reliability.profile import (
    OperationalProfile,
    dataset_proportional_profile,
)
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
            self.profile = dataset_proportional_profile(strata)
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
                if c not in {stratum_column, split_column, "item_id", "id",
                             "run_error"}
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
            self.profile = dataset_proportional_profile(strata)
            print(
                "NO OPERATIONAL PROFILE GIVEN. Falling back to the benchmark's\n"
                "own mix of strata, which asserts that your deployed workload\n"
                "looks like your test set. That assumption is the thing HIP-LLM\n"
                "exists to remove, and every probability below is conditional on\n"
                "it. Pass profile={...} with the mix you actually expect."
            )
        for column in columns:
            # A blank cell means "this node was not exercised on that item" —
            # a router sent the run to END before reaching it, say. That is a
            # *missing observation*, not a success and not a failure, so the row
            # is dropped for this component only and still counts for the others.
            values, kept = [], []
            for outcome, stratum in zip(working[column], strata):
                if outcome is None or (isinstance(outcome, float) and outcome != outcome):
                    continue
                values.append(int(outcome))
                kept.append(stratum)
            if not values:
                raise ValueError(
                    f"column {column!r} has no observations at all; every run "
                    "left this node unexercised, so nothing can be estimated "
                    "for it"
                )
            self._observations[column] = (
                values,
                kept,
            )
        return self

    # ------------------------------------------------------------ steps 3 + 4
    def run_and_observe(
        self,
        inputs: Sequence[Any],
        success: Mapping[str, Callable[[Any], Any]],
        *,
        stratum: Optional[Callable[[Any], str] | Sequence[str]] = None,
        profile: Optional[OperationalProfile | Mapping[str, float]] = None,
        invoke: Optional[Callable[[Any], Any]] = None,
        calibration_fraction: float = 0.75,
        on_error: str = "skip",
        progress: bool = True,
        calibrate: bool = True,
    ) -> Any:
        """Run the graph over ``inputs``, score every node, and calibrate.

        This is the one-cell entry point for a notebook that already builds and
        runs a LangGraph application: append this and the whole reliability
        analysis follows from the runs it performs.

        ::

            study = AgenticReliabilityStudy(graph, globals_ns=globals())
            study.run_and_observe(
                inputs=[{"smiles": s} for s in SMILES],
                stratum=lambda item: "large" if len(item["smiles"]) > 40 else "small",
                success={
                    "research_agent": lambda s: s.get("browser_status") == "ready",
                    "pixelrag_agent": lambda s: s.get("capture_status") == "captured",
                    "safety_agent":   lambda s: bool(s.get("workflow_succeeded")),
                },
                profile={"small": 0.6, "large": 0.4},
            )

        Parameters
        ----------
        inputs
            One graph input per benchmark item.  Each is passed to
            ``graph.invoke`` (or to ``invoke``).
        success
            ``{node_id: predicate}``.  Each predicate receives the **final
            state** and returns ``True`` (that node did its job), ``False`` (it
            did not), or ``None``.

            ``None`` means *not exercised* — a router sent the run to ``END``
            before this node ran — and is recorded as a missing observation
            rather than a failure.  That distinction matters: scoring an
            unreached node as failed would blame it for an upstream fault.
        stratum
            A callable mapping an input to its stratum label, or a ready-made
            sequence of labels, one per input.  Defaults to a single stratum,
            which is honest but gives up the whole point of an operational
            profile — pass one if the workload is not uniform.
        profile
            The operational profile.  Defaults to the observed frequencies of
            the stratum labels, recorded as such.
        invoke
            Custom runner, for a graph that needs streaming or a config.
            Receives one input and must return the final state.
        calibration_fraction
            Fraction of items used to fit basic-event probabilities; the rest
            are marked ``test`` and left out, so a held-out set survives.
        on_error
            What to do when a run raises.  ``"skip"`` (the default) drops the
            item and reports how many were dropped; ``"record"`` keeps the row
            with every node unscored and a ``run_error`` message.

            Neither blames a node. A crashed run leaves no state to score
            against, so marking its nodes as failures would penalise components
            that had already succeeded before the exception — the same mistake
            as scoring an unreached node. What is lost either way is visible:
            the count is printed, and ``"record"`` keeps the message.
        calibrate
            Run :meth:`calibrate` when the runs finish.  ``False`` stops after
            recording the outcomes.

        Returns
        -------
        pandas.DataFrame
            One row per item: ``item_id``, ``stratum``, one column per scored
            node, and ``split``.  Keep it — it is the measurement, and
            re-running the graph is the expensive part.
        """
        import pandas as pd

        if not inputs:
            raise ValueError("run_and_observe() needs at least one input")
        if not success:
            raise ValueError(
                "success must map node ids to predicates; without it there is "
                "nothing to measure. Each predicate takes the final state and "
                "returns True, False, or None for 'not exercised'."
            )
        if on_error not in ("record", "skip"):
            raise ValueError("on_error must be 'skip' or 'record'")

        runner = invoke or getattr(self.graph, "invoke", None)
        if runner is None:
            raise TypeError(
                "this study was built from a specification, not a runnable "
                "graph, so there is nothing to run. Pass invoke=... or "
                "construct the study from a compiled LangGraph."
            )

        if stratum is None:
            labels = ["all"] * len(inputs)
        elif callable(stratum):
            labels = [str(stratum(item)) for item in inputs]
        else:
            labels = [str(s) for s in stratum]
            if len(labels) != len(inputs):
                raise ValueError(
                    f"{len(labels)} stratum labels for {len(inputs)} inputs"
                )

        n_calibration = max(1, int(round(len(inputs) * calibration_fraction)))
        rows, skipped = [], []
        for index, item in enumerate(inputs):
            error = ""
            state = None
            try:
                state = runner(item)
            except Exception as exc:  # noqa: BLE001 - a run failing is data
                error = f"{type(exc).__name__}: {exc}"
                skipped.append((index, error))
                if on_error == "skip":
                    continue

            row = {"item_id": f"item_{index:04d}", "stratum": labels[index]}
            for node, predicate in success.items():
                if state is None:
                    # Nothing to score against; see `on_error` in the docstring.
                    row[node] = None
                    continue
                try:
                    verdict = predicate(state)
                except Exception:  # noqa: BLE001 - a predicate that cannot decide
                    verdict = None
                row[node] = None if verdict is None else int(bool(verdict))
            if on_error == "record":
                row["run_error"] = error
            row["split"] = "calibration" if index < n_calibration else "test"
            rows.append(row)

            if progress:
                scored = {k: row[k] for k in success}
                print(f"  {index + 1:>3}/{len(inputs)}  {labels[index]:<10} {scored}")

        if not rows:
            raise RuntimeError(
                "every run failed and on_error='skip', so nothing was measured"
            )
        if skipped:
            verb = "skipped" if on_error == "skip" else "recorded unscored"
            share = len(skipped) / len(inputs)
            print(
                f"\n{len(skipped)} of {len(inputs)} run(s) raised and were "
                f"{verb} ({share:.0%}):"
            )
            for index, message in skipped[:5]:
                print(f"  item {index}: {message}")
            if share > 0.1:
                print(
                    "  A tenth or more of the runs crashed. Those failures are "
                    "real and are NOT in the numbers below, so the estimate is "
                    "optimistic by an unknown amount — fix them first."
                )

        outcomes = pd.DataFrame(rows)
        self.observe(outcomes, profile=profile)
        if calibrate:
            self.run()
        return outcomes

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

    def operational_reliability(self, n_tasks: int = 1) -> str:
        """The measured claim, stated the way HIP-LLM defines it.

        This is the *measurement*: for each component, the probability that it
        fails on one task drawn from the stated operational profile, and the
        probability of failure-free operation over ``n_tasks``.

        It is deliberately separate from the fault tree. The tree decomposes
        each of these numbers over a component's internal failure modes so that
        it can be *propagated* through the architecture; that decomposition is a
        modelling step, not something that was observed. What was observed is
        here: a task either succeeded or it did not, whatever the reason.
        """
        if not self.evidence:
            raise StudyNotReady(
                "nothing has been measured yet; call observe() or "
                "run_and_observe() first"
            )
        assert self.profile is not None

        def fmt(value: float) -> str:
            """Four decimals normally; scientific once that would round to zero."""
            if value == 0.0 or value >= 1e-4:
                return f"{value:8.4f}"
            return f"{value:8.2e}"

        horizon = f"R({n_tasks} task{'s' if n_tasks != 1 else ''})"
        header = f"{'component':<20}{'P(fail per task)':>22}"
        if n_tasks > 1:
            header += f"{horizon:>22}"
        header += f"{'n':>7}"

        lines = [
            "Measured reliability under the operational profile",
            "=" * 50,
            "",
            self.profile.summary(),
            "",
            header,
            "-" * len(header),
        ]
        for name, evidence in self.evidence.items():
            low, high = evidence.interval
            row = f"{name:<20}[{fmt(low)},{fmt(high)}]"
            if n_tasks > 1:
                r_low, r_high = evidence.reliability(n_tasks)
                row += f" [{fmt(r_low)},{fmt(r_high)}]"
            lines.append(f"{row}  {evidence.n_trials:>5}")
        lines.append("")
        note = (
            "Each interval is a posterior envelope, not a confidence interval: it\n"
            "spans the admissible hyperparameter set as well as the sampling\n"
            "uncertainty."
        )
        if n_tasks > 1:
            note += (
                "\nR(n) is the probability of failure-free operation over n future\n"
                "tasks — HIP-LLM's definition of reliability — computed as E[p^n]\n"
                "per configuration and then enveloped, never as E[p]^n."
            )
        lines.append(note)
        method = next(iter(self.evidence.values())).method
        lines.append(f"\nmethod: {method}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Everything the study currently knows, in one printable block."""
        report = self._require_report()
        lines = [report.summary()]
        if self.profile is not None and self.calibration is None:
            lines.append("")
            lines.append(self.profile.summary())
        if self.calibration is not None:
            lines.append("")
            lines.append(self.operational_reliability(n_tasks=10))
            lines.append("")
            lines.append(
                "The fault tree below decomposes each measured probability over\n"
                "that component's failure modes so it can be propagated through\n"
                "the architecture. That split is a modelling step; the\n"
                "measurement is the table above."
            )
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
