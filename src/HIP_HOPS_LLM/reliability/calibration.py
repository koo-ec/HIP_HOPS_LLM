"""Turning measured outcomes into fault-tree basic-event probabilities.

This is the join between the two halves of the package.

HiP-HOPS gives the *structure*: which combinations of component faults produce a
system-level hazard.  Its basic-event probabilities, though, arrive as
engineering judgement --- ``llm.halluc = 0.15`` and so on --- which is fine for
ranking cut sets and useless for a reliability claim.

HIP-LLM gives the *numbers*: a hierarchical imprecise-Bayesian posterior for the
probability that a future item fails, given observed outcomes and an explicit
operational profile.  Crucially it returns an interval, because a few hundred
benchmark items do not identify a point.

:class:`EvidenceCalibrator` runs the second and writes its answer into the first.
The result is a fault tree whose leaves carry measured intervals with recorded
provenance, and therefore a top event that can be quoted as
``P(hazard) ∈ [lower, upper]`` rather than as a number nobody can defend.

Two design decisions worth stating.

**Where a component's measured failure rate goes.**  An LLM agent's subtle-value
deviation is ``VS-out = BE-HALLUC OR BE-NONDET OR VS-in``.  A measured
end-to-end wrong-answer rate for that agent, on correct input, is the probability
of the *union* of its internal events --- not of any one of them.  The default
policy therefore rescales the whole set so their OR reproduces the measurement
while their prior ratios are preserved: with weights ``w_i`` summing to one,
``p_i = 1 - (1 - P)**w_i``, which satisfies ``1 - prod(1 - p_i) = P`` exactly.

**Nothing is calibrated silently.**  Every event this touches gets an
``evidence`` string naming the sample size and the profile, and every component
the calibrator could *not* match is reported.  A network quietly running on
default priors is the specific failure this module exists to prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..faulttree.failure import FailureModel, FClass
from .profile import OperationalProfile

__all__ = [
    "ComponentEvidence",
    "CalibrationReport",
    "EvidenceCalibrator",
    "calibrate_failure_model",
    "distribute_union",
]

#: Basic-event kinds a measured *wrong answer* rate legitimately speaks about.
#: A measured accuracy says nothing about latency or unsafe execution, so those
#: keep their placeholders and are reported as uncalibrated.
VALUE_CLASSES = (FClass.VALUE_SUBTLE, FClass.VALUE_COARSE)

_GENERIC_TOKENS = {
    "agent", "node", "approach", "step", "llm", "model", "graph", "the", "a",
}


# --------------------------------------------------------------------------- #
# Distributing a union probability over its terms
# --------------------------------------------------------------------------- #
def distribute_union(
    target: float, weights: Sequence[float], mode: str = "share"
) -> np.ndarray:
    """Split a union probability over independent events.

    Given ``P(E1 or ... or En) = target`` and prior weights ``w_i``, return
    probabilities ``p_i`` with ``1 - prod(1 - p_i) = target``.

    ``mode="share"`` uses ``p_i = 1 - (1 - target) ** (w_i / sum(w))``, which is
    exact and keeps the events in roughly their prior proportions.
    ``mode="dominant"`` puts the whole probability on the heaviest event and
    leaves the rest at zero --- blunter, but easier to defend when only one of
    the events is really what was measured.
    """
    w = np.asarray(list(weights), dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty 1-D sequence")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    p = float(target)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"target probability must lie in [0, 1], got {p!r}")
    total = w.sum()
    if total <= 0:
        w = np.ones_like(w)
        total = w.sum()

    if mode == "dominant":
        out = np.zeros_like(w)
        out[int(np.argmax(w))] = p
        return out
    if mode != "share":
        raise ValueError("mode must be 'share' or 'dominant'")
    if p >= 1.0:
        return np.ones_like(w)
    share = w / total
    return 1.0 - np.power(1.0 - p, share)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class ComponentEvidence:
    """What was measured about one component, and what it implies.

    ``interval`` is the imprecise posterior for the probability that this
    component fails on an item drawn from the operational profile.  It is the
    number that goes into the fault tree.
    """

    component: str
    n_trials: int
    n_failures: int
    empirical: float
    point: float
    interval: Tuple[float, float]
    profile: OperationalProfile
    #: per-stratum observed failure rate, for the diagnostic table
    by_stratum: Dict[str, float] = field(default_factory=dict)
    #: the raw HIPLLM result, when the full inference was run
    posterior: Any = field(default=None, repr=False)
    method: str = "hip-llm imprecise posterior"

    @property
    def width(self) -> float:
        return self.interval[1] - self.interval[0]

    def evidence_string(self) -> str:
        return (
            f"{self.method}: {self.n_failures}/{self.n_trials} observed failures "
            f"under the profile {dict((k, round(v, 3)) for k, v in self.profile.items())}; "
            f"P ∈ [{self.interval[0]:.4f}, {self.interval[1]:.4f}]"
        )

    def summary(self) -> str:
        return (
            f"{self.component:<24} n={self.n_trials:<5} "
            f"empirical={self.empirical:.4f}  "
            f"P=[{self.interval[0]:.4f}, {self.interval[1]:.4f}]  "
            f"(width {self.width:.4f})"
        )


@dataclass
class CalibrationReport:
    """Which events were calibrated, which were not, and why."""

    evidence: Dict[str, ComponentEvidence] = field(default_factory=dict)
    #: basic event id -> (old probability, new probability, new interval)
    updated: Dict[str, Tuple[float, float, Tuple[float, float]]] = field(
        default_factory=dict
    )
    #: components in the model that no measurement matched
    uncalibrated_components: List[str] = field(default_factory=list)
    #: measurement keys that matched no component
    unmatched_evidence: List[str] = field(default_factory=list)
    #: basic events left on placeholders, with the reason
    skipped_events: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def n_updated(self) -> int:
        return len(self.updated)

    def to_frame(self):
        """The before/after table, for a notebook."""
        import pandas as pd

        rows = [
            {
                "basic event": eid,
                "placeholder P": round(old, 5),
                "calibrated P": round(new, 5),
                "lower": round(interval[0], 5),
                "upper": round(interval[1], 5),
            }
            for eid, (old, new, interval) in sorted(self.updated.items())
        ]
        return pd.DataFrame(rows)

    def evidence_frame(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "component": e.component,
                    "n": e.n_trials,
                    "failures": e.n_failures,
                    "empirical": round(e.empirical, 5),
                    "lower": round(e.interval[0], 5),
                    "upper": round(e.interval[1], 5),
                    "width": round(e.width, 5),
                }
                for e in self.evidence.values()
            ]
        )

    def summary(self) -> str:
        lines = ["Calibration", "-" * 11]
        for e in self.evidence.values():
            lines.append("  " + e.summary())
        lines.append(f"  {self.n_updated} basic event(s) updated from measurement")
        if self.uncalibrated_components:
            lines.append(
                "  no measurement matched: "
                + ", ".join(sorted(self.uncalibrated_components))
            )
        if self.unmatched_evidence:
            lines.append(
                "  measurement matched no component: "
                + ", ".join(sorted(self.unmatched_evidence))
            )
        if self.skipped_events:
            kinds = sorted({v for v in self.skipped_events.values()})
            lines.append(
                f"  {len(self.skipped_events)} event(s) kept their placeholder "
                f"({'; '.join(kinds)})"
            )
        lines.extend("  " + n for n in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The calibrator
# --------------------------------------------------------------------------- #
@dataclass
class EvidenceCalibrator:
    """Fit operational failure probabilities and write them into a fault tree.

    Parameters
    ----------
    profile
        The operational profile every estimate is conditional on.
    settings
        HIP-LLM inference settings.  Defaults to
        :func:`HIPLLM.quick_inference_settings`, which is interactive-speed;
        pass :func:`HIPLLM.paper_inference_settings` for published sizes.
    bound
        Which HIP-LLM envelope becomes the basic-event interval.  ``"credible"``
        (the default) uses the outer equal-tail credible envelope across
        hyperparameter configurations --- the widest, most cautious reading.
        ``"expected"`` uses the envelope of posterior means, which is tighter
        and is the right choice when the interval feeds a design decision rather
        than a safety argument.
    point
        Where the point estimate inside the interval comes from: ``"midpoint"``,
        ``"lower"``, ``"upper"``, or ``"empirical"``.
    policy
        How a component's measured probability is spread over its basic events;
        see :func:`distribute_union`.
    classes
        Failure classes a wrong-answer measurement is allowed to speak about.
        Latency and unsafe-execution events keep their placeholders --- a
        correctness benchmark says nothing about them.
    exact
        Run the full HIP-LLM hierarchical inference.  ``False`` substitutes a
        Jeffreys-interval approximation, which is far faster and adequate for
        exploration; it is labelled as such in the evidence string, so it can
        never be mistaken for the real thing in a report.
    """

    profile: OperationalProfile
    settings: Any = None
    bound: str = "credible"
    point: str = "midpoint"
    policy: str = "share"
    classes: Tuple[FClass, ...] = VALUE_CLASSES
    credible_level: float = 0.95
    exact: bool = True

    def __post_init__(self) -> None:
        self.profile = OperationalProfile.coerce(self.profile)
        if self.bound not in ("credible", "expected", "median"):
            raise ValueError("bound must be 'credible', 'expected' or 'median'")
        if self.point not in ("midpoint", "lower", "upper", "empirical"):
            raise ValueError(
                "point must be 'midpoint', 'lower', 'upper' or 'empirical'"
            )

    # -- measurement --------------------------------------------------------- #
    def fit_component(
        self,
        component: str,
        outcomes: Sequence[int | bool],
        strata: Sequence[str],
    ) -> ComponentEvidence:
        """Estimate one component's operational failure probability.

        ``outcomes[k]`` is ``1``/``True`` when the component answered item ``k``
        **correctly**, matching HIP-LLM's convention; ``strata[k]`` names the
        operational stratum of the same item.
        """
        y = np.asarray(list(outcomes))
        s = [str(x) for x in strata]
        if y.size == 0:
            raise ValueError(f"{component}: at least one outcome is required")
        if y.size != len(s):
            raise ValueError(
                f"{component}: {y.size} outcomes but {len(s)} stratum labels"
            )
        unknown = sorted(set(s) - set(self.profile.labels))
        if unknown:
            raise ValueError(
                f"{component}: outcomes fall in stratum(s) {unknown}, outside the "
                f"operational profile {list(self.profile.labels)}"
            )

        correct = y.astype(int)
        n_failures = int((correct == 0).sum())
        by_stratum = {
            label: float(1.0 - correct[[i for i, x in enumerate(s) if x == label]].mean())
            for label in self.profile.labels
            if label in set(s)
        }
        empirical = (
            self.profile.expected(by_stratum)
            if len(by_stratum) == len(self.profile)
            else float(n_failures / correct.size)
        )

        if self.exact:
            interval, posterior, method = self._hipllm_interval(correct, s)
        else:
            note = "Jeffreys interval (approximation; not the HIP-LLM posterior"
            if self.bound != "credible":
                # The three envelopes are a property of HIP-LLM's hyperparameter
                # family; the approximation has only one. Saying so beats
                # accepting the option and ignoring it.
                note += f"; bound={self.bound!r} does not apply to it"
            interval, posterior, method = (
                self._jeffreys_interval(correct, s),
                None,
                note + ")",
            )

        return ComponentEvidence(
            component=component,
            n_trials=int(correct.size),
            n_failures=n_failures,
            empirical=empirical,
            point=self._point_estimate(interval, empirical),
            interval=interval,
            profile=self.profile,
            by_stratum=by_stratum,
            posterior=posterior,
            method=method,
        )

    def fit_many(
        self, observations: Mapping[str, Any]
    ) -> Dict[str, ComponentEvidence]:
        """Fit several components at once.

        ``observations`` maps a component name to ``(outcomes, strata)``, or to a
        mapping with ``outcomes`` and ``strata`` keys::

            calibrator.fit_many({
                "react_agent": (react_outcomes, strata),
                "cot_agent":   (cot_outcomes, strata),
            })
        """
        out: Dict[str, ComponentEvidence] = {}
        for component, spec in observations.items():
            if isinstance(spec, Mapping):
                outcomes, strata = spec["outcomes"], spec["strata"]
            else:
                outcomes, strata = spec
            out[component] = self.fit_component(component, outcomes, strata)
        return out

    # -- writing into the model ---------------------------------------------- #
    def apply(
        self,
        failure_model: FailureModel,
        evidence: Mapping[str, ComponentEvidence],
        *,
        component_map: Optional[Mapping[str, str]] = None,
        in_place: bool = True,
    ) -> CalibrationReport:
        """Write measured intervals onto the model's basic events.

        ``component_map`` forces a measurement key onto a component id when the
        automatic name matching cannot be trusted.  Everything the matcher does
        or fails to do is recorded in the returned :class:`CalibrationReport`.
        """
        report = CalibrationReport(evidence=dict(evidence))
        assignments = self._match(failure_model, evidence, component_map)
        report.unmatched_evidence = sorted(
            set(evidence) - {k for k in assignments.values()}
        )

        touched_components = set()
        for component_id, key in assignments.items():
            measurement = evidence[key]
            events = [
                e
                for e in failure_model.events.values()
                if e.component == component_id
                and e.kind == "internal"
                and e.fclass in self.classes
            ]
            if not events:
                report.skipped_events[component_id] = (
                    f"no internal {'/'.join(c.value for c in self.classes)} basic "
                    "event to carry the measurement"
                )
                continue
            touched_components.add(component_id)
            # Weight by the *pre-calibration* prior, recorded once. Using the
            # current probability would make a second calibration split an
            # already-split value, silently moving numbers that were already
            # measured — see test_recalibrating_does_not_compound.
            for event in events:
                if event.baseline_prob is None:
                    event.baseline_prob = float(event.prob)
            weights = [max(e.baseline_prob, 1e-9) for e in events]
            lo = distribute_union(measurement.interval[0], weights, self.policy)
            hi = distribute_union(measurement.interval[1], weights, self.policy)
            mid = distribute_union(measurement.point, weights, self.policy)
            for event, p_lo, p_hi, p_mid in zip(events, lo, hi, mid):
                old = float(event.baseline_prob if event.baseline_prob is not None
                            else event.prob)
                event.prob = float(p_mid)
                event.prob_interval = (float(p_lo), float(p_hi))
                event.evidence = measurement.evidence_string()
                report.updated[event.id] = (old, event.prob, event.prob_interval)

        for event in failure_model.events.values():
            if event.id in report.updated:
                continue
            if event.component in touched_components:
                report.skipped_events[event.id] = (
                    f"class {event.fclass.value} is outside what a correctness "
                    "measurement can speak about"
                )

        analysed = {
            c for c in failure_model.system.components if not _is_boundary(failure_model, c)
        }
        report.uncalibrated_components = sorted(analysed - touched_components)
        if report.updated:
            failure_model.notes.append(
                f"{len(report.updated)} basic event(s) were calibrated from "
                f"measured outcomes under the operational profile "
                f"{dict((k, round(v, 3)) for k, v in self.profile.items())}."
            )
        if report.uncalibrated_components:
            failure_model.notes.append(
                "These components kept placeholder probabilities: "
                + ", ".join(report.uncalibrated_components)
                + ". Their contribution to every quantified result is engineering "
                "judgement, not measurement."
            )
        return report

    # -- internals ----------------------------------------------------------- #
    def _hipllm_interval(
        self, correct: np.ndarray, strata: Sequence[str]
    ) -> Tuple[Tuple[float, float], Any, str]:
        from HIPLLM import OperationalFailureProb, quick_inference_settings

        estimator = OperationalFailureProb(
            profile=dict(self.profile.items()),
            settings=self.settings or quick_inference_settings(),
            credible_level=self.credible_level,
        )
        result = estimator.fit(outcomes=correct.tolist(), strata=list(strata))
        if self.bound == "credible":
            interval = result.posterior_credible_envelope
        elif self.bound == "expected":
            interval = result.posterior_expected_failure_bounds
        else:
            interval = result.posterior_median_failure_bounds
        lo, hi = float(min(interval)), float(max(interval))
        return (lo, hi), result, (
            f"HIP-LLM hierarchical imprecise posterior ({self.bound} envelope, "
            f"level {self.credible_level:.2f})"
        )

    def _jeffreys_interval(
        self, correct: np.ndarray, strata: Sequence[str]
    ) -> Tuple[float, float]:
        """Profile-weighted Jeffreys interval --- the fast approximation."""
        from scipy.stats import beta

        alpha = 1.0 - self.credible_level
        lows: Dict[str, float] = {}
        highs: Dict[str, float] = {}
        for label in self.profile.labels:
            idx = [i for i, s in enumerate(strata) if s == label]
            if not idx:
                # No data in a stratum that carries weight: the interval there is
                # the whole unit interval, which is the honest answer.
                lows[label], highs[label] = 0.0, 1.0
                continue
            n = len(idx)
            failures = int((correct[idx] == 0).sum())
            lows[label] = (
                0.0 if failures == 0 else float(beta.ppf(alpha / 2, failures + 0.5, n - failures + 0.5))
            )
            highs[label] = (
                1.0 if failures == n else float(beta.ppf(1 - alpha / 2, failures + 0.5, n - failures + 0.5))
            )
        return (self.profile.expected(lows), self.profile.expected(highs))

    def _point_estimate(
        self, interval: Tuple[float, float], empirical: float
    ) -> float:
        lo, hi = interval
        if self.point == "lower":
            return lo
        if self.point == "upper":
            return hi
        if self.point == "empirical":
            return float(min(max(empirical, lo), hi))
        return 0.5 * (lo + hi)

    def _match(
        self,
        failure_model: FailureModel,
        evidence: Mapping[str, ComponentEvidence],
        component_map: Optional[Mapping[str, str]],
    ) -> Dict[str, str]:
        """``{component_id: evidence_key}`` by explicit map, then name tokens."""
        assignments: Dict[str, str] = {}
        components = list(failure_model.system.components)
        if component_map:
            for key, component_id in component_map.items():
                if component_id not in failure_model.system.components:
                    raise KeyError(
                        f"component_map points {key!r} at {component_id!r}, which is "
                        f"not in the model. Components are: {sorted(components)}"
                    )
                if key not in evidence:
                    raise KeyError(f"component_map names measurement {key!r}, which was not fitted")
                assignments[component_id] = key

        for component_id in components:
            if component_id in assignments:
                continue
            base = component_id.split("#")[0].replace("::router", "")
            tokens = _tokens(base) - _GENERIC_TOKENS
            if not tokens:
                continue
            best, best_score = None, 0
            for key in evidence:
                score = len(tokens & (_tokens(key) - _GENERIC_TOKENS))
                if score > best_score:
                    best, best_score = key, score
            if best is not None and best_score > 0:
                assignments[component_id] = best
        return assignments


def _tokens(text: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t and not t.isdigit()}


def _is_boundary(failure_model: FailureModel, component_id: str) -> bool:
    component = failure_model.system.components.get(component_id)
    return bool(component and component.is_boundary)


def calibrate_failure_model(
    failure_model: FailureModel,
    observations: Mapping[str, Any],
    profile: OperationalProfile | Mapping[str, float],
    **kwargs: Any,
) -> CalibrationReport:
    """Fit and apply in one call.

    ::

        report = calibrate_failure_model(
            model.failure_model,
            {"react_agent": (outcomes_a, strata), "cot_agent": (outcomes_b, strata)},
            profile={"short": 0.3, "long": 0.7},
        )
        print(report.summary())
    """
    calibrator = EvidenceCalibrator(
        profile=OperationalProfile.coerce(profile), **kwargs
    )
    evidence = calibrator.fit_many(observations)
    return calibrator.apply(failure_model, evidence)
