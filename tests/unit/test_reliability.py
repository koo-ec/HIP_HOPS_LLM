"""Operational profiles, evidence calibration, and the union-splitting rule."""

from __future__ import annotations

import numpy as np
import pytest

from HIP_HOPS_LLM import (
    EvidenceCalibrator,
    OperationalProfile,
    calibrate_failure_model,
    distribute_union,
    empirical_profile,
    stratify,
    uniform_profile,
)

PROFILE = {"short": 0.30, "medium": 0.50, "long": 0.20}


class TestOperationalProfile:
    def test_weights_are_normalised(self):
        profile = OperationalProfile({"a": 2.0, "b": 2.0})
        assert profile["a"] == pytest.approx(0.5)
        assert sum(profile.weights.values()) == pytest.approx(1.0)

    def test_normalisation_can_be_refused(self):
        with pytest.raises(ValueError, match="not 1.0"):
            OperationalProfile({"a": 0.2, "b": 0.2}, normalise=False)

    def test_an_empty_profile_is_rejected(self):
        with pytest.raises(ValueError, match="at least one stratum"):
            OperationalProfile({})

    def test_negative_weights_are_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            OperationalProfile({"a": -0.5, "b": 1.5})

    def test_expected_is_the_weighted_average(self):
        profile = OperationalProfile(PROFILE)
        assert profile.expected({"short": 0.0, "medium": 1.0, "long": 0.0}) == (
            pytest.approx(0.5)
        )

    def test_a_missing_stratum_is_an_error_not_a_silent_reweighting(self):
        profile = OperationalProfile(PROFILE)
        with pytest.raises(KeyError, match="no value for stratum"):
            profile.expected({"short": 0.1, "medium": 0.2})

    def test_empirical_profile_counts_frequencies(self):
        profile = empirical_profile(["a", "a", "a", "b"])
        assert profile["a"] == pytest.approx(0.75)
        assert "observed frequencies" in profile.provenance

    def test_uniform_profile_records_that_it_is_an_assumption(self):
        profile = uniform_profile(["a", "b", "c", "d"])
        assert profile["a"] == pytest.approx(0.25)
        assert "assumed" in profile.provenance

    def test_restricted_renormalises(self):
        profile = OperationalProfile(PROFILE).restricted_to(["short", "long"])
        assert sum(profile.weights.values()) == pytest.approx(1.0)
        assert profile["short"] == pytest.approx(0.3 / 0.5)

    def test_coerce_accepts_a_mapping_and_a_hipllm_profile(self):
        mapping = OperationalProfile.coerce(PROFILE)
        assert isinstance(mapping, OperationalProfile)
        roundtrip = OperationalProfile.coerce(mapping.to_hipllm())
        assert roundtrip.labels == mapping.labels
        assert np.allclose(roundtrip.vector, mapping.vector)

    def test_stratify_rejects_a_label_outside_the_profile(self):
        profile = OperationalProfile({"short": 1.0})
        with pytest.raises(ValueError, match="does not cover"):
            stratify([{"k": "long"}], "k", profile=profile)

    def test_stratify_accepts_a_callable(self):
        items = [{"n": 3}, {"n": 40}]
        labels = stratify(items, lambda i: "short" if i["n"] < 10 else "long")
        assert labels == ["short", "long"]

    def test_summary_names_its_provenance(self):
        assert "provenance" in OperationalProfile(PROFILE).summary()


class TestDistributeUnion:
    @pytest.mark.parametrize("target", [0.0, 0.05, 0.3, 0.75, 0.99])
    @pytest.mark.parametrize("weights", [[1.0], [1.0, 1.0], [0.15, 0.12, 0.01]])
    def test_the_union_reproduces_the_target_exactly(self, target, weights):
        p = distribute_union(target, weights)
        assert float(1.0 - np.prod(1.0 - p)) == pytest.approx(target, abs=1e-12)

    def test_shares_follow_the_prior_ratios(self):
        p = distribute_union(0.2, [0.3, 0.1])
        assert p[0] > p[1]

    def test_dominant_puts_everything_on_the_heaviest(self):
        p = distribute_union(0.4, [0.3, 0.1], mode="dominant")
        assert p[0] == pytest.approx(0.4)
        assert p[1] == pytest.approx(0.0)

    def test_certainty_saturates_every_term(self):
        assert np.allclose(distribute_union(1.0, [0.2, 0.8]), 1.0)

    def test_zero_weights_fall_back_to_equal_shares(self):
        p = distribute_union(0.3, [0.0, 0.0])
        assert p[0] == pytest.approx(p[1])

    def test_an_out_of_range_target_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            distribute_union(1.2, [1.0])

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="'share' or 'dominant'"):
            distribute_union(0.2, [1.0], mode="nope")


class TestEvidenceCalibrator:
    @pytest.fixture(scope="class")
    def calibrator(self):
        return EvidenceCalibrator(profile=PROFILE, exact=False)

    def test_fit_returns_an_interval_around_the_empirical_rate(self, calibrator):
        outcomes = [1] * 80 + [0] * 20
        strata = (["short"] * 30 + ["medium"] * 50 + ["long"] * 20)
        evidence = calibrator.fit_component("a", outcomes, strata)
        assert evidence.n_trials == 100
        assert evidence.n_failures == 20
        assert evidence.interval[0] <= evidence.point <= evidence.interval[1]
        assert evidence.width > 0

    def test_more_data_narrows_the_interval(self, calibrator):
        small = calibrator.fit_component(
            "a", [1, 0] * 15, ["short", "medium"] * 15
        )
        large = calibrator.fit_component(
            "a", [1, 0] * 300, ["short", "medium"] * 300
        )
        assert large.width < small.width

    def test_a_stratum_outside_the_profile_is_refused(self, calibrator):
        with pytest.raises(ValueError, match="outside the operational profile"):
            calibrator.fit_component("a", [1, 0], ["short", "enormous"])

    def test_mismatched_lengths_are_refused(self, calibrator):
        with pytest.raises(ValueError, match="stratum labels"):
            calibrator.fit_component("a", [1, 0, 1], ["short", "medium"])

    def test_empty_outcomes_are_refused(self, calibrator):
        with pytest.raises(ValueError, match="at least one outcome"):
            calibrator.fit_component("a", [], [])

    def test_the_exact_path_is_labelled_as_hip_llm(self):
        exact = EvidenceCalibrator(profile=PROFILE, exact=True)
        evidence = exact.fit_component(
            "a", [1, 1, 0, 1] * 10, ["short", "medium", "long", "short"] * 10
        )
        assert "HIP-LLM" in evidence.method

    def test_the_approximation_says_so(self, calibrator):
        evidence = calibrator.fit_component("a", [1, 0], ["short", "long"])
        assert "approximation" in evidence.method.lower()

    def test_the_bound_choice_changes_the_width(self):
        outcomes, strata = [1, 1, 0, 1] * 20, ["short", "medium", "long", "short"] * 20
        wide = EvidenceCalibrator(profile=PROFILE, bound="credible").fit_component(
            "a", outcomes, strata
        )
        tight = EvidenceCalibrator(profile=PROFILE, bound="expected").fit_component(
            "a", outcomes, strata
        )
        assert wide.width >= tight.width


class TestApplication:
    def test_calibration_replaces_placeholders(self, study, outcomes):
        import copy

        report = copy.deepcopy(study.report)
        before = {e.id: e.prob for e in report.failure_model.events.values()}
        result = calibrate_failure_model(
            report.failure_model,
            {
                "react_agent": (outcomes["react_agent"], outcomes["stratum"]),
                "cot_agent": (outcomes["cot_agent"], outcomes["stratum"]),
            },
            profile=PROFILE,
            exact=False,
        )
        assert result.n_updated > 0
        for eid in result.updated:
            assert report.failure_model.events[eid].prob != before[eid]
            assert report.failure_model.events[eid].prob_interval is not None

    def test_calibrated_events_record_their_evidence(self, study, outcomes):
        import copy

        report = copy.deepcopy(study.report)
        result = calibrate_failure_model(
            report.failure_model,
            {"react_agent": (outcomes["react_agent"], outcomes["stratum"])},
            profile=PROFILE,
            exact=False,
        )
        for eid in result.updated:
            evidence = report.failure_model.events[eid].evidence
            assert "observed failures" in evidence
            assert "placeholder" not in evidence.lower()

    def test_uncalibrated_components_are_named(self, study, outcomes):
        import copy

        report = copy.deepcopy(study.report)
        result = calibrate_failure_model(
            report.failure_model,
            {"react_agent": (outcomes["react_agent"], outcomes["stratum"])},
            profile=PROFILE,
            exact=False,
        )
        assert "aggregator" in result.uncalibrated_components
        assert any("placeholder" in n for n in report.failure_model.notes)

    def test_latency_events_keep_their_placeholders(self, study, outcomes):
        """A correctness benchmark says nothing about latency."""
        import copy

        report = copy.deepcopy(study.report)
        result = calibrate_failure_model(
            report.failure_model,
            {"react_agent": (outcomes["react_agent"], outcomes["stratum"])},
            profile=PROFILE,
            exact=False,
        )
        late = [e for e in report.failure_model.events.values() if e.id.endswith("LATE")]
        assert late
        assert all(e.id not in result.updated for e in late)

    def test_the_union_of_calibrated_events_matches_the_measurement(
        self, study, outcomes
    ):
        import copy

        report = copy.deepcopy(study.report)
        calibrator = EvidenceCalibrator(profile=PROFILE, exact=False)
        evidence = calibrator.fit_many(
            {"react_agent": (outcomes["react_agent"], outcomes["stratum"])}
        )
        calibrator.apply(report.failure_model, evidence)
        touched = [
            e
            for e in report.failure_model.events.values()
            if e.component == "react_agent" and e.prob_interval is not None
        ]
        union = 1.0 - np.prod([1.0 - e.prob for e in touched])
        assert union == pytest.approx(evidence["react_agent"].point, abs=1e-9)

    def test_an_explicit_component_map_is_honoured(self, study, outcomes):
        import copy

        report = copy.deepcopy(study.report)
        calibrator = EvidenceCalibrator(profile=PROFILE, exact=False)
        evidence = calibrator.fit_many(
            {"measurement_A": (outcomes["react_agent"], outcomes["stratum"])}
        )
        result = calibrator.apply(
            report.failure_model,
            evidence,
            component_map={"measurement_A": "cot_agent"},
        )
        assert any(eid.startswith("BE-cot_agent") for eid in result.updated)

    def test_a_component_map_pointing_nowhere_is_refused(self, study, outcomes):
        import copy

        report = copy.deepcopy(study.report)
        calibrator = EvidenceCalibrator(profile=PROFILE, exact=False)
        evidence = calibrator.fit_many(
            {"m": (outcomes["react_agent"], outcomes["stratum"])}
        )
        with pytest.raises(KeyError, match="not in the model"):
            calibrator.apply(
                report.failure_model, evidence, component_map={"m": "ghost"}
            )

    def test_the_report_renders(self, study, outcomes):
        import copy

        report = copy.deepcopy(study.report)
        result = calibrate_failure_model(
            report.failure_model,
            {"react_agent": (outcomes["react_agent"], outcomes["stratum"])},
            profile=PROFILE,
            exact=False,
        )
        assert "Calibration" in result.summary()
        assert not result.to_frame().empty
        assert not result.evidence_frame().empty
