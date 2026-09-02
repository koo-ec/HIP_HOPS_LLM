"""The measured claim: P(fail per task | OP), and R(n) over future tasks.

HIP-LLM defines reliability as *the probability of failure-free operation over a
specified number of future tasks under a given operational profile*
(Aghazadeh-Chakherlou et al., 2026). That statement, not a benchmark accuracy and
not a decomposition over failure modes, is what this package measures. These
tests pin it down.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from hiphopsllm import (  # noqa: E402
    AgenticReliabilityStudy,
    EvidenceCalibrator,
    OperationalProfile,
    StudyNotReady,
    dataset_proportional_profile,
    empirical_profile,
)

PROFILE = {"short": 0.30, "medium": 0.50, "long": 0.20}


@pytest.fixture(scope="module")
def evidence():
    """A component measured the exact way, so the posterior family exists."""
    calibrator = EvidenceCalibrator(profile=PROFILE, exact=True)
    outcomes = ([1] * 80 + [0] * 20) * 2
    strata = (["short"] * 30 + ["medium"] * 50 + ["long"] * 20) * 2
    return calibrator.fit_component("gpt-4o-mini", outcomes, strata)


class TestTheMeasuredQuantity:
    def test_the_failure_probability_is_per_task_under_the_profile(self, evidence):
        low, high = evidence.interval
        assert 0.0 <= low <= high <= 1.0
        assert evidence.profile.weights == OperationalProfile(PROFILE).weights

    def test_success_is_the_complement_of_failure(self, evidence):
        low, high = evidence.interval
        assert evidence.success_interval == pytest.approx((1.0 - high, 1.0 - low))

    def test_the_statement_names_the_profile_and_the_sample(self, evidence):
        text = evidence.statement()
        assert "gpt-4o-mini fails on a task with probability" in text
        assert "under the operational profile" in text
        assert "short 30%" in text and "long 20%" in text
        assert f"{evidence.n_failures}/{evidence.n_trials}" in text

    def test_the_statement_is_not_about_a_failure_mode(self, evidence):
        """The measurement is task failure, whatever the cause. Nothing in the
        headline claim should name hallucination or any other single mode."""
        text = evidence.statement(n_tasks=10).lower()
        for word in ("halluc", "format", "truncat", "nondet"):
            assert word not in text


class TestReliabilityOverFutureTasks:
    def test_one_task_is_the_complement_of_failure(self, evidence):
        low, high = evidence.reliability(1)
        s_low, s_high = evidence.success_interval
        # Envelope of E[p] over configurations, so it sits inside the interval.
        assert s_low - 1e-9 <= low <= high <= s_high + 1e-9

    def test_reliability_falls_as_the_horizon_grows(self, evidence):
        previous = evidence.reliability(1)
        for n in (2, 5, 10, 50):
            current = evidence.reliability(n)
            assert current[0] <= previous[0] + 1e-12
            assert current[1] <= previous[1] + 1e-12
            previous = current

    def test_it_stays_a_probability(self, evidence):
        for n in (1, 3, 25, 200):
            low, high = evidence.reliability(n)
            assert 0.0 <= low <= high <= 1.0

    def test_it_is_not_the_naive_power_of_the_mean(self, evidence):
        """E[p^n] >= E[p]^n by Jensen; the naive form understates reliability."""
        naive = evidence.reliability(1)[1] ** 20
        proper = evidence.reliability(20)[1]
        assert proper >= naive - 1e-12

    def test_a_horizon_below_one_is_refused(self, evidence):
        with pytest.raises(ValueError, match="at least 1"):
            evidence.reliability(0)

    def test_the_approximation_path_still_answers(self):
        calibrator = EvidenceCalibrator(profile=PROFILE, exact=False)
        evidence = calibrator.fit_component(
            "fast", [1, 1, 0, 1] * 10, ["short", "medium", "long", "short"] * 10
        )
        assert evidence.posterior is None
        low, high = evidence.reliability(5)
        assert 0.0 <= low <= high <= 1.0


class TestTheStudyReport:
    @pytest.fixture(scope="class")
    def study(self, request):
        from hiphopsllm import load_example, load_outcomes

        s = AgenticReliabilityStudy(load_example("parallel_aggregator"))
        s.observe(load_outcomes(), profile=PROFILE).run()
        return s

    def test_it_leads_with_the_profile_and_the_measurement(self, study):
        text = study.operational_reliability()
        assert "Measured reliability under the operational profile" in text
        assert "P(fail per task)" in text
        for component in study.evidence:
            assert component in text

    def test_a_single_task_horizon_omits_the_reliability_column(self, study):
        assert "R(1 task)" not in study.operational_reliability()

    def test_a_longer_horizon_adds_it(self, study):
        text = study.operational_reliability(n_tasks=10)
        assert "R(10 tasks)" in text
        assert "failure-free operation over n future" in text

    def test_a_tiny_reliability_is_shown_in_scientific_notation(self, study):
        """R(500) is real but rounds to zero at four decimals; printing 0.0000
        would read as a bug rather than as a very small number."""
        assert "e-" in study.operational_reliability(n_tasks=500)

    def test_it_says_the_interval_is_an_envelope_not_a_confidence_interval(self, study):
        assert "posterior envelope" in study.operational_reliability()

    def test_it_records_the_inference_method(self, study):
        assert "method:" in study.operational_reliability()

    def test_the_study_summary_leads_with_the_measurement(self, study):
        summary = study.summary()
        measurement = summary.index("Measured reliability under the operational")
        decomposition = summary.index("Calibration")
        assert measurement < decomposition, (
            "the measured claim must come before the fault-tree decomposition"
        )

    def test_the_summary_says_the_decomposition_is_a_modelling_step(self, study):
        assert "modelling step" in study.summary()

    def test_it_refuses_before_anything_is_measured(self):
        from hiphopsllm import load_example

        with pytest.raises(StudyNotReady, match="nothing has been measured"):
            AgenticReliabilityStudy(load_example("parallel_aggregator")).\
                operational_reliability()


class TestProfileProvenance:
    def test_a_dataset_proportional_profile_says_what_it_assumes(self):
        profile = dataset_proportional_profile(["a"] * 3 + ["b"])
        assert profile["a"] == pytest.approx(0.75)
        assert "ASSERTS" in profile.provenance
        assert "not a measurement" in profile.provenance

    def test_it_is_distinct_from_a_profile_measured_in_production(self):
        """Identical arithmetic, different claim. That is why they are separate."""
        labels = ["a"] * 3 + ["b"]
        dataset = dataset_proportional_profile(labels)
        production = empirical_profile(labels)
        assert dataset.weights == production.weights
        assert dataset.provenance != production.provenance
        assert "observed frequencies" in production.provenance

    def test_defaulting_the_profile_warns_that_it_is_an_assumption(self, capsys):
        from hiphopsllm import load_example, load_outcomes

        study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
        study.observe(load_outcomes())          # no profile given
        out = capsys.readouterr().out
        assert "NO OPERATIONAL PROFILE GIVEN" in out
        assert "test set" in out
        assert "ASSERTS" in study.profile.provenance

    def test_an_explicit_profile_is_not_second_guessed(self, capsys):
        from hiphopsllm import load_example, load_outcomes

        study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
        study.observe(load_outcomes(), profile=PROFILE)
        assert "NO OPERATIONAL PROFILE GIVEN" not in capsys.readouterr().out
        assert study.profile.provenance == "declared by the analyst"


class TestTheProfileActuallyChangesTheAnswer:
    def test_reweighting_the_workload_moves_the_failure_probability(self):
        """The whole point: the same measurements under a different workload are
        a different reliability claim."""
        calibrator_easy = EvidenceCalibrator(
            profile={"short": 0.9, "medium": 0.05, "long": 0.05}, exact=False
        )
        calibrator_hard = EvidenceCalibrator(
            profile={"short": 0.05, "medium": 0.05, "long": 0.9}, exact=False
        )
        # Short items nearly always pass; long ones nearly always fail.
        outcomes = [1] * 40 + [1] * 40 + [0] * 40
        strata = ["short"] * 40 + ["medium"] * 40 + ["long"] * 40
        easy = calibrator_easy.fit_component("m", outcomes, strata)
        hard = calibrator_hard.fit_component("m", outcomes, strata)
        assert hard.empirical > easy.empirical + 0.3, (
            "a workload dominated by the hard stratum must report a much higher "
            "failure probability from the very same observations"
        )

    def test_the_per_stratum_rates_are_kept_for_inspection(self):
        calibrator = EvidenceCalibrator(profile=PROFILE, exact=False)
        evidence = calibrator.fit_component(
            "m",
            [1] * 30 + [1] * 50 + [0] * 20,
            ["short"] * 30 + ["medium"] * 50 + ["long"] * 20,
        )
        assert set(evidence.by_stratum) == {"short", "medium", "long"}
        assert evidence.by_stratum["long"] == pytest.approx(1.0)
        assert evidence.by_stratum["short"] == pytest.approx(0.0)

    def test_the_profile_weighted_average_is_what_is_reported(self):
        calibrator = EvidenceCalibrator(profile=PROFILE, exact=False)
        evidence = calibrator.fit_component(
            "m",
            [1] * 30 + [1] * 50 + [0] * 20,
            ["short"] * 30 + ["medium"] * 50 + ["long"] * 20,
        )
        expected = np.dot(
            [PROFILE[k] for k in ("short", "medium", "long")],
            [evidence.by_stratum[k] for k in ("short", "medium", "long")],
        )
        assert evidence.empirical == pytest.approx(expected)
