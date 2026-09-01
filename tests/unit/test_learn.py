"""Fitting conditional probability tables from observed agent outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from HIP_HOPS_LLM.bayes.cpt import FAIL
from HIP_HOPS_LLM.bayes.learn import (
    CPTLearningError,
    fit_cpts,
    learn_cpt,
    learn_gate,
)


@pytest.fixture
def failures():
    """A frame of failure indicators where the gate is a genuine OR."""
    rng = np.random.default_rng(11)
    a = rng.integers(0, 2, 400)
    b = rng.integers(0, 2, 400)
    return pd.DataFrame({"a": a, "b": b, "y": np.maximum(a, b)})


class TestLearnCPT:
    def test_a_root_prior_matches_the_observed_rate(self):
        frame = pd.DataFrame({"a": [1] * 30 + [0] * 70})
        learned = learn_cpt(frame, "a", alpha=0.0000001)
        assert learned.cpt.prior_fail == pytest.approx(0.30, abs=1e-4)

    def test_smoothing_pulls_towards_the_prior_mean(self):
        frame = pd.DataFrame({"a": [1] * 10})
        unsmoothed = learn_cpt(frame, "a", alpha=1e-9).cpt.prior_fail
        smoothed = learn_cpt(frame, "a", alpha=1.0).cpt.prior_fail
        assert unsmoothed > smoothed
        assert smoothed < 1.0, "an unsmoothed table would claim certainty"

    def test_a_deterministic_or_is_recovered(self, failures):
        learned = learn_cpt(failures, "y", ["a", "b"], alpha=1e-9)
        table = learned.cpt.table[..., FAIL]
        assert table[(0, 0)] == pytest.approx(0.0, abs=1e-6)
        assert table[(1, 0)] == pytest.approx(1.0, abs=1e-6)
        assert table[(1, 1)] == pytest.approx(1.0, abs=1e-6)

    def test_counts_and_coverage_are_reported(self, failures):
        learned = learn_cpt(failures, "y", ["a", "b"])
        assert learned.n_observations == len(failures)
        assert learned.total_rows == 4
        assert learned.coverage == pytest.approx(1.0)

    def test_an_unobserved_row_is_flagged_as_prior_dominated(self):
        frame = pd.DataFrame({"a": [0, 0, 1], "b": [0, 0, 0], "y": [0, 0, 1]})
        learned = learn_cpt(frame, "y", ["a", "b"])
        assert learned.prior_dominated_rows == 2
        assert learned.coverage == pytest.approx(0.5)
        assert "prior-dominated" in learned.summary()

    def test_tables_are_normalised(self, failures):
        table = learn_cpt(failures, "y", ["a", "b"]).cpt.table
        assert np.allclose(table.sum(axis=-1), 1.0)

    def test_string_states_are_accepted(self):
        frame = pd.DataFrame({"a": ["Fail", "OK", "OK", "OK"]})
        assert learn_cpt(frame, "a", alpha=1e-9).cpt.prior_fail == pytest.approx(0.25)

    def test_a_missing_column_is_named(self, failures):
        with pytest.raises(CPTLearningError, match="no column"):
            learn_cpt(failures, "y", ["a", "nope"])

    def test_a_non_positive_alpha_is_refused(self, failures):
        with pytest.raises(CPTLearningError, match="alpha must be positive"):
            learn_cpt(failures, "y", ["a"], alpha=0.0)


class TestSplitGuard:
    def test_test_rows_are_refused(self, failures):
        frame = failures.copy()
        frame["split"] = ["calibration"] * 200 + ["test"] * 200
        with pytest.raises(CPTLearningError, match="calibration split only"):
            learn_cpt(frame, "y", ["a", "b"])

    def test_a_calibration_only_frame_is_accepted(self, failures):
        frame = failures.copy()
        frame["split"] = "calibration"
        assert learn_cpt(frame, "y", ["a", "b"]).n_observations == len(frame)

    def test_the_guard_can_be_turned_off_deliberately(self, failures):
        frame = failures.copy()
        frame["split"] = "test"
        assert learn_cpt(frame, "y", ["a"], check_split=False).n_observations > 0


class TestLearnGate:
    def test_an_or_gate_is_recognised(self, failures):
        _, distances = learn_gate(failures, "y", ["a", "b"], alpha=1e-9)
        assert distances["nearest"] == "or"
        assert distances["or"] == pytest.approx(0.0, abs=1e-6)

    def test_an_and_gate_is_recognised(self):
        rng = np.random.default_rng(3)
        a, b = rng.integers(0, 2, 400), rng.integers(0, 2, 400)
        frame = pd.DataFrame({"a": a, "b": b, "y": np.minimum(a, b)})
        _, distances = learn_gate(frame, "y", ["a", "b"], alpha=1e-9)
        assert distances["nearest"] == "and"

    def test_a_repairing_reviewer_is_neither(self):
        """The motivating case: a reviewer repairs some upstream errors, so the
        fitted table sits between AND and OR and neither gate describes it."""
        rng = np.random.default_rng(5)
        a = rng.integers(0, 2, 2000)
        b = rng.integers(0, 2, 2000)
        upstream = np.maximum(a, b)
        repaired = upstream & (rng.random(2000) > 0.55)
        frame = pd.DataFrame({"a": a, "b": b, "y": repaired.astype(int)})
        _, distances = learn_gate(frame, "y", ["a", "b"], alpha=1e-9)
        assert distances["or"] > 0.2
        assert distances["and"] > 0.1


class TestFitCPTs:
    STRUCTURE = {"react": [], "cot": [], "aggregator": ["react", "cot"]}

    def test_a_whole_network_is_fitted(self, outcomes):
        cpts, fits = fit_cpts(
            outcomes[outcomes["split"] == "calibration"],
            {
                "react_agent": [],
                "cot_agent": [],
                "aggregator": ["react_agent", "cot_agent"],
            },
            outcomes_are_failures=False,
        )
        assert set(cpts.order) == {"react_agent", "cot_agent", "aggregator"}
        assert cpts.top == "aggregator"
        assert all(f.n_observations > 0 for f in fits.values())

    def test_correctness_columns_are_inverted_once(self):
        frame = pd.DataFrame({"a": [1, 1, 1, 0]})   # 1 = correct
        cpts, _ = fit_cpts(frame, {"a": []}, outcomes_are_failures=False)
        assert cpts["a"].prior_fail == pytest.approx(0.4, abs=0.15)
        assert cpts["a"].prior_fail < 0.5

    def test_the_fitted_network_answers_queries(self, outcomes):
        from HIP_HOPS_LLM import BayesianNetwork

        cpts, _ = fit_cpts(
            outcomes[outcomes["split"] == "calibration"],
            {
                "react_agent": [],
                "cot_agent": [],
                "aggregator": ["react_agent", "cot_agent"],
            },
            outcomes_are_failures=False,
        )
        bn = BayesianNetwork(cpts=cpts, name="learned")
        marginal = bn.p_fail("aggregator")
        conditional = bn.p_fail(
            "aggregator", evidence={"react_agent": "Fail", "cot_agent": "Fail"}
        )
        assert 0.0 < marginal < 1.0
        assert conditional > marginal, (
            "both drafts being wrong must raise the aggregator's failure "
            "probability above its marginal"
        )

    def test_prior_dominated_rows_are_noted(self):
        frame = pd.DataFrame({"a": [0, 0], "b": [0, 0], "y": [0, 0]})
        cpts, _ = fit_cpts(frame, {"a": [], "b": [], "y": ["a", "b"]})
        assert any("prior" in n for n in cpts.notes)
