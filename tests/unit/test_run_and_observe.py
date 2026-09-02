"""``run_and_observe``: run the graph, score every node, calibrate — in one call.

This is the entry point for a notebook that already builds and runs a LangGraph
application. Its correctness hinges on one thing: never attributing a failure to
a component that did not cause it. A node that a router never reached, and a node
in a run that crashed outright, must both be *missing observations* rather than
failures, or the analysis blames whoever happens to be downstream.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from hiphopsllm import AgenticReliabilityStudy, OperationalProfile  # noqa: E402

SPEC = {
    "name": "serial pipeline",
    "nodes": {
        "__start__": {"role": "source"},
        "first": {"role": "llm_agent", "source_code": "out = model.generate(x)"},
        "second": {"role": "llm_agent", "source_code": "out = model.generate(y)"},
        "__end__": {"role": "sink"},
    },
    "edges": [["__start__", "first"], ["first", "second"], ["second", "__end__"]],
}

SUCCESS = {
    "first": lambda s: s.get("first_ok"),
    "second": lambda s: s.get("second_ok"),
}


def make_runner(script):
    """A fake ``graph.invoke``: yields one prepared final state per call."""
    states = list(script)

    def invoke(item):
        state = states.pop(0)
        if isinstance(state, Exception):
            raise state
        return state

    return invoke


@pytest.fixture
def study():
    return AgenticReliabilityStudy(SPEC, name="serial", exact_inference=False)


class TestBasics:
    def test_it_runs_scores_and_calibrates_in_one_call(self, study):
        script = [{"first_ok": True, "second_ok": True}] * 8 + [
            {"first_ok": False, "second_ok": None}
        ] * 2
        frame = study.run_and_observe(
            inputs=[{"i": i} for i in range(10)],
            success=SUCCESS,
            invoke=make_runner(script),
            profile={"all": 1.0},
            progress=False,
        )
        assert list(frame.columns) == [
            "item_id", "stratum", "first", "second", "split"
        ]
        assert len(frame) == 10
        assert study.calibration is not None
        assert study.evidence["first"].n_failures == 2 or study.evidence["first"].n_trials

    def test_it_returns_the_measurement_for_keeping(self, study):
        frame = study.run_and_observe(
            inputs=[{}, {}],
            success=SUCCESS,
            invoke=make_runner([{"first_ok": True, "second_ok": True}] * 2),
            profile={"all": 1.0},
            progress=False,
        )
        assert frame["item_id"].tolist() == ["item_0000", "item_0001"]

    def test_calibration_can_be_deferred(self, study):
        study.run_and_observe(
            inputs=[{}, {}],
            success=SUCCESS,
            invoke=make_runner([{"first_ok": True, "second_ok": True}] * 2),
            profile={"all": 1.0},
            calibrate=False,
            progress=False,
        )
        assert study.calibration is None
        assert study._observations

    def test_a_custom_invoke_is_used(self, study):
        calls = []

        def invoke(item):
            calls.append(item)
            return {"first_ok": True, "second_ok": True}

        study.run_and_observe(
            inputs=[{"n": 1}, {"n": 2}], success=SUCCESS, invoke=invoke,
            profile={"all": 1.0}, progress=False,
        )
        assert calls == [{"n": 1}, {"n": 2}]


class TestNotExercised:
    def test_none_is_a_missing_observation_not_a_failure(self, study):
        """The core rule: a node a router never reached did not fail."""
        script = [{"first_ok": True, "second_ok": True}] * 6 + [
            {"first_ok": False, "second_ok": None}
        ] * 4
        study.run_and_observe(
            inputs=[{}] * 10, success=SUCCESS, invoke=make_runner(script),
            profile={"all": 1.0}, progress=False,
        )
        # `first` was scored on every item; `second` only where it ran.
        assert study.evidence["first"].n_trials > study.evidence["second"].n_trials
        assert study.evidence["second"].n_failures == 0

    def test_scoring_an_unreached_node_as_failed_would_be_worse(self, study):
        """Contrast: if None became 0, `second` would look far worse than it is."""
        script = [{"first_ok": True, "second_ok": True}] * 5 + [
            {"first_ok": False, "second_ok": None}
        ] * 5
        study.run_and_observe(
            inputs=[{}] * 10, success=SUCCESS, invoke=make_runner(script),
            profile={"all": 1.0}, progress=False,
        )
        assert study.evidence["second"].empirical == pytest.approx(0.0)

    def test_a_node_never_exercised_at_all_is_refused_clearly(self, study):
        script = [{"first_ok": False, "second_ok": None}] * 4
        with pytest.raises(ValueError, match="no observations at all"):
            study.run_and_observe(
                inputs=[{}] * 4, success=SUCCESS, invoke=make_runner(script),
                profile={"all": 1.0}, progress=False,
            )


class TestErrors:
    def test_a_crashed_run_is_skipped_by_default(self, study, capsys):
        script = [{"first_ok": True, "second_ok": True}] * 6 + [RuntimeError("boom")] * 2
        frame = study.run_and_observe(
            inputs=[{}] * 8, success=SUCCESS, invoke=make_runner(script),
            profile={"all": 1.0}, progress=False,
        )
        assert len(frame) == 6
        assert "raised" in capsys.readouterr().out

    def test_a_crashed_run_never_blames_a_node(self, study):
        """A crash leaves no state to score, so marking its nodes as failures
        would penalise components that had already succeeded."""
        script = [{"first_ok": True, "second_ok": True}] * 6 + [RuntimeError("boom")] * 2
        study.run_and_observe(
            inputs=[{}] * 8, success=SUCCESS, invoke=make_runner(script),
            on_error="record", profile={"all": 1.0}, progress=False,
        )
        assert study.evidence["first"].n_failures == 0
        assert study.evidence["second"].n_failures == 0

    def test_record_keeps_the_row_and_the_message(self, study):
        script = [{"first_ok": True, "second_ok": True}] * 6 + [ValueError("nope")] * 2
        frame = study.run_and_observe(
            inputs=[{}] * 8, success=SUCCESS, invoke=make_runner(script),
            on_error="record", profile={"all": 1.0}, progress=False,
        )
        assert len(frame) == 8
        assert "run_error" in frame.columns
        assert any("nope" in str(v) for v in frame["run_error"])

    def test_a_high_crash_rate_is_called_out(self, study, capsys):
        script = [{"first_ok": True, "second_ok": True}] * 5 + [RuntimeError("x")] * 5
        study.run_and_observe(
            inputs=[{}] * 10, success=SUCCESS, invoke=make_runner(script),
            profile={"all": 1.0}, progress=False,
        )
        assert "optimistic" in capsys.readouterr().out

    def test_a_predicate_that_raises_scores_nothing_rather_than_a_failure(self, study):
        """A predicate that cannot decide has not observed a failure. Here every
        one of its verdicts is unknown, so the run is refused outright rather
        than reporting a component nobody measured."""

        def explodes(state):
            raise KeyError("missing key")

        with pytest.raises(ValueError, match="no observations at all"):
            study.run_and_observe(
                inputs=[{}] * 4,
                success={"first": lambda s: True, "second": explodes},
                invoke=make_runner([{"first_ok": True}] * 4),
                profile={"all": 1.0},
                calibrate=False,
                progress=False,
            )

    def test_an_occasional_undecidable_verdict_is_merely_dropped(self, study):
        calls = {"n": 0}

        def sometimes(state):
            calls["n"] += 1
            if calls["n"] % 2:
                raise KeyError("missing key")
            return True

        study.run_and_observe(
            inputs=[{}] * 6,
            success={"first": lambda s: True, "second": sometimes},
            invoke=make_runner([{"first_ok": True}] * 6),
            profile={"all": 1.0},
            calibrate=False,
            progress=False,
        )
        assert len(study._observations["second"][0]) < 6
        assert study._observations["second"][0]

    def test_every_run_failing_is_an_explicit_error(self, study):
        with pytest.raises(RuntimeError, match="every run failed"):
            study.run_and_observe(
                inputs=[{}] * 3, success=SUCCESS,
                invoke=make_runner([RuntimeError("x")] * 3),
                profile={"all": 1.0}, progress=False,
            )


class TestStrata:
    def test_a_callable_stratifies_each_input(self, study):
        frame = study.run_and_observe(
            inputs=[{"n": 1}, {"n": 50}, {"n": 2}, {"n": 90}],
            success=SUCCESS,
            stratum=lambda item: "big" if item["n"] > 10 else "small",
            invoke=make_runner([{"first_ok": True, "second_ok": True}] * 4),
            profile={"small": 0.5, "big": 0.5},
            progress=False,
        )
        assert frame["stratum"].tolist() == ["small", "big", "small", "big"]

    def test_a_ready_made_sequence_is_accepted(self, study):
        frame = study.run_and_observe(
            inputs=[{}] * 3, success=SUCCESS, stratum=["a", "b", "a"],
            invoke=make_runner([{"first_ok": True, "second_ok": True}] * 3),
            profile={"a": 0.7, "b": 0.3}, progress=False,
        )
        assert frame["stratum"].tolist() == ["a", "b", "a"]

    def test_a_mismatched_label_count_is_refused(self, study):
        with pytest.raises(ValueError, match="stratum labels for"):
            study.run_and_observe(
                inputs=[{}] * 3, success=SUCCESS, stratum=["a", "b"],
                invoke=make_runner([{}] * 3), progress=False,
            )

    def test_without_strata_everything_is_one_stratum(self, study):
        study.run_and_observe(
            inputs=[{}] * 4, success=SUCCESS,
            invoke=make_runner([{"first_ok": True, "second_ok": True}] * 4),
            progress=False,
        )
        assert len(study.profile) == 1

    def test_the_profile_defaults_to_the_observed_frequencies(self, study):
        """Of the *calibration* rows: the profile describes what was measured."""
        study.run_and_observe(
            inputs=[{}] * 4, success=SUCCESS, stratum=["a", "a", "a", "b"],
            invoke=make_runner([{"first_ok": True, "second_ok": True}] * 4),
            calibration_fraction=1.0, progress=False,
        )
        assert isinstance(study.profile, OperationalProfile)
        assert study.profile["a"] == pytest.approx(0.75)
        assert "observed frequencies" in study.profile.provenance


class TestSplit:
    def test_a_held_out_set_is_kept(self, study):
        frame = study.run_and_observe(
            inputs=[{}] * 20, success=SUCCESS,
            invoke=make_runner([{"first_ok": True, "second_ok": True}] * 20),
            calibration_fraction=0.75, profile={"all": 1.0}, progress=False,
        )
        assert (frame["split"] == "calibration").sum() == 15
        assert (frame["split"] == "test").sum() == 5
        assert study.evidence["first"].n_trials == 15

    def test_at_least_one_calibration_item_survives_a_tiny_fraction(self, study):
        frame = study.run_and_observe(
            inputs=[{}] * 4, success=SUCCESS,
            invoke=make_runner([{"first_ok": True, "second_ok": True}] * 4),
            calibration_fraction=0.0, profile={"all": 1.0}, progress=False,
        )
        assert (frame["split"] == "calibration").sum() >= 1


class TestGuards:
    def test_no_inputs_is_refused(self, study):
        with pytest.raises(ValueError, match="at least one input"):
            study.run_and_observe(inputs=[], success=SUCCESS, progress=False)

    def test_no_success_predicates_is_refused(self, study):
        with pytest.raises(ValueError, match="nothing to measure"):
            study.run_and_observe(inputs=[{}], success={}, progress=False)

    def test_an_unknown_on_error_is_refused(self, study):
        with pytest.raises(ValueError, match="'skip' or 'record'"):
            study.run_and_observe(
                inputs=[{}], success=SUCCESS, on_error="explode", progress=False
            )

    def test_a_specification_without_invoke_says_why(self, study):
        with pytest.raises(TypeError, match="nothing to run"):
            study.run_and_observe(inputs=[{}], success=SUCCESS, progress=False)


class TestTwoCellWorkflow:
    def test_the_documented_two_cells(self, study):
        """Cell one measures and calibrates; cell two draws the network."""
        study.run_and_observe(
            inputs=[{}] * 12,
            success=SUCCESS,
            stratum=lambda item: "all",
            invoke=make_runner(
                [{"first_ok": True, "second_ok": True}] * 9
                + [{"first_ok": False, "second_ok": None}] * 3
            ),
            profile={"all": 1.0},
            progress=False,
        )
        network = study.bayesnet("H2")
        assert 0.0 < network.p_fail() < 1.0
        assert network.view().figure() is not None
        envelope = study.hazard_probability("H2")
        assert envelope.lower <= envelope.upper
