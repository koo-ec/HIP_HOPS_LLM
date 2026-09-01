"""The end-to-end study: graph in, calibrated Bayesian network out."""

from __future__ import annotations

import pytest

from HIP_HOPS_LLM import (
    EXAMPLES,
    AgenticReliabilityStudy,
    StudyNotReady,
    load_example,
    load_outcomes,
)

PROFILE = {"short": 0.30, "medium": 0.50, "long": 0.20}


class TestTenLineWorkflow:
    def test_the_documented_pipeline_runs(self):
        """Exactly the example in the package docstring."""
        study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
        study.observe(load_outcomes(), profile=PROFILE)
        study.run()
        text = study.summary()
        network = study.bayesnet("H2")

        assert "HiP-HOPS analysis" in text
        assert 0.0 < network.p_fail() < 1.0

    def test_it_chains(self):
        study = (
            AgenticReliabilityStudy(load_example("parallel_aggregator"))
            .observe(load_outcomes(), profile=PROFILE)
            .run()
        )
        assert study.calibration is not None

    def test_hazard_probability_is_an_interval(self):
        study = (
            AgenticReliabilityStudy(load_example("parallel_aggregator"))
            .observe(load_outcomes(), profile=PROFILE)
            .run()
        )
        envelope = study.hazard_probability("H2")
        assert envelope.lower < envelope.upper
        assert envelope.contains(study.bayesnet("H2").p_fail())


class TestOrdering:
    def test_calibrate_without_outcomes_is_refused(self, parallel_spec):
        study = AgenticReliabilityStudy(parallel_spec)
        with pytest.raises(StudyNotReady, match="observe"):
            study.calibrate()

    def test_calibrate_without_a_profile_is_refused(self, parallel_spec, outcomes):
        study = AgenticReliabilityStudy(parallel_spec)
        study._observations["react_agent"] = (
            list(outcomes["react_agent"]),
            list(outcomes["stratum"]),
        )
        with pytest.raises(StudyNotReady, match="operational profile"):
            study.calibrate()

    def test_analysis_happens_on_demand(self, parallel_spec):
        study = AgenticReliabilityStudy(parallel_spec)
        assert study.report is None
        assert study.hazards_found()
        assert study.report is not None

    def test_an_uncalibrated_summary_says_so_loudly(self, parallel_spec):
        study = AgenticReliabilityStudy(parallel_spec)
        assert "NOT CALIBRATED" in study.summary()

    def test_a_calibrated_summary_reports_the_evidence(self, calibrated):
        text = calibrated.summary()
        assert "NOT CALIBRATED" not in text
        assert "Calibration" in text
        assert "operational profile" in text


class TestObserveShapes:
    def test_a_table_is_read(self, parallel_spec, outcomes):
        study = AgenticReliabilityStudy(parallel_spec)
        study.observe(outcomes, profile=PROFILE)
        assert set(study._observations) == {"react_agent", "cot_agent", "aggregator"}

    def test_the_calibration_split_is_used_alone(self, parallel_spec, outcomes):
        study = AgenticReliabilityStudy(parallel_spec)
        study.observe(outcomes, profile=PROFILE)
        n_calibration = int((outcomes["split"] == "calibration").sum())
        assert len(study._observations["react_agent"][0]) == n_calibration
        assert n_calibration < len(outcomes)

    def test_one_component_at_a_time(self, parallel_spec):
        study = AgenticReliabilityStudy(parallel_spec)
        study.observe([1, 1, 0, 1], ["short"] * 4, component="react_agent",
                      profile={"short": 1.0})
        assert "react_agent" in study._observations

    def test_system_level_outcomes_reproduce_the_hipllm_api(self, parallel_spec):
        """The user's original HIP-LLM snippet, through this package."""
        study = AgenticReliabilityStudy(parallel_spec)
        study.observe(
            [1, 1, 0, 1, 0, 0, 1, 0],
            ["short"] * 4 + ["long"] * 4,
            profile={"short": 0.30, "long": 0.70},
        )
        result = study.operational_failure
        assert result is not None
        summary = result.summary()
        assert 0.0 <= summary["empirical_operational_failure_probability"] <= 1.0
        low, high = result.posterior_credible_envelope
        assert low <= high

    def test_a_missing_stratum_column_is_named(self, parallel_spec, outcomes):
        study = AgenticReliabilityStudy(parallel_spec)
        with pytest.raises(KeyError, match="stratum_column"):
            study.observe(outcomes.drop(columns=["stratum"]), profile=PROFILE)

    def test_outcomes_without_strata_are_refused(self, parallel_spec):
        study = AgenticReliabilityStudy(parallel_spec)
        with pytest.raises(ValueError, match="stratum labels"):
            study.observe([1, 0, 1])


class TestCalibrationChangesTheAnswer:
    def test_the_top_event_probability_moves(self, parallel_spec, outcomes):
        before = AgenticReliabilityStudy(parallel_spec, name="before")
        before.analyse()
        placeholder = before.report.analysis("H2").quant.top_probability

        after = AgenticReliabilityStudy(parallel_spec, name="after")
        after.observe(outcomes, profile=PROFILE).run()
        measured = after.report.analysis("H2").quant.top_probability

        assert placeholder != pytest.approx(measured)

    def test_the_trees_are_resynthesised_after_calibration(self, calibrated):
        """Cut-set quantification must see the calibrated numbers, not the old ones."""
        analysis = calibrated.report.analysis("H2")
        network = calibrated.bayesnet("H2")
        assert analysis.quant.top_probability >= network.p_fail() - 1e-9

    def test_the_cut_set_structure_is_unchanged_by_calibration(
        self, study, calibrated
    ):
        """Calibration changes numbers, never structure."""
        before = {frozenset(s) for s in study.report.analysis("H2").cuts.sets}
        after = {frozenset(s) for s in calibrated.report.analysis("H2").cuts.sets}
        assert before == after


class TestOutputs:
    def test_fmea_and_single_points_render(self, calibrated):
        assert not calibrated.fmea().empty
        assert calibrated.single_points()

    def test_cpts_are_available_per_hazard(self, calibrated):
        cpts = calibrated.cpts("H2")
        assert cpts.top
        assert not cpts.to_frame().empty

    def test_save_writes_every_artefact(self, calibrated, tmp_path):
        written = calibrated.save(str(tmp_path))
        assert written
        names = {p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in written}
        assert any(n.endswith("_report.md") for n in names)
        assert any(n.endswith("_cutsets.csv") for n in names)
        assert any(n.endswith("_fmea.csv") for n in names)
        assert "calibration.csv" in names
        assert "evidence.csv" in names
        for path in written:
            assert tmp_path.joinpath(path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]).exists()

    def test_plots_render(self, calibrated):
        assert calibrated.plot("H2") is not None
        assert calibrated.plot_architecture() is not None

    def test_the_network_draws_without_graphviz(self, calibrated, tmp_path):
        view = calibrated.bayesnet("H2").view()
        written = view.to_png(str(tmp_path / "bn.png"))
        assert written == str(tmp_path / "bn.png")
        assert (tmp_path / "bn.png").stat().st_size > 1000
        assert "digraph" in view.to_dot()
        assert view.to_svg().lstrip().startswith(("<?xml", "<svg"))


@pytest.mark.parametrize("key", sorted(EXAMPLES))
class TestEveryExample:
    def test_analyses_and_converts(self, key):
        study = AgenticReliabilityStudy(load_example(key), name=key)
        study.analyse()
        assert study.hazards_found()
        for hazard in study.hazards_found():
            network = study.bayesnet(hazard)

            check = network.cross_check()
            assert set(check) == {
                "exact",
                "pyagrum",
                "difference",
                "relative_difference",
                "agree",
                "compared",
            }, "cross_check must return the same keys whether or not pyAgrum is here"
            if check["compared"]:
                assert check["agree"] == 1.0, (
                    f"{key}/{hazard}: exact and pyAgrum disagree — "
                    f"{check['exact']!r} vs {check['pyagrum']!r}"
                )
            else:
                # Without pyAgrum the comparison cannot run; saying so is not the
                # same as saying the engines disagree.
                assert check["agree"] != check["agree"], "agree should be NaN"
                assert 0.0 <= check["exact"] <= 1.0

            comparison = network.compare_with_cutsets(study.report.analysis(hazard))
            assert comparison["bound_overestimate"] >= -1e-9
