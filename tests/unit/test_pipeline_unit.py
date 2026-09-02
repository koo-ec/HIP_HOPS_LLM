"""AgenticReliabilityStudy at the unit level: state, guards and accessors.

The integration tests run the pipeline end to end. These cover the state machine
around it — what happens before each step is legal, what the error messages say,
and the accessors that only appear in a notebook.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from hiphopsllm import (  # noqa: E402
    AgenticReliabilityStudy,
    OperationalProfile,
    StudyNotReady,
)

PROFILE = {"short": 0.30, "medium": 0.50, "long": 0.20}


@pytest.fixture
def fresh(parallel_spec):
    return AgenticReliabilityStudy(parallel_spec, name="fresh")


class TestConstruction:
    def test_a_mapping_profile_is_coerced_at_construction(self, parallel_spec):
        study = AgenticReliabilityStudy(parallel_spec, profile=PROFILE)
        assert isinstance(study.profile, OperationalProfile)

    def test_a_profile_object_passes_through(self, parallel_spec):
        profile = OperationalProfile(PROFILE)
        assert AgenticReliabilityStudy(parallel_spec, profile=profile).profile is profile

    def test_no_profile_stays_none(self, fresh):
        assert fresh.profile is None

    def test_nothing_is_computed_eagerly(self, fresh):
        assert fresh.report is None
        assert fresh.calibration is None
        assert fresh.evidence == {}
        assert fresh.operational_failure is None

    def test_repr_reports_the_stage(self, fresh, calibrated):
        assert "not analysed" in repr(fresh)
        fresh.analyse()
        assert "analysed" in repr(fresh)
        assert "calibrated" in repr(calibrated)


class TestGuards:
    def test_calibrate_before_observe_names_the_missing_step(self, fresh):
        with pytest.raises(StudyNotReady) as excinfo:
            fresh.calibrate()
        assert "observe()" in str(excinfo.value)
        assert "placeholder" in str(excinfo.value)

    def test_calibrate_without_a_profile_explains_why_it_matters(
        self, fresh, outcomes
    ):
        fresh._observations["react_agent"] = (
            list(outcomes["react_agent"]), list(outcomes["stratum"])
        )
        with pytest.raises(StudyNotReady, match="never inferred silently"):
            fresh.calibrate()

    def test_observe_with_no_outcomes_at_all(self, fresh):
        with pytest.raises(ValueError, match="needs outcomes"):
            fresh.observe()

    def test_observe_rejects_a_frame_with_no_binary_columns(self, fresh):
        frame = pd.DataFrame({"stratum": ["short"] * 3, "text": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="component_columns"):
            fresh.observe(frame, profile={"short": 1.0})

    def test_run_is_idempotent(self, parallel_spec, outcomes):
        study = AgenticReliabilityStudy(parallel_spec, exact_inference=False)
        study.observe(outcomes, profile=PROFILE).run()
        first = study.hazard_probability("H2").as_tuple()
        study.run()
        assert study.hazard_probability("H2").as_tuple() == pytest.approx(
            first, rel=1e-15, abs=0.0
        )


class TestObserveShapes:
    def test_explicit_component_columns_are_honoured(self, fresh, outcomes):
        fresh.observe(outcomes, profile=PROFILE, component_columns=["react_agent"])
        assert set(fresh._observations) == {"react_agent"}

    def test_the_split_column_can_be_disabled(self, fresh, outcomes):
        fresh.observe(outcomes, profile=PROFILE, split_column=None)
        assert len(fresh._observations["react_agent"][0]) == len(outcomes)

    def test_a_different_calibration_label_is_honoured(self, fresh, outcomes):
        frame = outcomes.copy()
        frame["split"] = frame["split"].replace({"calibration": "train"})
        fresh.observe(frame, profile=PROFILE, calibration_split="train")
        expected = int((frame["split"] == "train").sum())
        assert len(fresh._observations["react_agent"][0]) == expected

    def test_a_frame_without_a_split_column_uses_every_row(self, fresh, outcomes):
        frame = outcomes.drop(columns=["split"])
        fresh.observe(frame, profile=PROFILE)
        assert len(fresh._observations["react_agent"][0]) == len(frame)

    def test_a_missing_profile_falls_back_to_the_benchmark_mix_and_says_so(
        self, fresh, outcomes, capsys
    ):
        """It must not present the dataset's own composition as a measured
        operational profile: that conflation is what HIP-LLM exists to remove."""
        fresh.observe(outcomes)
        assert isinstance(fresh.profile, OperationalProfile)
        assert "ASSERTS" in fresh.profile.provenance
        assert "not a measurement" in fresh.profile.provenance
        assert "NO OPERATIONAL PROFILE GIVEN" in capsys.readouterr().out

    def test_a_stratum_column_under_another_name(self, fresh, outcomes):
        frame = outcomes.rename(columns={"stratum": "bucket"})
        fresh.observe(frame, profile=PROFILE, stratum_column="bucket")
        assert fresh._observations

    def test_a_component_map_is_stored_for_calibration(self, fresh, outcomes):
        fresh.observe(outcomes, profile=PROFILE, component_map={"react_agent": "cot_agent"})
        assert fresh._component_map == {"react_agent": "cot_agent"}

    def test_system_level_observation_does_not_populate_components(self, fresh):
        fresh.observe([1, 0, 1, 0], ["short"] * 4, profile={"short": 1.0})
        assert fresh._observations == {}
        assert fresh.operational_failure is not None


class TestAccessors:
    def test_accessors_trigger_analysis_on_demand(self, parallel_spec):
        for accessor in (
            lambda s: s.system,
            lambda s: s.failure_model,
            lambda s: s.hazards_found(),
            lambda s: s.cut_sets("H2"),
            lambda s: s.single_points(),
            lambda s: s.fmea(),
        ):
            study = AgenticReliabilityStudy(parallel_spec)
            assert study.report is None
            accessor(study)
            assert study.report is not None

    def test_fmea_is_a_frame_with_the_expected_columns(self, study):
        frame = study.fmea()
        assert {"component", "event", "class", "P", "severity", "mitigation"} <= set(
            frame.columns
        )
        assert len(frame) == len(study.report.fmea())

    def test_cpts_accepts_builder_options(self, study):
        soft = study.cpts("H2", soft_gates=True)
        assert any(c.kind == "noisy_or" for c in soft.cpts.values())

    def test_bayesnet_names_include_the_study_and_hazard(self, study):
        assert "H2" in study.bayesnet("H2").name
        assert study.name in study.bayesnet("H2").name

    def test_hazards_found_is_sorted(self, study):
        assert study.hazards_found() == sorted(study.hazards_found())


class TestSummary:
    def test_an_uncalibrated_summary_shouts(self, fresh):
        assert "NOT CALIBRATED" in fresh.summary()

    def test_a_profile_without_calibration_is_still_shown(self, parallel_spec):
        study = AgenticReliabilityStudy(parallel_spec, profile=PROFILE)
        summary = study.summary()
        assert "operational profile" in summary
        assert "NOT CALIBRATED" in summary

    def test_a_system_level_result_is_summarised(self, fresh):
        fresh.observe(
            [1, 1, 0, 1, 0, 0, 1, 0],
            ["short"] * 4 + ["long"] * 4,
            profile={"short": 0.30, "long": 0.70},
        )
        summary = fresh.summary()
        assert "System-level operational failure probability" in summary
        assert "empirical_operational_failure_probability" in summary

    def test_the_calibrated_summary_carries_the_evidence(self, calibrated):
        summary = calibrated.summary()
        assert "Calibration" in summary
        assert "updated from measurement" in summary


class TestCalibrationOptions:
    def test_the_policy_can_be_changed(self, parallel_spec, outcomes):
        study = AgenticReliabilityStudy(parallel_spec, exact_inference=False)
        study.observe(outcomes, profile=PROFILE)
        study.calibrate(policy="dominant")
        touched = [
            e
            for e in study.failure_model.events.values()
            if e.component == "react_agent" and e.prob_interval is not None
        ]
        # "dominant" puts everything on one event and zero on the rest
        assert sum(1 for e in touched if e.prob > 0) == 1

    def test_the_bound_choice_reaches_the_envelope(self, parallel_spec, outcomes):
        """The three envelopes come from HIP-LLM's hyperparameter family, so the
        choice only means something on the exact path."""
        wide = AgenticReliabilityStudy(parallel_spec, bound="credible")
        tight = AgenticReliabilityStudy(parallel_spec, bound="expected")
        for study in (wide, tight):
            study.observe(outcomes, profile=PROFILE).run()
        assert wide.hazard_probability("H2").width > tight.hazard_probability(
            "H2"
        ).width

    def test_the_approximation_says_the_bound_does_not_apply(
        self, parallel_spec, outcomes
    ):
        """Accepting an option and ignoring it is exactly the silent behaviour
        this package exists to avoid."""
        study = AgenticReliabilityStudy(
            parallel_spec, exact_inference=False, bound="expected"
        )
        study.observe(outcomes, profile=PROFILE).run()
        method = next(iter(study.evidence.values())).method
        assert "does not apply" in method

    def test_calibration_is_recorded_on_the_failure_model_notes(self, calibrated):
        notes = " ".join(calibrated.failure_model.notes)
        assert "calibrated from" in notes

    def test_recalibrating_does_not_compound(self, parallel_spec, outcomes):
        """Calling calibrate() twice must land on the same numbers, not apply
        the union split to already-split events."""
        study = AgenticReliabilityStudy(parallel_spec, exact_inference=False)
        study.observe(outcomes, profile=PROFILE)
        study.calibrate()
        first = {e.id: e.prob for e in study.failure_model.events.values()}
        study.calibrate()
        second = {e.id: e.prob for e in study.failure_model.events.values()}
        for key in first:
            assert second[key] == pytest.approx(first[key]), key


class TestSave:
    def test_calibration_artefacts_are_written_only_when_calibrated(
        self, study, calibrated, tmp_path
    ):
        plain = {p.replace("\\", "/").rsplit("/", 1)[-1] for p in study.save(str(tmp_path / "a"))}
        full = {p.replace("\\", "/").rsplit("/", 1)[-1] for p in calibrated.save(str(tmp_path / "b"))}
        assert "calibration.csv" not in plain
        assert {"calibration.csv", "evidence.csv"} <= full

    def test_the_evidence_csv_records_sample_sizes(self, calibrated, tmp_path):
        calibrated.save(str(tmp_path))
        frame = pd.read_csv(tmp_path / "evidence.csv")
        assert {"component", "n", "failures", "lower", "upper"} <= set(frame.columns)
        assert (frame["n"] > 0).all()
        assert (frame["lower"] <= frame["upper"]).all()

    def test_the_calibration_csv_shows_before_and_after(self, calibrated, tmp_path):
        calibrated.save(str(tmp_path))
        frame = pd.read_csv(tmp_path / "calibration.csv")
        assert {"basic event", "placeholder P", "calibrated P"} <= set(frame.columns)
        assert not np.allclose(frame["placeholder P"], frame["calibrated P"])
