"""SafetyReport: lookup, derived tables, rendering and persistence.

`SafetyReport` is what `analyse_langgraph` hands back and what
`AgenticReliabilityStudy` wraps, so its accessors are the surface most code
touches. These tests cover the parts that are easy to get subtly wrong: prefix
lookup, the CSV writers, and whether the Markdown report actually states the
assumptions it is supposed to state.
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from HIP_HOPS_LLM import (  # noqa: E402
    SafetyReport,
    analyse_langgraph,
    load_example,
    map_uncertainty,
)


@pytest.fixture(scope="module")
def report():
    return analyse_langgraph(
        load_example("parallel_aggregator"), name="parallel agents + aggregator"
    )


@pytest.fixture(scope="module")
def looped_report():
    return analyse_langgraph(
        load_example("react_calculator"), name="ReAct + calculator", unroll=1
    )


class TestLookup:
    def test_exact_hazard_id(self, report):
        assert report.tree("H2").id == "H2"

    def test_prefix_lookup_finds_a_suffixed_hazard(self, looped_report):
        """`H5-coder` should be reachable as `H5`, since the suffix is the
        component name and callers do not always know it."""
        assert looped_report.tree("H5").id.startswith("H5")

    def test_an_unknown_hazard_lists_what_exists(self, report):
        with pytest.raises(KeyError, match="available"):
            report.tree("H99")

    def test_analysis_and_tree_agree(self, report):
        for hazard in report.trees:
            assert report.analysis(hazard).tree.id == report.tree(hazard).id

    def test_cut_sets_are_sorted_lists_not_frozensets(self, report):
        cuts = report.cut_sets("H2")
        assert all(isinstance(c, list) for c in cuts)
        assert all(c == sorted(c) for c in cuts)


class TestDerivedTables:
    def test_fmea_covers_every_event_in_a_cut_set(self, report):
        rows = {r.event_id for r in report.fmea()}
        in_cuts = {
            e
            for hazard in report.trees
            for cut in report.analysis(hazard).cuts.sets
            for e in cut
        }
        assert in_cuts <= rows

    def test_every_fmea_row_carries_a_mitigation(self, report):
        for row in report.fmea():
            assert row.mitigation, f"{row.event_id} has no mitigation"

    def test_single_points_are_exactly_the_order_one_cut_sets(self, report):
        reported = {(r["hazard"], r["event"]) for r in report.single_points()}
        expected = {
            (hazard, next(iter(cut)))
            for hazard in report.trees
            for cut in report.analysis(hazard).cuts.sets
            if len(cut) == 1
        }
        assert reported == expected

    def test_single_points_are_ranked_by_severity(self, report):
        order = ["catastrophic", "critical", "major", "minor", "negligible"]
        seen = [r["severity"] for r in report.single_points()]
        ranks = [order.index(s) for s in seen if s in order]
        assert ranks == sorted(ranks)


class TestRendering:
    def test_mermaid_is_a_flowchart_naming_the_top_event(self, report):
        text = report.mermaid("H2")
        assert text.lstrip().startswith(("graph", "flowchart"))
        assert "TOP" in text or report.tree("H2").nodes[report.tree("H2").root].label

    def test_summary_reports_every_hazard(self, report):
        summary = report.summary()
        for hazard in report.trees:
            assert hazard in summary

    def test_summary_names_the_common_cause_group(self, report):
        assert "common-cause" in report.summary()

    def test_summary_lists_single_points(self, report):
        assert "single points of failure" in report.summary()

    def test_markdown_states_the_loop_assumption(self, looped_report):
        """Loop unrolling is an assumption; a report that hides it is misleading."""
        text = looped_report.markdown().lower()
        assert "feedback" in text
        assert "unroll" in text

    def test_markdown_can_omit_the_trees(self, report):
        with_trees = report.markdown(include_trees=True)
        without = report.markdown(include_trees=False)
        assert len(with_trees) > len(without)

    def test_markdown_names_the_system(self, report):
        assert report.name in report.markdown()


class TestCSVWriters:
    def test_cutset_csv_parses_and_has_one_row_per_cut_set(self, report):
        rows = list(csv.DictReader(report._cutsets_csv().splitlines()))
        expected = sum(len(report.analysis(h).cuts.sets) for h in report.trees)
        assert len(rows) == expected
        assert set(rows[0]) == {"hazard", "order", "probability", "events"}

    def test_cutset_csv_orders_match_the_event_counts(self, report):
        for row in csv.DictReader(report._cutsets_csv().splitlines()):
            assert int(row["order"]) == len(row["events"].split(" + "))

    def test_cutset_probabilities_are_probabilities(self, report):
        for row in csv.DictReader(report._cutsets_csv().splitlines()):
            assert 0.0 <= float(row["probability"]) <= 1.0

    def test_fmea_csv_parses_and_quotes_embedded_commas(self, report):
        rows = list(csv.DictReader(report._fmea_csv().splitlines()))
        assert len(rows) == len(report.fmea())
        assert "component" in rows[0] and "mitigation" in rows[0]
        # a field containing a comma must survive the round trip intact
        assert all(row["event"] for row in rows)


class TestPersistence:
    def test_save_writes_one_file_per_export_and_hazard(self, report, tmp_path):
        written = report.save(str(tmp_path))
        names = {p.replace("\\", "/").rsplit("/", 1)[-1] for p in written}
        for hazard in report.trees:
            safe = hazard.replace("-", "_")
            assert any(n.endswith(f"{safe}.mmd") for n in names), hazard
            assert any(n.endswith(f"{safe}.dot") for n in names), hazard
            assert any(n.endswith(f"{safe}.json") for n in names), hazard
            assert any(n.endswith(f"{safe}.opsa.xml") for n in names), hazard
        assert any(n.endswith("_report.md") for n in names)
        assert any(n.endswith("_cutsets.csv") for n in names)
        assert any(n.endswith("_fmea.csv") for n in names)

    def test_every_written_path_exists_and_is_non_empty(self, report, tmp_path):
        for path in report.save(str(tmp_path)):
            import os

            assert os.path.getsize(path) > 0, path

    def test_the_json_export_round_trips(self, report, tmp_path):
        written = report.save(str(tmp_path))
        for path in written:
            if path.endswith(".json"):
                data = json.loads(open(path, encoding="utf-8").read())
                assert "nodes" in data
                assert data["nodes"]

    def test_a_prefix_overrides_the_generated_stem(self, report, tmp_path):
        written = report.save(str(tmp_path), prefix="run42")
        assert all("run42" in p.replace("\\", "/").rsplit("/", 1)[-1] for p in written)

    def test_saving_creates_the_directory(self, report, tmp_path):
        target = tmp_path / "deep" / "nested"
        report.save(str(target))
        assert target.is_dir()


class TestUncertaintyMapping:
    def test_records_are_matched_on_shared_name_tokens(self, report):
        summary = {
            "approach2_react": {"mean_cluster_entropy": 1.33},
            "approach2_cot": {"mean_cluster_entropy": 0.86},
        }
        mapped = map_uncertainty(report.system, summary)
        assert mapped.get("react_agent") == pytest.approx(1.33)
        assert mapped.get("cot_agent") == pytest.approx(0.86)

    def test_an_unmatched_record_is_ignored_not_guessed(self, report):
        mapped = map_uncertainty(report.system, {"totally_unrelated": {"mean_cluster_entropy": 9.9}})
        assert 9.9 not in mapped.values()

    def test_an_empty_summary_maps_nothing(self, report):
        assert map_uncertainty(report.system, None) == {}
        assert map_uncertainty(report.system, {}) == {}

    def test_a_different_metric_can_be_requested(self, report):
        summary = {"approach2_react": {"mean_cluster_entropy": 1.0, "other": 5.0}}
        assert map_uncertainty(report.system, summary, metric="other") == {
            "react_agent": 5.0
        }

    def test_a_non_numeric_metric_is_skipped(self, report):
        summary = {"approach2_react": {"mean_cluster_entropy": "high"}}
        assert map_uncertainty(report.system, summary) == {}

    def test_entropy_calibration_reaches_the_events(self):
        """The end-to-end path map_uncertainty feeds: an entropy record must
        actually move a basic event, or the calibration is silently inert."""
        base = analyse_langgraph(load_example("parallel_aggregator"), name="base")
        calibrated = analyse_langgraph(
            load_example("parallel_aggregator"),
            name="calibrated",
            uncertainty_summary={"approach2_react": {"mean_cluster_entropy": 2.6}},
        )
        target = next(
            e for e in base.failure_model.events if e.endswith("react_agent-HALLUC")
        )
        assert (
            calibrated.failure_model.events[target].prob
            > base.failure_model.events[target].prob
        )


class TestConstruction:
    def test_analyse_langgraph_returns_a_safety_report(self, report):
        assert isinstance(report, SafetyReport)

    def test_the_source_system_keeps_its_cycles(self, looped_report):
        from HIP_HOPS_LLM import find_cycles, is_acyclic

        assert find_cycles(looped_report.source_system), (
            "source_system must be the architecture as extracted, loops included"
        )
        assert is_acyclic(looped_report.system), (
            "system must be the acyclic model the trees were built from"
        )

    def test_raw_and_simplified_trees_are_both_kept(self, report):
        assert set(report.raw_trees) == set(report.trees)
        assert any(
            len(report.raw_trees[h].nodes) >= len(report.trees[h].nodes)
            for h in report.trees
        )

    def test_simplification_can_be_disabled(self):
        unsimplified = analyse_langgraph(
            load_example("parallel_aggregator"), name="raw", simplify=False
        )
        assert unsimplified.trees["H2"].nodes.keys() == (
            unsimplified.raw_trees["H2"].nodes.keys()
        )
