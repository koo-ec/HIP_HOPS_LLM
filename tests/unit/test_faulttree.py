"""Failure annotation, fault tree synthesis, cut sets and quantification."""

from __future__ import annotations

import itertools

import pytest

from HIP_HOPS_LLM import (
    AND,
    OR,
    AgenticReliabilityStudy,
    FClass,
    annotate_system,
    cut_sets,
    default_hazards,
    extract_architecture,
    make_acyclic,
    synthesise_all,
    to_dot,
    to_json,
    to_mermaid,
    to_openpsa_xml,
)


@pytest.fixture(scope="module")
def annotated(request):
    spec = request.getfixturevalue("parallel_spec")
    model, _ = make_acyclic(extract_architecture(spec), unroll=1)
    return model, annotate_system(model)


class TestAnnotation:
    def test_every_non_boundary_component_has_logic(self, annotated):
        model, fmodel = annotated
        for cid, component in model.components.items():
            assert cid in fmodel.logic, f"{cid} has no failure logic"

    def test_llm_agents_carry_a_hallucination_event(self, annotated):
        _, fmodel = annotated
        halluc = [
            e
            for e in fmodel.events.values()
            if e.component == "react_agent" and "HALLUC" in e.id
        ]
        assert halluc
        assert halluc[0].fclass is FClass.VALUE_SUBTLE

    def test_shared_snapshot_produces_a_common_cause_event(self, annotated):
        _, fmodel = annotated
        ccf = [e for e in fmodel.events.values() if e.kind == "ccf"]
        assert ccf, "a shared model snapshot must produce a common-cause event"

    def test_every_event_carries_its_provenance(self, annotated):
        _, fmodel = annotated
        for event in fmodel.events.values():
            assert event.evidence, f"{event.id} has no evidence string"

    def test_default_probabilities_are_declared_as_placeholders(self, annotated):
        _, fmodel = annotated
        placeholders = [
            e for e in fmodel.events.values() if "placeholder" in e.evidence.lower()
        ]
        assert placeholders, (
            "uncalibrated events must say so; a silent default is the failure "
            "mode this package exists to prevent"
        )

    def test_probability_overrides_are_applied(self, annotated):
        model, base = annotated
        target = next(e.id for e in base.events.values() if "HALLUC" in e.id)
        fmodel = annotate_system(model, probability_overrides={target: 0.42})
        assert fmodel.events[target].prob == pytest.approx(0.42)
        assert "override" in fmodel.events[target].evidence

    def test_an_unknown_override_id_is_refused(self, annotated):
        """A mistyped id must not silently leave the placeholder in place."""
        model, _ = annotated
        with pytest.raises(KeyError, match="does not have"):
            annotate_system(model, probability_overrides={"BE-nope-NOPE": 0.4})

    def test_an_out_of_range_override_is_refused(self, annotated):
        model, base = annotated
        target = next(iter(base.events))
        with pytest.raises(ValueError, match="not a probability"):
            annotate_system(model, probability_overrides={target: 1.4})

    def test_entropy_calibration_moves_the_probability(self, annotated):
        model, base = annotated
        calibrated = annotate_system(model, entropy_by_component={"react_agent": 2.5})
        before = next(e for e in base.events.values() if e.id.endswith("react_agent-HALLUC"))
        after = next(e for e in calibrated.events.values() if e.id == before.id)
        assert after.prob > before.prob


class TestBooleanAlgebra:
    def test_or_flattens_and_dedups(self):
        from HIP_HOPS_LLM.faulttree.failure import BasicEventRef

        a, b = BasicEventRef("a"), BasicEventRef("b")
        assert str(OR(a, OR(a, b))) == str(OR(a, b))

    def test_and_of_one_term_is_that_term(self):
        from HIP_HOPS_LLM.faulttree.failure import BasicEventRef

        a = BasicEventRef("a")
        assert AND(a) is a or str(AND(a)) == "a"


class TestSynthesis:
    def test_a_tree_is_produced_for_each_hazard(self, annotated):
        model, fmodel = annotated
        hazards = default_hazards(model)
        trees = synthesise_all(fmodel, hazards)
        assert set(trees) == {h.id for h in hazards}

    def test_trees_are_acyclic(self, study):
        for tree in study.report.trees.values():
            tree.verify_acyclic()

    def test_every_leaf_names_a_basic_event(self, study):
        for tree in study.report.trees.values():
            for node in tree.nodes.values():
                if node.ntype == "basic":
                    assert node.event_id, f"{node.id} is a basic event with no id"

    def test_simplification_preserves_the_cut_sets(self, study):
        report = study.report
        for hid, tree in report.trees.items():
            raw = report.raw_trees[hid]
            simplified_sets = {
                frozenset(s) for s in cut_sets(tree, report.failure_model).sets
            }
            raw_sets = {frozenset(s) for s in cut_sets(raw, report.failure_model).sets}
            assert simplified_sets == raw_sets, (
                f"{hid}: simplification changed the Boolean function"
            )


class TestCutSets:
    def test_cut_sets_are_minimal(self, study):
        for hid in study.report.trees:
            sets = [frozenset(s) for s in study.report.analysis(hid).cuts.sets]
            for a, b in itertools.permutations(sets, 2):
                assert not a < b, f"{hid}: {sorted(a)} is a subset of {sorted(b)}"

    def test_every_cut_set_actually_causes_the_top_event(self, study):
        """A cut set must make the Boolean function true; verify by evaluation."""
        report = study.report
        for hid, tree in report.trees.items():
            analysis = report.analysis(hid)
            for cs in list(analysis.cuts.sets)[:20]:
                assert _evaluate(tree, set(cs)), (
                    f"{hid}: cut set {sorted(cs)} does not produce the top event"
                )

    def test_removing_any_event_breaks_a_cut_set(self, study):
        report = study.report
        tree = report.tree("H2")
        for cs in list(report.analysis("H2").cuts.sets)[:15]:
            for dropped in cs:
                remaining = set(cs) - {dropped}
                assert not _evaluate(tree, remaining), (
                    f"cut set {sorted(cs)} is not minimal: {sorted(remaining)} "
                    "already causes the top event"
                )

    def test_parallel_agents_raise_the_cut_set_order(self, study, react_spec):
        """The vote is real: most H2 cut sets need both agents to be wrong."""
        parallel_orders = [len(s) for s in study.report.analysis("H2").cuts.sets]
        serial = AgenticReliabilityStudy(react_spec, name="serial")
        serial.analyse()
        serial_orders = [len(s) for s in serial.report.analysis("H2").cuts.sets]
        assert max(parallel_orders) > max(serial_orders)
        assert all(o == 1 for o in serial_orders), (
            "the serial architecture has no redundancy anywhere"
        )

    def test_the_shared_snapshot_is_still_an_order_one_cut_set(self, study):
        """The redundancy is architectural only: both agents share a snapshot."""
        order_1 = {
            next(iter(s)) for s in study.report.analysis("H2").cuts.sets if len(s) == 1
        }
        assert any(e.startswith("CCF-LLM") for e in order_1), (
            "a shared model snapshot must defeat the vote as an order-1 cut set"
        )


class TestQuantification:
    def test_probabilities_are_in_range(self, study):
        for hid in study.report.trees:
            p = study.report.analysis(hid).quant.top_probability
            assert 0.0 <= p <= 1.0

    def test_mcub_is_bounded_by_the_rare_event_sum(self, study):
        for hid in study.report.trees:
            quant = study.report.analysis(hid).quant
            assert quant.top_probability <= quant.rare_event_sum + 1e-12

    def test_importance_ranks_every_event_in_a_cut_set(self, study):
        analysis = study.report.analysis("H2")
        ranked = {row.event_id for row in analysis.importance}
        in_cuts = {e for s in analysis.cuts.sets for e in s}
        assert in_cuts <= ranked


class TestExports:
    def test_mermaid_dot_json_and_openpsa_are_produced(self, study):
        tree = study.report.tree("H2")
        assert to_mermaid(tree).startswith("graph")
        assert "digraph" in to_dot(tree)
        assert '"nodes"' in to_json(tree)
        xml = to_openpsa_xml(tree, name="t")
        assert xml.lstrip().startswith("<?xml") or "<opsa-mef" in xml

    def test_fmea_covers_every_basic_event_in_a_cut_set(self, study):
        rows = {r.event_id for r in study.report.fmea()}
        in_cuts = {
            e
            for hid in study.report.trees
            for s in study.report.analysis(hid).cuts.sets
            for e in s
        }
        assert in_cuts <= rows

    def test_markdown_report_mentions_the_loop_handling(self, react_spec):
        study = AgenticReliabilityStudy(react_spec, name="r")
        study.analyse()
        text = study.report.markdown()
        assert "feedback" in text.lower()


def _evaluate(tree, failed: set) -> bool:
    """Evaluate the tree's Boolean function with exactly ``failed`` set true."""
    memo: dict = {}

    def value(nid: str) -> bool:
        if nid in memo:
            return memo[nid]
        node = tree.nodes[nid]
        if node.ntype in ("basic", "undeveloped"):
            out = node.event_id in failed
        elif node.ntype == "house":
            out = True
        elif not node.children:
            out = False
        elif (node.gate or "OR").upper() == "AND":
            out = all(value(c) for c in node.children)
        else:
            out = any(value(c) for c in node.children)
        memo[nid] = out
        return out

    return value(tree.root)
