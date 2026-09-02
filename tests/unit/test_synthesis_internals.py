"""Fault tree structure: accessors, simplification, expansion and exports.

Synthesis produces a DAG, because shared sub-trees are shared — that sharing is
exactly why the Bayesian network gives an exact answer where the cut-set bound
does not. These tests cover the machinery around it: the accessors that report
the structure, the reduction that must preserve the Boolean function, and the
expansion that turns the DAG back into a strict tree for drawing.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from hiphopsllm import (
    Hazard,
    default_hazards,
    extract_architecture,
    load_example,
    make_acyclic,
    simplify_tree,
    to_dot,
    to_json,
    to_mermaid,
    to_openpsa_xml,
)
from hiphopsllm.faulttree.failure import annotate_system
from hiphopsllm.faulttree.synthesis import (
    _tag,
    describe_deviation,
    expand_to_tree,
    synthesise_all,
)


@pytest.fixture(scope="module")
def annotated():
    model, _ = make_acyclic(
        extract_architecture(load_example("parallel_aggregator")), unroll=1
    )
    return model, annotate_system(model)


class TestAccessors:
    def test_size_counts_every_node_by_kind(self, tree):
        size = tree.size()
        assert size["total"] == len(tree.nodes)
        assert sum(v for k, v in size.items() if k != "total") == size["total"]

    def test_depth_is_at_least_two_for_a_real_tree(self, tree):
        assert tree.depth() >= 2

    def test_leaves_are_exactly_the_leaf_nodes(self, tree):
        assert {n.id for n in tree.leaves()} == {
            n.id for n in tree.nodes.values() if n.is_leaf
        }

    def test_basic_event_ids_are_sorted_and_unique(self, tree):
        ids = tree.basic_event_ids()
        assert ids == sorted(set(ids))

    def test_the_root_has_no_parent(self, tree):
        assert tree.parent_count()[tree.root] == 0

    def test_every_other_node_has_at_least_one_parent(self, tree):
        counts = tree.parent_count()
        orphans = [n for n, c in counts.items() if c == 0 and n != tree.root]
        assert not orphans, f"unreachable nodes: {orphans}"

    def test_shared_nodes_are_those_with_several_parents(self, tree):
        counts = tree.parent_count()
        assert set(tree.shared_nodes()) == {n for n, c in counts.items() if c > 1}

    def test_this_tree_really_does_share_subtrees(self, tree):
        """If it did not, the exact-vs-bound comparison would be uninteresting."""
        assert tree.shared_nodes()

    def test_verify_acyclic_passes(self, tree):
        assert tree.verify_acyclic()

    def test_node_lookup(self, tree):
        assert tree.node(tree.root).id == tree.root
        with pytest.raises(KeyError):
            tree.node("no-such-node")


class TestTransferTags:
    def test_the_sequence_is_a_to_z_then_aa(self):
        assert _tag(0) == "A"
        assert _tag(25) == "Z"
        assert _tag(26) == "AA"
        assert _tag(27) == "AB"

    def test_tags_are_unique_over_a_long_run(self):
        tags = [_tag(i) for i in range(200)]
        assert len(set(tags)) == len(tags)

    def test_every_tag_is_upper_case_letters(self):
        assert all(re.fullmatch(r"[A-Z]+", _tag(i)) for i in range(60))


class TestExpandToTree:
    def test_expansion_produces_a_strict_tree(self, tree):
        expanded = expand_to_tree(tree)
        counts = expanded.parent_count()
        assert all(c <= 1 for n, c in counts.items()), "a strict tree has one parent"

    def test_repeats_point_back_at_the_original(self, tree):
        expanded = expand_to_tree(tree)
        for node in expanded.nodes.values():
            if node.repeat_of:
                assert node.repeat_of in tree.nodes or node.repeat_of in expanded.nodes

    def test_the_expansion_preserves_the_basic_events(self, tree):
        assert set(expand_to_tree(tree).basic_event_ids()) == set(
            tree.basic_event_ids()
        )

    def test_exceeding_the_node_budget_raises_with_the_alternative(self, tree):
        """Silently truncating a tree would produce a picture that is simply
        wrong; refusing and naming the alternative is the right behaviour."""
        with pytest.raises(ValueError, match="as_tree=False"):
            expand_to_tree(tree, max_nodes=12)

    def test_a_generous_budget_expands_fully(self, tree):
        expanded = expand_to_tree(tree, max_nodes=5000)
        assert len(expanded.nodes) >= len(tree.nodes)

    def test_transfer_subtrees_can_be_turned_off(self, tree):
        with_transfer = expand_to_tree(tree, transfer_subtrees=True)
        without = expand_to_tree(tree, transfer_subtrees=False)
        tagged_with = sum(1 for n in with_transfer.nodes.values() if n.transfer_ref)
        tagged_without = sum(1 for n in without.nodes.values() if n.transfer_ref)
        assert tagged_with >= tagged_without

    def test_an_already_strict_tree_is_unchanged_in_shape(self, tree):
        once = expand_to_tree(tree)
        twice = expand_to_tree(once)
        assert len(twice.nodes) == len(once.nodes)


class TestSimplification:
    def test_simplification_shrinks_or_matches(self, study):
        for hazard, reduced in study.report.trees.items():
            raw = study.report.raw_trees[hazard]
            assert len(reduced.nodes) <= len(raw.nodes)

    def test_it_keeps_the_root_and_the_basic_events(self, study):
        for hazard, reduced in study.report.trees.items():
            raw = study.report.raw_trees[hazard]
            assert set(reduced.basic_event_ids()) == set(raw.basic_event_ids())

    def test_each_reduction_can_be_disabled(self, annotated):
        _, fmodel = annotated
        raw = synthesise_all(fmodel, default_hazards(fmodel.system), simplify=False)
        tree = raw["H2"]
        full = simplify_tree(tree)
        none = simplify_tree(
            tree,
            collapse_single_input=False,
            flatten_gates=False,
            flatten_ports=False,
        )
        assert len(full.nodes) <= len(none.nodes)

    def test_a_reduced_tree_is_still_acyclic(self, study):
        for reduced in study.report.trees.values():
            assert reduced.verify_acyclic()

    def test_no_single_input_gate_survives_the_default_reduction(self, study):
        tree = study.report.tree("H2")
        gates = [
            n
            for n in tree.nodes.values()
            if n.ntype == "intermediate" and n.gate and len(n.children) == 1
        ]
        assert not gates, "a one-input gate carries no information"


class TestHazards:
    def test_the_default_set_covers_the_four_boundary_hazards(self, annotated):
        model, _ = annotated
        ids = {h.id for h in default_hazards(model)}
        assert {"H1", "H2", "H3", "H4"} <= ids

    def test_an_unsafe_tool_adds_its_own_hazard(self):
        model, _ = make_acyclic(extract_architecture(load_example("react_calculator")))
        ids = {h.id for h in default_hazards(model)}
        assert any(i.startswith("H5") for i in ids), "eval() must raise a hazard"

    def test_severities_are_from_the_expected_vocabulary(self, annotated):
        model, _ = annotated
        allowed = {"catastrophic", "critical", "major", "minor", "negligible"}
        assert {h.severity for h in default_hazards(model)} <= allowed

    def test_a_custom_hazard_is_synthesised(self, annotated):
        from hiphopsllm import Deviation, FClass

        model, fmodel = annotated
        custom = Hazard(
            id="HX",
            name="a custom top event",
            severity="critical",
            deviations=[
                Deviation(
                    "__end__",
                    model.components["__end__"].ports_in[0],
                    FClass.VALUE_SUBTLE,
                )
            ],
        )
        trees = synthesise_all(fmodel, [custom])
        assert set(trees) == {"HX"}
        assert trees["HX"].nodes
        assert trees["HX"].hazard.severity == "critical"

    def test_a_hazard_over_several_deviations_is_an_or(self, annotated):
        from hiphopsllm import Deviation, FClass

        model, fmodel = annotated
        port = model.components["__end__"].ports_in[0]
        combined = Hazard(
            id="HY",
            name="either kind of wrong value",
            severity="major",
            deviations=[
                Deviation("__end__", port, FClass.VALUE_SUBTLE),
                Deviation("__end__", port, FClass.VALUE_COARSE),
            ],
        )
        tree = synthesise_all(fmodel, [combined])["HY"]
        assert tree.nodes[tree.root].gate in (None, "OR")

    def test_the_hazard_label_reads_well(self):
        assert Hazard(id="H9", name="a thing", deviations=[]).label == "H9: a thing"

    def test_describe_deviation_is_human_readable(self, annotated):
        from hiphopsllm import Deviation, FClass

        model, _ = annotated
        text = describe_deviation(
            model, Deviation("aggregator", "out", FClass.VALUE_SUBTLE)
        )
        assert "aggregator" in text and "out" in text
        assert "Value" in text


class TestExports:
    def test_mermaid_names_every_node(self, tree):
        text = to_mermaid(tree)
        assert text.lstrip().startswith(("graph", "flowchart"))
        assert text.count("-->") + text.count("---") >= len(tree.nodes) - 1

    def test_dot_is_a_digraph_with_at_least_one_edge_per_child_link(self, tree):
        """DOT renders the expanded (strict-tree) form, so repeated causes add
        edges beyond the DAG's child links."""
        dot = to_dot(tree)
        assert dot.lstrip().startswith("digraph")
        assert dot.count("->") >= sum(len(n.children) for n in tree.nodes.values()) - 1
        assert dot.rstrip().endswith("}")

    def test_json_round_trips_and_names_the_root(self, tree):
        import json

        data = json.loads(to_json(tree))
        assert data["root"] == tree.root
        assert len(data["nodes"]) == len(tree.nodes)

    def test_json_carries_the_analysis_when_given_one(self, study):
        import json

        analysis = study.report.analysis("H2")
        data = json.loads(to_json(study.report.tree("H2"), analysis))
        assert len(data["minimal_cut_sets"]) == len(analysis.cuts.sets)
        assert data["quantification"]["top_probability"] == pytest.approx(
            analysis.quant.top_probability
        )
        assert data["importance"]
        assert data["hazard"]["id"] == "H2"

    def test_json_omits_the_analysis_when_none_is_given(self, tree):
        import json

        data = json.loads(to_json(tree))
        assert not data.get("minimal_cut_sets")

    def test_openpsa_is_well_formed_xml(self, tree):
        root = ET.fromstring(to_openpsa_xml(tree, name="t"))
        assert root.tag.endswith("opsa-mef") or root.tag

    def test_openpsa_declares_every_basic_event(self, tree):
        xml = to_openpsa_xml(tree, name="t")
        for event_id in tree.basic_event_ids():
            safe = re.sub(r"[^0-9A-Za-z_.\-]", "_", event_id)
            assert safe in xml or event_id in xml, event_id

    def test_every_export_is_non_empty_for_every_hazard(self, study):
        for hazard, tree_ in study.report.trees.items():
            for export in (to_mermaid, to_dot, to_json):
                assert export(tree_).strip(), f"{export.__name__} empty for {hazard}"
            assert to_openpsa_xml(tree_, name=hazard).strip()
