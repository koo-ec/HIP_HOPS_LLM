"""Conditional probability tables built from fault trees."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from hiphopsllm.bayes.cpt import (
    FAIL,
    MAX_GATE_INPUTS,
    OK,
    CPT,
    CPTBuilder,
    GateType,
    deterministic_gate_cpt,
    fault_tree_to_cpts,
    k_of_n_cpt,
    noisy_or_cpt,
    prior_cpt,
    sanitise,
)


class TestGateTables:
    def test_or_fails_exactly_when_any_input_fails(self):
        table = deterministic_gate_cpt(3, "OR")
        for combo in itertools.product((0, 1), repeat=3):
            assert table[combo][FAIL] == float(any(combo))

    def test_and_fails_exactly_when_all_inputs_fail(self):
        table = deterministic_gate_cpt(3, "AND")
        for combo in itertools.product((0, 1), repeat=3):
            assert table[combo][FAIL] == float(all(combo))

    def test_tables_are_normalised(self):
        for n in (1, 2, 4):
            for gate in ("AND", "OR"):
                table = deterministic_gate_cpt(n, gate)
                assert np.allclose(table.sum(axis=-1), 1.0)

    def test_k_of_n_generalises_and_and_or(self):
        n = 4
        assert np.array_equal(k_of_n_cpt(n, 1), deterministic_gate_cpt(n, "OR"))
        assert np.array_equal(k_of_n_cpt(n, n), deterministic_gate_cpt(n, "AND"))

    def test_two_of_three_is_a_majority_vote(self):
        table = k_of_n_cpt(3, 2)
        assert table[(1, 1, 0)][FAIL] == 1.0
        assert table[(1, 0, 0)][FAIL] == 0.0

    def test_k_out_of_range_is_rejected(self):
        with pytest.raises(ValueError, match="1 <= k <= n"):
            k_of_n_cpt(3, 4)

    def test_wide_gates_are_refused_with_an_explanation(self):
        with pytest.raises(ValueError, match="cut sets"):
            deterministic_gate_cpt(MAX_GATE_INPUTS + 1, "OR")


class TestNoisyOr:
    def test_unit_links_and_no_leak_reproduce_the_deterministic_or(self):
        soft = noisy_or_cpt([1.0, 1.0, 1.0], leak=0.0)
        assert np.allclose(soft, deterministic_gate_cpt(3, "OR"))

    def test_the_leak_fires_with_every_input_ok(self):
        table = noisy_or_cpt([0.5, 0.5], leak=0.02)
        assert table[(0, 0)][FAIL] == pytest.approx(0.02)

    def test_a_single_failed_input_propagates_with_its_link_probability(self):
        table = noisy_or_cpt([0.7, 0.3], leak=0.0)
        assert table[(1, 0)][FAIL] == pytest.approx(0.7)
        assert table[(0, 1)][FAIL] == pytest.approx(0.3)

    def test_two_failed_inputs_combine_independently(self):
        table = noisy_or_cpt([0.7, 0.3], leak=0.0)
        assert table[(1, 1)][FAIL] == pytest.approx(1 - 0.3 * 0.7)

    def test_soft_is_never_more_likely_than_deterministic(self):
        soft = noisy_or_cpt([0.6, 0.6, 0.6], leak=0.0)
        hard = deterministic_gate_cpt(3, "OR")
        assert np.all(soft[..., FAIL] <= hard[..., FAIL] + 1e-12)

    def test_invalid_link_probabilities_are_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            noisy_or_cpt([0.5, 1.5])


class TestPriors:
    def test_prior_is_ok_then_fail(self):
        assert prior_cpt(0.25)[OK] == pytest.approx(0.75)
        assert prior_cpt(0.25)[FAIL] == pytest.approx(0.25)

    def test_out_of_range_prior_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            prior_cpt(1.5)


class TestCPTValidation:
    def test_shape_is_checked(self):
        with pytest.raises(ValueError, match="expected"):
            CPT(variable="v", parents=("a",), table=np.array([0.5, 0.5]))

    def test_normalisation_is_checked(self):
        with pytest.raises(ValueError, match="not normalised"):
            CPT(variable="v", parents=(), table=np.array([0.5, 0.4]))

    def test_rows_render_parent_states(self):
        cpt = CPT(
            variable="g", parents=("a", "b"), table=deterministic_gate_cpt(2, "AND")
        )
        rows = cpt.rows()
        assert len(rows) == 4
        assert {"a": "Fail", "b": "Fail", "P(Fail)": 1.0} in rows


class TestConversion:
    def test_every_tree_node_becomes_a_variable(self, tree, failure_model):
        cpts = fault_tree_to_cpts(tree, failure_model)
        assert len(cpts) == len(tree.nodes)
        assert cpts.top == cpts.variable_of[tree.root]

    def test_the_order_is_topological(self, tree, failure_model):
        cpts = fault_tree_to_cpts(tree, failure_model)
        seen: set = set()
        for var in cpts.order:
            for parent in cpts[var].parents:
                assert parent in seen, f"{var} precedes its parent {parent}"
            seen.add(var)

    def test_basic_events_become_roots_with_their_probability(
        self, tree, failure_model
    ):
        cpts = fault_tree_to_cpts(tree, failure_model)
        for event_id, var in cpts.event_variable.items():
            assert cpts[var].is_root
            expected = failure_model.events[event_id].prob
            assert cpts[var].prior_fail == pytest.approx(expected)

    def test_gates_keep_their_semantics(self, tree, failure_model):
        cpts = fault_tree_to_cpts(tree, failure_model)
        for nid, node in tree.nodes.items():
            if node.ntype not in ("basic", "undeveloped", "house") and node.children:
                cpt = cpts[cpts.variable_of[nid]]
                expected = GateType.AND if node.gate == "AND" else GateType.OR
                assert cpt.gate is expected

    def test_lookup_by_node_id_variable_or_event(self, tree, failure_model):
        cpts = fault_tree_to_cpts(tree, failure_model)
        event_id, var = next(iter(cpts.event_variable.items()))
        assert cpts[event_id] is cpts[var]
        assert cpts.resolve(event_id) == var

    def test_an_unknown_name_lists_what_is_available(self, tree, failure_model):
        cpts = fault_tree_to_cpts(tree, failure_model)
        with pytest.raises(KeyError, match="Basic events here are"):
            cpts.resolve("no-such-event")

    def test_bounds_bracket_the_point_estimate(self, calibrated):
        report = calibrated.report
        tree, fmodel = report.tree("H2"), report.failure_model
        low = fault_tree_to_cpts(tree, fmodel, bound="lower")
        mid = fault_tree_to_cpts(tree, fmodel, bound="point")
        high = fault_tree_to_cpts(tree, fmodel, bound="upper")
        for var in low.roots:
            assert (
                low[var].prior_fail
                <= mid[var].prior_fail + 1e-12
                <= high[var].prior_fail + 1e-12
            )

    def test_soft_gates_are_recorded_as_a_departure(self, tree, failure_model):
        cpts = fault_tree_to_cpts(tree, failure_model, soft_gates=True)
        assert any("noisy-OR" in n or "noisy_or" in n for n in cpts.notes)
        assert any(c.kind == "noisy_or" for c in cpts.cpts.values())

    def test_gate_overrides_replace_the_logic(self, tree, failure_model):
        gate_node = next(
            nid
            for nid, node in tree.nodes.items()
            if node.gate == "OR" and len(node.children) >= 3
        )
        cpts = fault_tree_to_cpts(
            tree, failure_model, gate_overrides={gate_node: ("KOFN", 2)}
        )
        assert cpts[cpts.variable_of[gate_node]].gate is GateType.KOFN

    def test_missing_probabilities_are_reported_not_hidden(self, tree):
        """A basic event with no probability anywhere must be named in the notes."""
        import copy

        from hiphopsllm.faulttree.failure import FailureModel

        bare = copy.deepcopy(tree)
        bare.events = {}
        empty = FailureModel(system=None, logic={}, events={})
        cpts = CPTBuilder(default_prob=0.5).build(bare, empty)
        assert any("default prior" in n for n in cpts.notes)
        for var in cpts.roots:
            if cpts[var].kind == "prior" and cpts[var].evidence.startswith("no basic"):
                assert cpts[var].prior_fail == 0.5

    def test_hipmas_order_flips_the_states(self, tree, failure_model):
        cpts = fault_tree_to_cpts(tree, failure_model)
        flipped = cpts.to_hipmas_order()
        var = cpts.roots[0]
        assert flipped[var][0] == pytest.approx(cpts[var].prior_fail)


class TestNaming:
    def test_illegal_characters_are_replaced(self):
        assert sanitise("BE-a.b@c") == "BE_a_b_c"

    def test_a_leading_digit_is_prefixed(self):
        assert sanitise("2fast").startswith("n_")

    def test_names_stay_unique_after_sanitising(self, tree, failure_model):
        cpts = fault_tree_to_cpts(tree, failure_model)
        assert len(set(cpts.order)) == len(cpts.order)


def test_summary_and_frame_render(tree, failure_model):
    cpts = fault_tree_to_cpts(tree, failure_model)
    assert "CPT set" in cpts.summary()
    frame = cpts.to_frame()
    assert len(frame) == sum(c.n_rows for c in cpts.cpts.values())
