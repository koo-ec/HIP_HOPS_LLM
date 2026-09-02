"""Bayesian networks: exact inference, cross-checks, evidence, envelopes."""

from __future__ import annotations

import pytest

from hiphopsllm import BayesianNetwork, Envelope, fault_tree_to_bayesnet

pyagrum = pytest.importorskip("pyagrum", reason="pyAgrum is optional")


@pytest.fixture(scope="module")
def net(request):
    study = request.getfixturevalue("study")
    return BayesianNetwork.from_fault_tree(
        study.report.tree("H2"), study.report.failure_model, name="H2"
    )


class TestConstruction:
    def test_from_fault_tree_covers_every_node(self, net, tree):
        assert len(net.cpts) == len(tree.nodes)

    def test_arcs_run_from_causes_to_effects(self, net):
        agrum = net.net
        for var in net.cpts.order:
            for parent in net.cpts[var].parents:
                assert agrum.existsArc(parent, var)

    def test_state_labels_are_ok_then_fail(self, net):
        variable = net.net.variable(net.top)
        assert variable.label(0) == "OK"
        assert variable.label(1) == "Fail"

    def test_basic_events_are_addressable_by_name(self, net):
        assert net.basic_events
        for event_id in net.basic_events:
            assert net.resolve(event_id) in net.cpts.order


class TestInference:
    def test_exact_and_pyagrum_agree(self, net):
        check = net.cross_check()
        assert check["compared"] == 1.0, "pyAgrum is installed, so it must be used"
        assert check["agree"] == 1.0, (
            "the two construction paths disagree; one of them is wrong"
        )
        assert check["relative_difference"] < 1e-9

    def test_top_probability_is_a_probability(self, net):
        assert 0.0 <= net.p_fail() <= 1.0

    def test_the_cut_set_bound_never_under_estimates(self, study):
        for hid in study.report.trees:
            bn = BayesianNetwork.from_fault_tree(
                study.report.tree(hid), study.report.failure_model
            )
            comparison = bn.compare_with_cutsets(study.report.analysis(hid))
            assert comparison["bound_overestimate"] >= -1e-9, (
                f"{hid}: the minimal cut upper bound came out below the exact "
                "value, so the tree and the network have drifted apart"
            )

    def test_shared_events_make_the_bound_strictly_loose(self, study):
        """Where cut sets share events, MCUB must over-estimate — that is the
        whole reason for building the network."""
        bn = BayesianNetwork.from_fault_tree(
            study.report.tree("H2"), study.report.failure_model
        )
        gap = bn.compare_with_cutsets(study.report.analysis("H2"))
        assert gap["bound_overestimate"] > 0.0

    def test_a_single_point_of_failure_forces_the_top_event(self, net, study):
        """Conditioning an order-1 cut set to Fail must make P(top) exactly 1."""
        singles = [
            next(iter(s)) for s in study.report.analysis("H2").cuts.sets if len(s) == 1
        ]
        assert singles, "H2 has no single point of failure to test with"
        for event in singles:
            assert net.p_fail(evidence={event: "Fail"}) == pytest.approx(1.0)

    def test_a_higher_order_cut_set_needs_all_of_its_events(self, net, study):
        pairs = [s for s in study.report.analysis("H2").cuts.sets if len(s) == 2]
        assert pairs
        a, b = sorted(pairs[0])
        both = net.p_fail(evidence={a: "Fail", b: "Fail"})
        one = net.p_fail(evidence={a: "Fail", b: "OK"})
        assert both == pytest.approx(1.0)
        assert one < 1.0

    def test_evidence_accepts_several_spellings(self, net):
        event = net.basic_events[0]
        a = net.p_fail(evidence={event: "Fail"})
        b = net.p_fail(evidence={event: 1})
        c = net.p_fail(evidence={event: True})
        assert a == pytest.approx(b) == pytest.approx(c)

    def test_unknown_evidence_names_are_rejected_helpfully(self, net):
        with pytest.raises(KeyError, match="Basic events here are"):
            net.p_fail(evidence={"not-an-event": "Fail"})

    def test_monotone_in_every_basic_event(self, study):
        """A coherent fault tree's top probability never falls when a leaf's
        probability rises. This is what licenses the interval arithmetic."""
        tree, fmodel = study.report.tree("H2"), study.report.failure_model
        base = BayesianNetwork.from_fault_tree(tree, fmodel).p_fail()
        for event_id in list(fmodel.events)[:6]:
            event = fmodel.events[event_id]
            original = event.prob
            event.prob = min(1.0, original + 0.05)
            raised = BayesianNetwork.from_fault_tree(tree, fmodel).p_fail()
            event.prob = original
            assert raised >= base - 1e-12, f"{event_id} broke monotonicity"


class TestDiagnosis:
    def test_posteriors_are_ranked(self, net):
        event = net.basic_events[0]
        posterior = net.posteriors({event: "Fail"})
        values = list(posterior.values())
        assert values == sorted(values, reverse=True)

    def test_the_observed_event_has_posterior_one(self, net):
        event = net.basic_events[0]
        assert net.posteriors({event: "Fail"})[event] == pytest.approx(1.0)

    def test_evidence_raises_the_posterior_of_related_causes(self, net):
        prior = net.posteriors()
        posterior = net.posteriors({net.top: "Fail"})
        assert any(posterior[k] > prior[k] + 1e-9 for k in prior)

    def test_most_probable_explanation_assigns_every_basic_event(self, net):
        mpe = net.most_probable_explanation()
        assert set(mpe) == set(net.basic_events)
        assert set(mpe.values()) <= {"OK", "Fail"}

    def test_the_legacy_evidence_posterior_name_still_works(self, net):
        event = net.basic_events[0]
        assert net.evidence_posterior({event: "Fail"}) == net.posteriors(
            {event: "Fail"}
        )


class TestImprecise:
    def test_envelope_brackets_the_point_estimate(self, calibrated):
        point = calibrated.bayesnet("H2").p_fail()
        envelope = calibrated.imprecise_bayesnet("H2").envelope()
        assert envelope.contains(point), f"{point} outside {envelope}"

    def test_the_envelope_has_positive_width_after_calibration(self, calibrated):
        assert calibrated.imprecise_bayesnet("H2").envelope().width > 0.0

    def test_an_uncalibrated_tree_has_a_degenerate_envelope(self, study):
        envelope = study.imprecise_bayesnet("H2").envelope()
        assert envelope.width == pytest.approx(0.0, abs=1e-12)

    def test_envelope_rejects_an_inverted_interval(self):
        with pytest.raises(ValueError, match="exceeds"):
            Envelope(0.6, 0.4)

    def test_posterior_envelopes_cover_every_basic_event(self, calibrated):
        envelopes = calibrated.imprecise_bayesnet("H2").posterior_envelopes()
        assert envelopes
        assert all(e.lower <= e.upper for e in envelopes.values())


class TestWithoutPyagrum:
    def test_exact_engine_needs_only_numpy(self, net):
        assert 0.0 <= net.p_fail(engine="exact") <= 1.0

    def test_exact_matches_a_hand_computed_or(self):
        """Two independent events under an OR: P = 1 - (1-a)(1-b)."""
        from hiphopsllm.bayes.cpt import CPT, CPTSet, deterministic_gate_cpt, prior_cpt

        cs = CPTSet(name="or2")
        cs.add(CPT(variable="a", parents=(), table=prior_cpt(0.1)))
        cs.add(CPT(variable="b", parents=(), table=prior_cpt(0.2)))
        cs.add(
            CPT(
                variable="top",
                parents=("a", "b"),
                table=deterministic_gate_cpt(2, "OR"),
            )
        )
        cs.top = "top"
        bn = BayesianNetwork(cpts=cs, name="or2")
        assert bn.p_fail(engine="exact") == pytest.approx(1 - 0.9 * 0.8)
        assert bn.p_fail(engine="pyagrum") == pytest.approx(1 - 0.9 * 0.8)

    def test_and_of_two_events(self):
        from hiphopsllm.bayes.cpt import CPT, CPTSet, deterministic_gate_cpt, prior_cpt

        cs = CPTSet(name="and2")
        cs.add(CPT(variable="a", parents=(), table=prior_cpt(0.1)))
        cs.add(CPT(variable="b", parents=(), table=prior_cpt(0.2)))
        cs.add(
            CPT(
                variable="top",
                parents=("a", "b"),
                table=deterministic_gate_cpt(2, "AND"),
            )
        )
        cs.top = "top"
        assert BayesianNetwork(cpts=cs).p_fail(engine="exact") == pytest.approx(0.02)

    def test_a_shared_cause_is_not_double_counted(self):
        """Both branches depend on one event; the exact answer is 0.1, not 0.19."""
        from hiphopsllm.bayes.cpt import CPT, CPTSet, deterministic_gate_cpt, prior_cpt

        cs = CPTSet(name="shared")
        cs.add(CPT(variable="c", parents=(), table=prior_cpt(0.1)))
        cs.add(CPT(variable="l", parents=("c",), table=deterministic_gate_cpt(1, "OR")))
        cs.add(CPT(variable="r", parents=("c",), table=deterministic_gate_cpt(1, "OR")))
        cs.add(
            CPT(
                variable="top",
                parents=("l", "r"),
                table=deterministic_gate_cpt(2, "AND"),
            )
        )
        cs.top = "top"
        assert BayesianNetwork(cpts=cs).p_fail(engine="exact") == pytest.approx(0.1)


def test_functional_alias_matches_the_classmethod(tree, failure_model):
    a = fault_tree_to_bayesnet(tree, failure_model).p_fail()
    b = BayesianNetwork.from_fault_tree(tree, failure_model).p_fail()
    assert a == pytest.approx(b)
