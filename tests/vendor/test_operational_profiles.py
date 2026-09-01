"""Operational-profile tests (specification items 16-20)."""

from __future__ import annotations

import numpy as np
import pytest

from hip_llm.operational_profile import (
    aggregate,
    dataset_proportional_op,
    flatten_hierarchical_op,
    perturb_and_renormalise,
    unflatten_op,
    validate_weights,
)
from hip_llm.posterior import run_model
from hip_llm.schemas import DomainData, OperationalProfile, SubdomainData

PAPER_OMEGA_1 = np.array([0.204, 0.796])
PAPER_OMEGA_2 = np.array([0.483, 0.517])
PAPER_W = np.array([0.149, 0.851])


# --- 16. every OP vector sums to one ---------------------------------------- #
@pytest.mark.parametrize("w", [PAPER_OMEGA_1, PAPER_OMEGA_2, PAPER_W])
def test_paper_weights_sum_to_one(w):
    assert validate_weights(w).sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "bad,match",
    [
        (np.array([0.5, 0.4]), "sum to"),
        (np.array([1.2, -0.2]), "negative"),
        (np.array([0.5, np.nan]), "non-finite"),
        (np.array([0.5, np.inf]), "non-finite"),
    ],
)
def test_invalid_weight_vectors_are_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        validate_weights(bad)


def test_operational_profile_schema_rejects_label_mismatch():
    with pytest.raises(ValueError, match="labels"):
        OperationalProfile(level="domain", labels=("a",), weights=np.array([0.5, 0.5]))


def test_domain_schema_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="omega weights sum"):
        DomainData(
            "D", (SubdomainData("A", 1, 2), SubdomainData("B", 1, 2)), np.array([0.5, 0.4])
        )


# --- 17, 18. aggregation stays in [0,1] and matches manual sums ------------- #
def test_aggregation_matches_a_manual_weighted_sum():
    theta = np.array([[0.4, 0.9], [0.5, 0.8]])
    got = aggregate(theta, PAPER_OMEGA_2, axis=-1)
    want = np.array(
        [0.483 * 0.4 + 0.517 * 0.9, 0.483 * 0.5 + 0.517 * 0.8]
    )
    assert np.allclose(got, want)
    assert np.all((got >= 0.0) & (got <= 1.0))


def test_llm_aggregation_matches_a_manual_weighted_sum(toy_model, fast_settings, wide_interval):
    domain_sets, llm = run_model(toy_model, [wide_interval, wide_interval], fast_settings)
    manual = np.zeros_like(llm.p_L)
    for i, ds in enumerate(domain_sets):
        manual += toy_model.W[i] * ds.p[llm.pair_index[:, i], :]
    assert np.allclose(llm.p_L, manual)
    assert np.all((llm.p_L >= 0.0) & (llm.p_L <= 1.0))


def test_flatten_and_unflatten_round_trip():
    """Paper footnote 21: OP_ij = W_i * Omega_ij, W_i = sum_j OP_ij."""
    flat = flatten_hierarchical_op(PAPER_W, [PAPER_OMEGA_1, PAPER_OMEGA_2])
    assert flat.sum() == pytest.approx(1.0, abs=1e-12)
    W, omegas = unflatten_op(flat, [2, 2])
    assert np.allclose(W, PAPER_W)
    assert np.allclose(omegas[0], PAPER_OMEGA_1)
    assert np.allclose(omegas[1], PAPER_OMEGA_2)


def test_flat_and_hierarchical_aggregation_are_identical():
    theta = np.array([0.479, 0.490, 0.910, 0.920])
    hierarchical = PAPER_W[0] * (PAPER_OMEGA_1 @ theta[:2]) + PAPER_W[1] * (PAPER_OMEGA_2 @ theta[2:])
    flat = flatten_hierarchical_op(PAPER_W, [PAPER_OMEGA_1, PAPER_OMEGA_2]) @ theta
    assert hierarchical == pytest.approx(flat, abs=1e-15)


# --- 19. equal subdomain probabilities make weights irrelevant -------------- #
def test_equal_subdomain_probabilities_make_the_op_irrelevant():
    """Section 14 Case 4."""
    theta = np.full((500, 4), 0.73)
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.dirichlet(np.ones(4))
        assert np.allclose(aggregate(theta, w), 0.73)


# --- 20. moving weight to a weaker subdomain cannot raise the aggregate ----- #
def test_shifting_weight_to_the_weaker_subdomain_cannot_increase_the_aggregate():
    theta = np.array([0.45, 0.92])                       # index 0 is weaker
    base = np.array([0.483, 0.517])
    previous = float(aggregate(theta, base))
    for extra in (0.1, 0.2, 0.3, 0.4):
        w = np.array([base[0] + extra, base[1] - extra])
        w = w / w.sum()
        current = float(aggregate(theta, w))
        assert current <= previous + 1e-12
        previous = current


def test_rq3_omega_sweep_shifts_the_domain_posterior_in_the_expected_direction(
    fast_settings, wide_interval
):
    """RQ3: with theta_BoolQ > theta_RACE-H, raising Omega_RACE must lower p_2."""
    from hip_llm.envelopes import quantile_envelope
    from hip_llm.posterior import run_domain

    medians = []
    for omega_race in (0.10, 0.517, 0.90):
        d = DomainData(
            "D2",
            (SubdomainData("BoolQ", 73, 80, 0.909), SubdomainData("RACE-H", 44, 80, 0.552)),
            np.array([1.0 - omega_race, omega_race]),
        )
        ds = run_domain(d, wide_interval, fast_settings, domain_index=1)
        medians.append(np.mean(quantile_envelope(ds.p, 0.5)))
    assert medians[0] > medians[1] > medians[2], medians


# --- perturbation ----------------------------------------------------------- #
def test_perturbation_stays_a_probability_vector_and_respects_its_magnitude():
    rng = np.random.default_rng(7)
    base = np.array([0.10, 0.20, 0.30, 0.40])
    for magnitude in (0.0, 0.07, 0.20):
        out = perturb_and_renormalise(base, magnitude, rng)
        assert out.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(out >= 0.0)
        if magnitude == 0.0:
            assert np.allclose(out, base)


def test_perturbation_magnitude_has_no_default():
    """The 0.20 (paper) vs 0.07 (repository) conflict must never be resolved silently."""
    import inspect

    sig = inspect.signature(perturb_and_renormalise)
    assert sig.parameters["magnitude"].default is inspect.Parameter.empty


def test_perturbation_rejects_an_out_of_range_magnitude():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        perturb_and_renormalise(np.array([0.5, 0.5]), 1.5, rng)


def test_dataset_proportional_op_matches_the_paper_style_construction():
    op = dataset_proportional_op(np.array([204.0, 796.0]), "subdomain", ("MBPP", "DS-1000"))
    assert np.allclose(op.weights, [0.204, 0.796])
    assert op.weights.sum() == pytest.approx(1.0, abs=1e-15)
