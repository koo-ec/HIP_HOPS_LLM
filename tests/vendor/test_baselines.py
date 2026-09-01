"""RQ5 baseline tests, plus the coin-flip imprecise-probability example (Table 2)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import stats

from hip_llm.baselines import (
    bb_informative,
    bb_uninformative,
    generate_synthetic_counts,
    hibayes_partial_pooling,
    rq5_blocking_checklist,
)
from hip_llm.schemas import load_yaml

OP = np.array([0.10, 0.20, 0.30, 0.40])
COUNTS = np.array([45.0, 260.0, 900.0, 255.0])
TRIALS = np.array([100.0, 500.0, 1000.0, 300.0])


# --- Table 2: the coin-flip example ----------------------------------------- #
def test_table2_precise_posterior_is_beta_5_9():
    """n = 10, k = 3, Beta(2,2) prior -> Beta(5, 9), mean 5/14 = 0.357."""
    n, k, a, b = 10, 3, 2, 2
    post_a, post_b = a + k, b + (n - k)
    assert (post_a, post_b) == (5, 9)
    assert post_a / (post_a + post_b) == pytest.approx(0.36, abs=0.005)


def test_table2_imprecise_posterior_mean_interval_is_analytically_0_29_to_0_43():
    """Discrepancy D9: the printed [0.31, 0.38] does not follow from the printed credal set.

    Posterior mean is (3+alpha)/(10+alpha+beta), monotone increasing in alpha and
    decreasing in beta, so its extrema over the rectangle alpha, beta in [1, 3]
    are at (1, 3) and (3, 1): [4/14, 6/14] = [0.2857, 0.4286] -> [0.29, 0.43].
    """
    n, k = 10, 3
    grid = np.linspace(1, 3, 401)
    means = [(k + a) / (n + a + b) for a in grid for b in grid]
    lo, hi = min(means), max(means)

    assert lo == pytest.approx(4 / 14, abs=1e-12)
    assert hi == pytest.approx(6 / 14, abs=1e-12)
    assert (round(lo, 2), round(hi, 2)) == (0.29, 0.43)

    # The paper prints [0.31, 0.38]; record the disagreement rather than matching it.
    assert (round(lo, 2), round(hi, 2)) != (0.31, 0.38)


def test_table2_printed_interval_corresponds_to_a_different_credal_set():
    """Evidence for D9: [0.31, 0.38] is exactly alpha + beta = 3 with alpha in [1, 2]."""
    assert round(4 / 13, 2) == 0.31
    assert round(5 / 13, 2) == 0.38
    alphas = np.linspace(1, 2, 401)
    means = (3 + alphas) / (10 + 3)          # beta = 3 - alpha, so alpha + beta = 3
    assert (round(float(means.min()), 2), round(float(means.max()), 2)) == (0.31, 0.38)


def test_table2_precise_column_is_correct():
    """The classical-Bayesian column of Table 2 is internally consistent."""
    assert stats.beta.mean(5, 9) == pytest.approx(5 / 14, abs=1e-12)
    assert round(float(stats.beta.mean(5, 9)), 2) == 0.36


def test_table2_precise_posterior_lies_inside_the_full_credal_envelope():
    """Beta(5,9) is an interior member, so its density must sit inside the envelope.

    The envelope must be taken over the whole rectangle, not only its corners:
    the density envelope of a Beta family is not attained at the corners alone.
    """
    x = np.linspace(1e-6, 1 - 1e-6, 4000)
    grid = np.linspace(1, 3, 41)
    stack = np.array([stats.beta.pdf(x, 3 + a, 7 + b) for a in grid for b in grid])
    lo, hi = stack.min(axis=0), stack.max(axis=0)
    assert np.all(lo <= hi)
    precise = stats.beta.pdf(x, 5, 9)        # alpha = beta = 2, an interior point
    assert np.all(precise >= lo - 1e-9) and np.all(precise <= hi + 1e-9)


# --- BB-UnInf --------------------------------------------------------------- #
def test_bb_uninformative_matches_the_analytical_beta_posterior():
    rng = np.random.default_rng(11)
    res = bb_uninformative(COUNTS, TRIALS, OP, 200_000, rng)
    expected_mean = (1 + COUNTS) / (2 + TRIALS)
    assert np.allclose(res.theta.mean(axis=0), expected_mean, atol=5e-3)
    assert res.median == pytest.approx(float(expected_mean @ OP), abs=5e-3)


def test_bb_uninformative_subdomains_are_independent():
    rng = np.random.default_rng(12)
    res = bb_uninformative(COUNTS, TRIALS, OP, 100_000, rng)
    corr = np.corrcoef(res.theta.T)
    off = corr[~np.eye(4, dtype=bool)]
    assert np.all(np.abs(off) < 0.02)


# --- BB-Inf ----------------------------------------------------------------- #
def test_bb_informative_prior_mean_is_the_ground_truth():
    theta_gt = np.array([0.45, 0.52, 0.90, 0.85])
    rng = np.random.default_rng(13)
    # With zero data the posterior mean must equal the prior mean exactly.
    res = bb_informative(np.zeros(4), np.zeros(4), OP, theta_gt, 60.0, 200_000, rng)
    assert np.allclose(res.theta.mean(axis=0), theta_gt, atol=5e-3)


def test_bb_informative_prior_strength_is_required():
    """The paper never states the strength, so the code must not invent one."""
    sig = inspect.signature(bb_informative)
    assert sig.parameters["prior_strength"].default is inspect.Parameter.empty


def test_stronger_informative_prior_pulls_harder_toward_the_ground_truth():
    theta_gt = np.array([0.60, 0.60, 0.60, 0.60])
    rng = np.random.default_rng(14)
    weak = bb_informative(COUNTS, TRIALS, OP, theta_gt, 1.0, 40_000, rng).theta.mean(axis=0)
    strong = bb_informative(COUNTS, TRIALS, OP, theta_gt, 5000.0, 40_000, rng).theta.mean(axis=0)
    assert np.all(np.abs(strong - 0.60) < np.abs(weak - 0.60))


# --- HiBayES ---------------------------------------------------------------- #
def test_hibayes_priors_are_required():
    sig = inspect.signature(hibayes_partial_pooling)
    for name in ("alpha_prior_sd", "sigma_prior_scale"):
        assert sig.parameters[name].default is inspect.Parameter.empty
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_hibayes_recovers_a_common_rate_under_strong_pooling():
    """With sigma forced small, all subdomains shrink to the pooled rate."""
    counts = np.array([60.0, 300.0, 600.0, 180.0])
    trials = np.array([100.0, 500.0, 1000.0, 300.0])
    pooled = counts.sum() / trials.sum()
    rng = np.random.default_rng(15)
    res = hibayes_partial_pooling(
        counts, trials, OP, 20_000, rng, alpha_prior_sd=3.0, sigma_prior_scale=0.02
    )
    assert np.allclose(res.theta.mean(axis=0), pooled, atol=0.03)


def test_hibayes_partial_pooling_sits_between_no_pooling_and_full_pooling():
    counts = np.array([20.0, 300.0, 900.0, 90.0])
    trials = np.array([100.0, 500.0, 1000.0, 300.0])
    empirical = counts / trials
    pooled = counts.sum() / trials.sum()
    rng = np.random.default_rng(16)
    res = hibayes_partial_pooling(
        counts, trials, OP, 20_000, rng, alpha_prior_sd=3.0, sigma_prior_scale=1.0
    )
    means = res.theta.mean(axis=0)
    for j in range(4):
        lo, hi = sorted((empirical[j], pooled))
        assert lo - 0.05 <= means[j] <= hi + 0.05, (j, means[j], empirical[j], pooled)
    # And it is strictly shrunk relative to the unpooled estimate.
    assert np.sum(np.abs(means - pooled)) < np.sum(np.abs(empirical - pooled))


def test_hibayes_is_deterministic_given_the_seed():
    a = hibayes_partial_pooling(COUNTS, TRIALS, OP, 5_000, np.random.default_rng(3),
                                alpha_prior_sd=2.0, sigma_prior_scale=1.0)
    b = hibayes_partial_pooling(COUNTS, TRIALS, OP, 5_000, np.random.default_rng(3),
                                alpha_prior_sd=2.0, sigma_prior_scale=1.0)
    assert np.array_equal(a.p_L, b.p_L)


# --- credible intervals and errors ------------------------------------------ #
def test_credible_interval_and_error_definitions():
    rng = np.random.default_rng(17)
    res = bb_uninformative(COUNTS, TRIALS, OP, 50_000, rng)
    lo, hi = res.credible_interval(0.90)
    assert lo < res.median < hi
    assert lo == pytest.approx(float(np.quantile(res.p_L, 0.05)))
    assert hi == pytest.approx(float(np.quantile(res.p_L, 0.95)))
    assert res.error(0.5860) == pytest.approx(abs(res.median - 0.5860))


# --- synthetic data generation ---------------------------------------------- #
def test_synthetic_counts_are_binomial_and_reproducible():
    theta_gt = np.array([0.45, 0.52, 0.90, 0.85])
    sizes = [100, 500, 1000, 300]
    a, _ = generate_synthetic_counts(theta_gt, sizes, np.random.default_rng(99))
    b, N = generate_synthetic_counts(theta_gt, sizes, np.random.default_rng(99))
    assert np.array_equal(a, b)
    assert np.all(a <= N) and np.all(a >= 0)
    # Frequency check across many replicates.
    reps = np.array(
        [generate_synthetic_counts(theta_gt, sizes, np.random.default_rng(s))[0] for s in range(400)]
    )
    assert np.allclose((reps / np.array(sizes)).mean(axis=0), theta_gt, atol=0.01)


# --- RQ5 is blocked --------------------------------------------------------- #
def test_rq5_config_ships_blocked(root):
    cfg = load_yaml(root / "configs" / "synthetic_rq5.yaml")
    assert cfg["status"] == "blocked_by_missing_source_information"
    assert cfg["allow_illustrative_run"] is False
    assert cfg["p_L_ground_truth"] == 0.5860


def test_rq5_blocking_checklist_flags_every_missing_input(root):
    cfg = load_yaml(root / "configs" / "synthetic_rq5.yaml")
    checklist = rq5_blocking_checklist(cfg)
    assert len(checklist) >= 7
    assert all(item["resolved"] is False for item in checklist), [
        i["quantity"] for i in checklist if i["resolved"]
    ]
    quantities = " ".join(i["quantity"] for i in checklist).lower()
    for expected in ("subdomain reliability", "operational profile", "prior strength",
                     "hibayes", "perturbation", "seed"):
        assert expected in quantities


def test_rq5_checklist_clears_once_values_are_supplied(root):
    cfg = load_yaml(root / "configs" / "synthetic_rq5.yaml")
    ill = cfg["illustrative"]
    cfg["ground_truth"] = {"theta_gt": ill["theta_gt"], "op_gt": ill["op_gt"]}
    cfg["operational_profiles"]["op_data"] = ill["op_data"]
    cfg["operational_profiles"]["perturbation_magnitude"] = ill["perturbation_magnitude"]
    cfg["baselines"]["bb_inf"]["prior_strength"] = ill["bb_inf_prior_strength"]
    cfg["baselines"]["hibayes"]["alpha_prior_sd"] = ill["hibayes_alpha_prior_sd"]
    cfg["baselines"]["hibayes"]["sigma_prior_scale"] = ill["hibayes_sigma_prior_scale"]
    cfg["seeds"]["synthetic"] = ill["seed"]
    checklist = rq5_blocking_checklist(cfg)
    assert all(item["resolved"] for item in checklist)


def test_ground_truth_aggregate_is_underdetermined():
    """One equation, seven free unknowns: the reason RQ5 cannot be reconstructed."""
    rng = np.random.default_rng(5)
    solutions = 0
    for _ in range(2000):
        op = rng.dirichlet(np.ones(4))
        theta = rng.uniform(0.2, 0.99, size=3)
        remaining = 0.5860 - float(op[:3] @ theta)
        last = remaining / op[3]
        if 0.0 < last < 1.0:
            solutions += 1
    assert solutions > 100, "the aggregate admits many distinct parameter sets"
