"""Future-reliability tests (specification items 21-25).

Paper Theorems 4-6 and Appendix A.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import TOL_CDF_SUP, TOL_EXPECTED_RELIABILITY
from hip_llm.envelopes import cdf_family, default_t_grid
from hip_llm.posterior import run_domain
from hip_llm.reliability import (
    default_horizons,
    expected_reliability_envelope,
    expected_reliability_per_config,
    reliability_cdf_envelope,
    reliability_from_p,
    transformed_cdf,
)
from hip_llm.validation import check_reliability_monotone


# --- 21, 22. R(1) = p and R(n) = p^n ---------------------------------------- #
def test_r_of_one_is_p():
    p = np.linspace(0.0, 1.0, 101)
    assert np.array_equal(reliability_from_p(p, 1), p)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 60])
def test_r_of_n_is_p_to_the_n(n):
    p = np.linspace(0.0, 1.0, 101)
    assert np.allclose(reliability_from_p(p, n), p**n)


# --- 23. non-increasing in n ------------------------------------------------ #
def test_reliability_is_non_increasing_in_the_horizon():
    p = np.linspace(0.0, 1.0, 501)
    horizons = default_horizons()
    values = np.array([reliability_from_p(p, n) for n in horizons])
    assert np.all(np.diff(values, axis=0) <= 1e-15)


def test_expected_reliability_envelope_is_monotone(toy_domain, fast_settings, wide_interval):
    ds = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    env = expected_reliability_envelope(ds.p)
    assert check_reliability_monotone(env.lower, env.upper).passed
    assert np.all(env.width >= -1e-15)


# --- 24. the transformed-CDF identity --------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_transformed_cdf_identity(toy_domain, fast_settings, wide_interval, n):
    """F_{p^n}(t) = F_p(t^{1/n}) -- Theorem 4."""
    ds = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    t = default_t_grid()

    F_p = cdf_family(ds.p, t)
    analytic = transformed_cdf(F_p, t, n)
    empirical = cdf_family(ds.p**n, t)

    sup = float(np.max(np.abs(analytic - empirical)))
    assert sup < TOL_CDF_SUP, f"n={n}: sup-norm {sup:.4f} exceeds {TOL_CDF_SUP}"


def test_transformed_cdf_identity_exactly_on_a_beta(n=4):
    """With an analytical Beta the identity must hold to interpolation error only."""
    from scipy import stats

    t = np.linspace(0.0, 1.0, 2001)
    F_p = stats.beta.cdf(t, 8.0, 3.0)
    got = transformed_cdf(F_p, t, n)
    want = stats.beta.cdf(t ** (1.0 / n), 8.0, 3.0)
    assert np.max(np.abs(got - want)) < 1e-6


# --- 25. Monte-Carlo vs transformed-CDF agreement --------------------------- #
def test_monte_carlo_and_transform_agree(toy_domain, fast_settings, wide_interval):
    ds = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    for n in (2, 5, 10):
        transform_env = reliability_cdf_envelope(ds.p, n)
        from hip_llm.envelopes import cdf_envelope

        mc_env = cdf_envelope(ds.p**n, quantity=f"MC R({n})")
        sup_lower = float(np.max(np.abs(transform_env.lower - mc_env.lower)))
        sup_upper = float(np.max(np.abs(transform_env.upper - mc_env.upper)))
        assert max(sup_lower, sup_upper) < TOL_CDF_SUP, (n, sup_lower, sup_upper)


# --- E[p^n] != E[p]^n ------------------------------------------------------- #
def test_expected_reliability_is_not_the_naive_power_of_the_mean(toy_domain, fast_settings,
                                                                 wide_interval):
    """Guard against the Jensen-inequality shortcut the specification forbids."""
    ds = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    horizons = np.array([1.0, 5.0, 20.0])
    correct = expected_reliability_per_config(ds.p, horizons)
    naive = ds.p.mean(axis=1)[:, None] ** horizons[None, :]

    assert np.allclose(correct[:, 0], naive[:, 0])          # equal at n = 1
    assert np.all(correct[:, 1:] >= naive[:, 1:] - 1e-12)   # Jensen: E[p^n] >= E[p]^n

    # At n = 5 the absolute error already exceeds nothing-to-see-here, and the
    # relative error grows without bound, so the shortcut is measurably wrong.
    assert np.max(correct[:, 1] / naive[:, 1] - 1.0) > 0.05
    assert np.max(correct[:, 2] / naive[:, 2] - 1.0) > 1.0

    # On a high-reliability domain the ABSOLUTE error also exceeds the declared
    # expected-reliability tolerance, which is what the specification forbids.
    high = np.clip(ds.p + 0.45, 0.0, 1.0)
    corr_h = expected_reliability_per_config(high, np.array([20.0]))
    naive_h = high.mean(axis=1)[:, None] ** np.array([20.0])[None, :]
    assert np.max(corr_h - naive_h) > TOL_EXPECTED_RELIABILITY


def test_expected_reliability_matches_the_beta_closed_form():
    """For a Beta posterior, E[theta^n] = B(a+n, b) / B(a, b) (paper Section 3.1)."""
    from scipy.special import betaln

    a, b, n = 40.0, 42.0, 5
    rng = np.random.default_rng(2)
    draws = rng.beta(a, b, size=(1, 400_000))
    mc = float(expected_reliability_per_config(draws, np.array([float(n)]))[0, 0])
    closed = float(np.exp(betaln(a + n, b) - betaln(a, b)))
    assert mc == pytest.approx(closed, abs=1e-3)


def test_envelope_bounds_come_from_per_configuration_expectations(toy_domain, fast_settings,
                                                                  wide_interval):
    ds = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    h = default_horizons()
    per_config = expected_reliability_per_config(ds.p, h)
    env = expected_reliability_envelope(ds.p, h)
    assert np.allclose(env.lower, per_config.min(axis=0))
    assert np.allclose(env.upper, per_config.max(axis=0))
