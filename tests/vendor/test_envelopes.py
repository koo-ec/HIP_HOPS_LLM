"""CDF and envelope tests (specification items 5-10)."""

from __future__ import annotations

import numpy as np
import pytest

from hip_llm.envelopes import (
    cdf_envelope,
    cdf_family,
    default_t_grid,
    quantile_envelope,
    quantiles_from_cdf_envelope,
    summarise_envelope,
)
from hip_llm.posterior import run_domain, run_model
from hip_llm.schemas import CDFEnvelope


@pytest.fixture
def domain_samples(toy_domain, fast_settings, wide_interval):
    return run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)


# --- 5, 6, 7. every CDF is a valid CDF -------------------------------------- #
def test_every_cdf_is_non_decreasing_in_unit_range_with_correct_endpoints(domain_samples):
    F = cdf_family(domain_samples.p, default_t_grid())
    assert np.all(np.diff(F, axis=1) >= -1e-12)
    assert np.all(F >= 0.0) and np.all(F <= 1.0)
    assert np.allclose(F[:, 0], 0.0)
    assert np.allclose(F[:, -1], 1.0)


def test_cdf_matches_a_brute_force_count(domain_samples):
    t = default_t_grid(21)
    F = cdf_family(domain_samples.p[:3], t)
    for k in range(3):
        brute = np.array([(domain_samples.p[k] <= ti).mean() for ti in t])
        assert np.allclose(F[k], brute)


# --- 8. lower <= upper ------------------------------------------------------ #
def test_lower_envelope_never_exceeds_upper(domain_samples):
    env = cdf_envelope(domain_samples.p, quantity="p_1")
    assert np.all(env.lower <= env.upper + 1e-12)
    assert env.area >= 0.0
    assert env.max_separation >= 0.0


def test_cdf_envelope_schema_rejects_a_crossed_envelope():
    t = np.linspace(0, 1, 11)
    lower = np.linspace(0.2, 1.0, 11)
    upper = np.linspace(0.1, 0.9, 11)   # valid CDFs, but strictly below `lower`
    with pytest.raises(ValueError, match="lower CDF envelope exceeds"):
        CDFEnvelope(quantity="bad", t_grid=t, lower=lower, upper=upper)


def test_cdf_envelope_schema_rejects_a_non_monotone_cdf():
    t = np.linspace(0, 1, 5)
    bad = np.array([0.0, 0.5, 0.2, 0.8, 1.0])
    with pytest.raises(ValueError, match="non-decreasing"):
        CDFEnvelope(quantity="bad", t_grid=t, lower=bad, upper=np.linspace(0, 1, 5))


# --- 9. a singleton hyperparameter set collapses the envelope --------------- #
def test_precise_hyperparameters_collapse_the_envelope(toy_domain, fast_settings, precise_interval):
    """Section 14 Case 5: a degenerate credal set yields ONE posterior."""
    ds = run_domain(toy_domain, precise_interval, fast_settings, domain_index=0)
    env = cdf_envelope(ds.p, quantity="p_1")
    # All configurations are identical, so every configuration shares one RNG-free
    # hyperposterior; residual spread is pure Monte-Carlo noise across streams.
    assert env.max_separation < 0.05
    lo, hi = quantile_envelope(ds.p, 0.5)
    assert hi - lo < 0.01

    # With a single configuration the collapse must be exact.
    single = run_domain(
        toy_domain, precise_interval, fast_settings, domain_index=0,
        configs=[ds.configs[0]],
    )
    env1 = cdf_envelope(single.p, quantity="p_1")
    assert np.array_equal(env1.lower, env1.upper)
    assert env1.max_separation == 0.0
    assert env1.area == 0.0


def test_wider_intervals_give_wider_envelopes(toy_domain, fast_settings, wide_interval,
                                              precise_interval):
    wide = cdf_envelope(run_domain(toy_domain, wide_interval, fast_settings, 0).p, quantity="w")
    tight = cdf_envelope(run_domain(toy_domain, precise_interval, fast_settings, 0).p, quantity="t")
    assert wide.area > tight.area


# --- 10. quantiles are ordered and each median sits inside its own interval -- #
def test_quantiles_are_ordered_and_medians_lie_inside_their_own_intervals(domain_samples):
    s = summarise_envelope(domain_samples.p, "p_1")
    assert s.q05_lower <= s.median_lower <= s.q95_upper
    assert s.q05_lower <= s.median_upper <= s.q95_upper
    assert s.median_lower <= s.median_upper
    assert s.q05_lower <= s.q05_upper
    assert s.q95_lower <= s.q95_upper

    # Per configuration, the strict invariant that paper Table 5 violates.
    for k in range(domain_samples.n_configs):
        q05, q50, q95 = np.quantile(domain_samples.p[k], [0.05, 0.50, 0.95])
        assert q05 <= q50 <= q95


def test_envelope_of_medians_lies_inside_the_widest_credible_interval(domain_samples):
    """min_h Q05 <= min_h Q50 and max_h Q50 <= max_h Q95, by within-posterior monotonicity."""
    m_lo, m_hi = quantile_envelope(domain_samples.p, 0.50)
    c_lo, _ = quantile_envelope(domain_samples.p, 0.05)
    _, c_hi = quantile_envelope(domain_samples.p, 0.95)
    assert c_lo <= m_lo and m_hi <= c_hi


def test_the_two_quantile_definitions_are_reported_separately(domain_samples):
    """Q1 (envelope of quantiles) and Q2 (quantiles of the envelope) must not be mixed."""
    env = cdf_envelope(domain_samples.p, quantity="p_1")
    q1 = quantile_envelope(domain_samples.p, 0.5)
    q2 = quantiles_from_cdf_envelope(env, 0.5)
    assert q2[0] <= q2[1]
    # Q2 is at least as wide as Q1 up to the grid resolution of 1/200.
    assert q2[0] <= q1[0] + 0.006 and q2[1] >= q1[1] - 0.006


# --- LLM-level envelope ----------------------------------------------------- #
def test_llm_envelope_is_valid_and_pairs_are_joint(toy_model, fast_settings, wide_interval):
    domain_sets, llm = run_model(toy_model, [wide_interval, wide_interval], fast_settings)
    env = cdf_envelope(llm.p_L, quantity="p_L")
    assert np.all(env.lower <= env.upper + 1e-12)
    assert llm.pair_index.shape[1] == 2
    # Distinct joint tuples, i.e. domains were not paired index-by-index.
    assert len({tuple(r) for r in llm.pair_index}) == llm.pair_index.shape[0]
    recomputed = (
        toy_model.W[0] * domain_sets[0].p[llm.pair_index[:, 0], :]
        + toy_model.W[1] * domain_sets[1].p[llm.pair_index[:, 1], :]
    )
    assert np.allclose(llm.p_L, recomputed)


def test_exact_cartesian_pairing_enumerates_the_full_product(toy_model, fast_settings,
                                                             wide_interval):
    _, llm = run_model(
        toy_model, [wide_interval, wide_interval], fast_settings, pairing_mode="exact_cartesian"
    )
    K = fast_settings.K_per_domain
    assert llm.n_configs == K * K
    assert len({tuple(r) for r in llm.pair_index}) == K * K


@pytest.mark.parametrize("m", [2, 8, 9, 12])
def test_pairing_survives_a_product_too_large_to_index(m):
    """K^m overflows int64 for m >= 9 at K = 160; tuples must still be drawn jointly."""
    from hip_llm.grids import pair_llm_configurations

    pairs = pair_llm_configurations(K_per_domain=160, n_domains=m, max_pairs=512, seed=999)
    assert pairs.shape == (512, m)
    assert pairs.min() >= 0 and pairs.max() < 160
    assert len({tuple(r) for r in pairs}) == 512          # sampled without replacement
    again = pair_llm_configurations(K_per_domain=160, n_domains=m, max_pairs=512, seed=999)
    assert np.array_equal(pairs, again)                    # reproducible
    if m >= 2:
        # Domains are not paired by a shared ordering: the per-domain index
        # columns must be mutually uncorrelated.
        corr = np.corrcoef(pairs.T.astype(float))
        assert np.all(np.abs(corr[~np.eye(m, dtype=bool)]) < 0.25)


def test_capped_pairing_is_a_strict_subsample(toy_model, fast_settings, wide_interval):
    _, capped = run_model(toy_model, [wide_interval, wide_interval], fast_settings)
    _, full = run_model(
        toy_model, [wide_interval, wide_interval], fast_settings, pairing_mode="exact_cartesian"
    )
    assert capped.n_configs == fast_settings.max_llm_configuration_pairs
    assert capped.n_configs < full.n_configs
    # A subsample can only narrow (never widen) the envelope.
    e_cap = cdf_envelope(capped.p_L, quantity="capped")
    e_full = cdf_envelope(full.p_L, quantity="full")
    assert e_cap.area <= e_full.area + 1e-9
