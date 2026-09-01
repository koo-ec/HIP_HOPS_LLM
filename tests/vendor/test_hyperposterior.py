"""Grid construction, quadrature weights and hyperposterior behaviour.

Covers specification items 28-29 (no NaN/inf at extreme admissible
hyperparameters; log-space vs high-precision agreement) and Section 13.4-13.5
(nu-grid alternatives, Jacobian-correct quadrature weights).
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest
from scipy import integrate

from hip_llm.grids import (
    CONFIG_SAMPLING_SCHEMES,
    NU_GRID_SCHEMES,
    StrictModeError,
    build_grid,
    build_nu_axis,
    sample_configurations,
)
from hip_llm.hyperposterior import HyperposteriorCache, log_hyperposterior
from hip_llm.numerics import log_beta_binomial_evidence, log_beta_pdf, log_gamma_pdf
from hip_llm.schemas import HyperparameterConfiguration


# --- 28. extreme admissible hyperparameters stay finite --------------------- #
@pytest.mark.parametrize("nu_scheme", sorted(NU_GRID_SCHEMES))
def test_no_nan_at_extreme_admissible_corners(toy_domain, nu_scheme):
    """All 16 corners of the paper's admissible box must produce finite output."""
    corners = [(a, b, c, d) for a in (1, 12) for b in (1, 12) for c in (1, 25) for d in (1, 25)]
    for corner in corners:
        cfg = HyperparameterConfiguration(*map(float, corner))
        grid = build_grid(40, 50, nu_scheme=nu_scheme, config=cfg)
        probs, log_norm = log_hyperposterior(toy_domain, cfg, grid)
        assert np.all(np.isfinite(probs)), f"non-finite probs at {corner} ({nu_scheme})"
        assert np.isfinite(log_norm), f"non-finite evidence at {corner} ({nu_scheme})"
        assert probs.sum() == pytest.approx(1.0, abs=1e-12)


def test_extreme_counts_stay_finite(nu_scheme="log"):
    """C = 0 and C = N are legal and must not produce infinities."""
    from hip_llm.schemas import DomainData, SubdomainData

    for C in (0, 80):
        d = DomainData(
            "D", (SubdomainData("A", C, 80), SubdomainData("B", 40, 80)), np.array([0.5, 0.5])
        )
        cfg = HyperparameterConfiguration(1.0, 1.0, 1.0, 25.0)
        grid = build_grid(40, 50, nu_scheme=nu_scheme, config=cfg)
        probs, log_norm = log_hyperposterior(d, cfg, grid)
        assert np.all(np.isfinite(probs)) and np.isfinite(log_norm)


# --- 29. log-space vs high-precision decimal reference ---------------------- #
def test_log_space_agrees_with_high_precision_reference():
    """Compare the log Beta-Binomial evidence against a 60-digit Decimal computation."""
    getcontext().prec = 60
    C, N, mu, nu = 3.0, 10.0, 0.4, 6.0
    a, b = mu * nu, (1 - mu) * nu

    def dec_beta(x: Decimal, y: Decimal) -> Decimal:
        # B(x,y) = Gamma(x)Gamma(y)/Gamma(x+y); use lgamma via float then refine.
        from math import lgamma

        return Decimal(lgamma(float(x))) + Decimal(lgamma(float(y))) - Decimal(lgamma(float(x + y)))

    ref = dec_beta(Decimal(C) + Decimal(a), Decimal(N - C) + Decimal(b)) - dec_beta(
        Decimal(a), Decimal(b)
    )
    got = log_beta_binomial_evidence(np.array([C]), np.array([N]), np.array([mu]), np.array([nu]))[0]
    assert float(ref) == pytest.approx(float(got), abs=1e-10)


# --- Section 13.5: quadrature weights and the Jacobian ---------------------- #
@pytest.mark.parametrize("scheme", ["log", "linear"])
def test_grid_weights_integrate_the_gamma_prior_to_one(scheme):
    """sum_k Gamma(nu_k) * Delta nu_k ~ 1: the cell widths carry the Jacobian."""
    for c, d in [(1.0, 1.0), (25.0, 1.0), (3.0, 0.5), (25.0, 25.0)]:
        nodes, log_w, absorbs = build_nu_axis(2000, scheme, c=c, d=d, nu_lo=1e-5, nu_hi=400.0)
        assert not absorbs
        mass = float(np.sum(np.exp(log_gamma_pdf(nodes, c, rate=d) + log_w)))
        assert mass == pytest.approx(1.0, abs=5e-3), f"{scheme} c={c} d={d}: mass={mass}"


def test_gamma_quantile_grid_absorbs_the_prior_exactly():
    """The quantile axis carries uniform prior mass 1/n by construction."""
    nodes, log_w, absorbs = build_nu_axis(500, "gamma_quantile", c=4.0, d=2.0)
    assert absorbs
    assert np.allclose(np.exp(log_w), 1.0 / 500)
    # E[nu] under the discretised prior must approach c/d = 2.
    assert float(np.mean(nodes)) == pytest.approx(2.0, rel=5e-3)


def test_treating_a_log_grid_as_uniform_would_be_wrong():
    """Guard against the classic bug of ignoring non-uniform cell widths."""
    nodes, log_w, _ = build_nu_axis(50, "log", nu_lo=1e-3, nu_hi=250.0)
    widths = np.exp(log_w)
    assert widths.max() / widths.min() > 1e4          # strongly non-uniform
    naive = float(np.sum(np.exp(log_gamma_pdf(nodes, 3.0, rate=1.0)) * widths.mean()))
    correct = float(np.sum(np.exp(log_gamma_pdf(nodes, 3.0, rate=1.0) + log_w)))
    assert abs(correct - 1.0) < abs(naive - 1.0)


def test_grid_normaliser_matches_dblquad_on_a_small_problem(toy_domain):
    """Section 13.5: validate the grid approximation against scipy.dblquad."""
    cfg = HyperparameterConfiguration(3.0, 2.0, 6.0, 3.0)

    def integrand(nu: float, mu: float) -> float:
        ll = log_beta_binomial_evidence(
            toy_domain.counts, toy_domain.trials, np.array([mu]), np.array([nu])
        )[0]
        return float(
            np.exp(
                ll
                + log_beta_pdf(np.array([mu]), cfg.a, cfg.b)[0]
                + log_gamma_pdf(np.array([nu]), cfg.c, rate=cfg.d)[0]
            )
        )

    ref, err = integrate.dblquad(
        integrand, 1e-8, 1 - 1e-8, lambda _: 1e-8, lambda _: 200.0, epsabs=1e-14, epsrel=1e-10
    )
    for scheme in ("log", "linear", "gamma_quantile"):
        grid = build_grid(300, 300, nu_scheme=scheme, config=cfg,
                          nu_params={"nu_lo": 1e-5, "nu_hi": 300.0})
        _, log_norm = log_hyperposterior(toy_domain, cfg, grid)
        assert np.exp(log_norm) == pytest.approx(ref, rel=5e-3, abs=100 * err), scheme


def test_all_nu_schemes_agree_on_the_posterior_mean(toy_domain):
    """The unresolved nu-axis choice must not move the posterior materially."""
    cfg = HyperparameterConfiguration(4.0, 4.0, 8.0, 2.0)
    means = []
    for scheme in ("log", "linear", "gamma_quantile"):
        grid = build_grid(120, 120, nu_scheme=scheme, config=cfg)
        probs, _ = log_hyperposterior(toy_domain, cfg, grid)
        means.append(float(np.dot(probs, grid.mu)))
    assert max(means) - min(means) < 5e-3, f"mu posterior means diverge: {means}"


# --- strict mode ------------------------------------------------------------ #
def test_strict_mode_refuses_the_unresolved_grid():
    with pytest.raises(StrictModeError, match="nu"):
        build_grid(40, 50, strict_exact=True,
                   config=HyperparameterConfiguration(1.0, 1.0, 1.0, 1.0))


def test_strict_mode_refuses_the_unresolved_configuration_rule(wide_interval):
    with pytest.raises(StrictModeError, match="configurations"):
        sample_configurations(wide_interval, K=160, seed=123, strict_exact=True)


# --- configuration sampling ------------------------------------------------- #
@pytest.mark.parametrize("scheme", sorted(CONFIG_SAMPLING_SCHEMES))
def test_configurations_stay_inside_the_admissible_box(wide_interval, scheme):
    cfgs = sample_configurations(wide_interval, K=160, seed=123, scheme=scheme)
    assert len(cfgs) == 160
    arr = np.array([c.as_array() for c in cfgs])
    lo, hi = wide_interval.bounds[:, 0], wide_interval.bounds[:, 1]
    assert np.all(arr >= lo - 1e-12) and np.all(arr <= hi + 1e-12)


def test_precise_interval_collapses_to_identical_configurations(precise_interval):
    cfgs = sample_configurations(precise_interval, K=25, seed=1)
    arr = np.array([c.as_array() for c in cfgs])
    assert np.allclose(arr, arr[0])
    assert precise_interval.is_precise


def test_corner_scheme_includes_every_box_corner(wide_interval):
    cfgs = sample_configurations(
        wide_interval, K=160, seed=5, scheme="interval_corners_plus_interior"
    )
    arr = np.array([c.as_array() for c in cfgs])
    lo, hi = wide_interval.bounds[:, 0], wide_interval.bounds[:, 1]
    corners = {tuple(v) for v in np.array(np.meshgrid(*zip(lo, hi), indexing="ij")).reshape(4, -1).T}
    found = {tuple(v) for v in arr[:16]}
    assert corners == found


# --- caching must not change results ---------------------------------------- #
def test_cache_returns_identical_hyperposteriors(toy_domain):
    cache = HyperposteriorCache()
    cfg = HyperparameterConfiguration(3.0, 4.0, 5.0, 2.0)
    grid = build_grid(40, 50, config=cfg)
    first = cache.get(toy_domain, cfg, grid)
    second = cache.get(toy_domain, cfg, grid)
    assert cache.hits == 1 and cache.misses == 1
    assert np.array_equal(first.probs, second.probs)
    direct, _ = log_hyperposterior(toy_domain, cfg, grid)
    assert np.array_equal(first.probs, direct)


def test_cache_key_separates_different_data(toy_domain):
    from hip_llm.schemas import DomainData, SubdomainData

    other = DomainData(
        "D1", (SubdomainData("A", 10, 80), SubdomainData("B", 39, 80)), np.array([0.5, 0.5])
    )
    cache = HyperposteriorCache()
    cfg = HyperparameterConfiguration(3.0, 4.0, 5.0, 2.0)
    grid = build_grid(20, 20, config=cfg)
    a = cache.get(toy_domain, cfg, grid)
    b = cache.get(other, cfg, grid)
    assert cache.misses == 2
    assert not np.allclose(a.probs, b.probs)
