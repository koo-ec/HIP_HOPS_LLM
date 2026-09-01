"""Distribution and conjugacy tests (specification items 1-4).

Also covers the analytical special cases of specification Section 14.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from conftest import TOL_MC_MEAN, TOL_MC_VAR
from hip_llm.grids import build_grid
from hip_llm.hyperposterior import Hyperposterior, log_hyperposterior
from hip_llm.numerics import (
    log_beta_binomial_evidence,
    log_beta_binomial_pmf,
    log_beta_pdf,
    log_gamma_pdf,
)
from hip_llm.posterior import sample_domain_posterior
from hip_llm.schemas import DomainData, HyperparameterConfiguration, SubdomainData


# --- 1. sampled vs analytical conditional Beta posterior --------------------- #
@pytest.mark.parametrize("mu,nu", [(0.5, 4.0), (0.8, 20.0), (0.3, 1.5)])
def test_conditional_beta_moments_match_analytical(toy_domain: DomainData, mu, nu):
    """For fixed (mu, nu) the draws must match Beta(C+mu*nu, N-C+(1-mu)*nu)."""
    rng = np.random.default_rng(20260728)
    C, N = toy_domain.counts, toy_domain.trials
    alpha = C + mu * nu
    beta = (N - C) + (1.0 - mu) * nu
    draws = rng.beta(
        np.broadcast_to(alpha, (200_000, alpha.size)),
        np.broadcast_to(beta, (200_000, beta.size)),
    )

    exp_mean = alpha / (alpha + beta)
    exp_var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))

    assert np.allclose(draws.mean(axis=0), exp_mean, atol=TOL_MC_MEAN)
    assert np.allclose(draws.var(axis=0), exp_var, rtol=TOL_MC_VAR)


def test_sampler_produces_the_analytical_conditional_posterior(toy_domain: DomainData):
    """Case 1 of Section 14: degenerate hyperposterior -> exact Beta posterior."""
    cfg = HyperparameterConfiguration(2.0, 2.0, 3.0, 1.0)
    grid = build_grid(4, 4, config=cfg)
    # Collapse the hyperposterior onto a single (mu, nu) cell.
    probs = np.zeros(grid.size)
    probs[7] = 1.0
    hp = Hyperposterior("D1", cfg, grid, probs, 0.0)
    mu, nu = grid.mu[7], grid.nu[7]

    ps = sample_domain_posterior(toy_domain, hp, 120_000, np.random.default_rng(11))
    alpha = toy_domain.counts + mu * nu
    beta = (toy_domain.trials - toy_domain.counts) + (1.0 - mu) * nu
    expected = alpha / (alpha + beta)
    assert np.allclose(ps.theta.mean(axis=0), expected, atol=TOL_MC_MEAN)
    assert np.allclose(ps.mu, mu) and np.allclose(ps.nu, nu)


# --- 2. Beta-Binomial evidence vs direct numerical integration --------------- #
@pytest.mark.parametrize("C,N,mu,nu", [(3, 10, 0.5, 4.0), (7, 12, 0.7, 9.0), (0, 5, 0.25, 2.0)])
def test_beta_binomial_evidence_matches_quadrature(C, N, mu, nu):
    """int_0^1 Binom(C|N,t) Beta(t|mu*nu,(1-mu)*nu) dt must equal the closed form."""
    a, b = mu * nu, (1.0 - mu) * nu

    def integrand(t: float) -> float:
        return float(stats.binom.pmf(C, N, t) * stats.beta.pdf(t, a, b))

    numeric, err = integrate.quad(integrand, 0.0, 1.0, limit=400)
    closed = float(np.exp(log_beta_binomial_pmf(np.array(C), np.array(N), a, b)))
    assert numeric == pytest.approx(closed, rel=1e-6, abs=max(1e-12, 10 * err))

    # And the vectorised evidence used by the hyperposterior agrees once the
    # binomial coefficient is restored.
    vec = log_beta_binomial_evidence(
        np.array([C]), np.array([N]), np.array([mu]), np.array([nu]), include_binomial_coeff=True
    )
    assert float(vec[0]) == pytest.approx(np.log(closed), rel=1e-10)


def test_omitting_binomial_coefficient_does_not_change_the_hyperposterior(toy_domain):
    """The dropped binomial coefficients are constant in (mu, nu) and must cancel."""
    cfg = HyperparameterConfiguration(2.0, 3.0, 4.0, 1.0)
    grid = build_grid(12, 12, config=cfg)
    with_coeff = log_beta_binomial_evidence(
        toy_domain.counts, toy_domain.trials, grid.mu, grid.nu, include_binomial_coeff=True
    )
    without = log_beta_binomial_evidence(
        toy_domain.counts, toy_domain.trials, grid.mu, grid.nu, include_binomial_coeff=False
    )
    diff = with_coeff - without
    assert np.allclose(diff, diff[0])          # constant offset
    assert diff[0] == pytest.approx(0.0, abs=1e-9) or diff[0] != 0.0


# --- 3. Gamma uses shape-RATE, not shape-scale ------------------------------- #
@pytest.mark.parametrize("c,d", [(3.0, 2.0), (25.0, 1.0), (1.0, 25.0)])
def test_gamma_is_shape_rate_not_shape_scale(c, d):
    """Paper footnote 16: E[nu] = c/d and Var[nu] = c/d^2 under shape-rate."""
    # Grid adapted to the distribution: Gamma(1, rate=25) has mean 0.04, so a
    # fixed 0.01-60 grid would miss most of its mass.
    upper = stats.gamma.ppf(1 - 1e-12, a=c, scale=1.0 / d)
    x = np.linspace(1e-12, upper, 400_000)
    pdf = np.exp(log_gamma_pdf(x, c, rate=d))
    mass = np.trapezoid(pdf, x)
    mean = np.trapezoid(x * pdf, x)
    var = np.trapezoid((x - c / d) ** 2 * pdf, x)

    assert mass == pytest.approx(1.0, abs=2e-3)
    assert mean == pytest.approx(c / d, rel=2e-3)
    assert var == pytest.approx(c / d**2, rel=2e-2)

    # Explicitly differs from the shape-SCALE reading whenever d != 1.
    assert np.allclose(pdf, stats.gamma.pdf(x, a=c, scale=1.0 / d))
    if d != 1.0:
        assert not np.allclose(pdf, stats.gamma.pdf(x, a=c, scale=d))


def test_beta_pdf_matches_scipy():
    x = np.linspace(1e-4, 1 - 1e-4, 5000)
    for a, b in [(1.0, 1.0), (2.0, 5.0), (12.0, 1.0)]:
        assert np.allclose(np.exp(log_beta_pdf(x, a, b)), stats.beta.pdf(x, a, b), rtol=1e-10)


# --- 4. hyperposterior is finite, non-negative and normalised ---------------- #
@pytest.mark.parametrize("nu_scheme", ["log", "linear", "gamma_quantile"])
@pytest.mark.parametrize(
    "cfg",
    [(1.0, 1.0, 1.0, 1.0), (12.0, 12.0, 25.0, 25.0), (1.0, 12.0, 25.0, 1.0), (12.0, 1.0, 1.0, 25.0)],
)
def test_hyperposterior_is_finite_nonnegative_normalised(toy_domain, nu_scheme, cfg):
    config = HyperparameterConfiguration(*cfg)
    grid = build_grid(40, 50, nu_scheme=nu_scheme, config=config)
    probs, log_norm = log_hyperposterior(toy_domain, config, grid)
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.isfinite(log_norm)


def test_hyperposterior_reproduces_a_known_2d_integral(toy_domain):
    """The grid normaliser must agree with scipy.dblquad on the same integrand."""
    cfg = HyperparameterConfiguration(2.0, 3.0, 4.0, 2.0)

    def integrand(nu: float, mu: float) -> float:
        ll = log_beta_binomial_evidence(
            toy_domain.counts, toy_domain.trials, np.array([mu]), np.array([nu])
        )[0]
        return float(np.exp(ll + log_beta_pdf(np.array([mu]), cfg.a, cfg.b)[0]
                            + log_gamma_pdf(np.array([nu]), cfg.c, rate=cfg.d)[0]))

    reference, err = integrate.dblquad(
        integrand, 1e-6, 1 - 1e-6, lambda _: 1e-6, lambda _: 120.0, epsabs=1e-13, epsrel=1e-9
    )
    grid = build_grid(400, 400, nu_scheme="log", config=cfg, nu_params={"nu_lo": 1e-4, "nu_hi": 400.0})
    _, log_norm = log_hyperposterior(toy_domain, cfg, grid)
    assert np.exp(log_norm) == pytest.approx(reference, rel=2e-3, abs=10 * err)


# --- Section 14 analytical special cases ------------------------------------ #
def test_case2_single_subdomain_domain_equals_subdomain(fast_settings, wide_interval):
    """Case 2: with Omega = [1] the domain posterior IS the subdomain posterior."""
    from hip_llm.posterior import run_domain

    d = DomainData("D1", (SubdomainData("only", 38, 80, 0.475),), np.array([1.0]))
    ds = run_domain(d, wide_interval, fast_settings, domain_index=0)
    assert np.allclose(ds.p, ds.theta[:, :, 0])


def test_case3_single_domain_llm_equals_domain(fast_settings, wide_interval, toy_domain):
    """Case 3: with W = [1] the LLM posterior IS the domain posterior."""
    from hip_llm.posterior import run_model
    from hip_llm.schemas import ModelResult

    model = ModelResult("one-domain", (toy_domain,), np.array([1.0]), "unit test")
    domain_sets, llm = run_model(model, [wide_interval], fast_settings)
    assert np.allclose(llm.p_L, domain_sets[0].p[llm.pair_index[:, 0], :])


def test_case6_large_data_limit_shrinks_the_envelope(wide_interval):
    """Case 6: as N grows with C/N fixed, prior influence and envelope width fall."""
    from hip_llm.envelopes import summarise_envelope
    from hip_llm.posterior import run_domain
    from hip_llm.schemas import GlobalSettings

    st = GlobalSettings(
        n_mu=30, n_nu=30, cdf_points_T=201, S=4000, K_per_domain=16,
        max_llm_configuration_pairs=64, seed_global=7, seed_configs=123, seed_pairs=999,
        config_sampling="uniform_random", nu_grid_scheme="log", mu_grid_scheme="midpoint",
    )
    widths = []
    for N in (40, 400, 4000):
        d = DomainData(
            "D1",
            (SubdomainData("A", int(0.5 * N), N, 0.5), SubdomainData("B", int(0.6 * N), N, 0.6)),
            np.array([0.5, 0.5]),
        )
        ds = run_domain(d, wide_interval, st, domain_index=0)
        s = summarise_envelope(ds.p, "p")
        widths.append((s.median_upper - s.median_lower, s.q95_upper - s.q05_lower))

    median_widths = [w[0] for w in widths]
    posterior_widths = [w[1] for w in widths]
    assert median_widths[0] > median_widths[1] > median_widths[2]
    assert posterior_widths[0] > posterior_widths[1] > posterior_widths[2]
