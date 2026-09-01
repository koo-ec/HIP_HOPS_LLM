"""Hierarchical-dependence tests (specification items 11-15).

These are the tests that would catch the single most damaging implementation
error available in this model: aggregating independently-drawn subdomain
marginals instead of jointly-drawn conditionals, which silently destroys the
partial pooling the whole paper is about.
"""

from __future__ import annotations

import numpy as np
import pytest

from hip_llm.grids import build_grid
from hip_llm.hyperposterior import Hyperposterior, log_hyperposterior
from hip_llm.posterior import run_domain, run_model, sample_domain_posterior
from hip_llm.schemas import DomainData, HyperparameterConfiguration, SubdomainData


def _degenerate_hyperposterior(domain, cfg, cell=5):
    grid = build_grid(8, 8, config=cfg)
    probs = np.zeros(grid.size)
    probs[cell] = 1.0
    return Hyperposterior(domain.name, cfg, grid, probs, 0.0), grid.mu[cell], grid.nu[cell]


# A configuration whose Gamma(c, rate=d) prior puts nu near 25, i.e. comparable
# to the number of observations, so that pooling is actually exercised.  See
# discrepancy D10: the paper's own baseline box makes E[nu | data] ~ 1, under
# which pooling is real but numerically negligible.  Both regimes are tested.
STRONG_POOLING = HyperparameterConfiguration(1.0, 1.0, 25.0, 1.0)


def _sample_under(domain, cfg, S=200_000, seed=1, n_grid=60):
    grid = build_grid(n_grid, n_grid, config=cfg)
    probs, _ = log_hyperposterior(domain, cfg, grid)
    hp = Hyperposterior(domain.name, cfg, grid, probs, 0.0)
    return sample_domain_posterior(domain, hp, S, np.random.default_rng(seed))


def _small_n_domain(acc_a=0.25, acc_b=0.75, N=8):
    return DomainData(
        "D",
        (SubdomainData("A", round(acc_a * N), N), SubdomainData("B", round(acc_b * N), N)),
        np.array([0.5, 0.5]),
    )


# --- 11. conditional independence given (mu, nu) ---------------------------- #
def test_subdomains_are_independent_given_fixed_mu_nu(toy_domain):
    cfg = HyperparameterConfiguration(2.0, 2.0, 4.0, 1.0)
    hp, _, _ = _degenerate_hyperposterior(toy_domain, cfg)
    ps = sample_domain_posterior(toy_domain, hp, 200_000, np.random.default_rng(3))
    r = np.corrcoef(ps.theta[:, 0], ps.theta[:, 1])[0, 1]
    assert abs(r) < 0.01, f"conditional correlation should vanish, got {r:.4f}"


# --- 12. marginal positive dependence after integrating out (mu, nu) -------- #
@pytest.mark.parametrize("N,floor", [(8, 0.40), (20, 0.25), (80, 0.07)])
def test_marginal_within_domain_dependence_is_positive(N, floor):
    """After marginalising the shared (mu, nu) the subdomains are positively dependent.

    Exercised in the regime the mechanism is designed for (prior strength nu
    comparable to N).  Correlation necessarily decays as N grows because the
    subdomain's own data dominates the shared prior.
    """
    ps = _sample_under(_small_n_domain(N=N), STRONG_POOLING)
    r = float(np.corrcoef(ps.theta[:, 0], ps.theta[:, 1])[0, 1])
    assert r > floor, f"N={N}: expected r > {floor}, got {r:.4f}"

    # Equivalent variance statement: Cov > 0 <=> Var(sum) > sum of variances.
    var_sum = float(np.var(ps.theta.sum(axis=1)))
    independent_sum = float(np.var(ps.theta[:, 0]) + np.var(ps.theta[:, 1]))
    assert var_sum > independent_sum * 1.05


def test_dependence_is_positive_but_weak_at_the_paper_baseline(toy_domain, wide_interval):
    """Discrepancy D10, verified numerically.

    Under the paper's own box (a,b in [1,12], c,d in [1,25]) the hyperposterior
    puts nu near 1, so against N = 80 observations pooling barely operates.  The
    dependence is still positive -- the mechanism is intact -- but ~0.01, an order
    of magnitude below what the model delivers when nu is large.
    """
    from hip_llm.schemas import GlobalSettings

    st = GlobalSettings(
        n_mu=40, n_nu=50, cdf_points_T=201, S=20_000, K_per_domain=8,
        max_llm_configuration_pairs=16, seed_global=7, seed_configs=123, seed_pairs=999,
        config_sampling="uniform_random", nu_grid_scheme="log", mu_grid_scheme="midpoint",
    )
    ds = run_domain(toy_domain, wide_interval, st, domain_index=0)
    r_baseline = float(
        np.corrcoef(ds.theta[:, :, 0].ravel(), ds.theta[:, :, 1].ravel())[0, 1]
    )
    assert 0.0 < r_baseline < 0.05, f"expected a small positive correlation, got {r_baseline:.4f}"

    # The same data under a strong-nu configuration is an order of magnitude larger.
    strong = _sample_under(toy_domain, STRONG_POOLING)
    r_strong = float(np.corrcoef(strong.theta[:, 0], strong.theta[:, 1])[0, 1])
    assert r_strong > 5 * r_baseline

    # And the posterior mean of nu is what explains the difference.
    assert float(ds.nu.mean()) < 6.0
    assert float(strong.nu.mean()) > 15.0


def test_independent_marginal_sampling_would_destroy_the_dependence():
    """The incorrect implementation must be measurably different from ours."""
    ps = _sample_under(_small_n_domain(N=8), STRONG_POOLING)
    correct = float(np.corrcoef(ps.theta[:, 0], ps.theta[:, 1])[0, 1])

    rng = np.random.default_rng(99)
    broken_b = ps.theta[rng.permutation(ps.n_samples), 1]   # break the joint pairing
    broken = float(np.corrcoef(ps.theta[:, 0], broken_b)[0, 1])

    assert correct > 0.40
    assert abs(broken) < 0.02
    assert correct - broken > 0.38


# --- 13. cross-domain independence ------------------------------------------ #
def test_domains_are_independent(toy_model, fast_settings, wide_interval):
    domain_sets, llm = run_model(toy_model, [wide_interval, wide_interval], fast_settings)
    p1 = domain_sets[0].p[llm.pair_index[:, 0], :].ravel()
    p2 = domain_sets[1].p[llm.pair_index[:, 1], :].ravel()
    r = float(np.corrcoef(p1, p2)[0, 1])
    assert abs(r) < 0.05, f"cross-domain correlation should vanish, got {r:.4f}"


# --- 14. a sparse subdomain is shrunk more strongly ------------------------- #
def test_sparse_subdomain_is_pulled_harder_toward_the_domain_mean(sparse_rich_domain):
    """Both subdomains have C/N = 0.5; the sparse one must move further.

    With a prior mean pulled upward (a > b) the sparse subdomain's posterior mean
    must exceed the data-rich one's, because the rich subdomain is dominated by
    its own 800 observations.
    """
    cfg = HyperparameterConfiguration(10.0, 2.0, 20.0, 1.0)   # prior mean 0.83, strong pooling
    grid = build_grid(60, 60, config=cfg)
    probs, _ = log_hyperposterior(sparse_rich_domain, cfg, grid)
    hp = Hyperposterior(sparse_rich_domain.name, cfg, grid, probs, 0.0)
    ps = sample_domain_posterior(sparse_rich_domain, hp, 60_000, np.random.default_rng(4))

    sparse_mean, rich_mean = ps.theta[:, 0].mean(), ps.theta[:, 1].mean()
    assert sparse_mean > rich_mean + 0.02, (sparse_mean, rich_mean)
    assert abs(rich_mean - 0.5) < abs(sparse_mean - 0.5)


# --- 15. increasing nu strengthens pooling ---------------------------------- #
def test_larger_nu_strengthens_pooling():
    """Two subdomains with very different rates converge as nu grows."""
    d = DomainData(
        "D",
        (SubdomainData("low", 20, 80, 0.25), SubdomainData("high", 60, 80, 0.75)),
        np.array([0.5, 0.5]),
    )
    gaps = []
    for nu in (0.5, 5.0, 50.0, 500.0):
        cfg = HyperparameterConfiguration(2.0, 2.0, 1.0, 1.0)
        grid = build_grid(8, 8, config=cfg)
        probs = np.zeros(grid.size)
        # place all mass on a cell with mu ~ 0.5 and the requested nu
        mu_target = 0.5
        idx = int(np.argmin((grid.mu - mu_target) ** 2))
        probs[idx] = 1.0
        mu_val = grid.mu[idx]
        alpha = d.counts + mu_val * nu
        beta = (d.trials - d.counts) + (1 - mu_val) * nu
        means = alpha / (alpha + beta)
        gaps.append(float(abs(means[1] - means[0])))
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3], gaps


@pytest.mark.parametrize("N,min_shift", [(8, 0.12), (20, 0.08), (80, 0.03)])
def test_pooling_moves_a_subdomain_toward_its_partner(N, min_shift):
    """Adding a high-accuracy partner must raise a low-accuracy subdomain's posterior."""
    alone = DomainData(
        "D",
        (SubdomainData("A", round(0.5 * N), N), SubdomainData("A2", round(0.5 * N), N)),
        np.array([0.5, 0.5]),
    )
    with_partner = DomainData(
        "D",
        (SubdomainData("A", round(0.5 * N), N), SubdomainData("B", round(0.95 * N), N)),
        np.array([0.5, 0.5]),
    )
    m_alone = float(_sample_under(alone, STRONG_POOLING, S=100_000, seed=2).theta[:, 0].mean())
    m_pooled = float(_sample_under(with_partner, STRONG_POOLING, S=100_000, seed=2).theta[:, 0].mean())
    assert m_alone == pytest.approx(0.5, abs=0.01)
    assert m_pooled > m_alone + min_shift, (N, m_alone, m_pooled)


def test_same_shared_latents_are_used_across_subdomains(toy_domain, fast_settings, wide_interval):
    """Structural check: one (mu, nu) pair per Monte-Carlo draw, shared by all j."""
    ds = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    assert ds.mu.shape == (ds.n_configs, ds.n_samples)
    assert ds.nu.shape == (ds.n_configs, ds.n_samples)
    # Re-deriving the conditional Beta means from the recorded latents must
    # reproduce the sample means: proof the draws used exactly those latents.
    k = 0
    mu, nu = ds.mu[k], ds.nu[k]
    C, N = toy_domain.counts, toy_domain.trials
    alpha = C[None, :] + (mu * nu)[:, None]
    beta = (N - C)[None, :] + ((1 - mu) * nu)[:, None]
    expected = (alpha / (alpha + beta)).mean(axis=0)
    assert np.allclose(ds.theta[k].mean(axis=0), expected, atol=0.01)
