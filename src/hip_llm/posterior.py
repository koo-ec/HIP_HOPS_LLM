"""Monte-Carlo posterior sampling and hierarchical aggregation.

The sampling procedure follows paper Appendix B ("Simulation") exactly:

1. evaluate the hyperposterior on the :math:`(\\mu,\\nu)` grid in log space with
   quadrature weights, normalised by ``logsumexp``;
2. draw :math:`S` grid indices from that discrete hyperposterior;
3. for each drawn pair, draw **all** subdomain :math:`\\theta_{ij}` conditionally
   from :math:`\\mathrm{Beta}(C_{ij}+\\mu\\nu,\\;N_{ij}-C_{ij}+(1-\\mu)\\nu)`
   using the *same* shared :math:`(\\mu,\\nu)` on that Monte-Carlo draw;
4. aggregate through :math:`\\Omega_{ij}` and :math:`W_i`.

Step 3 is what preserves partial pooling and the marginal within-domain
dependence: aggregating independently-drawn marginals instead would silently
destroy it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .grids import build_grid, pair_llm_configurations, sample_configurations
from .hyperposterior import Hyperposterior, HyperposteriorCache, log_hyperposterior
from .numerics import assert_finite, child_generator
from .schemas import (
    DomainData,
    GlobalSettings,
    HyperparameterConfiguration,
    HyperparameterInterval,
    HyperposteriorGrid,
    ModelResult,
    PosteriorSamples,
)

__all__ = [
    "sample_domain_posterior",
    "DomainPosteriorSet",
    "LLMPosteriorSet",
    "run_domain",
    "run_model",
]


def sample_domain_posterior(
    domain: DomainData,
    hyperposterior: Hyperposterior,
    S: int,
    rng: np.random.Generator,
) -> PosteriorSamples:
    """Draw ``S`` joint posterior samples for one domain under one configuration.

    Returns a :class:`~hip_llm.schemas.PosteriorSamples` whose ``theta`` rows are
    *jointly* drawn under a shared latent :math:`(\\mu,\\nu)`, and whose ``p``
    is :math:`p_i=\\sum_j\\Omega_{ij}\\theta_{ij}` computed within each draw
    (paper Theorem 2).
    """
    if S <= 0:
        raise ValueError("S must be positive")

    idx = hyperposterior.draw_indices(rng, S)
    mu = hyperposterior.grid.mu[idx]
    nu = hyperposterior.grid.nu[idx]

    C = domain.counts[None, :]          # (1, n_i)
    N = domain.trials[None, :]
    alpha = C + (mu * nu)[:, None]      # (S, n_i) -- shared (mu, nu) across j
    beta = (N - C) + ((1.0 - mu) * nu)[:, None]

    assert_finite(alpha, "conditional Beta alpha")
    assert_finite(beta, "conditional Beta beta")
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise FloatingPointError("non-positive Beta parameters in conditional posterior")

    theta = rng.beta(alpha, beta)
    # Guard against exact 0/1 draws from extreme shapes before downstream powers.
    theta = np.clip(theta, 0.0, 1.0)

    p = theta @ domain.omega
    p = np.clip(p, 0.0, 1.0)

    return PosteriorSamples(
        domain=domain.name,
        config=hyperposterior.config,
        theta=theta,
        mu=mu,
        nu=nu,
        p=p,
        subdomain_names=domain.subdomain_names,
    )


@dataclass(frozen=True)
class DomainPosteriorSet:
    """The family of posteriors for one domain, indexed by :math:`h_i\\in\\mathcal{H}_i`."""

    domain: str
    subdomain_names: tuple[str, ...]
    configs: tuple[HyperparameterConfiguration, ...]
    theta: np.ndarray          # (K, S, n_i)
    p: np.ndarray              # (K, S)
    mu: np.ndarray             # (K, S)
    nu: np.ndarray             # (K, S)
    meta: Mapping[str, object]

    def __post_init__(self) -> None:
        K, S, n = self.theta.shape
        if self.p.shape != (K, S):
            raise ValueError(f"p has shape {self.p.shape}, expected {(K, S)}")
        if len(self.configs) != K:
            raise ValueError("configs length does not match the first axis of theta")
        if n != len(self.subdomain_names):
            raise ValueError("theta's last axis does not match the subdomain names")

    @property
    def n_configs(self) -> int:
        return self.theta.shape[0]

    @property
    def n_samples(self) -> int:
        return self.theta.shape[1]

    def subdomain_samples(self, name: str) -> np.ndarray:
        """``(K, S)`` posterior draws of :math:`\\theta_{ij}` for one subdomain."""
        j = self.subdomain_names.index(name)
        return self.theta[:, :, j]


@dataclass(frozen=True)
class LLMPosteriorSet:
    """LLM-level posterior family, one row per admissible configuration tuple."""

    model: str
    p_L: np.ndarray                    # (n_pairs, S)
    pair_index: np.ndarray             # (n_pairs, m) per-domain config indices
    domain_names: tuple[str, ...]
    meta: Mapping[str, object]

    @property
    def n_configs(self) -> int:
        return self.p_L.shape[0]

    @property
    def n_samples(self) -> int:
        return self.p_L.shape[1]


def run_domain(
    domain: DomainData,
    interval: HyperparameterInterval,
    settings: GlobalSettings,
    domain_index: int,
    model_index: int = 0,
    cache: HyperposteriorCache | None = None,
    configs: Sequence[HyperparameterConfiguration] | None = None,
    progress: object | None = None,
) -> DomainPosteriorSet:
    """Run the full imprecise inference for one domain.

    RNG streams are addressed deterministically as
    ``child_generator(settings.seed_global, (model_index, domain_index, k))`` so
    that every (model, domain, configuration) triple owns an independent,
    reproducible stream regardless of evaluation order or parallelism.
    """
    if configs is None:
        configs = sample_configurations(
            interval,
            K=settings.K_per_domain,
            seed=settings.seed_configs,
            scheme=settings.config_sampling,
            strict_exact=settings.strict_exact,
        )

    K = len(configs)
    S = settings.S
    n_i = domain.n_subdomains

    theta = np.empty((K, S, n_i), dtype=float)
    p = np.empty((K, S), dtype=float)
    mu = np.empty((K, S), dtype=float)
    nu = np.empty((K, S), dtype=float)

    # A configuration-independent grid can be built once and reused; a
    # quantile-transformed grid must be rebuilt per configuration.
    shared_grid: HyperposteriorGrid | None = None
    probe = build_grid(
        n_mu=settings.n_mu,
        n_nu=settings.n_nu,
        mu_scheme=settings.mu_grid_scheme,
        nu_scheme=settings.nu_grid_scheme,
        config=configs[0],
        strict_exact=settings.strict_exact,
        nu_params=settings.nu_grid_params,
    )
    if not probe.meta.get("config_dependent", False):
        shared_grid = probe

    iterator: Iterable[int] = range(K)
    if progress is not None:
        iterator = progress(iterator, total=K, desc=f"{domain.name}: configs")

    for k in iterator:
        cfg = configs[k]
        grid = shared_grid
        if grid is None:
            grid = build_grid(
                n_mu=settings.n_mu,
                n_nu=settings.n_nu,
                mu_scheme=settings.mu_grid_scheme,
                nu_scheme=settings.nu_grid_scheme,
                config=cfg,
                strict_exact=settings.strict_exact,
                nu_params=settings.nu_grid_params,
            )

        if cache is not None:
            hp = cache.get(domain, cfg, grid)
        else:
            probs, log_norm = log_hyperposterior(domain, cfg, grid)
            hp = Hyperposterior(
                domain=domain.name, config=cfg, grid=grid, probs=probs, log_evidence=log_norm
            )

        rng = child_generator(settings.seed_global, (model_index, domain_index, k))
        ps = sample_domain_posterior(domain, hp, S, rng)
        theta[k] = ps.theta
        p[k] = ps.p
        mu[k] = ps.mu
        nu[k] = ps.nu

    return DomainPosteriorSet(
        domain=domain.name,
        subdomain_names=domain.subdomain_names,
        configs=tuple(configs),
        theta=theta,
        p=p,
        mu=mu,
        nu=nu,
        meta={
            "S": S,
            "K": K,
            "G": settings.G,
            "grid_scheme": probe.scheme,
            "config_sampling": settings.config_sampling,
            "seed_global": settings.seed_global,
            "seed_configs": settings.seed_configs,
            "model_index": model_index,
            "domain_index": domain_index,
        },
    )


def run_model(
    model: ModelResult,
    intervals: Sequence[HyperparameterInterval],
    settings: GlobalSettings,
    model_index: int = 0,
    cache: HyperposteriorCache | None = None,
    pairing_mode: str = "capped_random",
    progress: object | None = None,
) -> tuple[tuple[DomainPosteriorSet, ...], LLMPosteriorSet]:
    """Run every domain of one LLM and aggregate to the LLM level.

    Domain posteriors use independent Monte-Carlo streams (cross-domain
    independence, paper Theorem 3) while within-domain dependence is preserved by
    the shared latent draws inside :func:`sample_domain_posterior`.

    LLM-level aggregation pairs *tuples* of domain configurations, never pairing
    domains independently -- see :func:`hip_llm.grids.pair_llm_configurations`.
    """
    if len(intervals) != model.n_domains:
        raise ValueError(
            f"{len(intervals)} hyperparameter intervals for {model.n_domains} domains"
        )

    domain_sets = tuple(
        run_domain(
            domain=d,
            interval=intervals[i],
            settings=settings,
            domain_index=i,
            model_index=model_index,
            cache=cache,
            progress=progress,
        )
        for i, d in enumerate(model.domains)
    )

    K_values = {ds.n_configs for ds in domain_sets}
    if len(K_values) != 1:
        raise ValueError(f"domains produced differing configuration counts: {K_values}")
    K = K_values.pop()

    pair_index = pair_llm_configurations(
        K_per_domain=K,
        n_domains=model.n_domains,
        max_pairs=settings.max_llm_configuration_pairs,
        seed=settings.seed_pairs,
        mode=pairing_mode,
    )

    p_L = np.zeros((pair_index.shape[0], settings.S), dtype=float)
    for i, ds in enumerate(domain_sets):
        p_L += model.W[i] * ds.p[pair_index[:, i], :]
    p_L = np.clip(assert_finite(p_L, "p_L"), 0.0, 1.0)

    llm_set = LLMPosteriorSet(
        model=model.model,
        p_L=p_L,
        pair_index=pair_index,
        domain_names=tuple(d.name for d in model.domains),
        meta={
            "pairing_mode": pairing_mode,
            "n_pairs": int(pair_index.shape[0]),
            "max_llm_configuration_pairs": settings.max_llm_configuration_pairs,
            "seed_pairs": settings.seed_pairs,
            "W": model.W.tolist(),
            "source_label": model.source_label,
        },
    )
    return domain_sets, llm_set
