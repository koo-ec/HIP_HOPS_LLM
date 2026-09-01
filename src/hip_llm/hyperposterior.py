"""Domain-level hyperposterior :math:`\\Pr(\\mu_i,\\nu_i \\mid C_i, h_i)`.

Implements paper Theorem 2 / Lemma 1 (Eq. A.9--A.11):

.. math::

    \\Pr(\\mu_i,\\nu_i\\mid C_i,h_i)
      \\;\\propto\\;
      \\Pr(C_i\\mid\\mu_i,\\nu_i)\\,
      \\mathrm{Beta}(\\mu_i\\mid a_i,b_i)\\,
      \\mathrm{Gamma}(\\nu_i\\mid c_i,\\ \\mathrm{rate}=d_i)

evaluated on the discretised grid, in log space, with the quadrature cell
weights included and normalised via ``logsumexp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .grids import build_grid
from .numerics import (
    assert_finite,
    log_beta_binomial_evidence,
    log_beta_pdf,
    log_gamma_pdf,
    normalise_log_weights,
)
from .schemas import (
    DomainData,
    HyperparameterConfiguration,
    HyperposteriorGrid,
    config_hash,
)

__all__ = ["Hyperposterior", "log_hyperposterior", "HyperposteriorCache"]


@dataclass(frozen=True)
class Hyperposterior:
    """Normalised discrete hyperposterior over the :math:`(\\mu,\\nu)` grid."""

    domain: str
    config: HyperparameterConfiguration
    grid: HyperposteriorGrid
    probs: np.ndarray
    log_evidence: float

    def __post_init__(self) -> None:
        p = assert_finite(np.asarray(self.probs, dtype=float), "hyperposterior probs")
        if p.shape != (self.grid.size,):
            raise ValueError(f"probs has shape {p.shape}, expected ({self.grid.size},)")
        if np.any(p < 0.0):
            raise ValueError("hyperposterior contains negative probability mass")
        if abs(p.sum() - 1.0) > 1e-9:
            raise ValueError(f"hyperposterior sums to {p.sum():.17g}, expected 1")
        object.__setattr__(self, "probs", p)

    @property
    def mean_mu(self) -> float:
        return float(np.dot(self.probs, self.grid.mu))

    @property
    def mean_nu(self) -> float:
        return float(np.dot(self.probs, self.grid.nu))

    def draw_indices(self, rng: np.random.Generator, S: int) -> np.ndarray:
        """Draw ``S`` grid-cell indices from the discrete hyperposterior."""
        if S <= 0:
            raise ValueError("S must be positive")
        return rng.choice(self.grid.size, size=S, replace=True, p=self.probs)


def log_hyperposterior(
    domain: DomainData,
    config: HyperparameterConfiguration,
    grid: HyperposteriorGrid,
) -> tuple[np.ndarray, float]:
    """Unnormalised log hyperposterior on the grid, plus its log normaliser.

    The log weight of grid cell :math:`g` is

    .. math::

        \\log w_g = \\log \\Pr(C_i\\mid\\mu_g,\\nu_g)
                    + \\log \\mathrm{Beta}(\\mu_g\\mid a,b)
                    + \\log \\mathrm{Gamma}(\\nu_g\\mid c,\\mathrm{rate}=d)
                    + \\log(\\Delta\\mu_g\\,\\Delta\\nu_g)

    with the prior terms *omitted* on any axis whose quadrature weights already
    absorb them (quantile-transformed axes -- see :mod:`hip_llm.grids`).  The
    binomial coefficients of the likelihood are constant in :math:`(\\mu,\\nu)`
    and therefore cancel in the normalisation; they are omitted here and this is
    the only quantity dropped.
    """
    log_w = log_beta_binomial_evidence(
        domain.counts, domain.trials, grid.mu, grid.nu, include_binomial_coeff=False
    )

    if not grid.absorbs_mu_prior:
        log_w = log_w + log_beta_pdf(grid.mu, config.a, config.b)
    if not grid.absorbs_nu_prior:
        log_w = log_w + log_gamma_pdf(grid.nu, config.c, rate=config.d)

    log_w = log_w + grid.log_cell_weight
    assert_finite(log_w, "log hyperposterior weights")

    probs, log_norm = normalise_log_weights(log_w)
    return probs, log_norm


class HyperposteriorCache:
    """Cache hyperposteriors keyed by ``(data hash, grid hash, configuration)``.

    Required by the performance section of the specification: the grid-based
    evaluation is by far the dominant cost (paper Fig. 11f puts subdomain-level
    posterior computation at >99% of runtime), and the same
    ``(domain, config)`` pair is revisited by several research questions.

    The cache never changes the numerical result -- it is keyed on everything
    that the result depends on.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self._store: dict[str, Hyperposterior] = {}
        self._max_entries = int(max_entries)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(domain: DomainData, config: HyperparameterConfiguration, grid: HyperposteriorGrid) -> str:
        return config_hash(
            {
                "data": domain.data_key(),
                "grid": grid.grid_key(),
                "config": config.as_array().tolist(),
            }
        )

    def get(
        self,
        domain: DomainData,
        config: HyperparameterConfiguration,
        grid: HyperposteriorGrid,
    ) -> Hyperposterior:
        key = self._key(domain, config, grid)
        cached = self._store.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        probs, log_norm = log_hyperposterior(domain, config, grid)
        hp = Hyperposterior(
            domain=domain.name, config=config, grid=grid, probs=probs, log_evidence=log_norm
        )
        if len(self._store) < self._max_entries:
            self._store[key] = hp
        return hp

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> Mapping[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "entries": len(self._store),
        }

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0


def build_hyperposterior(
    domain: DomainData,
    config: HyperparameterConfiguration,
    n_mu: int,
    n_nu: int,
    mu_scheme: str = "midpoint",
    nu_scheme: str = "log",
    strict_exact: bool = False,
    nu_params: Mapping[str, object] | None = None,
) -> Hyperposterior:
    """Convenience wrapper: build the grid for ``config`` then evaluate on it.

    Necessary because the ``gamma_quantile`` / ``beta_quantile`` schemes make the
    grid itself configuration-dependent.
    """
    grid = build_grid(
        n_mu=n_mu,
        n_nu=n_nu,
        mu_scheme=mu_scheme,
        nu_scheme=nu_scheme,
        config=config,
        strict_exact=strict_exact,
        nu_params=nu_params,
    )
    probs, log_norm = log_hyperposterior(domain, config, grid)
    return Hyperposterior(
        domain=domain.name, config=config, grid=grid, probs=probs, log_evidence=log_norm
    )
