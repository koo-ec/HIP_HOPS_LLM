"""RQ5 baselines: BB-UnInf, BB-Inf, HiBayES, and the RQ5 blocking checklist.

Paper Section 4.3.5 compares HIP-LLM against

* **BB-UnInf** -- independent Beta--Binomial per subdomain with ``Beta(1,1)``;
* **BB-Inf**   -- independent Beta--Binomial with informative priors *"centered
  on the GT reliabilities"* (the prior **strength** is never stated);
* **HiBayES** [31] -- *"a hierarchical Bayesian model with partial pooling across
  subdomains"* (no prior specification, no link function, no code reference).

REPRODUCIBILITY GAP
-------------------
Exact numerical reproduction of Tables 4 and 5 is **blocked**.  The paper
publishes only the aggregate ``p_L^GT = 0.5860``; the underlying ground-truth
subdomain reliability vector ``theta^GT`` and ground-truth OP vector ``OP^GT``
are never given, and ``sum_ij OP_ij theta_ij = 0.5860`` is one equation in seven
unknowns (4 reliabilities + 4 weights - 1 simplex constraint).  Infinitely many
parameter sets reproduce the aggregate, so inferring one would be fabrication.
The official repository contains no code and no synthetic-experiment data.

:func:`rq5_blocking_checklist` enumerates every missing quantity.  The estimators
below are fully implemented and unit-tested, so that RQ5 runs the moment the
missing values are supplied in ``configs/synthetic_rq5.yaml`` -- but every
estimator that needs an unstated setting takes it as a **required argument with
no default**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import norm

from .numerics import assert_finite, log_binom_coeff
from .operational_profile import validate_weights

__all__ = [
    "BaselineResult",
    "bb_uninformative",
    "bb_informative",
    "hibayes_partial_pooling",
    "generate_synthetic_counts",
    "rq5_blocking_checklist",
    "RQ5BlockedError",
]


class RQ5BlockedError(RuntimeError):
    """Raised when RQ5 is requested without the missing ground-truth settings."""


@dataclass(frozen=True)
class BaselineResult:
    """Posterior draws of :math:`p_L` from a single-posterior baseline."""

    method: str
    p_L: np.ndarray
    theta: np.ndarray
    meta: dict

    @property
    def median(self) -> float:
        return float(np.median(self.p_L))

    def error(self, p_gt: float) -> float:
        return float(abs(self.median - p_gt))

    def credible_interval(self, level: float = 0.90) -> tuple[float, float]:
        lo = (1.0 - level) / 2.0
        return (
            float(np.quantile(self.p_L, lo)),
            float(np.quantile(self.p_L, 1.0 - lo)),
        )


# --------------------------------------------------------------------------- #
# independent Beta-Binomial baselines
# --------------------------------------------------------------------------- #
def _bb_draw(
    counts: np.ndarray,
    trials: np.ndarray,
    alpha0: np.ndarray,
    beta0: np.ndarray,
    op: np.ndarray,
    S: int,
    rng: np.random.Generator,
    method: str,
    meta: dict,
) -> BaselineResult:
    alpha = np.asarray(alpha0, dtype=float) + np.asarray(counts, dtype=float)
    beta = np.asarray(beta0, dtype=float) + (np.asarray(trials, dtype=float) - np.asarray(counts, dtype=float))
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("Beta posterior parameters must be positive")
    theta = rng.beta(alpha[None, :], beta[None, :], size=(S, alpha.size))
    p_L = theta @ validate_weights(op, f"{method} OP")
    return BaselineResult(
        method=method,
        p_L=assert_finite(p_L, f"{method} p_L"),
        theta=theta,
        meta={**meta, "alpha_post": alpha.tolist(), "beta_post": beta.tolist()},
    )


def bb_uninformative(
    counts: np.ndarray,
    trials: np.ndarray,
    op: np.ndarray,
    S: int,
    rng: np.random.Generator,
) -> BaselineResult:
    """Independent Beta--Binomial with ``Beta(1,1)`` priors (paper: BB-UnInf).

    Subdomains are modelled independently -- there is no pooling -- so
    ``theta_j | C_j ~ Beta(1 + C_j, 1 + N_j - C_j)`` exactly.
    """
    k = np.asarray(counts).size
    return _bb_draw(
        counts, trials, np.ones(k), np.ones(k), op, S, rng, "BB-UnInf", {"prior": "Beta(1,1)"}
    )


def bb_informative(
    counts: np.ndarray,
    trials: np.ndarray,
    op: np.ndarray,
    theta_gt: np.ndarray,
    prior_strength: float,
    S: int,
    rng: np.random.Generator,
) -> BaselineResult:
    """Independent Beta--Binomial with priors centred on the GT reliabilities.

    ``alpha0_j = s * theta_gt_j``, ``beta0_j = s * (1 - theta_gt_j)``, so the
    prior mean is exactly ``theta_gt_j`` and ``s`` is the equivalent prior sample
    size.

    ``prior_strength`` (``s``) is a **required argument**: the paper says only
    "informative priors centered on the GT reliabilities" and never states the
    strength, which materially changes Table 4's Small-N numbers.
    """
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    gt = np.asarray(theta_gt, dtype=float)
    if np.any(gt <= 0) or np.any(gt >= 1):
        raise ValueError("theta_gt must lie strictly inside (0, 1)")
    return _bb_draw(
        counts,
        trials,
        prior_strength * gt,
        prior_strength * (1.0 - gt),
        op,
        S,
        rng,
        "BB-Inf",
        {"prior": f"Beta(s*theta_gt, s*(1-theta_gt)), s={prior_strength}", "prior_strength": prior_strength},
    )


# --------------------------------------------------------------------------- #
# HiBayES-style hierarchical partial pooling
# --------------------------------------------------------------------------- #
def _gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for ``int f(u) N(u|0,1) du ~ sum_k w_k f(x_k)``."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / np.sqrt(2.0 * np.pi)


def hibayes_partial_pooling(
    counts: np.ndarray,
    trials: np.ndarray,
    op: np.ndarray,
    S: int,
    rng: np.random.Generator,
    *,
    alpha_prior_sd: float,
    sigma_prior_scale: float,
    n_alpha_grid: int = 121,
    n_sigma_grid: int = 81,
    n_quad: int = 41,
) -> BaselineResult:
    """Hierarchical Binomial GLM with partial pooling (HiBayES family, ref. [31]).

    Model
    -----
    .. math::

        C_j \\sim \\mathrm{Binomial}(N_j, \\mathrm{logit}^{-1}(\\alpha + u_j)),\\quad
        u_j \\sim \\mathcal{N}(0, \\sigma^2),\\quad
        \\alpha \\sim \\mathcal{N}(0, \\sigma_\\alpha^2),\\quad
        \\sigma \\sim \\mathrm{Half}\\mathcal{N}(\\tau)

    This is the multilevel Binomial GLM with a random subdomain intercept that
    HiBayES describes; partial pooling is produced by the shared :math:`\\sigma`.
    Inference is exact-to-quadrature rather than MCMC: a 2-D grid over
    :math:`(\\alpha,\\sigma)` with Gauss--Hermite marginalisation of each
    :math:`u_j`, then conditional inverse-CDF sampling of :math:`u_j`.  This
    makes the baseline deterministic given the seed and free of convergence
    diagnostics.

    ``alpha_prior_sd`` (:math:`\\sigma_\\alpha`) and ``sigma_prior_scale``
    (:math:`\\tau`) are **required keyword arguments**.  The HIP-LLM paper cites
    HiBayES but states no prior specification, and the official repository
    contains no code, so no default would be defensible.
    """
    if alpha_prior_sd <= 0 or sigma_prior_scale <= 0:
        raise ValueError("alpha_prior_sd and sigma_prior_scale must be positive")

    C = np.asarray(counts, dtype=float)
    N = np.asarray(trials, dtype=float)
    if C.shape != N.shape or C.ndim != 1:
        raise ValueError("counts and trials must be 1-D arrays of equal length")
    J = C.size

    alpha_grid = np.linspace(-4.0 * alpha_prior_sd, 4.0 * alpha_prior_sd, n_alpha_grid)
    sigma_grid = np.linspace(1e-3, 4.0 * sigma_prior_scale, n_sigma_grid)
    gh_x, gh_w = _gauss_hermite(n_quad)
    log_gh_w = np.log(gh_w)
    log_binom = log_binom_coeff(N, C)

    # log p(C_j | alpha, sigma) marginalised over u_j, for every (alpha, sigma, j)
    A = alpha_grid[:, None, None, None]                 # (A,1,1,1)
    Sg = sigma_grid[None, :, None, None]                # (1,Sg,1,1)
    U = gh_x[None, None, :, None]                       # (1,1,Q,1)
    eta = A + Sg * U                                    # (A,Sg,Q,1)
    logp = -np.logaddexp(0.0, -eta)                     # log sigmoid
    log1mp = -np.logaddexp(0.0, eta)                    # log(1 - sigmoid)
    ll = C[None, None, None, :] * logp + (N - C)[None, None, None, :] * log1mp
    log_marg_j = logsumexp(ll + log_gh_w[None, None, :, None], axis=2)   # (A,Sg,J)
    log_lik = log_marg_j.sum(axis=2) + log_binom.sum()                   # (A,Sg)

    log_prior = (
        norm.logpdf(alpha_grid, 0.0, alpha_prior_sd)[:, None]
        + (norm.logpdf(sigma_grid, 0.0, sigma_prior_scale) + np.log(2.0))[None, :]
    )
    d_alpha = float(alpha_grid[1] - alpha_grid[0])
    d_sigma = float(sigma_grid[1] - sigma_grid[0])
    log_w = assert_finite(log_lik + log_prior + np.log(d_alpha * d_sigma), "HiBayES log weights")
    flat = log_w.ravel()
    probs = np.exp(flat - logsumexp(flat))
    probs = probs / probs.sum()

    idx = rng.choice(flat.size, size=S, replace=True, p=probs)
    ai, si = np.unravel_index(idx, log_w.shape)
    alpha_s = alpha_grid[ai]
    sigma_s = sigma_grid[si]

    # Conditional posterior of u_j given (alpha, sigma, C_j): 1-D inverse-CDF
    # sampling on a fine standardised grid, vectorised over draws and subdomains.
    zgrid = np.linspace(-6.0, 6.0, 241)
    eta_s = alpha_s[:, None, None] + sigma_s[:, None, None] * zgrid[None, :, None]   # (S,Z,J)
    lp = -np.logaddexp(0.0, -eta_s)
    l1mp = -np.logaddexp(0.0, eta_s)
    logf = C[None, None, :] * lp + (N - C)[None, None, :] * l1mp
    logf = logf + norm.logpdf(zgrid)[None, :, None]
    logf -= logsumexp(logf, axis=1, keepdims=True)
    w = np.exp(logf)
    cdf = np.cumsum(w, axis=1)
    cdf /= cdf[:, -1:, :]
    u = rng.uniform(size=(S, 1, J))
    pick = (cdf < u).sum(axis=1)
    pick = np.clip(pick, 0, zgrid.size - 1)
    z_draw = zgrid[pick]                                                            # (S,J)

    theta = expit(alpha_s[:, None] + sigma_s[:, None] * z_draw)
    theta = np.clip(theta, 0.0, 1.0)
    p_L = theta @ validate_weights(op, "HiBayES OP")

    return BaselineResult(
        method="HiBayES",
        p_L=assert_finite(p_L, "HiBayES p_L"),
        theta=theta,
        meta={
            "model": "hierarchical Binomial GLM, random subdomain intercept",
            "alpha_prior_sd": alpha_prior_sd,
            "sigma_prior_scale": sigma_prior_scale,
            "inference": "2-D grid + Gauss-Hermite marginalisation",
            "reconstruction": True,
            "reconstruction_note": (
                "HiBayES prior specification is not stated in the HIP-LLM paper and "
                "no code exists in the official repository; the priors used here are "
                "supplied by the caller and are NOT recovered from an official source."
            ),
        },
    )


# --------------------------------------------------------------------------- #
# synthetic data generation for RQ5
# --------------------------------------------------------------------------- #
def generate_synthetic_counts(
    theta_gt: np.ndarray, sample_sizes: Sequence[int], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``C_j ~ Binomial(N_j, theta_gt_j)`` -- the paper's synthetic RQ5 data.

    This is one of the two places in the whole package where synthetic draws are
    legitimate, because the paper itself defines RQ5 as a synthetic simulation
    experiment with a known ground truth.
    """
    gt = np.asarray(theta_gt, dtype=float)
    N = np.asarray(sample_sizes, dtype=np.int64)
    if gt.shape != N.shape:
        raise ValueError("theta_gt and sample_sizes must have the same shape")
    if np.any(gt < 0) or np.any(gt > 1):
        raise ValueError("theta_gt must lie in [0, 1]")
    if np.any(N <= 0):
        raise ValueError("sample sizes must be positive")
    return rng.binomial(N, gt).astype(float), N.astype(float)


def rq5_blocking_checklist(cfg: dict) -> list[dict[str, object]]:
    """Enumerate every RQ5 input the paper leaves unspecified.

    Returns one row per required quantity with ``resolved`` set according to
    whether ``cfg`` supplies it.  The notebook prints this table instead of
    fabricating results whenever anything is unresolved.
    """
    gt = cfg.get("ground_truth", {}) or {}
    baselines = cfg.get("baselines", {}) or {}
    items = [
        {
            "quantity": "ground-truth subdomain reliability vector theta^GT",
            "paper_reference": "Section 4.3.5 (only the aggregate p_L^GT = 0.5860 is printed)",
            "supplied_key": "ground_truth.theta_gt",
            "resolved": gt.get("theta_gt") is not None,
            "why_it_matters": "Determines every synthetic count; not identified by the aggregate.",
        },
        {
            "quantity": "ground-truth operational profile OP^GT",
            "paper_reference": "Section 4.3.5, Eq. (9) and footnote 21",
            "supplied_key": "ground_truth.op_gt",
            "resolved": gt.get("op_gt") is not None,
            "why_it_matters": "Sets p_L^GT jointly with theta^GT; underdetermined by 0.5860 alone.",
        },
        {
            "quantity": "dataset-based operational profile OP^data",
            "paper_reference": "Section 4.3.5 ('proportional to dataset sizes' -- which sizes unstated)",
            "supplied_key": "operational_profiles.op_data",
            "resolved": (cfg.get("operational_profiles", {}) or {}).get("op_data") is not None,
            "why_it_matters": "Defines the OP-mismatch scenario rows of Tables 4 and 5.",
        },
        {
            "quantity": "BB-Inf informative prior strength s",
            "paper_reference": "Section 4.3.5 ('informative priors centered on the GT reliabilities')",
            "supplied_key": "baselines.bb_inf.prior_strength",
            "resolved": (baselines.get("bb_inf", {}) or {}).get("prior_strength") is not None,
            "why_it_matters": "Controls how far BB-Inf is pulled toward the GT in the Small-N regime.",
        },
        {
            "quantity": "HiBayES prior specification",
            "paper_reference": "Section 4.3.5, reference [31]; no priors, link or code given",
            "supplied_key": "baselines.hibayes.{alpha_prior_sd, sigma_prior_scale}",
            "resolved": all(
                (baselines.get("hibayes", {}) or {}).get(k) is not None
                for k in ("alpha_prior_sd", "sigma_prior_scale")
            ),
            "why_it_matters": "Determines the amount of pooling and hence the HiBayES rows.",
        },
        {
            "quantity": "OP perturbation magnitude for OP^approx",
            "paper_reference": (
                "Section 4.3.5 states +/-20%; repository settings.yaml states "
                "sampling.PERTURBATION = 0.07 for 'all figures' -- unreconciled conflict"
            ),
            "supplied_key": "operational_profiles.perturbation_magnitude",
            "resolved": (cfg.get("operational_profiles", {}) or {}).get("perturbation_magnitude")
            is not None,
            "why_it_matters": "0.20 vs 0.07 changes the OP^approx rows of both tables.",
        },
        {
            "quantity": "random seeds for the synthetic draws",
            "paper_reference": "not stated for RQ5 (settings.yaml seeds cover figures only)",
            "supplied_key": "seeds.synthetic",
            "resolved": (cfg.get("seeds", {}) or {}).get("synthetic") is not None,
            "why_it_matters": "Table 4/5 are single-realisation numbers; a different draw moves them.",
        },
    ]
    return items
