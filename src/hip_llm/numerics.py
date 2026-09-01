"""Numerically stable primitives shared by the HIP-LLM inference code.

Everything that can underflow is computed in log space with
``scipy.special.betaln`` / ``gammaln`` / ``xlogy`` / ``logsumexp``, and every
public entry point asserts finiteness of its result so that a silent ``nan``
cannot propagate into a plotted envelope.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.special import betaln, gammaln, logsumexp, xlog1py, xlogy

__all__ = [
    "assert_finite",
    "log_beta_pdf",
    "log_gamma_pdf",
    "log_beta_binomial_pmf",
    "log_beta_binomial_evidence",
    "normalise_log_weights",
    "spawn_generators",
    "child_generator",
    "empirical_cdf",
    "empirical_quantile",
    "log_binom_coeff",
]


def assert_finite(arr: np.ndarray, name: str) -> np.ndarray:
    """Raise if ``arr`` contains ``nan`` or ``inf``; return it unchanged."""
    a = np.asarray(arr)
    if not np.all(np.isfinite(a)):
        bad = int(np.count_nonzero(~np.isfinite(a)))
        raise FloatingPointError(
            f"{name} contains {bad} non-finite value(s) out of {a.size}"
        )
    return a


def log_binom_coeff(n: np.ndarray | float, k: np.ndarray | float) -> np.ndarray:
    """:math:`\\log\\binom{n}{k}` via ``gammaln``."""
    n_arr = np.asarray(n, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    return gammaln(n_arr + 1.0) - gammaln(k_arr + 1.0) - gammaln(n_arr - k_arr + 1.0)


def log_beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """:math:`\\log \\mathrm{Beta}(x \\mid a, b)` (paper Eq. A.4, :math:`\\mu_i` prior)."""
    x = np.asarray(x, dtype=float)
    if a <= 0 or b <= 0:
        raise ValueError(f"Beta shape parameters must be positive, got a={a}, b={b}")
    out = xlogy(a - 1.0, x) + xlog1py(b - 1.0, -x) - betaln(a, b)
    return assert_finite(out, "log_beta_pdf")


def log_gamma_pdf(x: np.ndarray, c: float, rate: float) -> np.ndarray:
    """:math:`\\log \\mathrm{Gamma}(x \\mid c, \\mathrm{rate}=d)`.

    The paper is explicit (footnote 16) that the **shape--rate** parameterisation
    is used, with :math:`\\mathbb{E}[\\nu_i]=c_i/d_i` and
    :math:`\\mathrm{Var}[\\nu_i]=c_i/d_i^2`.  Passing ``rate`` where a library
    expects ``scale`` would silently change the prior, so this function takes
    ``rate`` only.
    """
    x = np.asarray(x, dtype=float)
    if c <= 0 or rate <= 0:
        raise ValueError(f"Gamma shape/rate must be positive, got c={c}, rate={rate}")
    out = c * np.log(rate) - gammaln(c) + xlogy(c - 1.0, x) - rate * x
    return assert_finite(out, "log_gamma_pdf")


def log_beta_binomial_pmf(
    k: np.ndarray, n: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Full log Beta-Binomial pmf **including** the binomial coefficient.

    :math:`\\log\\binom{n}{k} + \\log B(k+\\alpha, n-k+\\beta) - \\log B(\\alpha,\\beta)`
    (paper Eq. A.10).
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    out = (
        log_binom_coeff(n, k)
        + betaln(k + alpha, n - k + beta)
        - betaln(alpha, beta)
    )
    return assert_finite(out, "log_beta_binomial_pmf")


def log_beta_binomial_evidence(
    counts: np.ndarray,
    trials: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    include_binomial_coeff: bool = False,
) -> np.ndarray:
    """Log marginal likelihood :math:`\\log \\Pr(C_i \\mid \\mu_i, \\nu_i)`.

    Implements paper Eq. (A.10) marginalised over :math:`\\boldsymbol\\theta_i`:

    .. math::

        \\log \\Pr(C_i\\mid\\mu_i,\\nu_i)
        = \\sum_j \\Big[\\log B\\big(C_{ij}+\\mu_i\\nu_i,\\;
                                   N_{ij}-C_{ij}+(1-\\mu_i)\\nu_i\\big)
                      - \\log B\\big(\\mu_i\\nu_i,\\;(1-\\mu_i)\\nu_i\\big)\\Big]
          \\;+\\; \\underbrace{\\textstyle\\sum_j \\log\\binom{N_{ij}}{C_{ij}}}_{\\text{constant in }(\\mu,\\nu)}

    The binomial coefficients do not depend on :math:`(\\mu_i,\\nu_i)`, so they
    cancel exactly when the hyperposterior is normalised over the
    :math:`(\\mu,\\nu)` grid.  They are omitted by default (``include_binomial_
    coeff=False``) purely for speed; set the flag to ``True`` when an absolute
    evidence value is required (e.g. when comparing against ``scipy.integrate``
    in the tests).

    Parameters
    ----------
    counts, trials:
        Shape ``(n_i,)`` vectors :math:`C_{ij}`, :math:`N_{ij}`.
    mu, nu:
        Shape ``(G,)`` grid nodes.

    Returns
    -------
    np.ndarray
        Shape ``(G,)`` log-evidence per grid cell.
    """
    counts = np.asarray(counts, dtype=float).reshape(1, -1)
    trials = np.asarray(trials, dtype=float).reshape(1, -1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    nu = np.asarray(nu, dtype=float).reshape(-1, 1)

    if np.any(mu <= 0.0) or np.any(mu >= 1.0):
        raise ValueError("mu nodes must lie strictly inside (0, 1)")
    if np.any(nu <= 0.0):
        raise ValueError("nu nodes must be strictly positive")
    if np.any(counts < 0) or np.any(counts > trials):
        raise ValueError("counts must satisfy 0 <= C <= N")

    alpha = mu * nu                      # (G, 1) broadcast against (1, n_i)
    beta = (1.0 - mu) * nu

    term = betaln(counts + alpha, trials - counts + beta) - betaln(alpha, beta)
    out = term.sum(axis=1)

    if include_binomial_coeff:
        out = out + float(log_binom_coeff(trials, counts).sum())

    return assert_finite(out, "log_beta_binomial_evidence")


def normalise_log_weights(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalise unnormalised log weights with ``logsumexp``.

    Returns
    -------
    (probs, log_norm)
        ``probs`` sums to exactly 1 (after a final renormalisation guarding
        against float round-off) and ``log_norm`` is the log normalising
        constant, i.e. the log evidence :math:`\\log \\Pr(C_i\\mid h_i)` up to
        the omitted binomial coefficients.
    """
    log_w = assert_finite(np.asarray(log_w, dtype=float), "log_w")
    log_norm = float(logsumexp(log_w))
    if not np.isfinite(log_norm):
        raise FloatingPointError("log normalising constant is not finite")
    probs = np.exp(log_w - log_norm)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError(f"degenerate hyperposterior normalisation (sum={total})")
    probs = probs / total
    if np.any(probs < 0.0):
        raise FloatingPointError("negative hyperposterior probability")
    return probs, log_norm


def spawn_generators(seed: int, n: int, spawn_key: Sequence[int] | None = None) -> list[np.random.Generator]:
    """Create ``n`` independent :class:`numpy.random.Generator` children.

    Uses :class:`numpy.random.SeedSequence` spawning so that streams for
    different domains / configurations / models are provably independent and
    fully reproducible from the recorded root seed.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    ss = np.random.SeedSequence(entropy=seed, spawn_key=tuple(spawn_key or ()))
    return [np.random.default_rng(s) for s in ss.spawn(n)]


def child_generator(seed: int, spawn_key: Sequence[int]) -> np.random.Generator:
    """Deterministic child generator addressed by an integer path.

    ``child_generator(7, (2, 15))`` always yields the same stream, letting each
    (domain, configuration) pair own an independent, addressable RNG without
    threading generator objects through the call stack.
    """
    ss = np.random.SeedSequence(entropy=seed, spawn_key=tuple(int(k) for k in spawn_key))
    return np.random.default_rng(ss)


def empirical_cdf(samples: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """:math:`\\hat F(t)=\\frac{1}{S}\\#\\{s: x_s \\le t\\}` evaluated on ``t_grid``.

    Implemented with ``searchsorted`` on the sorted sample, giving an exact
    ``O(S log S + T log S)`` empirical CDF (no binning error).
    """
    x = np.sort(np.asarray(samples, dtype=float))
    t = np.asarray(t_grid, dtype=float)
    counts = np.searchsorted(x, t, side="right")
    return counts / x.size


def empirical_quantile(samples: np.ndarray, q: float | Iterable[float]) -> np.ndarray:
    """Empirical quantile(s) using the default linear interpolation of ``np.quantile``."""
    return np.quantile(np.asarray(samples, dtype=float), q)
