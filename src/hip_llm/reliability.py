"""Future reliability :math:`R(n_F)=p^{\\,n_F}` (paper Theorems 4--6, Appendix A.5).

Two distinct objects are computed and never conflated:

* the **posterior CDF of the reliability random variable**
  :math:`F_{R(n_F)}(t)=F_p(t^{1/n_F})` (Theorem 4's exact transform), and
* the **posterior expected reliability**
  :math:`\\mathbb{E}[R(n_F)]=\\mathbb{E}[p^{\\,n_F}]`, which is what paper Fig. 8a
  plots.

The expectation is computed per configuration as ``mean(p_h ** n_F)`` and then
enveloped.  It is *never* approximated by ``mean(p) ** n_F``, since by Jensen's
inequality :math:`\\mathbb{E}[p^{n}]\\ge\\mathbb{E}[p]^{n}` for :math:`n\\ge 1`
with equality only for a degenerate posterior.
"""

from __future__ import annotations

import numpy as np

from .envelopes import cdf_family, default_t_grid
from .numerics import assert_finite
from .schemas import CDFEnvelope, ReliabilityEnvelope

__all__ = [
    "reliability_from_p",
    "expected_reliability_per_config",
    "expected_reliability_envelope",
    "reliability_cdf_envelope",
    "transformed_cdf",
    "default_horizons",
]


def default_horizons(n_max: int = 60) -> np.ndarray:
    """Horizon grid matching paper Fig. 8 (log x-axis from 1 to 60)."""
    base = np.unique(
        np.concatenate(
            [
                np.arange(1, 11),
                np.array([12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60]),
            ]
        )
    )
    return base[base <= n_max].astype(float)


def reliability_from_p(p: np.ndarray, n_F: float) -> np.ndarray:
    """:math:`R(n_F)=p^{\\,n_F}` applied elementwise (paper Appendix A.5.1)."""
    if n_F < 1:
        raise ValueError("n_F must be at least 1")
    p = np.asarray(p, dtype=float)
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p must lie in [0, 1]")
    return assert_finite(np.power(p, float(n_F)), "R(n_F)")


def expected_reliability_per_config(samples: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    """``(K, H)`` array of :math:`\\hat{\\mathbb{E}}[R(n_F)\\mid h]`.

    Monte-Carlo estimator from paper Appendix B:
    :math:`\\hat{\\mathbb{E}}[R_L(n_F)] = \\frac{1}{S}\\sum_s (p_L^{(s)})^{n_F}`.

    Evaluated one horizon at a time.  The obvious vectorisation
    ``np.power(s[:, :, None], h[None, None, :]).mean(axis=1)`` materialises a
    ``(K, S, H)`` temporary -- 258 MB at the paper's baseline with 21 horizons --
    which would dominate peak memory and mask the model's true memory scaling in
    RQ8.  The arithmetic is identical; only the temporary is avoided.
    """
    s = np.atleast_2d(np.asarray(samples, dtype=float))
    h = np.asarray(horizons, dtype=float)
    if np.any(h < 1):
        raise ValueError("horizons must be >= 1")
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("samples must lie in [0, 1]")

    out = np.empty((s.shape[0], h.size), dtype=float)
    for j, n in enumerate(h):
        np.mean(np.power(s, n), axis=1, out=out[:, j])
    return assert_finite(out, "expected reliability per configuration")


def expected_reliability_envelope(
    samples: np.ndarray,
    horizons: np.ndarray | None = None,
    quantity: str = "E[R_L(n_F)]",
    meta: dict | None = None,
) -> ReliabilityEnvelope:
    """Lower/upper envelopes of :math:`\\mathbb{E}[R(n_F)]` across configurations."""
    h = default_horizons() if horizons is None else np.asarray(horizons, dtype=float)
    per_config = expected_reliability_per_config(samples, h)
    return ReliabilityEnvelope(
        quantity=quantity,
        horizons=h,
        lower=per_config.min(axis=0),
        upper=per_config.max(axis=0),
        n_configs=per_config.shape[0],
        meta=meta or {},
    )


def transformed_cdf(cdf_p: np.ndarray, t_grid: np.ndarray, n_F: float) -> np.ndarray:
    """Exact CDF transform of Theorem 4: :math:`F_{p^{n}}(t)=F_p(t^{1/n})`.

    ``cdf_p`` must be the CDF of :math:`p` evaluated on ``t_grid``; the result is
    the CDF of :math:`p^{n_F}` on the *same* grid, obtained by interpolating
    ``cdf_p`` at :math:`t^{1/n_F}`.
    """
    if n_F < 1:
        raise ValueError("n_F must be at least 1")
    t = np.asarray(t_grid, dtype=float)
    F = np.asarray(cdf_p, dtype=float)
    if F.shape[-1] != t.size:
        raise ValueError("cdf_p's last axis must match t_grid")
    t_root = np.power(np.clip(t, 0.0, 1.0), 1.0 / float(n_F))
    if F.ndim == 1:
        return np.interp(t_root, t, F)
    return np.vstack([np.interp(t_root, t, row) for row in F])


def reliability_cdf_envelope(
    samples: np.ndarray,
    n_F: float,
    t_grid: np.ndarray | None = None,
    quantity: str | None = None,
    meta: dict | None = None,
) -> CDFEnvelope:
    """CDF envelope of :math:`R(n_F)=p^{\\,n_F}` via the exact Theorem-4 transform."""
    t = default_t_grid() if t_grid is None else np.asarray(t_grid, dtype=float)
    F_p = cdf_family(samples, t)
    F_R = transformed_cdf(F_p, t, n_F)
    # Interpolation of a monotone step function stays monotone, but clip to be safe.
    F_R = np.clip(F_R, 0.0, 1.0)
    F_R = np.maximum.accumulate(F_R, axis=1)
    return CDFEnvelope(
        quantity=quantity or f"R(n_F={int(n_F)})",
        t_grid=t,
        lower=F_R.min(axis=0),
        upper=F_R.max(axis=0),
        n_configs=F_R.shape[0],
        meta={**(meta or {}), "n_F": float(n_F), "method": "theorem4_transform"},
    )
