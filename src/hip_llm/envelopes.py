"""Pointwise CDF envelopes and quantile envelopes under imprecision.

Orientation matters and is stated once here, then respected everywhere:

    ``lower_CDF(t) = inf_h F_h(t)``  and  ``upper_CDF(t) = sup_h F_h(t)``.

Because a *lower* CDF is stochastically *larger*, the lower CDF curve is the
**optimistic** reliability bound and the upper CDF curve is the **conservative**
one.  Lower/upper CDFs are therefore never described as "lower/upper reliability
values" without this distinction.

Quantiles under imprecision (specification Section 13.7) use **Definition Q1**
throughout this package:

    Q1  For each configuration ``h`` compute ``Q_q(p^(h))``; report
        ``[min_h Q_q, max_h Q_q]``.

Q1 is what the paper itself uses for RQ5 (Section 4.3.5: "we first compute the
median of each posterior ... and then report the envelope of posterior medians",
and the CI is ``[min_h Q_0.05, max_h Q_0.95]``).  The alternative -- inverting
the lower/upper CDF envelopes -- is provided as
:func:`quantiles_from_cdf_envelope` for comparison only and is never mixed into
a figure or table that uses Q1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .numerics import assert_finite, empirical_cdf
from .schemas import CDFEnvelope

__all__ = [
    "default_t_grid",
    "cdf_family",
    "cdf_envelope",
    "quantile_envelope",
    "quantiles_from_cdf_envelope",
    "EnvelopeSummary",
    "summarise_envelope",
]


def default_t_grid(T: int = 201) -> np.ndarray:
    """The paper's fixed CDF evaluation grid: ``linspace(0, 1, T)`` with ``T = 201``."""
    if T < 2:
        raise ValueError("T must be at least 2")
    return np.linspace(0.0, 1.0, T)


def cdf_family(samples: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Empirical CDF of every configuration.

    Parameters
    ----------
    samples:
        ``(K, S)`` array; row ``k`` holds the ``S`` posterior draws under
        configuration ``h_k``.
    t_grid:
        ``(T,)`` evaluation grid.

    Returns
    -------
    np.ndarray
        ``(K, T)`` array with ``F[k, t] = mean(samples[k] <= t_grid[t])``.
    """
    s = np.atleast_2d(np.asarray(samples, dtype=float))
    t = np.asarray(t_grid, dtype=float)
    out = np.empty((s.shape[0], t.size), dtype=float)
    for k in range(s.shape[0]):
        out[k] = empirical_cdf(s[k], t)
    return assert_finite(out, "cdf family")


def cdf_envelope(
    samples: np.ndarray,
    t_grid: np.ndarray | None = None,
    quantity: str = "unnamed",
    meta: dict | None = None,
) -> CDFEnvelope:
    """Pointwise min/max CDF envelope across configurations (paper Theorems 1--3)."""
    t = default_t_grid() if t_grid is None else np.asarray(t_grid, dtype=float)
    F = cdf_family(samples, t)
    return CDFEnvelope(
        quantity=quantity,
        t_grid=t,
        lower=F.min(axis=0),
        upper=F.max(axis=0),
        n_configs=F.shape[0],
        meta=meta or {},
    )


def quantile_envelope(samples: np.ndarray, q: float) -> tuple[float, float]:
    """Definition Q1: ``[min_h Q_q(p^(h)), max_h Q_q(p^(h))]``.

    This is the paper's RQ5 definition of the median envelope and of the
    reported 90% interval bounds.
    """
    s = np.atleast_2d(np.asarray(samples, dtype=float))
    per_config = np.quantile(s, q, axis=1)
    assert_finite(per_config, f"per-configuration quantile q={q}")
    return float(per_config.min()), float(per_config.max())


def quantiles_from_cdf_envelope(envelope: CDFEnvelope, q: float) -> tuple[float, float]:
    """Definition Q2 (comparison only): invert the lower/upper CDF envelopes.

    Because ``lower_CDF <= upper_CDF`` pointwise, inverting them swaps the
    order: ``Q_q(upper_CDF) <= Q_q(lower_CDF)``.  The returned pair is ordered
    ``(low, high)``.  Never mixed with :func:`quantile_envelope` in the same
    figure or table.
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie strictly inside (0, 1)")
    t = envelope.t_grid

    def _invert(F: np.ndarray) -> float:
        idx = int(np.searchsorted(F, q, side="left"))
        idx = min(max(idx, 0), t.size - 1)
        return float(t[idx])

    a, b = _invert(envelope.upper), _invert(envelope.lower)
    return (min(a, b), max(a, b))


@dataclass(frozen=True)
class EnvelopeSummary:
    """Scalar summaries used to quantify RQ2/RQ3/RQ6 shifts."""

    quantity: str
    median_lower: float
    median_upper: float
    q05_lower: float
    q05_upper: float
    q95_lower: float
    q95_upper: float
    envelope_area: float
    max_cdf_separation: float
    n_configs: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "quantity": self.quantity,
            "median_lower": self.median_lower,
            "median_upper": self.median_upper,
            "q05_lower": self.q05_lower,
            "q05_upper": self.q05_upper,
            "q95_lower": self.q95_lower,
            "q95_upper": self.q95_upper,
            "envelope_area": self.envelope_area,
            "max_cdf_separation": self.max_cdf_separation,
            "n_configs": self.n_configs,
        }


def summarise_envelope(
    samples: np.ndarray,
    quantity: str,
    t_grid: np.ndarray | None = None,
) -> EnvelopeSummary:
    """Median / 5% / 95% envelopes (Q1), envelope area and max CDF separation."""
    env = cdf_envelope(samples, t_grid=t_grid, quantity=quantity)
    m_lo, m_hi = quantile_envelope(samples, 0.50)
    l_lo, l_hi = quantile_envelope(samples, 0.05)
    u_lo, u_hi = quantile_envelope(samples, 0.95)
    return EnvelopeSummary(
        quantity=quantity,
        median_lower=m_lo,
        median_upper=m_hi,
        q05_lower=l_lo,
        q05_upper=l_hi,
        q95_lower=u_lo,
        q95_upper=u_hi,
        envelope_area=env.area,
        max_cdf_separation=env.max_separation,
        n_configs=env.n_configs,
    )
