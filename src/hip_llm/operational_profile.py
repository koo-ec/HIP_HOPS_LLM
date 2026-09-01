"""Operational profiles: validation, aggregation and perturbation.

Paper Definition 2 and Section 3.2.1.  The hierarchy is

    subdomain -> domain :  p_i = sum_j Omega_ij theta_ij      (sum_j Omega_ij = 1)
    domain    -> LLM    :  p_L = sum_i W_i p_i                (sum_i W_i     = 1)

Paper footnote 21 (RQ5) notes the algebraically identical flat form
``OP_ij = W_i * Omega_ij`` with ``p_L = sum_ij OP_ij theta_ij``; both directions
of that equivalence are implemented here and covered by tests.
"""

from __future__ import annotations

import numpy as np

from .schemas import WEIGHT_SUM_TOL, OperationalProfile

__all__ = [
    "validate_weights",
    "aggregate",
    "flatten_hierarchical_op",
    "unflatten_op",
    "perturb_and_renormalise",
    "dataset_proportional_op",
]


def validate_weights(weights: np.ndarray, name: str = "OP") -> np.ndarray:
    """Reject anything that is not a probability vector."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError(f"{name}: expected a non-empty 1-D vector, got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name}: contains non-finite values")
    if np.any(w < 0.0):
        raise ValueError(f"{name}: negative weight(s) {w[w < 0]}")
    total = w.sum()
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError(f"{name}: weights sum to {total:.17g}, expected 1")
    return w


def aggregate(values: np.ndarray, weights: np.ndarray, axis: int = -1) -> np.ndarray:
    """Weighted aggregation :math:`\\sum_j w_j x_j` with validation and range check."""
    w = validate_weights(weights, "aggregation weights")
    x = np.asarray(values, dtype=float)
    if x.shape[axis] != w.size:
        raise ValueError(f"cannot aggregate axis of length {x.shape[axis]} with {w.size} weights")
    out = np.tensordot(x, w, axes=([axis], [0]))
    if np.any(out < -1e-12) or np.any(out > 1.0 + 1e-12):
        raise ValueError("weighted aggregate escaped [0, 1]")
    return np.clip(out, 0.0, 1.0)


def flatten_hierarchical_op(
    W: np.ndarray, omegas: list[np.ndarray]
) -> np.ndarray:
    """``OP_ij = W_i * Omega_ij`` (paper footnote 21)."""
    W = validate_weights(W, "W")
    if len(omegas) != W.size:
        raise ValueError("one Omega vector is required per domain")
    parts = [W[i] * validate_weights(np.asarray(om, dtype=float), f"Omega[{i}]") for i, om in enumerate(omegas)]
    return np.concatenate(parts)


def unflatten_op(op: np.ndarray, sizes: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Recover ``(W, [Omega_i])`` from a flat subdomain-level OP.

    ``W_i = sum_j OP_ij`` and ``Omega_ij = OP_ij / W_i`` (paper footnote 21).
    """
    op = validate_weights(op, "flat OP")
    if sum(sizes) != op.size:
        raise ValueError(f"sizes {sizes} do not partition an OP of length {op.size}")
    W = np.empty(len(sizes), dtype=float)
    omegas: list[np.ndarray] = []
    start = 0
    for i, n in enumerate(sizes):
        block = op[start : start + n]
        total = block.sum()
        if total <= 0:
            raise ValueError(f"domain {i} carries zero operational weight; Omega is undefined")
        W[i] = total
        omegas.append(block / total)
        start += n
    return W, omegas


def perturb_and_renormalise(
    op: np.ndarray, magnitude: float, rng: np.random.Generator
) -> np.ndarray:
    """Multiplicative :math:`\\pm` ``magnitude`` perturbation, then renormalise.

    Implements the paper's ``OP_approx`` scenario (Section 4.3.5): "The ground
    truth OP is perturbed by ±20% noise and renormalized", i.e.
    ``OP'_k ∝ OP_k * U_k`` with ``U_k ~ Uniform(1-magnitude, 1+magnitude)``.

    .. warning::
       The paper states ``magnitude = 0.20`` for RQ5, whereas the official
       repository's ``numerics/settings.yaml`` carries ``sampling.PERTURBATION:
       0.07`` among settings declared to apply to *all figures*.  The two values
       cannot be reconciled from any official source and the repository gives no
       code that consumes ``PERTURBATION``.  This function therefore takes the
       magnitude as an explicit argument and never picks one silently; see
       ``configs/synthetic_rq5.yaml``.
    """
    if not (0.0 <= magnitude < 1.0):
        raise ValueError("perturbation magnitude must lie in [0, 1)")
    base = validate_weights(op, "OP to perturb")
    factors = rng.uniform(1.0 - magnitude, 1.0 + magnitude, size=base.size)
    out = base * factors
    total = out.sum()
    if total <= 0:
        raise FloatingPointError("perturbed OP has non-positive mass")
    return out / total


def dataset_proportional_op(sizes: np.ndarray, level: str, labels: tuple[str, ...]) -> OperationalProfile:
    """OP weights proportional to dataset sizes (paper Section 4.2, Remark 7)."""
    s = np.asarray(sizes, dtype=float)
    if np.any(s < 0) or s.sum() <= 0:
        raise ValueError("dataset sizes must be non-negative with positive total")
    return OperationalProfile(level=level, labels=labels, weights=s / s.sum())
