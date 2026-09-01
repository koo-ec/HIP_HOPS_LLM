"""RQ8: controlled scalability sweeps with real wall-clock and memory measurements.

Reproduces paper Table 6 and Fig. 11 **from measurements taken on the machine
running this notebook**.  The paper's own numbers (Google Colab, Intel Xeon
@ 2.20 GHz, ~13 GB RAM, single process) are carried alongside as *reference
values only* and are never substituted for a measurement.

Peak memory is reported with :mod:`tracemalloc`, which tracks Python-level
allocations.  This matches the magnitude of the paper's Table 6 (7.6 MB at
``S = 500``); process RSS would be an order of magnitude larger because of the
interpreter and NumPy/BLAS working set.  Process RSS is recorded separately when
:mod:`psutil` is available.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .posterior import run_model
from .schemas import (
    DomainData,
    GlobalSettings,
    HyperparameterInterval,
    ModelResult,
    SubdomainData,
)

__all__ = [
    "BASELINE_CONFIG",
    "PAPER_TABLE6_REFERENCE",
    "PAPER_FIG11_REFERENCE_EXPONENTS",
    "SweepPoint",
    "SweepResult",
    "synthetic_model",
    "measure_once",
    "run_sweep",
    "fit_power_law",
    "fit_power_law_with_offset",
    "bootstrap_exponent_ci",
    "baseline_timing_breakdown",
]

#: Paper Section 4.2 / Fig. 11 caption.
BASELINE_CONFIG: dict[str, int] = {
    "m": 2,
    "n_bar": 2,
    "K": 160,
    "S": 3000,
    "G": 2000,
    "T": 201,
    "K_total_cap": 512,
}

#: Paper Table 6, retained as a *reference*, never as a measured output.
PAPER_TABLE6_REFERENCE: list[dict[str, object]] = [
    {"swept_parameter": "Number of domains m", "range": "1 -> 12", "paper_peak_memory_mb": "24.6 -> 146.7"},
    {"swept_parameter": "Subdomains per domain n_bar", "range": "2 -> 40", "paper_peak_memory_mb": "35.8 -> 314.2"},
    {"swept_parameter": "Monte Carlo samples S", "range": "500 -> 6000", "paper_peak_memory_mb": "7.6 -> 69.4"},
    {"swept_parameter": "Hyperparameter configs K", "range": "40 -> 320", "paper_peak_memory_mb": "19.1 -> 57.8"},
    {"swept_parameter": "Grid size G", "range": "1000 -> 4500", "paper_peak_memory_mb": "~35.7 (constant)"},
]

#: Exponents printed in the paper's Fig. 11 power-law fits (reference only).
PAPER_FIG11_REFERENCE_EXPONENTS: dict[str, float] = {
    "G": 1.00,
    "m": 1.00,
    "n_bar": 0.71,
    "K": 1.00,
    "S": 0.01,
}


@dataclass(frozen=True)
class SweepPoint:
    """A single measured point of a scalability sweep."""

    parameter: str
    value: float
    times: tuple[float, ...]
    peak_memory_mb: float
    rss_delta_mb: float | None
    stage_times: Mapping[str, float]

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.times))

    @property
    def ci95(self) -> tuple[float, float]:
        """Normal-approximation 95% CI of the mean over repeats."""
        t = np.asarray(self.times, dtype=float)
        if t.size < 2:
            return (float(t[0]), float(t[0]))
        half = 1.96 * float(t.std(ddof=1)) / np.sqrt(t.size)
        return (self.mean_time - half, self.mean_time + half)


@dataclass
class SweepResult:
    """A complete one-parameter-at-a-time sweep with its fitted power law."""

    parameter: str
    points: list[SweepPoint] = field(default_factory=list)
    time_exponent: float = float("nan")
    time_coefficient: float = float("nan")
    memory_exponent: float = float("nan")
    memory_coefficient: float = float("nan")
    #: Exponent of the offset-corrected model ``t = t0 + c * x**alpha``.  A pure
    #: power-law fit is diluted downward whenever a parameter-independent fixed
    #: cost is a large share of a short runtime, which is exactly the regime a
    #: fast implementation lands in.
    time_exponent_offset_corrected: float = float("nan")
    time_offset: float = float("nan")
    #: Bootstrap 95% CI of the offset-corrected exponent, resampling the repeat
    #: timings at each sweep point.  This is the honest answer to "is this
    #: exponent actually resolved?" -- far better than a hand-set threshold.
    exponent_ci_low: float = float("nan")
    exponent_ci_high: float = float("nan")

    @property
    def x(self) -> np.ndarray:
        return np.array([p.value for p in self.points], dtype=float)

    @property
    def mean_times(self) -> np.ndarray:
        return np.array([p.mean_time for p in self.points], dtype=float)

    @property
    def ci_low(self) -> np.ndarray:
        return np.array([p.ci95[0] for p in self.points], dtype=float)

    @property
    def ci_high(self) -> np.ndarray:
        return np.array([p.ci95[1] for p in self.points], dtype=float)

    @property
    def peak_memory(self) -> np.ndarray:
        return np.array([p.peak_memory_mb for p in self.points], dtype=float)


def synthetic_model(m: int, n_bar: int, seed: int = 20260728, N_per_subdomain: int = 80) -> ModelResult:
    """Build an ``m x n_bar`` hierarchy for timing.

    The counts are irrelevant to the *cost*, which depends only on the shape;
    they are drawn once from a fixed seed so timings are comparable across
    repeats.  This model is used **exclusively** for RQ8 timing and never feeds
    a reported reliability result.
    """
    if m < 1 or n_bar < 1:
        raise ValueError("m and n_bar must be >= 1")
    rng = np.random.default_rng(seed)
    domains = []
    for i in range(m):
        acc = rng.uniform(0.40, 0.95, size=n_bar)
        subs = tuple(
            SubdomainData(
                name=f"S{i + 1}{j + 1}",
                successes=int(round(a * N_per_subdomain)),
                trials=N_per_subdomain,
                source_accuracy=float(a),
            )
            for j, a in enumerate(acc)
        )
        domains.append(DomainData(f"D{i + 1}", subs, np.full(n_bar, 1.0 / n_bar)))
    return ModelResult("synthetic-scalability", tuple(domains), np.full(m, 1.0 / m), "synthetic_timing_only")


def _settings(K: int, S: int, G: int, n_mu: int = 40) -> GlobalSettings:
    """Baseline settings with ``G = n_mu * n_nu`` honoured as closely as possible."""
    n_nu = max(2, int(round(G / n_mu)))
    return GlobalSettings(
        n_mu=n_mu,
        n_nu=n_nu,
        cdf_points_T=BASELINE_CONFIG["T"],
        S=S,
        K_per_domain=K,
        max_llm_configuration_pairs=BASELINE_CONFIG["K_total_cap"],
        seed_global=7,
        seed_configs=123,
        seed_pairs=999,
        config_sampling="uniform_random",
        nu_grid_scheme="log",
        mu_grid_scheme="midpoint",
    )


def measure_once(
    m: int, n_bar: int, K: int, S: int, G: int, repeats: int = 3
) -> tuple[list[float], float, float | None, dict[str, float]]:
    """Time and profile ``repeats`` executions of the full pipeline.

    Returns ``(times, peak_tracemalloc_mb, rss_delta_mb, stage_times)``.  Memory
    is profiled on a single dedicated execution so that ``tracemalloc``'s
    overhead never contaminates the reported wall-clock times.
    """
    model = synthetic_model(m, n_bar)
    intervals = [HyperparameterInterval((1, 12), (1, 12), (1, 25), (1, 25))] * m
    st = _settings(K, S, G)

    times: list[float] = []
    for _ in range(max(1, repeats)):
        gc.collect()
        t0 = time.perf_counter()
        run_model(model, intervals, st, model_index=0, cache=None)
        times.append(time.perf_counter() - t0)

    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        rss_before = proc.memory_info().rss
    except Exception:
        proc = None
        rss_before = None

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    domain_sets, llm_set = run_model(model, intervals, st, model_index=0, cache=None)
    subdomain_and_domain = time.perf_counter() - t0

    t1 = time.perf_counter()
    from .envelopes import cdf_envelope
    from .reliability import expected_reliability_envelope

    cdf_envelope(llm_set.p_L, quantity="p_L")
    for ds in domain_sets:
        cdf_envelope(ds.p, quantity=f"p_{ds.domain}")
    expected_reliability_envelope(llm_set.p_L)
    envelope_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_delta = None
    if proc is not None and rss_before is not None:
        rss_delta = (proc.memory_info().rss - rss_before) / 2**20

    stage_times = {
        "Subdomain posteriors": subdomain_and_domain,
        "Domain/LLM envelopes": envelope_time,
    }
    return times, peak / 2**20, rss_delta, stage_times


def run_sweep(
    parameter: str,
    values: Sequence[float],
    repeats: int = 3,
    baseline: Mapping[str, int] | None = None,
    progress: Callable | None = None,
) -> SweepResult:
    """Sweep one parameter with all others held at the paper's baseline."""
    base = dict(BASELINE_CONFIG if baseline is None else baseline)
    if parameter not in ("m", "n_bar", "K", "S", "G"):
        raise ValueError(f"unknown sweep parameter {parameter!r}")

    result = SweepResult(parameter=parameter)
    iterator = values if progress is None else progress(values, desc=f"sweep {parameter}")
    for v in iterator:
        cfg = dict(base)
        cfg[parameter] = int(v)
        times, peak_mb, rss_mb, stages = measure_once(
            m=int(cfg["m"]), n_bar=int(cfg["n_bar"]), K=int(cfg["K"]),
            S=int(cfg["S"]), G=int(cfg["G"]), repeats=repeats,
        )
        result.points.append(
            SweepPoint(parameter, float(v), tuple(times), peak_mb, rss_mb, stages)
        )

    result.time_exponent, result.time_coefficient = fit_power_law(result.x, result.mean_times)
    result.memory_exponent, result.memory_coefficient = fit_power_law(result.x, result.peak_memory)
    (result.time_exponent_offset_corrected, _, result.time_offset) = fit_power_law_with_offset(
        result.x, result.mean_times
    )
    result.exponent_ci_low, result.exponent_ci_high = bootstrap_exponent_ci(result)
    return result


def bootstrap_exponent_ci(
    result: "SweepResult", n_boot: int = 400, seed: int = 20260728
) -> tuple[float, float]:
    """Bootstrap a 95% CI for the offset-corrected exponent.

    Resamples the repeat timings at each sweep point (with replacement) and
    refits.  Whether an exponent is *resolved* is exactly the question of how
    wide this interval is, which is a measurable quantity rather than a
    hand-picked threshold on runtime spread.  Returns ``(nan, nan)`` when there
    are too few repeats to resample.
    """
    per_point = [np.asarray(p.times, dtype=float) for p in result.points]
    if len(per_point) < 3 or min(t.size for t in per_point) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    x = result.x
    draws: list[float] = []
    for _ in range(int(n_boot)):
        y = np.array([float(rng.choice(t)) for t in per_point])
        alpha, _, _ = fit_power_law_with_offset(x, y)
        if np.isfinite(alpha):
            draws.append(alpha)
    if len(draws) < 20:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``y = c * x**alpha`` in log-log space.

    Returns ``(alpha, c)``.  Returns ``(nan, nan)`` if fewer than two positive
    points are available rather than emitting a meaningless exponent.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return (float("nan"), float("nan"))
    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return (float(slope), float(np.exp(intercept)))


def fit_power_law_with_offset(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit ``y = t0 + c * x**alpha`` and return ``(alpha, c, t0)``.

    A plain log-log fit understates ``alpha`` whenever a parameter-independent
    fixed cost ``t0`` is a large share of the measured time.  That dilution is
    unavoidable for a fast implementation -- if the whole baseline run takes 2 s
    and 0.8 s of it does not depend on the swept parameter, the naive exponent is
    pulled toward zero even when the dependent part is exactly linear.

    ``t0`` is profiled over a coarse grid and ``(alpha, c)`` are fitted in
    log-log space for each candidate, keeping the best least-squares residual.
    Falls back to :func:`fit_power_law` when fewer than three points are usable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        alpha, c = fit_power_law(x, y)
        return (alpha, c, 0.0)

    xs, ys = x[mask], y[mask]
    plain_alpha, plain_c = fit_power_law(xs, ys)
    plain_sse = float(np.sum((plain_c * xs**plain_alpha - ys) ** 2))

    best = (float("nan"), float("nan"), 0.0, np.inf)
    # Cap the offset at 80% of the smallest measurement: beyond that the residual
    # spans less than a fifth of the data and the exponent stops being identified.
    for frac in np.linspace(0.0, 0.80, 81):
        t0 = frac * float(ys.min())
        resid_y = ys - t0
        if np.any(resid_y <= 0):
            continue
        slope, intercept = np.polyfit(np.log(xs), np.log(resid_y), 1)
        pred = np.exp(intercept) * xs**slope + t0
        sse = float(np.sum((pred - ys) ** 2))
        if sse < best[3]:
            best = (float(slope), float(np.exp(intercept)), float(t0), sse)

    alpha, c, t0, sse = best
    # Reject an offset fit that buys little accuracy or that runs away: with
    # noisy short timings the extra free parameter can chase noise and produce a
    # meaningless exponent. Returning the plain fit is the honest fallback.
    identifiable = (
        np.isfinite(alpha)
        and sse < 0.8 * max(plain_sse, 1e-30)
        and abs(alpha) < 2.0 * abs(plain_alpha) + 1.0
        and abs(alpha) <= 3.0
    )
    if not identifiable:
        return (plain_alpha, plain_c, 0.0)
    return (alpha, c, t0)


def baseline_timing_breakdown(repeats: int = 3) -> dict[str, float]:
    """Paper Fig. 11f: stage split at the baseline configuration."""
    _, _, _, stages = measure_once(
        m=BASELINE_CONFIG["m"],
        n_bar=BASELINE_CONFIG["n_bar"],
        K=BASELINE_CONFIG["K"],
        S=BASELINE_CONFIG["S"],
        G=BASELINE_CONFIG["G"],
        repeats=repeats,
    )
    return stages
