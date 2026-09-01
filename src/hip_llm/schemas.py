"""Validated data schemas for the HIP-LLM replication.

Every structure that crosses a module boundary is defined here as a frozen
dataclass with eager validation.  Invalid probability vectors, negative counts,
counts above the sample size, malformed intervals and non-finite values are
rejected at construction time rather than producing silently wrong posteriors.

Paper cross-references use the numbering of

    R. Aghazadeh-Chakherlou, Q. Guo, S. Khastgir, P. Popov, X. Zhang, X. Zhao,
    "A hierarchical imprecise probability approach to reliability assessment of
    large language models", Reliability Engineering & System Safety 272 (2026)
    112615.  doi:10.1016/j.ress.2026.112615
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

__all__ = [
    "WEIGHT_SUM_TOL",
    "SubdomainData",
    "DomainData",
    "ModelResult",
    "BenchmarkResult",
    "OperationalProfile",
    "HyperparameterInterval",
    "HyperparameterConfiguration",
    "HyperposteriorGrid",
    "PosteriorSamples",
    "CDFEnvelope",
    "ReliabilityEnvelope",
    "ReproductionStatus",
    "ReproductionRecord",
    "SourceRecord",
    "RunMode",
    "GlobalSettings",
    "load_yaml",
    "config_hash",
    "sha256_file",
]

# Tolerance used when checking that an operational-profile vector sums to one.
WEIGHT_SUM_TOL = 1e-9


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {arr}")
    return arr


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file (streamed, 1 MiB blocks)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, failing loudly if it is missing or not a mapping."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"{p} must contain a YAML mapping at the top level")
    return data


def config_hash(obj: Any) -> str:
    """Stable SHA-256 over a JSON-serialisable configuration object.

    Used to key caches and to stamp every figure/table with the configuration
    that produced it.
    """

    def _default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        raise TypeError(f"cannot hash object of type {type(o)!r}")

    blob = json.dumps(obj, sort_keys=True, default=_default, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# observed evaluation data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubdomainData:
    """Observed evaluation data for one subdomain :math:`S_{ij}`.

    Corresponds to :math:`C_{ij} \\mid \\theta_{ij}, N_{ij} \\sim
    \\mathrm{Binomial}(N_{ij}, \\theta_{ij})` (paper Section 3.2.1, Appendix A.1).

    Attributes
    ----------
    name:
        Subdomain label, e.g. ``"MBPP"``.
    successes:
        :math:`C_{ij}`, number of correct responses.
    trials:
        :math:`N_{ij}`, number of evaluated tasks.
    source_accuracy:
        The published accuracy :math:`\\hat\\theta_{ij}` this record was derived
        from, retained for provenance.  ``None`` when counts are primary data.
    """

    name: str
    successes: int
    trials: int
    source_accuracy: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("subdomain name must be a non-empty string")
        if int(self.trials) != self.trials or self.trials <= 0:
            raise ValueError(f"{self.name}: trials must be a positive integer, got {self.trials!r}")
        if int(self.successes) != self.successes or self.successes < 0:
            raise ValueError(
                f"{self.name}: successes must be a non-negative integer, got {self.successes!r}"
            )
        if self.successes > self.trials:
            raise ValueError(
                f"{self.name}: successes ({self.successes}) exceed trials ({self.trials})"
            )
        if self.source_accuracy is not None:
            a = float(self.source_accuracy)
            if not np.isfinite(a) or not (0.0 <= a <= 1.0):
                raise ValueError(f"{self.name}: source_accuracy must lie in [0,1], got {a}")

    @property
    def empirical_accuracy(self) -> float:
        """Effective accuracy after rounding to integer counts, :math:`C_{ij}/N_{ij}`."""
        return self.successes / self.trials


@dataclass(frozen=True)
class DomainData:
    """One domain :math:`D_i`: its subdomains plus its within-domain OP weights."""

    name: str
    subdomains: tuple[SubdomainData, ...]
    omega: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if not self.subdomains:
            raise ValueError(f"domain {self.name} must contain at least one subdomain")
        omega = _as_float_array(self.omega, f"omega[{self.name}]")
        if omega.size != len(self.subdomains):
            raise ValueError(
                f"domain {self.name}: {omega.size} omega weights for "
                f"{len(self.subdomains)} subdomains"
            )
        if np.any(omega < 0.0):
            raise ValueError(f"domain {self.name}: omega weights must be non-negative")
        if abs(omega.sum() - 1.0) > 1e-8:
            raise ValueError(
                f"domain {self.name}: omega weights sum to {omega.sum():.12g}, expected 1"
            )
        object.__setattr__(self, "omega", omega)

    @property
    def n_subdomains(self) -> int:
        return len(self.subdomains)

    @property
    def counts(self) -> np.ndarray:
        """Vector :math:`(C_{i1},\\dots,C_{in_i})`."""
        return np.array([s.successes for s in self.subdomains], dtype=float)

    @property
    def trials(self) -> np.ndarray:
        """Vector :math:`(N_{i1},\\dots,N_{in_i})`."""
        return np.array([s.trials for s in self.subdomains], dtype=float)

    @property
    def subdomain_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.subdomains)

    def data_key(self) -> str:
        """Hashable key over the observed data, used for caching."""
        return config_hash(
            {"name": self.name, "C": self.counts.tolist(), "N": self.trials.tolist()}
        )


@dataclass(frozen=True)
class ModelResult:
    """The full hierarchy for one LLM: independent domains plus domain weights :math:`W_i`."""

    model: str
    domains: tuple[DomainData, ...]
    W: np.ndarray = field(repr=False)
    source_label: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.domains:
            raise ValueError(f"model {self.model} must contain at least one domain")
        W = _as_float_array(self.W, f"W[{self.model}]")
        if W.size != len(self.domains):
            raise ValueError(
                f"model {self.model}: {W.size} domain weights for {len(self.domains)} domains"
            )
        if np.any(W < 0.0):
            raise ValueError(f"model {self.model}: domain weights must be non-negative")
        if abs(W.sum() - 1.0) > 1e-8:
            raise ValueError(f"model {self.model}: domain weights sum to {W.sum():.12g}, expected 1")
        object.__setattr__(self, "W", W)

    @property
    def n_domains(self) -> int:
        return len(self.domains)

    def domain(self, name: str) -> DomainData:
        for d in self.domains:
            if d.name == name:
                return d
        raise KeyError(f"model {self.model} has no domain {name!r}")


@dataclass(frozen=True)
class BenchmarkResult:
    """Per-task outcome record produced by a real benchmark evaluation run."""

    benchmark: str
    model_snapshot: str
    task_ids: tuple[str, ...]
    outcomes: tuple[int, ...]
    success_definition: str = "pass@1"
    n_api_errors: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.task_ids) != len(self.outcomes):
            raise ValueError("task_ids and outcomes must have the same length")
        if any(o not in (0, 1) for o in self.outcomes):
            raise ValueError("outcomes must be binary 0/1 (API errors are recorded separately)")
        if self.n_api_errors < 0:
            raise ValueError("n_api_errors must be non-negative")

    @property
    def successes(self) -> int:
        return int(sum(self.outcomes))

    @property
    def trials(self) -> int:
        return len(self.outcomes)

    def to_subdomain(self, name: str) -> SubdomainData:
        """Task reliability *conditional on a valid model response* (paper's experiment)."""
        return SubdomainData(name=name, successes=self.successes, trials=self.trials)

    def service_level_subdomain(self, name: str) -> SubdomainData:
        """Service-level reliability: API failures counted as failures.

        Kept strictly separate from :meth:`to_subdomain`; the paper reports only
        the conditional quantity.
        """
        return SubdomainData(
            name=f"{name}[service-level]",
            successes=self.successes,
            trials=self.trials + self.n_api_errors,
        )


# --------------------------------------------------------------------------- #
# operational profile
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OperationalProfile:
    """A validated probability vector over (sub-)domains (paper Definition 2)."""

    level: str
    labels: tuple[str, ...]
    weights: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        w = _as_float_array(self.weights, f"OP[{self.level}]")
        if len(self.labels) != w.size:
            raise ValueError(
                f"OP[{self.level}]: {len(self.labels)} labels for {w.size} weights"
            )
        if np.any(w < 0.0):
            raise ValueError(f"OP[{self.level}]: weights must be non-negative, got {w}")
        if abs(w.sum() - 1.0) > WEIGHT_SUM_TOL:
            raise ValueError(
                f"OP[{self.level}]: weights sum to {w.sum():.17g}, expected 1 "
                f"(tolerance {WEIGHT_SUM_TOL})"
            )
        object.__setattr__(self, "weights", w)

    def __len__(self) -> int:
        return self.weights.size


# --------------------------------------------------------------------------- #
# imprecise hyper-hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HyperparameterInterval:
    """One admissible hyper-rectangle :math:`\\mathcal{H}_i` (paper Eq. 4).

    :math:`a_i\\in[a^{\\min}_i,a^{\\max}_i]`, and likewise for
    :math:`b_i, c_i, d_i`, with :math:`\\mu_i\\sim\\mathrm{Beta}(a_i,b_i)` and
    :math:`\\nu_i\\sim\\mathrm{Gamma}(c_i,\\ \\mathrm{rate}=d_i)`.
    """

    a: tuple[float, float]
    b: tuple[float, float]
    c: tuple[float, float]
    d: tuple[float, float]

    def __post_init__(self) -> None:
        for label in ("a", "b", "c", "d"):
            lo, hi = getattr(self, label)
            lo, hi = float(lo), float(hi)
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(f"interval {label} must be finite, got [{lo}, {hi}]")
            if lo <= 0.0:
                raise ValueError(
                    f"interval {label} lower bound must be strictly positive "
                    f"(Beta/Gamma shape and rate), got {lo}"
                )
            if hi < lo:
                raise ValueError(f"interval {label} is malformed: [{lo}, {hi}]")
            object.__setattr__(self, label, (lo, hi))

    @property
    def bounds(self) -> np.ndarray:
        """``(4, 2)`` array of ``[lo, hi]`` rows ordered ``a, b, c, d``."""
        return np.array([self.a, self.b, self.c, self.d], dtype=float)

    @property
    def is_precise(self) -> bool:
        """True when every interval collapses to a point (paper's precise special case)."""
        return bool(np.all(self.bounds[:, 0] == self.bounds[:, 1]))

    def with_replaced(self, **kwargs: tuple[float, float]) -> "HyperparameterInterval":
        """Return a copy with selected intervals replaced (used by RQ2)."""
        current = {"a": self.a, "b": self.b, "c": self.c, "d": self.d}
        unknown = set(kwargs) - set(current)
        if unknown:
            raise KeyError(f"unknown interval name(s): {sorted(unknown)}")
        current.update(kwargs)
        return HyperparameterInterval(**current)  # type: ignore[arg-type]


@dataclass(frozen=True)
class HyperparameterConfiguration:
    """A single admissible :math:`h_i=(a_i,b_i,c_i,d_i)\\in\\mathcal{H}_i`."""

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        for label in ("a", "b", "c", "d"):
            v = float(getattr(self, label))
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError(f"hyperparameter {label} must be finite and positive, got {v}")
            object.__setattr__(self, label, v)

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c, self.d], dtype=float)


@dataclass(frozen=True)
class HyperposteriorGrid:
    """Discretised :math:`(\\mu,\\nu)` integration grid with quadrature weights.

    ``mu`` and ``nu`` are the flattened node coordinates of the :math:`G=n_\\mu
    n_\\nu` cells; ``log_cell_weight`` holds :math:`\\log(\\Delta\\mu_g
    \\Delta\\nu_g)` for a plain product rule, or the log of the *prior* cell mass
    when a quantile-transformed axis absorbs the prior density (see
    :mod:`hip_llm.grids`).  ``absorbs_mu_prior`` / ``absorbs_nu_prior`` record
    which prior factors have already been folded into the weights so that
    :mod:`hip_llm.hyperposterior` does not double-count them.
    """

    mu: np.ndarray = field(repr=False)
    nu: np.ndarray = field(repr=False)
    log_cell_weight: np.ndarray = field(repr=False)
    n_mu: int
    n_nu: int
    scheme: str
    absorbs_mu_prior: bool = False
    absorbs_nu_prior: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("mu", "nu", "log_cell_weight"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, arr)
        if not (self.mu.size == self.nu.size == self.log_cell_weight.size):
            raise ValueError("mu, nu and log_cell_weight must have equal length")
        if self.mu.size != self.n_mu * self.n_nu:
            raise ValueError(
                f"grid has {self.mu.size} cells but n_mu*n_nu = {self.n_mu * self.n_nu}"
            )
        if np.any(self.mu <= 0.0) or np.any(self.mu >= 1.0):
            raise ValueError("mu nodes must lie strictly inside (0, 1)")
        if np.any(self.nu <= 0.0):
            raise ValueError("nu nodes must be strictly positive")

    @property
    def size(self) -> int:
        return self.mu.size

    def grid_key(self) -> str:
        return config_hash(
            {
                "scheme": self.scheme,
                "n_mu": self.n_mu,
                "n_nu": self.n_nu,
                "mu": np.round(self.mu, 12).tolist(),
                "nu": np.round(self.nu, 12).tolist(),
                "w": np.round(self.log_cell_weight, 12).tolist(),
                "absorbs": [self.absorbs_mu_prior, self.absorbs_nu_prior],
            }
        )


# --------------------------------------------------------------------------- #
# posterior products
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PosteriorSamples:
    """Monte-Carlo draws for one domain under one hyperparameter configuration.

    ``theta`` has shape ``(S, n_i)`` and ``mu``/``nu`` have shape ``(S,)``.  The
    :math:`s`-th row of ``theta`` was drawn under the *shared* latent pair
    ``(mu[s], nu[s])``; this is what induces the marginal within-domain
    dependence described in paper Section 3.2.1.
    """

    domain: str
    config: HyperparameterConfiguration
    theta: np.ndarray = field(repr=False)
    mu: np.ndarray = field(repr=False)
    nu: np.ndarray = field(repr=False)
    p: np.ndarray = field(repr=False)
    subdomain_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float)
        if theta.ndim != 2:
            raise ValueError("theta must be a (S, n_i) array")
        for name in ("mu", "nu", "p"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (theta.shape[0],):
                raise ValueError(f"{name} must have shape ({theta.shape[0]},)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, arr)
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta contains non-finite values")
        if np.any(theta < 0.0) or np.any(theta > 1.0):
            raise ValueError("theta draws must lie in [0, 1]")
        if np.any(self.p < 0.0) or np.any(self.p > 1.0):
            raise ValueError("domain reliability draws must lie in [0, 1]")
        object.__setattr__(self, "theta", theta)

    @property
    def n_samples(self) -> int:
        return self.theta.shape[0]


@dataclass(frozen=True)
class CDFEnvelope:
    """Pointwise lower/upper CDF envelopes over the admissible set.

    ``lower[t] = inf_h F_h(t)`` and ``upper[t] = sup_h F_h(t)`` (paper Theorems
    1--3).  Note the orientation: a *lower* CDF corresponds to a stochastically
    *larger* reliability distribution.
    """

    quantity: str
    t_grid: np.ndarray = field(repr=False)
    lower: np.ndarray = field(repr=False)
    upper: np.ndarray = field(repr=False)
    n_configs: int = 0
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = _as_float_array(self.t_grid, "t_grid")
        lo = _as_float_array(self.lower, "lower")
        up = _as_float_array(self.upper, "upper")
        if not (t.size == lo.size == up.size):
            raise ValueError("t_grid, lower and upper must have equal length")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t_grid must be strictly increasing")
        for name, arr in (("lower", lo), ("upper", up)):
            if np.any(arr < -1e-12) or np.any(arr > 1.0 + 1e-12):
                raise ValueError(f"{name} CDF values must lie in [0, 1]")
            if np.any(np.diff(arr) < -1e-12):
                raise ValueError(f"{name} CDF must be non-decreasing")
        if np.any(lo > up + 1e-12):
            raise ValueError("lower CDF envelope exceeds upper CDF envelope")
        object.__setattr__(self, "t_grid", t)
        object.__setattr__(self, "lower", np.clip(lo, 0.0, 1.0))
        object.__setattr__(self, "upper", np.clip(up, 0.0, 1.0))

    @property
    def area(self) -> float:
        """Area between the CDF envelopes -- a scalar measure of imprecision."""
        return float(np.trapezoid(self.upper - self.lower, self.t_grid))

    @property
    def max_separation(self) -> float:
        """Sup-norm separation between the two envelope curves."""
        return float(np.max(self.upper - self.lower))


@dataclass(frozen=True)
class ReliabilityEnvelope:
    """Lower/upper envelopes of :math:`\\mathbb{E}[R(n_F)]` across configurations.

    Paper Theorems 4--6 / RQ4.  Computed as
    ``min_h mean(p_h**n_F)`` and ``max_h mean(p_h**n_F)`` -- never as
    ``mean(p)**n_F``, because :math:`\\mathbb{E}[p^{n}]\\neq\\mathbb{E}[p]^{n}`.
    """

    quantity: str
    horizons: np.ndarray = field(repr=False)
    lower: np.ndarray = field(repr=False)
    upper: np.ndarray = field(repr=False)
    n_configs: int = 0
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        h = _as_float_array(self.horizons, "horizons")
        lo = _as_float_array(self.lower, "lower")
        up = _as_float_array(self.upper, "upper")
        if not (h.size == lo.size == up.size):
            raise ValueError("horizons, lower and upper must have equal length")
        if np.any(h < 1):
            raise ValueError("horizons must be >= 1")
        if np.any(np.diff(h) <= 0):
            raise ValueError("horizons must be strictly increasing")
        for name, arr in (("lower", lo), ("upper", up)):
            if np.any(arr < -1e-12) or np.any(arr > 1.0 + 1e-12):
                raise ValueError(f"{name} expected reliability must lie in [0, 1]")
        if np.any(lo > up + 1e-12):
            raise ValueError("lower reliability envelope exceeds upper envelope")
        object.__setattr__(self, "horizons", h)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", up)

    @property
    def width(self) -> np.ndarray:
        """Envelope width (upper - lower), plotted in paper Fig. 8b."""
        return self.upper - self.lower


# --------------------------------------------------------------------------- #
# provenance / reproduction bookkeeping
# --------------------------------------------------------------------------- #
class ReproductionStatus(str, Enum):
    """Classification required by the reproduction report (master spec Section 8.15)."""

    EXACT = "exact"
    STATISTICALLY_EQUIVALENT = "statistically_equivalent_within_tolerance"
    CONTEMPORARY_RERUN = "faithful_contemporary_rerun"
    RECONSTRUCTED = "reconstructed_with_labelled_assumption"
    BLOCKED = "blocked_by_missing_source_information"
    INCONSISTENT_SOURCE = "inconsistent_source_artifact"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class ReproductionRecord:
    """One row of the paper-audit / reproduction-status table."""

    paper_item: str
    inputs: str
    notebook_section: str
    output_file: str
    status: ReproductionStatus
    note: str = ""

    def as_row(self) -> dict[str, str]:
        return {
            "paper_item": self.paper_item,
            "inputs": self.inputs,
            "notebook_section": self.notebook_section,
            "output_file": self.output_file,
            "status": self.status.value,
            "note": self.note,
        }


@dataclass(frozen=True)
class SourceRecord:
    """One entry of ``data/provenance_manifest.yaml``."""

    source_id: str
    role: str
    location: str
    retrieved: str
    sha256: str | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "role": self.role,
            "location": self.location,
            "retrieved": self.retrieved,
            "sha256": self.sha256,
            "note": self.note,
        }


class RunMode(str, Enum):
    """Top-level data mode (master spec Section 6)."""

    PUBLISHED_NUMERICS = "published_numerics"
    LIVE_API = "live_api"


@dataclass(frozen=True)
class GlobalSettings:
    """The paper's baseline experimental configuration (Section 4.2, Appendix B)."""

    n_mu: int
    n_nu: int
    cdf_points_T: int
    S: int
    K_per_domain: int
    max_llm_configuration_pairs: int
    seed_global: int
    seed_configs: int
    seed_pairs: int
    config_sampling: str
    nu_grid_scheme: str
    mu_grid_scheme: str
    nu_grid_params: Mapping[str, Any] = field(default_factory=dict)
    strict_exact: bool = False

    def __post_init__(self) -> None:
        for name in (
            "n_mu",
            "n_nu",
            "cdf_points_T",
            "S",
            "K_per_domain",
            "max_llm_configuration_pairs",
        ):
            v = getattr(self, name)
            if int(v) != v or v <= 0:
                raise ValueError(f"{name} must be a positive integer, got {v!r}")

    @property
    def G(self) -> int:
        """Total integration grid size :math:`G=n_\\mu n_\\nu`."""
        return self.n_mu * self.n_nu

    def hash(self) -> str:
        return config_hash(asdict(self))
