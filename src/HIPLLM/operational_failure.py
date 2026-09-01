"""Operational-profile failure probability from labelled benchmark outcomes.

This module is deliberately separate from :class:`HIPLLM.FailureProb`.

``FailureProb`` transforms token probabilities returned by a language-model
provider. It is a prompt-level confidence heuristic and does not use an
operational profile. ``OperationalFailureProb`` instead estimates the
probability that a future benchmark item will fail when items are drawn from an
explicit operational profile. It requires observed binary correctness labels
and uses the repository's hierarchical imprecise-Bayesian inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hip_llm.posterior import run_domain
from hip_llm.schemas import (
    DomainData,
    GlobalSettings,
    HyperparameterInterval,
    OperationalProfile,
    SubdomainData,
)

__all__ = [
    "OperationalFailureProb",
    "OperationalFailureResult",
    "quick_inference_settings",
    "paper_inference_settings",
]


def quick_inference_settings(
    *,
    seed: int = 7,
    samples: int = 1500,
    configurations: int = 48,
) -> GlobalSettings:
    """Return a Colab-friendly inference configuration.

    The settings retain the same model and reconstruction choices as the paper
    implementation, but use smaller Monte-Carlo and configuration counts for an
    interactive benchmark run. Use :func:`paper_inference_settings` for the
    published baseline sizes.
    """
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if configurations < 4:
        raise ValueError("configurations must be at least 4")
    return GlobalSettings(
        n_mu=20,
        n_nu=25,
        cdf_points_T=201,
        S=int(samples),
        K_per_domain=int(configurations),
        max_llm_configuration_pairs=int(configurations),
        seed_global=int(seed),
        seed_configs=int(seed) + 116,
        seed_pairs=int(seed) + 992,
        config_sampling="latin_hypercube",
        nu_grid_scheme="log",
        mu_grid_scheme="midpoint",
        nu_grid_params={"nu_lo": 1e-3, "nu_hi": 250.0},
        strict_exact=False,
    )


def paper_inference_settings() -> GlobalSettings:
    """Return the paper-sized baseline inference settings.

    The finite ``nu`` grid and configuration-sampling rules are reconstructed
    because the paper does not publish them. They are therefore labelled as
    reconstruction choices even though the grid sizes, sample count and number
    of configurations match the published baseline.
    """
    return GlobalSettings(
        n_mu=40,
        n_nu=50,
        cdf_points_T=201,
        S=3000,
        K_per_domain=160,
        max_llm_configuration_pairs=512,
        seed_global=7,
        seed_configs=123,
        seed_pairs=999,
        config_sampling="uniform_random",
        nu_grid_scheme="log",
        mu_grid_scheme="midpoint",
        nu_grid_params={"nu_lo": 1e-3, "nu_hi": 250.0},
        strict_exact=False,
    )


def _coerce_profile(
    profile: OperationalProfile | Mapping[str, float],
) -> OperationalProfile:
    if isinstance(profile, OperationalProfile):
        if len(set(profile.labels)) != len(profile.labels):
            raise ValueError("operational-profile labels must be unique")
        return profile
    if not isinstance(profile, Mapping) or not profile:
        raise TypeError(
            "profile must be a non-empty OperationalProfile or mapping of "
            "{stratum_label: probability}"
        )
    keys = tuple(profile.keys())
    labels = tuple(str(label) for label in keys)
    if len(set(labels)) != len(labels):
        raise ValueError("operational-profile labels must be unique after string conversion")
    weights = np.asarray([profile[key] for key in keys], dtype=float)
    return OperationalProfile(level="benchmark_stratum", labels=labels, weights=weights)


@dataclass(frozen=True)
class OperationalFailureResult:
    """Posterior failure-probability family under an operational profile.

    ``failure_samples`` has shape ``(K, S)``: one posterior sample row per
    admissible hyper-hyperparameter configuration. The result therefore reports
    lower and upper bounds rather than hiding imprecision behind one number.
    """

    profile: OperationalProfile
    successes: np.ndarray = field(repr=False)
    trials: np.ndarray = field(repr=False)
    failure_samples: np.ndarray = field(repr=False)
    theta_samples: np.ndarray = field(repr=False)
    credible_level: float = 0.95
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        successes = np.asarray(self.successes, dtype=int)
        trials = np.asarray(self.trials, dtype=int)
        failure = np.asarray(self.failure_samples, dtype=float)
        theta = np.asarray(self.theta_samples, dtype=float)

        n = len(self.profile.labels)
        if successes.shape != (n,) or trials.shape != (n,):
            raise ValueError("successes and trials must match the operational-profile labels")
        if np.any(trials <= 0):
            raise ValueError("every operational-profile stratum must contain observations")
        if np.any(successes < 0) or np.any(successes > trials):
            raise ValueError("success counts must lie between zero and trials")
        if failure.ndim != 2 or failure.size == 0:
            raise ValueError("failure_samples must be a non-empty (K, S) array")
        if theta.shape != (failure.shape[0], failure.shape[1], n):
            raise ValueError(
                "theta_samples must have shape (K, S, number_of_profile_labels)"
            )
        if not np.all(np.isfinite(failure)) or np.any((failure < 0) | (failure > 1)):
            raise ValueError("failure_samples must be finite and lie in [0, 1]")
        if not np.all(np.isfinite(theta)) or np.any((theta < 0) | (theta > 1)):
            raise ValueError("theta_samples must be finite and lie in [0, 1]")
        if not (0.0 < float(self.credible_level) < 1.0):
            raise ValueError("credible_level must lie strictly between 0 and 1")

        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "failure_samples", failure)
        object.__setattr__(self, "theta_samples", theta)
        object.__setattr__(self, "credible_level", float(self.credible_level))

    @property
    def empirical_failure_by_stratum(self) -> np.ndarray:
        """Observed error rate in each operational stratum."""
        return 1.0 - self.successes / self.trials

    @property
    def empirical_operational_failure_probability(self) -> float:
        """Plug-in estimate using observed stratum error rates and OP weights."""
        return float(self.profile.weights @ self.empirical_failure_by_stratum)

    @property
    def posterior_expected_failure_bounds(self) -> tuple[float, float]:
        """Envelope of posterior expected failure across configurations."""
        means = self.failure_samples.mean(axis=1)
        return float(means.min()), float(means.max())

    @property
    def posterior_median_failure_bounds(self) -> tuple[float, float]:
        """Envelope of posterior median failure across configurations."""
        medians = np.median(self.failure_samples, axis=1)
        return float(medians.min()), float(medians.max())

    @property
    def posterior_credible_envelope(self) -> tuple[float, float]:
        """Outer equal-tail credible envelope across all configurations."""
        alpha = (1.0 - self.credible_level) / 2.0
        lower = np.quantile(self.failure_samples, alpha, axis=1)
        upper = np.quantile(self.failure_samples, 1.0 - alpha, axis=1)
        return float(lower.min()), float(upper.max())

    def summary(self) -> dict[str, Any]:
        """Return the operational estimate and imprecise posterior bounds."""
        expected_lo, expected_hi = self.posterior_expected_failure_bounds
        median_lo, median_hi = self.posterior_median_failure_bounds
        credible_lo, credible_hi = self.posterior_credible_envelope
        return {
            "empirical_operational_failure_probability": (
                self.empirical_operational_failure_probability
            ),
            "posterior_expected_failure_lower": expected_lo,
            "posterior_expected_failure_upper": expected_hi,
            "posterior_median_failure_lower": median_lo,
            "posterior_median_failure_upper": median_hi,
            "posterior_credible_lower": credible_lo,
            "posterior_credible_upper": credible_hi,
            "credible_level": self.credible_level,
            "n_configurations": int(self.failure_samples.shape[0]),
            "samples_per_configuration": int(self.failure_samples.shape[1]),
            "operational_profile": dict(
                zip(self.profile.labels, self.profile.weights.tolist(), strict=True)
            ),
            "metadata": dict(self.metadata),
        }

    def to_df(self):
        """Return one row per operational stratum as a pandas DataFrame."""
        import pandas as pd

        theta_means = self.theta_samples.mean(axis=1)
        return pd.DataFrame(
            {
                "stratum": self.profile.labels,
                "operational_weight": self.profile.weights,
                "successes": self.successes,
                "trials": self.trials,
                "empirical_reliability": self.successes / self.trials,
                "empirical_failure_probability": self.empirical_failure_by_stratum,
                "posterior_mean_reliability_lower": theta_means.min(axis=0),
                "posterior_mean_reliability_upper": theta_means.max(axis=0),
            }
        )


class OperationalFailureProb:
    """Estimate benchmark-level failure probability under an explicit OP.

    Parameters
    ----------
    profile:
        A probability mapping from benchmark stratum to target workload share,
        for example ``{"short": 0.3, "medium": 0.5, "long": 0.2}``.
        The profile is required and is never inferred silently.
    interval:
        Admissible hyper-hyperparameter interval. The default matches the
        ranges used by the HIP-LLM paper.
    settings:
        Numerical inference settings. The default is Colab-friendly; pass
        :func:`paper_inference_settings` for the published baseline sizes.
    credible_level:
        Equal-tail posterior credible level used by the summary.
    """

    def __init__(
        self,
        profile: OperationalProfile | Mapping[str, float],
        *,
        interval: HyperparameterInterval | None = None,
        settings: GlobalSettings | None = None,
        credible_level: float = 0.95,
    ) -> None:
        self.profile = _coerce_profile(profile)
        self.interval = interval or HyperparameterInterval(
            a=(1.0, 12.0),
            b=(1.0, 12.0),
            c=(1.0, 25.0),
            d=(1.0, 25.0),
        )
        self.settings = settings or quick_inference_settings()
        if not (0.0 < float(credible_level) < 1.0):
            raise ValueError("credible_level must lie strictly between 0 and 1")
        self.credible_level = float(credible_level)

    def fit(
        self,
        outcomes: Sequence[bool | int],
        strata: Sequence[str],
        *,
        domain_name: str = "operational_workload",
    ) -> OperationalFailureResult:
        """Fit from binary correctness outcomes and matching stratum labels.

        ``outcomes[k]`` is ``1``/``True`` when the model answered item ``k``
        correctly and ``0``/``False`` otherwise. ``strata[k]`` identifies the
        operational-profile subdomain for the same item.
        """
        if isinstance(outcomes, (str, bytes)) or isinstance(strata, (str, bytes)):
            raise TypeError("outcomes and strata must be sequences, not strings")
        if len(outcomes) == 0:
            raise ValueError("at least one benchmark outcome is required")
        if len(outcomes) != len(strata):
            raise ValueError("outcomes and strata must have the same length")

        numeric = np.asarray(outcomes)
        if numeric.ndim != 1:
            raise ValueError("outcomes must be one-dimensional")
        try:
            numeric_float = numeric.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("outcomes must contain only binary 0/1 values") from exc
        if not np.all(np.isfinite(numeric_float)) or not np.all(
            np.isin(numeric_float, (0.0, 1.0))
        ):
            raise ValueError("outcomes must contain only binary 0/1 values")
        y = numeric_float.astype(int)

        stratum_values = np.asarray([str(value) for value in strata], dtype=object)
        known = set(self.profile.labels)
        unknown = sorted(set(stratum_values.tolist()) - known)
        if unknown:
            raise ValueError(
                f"benchmark outcomes contain strata absent from the operational profile: {unknown}"
            )

        successes: list[int] = []
        trials: list[int] = []
        subdomains: list[SubdomainData] = []
        for label in self.profile.labels:
            mask = stratum_values == label
            n = int(mask.sum())
            if n == 0:
                raise ValueError(
                    f"operational-profile stratum {label!r} has no benchmark observations; "
                    "collect data for it or remove it from the profile"
                )
            c = int(y[mask].sum())
            successes.append(c)
            trials.append(n)
            subdomains.append(SubdomainData(name=label, successes=c, trials=n))

        domain = DomainData(
            name=domain_name,
            subdomains=tuple(subdomains),
            omega=self.profile.weights,
        )
        posterior = run_domain(
            domain=domain,
            interval=self.interval,
            settings=self.settings,
            domain_index=0,
            model_index=0,
        )

        failure = 1.0 - posterior.p
        return OperationalFailureResult(
            profile=self.profile,
            successes=np.asarray(successes, dtype=int),
            trials=np.asarray(trials, dtype=int),
            failure_samples=failure,
            theta_samples=posterior.theta,
            credible_level=self.credible_level,
            metadata={
                "domain": domain_name,
                "inference_grid": posterior.meta.get("grid_scheme"),
                "configuration_sampling": posterior.meta.get("config_sampling"),
                "reconstruction": True,
            },
        )
