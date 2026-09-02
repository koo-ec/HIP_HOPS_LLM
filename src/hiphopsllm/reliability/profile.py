"""Operational profiles: what the system will actually be asked to do.

A benchmark accuracy is a statement about the benchmark's mix of tasks.  A
*reliability* claim has to be a statement about the mix the system will meet in
service, and those two are rarely the same --- a model measured on a corpus that
is 18% long multi-hop questions, then deployed on a workload that is 70% of them,
does not keep its measured failure rate.

The operational profile is that mix: a partition of the input space into strata
with the probability of each.  Everything downstream --- the imprecise posterior,
the calibrated basic events, the top-event probability --- is conditional on it,
so it is a first-class object here rather than a dictionary passed around.

The class is deliberately thin over
:class:`hip_llm.schemas.OperationalProfile`, so a profile built here can be
handed straight to HIP-LLM's inference and back.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "OperationalProfile",
    "dataset_proportional_profile",
    "empirical_profile",
    "uniform_profile",
    "stratify",
]


@dataclass(frozen=True)
class OperationalProfile:
    """A partition of the input space into strata, with a weight on each.

    Build one from a mapping::

        profile = OperationalProfile({"short": 0.30, "long": 0.70})

    or from observed usage::

        profile = empirical_profile(["short", "long", "long", "long"])
    """

    weights: Dict[str, float]
    name: str = "operational profile"
    #: how the weights were arrived at --- printed in every report that uses them
    provenance: str = "declared by the analyst"
    #: renormalise weights that do not sum to one, rather than raising
    normalise: bool = True

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("an operational profile needs at least one stratum")
        cleaned: Dict[str, float] = {}
        for label, weight in self.weights.items():
            key = str(label)
            if key in cleaned:
                raise ValueError(f"stratum {key!r} appears twice")
            w = float(weight)
            if not np.isfinite(w) or w < 0.0:
                raise ValueError(f"weight for {key!r} must be finite and non-negative")
            cleaned[key] = w
        total = sum(cleaned.values())
        if total <= 0.0:
            raise ValueError("operational-profile weights sum to zero")
        if abs(total - 1.0) > 1e-9:
            if not self.normalise:
                raise ValueError(
                    f"operational-profile weights sum to {total:.6f}, not 1.0"
                )
            cleaned = {k: v / total for k, v in cleaned.items()}
        object.__setattr__(self, "weights", cleaned)

    # -- access -------------------------------------------------------------- #
    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(self.weights)

    @property
    def vector(self) -> np.ndarray:
        return np.asarray([self.weights[k] for k in self.labels], dtype=float)

    def __len__(self) -> int:
        return len(self.weights)

    def __iter__(self):
        return iter(self.weights)

    def __getitem__(self, label: str) -> float:
        return self.weights[str(label)]

    def items(self):
        return self.weights.items()

    # -- use ----------------------------------------------------------------- #
    def expected(self, per_stratum: Mapping[str, float]) -> float:
        """Profile-weighted average of a per-stratum quantity.

        Raises if a stratum has no value: a silently dropped stratum is a
        silently reweighted profile, which is the kind of error that survives all
        the way into a published number.
        """
        missing = [k for k in self.labels if k not in per_stratum]
        if missing:
            raise KeyError(
                f"no value for stratum(s) {missing}; the profile covers {list(self.labels)}"
            )
        return float(sum(self.weights[k] * float(per_stratum[k]) for k in self.labels))

    def restricted_to(self, labels: Sequence[str]) -> "OperationalProfile":
        """The profile conditioned on a subset of strata, renormalised."""
        keep = [str(x) for x in labels]
        unknown = [k for k in keep if k not in self.weights]
        if unknown:
            raise KeyError(f"unknown stratum(s) {unknown}")
        return OperationalProfile(
            {k: self.weights[k] for k in keep},
            name=f"{self.name} | {', '.join(keep)}",
            provenance=f"{self.provenance}; restricted to {keep} and renormalised",
        )

    def to_hipllm(self, level: str = "benchmark_stratum"):
        """The equivalent :class:`hip_llm.schemas.OperationalProfile`."""
        from hip_llm.schemas import OperationalProfile as _HipProfile

        return _HipProfile(level=level, labels=self.labels, weights=self.vector)

    @classmethod
    def coerce(
        cls, profile: "OperationalProfile | Mapping[str, float] | Any"
    ) -> "OperationalProfile":
        """Accept this class, a plain mapping, or a HIP-LLM profile."""
        if isinstance(profile, cls):
            return profile
        labels = getattr(profile, "labels", None)
        weights = getattr(profile, "weights", None)
        if labels is not None and weights is not None and not isinstance(profile, Mapping):
            return cls(
                {str(k): float(v) for k, v in zip(labels, np.asarray(weights).ravel())},
                provenance="converted from a HIP-LLM OperationalProfile",
            )
        if isinstance(profile, Mapping):
            return cls({str(k): float(v) for k, v in profile.items()})
        raise TypeError(
            "profile must be an OperationalProfile, a mapping of "
            "{stratum: weight}, or a hip_llm OperationalProfile"
        )

    def summary(self) -> str:
        width = max(len(k) for k in self.labels)
        lines = [f"{self.name}  ({len(self)} strata)"]
        for label in self.labels:
            w = self.weights[label]
            bar = "█" * int(round(w * 30))
            lines.append(f"  {label:<{width}}  {w:6.3f}  {bar}")
        lines.append(f"  provenance: {self.provenance}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def empirical_profile(
    strata: Iterable[str], name: str = "empirical operational profile"
) -> OperationalProfile:
    """The profile implied by an observed sequence of stratum labels.

    Use this when the workload has been logged.  It records that provenance, so a
    reader can tell a measured profile from an assumed one.
    """
    counts = Counter(str(s) for s in strata)
    if not counts:
        raise ValueError("no stratum labels were given")
    total = sum(counts.values())
    return OperationalProfile(
        {k: v / total for k, v in counts.items()},
        name=name,
        provenance=f"observed frequencies over {total} logged items",
    )


def dataset_proportional_profile(
    strata: Iterable[str], name: str = "dataset-proportional profile"
) -> "OperationalProfile":
    """Weights proportional to the *benchmark's own* composition.

    HIP-LLM names this choice explicitly (paper Section 4.2, Remark 7) rather
    than letting it happen by default, and so does this. The paper's whole
    argument is that a benchmark accuracy is a descriptive statistic about the
    dataset, and becomes a reliability claim only once it is reweighted to the
    mix of work the system will actually meet. Taking the dataset's own mix as
    that workload asserts they are the same — sometimes true, never automatic.

    Use :func:`empirical_profile` when the labels come from *production* traffic;
    the two are computed identically and differ only in what they claim, which
    is exactly why they are separate functions.
    """
    counts = Counter(str(s) for s in strata)
    if not counts:
        raise ValueError("no stratum labels were given")
    total = sum(counts.values())
    return OperationalProfile(
        {k: v / total for k, v in counts.items()},
        name=name,
        provenance=(
            f"proportional to the benchmark's own composition over {total} items "
            "(HIP-LLM Remark 7) — this ASSERTS that the deployed workload has the "
            "same mix as the dataset, which is a claim, not a measurement"
        ),
    )


def uniform_profile(
    labels: Sequence[str], name: str = "uniform operational profile"
) -> OperationalProfile:
    """Equal weight on every stratum --- the honest default when nothing is known.

    It is a *choice*, not an absence of one, and it is recorded as such.
    """
    keys = [str(x) for x in labels]
    if not keys:
        raise ValueError("no strata were given")
    return OperationalProfile(
        {k: 1.0 / len(keys) for k in keys},
        name=name,
        provenance="assumed uniform; no usage data was supplied",
    )


def stratify(
    items: Iterable[Any],
    key: Any,
    profile: Optional[OperationalProfile] = None,
) -> List[str]:
    """Label each item with its stratum, checking the labels against a profile.

    ``key`` is a callable, or the name of a mapping key / attribute.  Passing a
    ``profile`` turns an unexpected label into an error rather than a stratum
    that silently carries zero weight.
    """
    if callable(key):
        getter = key
    else:

        def getter(item: Any) -> Any:
            if isinstance(item, Mapping):
                return item[key]
            return getattr(item, key)

    labels = [str(getter(item)) for item in items]
    if profile is not None:
        unknown = sorted(set(labels) - set(profile.labels))
        if unknown:
            raise ValueError(
                f"item(s) fall in stratum(s) {unknown}, which the operational "
                f"profile does not cover ({list(profile.labels)}). Either extend "
                "the profile or reclassify the items; a stratum with no weight "
                "silently drops those items from every downstream estimate."
            )
    return labels
