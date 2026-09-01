"""Invariant checks, source-discrepancy detection and provenance capture.

Two families live here:

* **Mathematical invariants** that a correct implementation must satisfy
  (monotone CDFs, ordered envelopes, medians inside their own intervals, ...).
  These are applied to *our* computed output.
* **Source-integrity checks** that compare the paper's printed artifacts against
  the official repository numerics and against internal consistency.  These are
  applied to the *published* artifacts and are expected to FAIL for the
  discrepancies documented below -- failing is the finding, not a bug.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .schemas import CDFEnvelope, SourceRecord, sha256_file

__all__ = [
    "CheckResult",
    "check_cdf_valid",
    "check_envelope_ordered",
    "check_median_within_interval",
    "check_reliability_monotone",
    "compare_accuracy_sources",
    "check_table5_internal_consistency",
    "check_fig9_against_sources",
    "environment_report",
    "build_provenance",
    "assert_no_overwrite",
]


@dataclass
class CheckResult:
    """Outcome of one named check."""

    name: str
    passed: bool
    detail: str = ""
    data: Any = field(default=None, repr=False)

    def __bool__(self) -> bool:
        return self.passed

    def as_row(self) -> dict[str, Any]:
        return {"check": self.name, "passed": self.passed, "detail": self.detail}


# --------------------------------------------------------------------------- #
# invariants on computed output
# --------------------------------------------------------------------------- #
def check_cdf_valid(F: np.ndarray, name: str = "CDF", tol: float = 1e-9) -> CheckResult:
    """Non-decreasing, within [0,1], with endpoints approaching 0 and 1."""
    F = np.atleast_2d(np.asarray(F, dtype=float))
    problems: list[str] = []
    if np.any(F < -tol) or np.any(F > 1.0 + tol):
        problems.append(f"values outside [0,1] (min={F.min():.6g}, max={F.max():.6g})")
    if np.any(np.diff(F, axis=-1) < -tol):
        worst = float(np.diff(F, axis=-1).min())
        problems.append(f"non-monotone (largest decrease {worst:.3g})")
    if np.any(F[:, 0] > 1e-2):
        problems.append(f"left endpoint not near 0 (max={F[:, 0].max():.4g})")
    if np.any(F[:, -1] < 1.0 - 1e-2):
        problems.append(f"right endpoint not near 1 (min={F[:, -1].min():.4g})")
    return CheckResult(
        name=f"{name}: valid CDF",
        passed=not problems,
        detail="; ".join(problems) or "non-decreasing, in [0,1], endpoints 0 -> 1",
    )


def check_envelope_ordered(env: CDFEnvelope, tol: float = 1e-12) -> CheckResult:
    """``lower_CDF(t) <= upper_CDF(t)`` pointwise."""
    viol = int(np.count_nonzero(env.lower > env.upper + tol))
    return CheckResult(
        name=f"{env.quantity}: lower CDF <= upper CDF",
        passed=viol == 0,
        detail=f"{viol} violating grid point(s)" if viol else "ordered at every grid point",
    )


def check_median_within_interval(
    median: tuple[float, float] | float,
    interval: tuple[float, float],
    label: str,
    tol: float = 1e-9,
) -> CheckResult:
    """A posterior median must lie inside that posterior's own 5%-95% interval.

    This is the invariant violated by two printed HIP-LLM rows of the paper's
    Table 5 (see :func:`check_table5_internal_consistency`).
    """
    lo, hi = float(interval[0]), float(interval[1])
    med = (float(median), float(median)) if np.isscalar(median) else (float(median[0]), float(median[1]))  # type: ignore[arg-type]
    ok = (lo - tol) <= med[0] and med[1] <= (hi + tol)
    return CheckResult(
        name=f"{label}: median within its own 90% interval",
        passed=ok,
        detail=(
            f"median {med} vs interval [{lo}, {hi}]"
            + ("" if ok else "  <-- IMPOSSIBLE for a single posterior")
        ),
        data={"median": med, "interval": (lo, hi)},
    )


def check_reliability_monotone(lower: np.ndarray, upper: np.ndarray, tol: float = 1e-12) -> CheckResult:
    """Expected reliability must be non-increasing in the horizon :math:`n_F`."""
    d_lo = np.diff(np.asarray(lower, dtype=float))
    d_up = np.diff(np.asarray(upper, dtype=float))
    bad = int(np.count_nonzero(d_lo > tol) + np.count_nonzero(d_up > tol))
    return CheckResult(
        name="E[R(n_F)] non-increasing in n_F",
        passed=bad == 0,
        detail=f"{bad} increasing step(s)" if bad else "monotone non-increasing on both envelopes",
    )


# --------------------------------------------------------------------------- #
# source-integrity checks
# --------------------------------------------------------------------------- #
def compare_accuracy_sources(
    table3: pd.DataFrame, repo: pd.DataFrame, atol: float = 1e-12
) -> pd.DataFrame:
    """Cell-by-cell comparison of paper Table 3 vs the repository figure numerics.

    Both frames must have columns ``model, domain, subdomain, theta_hat``.
    Returns a frame with absolute and relative differences and a ``match`` flag.
    """
    key = ["model", "domain", "subdomain"]
    for name, df in (("table3", table3), ("repo", repo)):
        missing = set(key + ["theta_hat"]) - set(df.columns)
        if missing:
            raise ValueError(f"{name} frame is missing column(s): {sorted(missing)}")

    merged = table3.merge(repo, on=key, how="outer", suffixes=("_table3", "_repo"))
    merged["abs_diff"] = (merged["theta_hat_table3"] - merged["theta_hat_repo"]).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["rel_diff"] = merged["abs_diff"] / merged["theta_hat_repo"].abs()
    merged["match"] = merged["abs_diff"] <= atol
    return merged.sort_values(key).reset_index(drop=True)


def check_table5_internal_consistency(
    table: pd.DataFrame, table_label: str | None = None
) -> list[CheckResult]:
    """Detect printed rows whose median envelope escapes its own 90% interval.

    Expected columns: ``op, method, median_lower, median_upper, ci_lower,
    ci_upper``, plus an optional ``table`` column used to label each check.

    For the paper's Table 5 this fails on the HIP-LLM rows under ``OP^approx``
    (median [0.5752, 0.5764] vs interval [0.5794, 0.5978]) and ``OP^GT``
    (median [0.5784, 0.5797] vs interval [0.5832, 0.6024]).  Both medians lie
    strictly *below* the reported 5% quantile, which no single posterior --
    and no min/max envelope over a family of posteriors -- can produce.  Applied
    to Table 4 every row passes, so the defect is specific to Table 5.
    """
    out: list[CheckResult] = []
    for _, row in table.iterrows():
        name = table_label or (row["table"] if "table" in row.index else "printed table")
        out.append(
            check_median_within_interval(
                (row["median_lower"], row["median_upper"]),
                (row["ci_lower"], row["ci_upper"]),
                label=f"{name} / {row['op']} / {row['method']}",
            )
        )
    return out


def check_fig9_against_sources(
    fig9: Mapping[str, float],
    table3: pd.DataFrame,
    repo: pd.DataFrame,
    model: str = "Sonnet 4.5",
    subdomain: str = "MBPP",
) -> list[CheckResult]:
    """Compare the Fig. 9 captioned Pass@1 value against both accuracy sources."""

    def _lookup(df: pd.DataFrame, label: str) -> float | None:
        sel = df[(df["model"] == model) & (df["subdomain"] == subdomain)]
        if sel.empty:
            return None
        return float(sel["theta_hat"].iloc[0])

    pass1 = float(fig9["pass1_accuracy"])
    results: list[CheckResult] = []
    for label, df in (("paper Table 3", table3), ("repository figure numerics", repo)):
        val = _lookup(df, label)
        if val is None:
            results.append(CheckResult(f"Fig. 9 Pass@1 vs {label}", False, "value not found"))
            continue
        diff = abs(pass1 - val)
        results.append(
            CheckResult(
                name=f"Fig. 9 Pass@1 ({model}/{subdomain}) agrees with {label}",
                passed=diff <= 1e-12,
                detail=f"caption {pass1:.4f} vs {label} {val:.4f} (|diff| = {diff:.4f})",
                data={"caption": pass1, "source": val, "abs_diff": diff},
            )
        )
    results.append(
        CheckResult(
            name="Fig. 9 N differs from the baseline effective N",
            passed=int(fig9["N"]) == 80,
            detail=f"Fig. 9 uses N = {int(fig9['N'])}; the baseline figures use N = 80",
        )
    )
    return results


def assert_no_overwrite(path: str | Path, expected_sha256: str) -> CheckResult:
    """Guard a reference dataset against silent modification."""
    p = Path(path)
    if not p.is_file():
        return CheckResult(f"integrity: {p.name}", False, "file missing")
    actual = sha256_file(p)
    return CheckResult(
        name=f"integrity: {p.name}",
        passed=actual == expected_sha256,
        detail=("checksum matches" if actual == expected_sha256 else f"expected {expected_sha256}, got {actual}"),
        data={"expected": expected_sha256, "actual": actual},
    )


# --------------------------------------------------------------------------- #
# environment / provenance
# --------------------------------------------------------------------------- #
def _git(args: Sequence[str], cwd: Path, strip: bool = True) -> str | None:
    """Run a git command, returning stdout or ``None`` if git is unavailable/fails.

    ``strip=False`` preserves leading whitespace, which matters for
    ``git status --porcelain``: its two status columns mean an unstaged change is
    reported as ``" M path"``, and stripping would shift the first line's path by
    one character.
    """
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() if strip else out.stdout.rstrip("\n")


def _porcelain_paths(status: str) -> list[str]:
    """Extract paths from ``git status --porcelain`` output.

    The format is two status columns, a space, then the path (quoted when it
    contains unusual characters).  Renames appear as ``old -> new``; the new path
    is what we report.
    """
    import re

    paths: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path = re.sub(r"^.{2}\s+", "", line)
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip().strip('"'))
    return paths


def environment_report(root: str | Path = ".") -> dict[str, Any]:
    """Capture everything needed to reproduce or audit this run."""
    root = Path(root).resolve()
    packages: dict[str, str] = {}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "yaml",
        "pytest",
        "papermill",
        "nbformat",
        "psutil",
        "openai",
        "anthropic",
        "datasets",
    ):
        try:
            mod = __import__(name)
            packages[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            packages[name] = "not installed"

    try:
        import psutil  # type: ignore

        ram_gb = round(psutil.virtual_memory().total / 2**30, 2)
        cpu_count = psutil.cpu_count(logical=True)
    except Exception:
        import os

        ram_gb = None
        cpu_count = os.cpu_count()

    commit = _git(["rev-parse", "HEAD"], root)
    dirty = _git(["status", "--porcelain"], root, strip=False)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count_logical": cpu_count,
        "ram_total_gb": ram_gb,
        "packages": packages,
        "git_commit": commit,
        "git_clean": (dirty.strip() == "") if dirty is not None else None,
        "git_modified_files": _porcelain_paths(dirty) if dirty else [],
        "paper_reported_environment": {
            "python": "3.12.12",
            "numpy": "2.0.2",
            "pandas": "2.2.2",
            "matplotlib": "3.10.0",
            "runtime": "Google Colab CPU, Intel Xeon @ 2.20 GHz, ~13 GB RAM, single process",
            "note": (
                "SciPy is required by this replication but is NOT among the packages the "
                "paper reports; it is an implementation dependency of the reproduction, "
                "not of the original experiment."
            ),
        },
    }


def build_provenance(entries: Iterable[SourceRecord]) -> dict[str, Any]:
    """Assemble the provenance manifest payload."""
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [e.as_dict() for e in entries],
    }
