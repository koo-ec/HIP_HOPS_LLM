"""Real benchmark evaluation: MBPP, DS-1000, BoolQ, RACE-H.

Every evaluator here is either the benchmark's own official harness or a
deterministic string parser.  **No LLM is ever used as a judge**, which the
specification forbids and which would in any case make the "perfect test oracle"
assumption of paper Remark 4 unverifiable.

Generated code is executed only inside a sandbox:

* preferred -- a Docker container with ``--network=none``, a read-only rootfs,
  a memory cap and a wall-clock cap;
* fallback  -- a subprocess in a scratch working directory with a scrubbed
  environment (no ``*_API_KEY``, no ``AWS_*`` etc.), a socket-blocking preamble
  and a wall-clock cap.

The fallback is weaker than container isolation and must be opted into
explicitly; it is refused by default.  Failure modes (compile error, runtime
error, timeout, failed assertion) are recorded separately, as required.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "FailureKind",
    "ExecutionOutcome",
    "SandboxPolicy",
    "SandboxRefusedError",
    "run_python_sandboxed",
    "evaluate_mbpp_task",
    "parse_boolq_answer",
    "parse_race_answer",
    "pass_at_k",
    "BenchmarkLoadError",
    "load_benchmark_split",
    "load_accuracy_table",
    "accuracy_to_counts",
    "build_model_from_accuracies",
]


class FailureKind(str, Enum):
    """Failure taxonomy recorded for every executed code task."""

    NONE = "none"
    COMPILE_ERROR = "compile_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    ASSERTION_FAILED = "assertion_failed"
    SANDBOX_ERROR = "sandbox_error"
    UNPARSEABLE = "unparseable_response"


@dataclass(frozen=True)
class ExecutionOutcome:
    """Result of executing one generated solution against its tests."""

    passed: bool
    failure: FailureKind
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    detail: str = ""


@dataclass(frozen=True)
class SandboxPolicy:
    """Resource and isolation limits applied to every generated-code execution."""

    backend: str = "docker"                 # "docker" | "subprocess"
    docker_image: str = "python:3.12-slim"
    wall_time_s: float = 15.0
    memory_mb: int = 512
    cpus: float = 1.0
    allow_unsafe_subprocess: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ("docker", "subprocess"):
            raise ValueError("backend must be 'docker' or 'subprocess'")
        if self.backend == "subprocess" and not self.allow_unsafe_subprocess:
            raise SandboxRefusedError(
                "backend='subprocess' provides weaker isolation than a container: it cannot "
                "enforce a hard memory cap on every platform and relies on a Python-level "
                "network block. Set allow_unsafe_subprocess=true only if you accept that, "
                "or install Docker and use backend='docker'."
            )


class SandboxRefusedError(RuntimeError):
    """Raised rather than executing untrusted code under inadequate isolation."""


class BenchmarkLoadError(RuntimeError):
    """The official benchmark split could not be loaded."""


# Environment variables that must never reach generated code.
_ENV_DENY_PATTERNS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
    "AWS_", "AZURE_", "GOOGLE_", "GH_", "GITHUB_", "SSH_", "OPENAI", "ANTHROPIC",
)

# Prepended to every executed script in the subprocess backend.
_NETWORK_BLOCK = """
import socket as _s
class _Blocked(OSError):
    pass
def _deny(*a, **k):
    raise _Blocked('network access is disabled in the evaluation sandbox')
_s.socket = _deny
_s.create_connection = _deny
_s.socketpair = _deny
try:
    import urllib.request as _u
    _u.urlopen = _deny
except Exception:
    pass
"""


def _scrubbed_env() -> dict[str, str]:
    keep = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if any(p in upper for p in _ENV_DENY_PATTERNS):
            continue
        keep[k] = v
    keep["PYTHONDONTWRITEBYTECODE"] = "1"
    keep["HOME"] = keep.get("TEMP", keep.get("TMPDIR", "."))
    keep["NO_PROXY"] = "*"
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        keep.pop(var, None)
    return keep


def _classify(stderr: str, returncode: int, timed_out: bool) -> FailureKind:
    if timed_out:
        return FailureKind.TIMEOUT
    if returncode == 0:
        return FailureKind.NONE
    if "SyntaxError" in stderr or "IndentationError" in stderr or "TabError" in stderr:
        return FailureKind.COMPILE_ERROR
    if "AssertionError" in stderr:
        return FailureKind.ASSERTION_FAILED
    return FailureKind.RUNTIME_ERROR


def run_python_sandboxed(script: str, policy: SandboxPolicy) -> ExecutionOutcome:
    """Execute ``script`` under the configured sandbox and classify the outcome."""
    import time

    workdir = Path(tempfile.mkdtemp(prefix="hipllm_sbx_"))
    try:
        src = workdir / "solution_under_test.py"
        if policy.backend == "subprocess":
            src.write_text(_NETWORK_BLOCK + "\n" + script, encoding="utf-8")
            cmd = [sys.executable, "-I", "-S", "-B", str(src)]
            env = _scrubbed_env()
        else:
            src.write_text(script, encoding="utf-8")
            if shutil.which("docker") is None:
                raise SandboxRefusedError(
                    "backend='docker' was requested but the docker executable was not found."
                )
            cmd = [
                "docker", "run", "--rm",
                "--network=none",
                "--read-only",
                f"--memory={policy.memory_mb}m",
                f"--memory-swap={policy.memory_mb}m",
                f"--cpus={policy.cpus}",
                "--pids-limit=128",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--tmpfs", "/tmp:rw,size=64m,noexec",
                "-v", f"{workdir}:/work:ro",
                "-w", "/work",
                policy.docker_image,
                "python", "-I", "-S", "-B", "solution_under_test.py",
            ]
            env = _scrubbed_env()

        t0 = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=policy.wall_time_s, env=env, cwd=str(workdir),
            )
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            rc, out, err = -1, (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""), ""
        duration = time.perf_counter() - t0

        kind = _classify(err, rc, timed_out)
        return ExecutionOutcome(
            passed=(kind is FailureKind.NONE),
            failure=kind,
            stdout=out[-4000:],
            stderr=err[-4000:],
            duration_s=duration,
            detail=f"returncode={rc}",
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# MBPP  (unit-test based)
# --------------------------------------------------------------------------- #
_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_python_code(response: str) -> str | None:
    """Deterministically extract a Python program from a model response."""
    blocks = _CODE_FENCE.findall(response or "")
    if blocks:
        return max(blocks, key=len).strip()
    text = (response or "").strip()
    if not text:
        return None
    # A bare program is accepted only if it plausibly contains a definition.
    if re.search(r"^\s*(def|class|import|from)\s", text, re.MULTILINE):
        return text
    return None


def evaluate_mbpp_task(
    response: str, test_list: Sequence[str], policy: SandboxPolicy, setup_code: str = ""
) -> ExecutionOutcome:
    """Run the task's official ``assert`` tests against the generated solution."""
    code = extract_python_code(response)
    if code is None:
        return ExecutionOutcome(
            passed=False, failure=FailureKind.UNPARSEABLE,
            detail="no Python code block found in the response",
        )
    script = "\n".join([code, "", setup_code, "", *test_list, "", "print('__ALL_TESTS_PASSED__')"])
    return run_python_sandboxed(script, policy)


def pass_at_k(outcomes: Sequence[bool], k: int) -> bool:
    """``Pass@k``: success iff at least one of the first ``k`` generations passes.

    ``Pass@1`` is therefore success on the **first** generation only, exactly as
    the paper's footnote 20 defines it.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(outcomes) < k:
        raise ValueError(f"pass@{k} needs {k} generations, got {len(outcomes)}")
    return any(outcomes[:k])


# --------------------------------------------------------------------------- #
# BoolQ / RACE-H  (deterministic parsing, no judge model)
# --------------------------------------------------------------------------- #
_TRUE_TOKENS = ("yes", "true")
_FALSE_TOKENS = ("no", "false")


def parse_boolq_answer(response: str) -> bool | None:
    """Deterministically parse a yes/no answer.  Returns ``None`` if unparseable.

    Priority order: an explicit ``Answer:`` line, then the first standalone
    yes/no token.  Never asks another model.
    """
    text = (response or "").strip().lower()
    if not text:
        return None
    m = re.search(r"answer\s*[:\-]\s*\**\s*(yes|no|true|false)\b", text)
    if m:
        return m.group(1) in _TRUE_TOKENS
    m = re.search(r"\b(yes|no|true|false)\b", text)
    if m:
        return m.group(1) in _TRUE_TOKENS
    return None


def parse_race_answer(response: str, n_options: int = 4) -> int | None:
    """Deterministically parse a multiple-choice letter into a 0-based index."""
    if not (1 <= n_options <= 26):
        raise ValueError("n_options must be between 1 and 26")
    letters = "".join(chr(ord("A") + i) for i in range(n_options))
    text = (response or "").strip()
    if not text:
        return None
    m = re.search(rf"answer\s*[:\-]\s*\**\s*\(?([{letters}{letters.lower()}])\)?\b", text, re.IGNORECASE)
    if m:
        return ord(m.group(1).upper()) - ord("A")
    m = re.match(rf"^\(?([{letters}{letters.lower()}])\)?[\.\)\s]", text)
    if m:
        return ord(m.group(1).upper()) - ord("A")
    m = re.search(rf"\b([{letters}])\b", text)
    if m:
        return ord(m.group(1).upper()) - ord("A")
    return None


# --------------------------------------------------------------------------- #
# dataset loading
# --------------------------------------------------------------------------- #
_SUPPORTED = {
    "MBPP": ("google-research-datasets/mbpp", "full"),
    "DS-1000": ("xlangai/DS-1000", None),
    "BoolQ": ("google/boolq", None),
    "RACE-H": ("ehovy/race", "high"),
}


def load_benchmark_split(
    name: str,
    split: str,
    item_ids: Sequence[str] | None = None,
    local_path: str | Path | None = None,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    """Load the official split, optionally restricted to explicit item ids.

    The HIP-LLM paper does **not** state which split or which item subset it
    evaluated (only the resulting accuracies), so ``split`` and ``item_ids`` are
    required inputs of the live-API config rather than defaults chosen here.
    A missing dataset raises :class:`BenchmarkLoadError`; it is never replaced by
    a stand-in.
    """
    if name not in _SUPPORTED:
        raise BenchmarkLoadError(f"unknown benchmark {name!r}; expected one of {sorted(_SUPPORTED)}")

    if local_path is not None:
        p = Path(local_path)
        if not p.is_file():
            raise BenchmarkLoadError(f"local benchmark file not found: {p}")
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise BenchmarkLoadError(
                "the 'datasets' package is required to load official benchmark splits; "
                "install it or supply local_path to a JSONL export."
            ) from exc
        path, config = _SUPPORTED[name]
        try:
            ds = load_dataset(path, config, split=split, revision=revision)
        except Exception as exc:  # noqa: BLE001
            raise BenchmarkLoadError(f"failed to load {name} ({path}, split={split}): {exc}") from exc
        rows = [dict(r) for r in ds]

    if item_ids is not None:
        wanted = set(map(str, item_ids))
        id_field = next(
            (f for f in ("task_id", "id", "problem_id", "example_id") if rows and f in rows[0]),
            None,
        )
        if id_field is None:
            raise BenchmarkLoadError(f"{name}: no recognised id field to filter item_ids on")
        rows = [r for r in rows if str(r[id_field]) in wanted]
        if len(rows) != len(wanted):
            missing = wanted - {str(r[id_field]) for r in rows}
            raise BenchmarkLoadError(
                f"{name}: {len(missing)} requested item id(s) not present in split {split!r}: "
                f"{sorted(missing)[:10]}"
            )
    return rows


# --------------------------------------------------------------------------- #
# published-measurement ingestion (Mode A)
# --------------------------------------------------------------------------- #
def load_accuracy_table(path: str | Path, expected_sha256: str | None = None):
    """Load a published accuracy table and optionally verify its checksum.

    Expected schema: ``model, domain, subdomain, theta_hat``.  Verifying the
    checksum is how the reproduction guarantees that a reference dataset was not
    silently modified between runs.
    """
    import pandas as pd  # local import: pandas is not needed by the sandbox path

    from .schemas import sha256_file

    p = Path(path)
    if not p.is_file():
        raise BenchmarkLoadError(f"accuracy table not found: {p}")
    if expected_sha256 is not None:
        actual = sha256_file(p)
        if actual != expected_sha256:
            raise BenchmarkLoadError(
                f"{p.name}: checksum mismatch (expected {expected_sha256}, got {actual}). "
                f"Refusing to proceed with a modified reference dataset."
            )
    df = pd.read_csv(p, comment="#")
    required = {"model", "domain", "subdomain", "theta_hat"}
    missing = required - set(df.columns)
    if missing:
        raise BenchmarkLoadError(f"{p.name}: missing column(s) {sorted(missing)}")
    bad = df[(df["theta_hat"] < 0) | (df["theta_hat"] > 1) | df["theta_hat"].isna()]
    if not bad.empty:
        raise BenchmarkLoadError(f"{p.name}: theta_hat outside [0,1] in {len(bad)} row(s)")
    return df


def accuracy_to_counts(theta_hat: float, N: int) -> int:
    """``C_ij = round(theta_hat_ij * N_ij)`` -- the repository's stated rule.

    ``numerics/README.md``: "theta_hat in [0,1] is the point accuracy used to
    form counts C_ij = round(theta_hat_ij * N_ij) with N_ij = 80."  Uses
    round-half-away-from-zero so that e.g. ``0.5 * 80 = 40.0`` and
    ``0.94375 * 80 = 75.5 -> 76``, rather than NumPy's banker's rounding.
    """
    if not (0.0 <= theta_hat <= 1.0):
        raise ValueError(f"theta_hat must lie in [0,1], got {theta_hat}")
    if N <= 0:
        raise ValueError("N must be positive")
    from decimal import ROUND_HALF_UP, Decimal

    c = int(Decimal(str(theta_hat * N)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return max(0, min(N, c))


def build_model_from_accuracies(
    df,
    model: str,
    hierarchy: Mapping[str, Sequence[str]],
    omega: Mapping[str, Sequence[float]],
    W: Sequence[float],
    N_per_subdomain: int,
    source_label: str,
    overrides: Mapping[str, tuple[int, int]] | None = None,
):
    """Assemble a :class:`~hip_llm.schemas.ModelResult` from an accuracy table.

    ``overrides`` maps a subdomain name to an explicit ``(successes, trials)``
    pair, which is how the Fig. 9 experiment substitutes its ``N = 257`` MBPP
    counts and how RQ7 injects its stress-test ``theta_BoolQ`` values, without
    disturbing any other subdomain.
    """
    import numpy as np

    from .schemas import DomainData, ModelResult, SubdomainData

    overrides = dict(overrides or {})
    sub = df[df["model"] == model]
    if sub.empty:
        raise BenchmarkLoadError(f"model {model!r} not present in the accuracy table")

    domains = []
    for dname, subnames in hierarchy.items():
        records = []
        for sname in subnames:
            if sname in overrides:
                C, N = overrides[sname]
                acc = None
            else:
                row = sub[(sub["domain"] == dname) & (sub["subdomain"] == sname)]
                if row.empty:
                    raise BenchmarkLoadError(
                        f"{model}: no accuracy for {dname}/{sname} in the table"
                    )
                acc = float(row["theta_hat"].iloc[0])
                N = N_per_subdomain
                C = accuracy_to_counts(acc, N)
            records.append(SubdomainData(name=sname, successes=int(C), trials=int(N), source_accuracy=acc))
        domains.append(DomainData(dname, tuple(records), np.asarray(omega[dname], dtype=float)))

    return ModelResult(model, tuple(domains), np.asarray(W, dtype=float), source_label)
