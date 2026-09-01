"""Utilities for reproducible StrategyQA evaluation.

The loader reads the JSON files published by the official StrategyQA
repository. Answers are scored with a deterministic yes/no parser; a second
language model is never used as a judge.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = [
    "StrategyQALoadError",
    "decomposition_stratum",
    "load_strategyqa",
    "parse_strategyqa_answer",
]

_OFFICIAL_RAW_ROOT = (
    "https://raw.githubusercontent.com/eladsegal/strategyqa/"
    "{revision}/data/strategyqa/{split}.json"
)
_ALLOWED_SPLITS = frozenset({"train", "dev"})
_REVISION_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:final\s+)?answer\s*[:\-]\s*\**\s*(yes|no|true|false)\b",
    re.IGNORECASE,
)
_STANDALONE_ANSWER_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
_TRUE_TOKENS = frozenset({"yes", "true"})


class StrategyQALoadError(RuntimeError):
    """Raised when the official StrategyQA data cannot be loaded or validated."""


def parse_strategyqa_answer(text: str) -> bool | None:
    """Parse a generated yes/no answer without using an LLM judge.

    An explicit ``Answer: yes`` or ``Answer: no`` form takes priority. Otherwise,
    the first standalone yes/no/true/false token is used. ``None`` means that the
    response is not safely parseable and should be handled according to an
    explicitly chosen evaluation policy.
    """
    value = (text or "").strip()
    if not value:
        return None
    match = _EXPLICIT_ANSWER_RE.search(value) or _STANDALONE_ANSWER_RE.search(value)
    if match is None:
        return None
    return match.group(1).lower() in _TRUE_TOKENS


def decomposition_stratum(
    item: Mapping[str, Any],
    *,
    short_max: int = 2,
    medium_max: int = 3,
) -> str:
    """Map a StrategyQA item to a transparent decomposition-length stratum.

    The default strata are ``short`` for at most two decomposition steps,
    ``medium`` for three steps, and ``long`` for four or more steps. These labels
    are a workload-design choice, not an official StrategyQA category.
    """
    if short_max < 0 or medium_max < short_max:
        raise ValueError("require 0 <= short_max <= medium_max")
    decomposition = item.get("decomposition")
    if not isinstance(decomposition, Sequence) or isinstance(decomposition, (str, bytes)):
        raise ValueError("StrategyQA item has no valid decomposition sequence")
    n_steps = len(decomposition)
    if n_steps <= short_max:
        return "short"
    if n_steps <= medium_max:
        return "medium"
    return "long"


def _validate_rows(payload: Any, *, require_answers: bool) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise StrategyQALoadError("StrategyQA payload must be a non-empty JSON array")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = {"qid", "question", "decomposition"}
    if require_answers:
        required.add("answer")

    for index, value in enumerate(payload):
        if not isinstance(value, Mapping):
            raise StrategyQALoadError(f"StrategyQA row {index} is not a JSON object")
        row = dict(value)
        missing = sorted(required - set(row))
        if missing:
            raise StrategyQALoadError(f"StrategyQA row {index} is missing fields {missing}")
        qid = str(row["qid"])
        if not qid or qid in seen:
            raise StrategyQALoadError(f"StrategyQA row {index} has an empty or duplicate qid")
        seen.add(qid)
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise StrategyQALoadError(f"StrategyQA row {index} has an invalid question")
        if not isinstance(row["decomposition"], list):
            raise StrategyQALoadError(f"StrategyQA row {index} has an invalid decomposition")
        if require_answers and not isinstance(row["answer"], bool):
            raise StrategyQALoadError(f"StrategyQA row {index} has a non-boolean answer")
        rows.append(row)
    return rows


def load_strategyqa(
    split: str = "dev",
    *,
    local_path: str | Path | None = None,
    revision: str = "main",
    timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    """Load and validate an official labelled StrategyQA split.

    Parameters
    ----------
    split:
        ``"train"`` or ``"dev"``. The labelled development split is the normal
        choice for the Colab evaluation.
    local_path:
        Optional path to a previously downloaded JSON array. When supplied, no
        network request is made.
    revision:
        Official repository branch, tag or commit. Use a commit SHA when exact
        data provenance is required; ``"main"`` follows the latest repository
        version and is therefore not immutable.
    timeout_seconds:
        Network timeout used only when ``local_path`` is absent.
    """
    if split not in _ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(_ALLOWED_SPLITS)}, got {split!r}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    if local_path is not None:
        path = Path(local_path)
        if not path.is_file():
            raise StrategyQALoadError(f"StrategyQA file not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StrategyQALoadError(f"failed to read StrategyQA JSON from {path}: {exc}") from exc
    else:
        if not revision or not _REVISION_RE.fullmatch(revision) or ".." in revision:
            raise ValueError("revision contains unsupported characters")
        url = _OFFICIAL_RAW_ROOT.format(revision=revision, split=split)
        request = Request(url, headers={"User-Agent": "HIPLLM-StrategyQA-loader/1"})
        try:
            with urlopen(request, timeout=float(timeout_seconds)) as response:  # noqa: S310
                raw = response.read()
            payload = json.loads(raw.decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StrategyQALoadError(
                f"failed to load official StrategyQA split={split!r}, revision={revision!r}: {exc}"
            ) from exc

    return _validate_rows(payload, require_answers=True)
