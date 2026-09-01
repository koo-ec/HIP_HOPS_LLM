"""LangChain-compatible prompt-level token-confidence scoring.

This module does not implement the HIP-LLM operational-profile calculation.
Use :class:`HIPLLM.OperationalFailureProb` with labelled benchmark outcomes for
an operational-profile-weighted probability of failure.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .results import FailureProbResult

SUPPORTED_SCORERS = ("min_probability", "sequence_probability")


class LogprobsUnavailableError(RuntimeError):
    """Raised when an LLM response does not expose token log probabilities."""


def _candidate_logprob(candidate: Mapping[str, Any]) -> float:
    value = candidate.get("logprob", candidate.get("log_probability"))
    if value is None:
        raise LogprobsUnavailableError("a token entry did not contain a log probability")
    return float(value)


def _extract_token_logprobs(response: Any) -> np.ndarray:
    """Normalise OpenAI- and Google-style LangChain response metadata."""
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        metadata = {}

    entries: Any = None
    google_result = metadata.get("logprobs_result")
    if isinstance(google_result, Mapping):
        entries = google_result.get("chosen_candidates")
    elif isinstance(google_result, Sequence) and not isinstance(google_result, (str, bytes)):
        entries = google_result

    if entries is None:
        raw = metadata.get("logprobs")
        if isinstance(raw, Mapping):
            if isinstance(raw.get("content"), Sequence):
                entries = raw["content"]
            elif isinstance(raw.get("token_logprobs"), Sequence):
                entries = [{"logprob": value} for value in raw["token_logprobs"]]

    if not entries:
        raise LogprobsUnavailableError(
            "The model response did not include token log probabilities. Configure the "
            "LangChain chat model with logprobs enabled and use a model/provider that "
            "supports them (for ChatVertexAI, HIPLLM sets llm.logprobs=True)."
        )

    try:
        values = np.asarray([_candidate_logprob(entry) for entry in entries], dtype=float)
    except (TypeError, ValueError) as exc:
        raise LogprobsUnavailableError(
            "The model returned malformed token log probabilities"
        ) from exc
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise LogprobsUnavailableError("The model returned empty or non-finite log probabilities")
    if np.any(values > 1e-9):
        raise LogprobsUnavailableError(
            "The model returned a positive log probability, which would imply a probability above 1"
        )
    return values


class FailureProb:
    """Generate responses and calculate prompt-level token-confidence heuristics.

    The supported confidence scorers are ``min_probability`` (the least likely
    generated token) and ``sequence_probability`` (the geometric mean token
    probability). Their transformed score is ``1 - confidence``.

    This class does **not** use an operational profile and its transformed score
    is not, by itself, a calibrated probability that the answer is wrong. For a
    benchmark-level probability of failure under an explicit workload profile,
    collect binary correctness outcomes and use
    :class:`HIPLLM.OperationalFailureProb`.

    Parameters
    ----------
    llm:
        A LangChain-compatible chat model exposing ``ainvoke`` and token logprobs.
    scorers:
        Any subset of ``min_probability`` and ``sequence_probability``. The default
        is ``["min_probability"]``.
    system_prompt:
        Optional system instruction prepended to string prompts.
    max_concurrency:
        Maximum number of in-flight model requests.
    """

    def __init__(
        self,
        llm: Any,
        scorers: Sequence[str] | None = None,
        system_prompt: str | None = None,
        max_concurrency: int = 5,
    ) -> None:
        if llm is None or not callable(getattr(llm, "ainvoke", None)):
            raise TypeError("llm must provide an async ainvoke method")
        selected = list(scorers or ["min_probability"])
        unsupported = [name for name in selected if name not in SUPPORTED_SCORERS]
        if unsupported:
            raise ValueError(
                f"Unsupported scorer(s): {unsupported}. Supported scorers: {list(SUPPORTED_SCORERS)}"
            )
        if len(set(selected)) != len(selected):
            raise ValueError("scorers must not contain duplicates")
        if not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")

        self.llm = llm
        self.scorers = selected
        self.system_prompt = system_prompt
        self.max_concurrency = max_concurrency

    def _messages(self, prompt: Any) -> Any:
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("prompts must not contain empty strings")
            messages: list[tuple[str, str]] = []
            if self.system_prompt:
                messages.append(("system", self.system_prompt))
            messages.append(("human", prompt))
            return messages
        if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)) and prompt:
            return list(prompt)
        raise TypeError("each prompt must be a non-empty string or a non-empty message sequence")

    async def generate_and_score(self, prompts: Sequence[Any]) -> FailureProbResult:
        """Generate one response per prompt and calculate the selected heuristics."""
        if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)) or not prompts:
            raise ValueError("prompts must be a non-empty sequence")

        if hasattr(self.llm, "logprobs"):
            self.llm.logprobs = True

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def invoke(prompt: Any) -> Any:
            async with semaphore:
                return await self.llm.ainvoke(self._messages(prompt))

        responses = await asyncio.gather(*(invoke(prompt) for prompt in prompts))
        token_logprobs = [_extract_token_logprobs(response) for response in responses]

        data: dict[str, list[Any]] = {
            "prompt": list(prompts),
            "response": [str(getattr(response, "content", response)) for response in responses],
        }
        confidence: dict[str, list[float]] = {}
        if "min_probability" in self.scorers:
            confidence["min_probability"] = [
                float(math.exp(values.min())) for values in token_logprobs
            ]
        if "sequence_probability" in self.scorers:
            confidence["sequence_probability"] = [
                float(math.exp(values.mean())) for values in token_logprobs
            ]

        for name in self.scorers:
            data[name] = confidence[name]
            failure = [1.0 - value for value in confidence[name]]
            if len(self.scorers) == 1:
                data["failure_probability"] = failure
            else:
                data[f"{name}_failure_probability"] = failure

        model_name = (
            getattr(self.llm, "model_name", None)
            or getattr(self.llm, "model", None)
            or type(self.llm).__name__
        )
        return FailureProbResult(
            data=data,
            metadata={
                "model": str(model_name),
                "scorers": list(self.scorers),
                "failure_transform": "1 - confidence",
                "estimate_type": "token_confidence_heuristic",
                "uses_operational_profile": False,
                "calibration_required_for_error_probability": True,
            },
        )
