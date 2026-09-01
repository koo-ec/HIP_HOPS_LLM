"""Public API tests for prompt-level failure-probability scoring."""

from __future__ import annotations

import asyncio
import math

import pytest


class FakeResponse:
    def __init__(self, content: str, response_metadata: dict) -> None:
        self.content = content
        self.response_metadata = response_metadata


class FakeLLM:
    def __init__(self, responses: list[FakeResponse], model_name: str = "fake-model") -> None:
        self._responses = iter(responses)
        self.model_name = model_name
        self.temperature = 0.0
        self.logprobs = False
        self.calls: list[object] = []

    async def ainvoke(self, messages: object) -> FakeResponse:
        self.calls.append(messages)
        return next(self._responses)


def test_generate_and_score_returns_failure_probabilities_and_dataframe() -> None:
    from HIPLLM import FailureProb

    responses = [
        FakeResponse(
            "Paris",
            {"logprobs": {"content": [{"token": "Paris", "logprob": math.log(0.8)}]}},
        ),
        FakeResponse(
            "Four",
            {
                "logprobs": {
                    "content": [
                        {"token": "F", "logprob": math.log(0.9)},
                        {"token": "our", "logprob": math.log(0.6)},
                    ]
                }
            },
        ),
    ]
    llm = FakeLLM(responses)
    scorer = FailureProb(llm=llm, scorers=["min_probability"])

    result = asyncio.run(scorer.generate_and_score(["Capital of France?", "2 + 2?"]))
    frame = result.to_df()

    assert frame["prompt"].tolist() == ["Capital of France?", "2 + 2?"]
    assert frame["response"].tolist() == ["Paris", "Four"]
    assert frame["min_probability"].tolist() == pytest.approx([0.8, 0.6])
    assert frame["failure_probability"].tolist() == pytest.approx([0.2, 0.4])
    assert result.to_dict()["metadata"]["scorers"] == ["min_probability"]
    assert llm.logprobs is True
    assert len(llm.calls) == 2


def test_google_vertex_logprobs_result_is_supported() -> None:
    from HIPLLM import FailureProb

    llm = FakeLLM(
        [
            FakeResponse(
                "answer",
                {
                    "logprobs_result": {
                        "chosen_candidates": [
                            {"token": "ans", "log_probability": math.log(0.75)},
                            {"token": "wer", "log_probability": math.log(0.5)},
                        ]
                    }
                },
            )
        ],
        model_name="gemini-2.5-pro",
    )

    result = asyncio.run(
        FailureProb(llm=llm, scorers=["min_probability"]).generate_and_score(["Question"])
    )

    assert result.data["min_probability"] == pytest.approx([0.5])
    assert result.data["failure_probability"] == pytest.approx([0.5])


def test_current_chat_vertex_logprobs_list_is_supported() -> None:
    from HIPLLM import FailureProb

    llm = FakeLLM(
        [
            FakeResponse(
                "answer",
                {
                    "logprobs_result": [
                        {"token": "ans", "logprob": math.log(0.7), "top_logprobs": []},
                        {"token": "wer", "logprob": math.log(0.4), "top_logprobs": []},
                    ]
                },
            )
        ],
        model_name="gemini-2.5-pro",
    )

    result = asyncio.run(FailureProb(llm=llm).generate_and_score(["Question"]))

    assert result.data["min_probability"] == pytest.approx([0.4])
    assert result.data["failure_probability"] == pytest.approx([0.6])


def test_sequence_probability_has_an_explicit_failure_column() -> None:
    from HIPLLM import FailureProb

    llm = FakeLLM(
        [
            FakeResponse(
                "AB",
                {
                    "logprobs": {
                        "content": [
                            {"token": "A", "logprob": math.log(0.8)},
                            {"token": "B", "logprob": math.log(0.2)},
                        ]
                    }
                },
            )
        ]
    )

    result = asyncio.run(
        FailureProb(llm=llm, scorers=["min_probability", "sequence_probability"])
        .generate_and_score(["Prompt"])
    )

    assert result.data["min_probability"] == pytest.approx([0.2])
    assert result.data["min_probability_failure_probability"] == pytest.approx([0.8])
    assert result.data["sequence_probability"] == pytest.approx([0.4])
    assert result.data["sequence_probability_failure_probability"] == pytest.approx([0.6])
    assert "failure_probability" not in result.data


def test_invalid_scorer_is_rejected() -> None:
    from HIPLLM import FailureProb

    with pytest.raises(ValueError, match="Unsupported scorer"):
        FailureProb(llm=FakeLLM([]), scorers=["unknown"])


def test_missing_logprobs_has_actionable_error() -> None:
    from HIPLLM import FailureProb, LogprobsUnavailableError

    llm = FakeLLM([FakeResponse("answer", {})])

    with pytest.raises(LogprobsUnavailableError, match="log probabilities"):
        asyncio.run(FailureProb(llm=llm).generate_and_score(["Prompt"]))


def test_positive_logprob_is_rejected_before_it_can_create_negative_failure() -> None:
    from HIPLLM import FailureProb, LogprobsUnavailableError

    llm = FakeLLM(
        [FakeResponse("answer", {"logprobs_result": [{"token": "x", "logprob": 0.1}]})]
    )

    with pytest.raises(LogprobsUnavailableError, match="positive"):
        asyncio.run(FailureProb(llm=llm).generate_and_score(["Prompt"]))


def test_empty_prompts_are_rejected() -> None:
    from HIPLLM import FailureProb

    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(FailureProb(llm=FakeLLM([])).generate_and_score([]))
