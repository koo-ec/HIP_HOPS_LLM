"""Tests for explicit operational-profile failure inference and StrategyQA tools."""

from __future__ import annotations

import json

import numpy as np
import pytest

from HIPLLM import (
    OperationalFailureProb,
    StrategyQALoadError,
    decomposition_stratum,
    load_strategyqa,
    parse_strategyqa_answer,
)
from hip_llm.schemas import GlobalSettings, HyperparameterInterval


def _tiny_settings() -> GlobalSettings:
    return GlobalSettings(
        n_mu=4,
        n_nu=5,
        cdf_points_T=21,
        S=120,
        K_per_domain=4,
        max_llm_configuration_pairs=4,
        seed_global=7,
        seed_configs=123,
        seed_pairs=999,
        config_sampling="uniform_random",
        nu_grid_scheme="log",
        mu_grid_scheme="midpoint",
        nu_grid_params={"nu_lo": 1e-3, "nu_hi": 25.0},
        strict_exact=False,
    )


def test_operational_failure_uses_explicit_profile_and_labelled_outcomes() -> None:
    estimator = OperationalFailureProb(
        profile={"short": 0.25, "long": 0.75},
        interval=HyperparameterInterval(
            a=(2.0, 2.0), b=(2.0, 2.0), c=(3.0, 3.0), d=(2.0, 2.0)
        ),
        settings=_tiny_settings(),
    )
    # short: 3/4 correct -> 0.25 failure; long: 1/4 correct -> 0.75 failure
    result = estimator.fit(
        outcomes=[1, 1, 1, 0, 1, 0, 0, 0],
        strata=["short"] * 4 + ["long"] * 4,
    )

    assert result.empirical_operational_failure_probability == pytest.approx(0.625)
    assert result.failure_samples.shape == (4, 120)
    assert np.all((result.failure_samples >= 0.0) & (result.failure_samples <= 1.0))

    summary = result.summary()
    assert summary["operational_profile"] == {"short": 0.25, "long": 0.75}
    assert 0.0 <= summary["posterior_expected_failure_lower"] <= 1.0
    assert 0.0 <= summary["posterior_expected_failure_upper"] <= 1.0
    assert (
        summary["posterior_expected_failure_lower"]
        <= summary["posterior_expected_failure_upper"]
    )

    frame = result.to_df()
    assert frame["stratum"].tolist() == ["short", "long"]
    assert frame["operational_weight"].tolist() == pytest.approx([0.25, 0.75])
    assert frame["successes"].tolist() == [3, 1]
    assert frame["trials"].tolist() == [4, 4]


def test_operational_failure_rejects_invalid_or_unsupported_profiles() -> None:
    with pytest.raises(ValueError, match="weights sum"):
        OperationalFailureProb({"short": 0.3, "long": 0.3})

    estimator = OperationalFailureProb(
        {"short": 0.5, "long": 0.5}, settings=_tiny_settings()
    )
    with pytest.raises(ValueError, match="absent from the operational profile"):
        estimator.fit([1, 0], ["short", "unknown"])
    with pytest.raises(ValueError, match="has no benchmark observations"):
        estimator.fit([1, 0], ["short", "short"])
    with pytest.raises(ValueError, match="binary"):
        estimator.fit([1, 2], ["short", "long"])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Answer: YES", True),
        ("Final answer - no", False),
        ("true", True),
        ("False.", False),
        ("I cannot determine this", None),
        ("", None),
    ],
)
def test_strategyqa_parser_is_deterministic(text: str, expected: bool | None) -> None:
    assert parse_strategyqa_answer(text) is expected
    assert parse_strategyqa_answer(text) is parse_strategyqa_answer(text)


def test_strategyqa_strata_are_explicit_workload_categories() -> None:
    assert decomposition_stratum({"decomposition": ["a", "b"]}) == "short"
    assert decomposition_stratum({"decomposition": ["a", "b", "c"]}) == "medium"
    assert decomposition_stratum({"decomposition": ["a", "b", "c", "d"]}) == "long"


def test_strategyqa_loader_validates_local_official_shape(tmp_path) -> None:
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            [
                {
                    "qid": "q1",
                    "question": "Is water wet?",
                    "answer": True,
                    "decomposition": ["What is water?", "What does wet mean?"],
                }
            ]
        ),
        encoding="utf-8",
    )
    rows = load_strategyqa("dev", local_path=path)
    assert rows[0]["qid"] == "q1"
    assert rows[0]["answer"] is True

    path.write_text(json.dumps([{"qid": "q1"}]), encoding="utf-8")
    with pytest.raises(StrategyQALoadError, match="missing fields"):
        load_strategyqa("dev", local_path=path)


def test_token_heuristic_metadata_says_it_does_not_use_an_op() -> None:
    import asyncio
    import math

    from HIPLLM import FailureProb

    class Response:
        content = "yes"
        response_metadata = {
            "logprobs": {"content": [{"token": "yes", "logprob": math.log(0.8)}]}
        }

    class LLM:
        model_name = "fake"
        logprobs = False

        async def ainvoke(self, messages):
            return Response()

    result = asyncio.run(FailureProb(LLM()).generate_and_score(["Question?"]))
    assert result.metadata["estimate_type"] == "token_confidence_heuristic"
    assert result.metadata["uses_operational_profile"] is False
    assert result.metadata["calibration_required_for_error_probability"] is True
