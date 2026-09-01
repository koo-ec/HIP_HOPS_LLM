"""Example architectures and outcome data that ship with the package.

Every example in the documentation runs from these, so the docs are executable
without LangGraph, without a GPU and without a network connection.  The graph
specifications are the same architectures the source notebook analyses, recorded
as plain JSON; the outcome table is synthetic and says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Dict

__all__ = ["EXAMPLES", "ExampleSpec", "load_example", "load_outcomes", "describe_examples"]

_PACKAGE = "HIP_HOPS_LLM.data"


@dataclass(frozen=True)
class ExampleSpec:
    """One bundled architecture."""

    key: str
    filename: str
    title: str
    summary: str


EXAMPLES: Dict[str, ExampleSpec] = {
    "react_calculator": ExampleSpec(
        key="react_calculator",
        filename="react_calculator.json",
        title="ReAct + calculator",
        summary=(
            "One LLM agent, a calculator tool and a conditional router, with a "
            "generator->coder->generator feedback loop. Every cut set is of "
            "order 1: there is no redundancy anywhere in it."
        ),
    ),
    "parallel_aggregator": ExampleSpec(
        key="parallel_aggregator",
        filename="parallel_aggregator.json",
        title="parallel agents + aggregator",
        summary=(
            "Two agents answer in parallel and a judge selects between them. The "
            "vote does raise most cut sets to order 2 — but both agents run the "
            "same model snapshot, so the shared-snapshot common-cause event is "
            "still an order-1 cut set."
        ),
    ),
    "supervisor_workers": ExampleSpec(
        key="supervisor_workers",
        filename="supervisor_workers.json",
        title="supervisor + workers + verifier",
        summary=(
            "A supervisor routes to one of three specialists and a verifier "
            "checks the answer. Two specialists share a snapshot and one does "
            "not, so the effect of genuine model diversity is visible."
        ),
    ),
}


def load_example(key: str = "parallel_aggregator") -> Dict[str, Any]:
    """Return one bundled architecture as a graph specification.

    The result goes straight into anything that accepts a graph::

        from HIP_HOPS_LLM import AgenticReliabilityStudy, load_example

        study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
    """
    if key not in EXAMPLES:
        raise KeyError(
            f"unknown example {key!r}; available: {sorted(EXAMPLES)}"
        )
    text = (
        resources.files(_PACKAGE)
        .joinpath(EXAMPLES[key].filename)
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


def load_outcomes(as_frame: bool = True) -> Any:
    """Synthetic per-agent outcomes for the ``parallel_aggregator`` example.

    240 items labelled with a StrategyQA-shaped stratum (``short`` / ``medium`` /
    ``long``), one correctness column per agent, and a ``split`` column so the
    calibration/evaluation separation can be demonstrated.  ``1`` means the agent
    answered that item **correctly**.

    The data is generated, not measured: agent accuracy falls with decomposition
    length, the two agents' errors are correlated at 0.55 because they share a
    model snapshot, and the aggregator selects correctly 85% of the time.  Use it
    to learn the API, never as evidence about any real model.
    """
    handle = resources.files(_PACKAGE).joinpath("agent_outcomes.csv")
    if not as_frame:
        import csv
        import io as _io

        return list(csv.DictReader(_io.StringIO(handle.read_text(encoding="utf-8"))))
    import io as _io

    import pandas as pd

    frame = pd.read_csv(_io.StringIO(handle.read_text(encoding="utf-8")))
    for column in ("react_agent", "cot_agent", "aggregator"):
        frame[column] = frame[column].astype(int)
    return frame


def describe_examples() -> str:
    """A printable catalogue of what is bundled."""
    lines = ["Bundled examples", "=" * 16]
    for spec in EXAMPLES.values():
        lines.append(f"\n{spec.key}  —  {spec.title}")
        lines.append("  " + spec.summary)
    lines.append(
        "\nload_outcomes() returns 240 synthetic per-agent outcomes for "
        "'parallel_aggregator'."
    )
    return "\n".join(lines)
