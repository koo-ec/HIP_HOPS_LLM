"""Result containers for the high-level HIPLLM API."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import pandas as pd


class FailureProbResult:
    """Tabular prompt-level scores returned by :class:`FailureProb`.

    Parameters
    ----------
    data:
        Equal-length columns containing prompts, responses and scores.
    metadata:
        Information about the scorer configuration and language model.
    """

    def __init__(self, data: Mapping[str, Sequence[Any]], metadata: Mapping[str, Any]) -> None:
        lengths = {len(values) for values in data.values()}
        if len(lengths) > 1:
            raise ValueError("all result columns must have the same length")
        self.data = {key: list(values) for key, values in data.items()}
        self.metadata = dict(metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive dictionary copy of the complete result."""
        return deepcopy({"data": self.data, "metadata": self.metadata})

    def to_df(self) -> pd.DataFrame:
        """Return one row per prompt as a pandas DataFrame."""
        return pd.DataFrame(deepcopy(self.data))
