"""Bundled example systems, and reading/writing analysis artefacts."""

from __future__ import annotations

from .examples import (
    EXAMPLES,
    describe_examples,
    load_example,
    load_outcomes,
)
from .n8n import (
    N8nBlock,
    N8nWorkflow,
    analyse_n8n,
    load_n8n,
    n8n_study,
    n8n_to_spec,
)

__all__ = [
    "EXAMPLES",
    "describe_examples",
    "load_example",
    "load_outcomes",
    "N8nBlock",
    "N8nWorkflow",
    "analyse_n8n",
    "load_n8n",
    "n8n_study",
    "n8n_to_spec",
]
