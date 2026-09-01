"""Architecture extraction: a LangGraph application as an analysable system model.

The first HiP-HOPS phase is *model the system*: components with ports, and the
connections between them.  Here that model is read out of the object a LangGraph
notebook already renders --- ``graph.get_graph()`` --- rather than being drawn by
hand, so the analysis cannot drift away from the code it describes.

Agent graphs contain feedback (ReAct loops); fault trees cannot.  :mod:`.acyclic`
unrolls each loop to a stated depth and closes it with a *feedback-cut* component,
which keeps the loop's contribution in the tree instead of silently deleting it.
"""

from __future__ import annotations

from .acyclic import (
    CycleReport,
    find_back_edges,
    find_cycles,
    is_acyclic,
    make_acyclic,
)
from .extract import LangGraphExtractor
from .model import (
    Component,
    Connection,

    RawGraph,
    Role,
    SystemModel,
    build_system_model,
    classify_role,
    detect_resources,
    extract_architecture,
    parse_mermaid,
    raw_from_spec,
    source_of_function,
)

__all__ = [
    "Component",
    "Connection",
    "CycleReport",
    "LangGraphExtractor",

    "RawGraph",
    "Role",
    "SystemModel",
    "build_system_model",
    "classify_role",
    "detect_resources",
    "extract_architecture",
    "find_back_edges",
    "find_cycles",
    "is_acyclic",
    "make_acyclic",
    "parse_mermaid",
    "raw_from_spec",
    "source_of_function",
]
