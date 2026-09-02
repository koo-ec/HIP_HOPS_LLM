"""A configurable, reusable extractor for LangGraph architectures.

:func:`~hiphopsllm.architecture.model.extract_architecture` is a function that
takes ten keyword arguments.  That is fine for a single call in a notebook, but
awkward when the same conventions --- the same role overrides, the same shared
model snapshots, the same unroll depth --- have to be applied to several graphs
and compared.  :class:`LangGraphExtractor` holds those conventions as state and
applies them to any number of graphs::

    extractor = LangGraphExtractor(
        globals_ns=globals(),
        role_overrides={"coder": "tool"},
        unroll=2,
    )
    approach_1 = extractor.extract(graph_1, name="Approach 1")
    approach_2 = extractor.extract(graph_2, name="Approach 2")

The extractor also carries the loop-elimination step, so ``extract_acyclic``
returns the model the fault tree synthesiser can actually consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from .acyclic import CycleReport, make_acyclic
from .model import Role, SystemModel, extract_architecture

__all__ = ["LangGraphExtractor"]


@dataclass
class LangGraphExtractor:
    """Turn LangGraph objects into :class:`SystemModel` objects, under fixed conventions.

    Parameters
    ----------
    globals_ns
        Pass ``globals()`` from a notebook.  Node functions are then found by
        name and the *actual* model objects are interrogated, which is what makes
        shared-snapshot (common-cause) detection reliable.  Without it, role
        classification falls back to node names and edge topology alone.
    node_functions
        Explicit ``{node_id: function}`` mapping, for when the functions are not
        in ``globals_ns`` (a script, a class, an imported module).  Takes
        precedence over ``globals_ns``.
    role_overrides
        Force a component's archetype, e.g. ``{"verifier": Role.AGGREGATOR}``.
        Use this when a node's name and source do not reveal what it really is.
    resource_overrides
        Declare shared resources the source does not name, e.g.
        ``{"critic": {"llm": "gpt-4o-2024-11-20"}}``.  Components sharing a
        resource become a common-cause group, which is usually the difference
        between a redundant architecture and one that only looks redundant.
    unroll
        Iterations of each feedback loop represented explicitly (default 1).
    materialise_routers
        Turn ``add_conditional_edges`` into an explicit router component with its
        own failure logic, rather than an anonymous branch (default ``True``).
    """

    globals_ns: Optional[Dict[str, Any]] = None
    node_functions: Optional[Dict[str, Callable[..., Any]]] = None
    role_overrides: Dict[str, Role | str] = field(default_factory=dict)
    resource_overrides: Dict[str, Dict[str, str]] = field(default_factory=dict)
    unroll: int = 1
    materialise_routers: bool = True

    # -- fluent configuration ----------------------------------------------- #
    def with_roles(self, **roles: Role | str) -> "LangGraphExtractor":
        """Return a copy with additional role overrides."""
        merged = {**self.role_overrides, **roles}
        return self._replace(role_overrides=merged)

    def with_resources(self, **resources: Dict[str, str]) -> "LangGraphExtractor":
        """Return a copy with additional shared-resource declarations."""
        merged = {**self.resource_overrides, **resources}
        return self._replace(resource_overrides=merged)

    def _replace(self, **changes: Any) -> "LangGraphExtractor":
        base = {
            "globals_ns": self.globals_ns,
            "node_functions": self.node_functions,
            "role_overrides": dict(self.role_overrides),
            "resource_overrides": dict(self.resource_overrides),
            "unroll": self.unroll,
            "materialise_routers": self.materialise_routers,
        }
        base.update(changes)
        return LangGraphExtractor(**base)

    # -- extraction ---------------------------------------------------------- #
    def extract(self, graph: Any, name: str = "langgraph_system") -> SystemModel:
        """Read the architecture as it is, feedback loops and all.

        ``graph`` may be a compiled LangGraph, the drawable from
        ``graph.get_graph()``, mermaid text, a dict specification, or an
        already-built :class:`SystemModel` (returned unchanged).
        """
        return extract_architecture(
            graph,
            name=name,
            role_overrides=self.role_overrides or None,
            resource_overrides=self.resource_overrides or None,
            globals_ns=self.globals_ns,
            node_functions=self.node_functions,
            materialise_routers=self.materialise_routers,
        )

    def extract_acyclic(
        self, graph: Any, name: str = "langgraph_system"
    ) -> Tuple[SystemModel, CycleReport]:
        """Extract, then unroll feedback loops to :attr:`unroll` iterations.

        Returns the analysable (acyclic) model and the :class:`CycleReport`
        recording what was cut, so the loop handling is visible in the report
        rather than hidden in the tree.
        """
        return make_acyclic(self.extract(graph, name=name), unroll=self.unroll)

    __call__ = extract
