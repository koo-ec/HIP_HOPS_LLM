"""
hipgraph.acyclic — turning a cyclic agent graph into a loop-free analysis model.

Fault trees are, by definition, acyclic: a top event is refined into causes, and
no event may be its own cause.  Agentic LangGraph applications are *not* acyclic
— the ReAct pattern is a feedback loop (``generator -> coder -> generator``).
Classical HiP-HOPS meets the same problem with control loops in engineered
systems and resolves it by breaking the circular dependency explicitly rather
than letting the synthesis algorithm recurse for ever.

This module implements that step and records it, so the resulting fault tree can
be read as a *static* structure without hiding what was done to obtain it.

Two policies are offered, both producing a directed **acyclic** graph:

``unroll=k`` (default ``k = 1``)
    Iterations ``1..k`` of the loop body are represented explicitly as distinct
    component instances (``generator#1``, ``generator#2``, ...).  This exposes
    iteration-dependent behaviour (e.g. prompt growth) at the cost of a larger
    tree.

**Feedback cut with contribution preserved** (always applied at depth ``k``)
    The last back edge is replaced by a ``FEEDBACK`` pseudo-component that
    consumes the deviations the loop would have carried and delivers them to the
    system boundary, together with a *loop-exhaustion* basic event.  This is the
    conservative choice: simply deleting the back edge would silently remove the
    tool's contribution from the tree and produce an optimistic — that is,
    unsafe — result.

The intent is that no analysis result depends on an arbitrary recursion cut-off:
every loop is either unrolled a stated number of times or represented by a named
basic event that appears in the cut sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .model import Component, RawGraph, Role, SystemModel, build_system_model

__all__ = ["CycleReport", "find_cycles", "find_back_edges", "make_acyclic", "is_acyclic"]

Edge = Tuple[str, str, str, bool]  # src, dst, label, conditional


# --------------------------------------------------------------------------- #
# Cycle detection
# --------------------------------------------------------------------------- #
def _adjacency(edges: Sequence[Edge]) -> Dict[str, List[Edge]]:
    adj: Dict[str, List[Edge]] = {}
    for e in edges:
        adj.setdefault(e[0], []).append(e)
    return adj


def find_back_edges(nodes: Sequence[str], edges: Sequence[Edge]) -> List[Edge]:
    """Return the back edges of a depth-first forest (iterative DFS).

    A back edge points to a node currently on the DFS stack; removing the set of
    back edges is sufficient to make the graph acyclic.
    """
    adj = _adjacency(edges)
    colour: Dict[str, int] = {n: 0 for n in nodes}   # 0 white, 1 grey, 2 black
    back: List[Edge] = []

    ordered = list(nodes)
    for root in ordered:
        if colour.get(root, 0) != 0:
            continue
        stack: List[Tuple[str, int]] = [(root, 0)]
        colour[root] = 1
        while stack:
            node, idx = stack[-1]
            out = adj.get(node, [])
            if idx < len(out):
                stack[-1] = (node, idx + 1)
                edge = out[idx]
                nxt = edge[1]
                state = colour.get(nxt, 0)
                if state == 0:
                    colour[nxt] = 1
                    stack.append((nxt, 0))
                elif state == 1:
                    back.append(edge)
            else:
                colour[node] = 2
                stack.pop()
    return back


def _sccs(nodes: Sequence[str], edges: Sequence[Edge]) -> List[List[str]]:
    """Tarjan's strongly connected components (iterative)."""
    adj = _adjacency(edges)
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: Set[str] = set()
    stack: List[str] = []
    result: List[List[str]] = []
    counter = 0

    for root in nodes:
        if root in index:
            continue
        work: List[Tuple[str, int]] = [(root, 0)]
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            recursed = False
            out = adj.get(node, [])
            for i in range(pi, len(out)):
                nxt = out[i][1]
                work[-1] = (node, i + 1)
                if nxt not in index:
                    work.append((nxt, 0))
                    recursed = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if recursed:
                continue
            if low[node] == index[node]:
                comp: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == node:
                        break
                result.append(sorted(comp))
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return result


def find_cycles(model: SystemModel) -> List[List[str]]:
    """Components that lie on a cycle, grouped by strongly connected component."""
    nodes = list(model.components)
    edges = [(c.src, c.dst, c.label, c.conditional) for c in model.connections]
    self_loops = {c.src for c in model.connections if c.src == c.dst}
    out: List[List[str]] = []
    for comp in _sccs(nodes, edges):
        if len(comp) > 1 or (comp and comp[0] in self_loops):
            out.append(comp)
    return out


def is_acyclic(model: SystemModel) -> bool:
    return not find_cycles(model)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class CycleReport:
    """What was found and what was done about it — carried into the report."""

    cycles: List[List[str]] = field(default_factory=list)
    back_edges: List[Tuple[str, str]] = field(default_factory=list)
    unroll: int = 1
    feedback_components: List[str] = field(default_factory=list)
    replicated: Dict[str, List[str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def had_cycles(self) -> bool:
        return bool(self.cycles)

    def summary(self) -> str:
        if not self.cycles:
            return "The architecture is already acyclic; no loop handling was required."
        lines = [
            f"{len(self.cycles)} feedback loop(s) found; unrolled to depth {self.unroll} "
            f"and closed with {len(self.feedback_components)} feedback-cut component(s)."
        ]
        for cyc in self.cycles:
            lines.append("  loop: " + " -> ".join(cyc))
        for src, dst in self.back_edges:
            lines.append(f"  back edge cut: {src} -> {dst}")
        lines.extend("  " + n for n in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Transformation
# --------------------------------------------------------------------------- #
FEEDBACK_PREFIX = "feedback_cut@"


def make_acyclic(
    model: SystemModel,
    unroll: int = 1,
    boundary: Optional[str] = None,
) -> Tuple[SystemModel, CycleReport]:
    """Return a loop-free copy of ``model`` plus a :class:`CycleReport`.

    Parameters
    ----------
    unroll
        Number of loop iterations represented explicitly (``>= 1``).
    boundary
        Component that receives the feedback-cut output.  Defaults to the
        graph's sink (``__end__``), because an unresolved loop manifests at the
        system boundary as "no answer" or "answer too late".
    """
    unroll = max(1, int(unroll))
    report = CycleReport(unroll=unroll)
    cycles = find_cycles(model)
    edges: List[Edge] = [(c.src, c.dst, c.label, c.conditional) for c in model.connections]

    if not cycles:
        report.notes.append("No feedback loops detected.")
        return _rebuild(model, edges, {}, report), report

    report.cycles = cycles
    cycle_nodes: Set[str] = {n for cyc in cycles for n in cyc}
    back = find_back_edges(list(model.components), edges)
    back_set = {(e[0], e[1]) for e in back}
    report.back_edges = sorted(back_set)

    sinks = model.sinks()
    boundary_id = boundary or (sinks[0] if sinks else None)

    new_edges: List[Edge] = []
    new_components: Dict[str, Component] = {}
    replicated: Dict[str, List[str]] = {}

    def inst(node: str, i: int) -> str:
        if node not in cycle_nodes or unroll == 1:
            return node
        return f"{node}#{i}"

    # Replicate the loop bodies.
    for node in sorted(cycle_nodes):
        base = model.components[node]
        if unroll == 1:
            continue
        replicated[node] = []
        for i in range(1, unroll + 1):
            cid = f"{node}#{i}"
            clone = Component(
                id=cid,
                label=f"{base.label} (iteration {i})",
                role=base.role,
                resources=dict(base.resources),
                source_code=base.source_code,
                branches=list(base.branches),
                notes=list(base.notes) + [f"Unrolled instance {i} of {node}."],
                metadata=dict(base.metadata),
            )
            new_components[cid] = clone
            replicated[node].append(cid)

    for node, comp in model.components.items():
        if node in cycle_nodes and unroll > 1:
            continue
        new_components[node] = Component(
            id=comp.id,
            label=comp.label,
            role=comp.role,
            resources=dict(comp.resources),
            source_code=comp.source_code,
            branches=list(comp.branches),
            notes=list(comp.notes),
            metadata=dict(comp.metadata),
        )

    # Re-route edges over the unrolled instances.
    for src, dst, label, cond in edges:
        s_in, d_in = src in cycle_nodes, dst in cycle_nodes
        is_back = (src, dst) in back_set
        if s_in and d_in:
            for i in range(1, unroll + 1):
                if not is_back:
                    new_edges.append((inst(src, i), inst(dst, i), label, cond))
                elif i < unroll:
                    new_edges.append((inst(src, i), inst(dst, i + 1), label, cond))
                # the back edge of the last iteration is handled by the cut below
        elif s_in and not d_in:
            for i in range(1, unroll + 1):
                new_edges.append((inst(src, i), dst, label, cond))
        elif d_in and not s_in:
            new_edges.append((src, inst(dst, 1), label, cond))
        else:
            new_edges.append((src, dst, label, cond))

    # Close every cut back edge with a feedback component.
    for src, dst in sorted(back_set):
        cut_id = f"{FEEDBACK_PREFIX}{src}->{dst}"
        last_src = inst(src, unroll)
        new_components[cut_id] = Component(
            id=cut_id,
            label=f"feedback cut {src} -> {dst}",
            role=Role.FEEDBACK,
            notes=[
                f"Replaces the back edge {src} -> {dst}, cut after {unroll} modelled "
                "iteration(s). Deviations carried by the feedback path are delivered to "
                "the system boundary rather than discarded, so the tree stays conservative.",
            ],
            metadata={"loop_source": src, "loop_target": dst, "unroll": unroll},
        )
        report.feedback_components.append(cut_id)
        new_edges.append((last_src, cut_id, "loop cut", False))
        if boundary_id:
            new_edges.append((cut_id, boundary_id, "unresolved iteration", False))

    report.replicated = replicated
    report.notes.append(
        "Deleting a back edge outright would remove the feedback path's contribution "
        "from the fault tree and understate risk; the feedback-cut component preserves it."
    )
    if unroll == 1:
        report.notes.append(
            "unroll=1: a single pass through the loop body is modelled. Increase "
            "unroll to expose iteration-dependent effects such as prompt growth."
        )

    acyclic_model = _rebuild(model, new_edges, new_components, report)
    remaining = find_cycles(acyclic_model)
    if remaining:  # pragma: no cover - defensive
        raise RuntimeError(f"loop elimination failed; cycles remain: {remaining}")
    return acyclic_model, report


def _rebuild(
    model: SystemModel,
    edges: Sequence[Edge],
    components: Dict[str, Component],
    report: CycleReport,
) -> SystemModel:
    """Rebuild a :class:`SystemModel` (recomputing ports) from an edge list."""
    comps = components or model.components
    raw = RawGraph(name=model.name)
    for cid, comp in comps.items():
        raw.nodes[cid] = comp.label
        raw.node_meta[cid] = {
            "role": comp.role.value,
            "resources": dict(comp.resources),
            "source_code": comp.source_code,
            "branches": list(comp.branches),
            "notes": list(comp.notes),
        }
    for e in edges:
        raw.edges.append(e)
        for nid in (e[0], e[1]):
            if nid not in raw.nodes:
                raw.nodes[nid] = nid
    rebuilt = build_system_model(raw, materialise_routers=False)
    rebuilt.metadata = dict(model.metadata)
    rebuilt.metadata["acyclic"] = True
    rebuilt.metadata["unroll"] = report.unroll
    return rebuilt
