"""
hiphopsllm.faulttree.synthesis — fault tree synthesis (HiP-HOPS *Phase 2*).

HiP-HOPS does not ask the analyst to draw fault trees.  It *synthesises* them:
starting from a system-level hazard expressed as a deviation at the system
boundary, it walks the architecture backwards, and at each component substitutes
the local failure expression for the deviation being explained.  Input deviations
are resolved across connections into output deviations of the upstream
component, and the traversal continues until only basic events remain.

The same happens here, over the acyclic projection of the agent graph::

    hazard  H2  "wrong answer delivered, undetected"
      = VS-__end__.in
      = VS-aggregator.out                                  (across the connection)
      = BE-aggregator-SELECT AND (VS-react.out OR VS-cot.out)
        OR (VS-react.out AND VS-cot.out)
        OR CCF-LLM-...                                     (shared model snapshot)
      = ... until every leaf is a basic event

Termination is structural, not heuristic: the architecture is acyclic before
synthesis starts, expansions only ever move upstream, and every deviation is
expanded at most once (the result is memoised and shared, exactly as a transfer
gate is shared in a hand-drawn tree).  A defensive path check remains, so a
malformed annotation produces a clearly-labelled ``circular reference``
undeveloped event instead of an infinite recursion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..architecture.model import Role, SystemModel
from .failure import (
    And,
    BasicEvent,
    BasicEventRef,
    Const,
    Deviation,
    DevRef,
    Expr,
    FClass,
    FailureModel,
    Or,
)

__all__ = [
    "FTNode",
    "FaultTree",
    "Hazard",
    "default_hazards",
    "synthesise_fault_tree",
    "synthesise_all",
    "simplify_tree",
    "describe_deviation",
]


# --------------------------------------------------------------------------- #
# Fault tree data structure
# --------------------------------------------------------------------------- #
@dataclass
class FTNode:
    """One node of the synthesised tree.

    ``ntype``    top | intermediate | basic | undeveloped | house
    ``gate``     AND | OR | None (a single-cause pass-through)
    """

    id: str
    ntype: str
    label: str
    gate: Optional[str] = None
    children: List[str] = field(default_factory=list)
    deviation: Optional[str] = None
    event_id: Optional[str] = None
    component: Optional[str] = None
    #: "in" for an input-port deviation, "out" for an output-port one
    port_kind: Optional[str] = None
    #: set on a duplicate produced by :func:`expand_to_tree` — the id it copies
    repeat_of: Optional[str] = None
    #: transfer tag ("A", "B", …) linking a transfer symbol to its subtree
    transfer_ref: Optional[str] = None
    detail: str = ""

    @property
    def is_leaf(self) -> bool:
        return self.ntype in ("basic", "undeveloped", "house")


@dataclass
class FaultTree:
    """A synthesised static fault tree (a rooted DAG with shared sub-trees)."""

    id: str
    name: str
    root: str
    nodes: Dict[str, FTNode] = field(default_factory=dict)
    hazard: Optional["Hazard"] = None
    events: Dict[str, BasicEvent] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # -- queries ------------------------------------------------------------ #
    def node(self, nid: str) -> FTNode:
        return self.nodes[nid]

    def basic_event_ids(self) -> List[str]:
        return sorted({n.event_id for n in self.nodes.values()
                       if n.ntype == "basic" and n.event_id})

    def leaves(self) -> List[FTNode]:
        return [n for n in self.nodes.values() if n.is_leaf]

    def size(self) -> Dict[str, int]:
        kinds: Dict[str, int] = {}
        for n in self.nodes.values():
            kinds[n.ntype] = kinds.get(n.ntype, 0) + 1
        kinds["total"] = len(self.nodes)
        return kinds

    def depth(self) -> int:
        memo: Dict[str, int] = {}

        def _d(nid: str, seen: Tuple[str, ...] = ()) -> int:
            if nid in memo:
                return memo[nid]
            if nid in seen:                      # defensive; verify() rejects this
                return 0
            node = self.nodes[nid]
            value = 1 + max((_d(c, seen + (nid,)) for c in node.children), default=0)
            memo[nid] = value
            return value

        return _d(self.root)

    def parent_count(self) -> Dict[str, int]:
        count: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for child in node.children:
                count[child] = count.get(child, 0) + 1
        return count

    def simplified(self, **kwargs) -> "FaultTree":
        """Return a structurally reduced copy (see :func:`simplify_tree`)."""
        return simplify_tree(self, **kwargs)

    def shared_nodes(self) -> List[str]:
        """Nodes referenced by more than one parent (rendered as transfer gates)."""
        count: Dict[str, int] = {}
        for n in self.nodes.values():
            for c in n.children:
                count[c] = count.get(c, 0) + 1
        return sorted(k for k, v in count.items() if v > 1)

    # -- validation --------------------------------------------------------- #
    def verify_acyclic(self) -> bool:
        """True when no node is its own ancestor. A fault tree must satisfy this."""
        colour: Dict[str, int] = {}
        stack: List[Tuple[str, int]] = [(self.root, 0)]
        colour[self.root] = 1
        while stack:
            nid, idx = stack[-1]
            kids = self.nodes[nid].children
            if idx < len(kids):
                stack[-1] = (nid, idx + 1)
                child = kids[idx]
                state = colour.get(child, 0)
                if state == 1:
                    return False
                if state == 0:
                    colour[child] = 1
                    stack.append((child, 0))
            else:
                colour[nid] = 2
                stack.pop()
        return True


# --------------------------------------------------------------------------- #
# Hazards (top events)
# --------------------------------------------------------------------------- #
@dataclass
class Hazard:
    """A system-level effect to be analysed, anchored to a boundary deviation."""

    id: str
    name: str
    deviations: List[Deviation]
    severity: str = "major"
    description: str = ""
    detection: str = ""

    @property
    def label(self) -> str:
        return f"{self.id}: {self.name}"


_SEVERITY_ORDER = {"catastrophic": 4, "critical": 3, "major": 2, "minor": 1}


def default_hazards(model: SystemModel) -> List[Hazard]:
    """The standard hazard list for an agentic workflow.

    Anchored at the system boundary (the sink's input ports), plus one hazard
    per component that executes model-authored code, because that effect is not
    observable at the output port at all.
    """
    hazards: List[Hazard] = []
    sinks = model.sinks()
    boundary: List[Tuple[str, str]] = []
    for sid in sinks:
        comp = model.components[sid]
        for port in (comp.ports_in or ["in"]):
            boundary.append((sid, port))

    def devs(fclass: FClass) -> List[Deviation]:
        return [Deviation(cid, port, fclass) for cid, port in boundary]

    hazards.append(Hazard(
        id="H1", name="No answer delivered",
        deviations=devs(FClass.OMISSION), severity="major",
        description="The workflow terminates without producing a usable result: the run "
                    "aborts, a branch leads to END with nothing written, or the loop never "
                    "converges.",
        detection="Detectable — the caller observes a missing or empty result.",
    ))
    hazards.append(Hazard(
        id="H2", name="Incorrect answer delivered and accepted as correct",
        deviations=devs(FClass.VALUE_SUBTLE), severity="critical",
        description="A well-formed but wrong result reaches the caller. Nothing downstream "
                    "can distinguish it from a correct one, so it is acted upon.",
        detection="Not detectable by the system itself — requires an independent oracle.",
    ))
    hazards.append(Hazard(
        id="H3", name="Malformed answer delivered",
        deviations=devs(FClass.VALUE_COARSE), severity="minor",
        description="The result is present but violates the expected shape (missing final "
                    "answer field, truncated text, error string in place of a value).",
        detection="Detectable by a schema/parse check at the boundary.",
    ))
    hazards.append(Hazard(
        id="H4", name="Answer too late / budget exhausted",
        deviations=devs(FClass.LATE), severity="minor",
        description="Latency or token budget is exceeded, typically through repeated loop "
                    "iterations.",
        detection="Detectable — timeout or budget counter.",
    ))

    tools_with_exec = [
        c for c in model.components.values()
        if c.role is Role.TOOL and c.ports_out
    ]
    for comp in tools_with_exec:
        hazards.append(Hazard(
            id=f"H5-{comp.id}", name=f"Unsafe execution of model-authored code in {comp.id}",
            deviations=[Deviation(comp.id, comp.port_out(), FClass.COMMISSION)],
            severity="catastrophic",
            description="Code produced by the language model is executed with the host's "
                        "privileges. The effect is not visible at the workflow output, so it "
                        "is analysed as a top event of its own.",
            detection="Not detectable from the workflow result; requires host-level controls.",
        ))
    return hazards


# --------------------------------------------------------------------------- #
# Deviation descriptions
# --------------------------------------------------------------------------- #
_ROLE_NOUN = {
    Role.LLM_AGENT: "agent output",
    Role.TOOL: "tool observation",
    Role.ROUTER: "routing decision",
    Role.AGGREGATOR: "aggregated answer",
    Role.TRANSFORM: "node output",
    Role.SOURCE: "workflow input",
    Role.SINK: "workflow output",
    Role.FEEDBACK: "feedback path",
}


def describe_deviation(system: SystemModel, dev: Deviation) -> str:
    comp = system.components.get(dev.component)
    noun = _ROLE_NOUN.get(comp.role, "output") if comp else "output"
    return f"{dev.fclass.title} of {dev.component}.{dev.port} ({noun})"


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
class _Synthesiser:
    """Backward traversal of the architecture, composing local failure logic."""

    def __init__(self, fmodel: FailureModel, prune_empty: bool = True):
        self.fm = fmodel
        self.system = fmodel.system
        self.prune_empty = prune_empty
        self.nodes: Dict[str, FTNode] = {}
        self.memo: Dict[str, Optional[str]] = {}
        self.events: Dict[str, BasicEvent] = {}
        self.warnings: List[str] = []
        self._counter = 0

    # -- node factory ------------------------------------------------------- #
    def _nid(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def _add(self, node: FTNode) -> str:
        self.nodes[node.id] = node
        return node.id

    def _basic(self, event_id: str) -> Optional[str]:
        event = self.fm.events.get(event_id)
        if event is None:
            self.warnings.append(f"unknown basic event referenced: {event_id}")
            return None
        nid = f"BE::{event_id}"
        if nid not in self.nodes:
            self._add(FTNode(
                id=nid, ntype="basic", label=f"{event.id}\n{event.label}",
                event_id=event.id, component=event.component,
                detail=event.rationale,
            ))
        self.events[event_id] = event
        return nid

    def _undeveloped(self, label: str, detail: str = "", component: Optional[str] = None) -> str:
        return self._add(FTNode(
            id=self._nid("UND"), ntype="undeveloped", label=label,
            detail=detail, component=component,
        ))

    # -- expression expansion ---------------------------------------------- #
    def _build(self, expr: Expr, path: Tuple[str, ...]) -> Optional[str]:
        if isinstance(expr, Const):
            if not expr.value:
                return None
            return self._add(FTNode(id=self._nid("HOUSE"), ntype="house", label="TRUE"))
        if isinstance(expr, BasicEventRef):
            return self._basic(expr.event_id)
        if isinstance(expr, DevRef):
            return self._resolve(expr.deviation, path)
        if isinstance(expr, (And, Or)):
            gate = "AND" if isinstance(expr, And) else "OR"
            children = [c for c in (self._build(t, path) for t in expr.terms) if c]
            if not children:
                return None
            if len(children) == 1:
                return children[0]
            if gate == "AND" and len(children) < len(expr.terms):
                # A conjunction with an impossible term cannot occur at all.
                return None
            return self._add(FTNode(
                id=self._nid("G"), ntype="intermediate",
                label="all of the below" if gate == "AND" else "any of the below",
                gate=gate, children=children,
            ))
        raise TypeError(f"unsupported expression node: {expr!r}")

    # -- deviation resolution ---------------------------------------------- #
    def _resolve(self, dev: Deviation, path: Tuple[str, ...]) -> Optional[str]:
        key = dev.id
        # The path check must precede the memo lookup: a deviation still being
        # expanded is already in the memo (seeded below), so checking the memo
        # first would silently return "impossible" for a circular annotation
        # instead of flagging it — an optimistic, unsafe answer.
        if key in path:
            self.warnings.append(
                f"circular failure logic at {key}; represented as an undeveloped event"
            )
            return self._undeveloped(
                f"circular reference: {key}",
                "The annotation refers to itself. Analysis stops here; review the local "
                "failure logic of this component.",
                component=dev.component,
            )
        if key in self.memo:
            return self.memo[key]
        comp = self.system.components.get(dev.component)
        if comp is None:
            return None
        self.memo[key] = None  # provisional, prevents unbounded re-entry
        node = (self._resolve_input(dev, comp, path)
                if dev.port in comp.ports_in
                else self._resolve_output(dev, comp, path))
        self.memo[key] = node
        return node

    def _resolve_input(self, dev, comp, path: Tuple[str, ...]) -> Optional[str]:
        """An input deviation is caused by the upstream output or by the channel."""
        conns = self.system.incoming(comp.id, dev.port)
        if not conns:
            return None
        children: List[str] = []
        for conn in conns:
            upstream = Deviation(conn.src, conn.src_port, dev.fclass)
            up_node = self._resolve(upstream, path + (dev.id,))
            if up_node:
                children.append(up_node)
            for eid in self.fm.connection_events.get(conn.id, []):
                event = self.fm.events.get(eid)
                if event and event.fclass is dev.fclass:
                    be = self._basic(eid)
                    if be:
                        children.append(be)
        children = _dedup(children)
        if not children:
            return None
        if len(children) == 1:
            return children[0]
        return self._add(FTNode(
            id=self._nid("G"), ntype="intermediate",
            label=describe_deviation(self.system, dev), gate="OR",
            children=children, deviation=dev.id, component=comp.id, port_kind="in",
        ))

    def _resolve_output(self, dev, comp, path: Tuple[str, ...]) -> Optional[str]:
        expr = self.fm.expression(dev)
        if expr is None:
            if comp.role in (Role.SOURCE, Role.SINK) and not comp.ports_in:
                return None
            if self.prune_empty:
                return None
            return self._undeveloped(
                f"{describe_deviation(self.system, dev)} — not annotated",
                "No local failure logic was supplied for this output deviation.",
                component=comp.id,
            )
        child = self._build(expr, path + (dev.id,))
        if child is None:
            return None
        child_node = self.nodes[child]
        # Wrap in an intermediate event so the tree reads as a chain of deviations.
        gate = child_node.gate if child_node.ntype == "intermediate" and not child_node.deviation else None
        if gate and not child_node.deviation:
            node = FTNode(
                id=self._nid("E"), ntype="intermediate",
                label=describe_deviation(self.system, dev), gate=gate,
                children=list(child_node.children), deviation=dev.id, component=comp.id,
                port_kind="out", detail=f"local failure logic: {expr}",
            )
            del self.nodes[child]
            return self._add(node)
        return self._add(FTNode(
            id=self._nid("E"), ntype="intermediate",
            label=describe_deviation(self.system, dev), gate=None,
            children=[child], deviation=dev.id, component=comp.id,
            port_kind="out", detail=f"local failure logic: {expr}",
        ))

    # -- entry point -------------------------------------------------------- #
    def synthesise(self, hazard: Hazard) -> FaultTree:
        children: List[str] = []
        for dev in hazard.deviations:
            nid = self._resolve(dev, ())
            if nid:
                children.append(nid)
        children = _dedup(children)
        root = self._add(FTNode(
            id="TOP", ntype="top", label=hazard.label,
            gate="OR" if len(children) > 1 else None,
            children=children, detail=hazard.description,
        ))
        tree = FaultTree(
            id=hazard.id, name=hazard.name, root=root,
            nodes=self.nodes, hazard=hazard, events=dict(self.events),
            warnings=list(self.warnings),
        )
        if not children:
            tree.warnings.append(
                "No causes were found for this hazard: no annotated path reaches the "
                "boundary deviation. Check the architecture and the failure library."
            )
        if not tree.verify_acyclic():  # pragma: no cover - defensive
            raise RuntimeError(
                f"synthesised structure for {hazard.id} is not acyclic; refusing to "
                "return an invalid fault tree"
            )
        return tree


# --------------------------------------------------------------------------- #
# Structural simplification
# --------------------------------------------------------------------------- #
#: Labels of gates created purely to hold a Boolean combination — flattening
#: them into a parent of the same type removes no information.
_ANONYMOUS = {"all of the below", "any of the below"}


def simplify_tree(
    tree: FaultTree,
    collapse_single_input: bool = True,
    flatten_gates: bool = True,
    flatten_ports: bool = True,
    dedup_inputs: bool = True,
) -> FaultTree:
    """Reduce a synthesised tree to its informative structure.

    Synthesis is deliberately literal: it emits one intermediate event per
    deviation, so a chain of components produces a chain of one-input gates.
    That is faithful but tedious to read, and a one-input OR is not a gate at
    all. Three reductions are applied to a fixed point:

    ``collapse_single_input``
        A gate with a single input *is* that input. ``OR(BE-coder-PARSE)``
        becomes ``BE-coder-PARSE`` and the intervening event box disappears.
    ``flatten_gates``
        ``OR(a, OR(b, c))`` becomes ``OR(a, b, c)`` for the anonymous combination
        gates.
    ``flatten_ports``
        An *input*-port deviation is absorbed into the output deviation it
        causes: "omission at coder.in" under "omission of coder.out" is one step,
        not two. One node per component output survives, so the propagation
        between components stays visible while the port-level bookkeeping goes.
        Set ``False`` to keep every port explicitly.
    ``dedup_inputs``
        The same input listed twice under one gate is listed once.

    **The Boolean function is unchanged**: every reduction is an identity of
    Boolean algebra, so the minimal cut sets before and after are identical.
    ``test_synthesis_internals.py`` checks exactly that. The unreduced tree remains
    available (``report.raw_trees``) when the full propagation chain is wanted.
    """
    nodes: Dict[str, FTNode] = {
        nid: FTNode(
            id=node.id, ntype=node.ntype, label=node.label, gate=node.gate,
            children=list(node.children), deviation=node.deviation,
            event_id=node.event_id, component=node.component,
            port_kind=node.port_kind, detail=node.detail,
        )
        for nid, node in tree.nodes.items()
    }
    before = len(nodes)

    def parents_of() -> Dict[str, int]:
        count: Dict[str, int] = {nid: 0 for nid in nodes}
        for node in nodes.values():
            for child in node.children:
                count[child] = count.get(child, 0) + 1
        return count

    changed = True
    guard = 0
    while changed and guard < 100:
        changed = False
        guard += 1
        parents = parents_of()
        for node in nodes.values():
            rewritten: List[str] = []
            for cid in node.children:
                child = nodes.get(cid)
                if child is None:
                    continue
                # a gate (or pass-through) with one input is that input
                if (collapse_single_input and child.ntype == "intermediate"
                        and len(child.children) == 1):
                    rewritten.append(child.children[0])
                    changed = True
                    continue
                # OR under OR / AND under AND, for gates that carry no name of
                # their own: anonymous combinations, and input-port deviations
                # (which are absorbed into the output deviation they cause).
                absorbable = child.label in _ANONYMOUS or (
                    flatten_ports and child.port_kind == "in"
                )
                if (flatten_gates and child.ntype == "intermediate"
                        and child.gate is not None and child.gate == node.gate
                        and absorbable and parents.get(cid, 0) == 1):
                    rewritten.extend(child.children)
                    changed = True
                    continue
                rewritten.append(cid)
            if dedup_inputs:
                rewritten = _dedup(rewritten)
            if rewritten != node.children:
                node.children = rewritten
                changed = True

    # a gate is only a gate with two or more inputs — the top event included
    for node in nodes.values():
        if node.ntype in ("intermediate", "top") and node.gate and len(node.children) < 2:
            node.gate = None

    # drop everything no longer reachable from the top event
    reachable: Set[str] = set()
    stack = [tree.root]
    while stack:
        nid = stack.pop()
        if nid in reachable or nid not in nodes:
            continue
        reachable.add(nid)
        stack.extend(nodes[nid].children)

    reduced = FaultTree(
        id=tree.id, name=tree.name, root=tree.root,
        nodes={nid: nodes[nid] for nid in nodes if nid in reachable},
        hazard=tree.hazard, events=dict(tree.events), warnings=list(tree.warnings),
        notes=list(tree.notes),
    )
    removed = before - len(reduced.nodes)
    if removed:
        reduced.notes.append(
            f"Simplified: {before} nodes -> {len(reduced.nodes)} ({removed} removed) by "
            "collapsing single-input gates, flattening nested combination gates and "
            "de-duplicating inputs. The Boolean function and the minimal cut sets are "
            "unchanged."
        )
    if not reduced.verify_acyclic():  # pragma: no cover - defensive
        raise RuntimeError(f"simplification broke the tree for {tree.id}")
    return reduced


def _tag(index: int) -> str:
    """A, B, … Z, AA, AB, … for transfer symbols."""
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def expand_to_tree(
    tree: FaultTree, max_nodes: int = 500, transfer_subtrees: bool = True
) -> FaultTree:
    """Turn the shared-subtree DAG into a strict tree by repeating causes.

    Synthesis memoises each deviation, so one cause can feed several branches —
    correct, compact, and unreadable when drawn: its edges run right across the
    diagram. Conventional fault trees avoid this by repeating the event under
    each parent (or referencing it through a transfer gate). Repeating it makes
    every subtree local, which is what lets the diagram be drawn with no
    crossing lines at all.

    Two devices, both standard practice:

    * a shared **basic event** is simply repeated under each parent (marked
      ``repeat_of``); it is cheap and is how repeated events are normally shown;
    * a shared **subtree** is drawn once and referenced elsewhere by a
      **transfer symbol** carrying a tag ("A", "B", …), rather than being copied
      wholesale — which is what keeps the diagram from exploding in width.

    Cut sets are unaffected: a repeated event is still one event, and every copy
    carries the same ``event_id``.

    Raises :class:`ValueError` if the expansion would still exceed ``max_nodes``.
    """
    nodes: Dict[str, FTNode] = {}
    used: Dict[str, int] = {}
    tags: Dict[str, str] = {}          # original node id -> transfer tag
    first_copy: Dict[str, str] = {}    # original node id -> id of its first copy

    def copy(nid: str) -> str:
        if len(nodes) > max_nodes:
            raise ValueError(
                f"expanding {tree.id} to a strict tree exceeds {max_nodes} nodes; "
                "draw the shared-subtree form instead (as_tree=False)"
            )
        source = tree.nodes[nid]
        seen = used.get(nid, 0)
        used[nid] = seen + 1

        # a shared subtree is referenced, not repeated
        if seen and source.children and transfer_subtrees:
            tag = tags.setdefault(nid, _tag(len(tags)))
            transfer_id = f"{nid}~t{seen}"
            nodes[transfer_id] = FTNode(
                id=transfer_id, ntype="transfer", label=source.label,
                component=source.component, deviation=source.deviation,
                repeat_of=nid, transfer_ref=tag,
                detail="Transfer: this cause is developed elsewhere in the tree.",
            )
            return transfer_id

        new_id = nid if seen == 0 else f"{nid}~r{seen}"
        children = [copy(c) for c in source.children]
        nodes[new_id] = FTNode(
            id=new_id, ntype=source.ntype, label=source.label, gate=source.gate,
            children=children, deviation=source.deviation, event_id=source.event_id,
            component=source.component, port_kind=source.port_kind,
            repeat_of=(nid if seen else None), detail=source.detail,
        )
        first_copy.setdefault(nid, new_id)
        return new_id

    root = copy(tree.root)
    for original, tag in tags.items():          # label the subtree being referenced
        target = first_copy.get(original)
        if target in nodes:
            nodes[target].transfer_ref = tag

    expanded = FaultTree(
        id=tree.id, name=tree.name, root=root, nodes=nodes, hazard=tree.hazard,
        events=dict(tree.events), warnings=list(tree.warnings), notes=list(tree.notes),
    )
    repeats = sum(1 for n in nodes.values() if n.repeat_of and n.ntype != "transfer")
    if repeats or tags:
        expanded.notes.append(
            f"Drawn as a strict tree: {repeats} repeated event(s) and {len(tags)} "
            "transfer reference(s), so that no connector crosses the diagram."
        )
    return expanded


def _dedup(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def synthesise_fault_tree(
    fmodel: FailureModel, hazard: Hazard, simplify: bool = True
) -> FaultTree:
    """Synthesise the fault tree for one hazard."""
    tree = _Synthesiser(fmodel).synthesise(hazard)
    return simplify_tree(tree) if simplify else tree


def synthesise_all(
    fmodel: FailureModel,
    hazards: Optional[Sequence[Hazard]] = None,
    simplify: bool = True,
) -> Dict[str, FaultTree]:
    """Synthesise one fault tree per hazard, keyed by hazard id."""
    hazards = list(hazards) if hazards is not None else default_hazards(fmodel.system)
    trees: Dict[str, FaultTree] = {}
    for hazard in hazards:
        trees[hazard.id] = synthesise_fault_tree(fmodel, hazard, simplify=simplify)
    return trees
