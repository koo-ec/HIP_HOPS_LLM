"""
hiphopsllm.architecture.model — architecture meta-model and extraction from a LangGraph graph.

This is *Phase 0* of the HiP-HOPS process ("model the architecture").  HiP-HOPS
analyses a hierarchical model of components connected through ports; before any
failure logic can be attached we need that model.  In a LangGraph application the
architecture is already there — it is exactly the object the notebook renders
with::

    display(Image(graph.get_graph().draw_mermaid_png()))

``graph.get_graph()`` returns a drawable graph carrying ``.nodes`` and ``.edges``.
This module turns it into a :class:`SystemModel`: typed components, explicit
input/output **ports**, and directed **connections** between ports.  Three input
routes are supported so the analysis works inside the notebook *and* offline:

1. a compiled LangGraph / any object exposing ``get_graph()``;
2. the mermaid source string (``graph.get_graph().draw_mermaid()``);
3. a plain dict specification (:meth:`SystemModel.from_spec`).

Two modelling decisions are worth stating explicitly because they shape every
fault tree produced downstream:

*   **Conditional edges become a component.**  In LangGraph a router function is
    not a node — it is a callable attached to ``add_conditional_edges``.  It is
    nevertheless a real piece of software with its own failure modes (a regular
    expression that matches the wrong branch, or no branch at all).  We
    therefore materialise it as a distinct ``ROUTER`` component sitting between
    the deciding node and its successors.

*   **Fan-in is a shared-state hazard.**  Where two nodes write the same
    LangGraph state channel in one super-step, a non-reducer channel raises
    ``InvalidUpdateError`` and the run dies.  Fan-in connections are flagged here
    (``Connection.fan_in``) so the failure library can attach channel-contention
    events to them.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

START_ID = "__start__"
END_ID = "__end__"

__all__ = [
    "Role",
    "Component",
    "Connection",
    "SystemModel",
    "RawGraph",
    "parse_mermaid",
    "extract_architecture",
    "raw_from_spec",
    "source_of_function",
]


# --------------------------------------------------------------------------- #
# Component taxonomy
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    """Archetype of a component, which selects its default failure logic."""

    SOURCE = "source"          # __start__
    SINK = "sink"              # __end__
    LLM_AGENT = "llm_agent"    # a node whose output is produced by an LLM
    TOOL = "tool"              # deterministic executor (calculator, code, API)
    ROUTER = "router"          # conditional-edge decision function
    AGGREGATOR = "aggregator"  # fan-in node that combines several agents
    TRANSFORM = "transform"    # any other deterministic node
    FEEDBACK = "feedback_cut"  # pseudo-component replacing a cut feedback edge

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass
class Component:
    """One architectural block with ports, akin to a HiP-HOPS component."""

    id: str
    label: str
    role: Role
    ports_in: List[str] = field(default_factory=list)
    ports_out: List[str] = field(default_factory=list)
    #: shared physical/logical resources — the basis for common-cause grouping
    #: e.g. ``{"llm": "Qwen/Qwen2.5-Math-1.5B-Instruct", "runtime": "gpu:0"}``
    resources: Dict[str, str] = field(default_factory=dict)
    source_code: str = ""
    branches: List[str] = field(default_factory=list)   # routers only
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_boundary(self) -> bool:
        return self.role in (Role.SOURCE, Role.SINK)

    def port_in(self, index: int = 0) -> str:
        return self.ports_in[index] if self.ports_in else "in"

    def port_out(self, index: int = 0) -> str:
        return self.ports_out[index] if self.ports_out else "out"


@dataclass(frozen=True)
class Connection:
    """A directed port-to-port link (a HiP-HOPS 'connection'/channel)."""

    src: str
    src_port: str
    dst: str
    dst_port: str
    label: str = ""
    conditional: bool = False
    fan_in: bool = False       # target is written by >1 predecessor in a step
    parallel: bool = False     # source belongs to a parallel (fan-out) branch

    @property
    def id(self) -> str:
        return f"{self.src}.{self.src_port}->{self.dst}.{self.dst_port}"


@dataclass
class SystemModel:
    """The full architecture: components + connections, plus lookup helpers."""

    name: str
    components: Dict[str, Component] = field(default_factory=dict)
    connections: List[Connection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- lookup ------------------------------------------------------------- #
    def component(self, cid: str) -> Component:
        return self.components[cid]

    def incoming(self, cid: str, port: Optional[str] = None) -> List[Connection]:
        return [
            c for c in self.connections
            if c.dst == cid and (port is None or c.dst_port == port)
        ]

    def outgoing(self, cid: str, port: Optional[str] = None) -> List[Connection]:
        return [
            c for c in self.connections
            if c.src == cid and (port is None or c.src_port == port)
        ]

    def predecessors(self, cid: str) -> List[str]:
        return sorted({c.src for c in self.connections if c.dst == cid})

    def successors(self, cid: str) -> List[str]:
        return sorted({c.dst for c in self.connections if c.src == cid})

    def sinks(self) -> List[str]:
        explicit = [c.id for c in self.components.values() if c.role is Role.SINK]
        if explicit:
            return explicit
        return [cid for cid in self.components if not self.successors(cid)]

    def sources(self) -> List[str]:
        explicit = [c.id for c in self.components.values() if c.role is Role.SOURCE]
        if explicit:
            return explicit
        return [cid for cid in self.components if not self.predecessors(cid)]

    def by_role(self, role: Role) -> List[Component]:
        return [c for c in self.components.values() if c.role is role]

    # -- common-cause grouping --------------------------------------------- #
    def common_cause_groups(self) -> Dict[Tuple[str, str], List[str]]:
        """Group components that share a resource.

        A shared resource (the same model snapshot, the same GPU, the same
        tokenizer, the same prompt template) is a common-cause failure (CCF)
        candidate: redundancy built from components in one group does *not*
        deliver the failure independence the architecture appears to promise.
        """
        groups: Dict[Tuple[str, str], List[str]] = {}
        for comp in self.components.values():
            for kind, value in comp.resources.items():
                groups.setdefault((kind, value), []).append(comp.id)
        return {k: sorted(v) for k, v in groups.items() if len(v) > 1}

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_spec(cls, spec: Dict[str, Any], **kwargs: Any) -> "SystemModel":
        """Build a model from a plain dict (offline / unit-test route).

        ``spec`` schema::

            {"name": str,
             "nodes": {node_id: {"role": str|Role, "resources": {...},
                                 "source_code": str, "label": str}},
             "edges": [(src, dst) | (src, dst, label, conditional)]}

        ``kwargs`` are forwarded to :func:`build_system_model`, so role and
        resource overrides work on a specification exactly as they do on a live
        LangGraph.
        """
        return build_system_model(raw_from_spec(spec), **kwargs)

    # -- reporting ---------------------------------------------------------- #
    def architecture_table(self) -> List[Dict[str, str]]:
        rows = []
        for cid in sorted(self.components):
            comp = self.components[cid]
            rows.append(
                {
                    "component": cid,
                    "role": comp.role.value,
                    "in_ports": ", ".join(comp.ports_in) or "-",
                    "out_ports": ", ".join(comp.ports_out) or "-",
                    "resources": ", ".join(
                        f"{k}={v}" for k, v in sorted(comp.resources.items())
                    ) or "-",
                }
            )
        return rows

    def to_mermaid(self) -> str:
        lines = ["graph TD;"]
        for cid, comp in self.components.items():
            safe = _mm_id(cid)
            if comp.role is Role.SOURCE:
                lines.append(f'    {safe}(["{comp.label}"]):::boundary')
            elif comp.role is Role.SINK:
                lines.append(f'    {safe}(["{comp.label}"]):::boundary')
            elif comp.role is Role.ROUTER:
                lines.append(f'    {safe}{{"{comp.label}"}}:::router')
            else:
                lines.append(f'    {safe}["{comp.label}"]:::{comp.role.value}')
        for conn in self.connections:
            arrow = "-.->" if conn.conditional else "-->"
            lbl = f"|{conn.label}|" if conn.label else ""
            lines.append(f"    {_mm_id(conn.src)} {arrow}{lbl} {_mm_id(conn.dst)};")
        lines += [
            "    classDef boundary fill:#254E58,color:#fff,stroke:#254E58;",
            "    classDef llm_agent fill:#88BDBC,color:#112D32,stroke:#254E58;",
            "    classDef tool fill:#F5E9C9,color:#112D32,stroke:#254E58;",
            "    classDef router fill:#E6B89C,color:#112D32,stroke:#254E58;",
            "    classDef aggregator fill:#C5D8D1,color:#112D32,stroke:#254E58;",
            "    classDef transform fill:#EEE,color:#112D32,stroke:#254E58;",
        ]
        return "\n".join(lines)


def _mm_id(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", text)


# --------------------------------------------------------------------------- #
# Raw graph (intermediate representation)
# --------------------------------------------------------------------------- #
@dataclass
class RawGraph:
    """Node ids/labels and raw edges, before ports and roles are assigned."""

    name: str = "system"
    nodes: Dict[str, str] = field(default_factory=dict)
    edges: List[Tuple[str, str, str, bool]] = field(default_factory=list)
    node_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)


_MERMAID_EDGE = re.compile(
    r"^\s*(?P<src>[A-Za-z0-9_:.\-]+)\s*"
    r"(?P<arrow>"
    r"-\.\s*(?P<dlabel>.*?)\s*\.->"          # -. label .->   (conditional)
    r"|-\.->"                                 # -.->           (conditional)
    r"|-->\s*\|(?P<plabel>[^|]*)\|"           # -->|label|
    r"|-->"                                   # -->
    r"|==>"                                   # ==>
    r")\s*"
    r"(?P<dst>[A-Za-z0-9_:.\-]+)\s*;?\s*$"
)

_MERMAID_NODE = re.compile(
    r"^\s*(?P<id>[A-Za-z0-9_:.\-]+)\s*"
    r"(?:\(\[(?P<lbl1>.*?)\]\)|\[\[(?P<lbl2>.*?)\]\]|\{\{(?P<lbl3>.*?)\}\}"
    r"|\((?P<lbl4>.*?)\)|\[(?P<lbl5>.*?)\]|\{(?P<lbl6>.*?)\})"
    r"\s*(?::::\w+)?\s*;?\s*$"
)

_SKIP_PREFIXES = (
    "graph", "flowchart", "classDef", "class ", "subgraph", "end", "---",
    "config:", "%%", "linkStyle", "style ", "click ",
)


def _clean_label(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    return text.strip().strip('"').strip()


def parse_mermaid(mermaid: str, name: str = "system") -> RawGraph:
    """Parse the mermaid source produced by ``draw_mermaid()``.

    Used when the drawable graph object is unavailable (e.g. offline analysis of
    a notebook, or a mermaid diagram pasted from a report).
    """
    raw = RawGraph(name=name)
    for line in mermaid.splitlines():
        stripped = line.strip()
        if not stripped or any(stripped.startswith(p) for p in _SKIP_PREFIXES):
            continue
        m = _MERMAID_EDGE.match(stripped)
        if m:
            src, dst = m.group("src"), m.group("dst")
            label = _clean_label(m.group("dlabel") or m.group("plabel") or "")
            # a dotted arrow is LangGraph's conditional edge; read it off the
            # matched arrow rather than inferring it from surrounding text
            conditional = m.group("arrow").startswith("-.")
            raw.edges.append((src, dst, label, conditional))
            raw.nodes.setdefault(src, src)
            raw.nodes.setdefault(dst, dst)
            continue
        m = _MERMAID_NODE.match(stripped)
        if m:
            nid = m.group("id")
            label = _clean_label(
                next((m.group(g) for g in ("lbl1", "lbl2", "lbl3", "lbl4", "lbl5", "lbl6")
                      if m.group(g) is not None), nid)
            )
            raw.nodes[nid] = label or nid
    return raw


# --------------------------------------------------------------------------- #
# Extraction from a live LangGraph object
# --------------------------------------------------------------------------- #
def raw_from_spec(spec: Dict[str, Any]) -> "RawGraph":
    """Read a plain dict specification into a :class:`RawGraph`.

    Kept separate from :meth:`SystemModel.from_spec` so the specification route
    goes through exactly the same :func:`build_system_model` call as a live
    LangGraph, and therefore honours the same overrides.
    """
    raw = RawGraph(name=spec.get("name", "system"))
    for nid, meta in (spec.get("nodes") or {}).items():
        raw.nodes[nid] = str((meta or {}).get("label", nid))
        raw.node_meta[nid] = dict(meta or {})
    for edge in spec.get("edges", []):
        src, dst = edge[0], edge[1]
        label = edge[2] if len(edge) > 2 and edge[2] else ""
        cond = bool(edge[3]) if len(edge) > 3 else bool(label)
        raw.edges.append((src, dst, label, cond))
        for nid in (src, dst):
            raw.nodes.setdefault(nid, nid)
    return raw


def _drawable(graph_like: Any) -> Any:
    """Return the drawable graph (``graph.get_graph()``) from whatever we got."""
    if hasattr(graph_like, "nodes") and hasattr(graph_like, "edges"):
        return graph_like
    if hasattr(graph_like, "get_graph"):
        return graph_like.get_graph()
    raise TypeError(
        "Expected a compiled LangGraph (has .get_graph()), a drawable graph "
        f"(has .nodes/.edges) or a mermaid string; got {type(graph_like)!r}"
    )


def _resolve_callable(obj: Any, depth: int = 0) -> Optional[Callable[..., Any]]:
    """Best-effort unwrapping of a LangGraph node payload to its function."""
    if obj is None or depth > 6:
        return None
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return obj
    for attr in ("func", "_func", "fn", "callable", "node", "runnable", "bound", "action"):
        inner = getattr(obj, attr, None)
        if inner is not None and inner is not obj:
            found = _resolve_callable(inner, depth + 1)
            if found is not None:
                return found
    steps = getattr(obj, "steps", None) or getattr(obj, "steps__", None)
    if isinstance(steps, (list, tuple)):
        for step in steps:
            found = _resolve_callable(step, depth + 1)
            if found is not None:
                return found
    return None


def source_of_function(fn: Optional[Callable[..., Any]]) -> str:
    """``inspect.getsource``, but only when the text really is that function.

    In a notebook whose cell has been re-executed, ``linecache`` can hand back
    stale or unrelated lines. Classifying a component from someone else's source
    is worse than classifying it from none, so the result is discarded unless it
    contains the function's own ``def``.
    """
    if fn is None:
        return ""
    # LangGraph node payloads are frequently partials or Runnable wrappers, and
    # inspect.getsource() on one of those returns nothing. Unwrapping first is
    # what makes node_functions={"worker": partial(worker, ...)} work at all;
    # without it the component silently gets no source, and therefore no role
    # hint and no detected resources.
    fn = _resolve_callable(fn) or fn
    try:
        code = inspect.getsource(fn)
    except (OSError, TypeError):
        return ""
    name = getattr(fn, "__name__", "")
    if not name or not re.search(rf"(?:async\s+)?def\s+{re.escape(name)}\s*\(", code):
        return ""
    return code


def _source_of(obj: Any) -> str:
    return source_of_function(_resolve_callable(obj))


def _raw_from_drawable(drawable: Any, name: str) -> RawGraph:
    raw = RawGraph(name=name)
    nodes = getattr(drawable, "nodes", {}) or {}
    items = nodes.items() if hasattr(nodes, "items") else ((n, n) for n in nodes)
    for nid, node in items:
        label = getattr(node, "name", None) or str(nid)
        raw.nodes[str(nid)] = str(label)
        raw.node_meta[str(nid)] = {
            "data": getattr(node, "data", None),
            "metadata": getattr(node, "metadata", None) or {},
        }
    for edge in getattr(drawable, "edges", []) or []:
        src = str(getattr(edge, "source", None) or edge[0])
        dst = str(getattr(edge, "target", None) or edge[1])
        label = getattr(edge, "data", None)
        conditional = bool(getattr(edge, "conditional", False))
        raw.edges.append((src, dst, _clean_label(str(label)) if label else "", conditional))
        raw.nodes.setdefault(src, src)
        raw.nodes.setdefault(dst, dst)
    return raw


# --------------------------------------------------------------------------- #
# Role classification and resource detection
# --------------------------------------------------------------------------- #
_LLM_PATTERNS = (
    r"\.generate\s*\(", r"apply_chat_template", r"\bllm\b", r"ChatOpenAI",
    r"ChatAnthropic", r"\.invoke\s*\(\s*messages", r"tokenizer",
)
_TOOL_PATTERNS = (
    r"\beval\s*\(", r"\bexec\s*\(", r"subprocess", r"requests\.", r"\.run\s*\(",
    r"sympy", r"calculator", r"@tool\b",
)
_AGGREGATOR_NAMES = ("aggregat", "combin", "reduce", "vote", "judge", "merge", "consensus", "critic")
_TOOL_NAMES = ("tool", "coder", "code", "calc", "exec", "search", "retriev", "python")
_LLM_NAMES = ("agent", "generator", "llm", "react", "cot", "planner", "writer", "solver", "reason")


def _match_any(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_role(
    node_id: str,
    label: str,
    source_code: str,
    in_degree: int,
    has_conditional_out: bool,
) -> Role:
    """Heuristic role assignment; always overridable by the caller."""
    if node_id == START_ID:
        return Role.SOURCE
    if node_id == END_ID:
        return Role.SINK

    name = f"{node_id} {label}".lower()
    code = source_code or ""

    if any(k in name for k in _AGGREGATOR_NAMES) and in_degree >= 2:
        return Role.AGGREGATOR
    if _match_any(_TOOL_PATTERNS, code) and not _match_any([r"\.generate\s*\("], code):
        return Role.TOOL
    if any(k in name for k in _TOOL_NAMES) and not _match_any(_LLM_PATTERNS, code):
        return Role.TOOL
    if _match_any(_LLM_PATTERNS, code):
        return Role.AGGREGATOR if any(k in name for k in _AGGREGATOR_NAMES) else Role.LLM_AGENT
    if any(k in name for k in _LLM_NAMES):
        return Role.LLM_AGENT
    if any(k in name for k in _TOOL_NAMES):
        return Role.TOOL
    # Last resort. Routers are normally materialised as their own components by
    # build_system_model, so this only fires with materialise_routers=False, or
    # for a node whose name and source say nothing at all. A node that chooses
    # between successors is more usefully a router than an anonymous transform.
    if has_conditional_out:
        return Role.ROUTER
    return Role.TRANSFORM


# These match the *variable* a node generates through, so that a shared model
# reached by name — not by a string literal in the source — is still detected.
#
# The earlier patterns required a character before "model"/"tokenizer"
# (``[A-Za-z_][A-Za-z0-9_]*model``), so they matched ``my_model`` but never
# ``model`` or ``model_deep`` — the two names the source notebook actually uses.
# A snapshot shared through such a variable was therefore invisible, which turns
# a common-cause single point of failure into an apparently redundant
# architecture: the single most consequential extraction error this package can
# make. Match any identifier at the call site instead and filter by name.
_GENERATE_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*generate\s*\(", re.I)
_TOKENIZER_CALL = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
    r"(?:apply_chat_template|batch_decode|decode|encode|tokenize)\s*\(",
    re.I,
)


class _NameFilter:
    """A findall-compatible matcher that keeps only names containing ``needle``."""

    def __init__(self, pattern: "re.Pattern[str]", needle: str) -> None:
        self._pattern = pattern
        self._needle = needle.lower()

    def findall(self, text: str) -> List[str]:
        return [m for m in self._pattern.findall(text) if self._needle in m.lower()]


_MODEL_VAR = _NameFilter(_GENERATE_CALL, "model")
_TOKENIZER_VAR = _NameFilter(_TOKENIZER_CALL, "tokeniz")
_HF_ID = re.compile(r"[\"']([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)[\"']")


def _model_id_of(obj: Any) -> Optional[str]:
    for path in ("name_or_path", "config._name_or_path", "config.name_or_path", "model_name", "model"):
        cur: Any = obj
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            if isinstance(cur, str) and cur:
                return cur
        except AttributeError:
            continue
    return None


def detect_resources(
    source_code: str,
    globals_ns: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Infer the shared resources a node depends on (for CCF grouping).

    When the notebook's ``globals()`` is supplied the *actual* model objects are
    interrogated, so two agents that look independent but load the same snapshot
    are correctly identified as a common-cause group.
    """
    resources: Dict[str, str] = {}
    if not source_code:
        return resources

    model_vars = sorted(set(_MODEL_VAR.findall(source_code)))
    tok_vars = sorted(set(_TOKENIZER_VAR.findall(source_code)))

    resolved: List[str] = []
    if globals_ns:
        for var in model_vars:
            mid = _model_id_of(globals_ns.get(var))
            if mid:
                resolved.append(mid)
    if not resolved:
        literals = [m for m in _HF_ID.findall(source_code) if "/" in m]
        resolved = sorted(set(literals))
    if resolved:
        resources["llm"] = resolved[0] if len(resolved) == 1 else "|".join(sorted(set(resolved)))
    elif model_vars:
        resources["llm"] = f"var:{model_vars[0]}"

    if tok_vars:
        if globals_ns:
            tids = [t for t in (_model_id_of(globals_ns.get(v)) for v in tok_vars) if t]
            if tids:
                resources["tokenizer"] = tids[0]
        resources.setdefault("tokenizer", f"var:{tok_vars[0]}")

    if re.search(r"cuda|device_map|torch\.", source_code, re.I):
        resources["runtime"] = "local-gpu"
    if re.search(r"BitsAndBytes|load_in_4bit", source_code, re.I):
        resources["quantisation"] = "4bit-nf4"
    return resources


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_system_model(
    raw: RawGraph,
    role_overrides: Optional[Dict[str, Role | str]] = None,
    resource_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    globals_ns: Optional[Dict[str, Any]] = None,
    node_functions: Optional[Dict[str, Callable[..., Any]]] = None,
    materialise_routers: bool = True,
) -> SystemModel:
    """Turn a :class:`RawGraph` into a fully ported :class:`SystemModel`."""
    role_overrides = {k: Role(v) if not isinstance(v, Role) else v
                      for k, v in (role_overrides or {}).items()}
    resource_overrides = resource_overrides or {}
    node_functions = node_functions or {}

    # --- source code per node -------------------------------------------- #
    sources: Dict[str, str] = {}
    for nid in raw.nodes:
        meta = raw.node_meta.get(nid, {})
        code = str(meta.get("source_code", "") or "")
        if not code and nid in node_functions:
            code = source_of_function(node_functions[nid])
        if not code and "data" in meta:
            code = _source_of(meta["data"])
        if not code and globals_ns and callable(globals_ns.get(nid)):
            code = source_of_function(globals_ns[nid])
        sources[nid] = code

    in_degree = {nid: sum(1 for e in raw.edges if e[1] == nid) for nid in raw.nodes}
    cond_out = {nid: any(e[0] == nid and e[3] for e in raw.edges) for nid in raw.nodes}

    model = SystemModel(name=raw.name, metadata={"origin": "langgraph"})

    # --- components ------------------------------------------------------- #
    for nid, label in raw.nodes.items():
        meta = raw.node_meta.get(nid, {})
        role = role_overrides.get(nid)
        if role is None and meta.get("role"):
            role = Role(meta["role"])
        if role is None:
            role = classify_role(nid, label, sources[nid], in_degree.get(nid, 0), cond_out.get(nid, False))
        # An explicitly supplied (even empty) resource dict is authoritative;
        # only an absent key triggers detection. Without this a rebuild — e.g.
        # after loop elimination — would re-detect and reintroduce resources the
        # caller deliberately cleared.
        if "resources" in meta:
            resources = dict(meta.get("resources") or {})
        else:
            resources = detect_resources(sources[nid], globals_ns)
        resources.update(resource_overrides.get(nid, {}))
        if role in (Role.SOURCE, Role.SINK):
            resources = {}
        model.components[nid] = Component(
            id=nid,
            label=_clean_label(label) or nid,
            role=role,
            resources=resources,
            source_code=sources[nid],
            branches=list(meta.get("branches") or []),
            notes=list(meta.get("notes") or []),
            metadata={k: v for k, v in meta.items() if k != "data"},
        )

    # --- connections, materialising routers ------------------------------- #
    edges: List[Tuple[str, str, str, bool]] = []
    #: routers whose feed edge (node -> its own router) has been emitted. Tracked
    #: separately from component existence: a caller may pre-declare the router
    #: node to attach its source code, and the feed edge must still be created —
    #: without it the node and its router are disconnected and every loop through
    #: the router silently disappears from the analysis.
    router_fed: Set[str] = set()
    for src, dst, label, conditional in raw.edges:
        if conditional and materialise_routers:
            rid = f"{src}::router"
            if rid not in model.components:
                # The routing function is not the node's function: LangGraph keeps
                # it in add_conditional_edges(), and the drawable graph does not
                # carry it. Attributing the node's source to the router would give
                # the router the node's resources (and put it in the model's
                # common-cause group), so it starts with no source of its own —
                # pass node_functions={"<node>::router": router} to analyse it.
                router_src = source_of_function((node_functions or {}).get(rid))
                model.components[rid] = Component(
                    id=rid,
                    label=f"{model.components[src].label} router",
                    role=Role.ROUTER,
                    source_code=router_src,
                    resources=detect_resources(router_src, globals_ns) if router_src else {},
                    notes=[
                        "Materialised from add_conditional_edges(): the routing "
                        "function is analysed as a component in its own right."
                    ],
                )
            if rid not in router_fed:
                edges.append((src, rid, "", False))
                router_fed.add(rid)
            model.components[rid].branches.append(label or dst)
            edges.append((rid, dst, label, True))
        else:
            edges.append((src, dst, label, conditional))

    # --- ports ------------------------------------------------------------- #
    preds: Dict[str, List[int]] = {}
    for index, (src, dst, label, conditional) in enumerate(edges):
        preds.setdefault(dst, []).append(index)

    # One input port per incoming connection, with a unique name. Two branches
    # of the same router reaching END are distinct connections and must not
    # share a port: a shared name would collapse them into one deviation and
    # miscount the fan-in.
    port_of_edge: Dict[int, str] = {}
    for dst, indices in preds.items():
        if len(indices) == 1:
            port_of_edge[indices[0]] = "in"
            continue
        used: Set[str] = set()
        for index in indices:
            src, _dst, label, _cond = edges[index]
            name = f"in@{src}"
            if name in used:
                suffix = _clean_label(label) or str(index)
                name = f"in@{src}:{suffix}"
                counter = 2
                while name in used:
                    name = f"in@{src}:{suffix}{counter}"
                    counter += 1
            used.add(name)
            port_of_edge[index] = name

    # A node is on a parallel branch when one of its predecessors fans out to
    # several successors unconditionally: those branches run in the same
    # LangGraph super-step and therefore compete for the same state channels.
    fan_out_parents = {
        src for src in {e[0] for e in edges}
        if sum(1 for e in edges if e[0] == src and not e[3]) > 1
    }
    parallel_nodes = {e[1] for e in edges if e[0] in fan_out_parents and not e[3]}

    for cid, comp in model.components.items():
        incoming = preds.get(cid, [])
        comp.ports_in = [port_of_edge[i] for i in incoming]
        if comp.role is Role.ROUTER:
            comp.ports_out = [f"out@{b}" for b in comp.branches] or ["out"]
        else:
            comp.ports_out = ["out"] if any(e[0] == cid for e in edges) else []
        if comp.role is Role.SINK and not comp.ports_in:
            comp.ports_in = ["in"]

    for index, (src, dst, label, conditional) in enumerate(edges):
        incoming = preds.get(dst, [])
        dst_port = port_of_edge[index]
        src_comp = model.components[src]
        if src_comp.role is Role.ROUTER:
            src_port = f"out@{label or dst}"
            if src_port not in src_comp.ports_out:
                src_port = src_comp.ports_out[0]
        else:
            src_port = "out"
        model.connections.append(
            Connection(
                src=src,
                src_port=src_port,
                dst=dst,
                dst_port=dst_port,
                label=label,
                conditional=conditional,
                fan_in=len(incoming) > 1,
                parallel=src in parallel_nodes,
            )
        )

    # Aggregator promotion: a fan-in LLM node is an aggregator in practice.
    for comp in model.components.values():
        if comp.role is Role.LLM_AGENT and len(comp.ports_in) >= 2:
            if any(k in comp.id.lower() for k in _AGGREGATOR_NAMES):
                comp.role = Role.AGGREGATOR

    return model


def extract_architecture(
    graph_like: Any,
    name: str = "langgraph_system",
    role_overrides: Optional[Dict[str, Role | str]] = None,
    resource_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    globals_ns: Optional[Dict[str, Any]] = None,
    node_functions: Optional[Dict[str, Callable[..., Any]]] = None,
    materialise_routers: bool = True,
) -> SystemModel:
    """Extract a :class:`SystemModel` from a LangGraph object or mermaid text.

    This is the entry point that replaces ``graph.get_graph().draw_mermaid_png()``
    as the source of truth for the analysis::

        model = extract_architecture(graph, globals_ns=globals())
    """
    if isinstance(graph_like, SystemModel):
        return graph_like
    if isinstance(graph_like, dict):
        raw = raw_from_spec({"name": name, **graph_like})
    elif isinstance(graph_like, str):
        raw = parse_mermaid(graph_like, name=name)
    else:
        try:
            drawable = _drawable(graph_like)
            raw = _raw_from_drawable(drawable, name)
            if not raw.edges and hasattr(drawable, "draw_mermaid"):
                raw = parse_mermaid(drawable.draw_mermaid(), name=name)
        except TypeError:
            raise
    return build_system_model(
        raw,
        role_overrides=role_overrides,
        resource_overrides=resource_overrides,
        globals_ns=globals_ns,
        node_functions=node_functions,
        materialise_routers=materialise_routers,
    )
