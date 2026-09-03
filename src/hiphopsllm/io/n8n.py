"""
hiphopsllm.io.n8n — read an n8n workflow export into the architecture model.

An n8n workflow export is a JSON document with three parts that matter here::

    {"name": ...,
     "nodes":       [{"name": ..., "type": ..., "parameters": {...},
                      "credentials": {...}}, ...],
     "connections": {"<source node name>": {"<kind>": [[{"node": ...}, ...]]}}}

The translation performed here is *not* a syntactic rename.  Three decisions
shape every fault tree that comes out of it, and each is recorded on the block
it applies to so the analyst can see it and argue with it:

1.  **Not every n8n node is a component.**  A sticky note has no runtime
    behaviour, and a language-model sub-node is not a step in the flow: it is the
    *resource* an agent generates with.  Folding ``lmChatOpenAi`` into the
    agent's ``resources["llm"]`` is what lets two agents that share one model
    form a common-cause group instead of looking independent
    (:meth:`SystemModel.common_cause_groups`).  Modelled as separate LLM
    components they would each carry their own hallucination event and the
    shared snapshot would vanish from the analysis.

2.  **The ``ai_*`` connections run backwards.**  n8n draws a tool, a memory or a
    model *into* the agent, so the arrow in the JSON points from the sub-node to
    the agent.  For a memory or a parser that is also the direction failures
    propagate, and the edge is kept as-is.  For a **tool** it is not: the agent
    decides to call the tool, so the invocation runs agent to tool, the
    observation comes back (a loop, cut by :func:`make_acyclic`), and an outward
    action such as sending an email is delivered at the system boundary.

3.  **A branching n8n node keeps its own branch logic.**  In LangGraph the
    routing function is anonymous and is materialised as a ``<node>::router``
    component.  An n8n ``If`` or ``Switch`` *is* the router, so it is given
    :class:`Role.ROUTER` directly and its outgoing edges are not marked
    conditional, which would otherwise create a second, empty router beside it.

Everything else follows the ordinary pipeline: annotate, synthesise, quantify.

    >>> from hiphopsllm.io.n8n import load_n8n
    >>> wf = load_n8n("Gmail Agent.json")           # doctest: +SKIP
    >>> print(wf.ledger_markdown())                 # doctest: +SKIP
    >>> report = wf.analyse()                       # doctest: +SKIP
    >>> report.cut_sets("H2")                       # doctest: +SKIP
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..architecture.acyclic import make_acyclic
from ..architecture.model import Role, SystemModel, extract_architecture
from ..faulttree.failure import (
    OR,
    BasicEvent,
    ComponentFailureLogic,
    DevRef,
    Deviation,
    FClass,
    annotate_llm_agent,
    annotate_tool,
)

__all__ = [
    "N8nBlock",
    "N8nWorkflow",
    "load_n8n",
    "n8n_to_spec",
    "analyse_n8n",
    "n8n_study",
    "RULES",
    "SIDE_EFFECT_SERVICES",
    "READ_ONLY_OPERATIONS",
]

END_ID = "__end__"

#: n8n connection kinds whose sub-node supplies a capability rather than data
#: flowing along the main path.
AI_KINDS = (
    "ai_languageModel",
    "ai_memory",
    "ai_tool",
    "ai_embedding",
    "ai_outputParser",
    "ai_vectorStore",
    "ai_retriever",
    "ai_document",
    "ai_textSplitter",
    "ai_chain",
)

#: Services whose nodes act on the world.  Reaching one of these is not an
#: internal state change: an email leaves, a row is written, a message is posted.
SIDE_EFFECT_SERVICES = (
    "gmail", "emailsend", "microsoftoutlook", "slack", "telegram", "discord",
    "whatsapp", "twilio", "googlesheets", "googledrive", "googlecalendar",
    "notion", "airtable", "postgres", "mysql", "mongodb", "redis", "s3",
    "awss3", "hubspot", "salesforce", "jira", "github", "gitlab", "trello",
    "asana", "clickup", "stripe", "shopify", "webhookresponse", "respondtowebhook",
    "executecommand", "ssh", "ftp", "http request",
)

#: Operations that only read.  An n8n node's ``operation`` parameter is the only
#: honest signal available without executing the workflow; anything not on this
#: list is treated as acting on the world, because under-calling a side effect is
#: the more dangerous mistake.
READ_ONLY_OPERATIONS = (
    "get", "getall", "getmany", "read", "search", "lookup", "download",
    "list", "query", "select", "find",
)

_FROM_AI = re.compile(r"\$fromAI\s*\(")
_EXPRESSION = re.compile(r"\{\{")


# --------------------------------------------------------------------------- #
# Classification rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    """One classification rule: how a family of n8n node types is modelled.

    ``kind`` is ``"component"`` (it becomes a block of the architecture),
    ``"resource"`` (it is folded into the resources of the node it serves) or
    ``"excluded"`` (it has no runtime behaviour at all).
    """

    name: str
    test: Callable[[str, Dict[str, Any]], bool]
    kind: str
    role: Optional[Role]
    resource_kind: Optional[str] = None
    why: str = ""
    #: ``False`` where the node type never acts outside the workflow, whatever
    #: service name appears in its type. A chat-memory node writes to Postgres,
    #: but that write is internal state, not an outward action, and treating it
    #: as one would put it at the delivery boundary and inflate every hazard.
    acts: Optional[bool] = None


def _short(node_type: str) -> str:
    """``@n8n/n8n-nodes-langchain.lmChatOpenAi`` -> ``lmchatopenai``."""
    return node_type.rsplit(".", 1)[-1].lower()


def _is(*names: str) -> Callable[[str, Dict[str, Any]], bool]:
    lowered = tuple(n.lower() for n in names)
    return lambda t, node: _short(t) in lowered


def _endswith(*suffixes: str) -> Callable[[str, Dict[str, Any]], bool]:
    lowered = tuple(s.lower() for s in suffixes)
    return lambda t, node: _short(t).endswith(lowered)


def _startswith(*prefixes: str) -> Callable[[str, Dict[str, Any]], bool]:
    lowered = tuple(p.lower() for p in prefixes)
    return lambda t, node: _short(t).startswith(lowered)


#: Ordered; the first rule that matches wins.  The order matters: ``gmailTool``
#: must be read as a tool before it is read as a Gmail node, and a trigger must
#: be read as the boundary before anything else claims it.
RULES: Tuple[Rule, ...] = (
    Rule(
        "sticky note", _is("stickynote"), "excluded", None,
        why="A sticky note is documentation drawn on the canvas. It has no "
            "inputs, no outputs and no runtime behaviour, so it cannot fail and "
            "it is excluded from the architecture rather than given empty logic.",
    ),
    Rule(
        "trigger", lambda t, n: _short(t).endswith("trigger")
        or _short(t) in ("start", "cron", "webhook", "interval", "emailreadimap"),
        "component", Role.SOURCE,
        why="The trigger is the system boundary: it is where a task enters. Its "
            "failure modes are boundary modes (no task arrives when one should, "
            "or an ill-posed task arrives), not internal ones, so it is given the "
            "SOURCE annotation and no processing logic.",
    ),
    Rule(
        "language model sub-node",
        lambda t, n: _short(t).startswith(("lmchat", "lmopen", "lmcohere", "lmollama"))
        or _short(t).startswith("lm"),
        "resource", None, resource_kind="llm",
        why="A model sub-node is not a step in the flow, it is the resource the "
            "agent generates with. Folding it into resources['llm'] is what makes "
            "two agents on one model a common-cause group; modelled as its own "
            "component it would carry a second hallucination event and the shared "
            "snapshot would disappear from the analysis.",
    ),
    Rule(
        "embeddings sub-node", _startswith("embeddings"), "resource", None,
        resource_kind="embedding",
        why="Same argument as the language model: an embedding model is a shared "
            "resource of whatever retrieves with it, and two retrievers on one "
            "embedding model are not independent.",
    ),
    Rule(
        "memory sub-node", _startswith("memory"), "component", Role.TOOL, acts=False,
        why="Memory is a retrieval step the agent depends on. It has two "
            "distinct modes: the store is unreachable and nothing comes back "
            "(omission), or the wrong conversation comes back because the session "
            "key collides (a plausible, undetectable value deviation). The TOOL "
            "annotation carries exactly that pair.",
    ),
    Rule(
        "output parser sub-node", _startswith("outputparser"), "component", Role.TRANSFORM,
        acts=False,
        why="A parser is deterministic: it either fails loudly or it corrupts the "
            "payload. That is the TRANSFORM annotation.",
    ),
    Rule(
        "vector store / retriever sub-node",
        _startswith("vectorstore", "retriever", "documentdefaultdataloader", "textsplitter"),
        "component", Role.TOOL, acts=False,
        why="Retrieval supplies evidence the agent will trust. Returning nothing "
            "is an omission; returning the wrong passage is a subtle value "
            "deviation the agent cannot detect, which is the TOOL annotation.",
    ),
    Rule(
        "agent / LLM chain",
        lambda t, n: _short(t) in ("agent", "chainllm", "conversationalagent",
                                   "openaiassistant", "chainsummarization", "informationextractor",
                                   "textclassifier", "sentimentanalysis"),
        "component", Role.LLM_AGENT,
        why="A node whose output is produced by a language model. It gets the "
            "full LLM annotation: hallucination and sampling non-determinism as "
            "subtle value deviations, format violation and truncation as coarse "
            "ones, empty generation and context overflow as omissions, plus "
            "latency. It is also transparent to a subtle deviation arriving at "
            "its input: it has no way to detect one.",
    ),
    Rule(
        "tool node", _endswith("tool"), "component", Role.TOOL,
        why="A node attached to an agent by an ai_tool connection. The agent, not "
            "the flow, decides when to call it, so the invocation edge runs from "
            "the agent to the tool and the observation comes back as a loop.",
    ),
    Rule(
        "code node", _is("code", "function", "functionitem"), "component", Role.TOOL,
        why="Hand-written code executed inside the workflow. Parse failure, "
            "raised exception and 'runs cleanly but computes the wrong thing' are "
            "the three TOOL modes, and the third is the one nothing downstream "
            "can catch.",
    ),
    Rule(
        "branch node", _is("if", "switch", "filter"), "component", Role.ROUTER,
        why="The branch decision lives inside this node, unlike LangGraph where "
            "it lives in an anonymous callable. It is therefore given the ROUTER "
            "annotation directly (no branch matched, wrong branch taken, early "
            "termination) and its outgoing edges are not marked conditional, "
            "which would create a second empty router beside it.",
    ),
    Rule(
        "merge node", _is("merge", "aggregate", "compareDatasets".lower()),
        "component", Role.AGGREGATOR,
        why="A fan-in node. This is the only annotation in the library that "
            "expresses redundancy: omission and value deviations need ALL inputs "
            "to deviate, unless the selection itself is wrong. Any common cause "
            "shared by the inputs collapses that AND gate back to a single point "
            "of failure.",
    ),
    Rule(
        "HTTP request", _is("httprequest", "graphql", "webhookresponse", "respondtowebhook"),
        "component", Role.TOOL,
        why="An external call. It can return nothing, return an error body that "
            "the flow reads as data, or return a well-formed wrong answer.",
    ),
    Rule(
        "data shaping", _is("set", "noop", "splitout", "splitinbatches", "itemlists",
                            "datetime", "renamekeys", "editfields", "removeduplicates",
                            "limit", "sort", "html", "markdown", "extractfromfile"),
        "component", Role.TRANSFORM,
        why="Deterministic shaping of the payload. Two modes: it throws (omission) "
            "or it writes a malformed value (coarse value deviation).",
    ),
    Rule(
        "service node",
        lambda t, n: any(s in _short(t) for s in SIDE_EFFECT_SERVICES),
        "component", Role.TOOL,
        why="A node that talks to an outside service. Modelled as a tool because "
            "its failure modes are a tool's: no result, an error surfaced as a "
            "result, or a clean call that did the wrong thing.",
    ),
)

_FALLBACK = Rule(
    "unrecognised", lambda t, n: True, "component", Role.TRANSFORM,
    why="This node type is not in the rule table, so it is modelled "
        "conservatively as a deterministic transform: it can fail to produce "
        "output, or corrupt what it passes on. If it generates with a model, "
        "calls a service or branches, say so with role_overrides — the default "
        "will understate it.",
)


def classify(node: Dict[str, Any]) -> Rule:
    """Return the first rule matching this n8n node."""
    node_type = str(node.get("type", ""))
    for rule in RULES:
        if rule.test(node_type, node):
            return rule
    return _FALLBACK


# --------------------------------------------------------------------------- #
# Parameter inspection
# --------------------------------------------------------------------------- #
def _walk_strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: List[str] = []
        for key, item in value.items():
            out.extend(_walk_strings(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_walk_strings(item))
        return out
    return []


def _parameter_text(node: Dict[str, Any]) -> str:
    return "\n".join(_walk_strings(node.get("parameters", {})))


def _operation(node: Dict[str, Any]) -> str:
    params = node.get("parameters") or {}
    return str(params.get("operation", "") or "").lower()


def _model_name(node: Dict[str, Any]) -> str:
    """Best-effort model identifier of a language-model sub-node."""
    params = node.get("parameters") or {}
    model = params.get("model")
    if isinstance(model, dict):
        return str(model.get("value") or model.get("cachedResultName") or "model")
    if isinstance(model, str) and model:
        return model
    for key in ("modelName", "modelId", "deploymentName"):
        if params.get(key):
            return str(params[key])
    return _short(str(node.get("type", "model")))


def _credentials(node: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kind, meta in (node.get("credentials") or {}).items():
        if isinstance(meta, dict):
            out[str(kind)] = str(meta.get("name") or meta.get("id") or kind)
        else:
            out[str(kind)] = str(meta)
    return out


def _is_side_effect(node: Dict[str, Any], rule: Rule, role: Optional[Role]) -> bool:
    """Does reaching this node act on the world?

    Read-only operations are exempted, but only when the node says so. Anything
    unstated counts as acting, because treating a send as a read hides the one
    failure mode that cannot be undone.
    """
    if rule.acts is not None:
        return bool(rule.acts)
    if role is not Role.TOOL:
        return False
    node_type = _short(str(node.get("type", "")))
    base = node_type[: -len("tool")] if node_type.endswith("tool") else node_type
    if not any(service in base for service in SIDE_EFFECT_SERVICES):
        if base in ("code", "function", "functionitem", "executecommand", "ssh"):
            return True
        return False
    operation = _operation(node)
    if operation and operation in READ_ONLY_OPERATIONS:
        return False
    return True


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
@dataclass
class N8nBlock:
    """One n8n node, with the modelling decision taken for it and its reason."""

    name: str
    node_type: str
    type_version: str
    rule: Rule
    kind: str                                  # component | resource | excluded
    role: Optional[Role]
    resources: Dict[str, str] = field(default_factory=dict)
    credentials: Dict[str, str] = field(default_factory=dict)
    attached_to: List[str] = field(default_factory=list)   # resources / sub-nodes
    connection_kind: str = "main"
    side_effect: bool = False
    model_authored_args: bool = False
    notes: List[str] = field(default_factory=list)
    source_code: str = ""

    @property
    def role_name(self) -> str:
        if self.kind == "resource":
            return f"resource:{self.rule.resource_kind}"
        if self.kind == "excluded":
            return "excluded"
        return str(self.role)

    def flags(self) -> List[str]:
        out = []
        if self.side_effect:
            out.append("acts on the world")
        if self.model_authored_args:
            out.append("arguments written by the model ($fromAI)")
        if self.credentials:
            out.append("holds credentials")
        return out


# --------------------------------------------------------------------------- #
# The workflow
# --------------------------------------------------------------------------- #
@dataclass
class N8nWorkflow:
    """An n8n export, translated into a specification the analysis can read."""

    name: str
    raw: Dict[str, Any]
    blocks: Dict[str, N8nBlock] = field(default_factory=dict)
    edges: List[Tuple[str, str, str, bool]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    tool_feedback: bool = True
    host_resource: bool = True

    # -- specification ------------------------------------------------------ #
    def to_spec(self) -> Dict[str, Any]:
        """The dict specification consumed by :func:`extract_architecture`."""
        nodes: Dict[str, Any] = {}
        for name, block in self.blocks.items():
            if block.kind != "component":
                continue
            nodes[name] = {
                "label": name,
                "role": block.role.value if block.role else Role.TRANSFORM.value,
                "resources": dict(block.resources),
                "source_code": block.source_code,
                "notes": list(block.notes),
                "n8n_type": block.node_type,
            }
        nodes[END_ID] = {"label": "delivered", "role": Role.SINK.value, "resources": {}}
        return {"name": self.name, "nodes": nodes, "edges": list(self.edges)}

    def system(self, **kwargs: Any) -> SystemModel:
        return extract_architecture(self.to_spec(), name=self.name, **kwargs)

    # -- the annotation ledger --------------------------------------------- #
    def ledger(self) -> List[Dict[str, str]]:
        """One row per n8n node: what it was modelled as, and why."""
        rows: List[Dict[str, str]] = []
        for name, block in self.blocks.items():
            rows.append(
                {
                    "n8n node": name,
                    "n8n type": block.node_type.rsplit(".", 1)[-1],
                    "connection": block.connection_kind,
                    "modelled as": block.role_name,
                    "rule": block.rule.name,
                    "resources": ", ".join(f"{k}={v}" for k, v in sorted(block.resources.items())) or "-",
                    "flags": "; ".join(block.flags()) or "-",
                    "why": block.rule.why,
                }
            )
        return rows

    def ledger_frame(self):
        import pandas as pd

        return pd.DataFrame(self.ledger())

    def ledger_markdown(self) -> str:
        lines = [f"### Annotation ledger for `{self.name}`", ""]
        for row in self.ledger():
            lines.append(f"**{row['n8n node']}**  (`{row['n8n type']}`, {row['connection']})")
            lines.append("")
            lines.append(f"- modelled as: **{row['modelled as']}**  (rule: {row['rule']})")
            if row["resources"] != "-":
                lines.append(f"- resources: {row['resources']}")
            if row["flags"] != "-":
                lines.append(f"- flags: {row['flags']}")
            lines.append(f"- why: {row['why']}")
            lines.append("")
        if self.notes:
            lines.append("### Structural notes")
            lines.append("")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)

    # -- failure logic the library does not supply -------------------------- #
    def extra_logic(self, system: SystemModel) -> Dict[str, ComponentFailureLogic]:
        """Commission logic for a workflow that can act on the world.

        The library's annotations describe a system that *answers*. An n8n
        workflow with a Gmail tool also *does* things, and the failure that
        matters most there has no analogue in a question-answering pipeline: the
        agent acts when it should have stayed silent. That is a commission, and
        commission is the one guideword the default annotations leave empty for
        an agent, so it is added here rather than left to be discovered later.
        """
        extra: Dict[str, ComponentFailureLogic] = {}
        acting_agents = self._acting_agents()

        for cid, agent_tools in acting_agents.items():
            comp = system.components.get(cid)
            if comp is None or comp.role is not Role.LLM_AGENT:
                continue
            cfl = annotate_llm_agent(comp)
            ref = cfl.add_event(
                BasicEvent(
                    id=f"BE-{cid}-POLICY",
                    component=cid,
                    label="Acts outside the remit stated in its system message",
                    fclass=FClass.COMMISSION,
                    prob=0.07,
                    kind="internal",
                    rationale=(
                        "The remit ('reply only to business enquiries', 'never touch "
                        "personal mail') is expressed in prose inside the prompt, so it "
                        "is a preference of the model, not a constraint on the system. "
                        f"Nothing between this node and {', '.join(agent_tools)} can "
                        "refuse a call the model decides to make."
                    ),
                    mitigation=(
                        "Move the remit out of the prompt: a deterministic classifier "
                        "or filter node ahead of the agent, or a human approval step on "
                        "the acting tool. A prompt instruction is not an interlock."
                    ),
                )
            )
            for port in comp.ports_out or ["out"]:
                dev = Deviation(cid, port, FClass.COMMISSION)
                cfl.logic[dev] = OR(cfl.logic.get(dev), ref)
            cfl.notes.append(
                "This agent holds tools that act on the world, so its output "
                "deviations are not confined to the answer: a commission here "
                "leaves the system as an email."
            )
            extra[cid] = cfl

        for name, block in self.blocks.items():
            if not block.side_effect:
                continue
            comp = system.components.get(name)
            if comp is None or comp.role is not Role.TOOL:
                continue
            cfl = annotate_tool(comp)
            ref = cfl.add_event(
                BasicEvent(
                    id=f"BE-{name}-UNSOLICITED",
                    component=name,
                    label="Outward action performed that was not warranted",
                    fclass=FClass.COMMISSION,
                    prob=0.03,
                    kind="internal",
                    rationale=(
                        "The node performs an irreversible external action "
                        f"({block.node_type.rsplit('.', 1)[-1]}"
                        + (", with arguments written by the model" if block.model_authored_args else "")
                        + "). It executes whatever call reaches it; it has no view of "
                        "whether the call should have been made."
                    ),
                    mitigation=(
                        "Draft-then-approve rather than send; an allow-list of "
                        "recipients; rate limits that make a runaway loop visible "
                        "before it is expensive."
                    ),
                )
            )
            inbound = [DevRef(name, port, FClass.COMMISSION) for port in comp.ports_in]
            for port in comp.ports_out or ["out"]:
                dev = Deviation(name, port, FClass.COMMISSION)
                cfl.logic[dev] = OR(cfl.logic.get(dev), ref, *inbound)
            if block.model_authored_args:
                cfl.notes.append(
                    "The arguments of this call are authored by the language model "
                    "($fromAI), so a subtle value deviation upstream becomes the "
                    "content of a real outward action."
                )
            extra[name] = cfl
        return extra

    def _acting_agents(self) -> Dict[str, List[str]]:
        """Agents that own at least one tool which acts on the world."""
        out: Dict[str, List[str]] = {}
        for name, block in self.blocks.items():
            if block.side_effect and block.connection_kind == "ai_tool":
                for parent in block.attached_to:
                    out.setdefault(parent, []).append(name)
        return out

    # -- hazards ------------------------------------------------------------ #
    def hazards(self, system: SystemModel) -> List[Any]:
        """The hazard list, reworded for a workflow that acts on the world.

        The library generates one commission top event per tool and words it as
        unsafe code execution, because that is what commission means for a
        LangGraph tool. In n8n it usually means an outward action: an email
        sent, a row written, a message posted. Two changes are made here, both
        before synthesis so that the FMEA and the effect columns carry them:

        * a commission hazard on a node that acts outside the workflow is
          renamed and marked critical;
        * a commission hazard on a node with no commission logic behind it is
          dropped. Reporting it at P = 0 would read as a quantified claim when
          it is an absence of modelling.
        """
        from ..faulttree.synthesis import default_hazards

        kept: List[Any] = []
        dropped: List[str] = []
        for hazard in default_hazards(system):
            if not hazard.id.startswith("H5-"):
                kept.append(hazard)
                continue
            target = hazard.id[len("H5-"):]
            block = self.blocks.get(target)
            component = system.components.get(target)
            executes_code = bool(
                component is not None
                and re.search(r"\beval\s*\(|\bexec\s*\(", component.source_code or "")
            )
            if block is not None and block.side_effect:
                hazard.name = f"Unsolicited outward action by {target}"
                hazard.severity = "critical"
                hazard.description = (
                    "The workflow acts on the world when it should not have: an "
                    "email leaves, a record is written, a message is posted. The "
                    "action is not visible in the workflow's own output, and for "
                    "an outbound message it cannot be withdrawn."
                )
                hazard.detection = (
                    "Not detectable inside the workflow. Only an outside observer "
                    "(the recipient, an audit of the sent folder) sees it."
                )
                kept.append(hazard)
            elif executes_code:
                kept.append(hazard)
            else:
                dropped.append(target)
        if dropped:
            note = (
                "No commission hazard was raised for "
                + ", ".join(sorted(dropped))
                + ": these nodes do not act outside the workflow and do not "
                "execute model-authored code, so there is nothing for a "
                "commission top event to decompose."
            )
            if note not in self.notes:
                self.notes.append(note)
        return kept

    # -- one-call pipeline -------------------------------------------------- #
    def analyse(self, unroll: int = 1, **kwargs: Any) -> Any:
        """Extract, annotate (with the n8n commission logic) and synthesise."""
        from ..report import analyse_langgraph

        spec = self.to_spec()
        system = extract_architecture(spec, name=self.name)
        acyclic, _ = make_acyclic(system, unroll=unroll)
        extra = self.extra_logic(acyclic)
        merged = dict(kwargs.pop("extra_logic", {}) or {})
        merged.update(extra)
        kwargs.setdefault("hazards", self.hazards(acyclic))
        return analyse_langgraph(
            spec, name=self.name, unroll=unroll, extra_logic=merged, **kwargs
        )

    def study(self, unroll: int = 1, **kwargs: Any) -> Any:
        """An :class:`AgenticReliabilityStudy` with the analysis already run."""
        from ..pipeline import AgenticReliabilityStudy

        profile = kwargs.pop("profile", None)
        spec = self.to_spec()
        study = AgenticReliabilityStudy(
            spec, name=self.name, profile=profile, unroll=unroll
        )
        system = extract_architecture(spec, name=self.name)
        acyclic, _ = make_acyclic(system, unroll=unroll)
        # The study passes its own ``hazards`` attribute through to
        # analyse_langgraph, so the list is set on the study rather than handed
        # to analyse() as a keyword, which would arrive twice.
        if study.hazards is None:
            study.hazards = self.hazards(acyclic)
        study.report = study.analyse(extra_logic=self.extra_logic(acyclic), **kwargs)
        return study

    # -- reporting ---------------------------------------------------------- #
    def summary(self) -> str:
        components = [b for b in self.blocks.values() if b.kind == "component"]
        resources = [b for b in self.blocks.values() if b.kind == "resource"]
        excluded = [b for b in self.blocks.values() if b.kind == "excluded"]
        acting = [b.name for b in components if b.side_effect]
        lines = [
            f"n8n workflow: {self.name}",
            f"  {len(self.blocks)} nodes -> {len(components)} components, "
            f"{len(resources)} folded in as resources, {len(excluded)} excluded",
            f"  {len(self.edges)} edges (including the delivery boundary)",
        ]
        if acting:
            lines.append(f"  acts on the world through: {', '.join(acting)}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Reading the export
# --------------------------------------------------------------------------- #
def _load_json(source: Any) -> Dict[str, Any]:
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        text = str(source)
        if text.lstrip().startswith("{"):
            return json.loads(text)
        return json.loads(Path(text).read_text(encoding="utf-8"))
    raise TypeError(f"expected a path, a JSON string or a dict; got {type(source)!r}")


def load_n8n(
    source: Any,
    name: Optional[str] = None,
    *,
    tool_feedback: bool = True,
    host_resource: bool = True,
    role_overrides: Optional[Dict[str, Role | str]] = None,
) -> N8nWorkflow:
    """Read an n8n workflow export.

    Parameters
    ----------
    source
        Path to the exported ``.json``, the JSON text itself, or the parsed dict.
    tool_feedback
        Keep the observation edge from a tool back to its agent. It closes a
        loop, which :func:`make_acyclic` cuts, adding the two failures a bounded
        loop introduces on its own: not converging, and the latency of iterating.
        Set ``False`` for a strictly feed-forward reading.
    host_resource
        Give every component a shared ``runtime`` resource named after the n8n
        instance. It is true (one process, one host) and it makes the host a
        visible common cause instead of an unstated assumption.
    role_overrides
        Force the role of a node the rule table gets wrong.
    """
    raw = _load_json(source)
    if "nodes" not in raw:
        raise ValueError(
            "this JSON has no 'nodes' key, so it is not an n8n workflow export. "
            "Export with the workflow menu -> Download."
        )
    workflow_name = name or str(raw.get("name") or "n8n workflow")
    overrides = {k: (v if isinstance(v, Role) else Role(v))
                 for k, v in (role_overrides or {}).items()}

    wf = N8nWorkflow(
        name=workflow_name, raw=raw, tool_feedback=tool_feedback,
        host_resource=host_resource,
    )

    # --- blocks ----------------------------------------------------------- #
    for node in raw.get("nodes") or []:
        node_name = str(node.get("name") or node.get("id") or "node")
        rule = classify(node)
        role = overrides.get(node_name, rule.role)
        kind = rule.kind
        if node_name in overrides:
            kind = "component"
        block = N8nBlock(
            name=node_name,
            node_type=str(node.get("type", "")),
            type_version=str(node.get("typeVersion", "")),
            rule=rule,
            kind=kind,
            role=role,
            credentials=_credentials(node),
            side_effect=(kind == "component" and _is_side_effect(node, rule, role)),
            model_authored_args=bool(_FROM_AI.search(_parameter_text(node))),
        )
        if node_name in overrides:
            block.notes.append(
                f"Role forced to {role} by role_overrides; the rule table said "
                f"{rule.role}."
            )
        code = (node.get("parameters") or {}).get("jsCode") or (node.get("parameters") or {}).get("functionCode")
        if isinstance(code, str):
            block.source_code = code
        for credential_kind, credential_name in block.credentials.items():
            block.resources[f"credential:{credential_kind}"] = credential_name
        wf.blocks[node_name] = block

    if wf.host_resource:
        instance = str((raw.get("meta") or {}).get("instanceId") or "n8n")[:12]
        for block in wf.blocks.values():
            if block.kind == "component" and block.role not in (Role.SOURCE, Role.SINK):
                block.resources.setdefault("runtime", f"n8n:{instance}")

    # --- connections ------------------------------------------------------- #
    edges: List[Tuple[str, str, str, bool]] = []
    has_outgoing: Dict[str, bool] = {name: False for name in wf.blocks}

    for source_name, kinds in (raw.get("connections") or {}).items():
        if source_name not in wf.blocks:
            continue
        source_block = wf.blocks[source_name]
        for connection_kind, branches in (kinds or {}).items():
            branch_labels = _branch_labels(raw, source_name, len(branches or []))
            for index, targets in enumerate(branches or []):
                for target in targets or []:
                    target_name = str((target or {}).get("node", ""))
                    if target_name not in wf.blocks:
                        continue
                    label = branch_labels[index] if index < len(branch_labels) else ""
                    _wire(wf, edges, has_outgoing, source_block, wf.blocks[target_name],
                          connection_kind, label)

    # --- delivery boundary ------------------------------------------------- #
    for name, block in wf.blocks.items():
        if block.kind != "component" or block.role is Role.SOURCE:
            continue
        if block.side_effect:
            edges.append((name, END_ID, "outward action", False))
        elif not has_outgoing.get(name):
            edges.append((name, END_ID, "", False))

    if not any(dst == END_ID for _, dst, _, _ in edges):
        # Every component feeds another one, so nothing is delivered anywhere.
        # Anchoring the hazards on an arbitrary node would be worse than saying so.
        wf.notes.append(
            "No node delivers anything outside the workflow and every node has a "
            "successor, so there is no system boundary to anchor the hazards on. "
            "Add the delivering node explicitly."
        )

    wf.edges = edges
    _record_structural_notes(wf)
    return wf


def _branch_labels(raw: Dict[str, Any], source_name: str, count: int) -> List[str]:
    """Labels for a node's main output branches (``true``/``false``, or rules)."""
    if count <= 1:
        return [""]
    node = next((n for n in raw.get("nodes") or [] if n.get("name") == source_name), {})
    short = _short(str(node.get("type", "")))
    if short in ("if", "filter"):
        return ["true", "false"][:count] + [""] * max(0, count - 2)
    params = node.get("parameters") or {}
    rules = ((params.get("rules") or {}).get("values")
             or (params.get("rules") or {}).get("rules") or [])
    labels = []
    for index in range(count):
        if index < len(rules) and isinstance(rules[index], dict):
            labels.append(str(rules[index].get("outputKey") or f"branch {index}"))
        else:
            labels.append(f"branch {index}")
    return labels


def _wire(
    wf: N8nWorkflow,
    edges: List[Tuple[str, str, str, bool]],
    has_outgoing: Dict[str, bool],
    source: N8nBlock,
    target: N8nBlock,
    connection_kind: str,
    label: str,
) -> None:
    """Add the edges implied by one n8n connection.

    The direction of an ``ai_*`` connection in the JSON is 'sub-node into the
    node that uses it'. That is the direction failures propagate for a memory or
    a parser, and the reverse of it for a tool, which the agent decides to call.
    """
    if connection_kind == "main":
        if source.kind != "component" or target.kind != "component":
            return
        conditional = bool(label) and source.role is not Role.ROUTER
        edges.append((source.name, target.name, label, conditional))
        has_outgoing[source.name] = True
        return

    # sub-node connections
    source.connection_kind = connection_kind
    if source.name not in target.attached_to:
        source.attached_to.append(target.name)

    if source.kind == "resource":
        kind = source.rule.resource_kind or "resource"
        value = _model_name(_node_by_name(wf.raw, source.name) or {})
        target.resources[kind] = value
        for credential_kind, credential_name in source.credentials.items():
            target.resources.setdefault(f"credential:{credential_kind}", credential_name)
        return

    if source.kind != "component" or target.kind != "component":
        return

    if connection_kind == "ai_tool":
        edges.append((target.name, source.name, "invoke", False))
        has_outgoing[target.name] = True
        if wf.tool_feedback:
            edges.append((source.name, target.name, "observation", False))
            has_outgoing[source.name] = True
    else:
        edges.append((source.name, target.name, connection_kind.replace("ai_", ""), False))
        has_outgoing[source.name] = True


def _node_by_name(raw: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for node in raw.get("nodes") or []:
        if node.get("name") == name:
            return node
    return None


def _record_structural_notes(wf: N8nWorkflow) -> None:
    resources = [b.name for b in wf.blocks.values() if b.kind == "resource"]
    excluded = [b.name for b in wf.blocks.values() if b.kind == "excluded"]
    if resources:
        note = f"Folded in as resources rather than components: {', '.join(resources)}."
        shared = [b.name for b in wf.blocks.values()
                  if b.kind == "resource" and len(b.attached_to) > 1]
        if shared:
            note += (
                f" {', '.join(shared)} is attached to more than one node, so those "
                "nodes share a resource and form a common-cause group: they do not "
                "fail independently."
            )
        wf.notes.append(note)
    if excluded:
        wf.notes.append(f"Excluded (no runtime behaviour): {', '.join(excluded)}.")
    tools = [b.name for b in wf.blocks.values()
             if b.connection_kind == "ai_tool" and b.kind == "component"]
    if tools and wf.tool_feedback:
        wf.notes.append(
            "Each ai_tool connection became two edges (agent to tool, tool back to "
            "agent). The loop is cut by make_acyclic, which is what adds the "
            "iteration-budget and iteration-latency events."
        )
    shared: Dict[Tuple[str, str], List[str]] = {}
    for block in wf.blocks.values():
        if block.kind != "component":
            continue
        for kind, value in block.resources.items():
            if kind.startswith("credential:"):
                shared.setdefault((kind, value), []).append(block.name)
    for (kind, value), members in shared.items():
        if len(members) > 1:
            wf.notes.append(
                f"{len(members)} nodes share {kind.split(':', 1)[1]} '{value}': "
                f"{', '.join(sorted(members))}. Revoking or rate-limiting it removes "
                "all of them at once, so they are not independent."
            )


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def n8n_to_spec(source: Any, **kwargs: Any) -> Dict[str, Any]:
    """Read an n8n export straight to a graph specification dict."""
    return load_n8n(source, **kwargs).to_spec()


def analyse_n8n(source: Any, **kwargs: Any) -> Any:
    """Read an n8n export and run the whole structural analysis on it."""
    load_kwargs = {k: kwargs.pop(k) for k in
                   ("name", "tool_feedback", "host_resource", "role_overrides")
                   if k in kwargs}
    return load_n8n(source, **load_kwargs).analyse(**kwargs)


def n8n_study(source: Any, **kwargs: Any) -> Any:
    """Read an n8n export into a study ready for :meth:`observe`."""
    load_kwargs = {k: kwargs.pop(k) for k in
                   ("name", "tool_feedback", "host_resource", "role_overrides")
                   if k in kwargs}
    return load_n8n(source, **load_kwargs).study(**kwargs)
