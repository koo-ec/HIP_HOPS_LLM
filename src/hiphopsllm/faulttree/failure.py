"""
hiphopsllm.faulttree.failure — deviations, Boolean failure expressions and the annotation
library (HiP-HOPS *Phase 1*: "annotate components with local failure logic").

In HiP-HOPS every component carries an **IF-FMEA** table: for each *output
deviation* of the component, a logical expression over *input deviations* and
*internal basic events* that explains how that output deviation arises.  Fault
trees are never drawn by hand — they are synthesised by composing these local
tables along the architecture's connections.

Deviation notation follows the HiP-HOPS convention ``<class>-<component>.<port>``::

    O-coder.out      omission of the tool's output
    VS-generator.out subtle (plausible but wrong) value deviation

Failure classes are the classical guideword set, specialised for LLM-based
agents.  The distinction that matters most for agentic systems is between
**VALUE_COARSE** (wrong *and* detectable — malformed, unparsable, schema
violation) and **VALUE_SUBTLE** (wrong but plausible — the hallucination case,
which no downstream syntactic check will catch).  They have very different
propagation behaviour and very different consequences, so they are kept apart
throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

from ..architecture.model import Component, Role, SystemModel

__all__ = [
    "FClass",
    "Deviation",
    "Expr",
    "BasicEventRef",
    "DevRef",
    "And",
    "Or",
    "Const",
    "AND",
    "OR",
    "BasicEvent",
    "ComponentFailureLogic",
    "FailureModel",
    "annotate_system",
    "entropy_to_fail_prob",
]


# --------------------------------------------------------------------------- #
# Failure classes
# --------------------------------------------------------------------------- #
class FClass(str, Enum):
    """Deviation guidewords used by the annotation library."""

    OMISSION = "O"          # expected output absent
    COMMISSION = "C"        # output produced when none should be
    VALUE_COARSE = "VC"     # wrong and syntactically detectable
    VALUE_SUBTLE = "VS"     # wrong but plausible (undetectable at run time)
    EARLY = "E"             # produced before its preconditions are met
    LATE = "L"              # produced too late / budget exhausted

    @property
    def title(self) -> str:
        return {
            "O": "Omission",
            "C": "Commission",
            "VC": "Value (coarse, detectable)",
            "VS": "Value (subtle, undetectable)",
            "E": "Early",
            "L": "Late",
        }[self.value]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


#: The classes that propagate through a component by default.
PROPAGATING = (FClass.OMISSION, FClass.VALUE_COARSE, FClass.VALUE_SUBTLE, FClass.LATE)


@dataclass(frozen=True, order=True)
class Deviation:
    """A failure of class ``fclass`` observed at ``component.port``."""

    component: str
    port: str
    fclass: FClass

    def __str__(self) -> str:
        return f"{self.fclass.value}-{self.component}.{self.port}"

    @property
    def id(self) -> str:
        return str(self)


# --------------------------------------------------------------------------- #
# Boolean expression algebra
# --------------------------------------------------------------------------- #
class Expr:
    """Base class of the local failure-logic expression language."""

    def refs(self) -> List["DevRef"]:
        return []

    def events(self) -> List[str]:
        return []


@dataclass(frozen=True)
class Const(Expr):
    value: bool

    def __str__(self) -> str:
        return "TRUE" if self.value else "FALSE"


@dataclass(frozen=True)
class BasicEventRef(Expr):
    """Reference to an internal basic event (a leaf of the fault tree)."""

    event_id: str

    def events(self) -> List[str]:
        return [self.event_id]

    def __str__(self) -> str:
        return self.event_id


@dataclass(frozen=True)
class DevRef(Expr):
    """Reference to a deviation at one of this component's ports."""

    component: str
    port: str
    fclass: FClass

    @property
    def deviation(self) -> Deviation:
        return Deviation(self.component, self.port, self.fclass)

    def refs(self) -> List["DevRef"]:
        return [self]

    def __str__(self) -> str:
        return str(self.deviation)


@dataclass(frozen=True)
class And(Expr):
    terms: Tuple[Expr, ...]

    def refs(self) -> List[DevRef]:
        return [r for t in self.terms for r in t.refs()]

    def events(self) -> List[str]:
        return [e for t in self.terms for e in t.events()]

    def __str__(self) -> str:
        return "(" + " AND ".join(str(t) for t in self.terms) + ")"


@dataclass(frozen=True)
class Or(Expr):
    terms: Tuple[Expr, ...]

    def refs(self) -> List[DevRef]:
        return [r for t in self.terms for r in t.refs()]

    def events(self) -> List[str]:
        return [e for t in self.terms for e in t.events()]

    def __str__(self) -> str:
        return "(" + " OR ".join(str(t) for t in self.terms) + ")"


FALSE = Const(False)
TRUE = Const(True)


def _flatten(terms: Iterable[Optional[Expr]], cls) -> List[Expr]:
    out: List[Expr] = []
    for t in terms:
        if t is None:
            continue
        if isinstance(t, cls):
            out.extend(t.terms)
        else:
            out.append(t)
    return out


def OR(*terms: Optional[Expr]) -> Expr:
    """Build a simplified disjunction (flattens, drops FALSE, dedups)."""
    flat = [t for t in _flatten(terms, Or) if not (isinstance(t, Const) and not t.value)]
    if any(isinstance(t, Const) and t.value for t in flat):
        return TRUE
    unique: List[Expr] = []
    for t in flat:
        if t not in unique:
            unique.append(t)
    if not unique:
        return FALSE
    return unique[0] if len(unique) == 1 else Or(tuple(unique))


def AND(*terms: Optional[Expr]) -> Expr:
    """Build a simplified conjunction (flattens, drops TRUE, dedups)."""
    flat = [t for t in _flatten(terms, And) if not (isinstance(t, Const) and t.value)]
    if any(isinstance(t, Const) and not t.value for t in flat):
        return FALSE
    unique: List[Expr] = []
    for t in flat:
        if t not in unique:
            unique.append(t)
    if not unique:
        return TRUE
    return unique[0] if len(unique) == 1 else And(tuple(unique))


# --------------------------------------------------------------------------- #
# Basic events
# --------------------------------------------------------------------------- #
@dataclass
class BasicEvent:
    """A leaf failure: an internal fault of one component, or a shared cause.

    ``prob`` is the point estimate used for quantification.  ``prob_interval``
    optionally carries an imprecise (lower, upper) pair — useful when the
    estimate comes from a small sample, which is the normal situation for
    LLM failure rates.
    """

    id: str
    component: str
    label: str
    fclass: FClass
    prob: float = 0.05
    prob_interval: Optional[Tuple[float, float]] = None
    #: The probability this event had *before* any calibration touched it, set
    #: once by the calibrator. Calibration splits a component's measured
    #: probability over its events in proportion to their priors; without a
    #: stable baseline the second calibration would use the first one's output
    #: as weights and quietly move numbers that were already measured.
    baseline_prob: Optional[float] = None
    kind: str = "internal"          # internal | ccf | channel | loop_cut | boundary
    rationale: str = ""
    mitigation: str = ""
    evidence: str = "engineering judgement (placeholder — replace with measurement)"

    @property
    def interval(self) -> Tuple[float, float]:
        return self.prob_interval or (self.prob, self.prob)


@dataclass
class ComponentFailureLogic:
    """The IF-FMEA table of one component."""

    component: str
    role: Role
    #: output deviation -> Boolean expression over input deviations + basic events
    logic: Dict[Deviation, Expr] = field(default_factory=dict)
    events: Dict[str, BasicEvent] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def set(self, port: str, fclass: FClass, expr: Expr) -> None:
        if isinstance(expr, Const) and not expr.value:
            return
        self.logic[Deviation(self.component, port, fclass)] = expr

    def add_event(self, event: BasicEvent) -> BasicEventRef:
        self.events[event.id] = event
        return BasicEventRef(event.id)

    def table(self) -> List[Dict[str, str]]:
        return [
            {
                "output_deviation": str(dev),
                "failure_class": dev.fclass.title,
                "expression": str(expr),
            }
            for dev, expr in sorted(self.logic.items(), key=lambda kv: str(kv[0]))
        ]


@dataclass
class FailureModel:
    """The annotated system: architecture + per-component failure logic."""

    system: SystemModel
    logic: Dict[str, ComponentFailureLogic] = field(default_factory=dict)
    events: Dict[str, BasicEvent] = field(default_factory=dict)
    ccf_groups: Dict[str, List[str]] = field(default_factory=dict)
    connection_events: Dict[str, List[str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def expression(self, dev: Deviation) -> Optional[Expr]:
        cfl = self.logic.get(dev.component)
        return cfl.logic.get(dev) if cfl else None

    def event(self, eid: str) -> BasicEvent:
        return self.events[eid]


# --------------------------------------------------------------------------- #
# Probability calibration
# --------------------------------------------------------------------------- #
def entropy_to_fail_prob(
    entropy_value: Optional[float],
    ent_mid: float = 0.90,
    slope: float = 3.0,
    p_min: float = 0.01,
    p_max: float = 0.95,
) -> float:
    """Map semantic-cluster entropy to a failure probability.

    Identical calibration to the Bayesian-network cell of the source notebook,
    so the fault tree and the BN are quantified on the same scale.  Entropy is
    computed by ``HFSemanticUncertainty`` over K resamples of one prompt: high
    entropy means the agent's answer is unstable, which we read as an elevated
    probability of a subtle value deviation.
    """
    import math

    if entropy_value is None:
        return 0.10
    p = 1.0 / (1.0 + math.exp(-slope * (float(entropy_value) - ent_mid)))
    return float(max(p_min, min(p_max, p)))


# --------------------------------------------------------------------------- #
# Default probabilities (placeholders — documented as such in every report)
# --------------------------------------------------------------------------- #
DEFAULT_P: Dict[str, float] = {
    "llm.halluc": 0.15,
    "llm.format": 0.08,
    "llm.empty": 0.01,
    "llm.truncate": 0.05,
    "llm.context": 0.03,
    "llm.nondet": 0.12,
    "llm.late": 0.05,
    "tool.parse": 0.10,
    "tool.exec_error": 0.08,
    "tool.wrong_expr": 0.10,
    "tool.unsafe": 0.02,
    "router.no_match": 0.06,
    "router.misroute": 0.05,
    "router.early_stop": 0.07,
    "agg.select": 0.10,
    "agg.format": 0.05,
    "agg.empty": 0.01,
    "transform.fault": 0.03,
    "channel.loss": 0.01,
    "channel.contention": 0.05,
    "loop.cut": 0.10,
    "ccf.model": 0.08,
    "ccf.runtime": 0.01,
    "boundary.input": 0.01,
}


# --------------------------------------------------------------------------- #
# Annotation library — one builder per component archetype
# --------------------------------------------------------------------------- #
def _ev(
    cfl: ComponentFailureLogic,
    suffix: str,
    label: str,
    fclass: FClass,
    prob_key: str,
    rationale: str,
    mitigation: str,
    kind: str = "internal",
) -> BasicEventRef:
    event = BasicEvent(
        id=f"BE-{cfl.component}-{suffix}",
        component=cfl.component,
        label=label,
        fclass=fclass,
        prob=DEFAULT_P.get(prob_key, 0.05),
        kind=kind,
        rationale=rationale,
        mitigation=mitigation,
    )
    return cfl.add_event(event)


def _in_refs(comp: Component, fclass: FClass) -> List[DevRef]:
    return [DevRef(comp.id, p, fclass) for p in comp.ports_in]


def _any_input(comp: Component, fclass: FClass) -> Expr:
    return OR(*_in_refs(comp, fclass))


def _all_inputs(comp: Component, fclass: FClass) -> Expr:
    return AND(*_in_refs(comp, fclass))


def annotate_llm_agent(comp: Component) -> ComponentFailureLogic:
    """Local failure logic of a node whose output is produced by an LLM."""
    cfl = ComponentFailureLogic(comp.id, comp.role)
    out = comp.port_out()

    halluc = _ev(cfl, "HALLUC", "Plausible but incorrect content generated", FClass.VALUE_SUBTLE,
                 "llm.halluc",
                 "The model produces a well-formed answer that is factually or "
                 "arithmetically wrong. Not detectable by any syntactic check downstream.",
                 "Sample-based semantic uncertainty gate; require tool-verified arithmetic; "
                 "abstain above an entropy threshold.")
    nondet = _ev(cfl, "NONDET", "Sampling non-determinism changes the answer", FClass.VALUE_SUBTLE,
                 "llm.nondet",
                 "do_sample=True with temperature>0 makes the node a stochastic component: "
                 "the same input can yield a different answer on a re-run.",
                 "Pin seed and temperature for the deterministic path; use the K-sample "
                 "entropy estimate as an online confidence signal.")
    fmt = _ev(cfl, "FORMAT", "Required output format violated", FClass.VALUE_COARSE,
              "llm.format",
              "The ReAct contract (Thought/Action/Action Input/Observation/Final Answer) "
              "is not honoured, so downstream regular expressions cannot extract the fields.",
              "Constrained decoding or a schema validator with a bounded repair step.")
    trunc = _ev(cfl, "TRUNC", "Generation truncated at max_new_tokens", FClass.VALUE_COARSE,
                "llm.truncate",
                "The stop condition is never reached within the token budget, so the "
                "response is cut mid-structure.",
                "Budget headroom, explicit length checks, and a completeness assertion.")
    empty = _ev(cfl, "EMPTY", "No output produced (runtime/decode failure)", FClass.OMISSION,
                "llm.empty",
                "OOM, CUDA fault, tokenizer error or an empty decode leaves the state key unwritten.",
                "Try/except around generation with an explicit failure state; health checks.")
    ctx = _ev(cfl, "CTX", "Context window exceeded", FClass.OMISSION,
              "llm.context",
              "Prompt accumulation across iterations eventually exceeds the window.",
              "Bound the transcript, summarise history, assert token count before generation.")
    late = _ev(cfl, "LATE", "Response later than the latency budget", FClass.LATE,
               "llm.late",
               "Long generations under contention exceed the interaction deadline.",
               "Timeout with a defined degraded response.")

    cfl.set(out, FClass.OMISSION, OR(empty, ctx, _any_input(comp, FClass.OMISSION)))
    cfl.set(out, FClass.VALUE_COARSE, OR(fmt, trunc, _any_input(comp, FClass.VALUE_COARSE)))
    cfl.set(out, FClass.VALUE_SUBTLE, OR(halluc, nondet, _any_input(comp, FClass.VALUE_SUBTLE)))
    cfl.set(out, FClass.LATE, OR(late, _any_input(comp, FClass.LATE)))
    cfl.notes.append(
        "An LLM node trusts its input: a subtle value deviation arriving at the "
        "input propagates to the output unchanged (no error detection)."
    )
    return cfl


def annotate_tool(comp: Component) -> ComponentFailureLogic:
    """Local failure logic of a deterministic executor (calculator/code/API)."""
    cfl = ComponentFailureLogic(comp.id, comp.role)
    out = comp.port_out()
    code = comp.source_code or ""
    uses_eval = bool(__import__("re").search(r"\beval\s*\(|\bexec\s*\(", code))

    parse = _ev(cfl, "PARSE", "Action Input could not be extracted", FClass.OMISSION,
                "tool.parse",
                "The regular expression that lifts the tool call out of free text fails, "
                "so the tool has nothing to execute.",
                "Structured tool calling instead of regex over prose; explicit parse-failure branch.")
    exec_err = _ev(cfl, "EXECERR", "Execution raised; error text returned as an observation",
                   FClass.VALUE_COARSE, "tool.exec_error",
                   "The exception handler substitutes a guidance string for the result. That "
                   "string is fed back to the agent as if it were an observation.",
                   "Type the tool result (ok/error) instead of returning prose; cap retries.")
    wrong = _ev(cfl, "WRONGEXPR", "Executes correctly but computes the wrong thing",
                FClass.VALUE_SUBTLE, "tool.wrong_expr",
                "The expression is syntactically valid and evaluates cleanly, but does not "
                "encode the intent of the problem. The tool cannot detect this.",
                "Cross-check with an independent formulation; unit-test the tool contract.")

    cfl.set(out, FClass.OMISSION, OR(parse, _any_input(comp, FClass.OMISSION)))
    cfl.set(out, FClass.VALUE_COARSE, OR(exec_err, _any_input(comp, FClass.VALUE_COARSE)))
    cfl.set(out, FClass.VALUE_SUBTLE, OR(wrong, _any_input(comp, FClass.VALUE_SUBTLE)))

    if uses_eval:
        unsafe = _ev(cfl, "UNSAFE", "Model-authored code executed without a sandbox",
                     FClass.COMMISSION, "tool.unsafe",
                     "eval()/exec() is applied to text produced by the model, so any side "
                     "effect the model can express is executed with the host's privileges.",
                     "Restricted evaluator (AST allow-list) or an isolated sandbox process.")
        cfl.set(out, FClass.COMMISSION, unsafe)
        cfl.notes.append(
            "SECURITY: this component executes model-authored strings. The commission "
            "deviation is a top event in its own right, not merely a contributor."
        )
    return cfl


def annotate_router(comp: Component) -> ComponentFailureLogic:
    """Local failure logic of a conditional-edge decision function."""
    cfl = ComponentFailureLogic(comp.id, comp.role)

    no_match = _ev(cfl, "NOMATCH", "No branch condition matched", FClass.OMISSION,
                   "router.no_match",
                   "Neither the tool pattern nor the final-answer pattern matches, so control "
                   "leaves by the error branch and the run ends with nothing delivered.",
                   "Make the fallback branch explicit and observable; never route silently to END.")
    misroute = _ev(cfl, "MISROUTE", "Wrong branch selected", FClass.VALUE_COARSE,
                   "router.misroute",
                   "Pattern matching over free text selects a branch the reasoning did not intend.",
                   "Decide on structured fields, not on regexes over prose.")
    early = _ev(cfl, "EARLYSTOP", "Terminates before the task is complete", FClass.EARLY,
                "router.early_stop",
                "The termination token appears inside the reasoning text, so the run is "
                "declared complete while the answer is still being derived.",
                "Anchor the termination test to the end of the output and to a completeness check.")

    for port in (comp.ports_out or ["out"]):
        cfl.set(port, FClass.OMISSION, OR(no_match, _any_input(comp, FClass.OMISSION)))
        cfl.set(port, FClass.VALUE_COARSE, OR(misroute, _any_input(comp, FClass.VALUE_COARSE)))
        cfl.set(port, FClass.VALUE_SUBTLE, _any_input(comp, FClass.VALUE_SUBTLE))
        cfl.set(port, FClass.EARLY, early)
        cfl.set(port, FClass.LATE, _any_input(comp, FClass.LATE))
    cfl.notes.append(
        "The router adds no value deviation of its own beyond branch selection; it is "
        "transparent to subtle value errors, which is why a wrong answer can pass "
        "through the termination check unchallenged."
    )
    return cfl


def annotate_aggregator(comp: Component) -> ComponentFailureLogic:
    """Local failure logic of a fan-in node that combines redundant agents.

    This is where redundancy is expressed: an aggregator masks a *single* faulty
    input, so its omission and value deviations require **all** inputs to be
    deviated (an AND gate) — unless the aggregator itself mis-selects.
    """
    cfl = ComponentFailureLogic(comp.id, comp.role)
    out = comp.port_out()

    empty = _ev(cfl, "EMPTY", "Aggregation produced no output", FClass.OMISSION,
                "agg.empty",
                "The combining step itself fails (runtime error or empty generation).",
                "Fail closed to a defined 'no answer' response rather than an unwritten key.")
    select = _ev(cfl, "SELECT", "Selects the incorrect candidate answer", FClass.VALUE_SUBTLE,
                 "agg.select",
                 "With one correct and one incorrect candidate, the aggregator adjudicates "
                 "wrongly. This converts single-agent faults into system faults.",
                 "Adjudicate on verifiable evidence (recomputation, unit checks), not on prose "
                 "plausibility; escalate ties instead of guessing.")
    fmt = _ev(cfl, "FORMAT", "Combined answer malformed", FClass.VALUE_COARSE,
              "agg.format",
              "The final response does not expose the answer in the agreed shape.",
              "Validate the final schema at the system boundary.")
    own = _ev(cfl, "OWN", "Substitutes its own incorrect answer", FClass.VALUE_SUBTLE,
              "agg.select",
              "The aggregator is permitted to propose a solution of its own when it "
              "trusts neither candidate. That answer is generated by the same kind of "
              "stochastic reasoning as the candidates, but nothing reviews it — so the "
              "adjudicator becomes a single point of failure even with correct inputs.",
              "Restrict the aggregator to selection among candidates, or subject an "
              "aggregator-authored answer to the same verification as a candidate.")

    n_in = len(comp.ports_in)
    all_omitted = _all_inputs(comp, FClass.OMISSION)
    all_subtle = _all_inputs(comp, FClass.VALUE_SUBTLE)
    any_subtle = _any_input(comp, FClass.VALUE_SUBTLE)
    all_coarse = _all_inputs(comp, FClass.VALUE_COARSE)

    cfl.set(out, FClass.OMISSION, OR(empty, all_omitted))
    cfl.set(out, FClass.VALUE_COARSE, OR(fmt, all_coarse))
    # Wrong final answer if: every candidate is wrong (redundancy defeated), or
    # at least one candidate is wrong and the aggregator picks it, or the
    # aggregator authors a wrong answer of its own.
    cfl.set(out, FClass.VALUE_SUBTLE, OR(own, all_subtle, AND(any_subtle, select)))
    cfl.set(out, FClass.LATE, _any_input(comp, FClass.LATE))
    cfl.notes.append(
        f"Redundancy over {n_in} input(s): omission and value deviations are masked "
        "unless all inputs deviate, or the selection logic itself fails. Any common "
        "cause shared by the inputs collapses this AND gate into a single point of failure."
    )
    return cfl


def annotate_transform(comp: Component) -> ComponentFailureLogic:
    cfl = ComponentFailureLogic(comp.id, comp.role)
    out = comp.port_out()
    fault = _ev(cfl, "FAULT", "Internal fault of the node", FClass.OMISSION,
                "transform.fault",
                "Unhandled exception or missing state write in a deterministic node.",
                "Unit tests plus explicit state-contract assertions.")
    corrupt = _ev(cfl, "CORRUPT", "Node corrupts the payload", FClass.VALUE_COARSE,
                  "transform.fault",
                  "Transformation writes a malformed or wrongly typed value into the state.",
                  "Typed state channels; validate on write.")
    cfl.set(out, FClass.OMISSION, OR(fault, _any_input(comp, FClass.OMISSION)))
    cfl.set(out, FClass.VALUE_COARSE, OR(corrupt, _any_input(comp, FClass.VALUE_COARSE)))
    cfl.set(out, FClass.VALUE_SUBTLE, _any_input(comp, FClass.VALUE_SUBTLE))
    cfl.set(out, FClass.LATE, _any_input(comp, FClass.LATE))
    return cfl


def annotate_source(comp: Component) -> ComponentFailureLogic:
    """The system boundary: the request that enters the graph."""
    cfl = ComponentFailureLogic(comp.id, comp.role)
    out = comp.port_out()
    bad_input = _ev(cfl, "BADREQ", "Ill-posed or ambiguous request at the boundary",
                    FClass.VALUE_SUBTLE, "boundary.input",
                    "The task given to the graph is ambiguous or contradictory; no internal "
                    "component can recover from it.",
                    "Input validation and clarification protocol before the graph is entered.",
                    kind="boundary")
    missing = _ev(cfl, "NOREQ", "Required input key absent from the initial state",
                  FClass.OMISSION, "boundary.input",
                  "The graph is invoked without a key that downstream nodes read.",
                  "Validate the initial state against the State TypedDict before invoke().",
                  kind="boundary")
    cfl.set(out, FClass.OMISSION, missing)
    cfl.set(out, FClass.VALUE_SUBTLE, bad_input)
    return cfl


def annotate_sink(comp: Component) -> ComponentFailureLogic:
    """The system boundary where top events are observed. No logic of its own."""
    return ComponentFailureLogic(comp.id, comp.role)


def annotate_feedback(comp: Component) -> ComponentFailureLogic:
    """Local failure logic of the pseudo-component that closes a cut loop.

    It carries whatever the feedback path was carrying to the system boundary,
    and adds the two failures that a bounded loop introduces by itself:
    non-convergence within the iteration budget, and the latency of iterating.
    """
    cfl = ComponentFailureLogic(comp.id, comp.role)
    out = comp.port_out()
    src = comp.metadata.get("loop_source", "loop")
    depth = comp.metadata.get("unroll", 1)

    exhaust = _ev(cfl, "EXHAUST", "Iteration budget exhausted without a final answer",
                  FClass.OMISSION, "loop.cut",
                  f"The loop closed by {src} does not converge within the modelled "
                  f"{depth} iteration(s); the recursion limit is reached, or the agent "
                  "keeps re-entering the tool without ever emitting a final answer.",
                  "Hard iteration counter in the state with a defined give-up response; "
                  "monitor the loop count as a run-time safety signal.",
                  kind="loop_cut")
    slow = _ev(cfl, "ITERLATE", "Iterating exceeds the time/token budget", FClass.LATE,
               "loop.cut",
               "Each additional pass re-generates the whole transcript, so latency and "
               "token cost grow with iteration depth.",
               "Bound iterations and total tokens; degrade gracefully on the budget.",
               kind="loop_cut")

    cfl.set(out, FClass.OMISSION, OR(exhaust, _any_input(comp, FClass.OMISSION)))
    cfl.set(out, FClass.LATE, OR(slow, _any_input(comp, FClass.LATE)))
    cfl.set(out, FClass.VALUE_COARSE, _any_input(comp, FClass.VALUE_COARSE))
    cfl.set(out, FClass.VALUE_SUBTLE, _any_input(comp, FClass.VALUE_SUBTLE))
    cfl.notes.append(
        "Loop-cut boundary: deviations that the feedback path would have re-injected "
        "are delivered here instead of being discarded, keeping the acyclic model "
        "conservative with respect to the cyclic system."
    )
    return cfl


_LIBRARY = {
    Role.LLM_AGENT: annotate_llm_agent,
    Role.FEEDBACK: annotate_feedback,
    Role.TOOL: annotate_tool,
    Role.ROUTER: annotate_router,
    Role.AGGREGATOR: annotate_aggregator,
    Role.TRANSFORM: annotate_transform,
    Role.SOURCE: annotate_source,
    Role.SINK: annotate_sink,
}


# --------------------------------------------------------------------------- #
# Connection-level and common-cause annotation
# --------------------------------------------------------------------------- #
def _annotate_connections(model: FailureModel) -> None:
    """Attach channel failures to connections (LangGraph state channels)."""
    for conn in model.system.connections:
        eid = f"BE-CH-{conn.src}-{conn.dst}-LOSS"
        model.events[eid] = BasicEvent(
            id=eid,
            component=f"{conn.src}->{conn.dst}",
            label="State value not visible to the successor",
            fclass=FClass.OMISSION,
            prob=DEFAULT_P["channel.loss"],
            kind="channel",
            rationale="The producing node returns a key the consumer does not read, or the "
                      "key is overwritten before it is consumed.",
            mitigation="Single writer per channel; assert the state contract between nodes.",
        )
        model.connection_events.setdefault(conn.id, []).append(eid)

        if conn.fan_in and conn.parallel:
            cid = f"BE-CH-{conn.dst}-CONTENTION"
            if cid not in model.events:
                model.events[cid] = BasicEvent(
                    id=cid,
                    component=conn.dst,
                    label="Concurrent writes to a non-reducer state channel",
                    fclass=FClass.OMISSION,
                    prob=DEFAULT_P["channel.contention"],
                    kind="channel",
                    rationale="Two branches complete in the same super-step and write the same "
                              "LastValue key. LangGraph raises InvalidUpdateError and the whole "
                              "run is lost — an omission at the system boundary, not a local fault.",
                    mitigation="Annotated reducer channels (operator.add) or disjoint keys per "
                               "branch, aggregated by a single writer.",
                )
                # Contention destroys the whole run, so it bypasses the redundancy
                # of the fan-in node: it is attached to the node's output omission
                # directly, not to one of its input ports.
                target = model.system.components.get(conn.dst)
                cfl = model.logic.get(conn.dst)
                if target and cfl:
                    for port in (target.ports_out or ["out"]):
                        dev = Deviation(conn.dst, port, FClass.OMISSION)
                        cfl.logic[dev] = OR(cfl.logic.get(dev), BasicEventRef(cid))
                    cfl.notes.append(
                        "Fan-in node: concurrent state writes are a single point of failure "
                        "that the aggregation redundancy does not mask."
                    )


def _annotate_ccf(model: FailureModel) -> None:
    """Create common-cause basic events for shared resources.

    A CCF event is injected into the *subtle value* and *omission* logic of every
    member of the group, so a fault tree built over apparently redundant agents
    exposes the shared cause as a single point of failure.
    """
    for (kind, value), members in model.system.common_cause_groups().items():
        if kind not in ("llm", "runtime", "tokenizer", "quantisation"):
            continue
        if kind == "llm":
            eid = f"CCF-LLM-{_slug(value)}"
            event = BasicEvent(
                id=eid,
                component=" + ".join(members),
                label=f"Shared model snapshot fails identically ({value})",
                fclass=FClass.VALUE_SUBTLE,
                prob=DEFAULT_P["ccf.model"],
                kind="ccf",
                rationale=(
                    f"Components {', '.join(members)} all call the same model snapshot. "
                    "Their errors are correlated: a prompt the model gets wrong is wrong "
                    "for every replica, so the voting/aggregation AND gate degrades to an "
                    "OR gate for that class of input."
                ),
                mitigation="Diversify the redundant channels (different model family, different "
                           "prompt strategy, or a symbolic checker) before claiming redundancy.",
            )
        elif kind == "runtime":
            eid = f"CCF-RUNTIME-{_slug(value)}"
            event = BasicEvent(
                id=eid, component=" + ".join(members),
                label=f"Shared runtime fails ({value})",
                fclass=FClass.OMISSION, prob=DEFAULT_P["ccf.runtime"], kind="ccf",
                rationale="All components execute in one process on one accelerator; an OOM or "
                          "device fault removes every replica at once.",
                mitigation="Separate execution domains, or accept and declare the single point of failure.",
            )
        else:
            continue

        model.events[eid] = event
        model.ccf_groups[eid] = members
        ref = BasicEventRef(eid)
        for cid in members:
            cfl = model.logic.get(cid)
            comp = model.system.components.get(cid)
            if not cfl or not comp or not comp.ports_out:
                continue
            for port in comp.ports_out:
                dev = Deviation(cid, port, event.fclass)
                cfl.logic[dev] = OR(cfl.logic.get(dev), ref)


def _slug(text: str) -> str:
    import re as _re
    return _re.sub(r"[^0-9A-Za-z]+", "-", text).strip("-")[:40]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def annotate_system(
    system: SystemModel,
    probability_overrides: Optional[Dict[str, float]] = None,
    entropy_by_component: Optional[Dict[str, float]] = None,
    extra_logic: Optional[Dict[str, ComponentFailureLogic]] = None,
) -> FailureModel:
    """Annotate every component with its local failure logic (HiP-HOPS Phase 1).

    Parameters
    ----------
    entropy_by_component
        Measured semantic-cluster entropy per component (from the notebook's
        ``uncertainty_summary``).  Where present it replaces the placeholder
        probability of that component's hallucination / non-determinism events
        through :func:`entropy_to_fail_prob`.
    """
    model = FailureModel(system=system)

    for cid, comp in system.components.items():
        builder = _LIBRARY.get(comp.role, annotate_transform)
        cfl = builder(comp)
        if extra_logic and cid in extra_logic:
            override = extra_logic[cid]
            cfl.logic.update(override.logic)
            cfl.events.update(override.events)
            cfl.notes.extend(override.notes)
        model.logic[cid] = cfl
        model.events.update(cfl.events)

    _annotate_connections(model)
    _annotate_ccf(model)

    # --- quantitative calibration ----------------------------------------- #
    if entropy_by_component:
        for cid, entropy in entropy_by_component.items():
            p = entropy_to_fail_prob(entropy)
            for suffix, scale in (("HALLUC", 1.0), ("NONDET", 0.6), ("SELECT", 0.8), ("OWN", 0.9)):
                eid = f"BE-{cid}-{suffix}"
                if eid in model.events:
                    ev = model.events[eid]
                    ev.prob = round(min(0.95, p * scale), 4)
                    ev.evidence = (
                        f"measured: mean semantic-cluster entropy = {entropy:.4f} "
                        f"-> sigmoid calibration (ent_mid=0.90, slope=3.0)"
                    )
    if probability_overrides:
        unknown = [eid for eid in probability_overrides if eid not in model.events]
        if unknown:
            # Silently ignoring a mistyped id would leave the placeholder in
            # place and report a number the analyst believes they replaced.
            raise KeyError(
                f"probability_overrides names basic event(s) {sorted(unknown)}, "
                f"which this model does not have. Ids look like "
                f"'BE-<component>-<suffix>'; this model has "
                f"{sorted(model.events)[:8]}{' ...' if len(model.events) > 8 else ''}"
            )
        for eid, p in probability_overrides.items():
            if not 0.0 <= float(p) <= 1.0:
                raise ValueError(
                    f"probability_overrides[{eid!r}] = {p!r} is not a probability"
                )
            model.events[eid].prob = float(p)
            model.events[eid].evidence = "user-supplied override"

    if model.ccf_groups:
        model.notes.append(
            "Common-cause groups were detected. Redundant structures containing these "
            "components do not provide independent failure paths."
        )
    return model
