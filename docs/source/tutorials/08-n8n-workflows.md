# 8. Analysing an n8n workflow

LangGraph is not the only way people build agentic systems. An **n8n** workflow
is an agent architecture too: a trigger, an agent node, a model, some tools, a
memory. `hiphopsllm.io.n8n` reads an n8n JSON export straight into the same
architecture model, so everything in tutorials 1-7 applies unchanged.

```python
from hiphopsllm import load_n8n

workflow = load_n8n("Mail agent.json")     # a path, a file object, or a dict
print(workflow.summary())
```

```text
n8n workflow: Mail agent
  7 nodes -> 5 components, 1 folded in as resources, 1 excluded
  7 edges (including the delivery boundary)
  acts on the world through: Reply
  note: Folded in as resources rather than components: Chat Model.
  note: Excluded (no runtime behaviour): Sticky Note.
  note: Each ai_tool connection became two edges (agent to tool, tool back to
        agent). The loop is cut by make_acyclic, which is what adds the
        iteration-budget and iteration-latency events.
  note: 3 nodes share gmailOAuth2 'Mail account': Gmail Trigger, Read thread,
        Reply. Revoking or rate-limiting it removes all of them at once, so
        they are not independent.
```

Seven nodes became five components. That is not tidying: it is three modelling
decisions, and each changes the fault tree.

## The three decisions

**1. Not every n8n node is a component.** A sticky note has no runtime
behaviour. A language-model sub-node is not a step in the flow, it is the
*resource* the agent generates with, so it is folded into
`resources["llm"]`. That fold is what lets two agents sharing one model form a
common-cause group. Modelled as separate components they would each carry their
own hallucination event and the shared snapshot would vanish from the analysis
- which is exactly the error this package exists to catch.

**2. The `ai_*` connections run backwards.** n8n draws a tool, a memory or a
model *into* the agent, so the JSON arrow points from the sub-node to the agent.
For a memory or an output parser that is also the direction failures propagate,
and the edge is kept. For a **tool** it is not: the agent decides to call it, so
the invocation runs agent → tool, the observation comes back (a loop, cut by
`make_acyclic`), and an outward action such as sending an email is delivered at
the system boundary.

**3. A branching n8n node is already the router.** In LangGraph the routing
function is anonymous and gets materialised as a `<node>::router` component. An
n8n `If` or `Switch` *is* the router, so it is given `Role.ROUTER` directly and
its outgoing edges are not marked conditional - otherwise a second, empty router
would appear beside it.

## The ledger

Every decision is recorded per node, with its reason, so you can disagree with
it in the open rather than discovering it in a number:

```python
workflow.ledger_frame()          # pandas
workflow.ledger_markdown()       # for a report
workflow.ledger()                # the rows as dicts
```

| n8n node | type | connection | modelled as | resources | flags |
|---|---|---|---|---|---|
| Gmail Trigger | gmailTrigger | main | `source` | credential:gmailOAuth2 | holds credentials |
| AI Agent | agent | main | `llm_agent` | llm=gpt-4.1, runtime=n8n:n8n | - |
| Chat Model | lmChatOpenAi | ai_languageModel | **resource:llm** | credential:openAiApi | holds credentials |
| Reply | gmailTool | ai_tool | `tool` | credential:gmailOAuth2 | **acts on the world; arguments written by the model (`$fromAI`)** |
| Read thread | gmailTool | ai_tool | `tool` | credential:gmailOAuth2 | holds credentials |
| Memory | memoryPostgresChat | ai_memory | `tool` | credential:postgres | holds credentials |
| Sticky Note | stickyNote | main | **excluded** | - | - |

The `why` column (dropped above for width) carries a sentence per row. For the
model sub-node it reads:

> A model sub-node is not a step in the flow, it is the resource the agent
> generates with. Folding it into `resources['llm']` is what makes two agents on
> one model a common-cause group; modelled as its own component it would carry a
> second hallucination event and the shared snapshot would disappear from the
> analysis.

## Analysing it

```python
report = workflow.analyse()
print(report.summary())
```

```text
HiP-HOPS analysis — Mail agent
================================
components: 8  connections: 9  basic events: 35
1 feedback loop(s) found; unrolled to depth 1 and closed with 2 feedback-cut
component(s).
  loop: AI Agent -> Read thread -> Reply
  back edge cut: Read thread -> AI Agent
  back edge cut: Reply -> AI Agent

common-cause groups:
  credential:gmailOAuth2=Mail account: Read thread, Reply
  runtime=n8n:n8n: AI Agent, Memory, Read thread, Reply

hazard     sev             P(top)    MCS  SPOF  name
----------------------------------------------------
H1         major           0.4923     18    18  No answer delivered
H2         critical        0.4602      6     6  Incorrect answer delivered and accepted
H3         minor           0.3194      5     5  Malformed answer delivered
H4         minor           0.1900      2     2  Answer too late / budget exhausted
H5-Reply   critical        0.0979      2     2  Unsolicited outward action by Reply
```

## The commission hazard

`H5-Reply` is the one that does not appear for a read-only pipeline. A tool that
**acts on the world** - sends mail, writes a row, calls a webhook - can fail by
doing something nobody asked for, which is a *commission* failure rather than an
omission or a wrong value. The importer detects it from the node's operation and
adds the hazard automatically:

```python
[h.id for h in workflow.hazards(workflow.system())]
```

Two things drive it. `SIDE_EFFECT_SERVICES` says which n8n services can act
outwards at all; `READ_ONLY_OPERATIONS` says which of their operations do not
(`get`, `getAll`, `search`, …). `Read thread` uses `operation: "get"`, so it
gets no commission hazard; `Reply` uses `operation: "reply"`, so it does.

The hazard decomposes into two different faults with two different fixes. An
agent that owns an acting tool gets a `POLICY` basic event of its own:

> The remit ("reply only to business enquiries") is expressed in prose inside the
> prompt, so it is a preference of the model, not a constraint on the system.
> Nothing between this node and `Reply` can refuse a call the model decides to
> make.

and the acting tool gets `UNSOLICITED`:

> It executes whatever call reaches it; it has no view of whether the call should
> have been made.

The `$fromAI` marker is recorded separately. When a tool's arguments are written
by the model rather than by the flow, the block is flagged `arguments written by
the model ($fromAI)` and the component carries a note saying that a subtle value
deviation upstream becomes the *content* of a real outward action.

## Straight to a study

The same four entry points as everywhere else, at increasing levels of
convenience:

```python
from hiphopsllm import load_n8n, n8n_to_spec, analyse_n8n, n8n_study

workflow = load_n8n(source)        # N8nWorkflow: inspect the ledger first
spec     = n8n_to_spec(source)     # a plain dict spec, for AgenticReliabilityStudy
report   = analyse_n8n(source)     # SafetyReport: structure only
study    = n8n_study(source)       # AgenticReliabilityStudy, ready for .observe()
```

`n8n_study` is the one to use when you have measurements:

```python
study = n8n_study("Mail agent.json")
study.observe(outcomes, profile={"routine": 0.8, "unusual": 0.2})
study.run()
study.bayesnet("H5-Reply").show()
```

## Options

| Argument | Default | Effect |
|---|---|---|
| `tool_feedback` | `True` | Model the observation returning from a tool as a back edge. `False` treats tool calls as one-way, which removes the loop and its iteration-budget events. |
| `host_resource` | `True` | Give every component `runtime=n8n:<host>`, so a host outage is one common cause rather than several independent ones. |
| `role_overrides` | `None` | Force a node's role when the rules read it wrongly. The override is recorded in the ledger. |

## When a node type is unknown

The rules are matched in order and the first match wins; an unrecognised type
falls back to `Role.TRANSFORM` and **says so** in the ledger rather than being
dropped. If that is wrong for your workflow, either pass `role_overrides` or add
a rule - see [Extending the package](../development/extending.md#a-new-n8n-node-rule).

```{seealso}
[`hiphopsllm.io`](../api/io.md) for the full signatures, and
[Loop elimination](../concepts/hiphops.md) for what the feedback cut does to the
tree.
```
