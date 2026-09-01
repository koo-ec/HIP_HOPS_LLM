# 1. From a LangGraph to a fault tree

We will take the ReAct-plus-calculator architecture — one agent, one tool, one
router, and a feedback loop — and turn it into fault trees without drawing
anything.

```python
from HIP_HOPS_LLM import extract_architecture, load_example
import pandas as pd

model = extract_architecture(load_example("react_calculator"), name="ReAct + calculator")
pd.DataFrame(model.architecture_table()).set_index("component")
```

```text
                        role                                  in_ports                    out_ports                     resources
component
__end__                 sink  in@generator::router, in@…:end                                      -                             -
__start__             source                               -                                    out                             -
coder                   tool                              in                                    out                             -
generator          llm_agent            in@__start__, in@coder                                   out  llm=Qwen/…, runtime=cuda:0
generator::router     router                              in       out@coder, out@error, out@end                             -
```

Five components from a three-node graph. Two of them are worth stopping on.

## The router is a component

`generator::router` does not exist in the LangGraph node list. It is the callable
passed to `add_conditional_edges`, and the extractor materialises it, because a
regular expression that matches the wrong branch is a failure mode with
consequences:

```python
model.components["generator::router"].branches
# ['coder', 'error', 'end']
```

Two of those branches reach `__end__`, and they get **separate input ports**
(`in@generator::router` and `in@generator::router:end`). A shared port would
collapse them into one deviation and miscount the fan-in.

The router's source is not the node's source. LangGraph keeps the routing
function separately, so pass it explicitly if you want it analysed:

```python
extract_architecture(graph, node_functions={"generator::router": route_fn})
```

## The resources are read, not declared

```python
model.common_cause_groups()
# {('llm', 'Qwen/Qwen2.5-Math-1.5B-Instruct'): [...], ('runtime', 'cuda:0'): [...]}
```

Those came out of the node function's source text. In a notebook, pass
`globals_ns=globals()` and the extractor interrogates the *live* model objects
instead, which is more reliable:

```python
extract_architecture(graph, globals_ns=globals())
```

Declare what the source does not name:

```python
from HIP_HOPS_LLM import LangGraphExtractor

extractor = LangGraphExtractor(globals_ns=globals()).with_resources(
    critic={"llm": "gpt-4o-2024-11-20"},
    drafter={"llm": "gpt-4o-2024-11-20"},     # same snapshot → a CCF group
)
```

## The loop

```python
from HIP_HOPS_LLM import find_cycles, make_acyclic

find_cycles(model)
# [['coder', 'generator', 'generator::router']]
```

Fault trees are acyclic. Cutting the back edge outright would delete the tool's
contribution and **understate** risk, so instead the loop is unrolled and closed
with a feedback-cut component:

```python
acyclic, report = make_acyclic(model, unroll=1)
print(report.summary())
```

```text
1 feedback loop(s) found; unrolled to depth 1 and closed with 1 feedback-cut
component(s).
  loop: coder -> generator -> generator::router
  back edge cut: coder -> generator
  Deleting a back edge outright would remove the feedback path's contribution
  from the fault tree and understate risk; the feedback-cut component preserves it.
  unroll=1: a single pass through the loop body is modelled. Increase unroll to
  expose iteration-dependent effects such as prompt growth.
```

`unroll=2` models two passes and produces a larger tree. Use it when the loop's
iterations differ — a growing prompt, an accumulating scratchpad — and `unroll=1`
when they do not.

## Synthesis

```python
from HIP_HOPS_LLM import AgenticReliabilityStudy

study = AgenticReliabilityStudy(load_example("react_calculator"),
                                name="ReAct + calculator", unroll=1)
print(study.analyse().summary())
```

```text
components: 6  connections: 7  basic events: 24
1 feedback loop(s) found; unrolled to depth 1 …

hazard     sev             P(top)    MCS  SPOF  name
----------------------------------------------------
H1         major           0.3185     12    12  No answer delivered
H2         critical        0.3335      4     4  Incorrect answer delivered and accepted as correct
H3         minor           0.2361      4     4  Malformed answer delivered
H4         minor           0.1450      2     2  Answer too late / budget exhausted
H5-coder   catastrophic    0.0200      1     1  Unsafe execution of model-authored code in coder
```

`H5-coder` appears because `coder`'s source contains `eval(`. It is
catastrophic, and its single cut set is `BE-coder-UNSAFE` — nothing else has to
go wrong.

**Every cut set here is order 1.** There is no redundancy anywhere in this
architecture; any single fault reaches the boundary.

## Looking at the tree

```python
study.plot("H2")                       # matplotlib
print(study.report.mermaid("H2"))      # mermaid source
study.report.display("H2")             # inline in a notebook
```

Exports:

```python
from HIP_HOPS_LLM import to_dot, to_json, to_openpsa_xml

tree = study.report.tree("H2")
to_dot(tree)                           # Graphviz
to_json(tree, study.report.analysis("H2"))
to_openpsa_xml(tree, name="H2")        # Open-PSA MEF, for XFTA / SCRAM
```

## Next

[Tutorial 2](02-cut-sets-and-fmea.md) reads the structural result: which cut sets
matter, which events to fix first, and what the generated FMEA says.
