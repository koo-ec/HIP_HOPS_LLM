# Extending the package

Seven things people actually want to add, each with the files to touch and the
test that proves it worked. Every one of them ends the same way: **a small graph
you can check by hand**, because a cut set you cannot verify on paper is a cut
set you cannot trust from a machine.

## A new component archetype

You have a kind of node the library does not model - a retriever, a guardrail, a
human-in-the-loop approval step.

1. **Add the role.** `Role` in `architecture/model.py`. The value is what appears
   in reports and drawings, so it is prose (`"llm_agent"`, not `"LLMAGENT"`).
2. **Teach `classify_role` to recognise it.** It reads node source, then names,
   then falls back on `has_conditional_out`. Prefer a signal that survives
   refactoring - a resource, a decorator - over a name substring.
3. **Write the annotation builder** in `faulttree/failure.py`, next to
   `annotate_llm_agent` and `annotate_tool`. For each output `Deviation` the
   component can produce, give the Boolean expression over its input deviations
   and its own basic events.
4. **Give every basic event a `rationale` and a `mitigation`.** The FMEA is
   generated from these, so an event without them produces an FMEA row that
   tells the reader nothing. Write them for the *class*: what does a `VS` from
   this archetype actually look like?
5. **Add defaults to `DEFAULT_P`,** and say `placeholder` in the `evidence`
   string. A number nobody measured must never look measured.
6. **Register it** in the dispatch that `annotate_system` uses.

The decision that matters is transparency. Ask: *if a subtle value deviation
arrives at this component's input, can it detect it?* An LLM agent cannot, so its
logic passes `VS` straight through; a schema validator can, so it converts `VC`
to `O` and stops it. Get this wrong and the tree will show a check where there is
none.

```python
# the test to write: a two-node graph you can reason about on paper
spec = {"name": "t", "nodes": {...}, "edges": [...]}
report = analyse_langgraph(spec)
assert report.cut_sets("H2") == [["BE-guard-MISS"], ...]   # by hand, not by eye
```

## A new hazard

Hazards are data, not code. Pass them in:

```python
from hiphopsllm import AgenticReliabilityStudy, Hazard, FClass

study = AgenticReliabilityStudy(
    graph,
    hazards=[Hazard(id="HX", name="PII leaves the system", severity="critical",
                    component="__end__", port="in", fclass=FClass.COMMISSION)],
)
```

A hazard is *a deviation at a named port of a named component*. If the tree comes
back empty, that is a result and not a bug: nothing in this architecture can
cause that deviation at that port. Check you picked the right port before
concluding the system is safe.

To make a hazard appear automatically for a whole class of system, extend
`default_hazards` in `faulttree/synthesis.py`. `N8nWorkflow.hazards` shows the
other half of the pattern: it takes the defaults and *rewords* `H5-<tool>`, the
unsafe-execution hazard, into "unsolicited outward action" for the tools that act
on the world, and drops it for the ones that cannot.

## A new gate type

1. **Write the table constructor** in `bayes/cpt.py` beside `k_of_n_cpt`, and
   validate its fan-in through `_check_fan_in`.
2. **Add it to `GateType`** and to `CPTBuilder._gate_table`.
3. **Check monotonicity.** `ImpreciseBayesianNetwork` builds a lower and an upper
   twin network, and that is only valid because the top event is monotone in
   every basic event. If your gate is not monotone, say so in its docstring
   *and* append to `cpts.notes`, the way `soft_gates` does. Silently breaking
   the interval arithmetic is the worst failure mode this codebase has.
4. **Compare against a hand-computed table** in `tests/unit/test_cpt.py`. Not
   against another implementation - against arithmetic you did yourself.

## A new export format

`faulttree/export.py` takes a `FaultTree` and returns a string. Add the function,
export it from `faulttree/__init__.py` and the top-level `__init__.py`, and wire
it into `SafetyReport.save` if it should be written by default.

Two properties to test: the output **parses** in its target format, and it
contains every basic event id in `tree.basic_event_ids()`. An exporter that
quietly drops shared sub-trees is the classic bug here - `to_openpsa_xml` handles
them by reference, `to_mermaid` by repetition, and the tests say which.

## A new stratifier

A stratifier is any callable from an input item to a label. Nothing to register:

```python
from hiphopsllm import stratify

labels = stratify(items, lambda r: "long" if len(r["q"].split()) > 40 else "short",
                  profile=profile)
```

Pass `profile=` so an unexpected label raises instead of silently dropping those
items from every downstream estimate. If you want to ship one, put it beside
`decomposition_stratum` in `reliability/profile.py`.

The property a good stratifier has: failure probability varies **between** strata
more than **within** them. If it does not, the stratification buys nothing and
you have only made the intervals wider.

## A new n8n node rule

`RULES` in `io/n8n.py` is an ordered tuple; the first rule that matches wins, so
order is part of the meaning - `gmailTool` must be read as a tool before it is
read as a Gmail node.

```python
Rule(
    "vector store",
    _endswith("vectorstorepinecone", "vectorstoreqdrant"),
    kind="component",          # or "resource", or "excluded"
    role=Role.TOOL,
    why="A vector store is a retrieval step the agent depends on: it can be "
        "unreachable (omission) or return the wrong neighbours (a plausible, "
        "undetectable value deviation).",
)
```

The `why` string is not a comment. It reaches the ledger the analyst reads, so
write it for them: what this node *is* in failure terms, and why it is modelled
this way rather than the obvious alternative. `kind="resource"` also needs
`resource_kind` - that is how a model sub-node becomes `resources["llm"]` and
therefore how two agents on one model become a common-cause group.

Test it on a workflow dict, the way `tests/unit/test_n8n.py` does, and assert on
the ledger as well as the tree - the reason is as much a product here as the
number.

## A new plot

`viz/plots.py`. Take an optional `ax`, return the `ax`, use the `TOKENS` palette,
and never let a status colour be the only signal - pair it with a text tag, as
`plot_importance` does with `◆ single point`.

Test what breaks rather than what it looks like: every node is drawn, degenerate
input produces a figure or a clear error rather than a traceback from inside
matplotlib, and long labels do not escape their shapes. `tests/unit/test_viz_plots.py`
reads `ax.texts` and `ax.lines` to assert those.

## Things that will bite you

**Do not edit `src/HIPLLM/` or `src/hip_llm/`.** They are byte copies of
[`koo-ec/HIP_LLM`](https://github.com/koo-ec/HIP_LLM). Fix bugs upstream and
re-vendor - see [Vendoring](../vendoring.md). They are excluded from ruff for
this reason.

**pyAgrum is optional.** Anything in `bayes/` must also work through
`engine="exact"`. CI runs the whole suite once with pyAgrum uninstalled.

**Nothing fails silently.** If your code cannot do what was asked, raise with a
message naming what was wrong *and* what the valid options are. If a value is a
placeholder, its `evidence` string says so. This is the house rule the package
exists to embody - a Bayesian network running on default priors looks exactly
like one running on evidence.

**Provenance travels with numbers.** Every `BasicEvent` has an `evidence` string,
every `OperationalProfile` a `provenance`, every `ComponentEvidence` its
`method`, `n` and interval. A number that reaches a report without them is a bug.
