# 7. Your own graph

Everything so far ran on bundled specifications. Here is how to point it at a
real LangGraph application.

## The minimum

```python
from hiphopsllm import AgenticReliabilityStudy

study = AgenticReliabilityStudy(graph, name="my workflow", globals_ns=globals())
print(study.analyse().summary())
```

`graph` may be a compiled LangGraph, the drawable from `graph.get_graph()`, the
mermaid text from `draw_mermaid()`, a specification dict, or a `SystemModel`.

**Pass `globals_ns=globals()` in a notebook.** Without it, role classification
falls back to node names and edge topology; with it, the extractor finds the node
functions and interrogates the *live* model objects, which is what makes
shared-snapshot detection — and therefore the common-cause analysis — reliable.

In a script or a module, pass the functions explicitly:

```python
study = AgenticReliabilityStudy(
    graph,
    node_functions={
        "planner": planner,
        "worker": worker,
        "planner::router": route,     # the add_conditional_edges callable
    },
)
```

The `::router` key matters. LangGraph keeps routing functions outside the node
list, so the extractor materialises a router component but has no source for it
unless you supply one — and attributing the node's source to it would put the
router in the node's common-cause group, which is wrong.

## Check the extraction first

Before trusting any number, look at what was read:

```python
import pandas as pd

pd.DataFrame(study.system.architecture_table()).set_index("component")
```

Three questions to ask of that table.

**Are the roles right?** A node classified `transform` when it is really an LLM
agent gets deterministic failure logic and no hallucination event.

```python
study = AgenticReliabilityStudy(graph, role_overrides={"verifier": "aggregator"})
```

Roles: `source`, `sink`, `llm_agent`, `tool`, `router`, `aggregator`,
`transform`.

**Were the shared resources found?**

```python
study.system.common_cause_groups()
```

If two components call the same model through a wrapper the extractor cannot see
through, say so:

```python
study = AgenticReliabilityStudy(
    graph,
    resource_overrides={
        "drafter": {"llm": "gpt-4o-2024-11-20", "runtime": "api"},
        "critic":  {"llm": "gpt-4o-2024-11-20", "runtime": "api"},
    },
)
```

An undetected shared snapshot is the single most consequential extraction error:
it turns a common-cause single point of failure into an apparently redundant
architecture.

**Were the loops found?**

```python
from hiphopsllm import find_cycles

find_cycles(study.system)
print(study.report.cycle_report.summary())
```

## Are the default hazards yours?

The defaults are `H1` no answer, `H2` incorrect answer accepted, `H3` malformed
answer, `H4` too late, and `H5-<tool>` unsafe execution where a tool's source
contains `eval` or `exec`. If your system boundary is elsewhere — a tool that
writes to a database, an agent that sends email — declare it:

```python
from hiphopsllm import FClass, Hazard, default_hazards

hazards = default_hazards(study.system) + [
    Hazard(
        id="H6",
        name="Unintended write to the production database",
        severity="catastrophic",
        component="db_writer",
        port="out",
        fclass=FClass.COMMISSION,
    ),
]
study = AgenticReliabilityStudy(graph, hazards=hazards)
```

## Getting the outcome data

This is the part that takes real work, and it is worth doing properly.

**What to log.** One row per benchmark item: an id, the stratum, and a `1`/`0`
correctness column *per node*, not only for the system. Per-node columns are what
make the fault tree quantitative rather than illustrative.

```python
rows = []
for item in benchmark:
    state = graph.invoke({"question": item["question"]})
    rows.append({
        "item_id": item["id"],
        "stratum": stratum_of(item),
        "planner":  int(is_correct(state["plan"], item)),
        "worker":   int(is_correct(state["draft"], item)),
        "verifier": int(is_correct(state["final"], item)),
        "split": "calibration" if item["id"] in calibration_ids else "test",
    })
outcomes = pd.DataFrame(rows)
```

**Scoring intermediate nodes is the hard part.** A planner's "correctness" needs
a definition, and a bad one produces a confidently wrong reliability number. Two
approaches that work: score the intermediate output against the same ground truth
where that is meaningful, or define a node-specific oracle (did the plan name the
right tool? did the retrieval return the gold passage?) and say in the report
which you used. `ComponentEvidence.method` carries that string.

**Keep a real split.** Fit on calibration, evaluate on test. The package refuses
to fit CPTs on test rows and uses only the calibration split for basic events,
but it can only do that if the column is there.

**How many items?** Enough that the interval is narrow enough to act on. 160
items gave widths around 0.14 in the bundled example; roughly quadruple the data
to halve that. Check before you commit compute:

```python
from hiphopsllm import EvidenceCalibrator

pilot = EvidenceCalibrator(profile=profile, exact=False)
pilot.fit_component("worker", pilot_outcomes, pilot_strata).width
```

## Run it

```python
study.observe(outcomes, profile=profile)
study.run()
print(study.summary())
```

Then read the calibration report before anything else:

```python
report = study.calibration
report.uncalibrated_components   # no measurement reached these
report.unmatched_evidence        # these measurements matched no component
report.skipped_events            # these kept placeholders, and why
```

An empty `uncalibrated_components` and an empty `unmatched_evidence` mean the
name matching worked. Anything in either list is either a `component_map` you
need to supply or a measurement you need to collect:

```python
study.observe(outcomes, profile=profile,
              component_map={"pass_2_scores": "verifier"})
```

## A worked skeleton

```python
import pandas as pd
from hiphopsllm import AgenticReliabilityStudy, empirical_profile

profile = empirical_profile(production_stratum_labels)   # from logs, not guessed

study = AgenticReliabilityStudy(
    graph,
    name="production QA workflow",
    globals_ns=globals(),
    role_overrides={"verifier": "aggregator"},
    resource_overrides={"drafter": {"llm": MODEL_ID}, "critic": {"llm": MODEL_ID}},
    unroll=1,
)

# 1. structure first — this needs no data at all
study.analyse()
print(study.summary())
pd.DataFrame(study.single_points())

# 2. then measurement
study.observe(outcomes, profile=profile)
study.run()
print(study.calibration.summary())

# 3. then the answer, as an interval
print(study.hazard_probability("H2"))

# 4. and the artefacts
study.save("artifacts/2026-09-01/")
study.bayesnet("H2").view().to_png("artifacts/2026-09-01/bn_h2.png")
```

Run step 1 the day you write the graph. It costs nothing, needs no data, and its
findings — the order-1 cut sets, the undetected shared snapshot — are usually the
ones worth acting on first.
