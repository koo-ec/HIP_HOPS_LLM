# 5. End to end in ten lines

Everything from the first four tutorials, as one object.

```python
from HIP_HOPS_LLM import AgenticReliabilityStudy, load_example, load_outcomes

study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
study.observe(load_outcomes(), profile={"short": 0.3, "medium": 0.5, "long": 0.2})
study.run()

print(study.summary())
print(study.hazard_probability("H2"))
study.bayesnet("H2").show()
study.save("artifacts/")
```

Or as a chain:

```python
study = (
    AgenticReliabilityStudy(graph, globals_ns=globals())
    .observe(outcomes, profile=profile)
    .run()
)
```

`run()` calls `analyse()` if it has not been called, then `calibrate()` if
outcomes were given. Nothing happens implicitly: a study with no outcomes reports
placeholder probabilities and prints `NOT CALIBRATED` in capitals.

## Comparing architectures

The comparison is where this earns its keep. Same profile, same data, three
architectures:

```python
import pandas as pd
from HIP_HOPS_LLM import EXAMPLES

rows = []
for key in EXAMPLES:
    s = AgenticReliabilityStudy(load_example(key), name=key)
    s.analyse()
    a = s.report.analysis("H2")
    rows.append({
        "architecture": key,
        "components": len(s.system.components),
        "P(H2)": round(a.quant.top_probability, 4),
        "cut sets": len(a.cuts.sets),
        "min order": min(len(c) for c in a.cuts.sets),
        "SPOFs": len(a.single_points),
    })
pd.DataFrame(rows)
```

```text
          architecture  components   P(H2)  cut sets  min order  SPOFs
0     react_calculator           6  0.3335         4          1      4
1  parallel_aggregator           5  0.2787        11          1      3
2   supervisor_workers           8  0.4293        14          1      5
```

Two things worth noticing, and neither is obvious from an architecture diagram.

**More components is not more reliable.** `supervisor_workers` has the most parts
and the worst `P(H2)`. Its router, its three specialists and its verifier are all
in series on the value path — a supervisor that routes to the wrong specialist
and a verifier that passes a wrong answer are each single points of failure. Depth
without redundancy just adds ways to be wrong.

**Every one of them has `min order == 1`.** None of these architectures is free of
single points of failure for the critical hazard.

## Does diversity help?

The parallel architecture's redundancy is defeated by a shared model snapshot.
Test the counterfactual without changing any code:

```python
cases = {
    "as built (shared snapshot)": {},
    "cot_agent diversified": {
        "cot_agent": {"llm": "gpt-4o-mini", "runtime": "api"},
    },
    "all three diversified": {
        "cot_agent": {"llm": "gpt-4o-mini", "runtime": "api"},
        "aggregator": {"llm": "claude-sonnet-4-5", "runtime": "api2"},
    },
}
for name, overrides in cases.items():
    s = AgenticReliabilityStudy(load_example("parallel_aggregator"),
                                name=name, resource_overrides=overrides)
    s.analyse()
    a = s.report.analysis("H2")
    order_1 = sorted(next(iter(c)) for c in a.cuts.sets if len(c) == 1)
    print(f"{name:<28} P(H2)={a.quant.top_probability:.4f}  order-1={order_1}")
```

```text
as built (shared snapshot)   P(H2)=0.2787  order-1=['BE-__start__-BADREQ',
                                                    'BE-aggregator-OWN',
                                                    'CCF-LLM-Qwen-…']
cot_agent diversified        P(H2)=0.2787  order-1=['BE-__start__-BADREQ',
                                                    'BE-aggregator-OWN',
                                                    'CCF-LLM-Qwen-…']
all three diversified        P(H2)=0.2160  order-1=['BE-__start__-BADREQ',
                                                    'BE-aggregator-OWN']
```

**Diversifying one agent changes nothing.** The common-cause group shrinks from
three members to two — but the judge is still one of them, and the judge is on
the value path on its own, so `CCF-LLM-…` remains an order-1 cut set. You have to
diversify the judge too before the vote is worth anything, and then `P(H2)` drops
by a fifth.

This is a result you would be unlikely to reach by reasoning about the diagram,
and it is the kind of thing the package exists to surface.

## Profile sensitivity

The answer is conditional on the workload. Vary it:

```python
profiles = {
    "as measured": {"short": 0.30, "medium": 0.50, "long": 0.20},
    "harder":      {"short": 0.05, "medium": 0.15, "long": 0.80},
    "easier":      {"short": 0.70, "medium": 0.25, "long": 0.05},
}
for name, profile in profiles.items():
    s = AgenticReliabilityStudy(load_example("parallel_aggregator"))
    s.observe(load_outcomes(), profile=profile).run()
    print(f"{name:<12} P(H2) ∈ {s.hazard_probability('H2')}")
```

Nothing about the system changed between those three rows. Quoting a single
failure probability without saying which workload it is conditional on is
therefore not a claim about the system at all.

## What `save()` writes

```python
study.save("artifacts/")
```

```text
artifacts/agentic_workflow_report.md          the analysis, incl. loop and CCF notes
artifacts/agentic_workflow_architecture.mmd   the system as mermaid
artifacts/agentic_workflow_H2.mmd/.dot/.json  the tree, three ways
artifacts/agentic_workflow_H2.opsa.xml        Open-PSA MEF, for XFTA / SCRAM
artifacts/agentic_workflow_cutsets.csv        every cut set, with its probability
artifacts/agentic_workflow_fmea.csv           the generated FMEA
artifacts/calibration.csv                     placeholder → measured, per event
artifacts/evidence.csv                        n, failures, interval, per component
```

The last two are the provenance record. Six months later they are the difference
between a number someone can check and a number someone has to trust.
