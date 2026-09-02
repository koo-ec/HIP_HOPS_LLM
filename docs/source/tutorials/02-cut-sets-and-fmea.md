# 2. Cut sets, importance and the FMEA

The structural result is available before any measurement exists, and it is often
the more actionable half. This tutorial reads it.

```python
from hiphopsllm import AgenticReliabilityStudy, load_example

study = AgenticReliabilityStudy(load_example("react_calculator"),
                                name="ReAct + calculator")
study.analyse()
```

## Cut sets

A **minimal cut set** is a smallest combination of basic events sufficient to
produce the hazard. Its *order* is how many events it takes.

```python
for cs in sorted(study.cut_sets("H2"), key=lambda c: (len(c), c)):
    print(len(cs), " + ".join(cs))
```

```text
1 BE-__start__-BADREQ
1 BE-coder-WRONGEXPR
1 BE-generator-HALLUC
1 BE-generator-NONDET
```

Order 1 everywhere. Each of those alone delivers a wrong answer the user accepts.
That is the honest description of a single-agent-plus-tool architecture, and no
amount of prompt engineering changes the *structure*.

Compare the parallel architecture:

```python
parallel = AgenticReliabilityStudy(load_example("parallel_aggregator"))
parallel.analyse()
sorted(len(c) for c in parallel.cut_sets("H2"))
# [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
```

Eight of eleven are order 2 — the vote is real. Three are not, and they are the
interesting three:

```text
BE-__start__-BADREQ                        an ill-posed request
BE-aggregator-OWN                          the judge hallucinated its own answer
CCF-LLM-Qwen-Qwen2-5-Math-1-5B-Instruct    both agents are the same snapshot
```

The last one is why the redundancy is architectural only. Two agents running one
model do not fail independently, and the vote buys nothing against a fault they
share.

## Single points of failure

```python
import pandas as pd
pd.DataFrame(study.single_points())
```

Every order-1 cut set, across all hazards, ranked by severity. This is the fix
list.

## Importance

Two rankings, answering two different questions.

```python
analysis = study.report.analysis("H2")
pd.DataFrame([
    {"event": r.event_id, "P": r.probability,
     "Fussell-Vesely": round(r.fussell_vesely, 4),
     "Birnbaum": round(r.birnbaum, 4), "SPOF": r.single_point}
    for r in analysis.importance
])
```

```text
                 event     P  Fussell-Vesely  Birnbaum  SPOF
0  BE-generator-HALLUC  0.15          0.3947    0.7841  True
1  BE-generator-NONDET  0.12          0.3158    0.7573  True
2   BE-coder-WRONGEXPR  0.10          0.2632    0.7405  True
3  BE-__start__-BADREQ  0.01          0.0263    0.6732  True
```

**Fussell–Vesely** is the fraction of the top-event probability this event
contributes. Rank by it to decide *what to fix first* — here, hallucination in
the generator, which accounts for 39% of the risk.

**Birnbaum** is the sensitivity: how much the top event moves per unit change in
this event. Rank by it to decide *where measurement is most worth buying*. Note
that `BE-__start__-BADREQ` has a Fussell–Vesely of 0.026 but a Birnbaum of 0.67 —
it barely contributes at its current probability, but if that probability were
wrong, the answer would move a lot. That is exactly the event whose placeholder
you should replace with a measurement first.

## The FMEA

```python
study.fmea().head()
```

```text
   component                event            failure mode          class     P  … severity      mitigation
0      coder      BE-coder-UNSAFE  Model-authored code …    Commission  0.02  … catastrophic  Restricted evaluator (AST …
1  generator  BE-generator-HALLUC  Plausible but incorr…  Value (subtle) 0.15  … critical      Sample-based semantic unc…
2  generator  BE-generator-NONDET  Sampling non-determi…  Value (subtle) 0.12  … critical      Pin seed and temperature …
3      coder   BE-coder-WRONGEXPR  Executes correctly b…  Value (subtle) 0.10  … critical      Cross-check with an indep…
4  __start__  BE-__start__-BADREQ  Ill-posed or ambiguo…  Value (subtle) 0.01  … critical      Input validation and clar…
```

It is derived from the trees, not written separately, so it cannot disagree with
them. Each row carries the direct effect (which hazard this event reaches), the
further effects, the worst severity it can cause, and a mitigation appropriate to
its *class* — which is where keeping `VC` and `VS` apart pays off. A schema
validator is a complete answer to `VC` and no answer at all to `VS`.

## What to do with it

Three moves, in the order they usually pay:

**Break the order-1 cut sets.** `BE-coder-UNSAFE` is catastrophic and has a
one-line fix (a restricted AST evaluator instead of `eval`). Fix that before
tuning anything.

**Make the redundancy real.** If two agents share a snapshot, the common-cause
event dominates. Change one of them to a genuinely different model and re-run —
the `CCF-LLM-…` cut set disappears:

```python
study = AgenticReliabilityStudy(
    load_example("parallel_aggregator"),
    resource_overrides={"cot_agent": {"llm": "a-different-model", "runtime": "cuda:1"}},
)
study.analyse()
[c for c in study.cut_sets("H2") if len(c) == 1]
# the CCF cut set is gone
```

**Then measure.** Structure tells you where redundancy is missing; only
measurement tells you whether what remains is good enough.
[Tutorial 3](03-operational-profile.md) does that.

## Exports for other tools

```python
study.save("artifacts/")
```

Writes per-hazard mermaid, DOT, JSON and **Open-PSA MEF XML** (readable by XFTA
and SCRAM), plus `*_cutsets.csv` and `*_fmea.csv`, and the Markdown report that
carries the loop handling and common-cause notes.
