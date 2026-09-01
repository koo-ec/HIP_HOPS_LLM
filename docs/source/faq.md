# FAQ

## Do I need pyAgrum?

No. It is a genuine optional dependency: every probability the package reports is
also computable by exact enumeration in NumPy alone.

```python
network.p_fail(engine="exact")
```

The two engines are cross-checked against each other precisely so that the NumPy
one can be trusted:

```python
network.cross_check()    # {'agree': 1.0, 'relative_difference': 4.9e-16, …}
```

pyAgrum scales further — exact enumeration refuses above 24 basic events per tree
— so install it for large systems.

## Do I need LangGraph?

Only to analyse a live compiled graph. The extractor also accepts mermaid text
and specification dicts, which is how every example in these docs runs without
it.

## `bn.show()` produced a matplotlib figure, not pyAgrum. Why?

Graphviz's `dot` binary is not on your PATH; the caption says so. See
[Installation](install.md#graphviz). Nothing is wrong, and the numbers are
identical.

## The common-cause group was not detected

The extractor reads model ids from node source text and, when you pass
`globals_ns=globals()`, from the live objects. A model reached through a wrapper
it cannot see through will be missed. Declare it:

```python
AgenticReliabilityStudy(graph, resource_overrides={
    "drafter": {"llm": "gpt-4o-2024-11-20"},
    "critic":  {"llm": "gpt-4o-2024-11-20"},
})
```

Always check `study.system.common_cause_groups()` before trusting a redundancy
claim. An undetected shared snapshot turns a single point of failure into an
apparently redundant architecture — the most consequential extraction error there
is.

## Why is my `P(H2)` so high?

Most likely because it is uncalibrated. Placeholder probabilities are
deliberately pessimistic engineering judgement, and `study.summary()` prints
`NOT CALIBRATED` in capitals when they are in force. Measure, then look again.

If it *is* calibrated and still high, read the cut sets. Order-1 cut sets
dominate, and in agentic architectures they are usually a shared model snapshot
or an aggregator that can fail on its own.

## Can I use it without any outcome data?

Yes, and it is worth doing on day one. `study.analyse()` gives cut sets, single
points of failure, common-cause groups and an FMEA from the architecture alone.
Those findings are structural, cost nothing, and are often the ones worth acting
on first.

## What does an interval like `[0.17, 0.29]` actually mean?

That the probability the hazard occurs on one request drawn from the stated
operational profile lies in that range, given the observed data and HIP-LLM's
admissible hyperparameter set. It is wide because a few hundred items do not pin
it down. See [Imprecise probability](concepts/imprecise.md) for what you may and
may not conclude from one.

## Two architectures have overlapping envelopes. Which is better?

This much data cannot tell you. Overlapping envelopes establish that the
comparison is unresolved, not that the architectures are equivalent. Collect more
data, choose a tighter bound (`bound="expected"`), or compare them structurally —
cut-set orders and single points of failure are a real comparison that needs no
data at all.

## Why does calibration change the numbers but not the cut sets?

Because they are different things. Cut sets are a property of the Boolean
structure, which measurement does not touch; probabilities are a property of the
leaves, which is all measurement changes. A test asserts exactly this.

## Can I calibrate latency or omission events?

Yes, but not with a correctness benchmark — it says nothing about them, so they
are deliberately left on placeholders and reported as such. Measure them
separately and calibrate against the relevant class:

```python
from HIP_HOPS_LLM import EvidenceCalibrator, FClass

latency = EvidenceCalibrator(profile=profile, classes=(FClass.LATE,))
latency.apply(study.failure_model, latency.fit_many(within_budget_by_agent))
```

## My aggregator is a majority vote, not an AND

Override the gate:

```python
study.bayesnet("H2", gate_overrides={"G7": ("KOFN", 2)})
```

Or fit it from data — see [Tutorial 6](tutorials/06-learned-cpts.md).

## Exact enumeration refused

More than 24 basic events in one tree. Use `engine="pyagrum"`. If pyAgrum is also
too slow, the tree is large enough that the cut sets are the right tool.

## A component was left uncalibrated

`study.calibration.uncalibrated_components` lists them, and
`unmatched_evidence` lists measurements that reached no component. Components are
matched to measurements by shared name tokens; where that is unsafe, say so
explicitly:

```python
study.observe(outcomes, profile=profile, component_map={"pass_2": "verifier"})
```

## How do I cite this?

Cite the HIP-LLM paper and this repository; `CITATION.cff` has both.
