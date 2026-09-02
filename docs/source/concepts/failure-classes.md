# Failure classes

Six guidewords, adapted from the classical HAZOP/HiP-HOPS set to language-model
components.

| Code | Class | Meaning in an agentic system |
|---|---|---|
| `O` | Omission | No output: crash, empty generation, unwritten state key, dead branch |
| `C` | Commission | Output or effect when there should be none — model-authored code executed |
| `VC` | Value, coarse | Wrong **and detectable**: malformed, truncated, schema violation |
| `VS` | Value, subtle | Wrong but **plausible** — the hallucination case; no downstream check catches it |
| `E` | Early | Terminates before its preconditions are met (a premature "Final Answer") |
| `L` | Late | Latency or token budget exceeded |

```python
from hiphopsllm import FClass

FClass.VALUE_SUBTLE.value    # 'VS'
FClass.VALUE_SUBTLE.title    # 'Value (subtle, undetectable)'
```

## Why VC and VS are separate

This is the single most important modelling decision in the library, and it is
worth being explicit about why.

`VC` and `VS` propagate **identically** through the architecture. Both are "the
value is wrong". A naive model would merge them into one `V` class and halve the
size of every failure table.

They have **opposite consequences**. A `VC` deviation is caught at the system
boundary: the JSON does not parse, the schema check fires, the retry loop
triggers. The user sees an error — a reliability problem, but a *visible* one. A
`VS` deviation is delivered. The answer is fluent, well-formed, confidently
phrased and wrong, and nothing in the system notices.

Merging them produces one number for two hazards with different severities — and
it is exactly the wrong number, because most of the probability mass lives in the
detectable class while most of the *risk* lives in the subtle one. Keeping them
apart is what lets `H2` (incorrect answer accepted as correct, critical) be a
different question from `H3` (malformed answer delivered, minor).

It also changes what mitigations the FMEA proposes. A schema validator is a
complete answer to `VC` and no answer at all to `VS`; only independent
redundancy, an external oracle, or a human is.

## The archetype library

Each component role gets its local failure logic from a builder in
`hiphopsllm.faulttree.failure`. Abridged — the module has the events,
rationales and mitigations in full.

**LLM agent** — trusts its input, so `VS` passes straight through:

```text
O-out  = BE-EMPTY   OR BE-CTX     OR O-in
VC-out = BE-FORMAT  OR BE-TRUNC   OR VC-in
VS-out = BE-HALLUC  OR BE-NONDET  OR VS-in
L-out  = BE-LATE    OR L-in
```

**Tool / executor**:

```text
O-out  = BE-PARSE      OR O-in     # the tool call could not be extracted
VC-out = BE-EXECERR    OR VC-in    # exception text returned as an observation
VS-out = BE-WRONGEXPR  OR VS-in    # runs fine, computes the wrong thing
C-out  = BE-UNSAFE                 # only when eval()/exec() is present
```

**Router** (materialised from `add_conditional_edges`):

```text
O-out  = BE-NOMATCH    OR O-in     # no branch matched -> END with nothing
VC-out = BE-MISROUTE   OR VC-in
E-out  = BE-EARLYSTOP              # termination token seen inside the reasoning
VS-out = VS-in                     # transparent to subtle errors
```

Note the last line. A router cannot detect a subtly wrong answer, so it passes
one through unchanged. That is why adding a router to a chain does not improve
`H2`.

**Aggregator** — the only archetype whose value logic is a conjunction:

```text
VS-out = BE-SELECT OR BE-OWN OR (VS-in-1 AND VS-in-2 AND ...)
```

The `AND` is the redundancy: the aggregator delivers a subtly wrong answer only
if *every* input was subtly wrong — or if it mis-selected between good ones
(`BE-SELECT`), or hallucinated its own (`BE-OWN`). Those two `OR` terms are why
an aggregator is not free: it is itself a component that can fail, and in the
bundled example `BE-aggregator-OWN` is an order-1 cut set for the critical
hazard.

**Feedback cut** — the pseudo-component that closes an unrolled loop:

```text
O-out = BE-EXHAUST OR O-in        # the loop ran out of iterations
```

## Which classes a measurement may speak about

A correctness benchmark measures correctness. When `EvidenceCalibrator` writes
measured intervals into a model, it touches only the value classes:

```python
from hiphopsllm.reliability.calibration import VALUE_CLASSES

VALUE_CLASSES     # (FClass.VALUE_SUBTLE, FClass.VALUE_COARSE)
```

Latency (`L`) and omission (`O`) events keep their placeholders, and the
calibration report says how many, and why:

```text
8 event(s) kept their placeholder (class L is outside what a correctness
measurement can speak about; class O is outside what a correctness measurement
can speak about)
```

To calibrate those, measure them. Time each node, record a `1`/`0` for "within
budget", and calibrate a second time against the latency class:

```python
from hiphopsllm import EvidenceCalibrator, FClass

latency = EvidenceCalibrator(profile=profile, classes=(FClass.LATE,))
latency.apply(study.failure_model, latency.fit_many(within_budget_by_agent))
```
