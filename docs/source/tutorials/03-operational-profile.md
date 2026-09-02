# 3. Measuring under an operational profile

Structure told us where the risk *can* come from. Now we measure how much there
actually is.

## The data you need

One row per benchmark item, a stratum label, and a `1`/`0` correctness column per
component you want to calibrate. `1` means correct.

```python
from hiphopsllm import load_outcomes

outcomes = load_outcomes()
outcomes.head(3)
```

```text
     item_id  stratum  react_agent  cot_agent  aggregator        split
0  item_0000   medium            1          1           1  calibration
1  item_0001   medium            1          1           1  calibration
2  item_0002    short            1          1           1  calibration
```

Getting per-component columns means logging each node's answer separately, not
only the system's. If you only have end-to-end outcomes, you can still calibrate
the last component and use `study.operational_failure` for the system as a whole —
but per-component data is what makes the fault tree quantitative rather than
decorative.

## Calibrate

```python
from hiphopsllm import AgenticReliabilityStudy, load_example

study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
study.observe(outcomes, profile={"short": 0.30, "medium": 0.50, "long": 0.20})
study.run()

study.calibration.evidence_frame()
```

```text
     component    n  failures  empirical    lower    upper    width
0  react_agent  160        33    0.21208  0.15425  0.29774  0.14350
1    cot_agent  160        25    0.15262  0.10208  0.24045  0.13837
2   aggregator  160        39    0.23892  0.17422  0.33723  0.16301
```

`n=160` and not 240: the `split` column was honoured and only the calibration
rows used. `empirical` is the *profile-weighted* observed rate, not the raw one —
the stratum mix in the data is not the stratum mix in the workload.

Look at the per-stratum rates to see why that matters:

```python
{k: v.by_stratum for k, v in study.evidence.items()}
```

```text
react_agent  {'short': 0.051, 'medium': 0.151, 'long': 0.607}
cot_agent    {'short': 0.026, 'medium': 0.161, 'long': 0.321}
aggregator   {'short': 0.128, 'medium': 0.258, 'long': 0.357}
```

`react_agent` fails on 5% of short questions and 61% of long ones — a twelvefold
difference. A single pooled accuracy averages that away, and any deployment whose
long-question share differs from the benchmark's inherits a wrong number. This is
the entire argument for stratifying.

## What landed on the tree

```python
study.calibration.to_frame()
```

```text
              basic event  placeholder P  calibrated P    lower    upper
0    BE-aggregator-FORMAT           0.05       0.05736  0.03756  0.07897
1       BE-aggregator-OWN           0.10       0.11143  0.07371  0.15171
2    BE-aggregator-SELECT           0.10       0.11143  0.07371  0.15171
3     BE-cot_agent-FORMAT           0.08       0.03687  0.02130  0.05352
4     BE-cot_agent-HALLUC           0.15       0.06802  0.03957  0.09800
5     BE-cot_agent-NONDET           0.12       0.05480  0.03179  0.07920
6      BE-cot_agent-TRUNC           0.05       0.02321  0.01337  0.03380
7   BE-react_agent-FORMAT           0.08       0.04995  0.03295  0.06825
8   BE-react_agent-HALLUC           0.15       0.09160  0.06089  0.12414
9   BE-react_agent-NONDET           0.12       0.07397  0.04902  0.10061
10  BE-react_agent-TRUNC           0.05       0.03151  0.02072  0.04322
```

Note that `cot_agent` had the *lowest* measured failure rate and its events came
down the most: hallucination from a placeholder 0.15 to a measured 0.068. The
placeholders were pessimistic here; they are just as often optimistic, and there
is no way to know which without measuring.

### How one measurement becomes four events

`react_agent`'s measured probability is 0.212 — for the *component*. Its value
logic is a disjunction of four internal events, so the measurement is a statement
about their union, not about any one of them.

The calibrator rescales the set so that their OR reproduces the measurement while
their prior ratios are preserved. With weights `wᵢ` summing to one:

$$p_i = 1 - (1-P)^{w_i} \quad\Longrightarrow\quad 1 - \prod_i (1 - p_i) = P$$

exactly. You can check it:

```python
import numpy as np

events = [e for e in study.failure_model.events.values()
          if e.component == "react_agent" and e.prob_interval]
1 - np.prod([1 - e.prob for e in events])     # 0.21208 — the measurement
```

The alternative policy puts the whole probability on the heaviest event and zeroes
the rest — blunter, but easier to defend when only one of them is really what you
measured:

```python
study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
study.observe(outcomes, profile=profile)
study.calibrate(policy="dominant")
```

## What was *not* calibrated

```python
print(study.calibration.summary())
```

```text
  11 basic event(s) updated from measurement
  8 event(s) kept their placeholder (class L is outside what a correctness
  measurement can speak about; class O is outside what a correctness measurement
  can speak about)
```

A correctness benchmark measures correctness. Latency and omission events were
left alone and said so. If a component had matched no measurement at all, it
would be listed too:

```python
study.calibration.uncalibrated_components
study.calibration.unmatched_evidence      # measurements that matched no component
```

Components are matched to measurements by shared name tokens. Where that is not
safe — a measurement called `pass_2` for a component called `verifier` — say so:

```python
study.observe(outcomes, profile=profile,
              component_map={"pass_2": "verifier"})
```

A `component_map` pointing at a component that does not exist raises, and lists
the ones that do.

## Choosing the bound

Three envelopes, three questions. The default is the widest.

```python
study.bound = "credible"   # default: outer credible envelope, for a safety case
study.bound = "expected"   # range of posterior means, for a design decision
study.bound = "median"     # range of posterior medians, robust to skew
```

## Speed

Full hierarchical inference is the default and takes about a third of a second on
this data. For a fast sweep:

```python
AgenticReliabilityStudy(spec, exact_inference=False)
```

That substitutes a profile-weighted Jeffreys interval and labels every evidence
string `Jeffreys interval (approximation; not the HIP-LLM posterior)`, so a number
produced this way cannot be mistaken for the real thing in a report.

## Next

[Tutorial 4](04-fault-tree-to-bayesnet.md) turns the calibrated tree into a
Bayesian network — the exact top-event probability, and the posterior over causes.
