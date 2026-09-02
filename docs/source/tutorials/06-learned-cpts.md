# 6. Learned CPTs

The synthesised tree says an aggregator delivers a wrong answer when *both*
inputs are wrong. That is a modelling assumption, and in a real multi-agent
system it is usually wrong in an interesting direction: a judge repairs some
upstream errors and introduces others, so the true table is neither AND nor OR.

When per-node outcomes have been logged, stop assuming and estimate.

## Fit a gate

`learn_gate` needs *failure* indicators, so invert the correctness columns once,
explicitly:

```python
from hiphopsllm import load_outcomes
from hiphopsllm.bayes.learn import learn_gate

outcomes = load_outcomes()
calibration = outcomes[outcomes["split"] == "calibration"].copy()
for column in ("react_agent", "cot_agent", "aggregator"):
    calibration[column] = 1 - calibration[column]        # correct → failed

learned, distance = learn_gate(calibration, "aggregator", ["react_agent", "cot_agent"])
distance
```

```text
{'and': 0.1225, 'or': 0.4947, 'nearest': 'and'}
```

`distance` is the mean absolute difference between the fitted `P(Fail | parents)`
column and each deterministic table. The gate is much closer to AND than to OR —
the redundancy is real — but 0.12 away from it, and that gap is where the
interesting behaviour lives.

```python
import pandas as pd

pd.DataFrame(learned.cpt.rows())
```

```text
  react_agent cot_agent   P(Fail)
0          OK        OK  0.181818
1          OK      Fail  0.200000
2        Fail        OK  0.055556
3        Fail      Fail  0.947368
```

Read the four rows against the AND gate the tree assumed (`0, 0, 0, 1`):

**Row 0 — both drafts right, the judge fails 18% of the time.** The AND gate says
zero. This is `BE-aggregator-OWN` and `BE-aggregator-SELECT` in the synthesised
tree, and the data says together they account for a fifth of all failures. A
model without them is optimistic by exactly that much.

**Rows 1 and 2 are not symmetric** (0.20 vs 0.056). The judge recovers from a bad
`cot_agent` draft far more often than from a bad `react_agent` one. No
deterministic gate can express that, and no amount of tuning the basic-event
probabilities will produce it.

**Row 3 — both wrong, the judge is right 5% of the time.** It occasionally
produces a correct answer neither agent gave.

## How much does the assumption cost?

This is the motivating result of the HIP-MAS study, reproduced under known
ground truth: with a reviewer repairing 55% of upstream errors, the deterministic
AND-series gate mispredicted held-out failure by **+0.386**, while the learned-CPT
model was within **0.005**.

You can see the mechanism directly here:

```python
import numpy as np

fitted = learned.cpt.table[..., 1]
p_react, p_cot = 0.212, 0.153          # measured component failure rates

and_gate  = p_react * p_cot                                    # 0.032
learned_p = sum(
    fitted[(a, b)] * (p_react if a else 1 - p_react) * (p_cot if b else 1 - p_cot)
    for a in (0, 1) for b in (0, 1)
)
print(f"AND gate: {and_gate:.4f}   learned: {learned_p:.4f}")
```

The AND gate predicts about 3% because it can only fail when both inputs do. The
learned table predicts far more, because row 0 says the judge fails on its own.

## Fit a whole network

```python
from hiphopsllm import BayesianNetwork
from hiphopsllm.bayes.learn import fit_cpts

cpts, fits = fit_cpts(
    outcomes[outcomes["split"] == "calibration"],
    {"react_agent": [], "cot_agent": [], "aggregator": ["react_agent", "cot_agent"]},
    outcomes_are_failures=False,        # the columns hold 1 = correct
)
network = BayesianNetwork(cpts=cpts, name="learned")

network.p_fail("aggregator")
network.p_fail("aggregator", evidence={"react_agent": "Fail", "cot_agent": "Fail"})
```

The structure must be given parents-first, and `outcomes_are_failures` states the
polarity of the columns once, here, rather than silently everywhere.

## Two guards

**Smoothing is not cosmetic.** At pilot sample sizes several parent
configurations — "both agents wrong *and* they agree" — are observed a handful of
times or not at all. An unsmoothed maximum-likelihood estimate would put a hard
`0` or `1` in the table and make the network claim a certainty it has not earned.
`alpha=1` (Laplace) is the default; the raw counts come back with the table so a
report can say how much of it was measured:

```python
learned.n_observations          # 160
learned.total_rows              # 4
learned.prior_dominated_rows    # 0  — every configuration was observed
learned.coverage                # 1.0
print(learned.summary())
```

```text
aggregator: 160 observations over 4 rows, 0 prior-dominated
(coverage 100%, Dirichlet alpha=1.0)
```

A coverage below 1.0 means some rows of your CPT are the prior, not the data.
That is fine, as long as the report says so.

**A CPT is never fitted on a test split.** If the frame carries rows marked
`test`, `held_out` or `eval`, `learn_cpt` raises:

```text
CPTLearningError: learn_cpt('aggregator') was given rows marked ['test'] in
column 'split'. Fit CPTs on the calibration split only; a table fitted on the
evaluation set makes every downstream number optimistic and untestable.
```

Override only when you mean it: `check_split=False`.

## When to use which

| | Use the **synthesised** tree | Use a **learned** CPT |
|---|---|---|
| You have | an architecture, and per-component failure rates at most | per-item outcomes for every node |
| It gives you | cut sets, SPOFs, an FMEA, structural insight | an accurate predictive model of one gate |
| It cannot | say what a component *actually* does, only what its archetype does | tell you *why*, or generalise to an architecture you have not run |

They are complements. Use the tree to find where redundancy is missing; use a
learned table where you have data and the archetype is visibly wrong.
