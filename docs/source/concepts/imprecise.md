# Imprecise probability

You ran a benchmark. 160 items, 34 of them wrong. What is the probability the
next item fails?

`34/160 = 0.2125` is one answer, and it is the wrong shape of answer. It asserts
three significant figures from 160 observations. Ask the same question of a
second sample of 160 and you get a visibly different number, which means the
first one was never worth three figures.

HIP-LLM's answer is an interval, and this package carries that interval all the
way to the top event.

## Where the interval comes from

HIP-LLM's model is hierarchical: item outcomes within a stratum, stratum failure
rates within a domain, and hyperparameters above those. The hyperparameters are
not given a single prior; they are given an admissible *interval*, and the
posterior is computed at many configurations drawn from it. What comes back is a
family of posteriors, one per configuration — and the reported bounds are the
envelope over that family.

```python
from HIP_HOPS_LLM import OperationalFailureProb, quick_inference_settings

estimator = OperationalFailureProb(
    profile={"short": 0.30, "long": 0.70},
    settings=quick_inference_settings(samples=1500, configurations=48),
)
result = estimator.fit(outcomes=[1, 1, 0, 1, 0, 0, 1, 0],
                       strata=["short"] * 4 + ["long"] * 4)
result.summary()
```

Three envelopes are available, and they answer different questions:

| Envelope | What it is | Use it when |
|---|---|---|
| `posterior_expected_failure_bounds` | Range of the posterior *mean* across configurations | The interval feeds a design decision — tightest, and about the central estimate |
| `posterior_median_failure_bounds` | Range of the posterior *median* | Same, robust to a skewed posterior |
| `posterior_credible_envelope` | Outer equal-tail credible interval across *all* configurations | Making a safety argument — widest, and the most cautious reading |

`EvidenceCalibrator` defaults to `credible`. Change it deliberately:

```python
EvidenceCalibrator(profile=profile, bound="expected")   # tighter
```

## Carrying it through the fault tree

A **coherent** fault tree — one built only from AND and OR gates, with no negated
inputs — has a top-event probability that is *monotone non-decreasing* in every
basic-event probability. Raising any leaf can only raise the top.

That single fact does all the work. It means the interval on the top event is
obtained by evaluating at the **corners**: all lower bounds gives a guaranteed
lower bound, all upper bounds a guaranteed upper bound. No optimisation over the
interval box, no sampling, no approximation.

```python
imprecise = study.imprecise_bayesnet("H2")
imprecise.envelope()          # [0.172738, 0.291474]
```

Internally that is two complete Bayesian networks:

```python
imprecise.lower.p_fail()      # every leaf at its lower bound
imprecise.upper.p_fail()      # every leaf at its upper bound
```

The test suite verifies monotonicity directly, by perturbing each basic event and
checking the top event never falls. If a future gate type breaks coherence, that
test fails and the interval arithmetic stops being licensed — which is the point
of having it.

## What an interval licenses you to say

**Yes:** "Under this operational profile, the probability that an incorrect answer
is delivered and accepted lies in [0.173, 0.291]." That is a claim you can defend
from 160 items.

**Yes:** "Architecture A's envelope is [0.17, 0.29] and B's is [0.31, 0.44]; they
do not overlap, so B is worse under this profile." Non-overlapping envelopes are
a real comparison.

**No:** "The failure probability is 0.232." The midpoint is a convenience for
plotting, not a result. `ComponentEvidence.point` exists because the fault tree's
`prob` field needs a scalar, and it is always accompanied by `prob_interval`.

**Careful:** overlapping envelopes do not establish that two architectures are
equivalent; they establish that this much data cannot separate them. The
remedies are more data or a tighter bound choice, both of which are honest, and
narrowing the interval by fiat, which is not.

## Interval width is a measurement, too

Width tells you how much the estimate is worth:

```python
for name, e in study.evidence.items():
    print(f"{name:<14} [{e.interval[0]:.4f}, {e.interval[1]:.4f}]  width {e.width:.4f}")
```

```text
react_agent    [0.1542, 0.2977]  width 0.1435
cot_agent      [0.1021, 0.2405]  width 0.1384
aggregator     [0.1742, 0.3372]  width 0.1630
```

A width of 0.14 on a component whose point estimate is 0.21 is not a precise
measurement, and the report saying so is more useful than a report implying
otherwise. Collecting four times the data roughly halves it.

## The fast path, and when not to use it

Full hierarchical inference is the default. For exploration there is an
approximation:

```python
EvidenceCalibrator(profile=profile, exact=False)   # profile-weighted Jeffreys
```

It is far faster and adequate for "does this architecture change help?". It is
*not* the HIP-LLM posterior, and it labels itself accordingly in every evidence
string it writes:

```text
Jeffreys interval (approximation; not the HIP-LLM posterior): 34/160 observed
failures under the profile {'short': 0.3, 'medium': 0.5, 'long': 0.2};
P ∈ [0.1324, 0.3171]
```

so a number produced this way can never be mistaken for the real thing in a
report. Use the exact path for anything you publish — on the bundled example it
takes about a third of a second.

## Further reading

R. Aghazadeh-Chakherlou, Q. Guo, S. Khastgir, P. Popov, X. Zhang and X. Zhao,
"A hierarchical imprecise probability approach to reliability assessment of large
language models", *Reliability Engineering & System Safety* **272** (2026) 112615.
<https://doi.org/10.1016/j.ress.2026.112615>
