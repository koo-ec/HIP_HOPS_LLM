# Operational profiles

A benchmark accuracy is a statement about the benchmark's mix of tasks. A
*reliability* claim has to be a statement about the mix the system will meet in
service, and those two are rarely the same.

Concretely: a model measured on a corpus that is 18% long multi-hop questions,
then deployed on a workload that is 70% of them, does not keep its measured
failure rate. Nothing about the model changed; the question changed.

The operational profile is that mix — a partition of the input space into strata,
with the probability of each. Everything downstream is conditional on it.

## Building one

```python
from hiphopsllm import OperationalProfile, empirical_profile, uniform_profile

# declared
profile = OperationalProfile({"short": 0.30, "medium": 0.50, "long": 0.20})

# measured from logged usage
profile = empirical_profile(observed_stratum_labels)

# assumed, and recorded as an assumption
profile = uniform_profile(["short", "medium", "long"])
```

Each carries its own provenance, printed in every report that uses it:

```python
print(profile.summary())
```

```text
operational profile  (3 strata)
  short    0.300  █████████
  medium   0.500  ███████████████
  long     0.200  ██████
  provenance: declared by the analyst
```

That line is not decoration. "Declared by the analyst" and "observed frequencies
over 12,480 logged items" support very different claims, and a reader six months
later cannot tell them apart from the numbers alone.

## How this maps onto HIP-LLM's hierarchy

HIP-LLM's operational profile is hierarchical (paper Definition 2, Section 3.2.1):

```text
subdomain -> domain :  p_i   = sum_j  Omega_ij * theta_ij     (sum_j Omega_ij = 1)
domain    -> LLM    :  p_L   = sum_i  W_i * p_i               (sum_i W_i     = 1)
```

`theta_ij` is the failure probability in subdomain *j* of domain *i*; `Omega_ij`
and `W_i` are the profile weights. The paper also gives the algebraically
identical flat form `OP_ij = W_i * Omega_ij`.

**A stratum here is a HIP-LLM subdomain**, and this package uses the flat,
single-domain case: your `{label: weight}` mapping is `Omega`, and there is one
domain, so `p_L` is the profile-weighted average of the per-stratum failure
probabilities. That is what
`hiphopsllm.reliability.hipllm.OperationalFailureProb` is given, at
`level="benchmark_stratum"`.

If you need several domains, use the engine directly:
`hiphopsllm.reliability.hipllm.posterior.run_domain` with `DomainData` and
`SubdomainData` from the same module.

## What is actually reported

The measurement is **the probability that a component fails on one task drawn
from the profile**, and the reliability
**R(n) = P(failure-free over n future tasks)**, which is HIP-LLM's definition of
reliability. Neither is about any particular failure mode:

```python
study.operational_reliability(n_tasks=10)
study.evidence["react_agent"].statement(n_tasks=10)
study.evidence["react_agent"].reliability(100)
```

```text
react_agent fails on a task with probability [0.1542, 0.2977] under the
operational profile (short 30%, medium 50%, long 20%), from 33/160 observed
failures; failure-free over 10 tasks with probability [0.0736, 0.1003]
```

The fault tree then *decomposes* each of those numbers over the component's
failure modes so it can be propagated through the architecture. That split is a
modelling step; the measurement is the statement above.

## Choosing the strata

The strata must partition the input space, and they should be chosen so that
failure probability varies *between* strata more than *within* them. Otherwise
the stratification buys nothing.

For question answering, decomposition length works well, and HIP-LLM ships the
StrategyQA stratifier that produced it:

```python
from hiphopsllm import decomposition_stratum, load_strategyqa

items = load_strategyqa("train")
strata = [decomposition_stratum(item) for item in items]
```

:::{warning}
Use HIP-LLM's own `load_strategyqa` and `decomposition_stratum`. The Hugging Face
mirror `ChilleD/StrategyQA` exposes `facts`, not `decomposition`; substituting it
collapses every task into one stratum and silently destroys the profile. The
loader raises rather than falling back.
:::

For your own workload, `stratify` takes a callable and checks the labels against
the profile:

```python
from hiphopsllm import stratify

labels = stratify(
    requests,
    lambda r: "long" if len(r["question"].split()) > 40 else "short",
    profile=profile,      # an unexpected label is an error, not a zero weight
)
```

Passing the profile matters. A stratum with no weight silently drops those items
from every downstream estimate — the kind of error that survives all the way into
a published number.

## Using one

The profile-weighted average of any per-stratum quantity:

```python
profile.expected({"short": 0.05, "medium": 0.12, "long": 0.31})   # 0.137
```

A missing stratum raises rather than renormalising, for the same reason.

Conditioning on a subset — "what if we only served short questions?":

```python
profile.restricted_to(["short", "medium"])     # renormalised, provenance updated
```

## Sample size follows the profile

A stratum that carries 5% of the weight still needs enough items to estimate its
own failure rate; otherwise its interval is wide and the profile-weighted
envelope inherits that width. When a stratum has *no* items at all, the fast
Jeffreys path returns `[0, 1]` for it — the honest answer — and the envelope goes
correspondingly wide. That is a signal to collect data, not to drop the stratum.

## The two OperationalProfile classes

HIP-LLM's schema keeps parallel `labels`/`weights` arrays; this package's class is
mapping-shaped and more convenient. They convert both ways:

```python
from hiphopsllm import OperationalProfile
from hiphopsllm.reliability.hipllm import HIPLLMOperationalProfile

engine_profile = profile.to_hipllm()               # -> hip_llm.schemas version
OperationalProfile.coerce(engine_profile)          # -> back again
```

Anything in this package that takes a `profile=` argument accepts either, or a
plain `dict`.

## Never let the benchmark be the profile by default

`dataset_proportional_profile` exists so that using the dataset's own mix as the
workload is a *named choice* rather than something that happens quietly. HIP-LLM
names it too (paper Section 4.2, Remark 7).

```python
from hiphopsllm import dataset_proportional_profile, empirical_profile

dataset_proportional_profile(strata)   # "my test set looks like production"
empirical_profile(production_labels)   # production traffic, actually measured
```

The arithmetic is identical. The claim is not, which is why they are separate
functions and why their `provenance` strings differ. If you call `observe()`
without a profile you get the first one, and a warning saying so.

## Why it is never inferred

`AgenticReliabilityStudy.calibrate()` refuses to run without one:

```text
StudyNotReady: no operational profile has been set; pass profile=... to the
study or to observe(). Every failure probability is conditional on it, so it is
never inferred silently.
```

The obvious default — use the benchmark's own mix — is exactly the mistake this
layer exists to prevent. If the benchmark's mix *is* your workload, say so
explicitly with `profile=empirical_profile(strata)`, which records that choice.
