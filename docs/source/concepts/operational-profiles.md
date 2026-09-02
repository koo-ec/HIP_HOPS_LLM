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
