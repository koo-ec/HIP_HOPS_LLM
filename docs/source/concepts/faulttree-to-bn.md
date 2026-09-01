# Fault trees as Bayesian networks

A fault tree and a Bayesian network are two readings of the same object. Every
node of the tree becomes a binary variable; every gate becomes a conditional
probability table over its inputs; every basic event becomes a root with its
failure probability as prior.

Converting rather than re-authoring matters for a practical reason. The notebook
this package generalises hand-wired a pyAgrum network *beside* its fault tree,
and nothing kept the two consistent when the graph changed. Generating one from
the other removes that failure mode entirely.

## The conversion

```python
from HIP_HOPS_LLM import fault_tree_to_cpts

cpts = fault_tree_to_cpts(study.report.tree("H2"), study.failure_model)
print(cpts.summary())
```

```text
CPT set — FT_H2
  variables      15
  basic events   8
  gates          7 (0 soft, 7 deterministic)
  top event      TOP
  table rows     70
```

Inspect any table:

```python
cpts.to_frame("BE-react_agent-HALLUC")
```

State indices are `0 = OK` and `1 = Fail` throughout. (The HIP-MAS study code
uses the opposite order; `cpts.to_hipmas_order()` converts.)

## Gate semantics

| Gate | Table | When |
|---|---|---|
| `OR` | fails iff any input fails | the default; classical fault tree reading |
| `AND` | fails iff all inputs fail | redundancy — an aggregator's value logic |
| `KOFN` | fails iff at least *k* of *n* fail | a majority vote; `k=1` is OR, `k=n` is AND |
| `NOISY_OR` | each failed input propagates with probability `pᵢ`, plus a leak | when propagation is probabilistic, not certain |

```python
from HIP_HOPS_LLM import deterministic_gate_cpt, k_of_n_cpt, noisy_or_cpt

deterministic_gate_cpt(3, "OR")
k_of_n_cpt(3, 2)                       # two-of-three majority
noisy_or_cpt([0.7, 0.3], leak=0.02)
```

Noisy-OR is the honest option when an input deviation only *sometimes*
propagates — a malformed tool observation an agent occasionally recovers from —
and the leak covers causes the tree does not model, which for an LLM component
is never an empty set:

$$P(\text{child}=\text{Fail} \mid \text{parents}) = 1 - (1-\ell)\prod_{i \in \text{failed}} (1 - p_i)$$

Softening changes what the network means, so it says so:

```python
cpts = fault_tree_to_cpts(tree, fmodel, soft_gates=True)
cpts.notes
# ['OR gates were softened to noisy-OR: the network no longer encodes the same
#   Boolean function as the minimal cut sets']
```

Override individual gates where the synthesised structure is right but the logic
is not — a two-of-three aggregator being the usual case:

```python
fault_tree_to_cpts(tree, fmodel, gate_overrides={"G7": ("KOFN", 2)})
```

## What the network buys

### An exact top-event probability

Cut-set quantification uses the **minimal cut upper bound**:

$$P_{\text{MCUB}} = 1 - \prod_{c \in \text{cut sets}} \left(1 - \prod_{e \in c} p_e\right)$$

which treats cut sets as independent. They are not — in agentic architectures
they always share basic events, because every agent depends on the same request
and often on the same model snapshot. So MCUB over-estimates.

```python
network = study.bayesnet("H2")
network.compare_with_cutsets(study.report.analysis("H2"))
```

```text
{'exact_bayesnet': 0.2262, 'minimal_cut_upper_bound': 0.2307,
 'rare_event_sum': 0.2521, 'bound_overestimate': 0.0045}
```

Together they bracket the answer: the exact value below, the bound above. On a
coherent tree `bound_overestimate` can never be negative, and the test suite
asserts that on every hazard of every bundled example — a negative value would
mean the tree and the network had drifted apart.

### Diagnosis

Condition on what a run actually showed, and read the posterior over causes:

```python
network.posteriors({"BE-aggregator-OWN": "Fail"})
```

```text
BE-aggregator-OWN                          1.0000
BE-aggregator-SELECT                       0.1103
BE-react_agent-HALLUC                      0.0910
CCF-LLM-Qwen-Qwen2-5-Math-1-5B-Instruct    0.0800
```

Evidence keys may be basic-event ids, fault tree node ids or variable names, and
states may be written `"Fail"`, `1` or `True`. An unknown key lists what is
available rather than failing obscurely.

The single most probable joint assignment, where the tree is small enough for it
to be meaningful:

```python
network.most_probable_explanation({"TOP": "Fail"})
```

### Two engines, cross-checked

Every quantity is computable two ways: through pyAgrum, and by exact enumeration
over the basic events in NumPy alone. They share no code, so agreement is real
evidence that the conversion is right.

```python
network.cross_check()
```

```text
{'exact': 0.226231129671561, 'pyagrum': 0.22623112967156112,
 'difference': 1.11e-16, 'relative_difference': 4.9e-16, 'agree': 1.0,
 'compared': 1.0}
```

This also means **pyAgrum is optional**. Without it, `engine="exact"` answers
every question the package asks, up to about 24 basic events per tree.

## Learned tables

Where per-node outcomes have been logged, a gate's table can be *estimated*
rather than assumed. See [Tutorial 6](../tutorials/06-learned-cpts.md); the short
version is that a reviewer which repairs some upstream errors and introduces
others is neither AND nor OR, and assuming it is one of them can be wrong by a
lot:

> In the HIP-MAS synthetic ground-truth study, with a reviewer repairing 55% of
> upstream errors, the deterministic AND-series gate mispredicted held-out
> failure by **+0.386** while the learned-CPT model was within **0.005**.

```python
from HIP_HOPS_LLM import learn_gate

learned, distance = learn_gate(observations, "aggregator", ["react", "cot"])
distance    # {'and': 0.31, 'or': 0.44, 'nearest': 'and'}
```

## Limits

**Gate fan-in.** A deterministic table has `2ⁿ` rows, so gates are built up to
18 inputs and refuse beyond that with a message pointing at the cut sets. Raise
`HIP_HOPS_LLM.bayes.cpt.MAX_GATE_INPUTS` deliberately if you mean it.

**Exact enumeration.** Sums `2ᵏ` terms over `k` basic events; it refuses above 24
and suggests `engine="pyagrum"`, which uses junction-tree propagation and scales
much further.

**Most probable explanation.** Exhaustive, so it refuses above 22 basic events.
Rank causes with `posteriors()` instead.
