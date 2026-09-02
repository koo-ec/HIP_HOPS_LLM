# 4. Fault tree to Bayesian network

```python
from HIP_HOPS_LLM import AgenticReliabilityStudy, load_example, load_outcomes

study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
study.observe(load_outcomes(), profile={"short": 0.3, "medium": 0.5, "long": 0.2})
study.run()
```

## The tables

```python
cpts = study.cpts("H2")
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

Every node became a variable, in a valid topological order:

```python
cpts.order
# ['CCF_LLM_Qwen_Qwen2_5_Math_1_5B_Instruct', 'BE_aggregator_SELECT',
#  'BE___start___BADREQ', 'BE_cot_agent_NONDET', 'BE_cot_agent_HALLUC', 'E5',
#  'BE_react_agent_NONDET', 'BE_react_agent_HALLUC', 'E3', 'G7', 'G8', 'G6', ...]

cpts.priors
# {'CCF_LLM_…': 0.08, 'BE_aggregator_SELECT': 0.1114, …}
```

Look at one table:

```python
cpts.to_frame("BE-react_agent-HALLUC")
```

Variables can be addressed three ways — by variable name, by fault tree node id,
or by basic-event id — because those are the names that appear in cut sets, in
the FMEA and in evidence dictionaries:

```python
cpts.resolve("BE-react_agent-HALLUC")     # 'BE_react_agent_HALLUC'
cpts["BE-react_agent-HALLUC"] is cpts["BE_react_agent_HALLUC"]   # True
```

## The network

```python
network = study.bayesnet("H2")
print(network.summary())
```

```text
Bayesian network — agentic workflow — H2
  variables      15
  basic events   8
  gates          7 (0 soft, 7 deterministic)
  top event      TOP
  table rows     70
  P(top = Fail)  0.227643
```

### Verify it

```python
network.cross_check()
```

```text
{'exact': 0.2276430…, 'pyagrum': 0.2276430…,
 'difference': 1.1e-16, 'relative_difference': 4.9e-16,
 'agree': 1.0, 'compared': 1.0}
```

Two engines that share no code — pyAgrum's junction tree, and exact enumeration
in NumPy — agreeing to machine precision. Run this whenever you change gate
semantics; it is the cheapest evidence that the conversion is still right.

Because the second engine exists, **pyAgrum is optional**:

```python
network.p_fail(engine="exact")     # NumPy only
```

### Against the cut sets

```python
network.compare_with_cutsets(study.report.analysis("H2"))
```

```text
{'exact_bayesnet': 0.2276, 'minimal_cut_upper_bound': 0.2322,
 'rare_event_sum': 0.2521, 'bound_overestimate': 0.0046}
```

The bound is loose by 0.0046 because those cut sets share basic events —
`CCF-LLM-…` appears in several of them, and MCUB treats cut sets as independent.
Exact inference does not. On a coherent tree the overestimate can never be
negative; if it is, the tree and the network have drifted apart, and the test
suite checks that on every hazard of every bundled example.

## Diagnosis

Condition on what a run showed:

```python
network.posteriors({"BE-aggregator-OWN": "Fail"})
```

```text
BE-aggregator-OWN                          1.0000
BE-aggregator-SELECT                       0.1103
BE-react_agent-HALLUC                      0.0910
CCF-LLM-Qwen-Qwen2-5-Math-1-5B-Instruct    0.0800
BE-react_agent-NONDET                      0.0735
BE-cot_agent-HALLUC                        0.0679
```

States may be written `"Fail"`, `1`, or `True`. Ask about intermediate nodes too:

```python
network.posteriors({"TOP": "Fail"}, basic_events_only=False)
```

The single most likely joint explanation, given the hazard occurred:

```python
network.most_probable_explanation({"TOP": "Fail"})
```

```text
{'CCF-LLM-Qwen-Qwen2-5-Math-1-5B-Instruct': 'OK',
 'BE-aggregator-SELECT': 'OK',
 'BE-__start__-BADREQ': 'OK',
 'BE-cot_agent-NONDET': 'OK',
 'BE-cot_agent-HALLUC': 'OK',
 'BE-react_agent-NONDET': 'OK',
 'BE-react_agent-HALLUC': 'OK',
 'BE-aggregator-OWN': 'Fail'}
```

The most probable single story behind a wrong answer is not the agents failing —
it is the judge hallucinating its own. That is an order-1 cut set on a component
that exists to *improve* reliability, and it is the kind of thing the structural
view alone will not rank for you.

## The imprecise pair

```python
imprecise = study.imprecise_bayesnet("H2")
imprecise.envelope()          # [0.172738, 0.291474]
imprecise.lower.p_fail()      # every leaf at its lower bound
imprecise.upper.p_fail()      # every leaf at its upper bound
```

Per-cause envelopes:

```python
imprecise.posterior_envelopes()
```

```text
BE-aggregator-SELECT     [0.073713, 0.151711]
BE-aggregator-OWN        [0.073713, 0.151711]
BE-react_agent-HALLUC    [0.060891, 0.124137]
BE-react_agent-NONDET    [0.049017, 0.100608]
BE-cot_agent-HALLUC      [0.039574, 0.097997]
```

## Changing the gate logic

The synthesised structure can be right while the logic is not — a two-of-three
majority aggregator being the usual case:

```python
network = study.bayesnet("H2", gate_overrides={"G7": ("KOFN", 2)})
```

Soften every OR to a noisy-OR when propagation is probabilistic rather than
certain:

```python
network = study.bayesnet("H2", soft_gates=True, link_probability=0.85, leak=0.02)
network.cpts.notes
# ['OR gates were softened to noisy-OR: the network no longer encodes the same
#   Boolean function as the minimal cut sets']
```

That warning matters. Once gates are soft, the network and the cut sets no longer
describe the same object, and `compare_with_cutsets` is no longer a bound.

## Drawing

```python
network.show()                              # pyAgrum, or matplotlib
network.view(evidence={"BE-aggregator-OWN": "Fail"}).show()
network.view().to_png("bn_h2.png")
network.view().to_svg("bn_h2.svg")
network.view().to_dot()                     # Graphviz source, no dot binary needed
```

The backend is chosen by whether the Graphviz `dot` binary is genuinely callable
— not by whether `import pydot` succeeds, which proves nothing. Check for
yourself:

```python
from HIP_HOPS_LLM import graphviz_available

graphviz_available()                        # varies by environment
network.view().resolved_backend             # 'pyagrum' or 'matplotlib'
```

The matplotlib backend lays the network out in longest-path layers with a
barycentre pass to reduce edge crossings, shades each node by its posterior,
outlines evidence nodes in green, and says in the caption that Graphviz was
absent. Force either backend with `backend="pyagrum"` or `backend="matplotlib"`.

## Next

[Tutorial 5](05-end-to-end.md) puts the whole thing together and compares two
architectures under the same profile.
