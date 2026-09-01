# Concepts

Five ideas, each of which changes what you do with the library.

| | |
|---|---|
| [**HiP-HOPS for agent graphs**](hiphops.md) | Why the analysis is *synthesised* from the architecture rather than drawn, and what happens to feedback loops. |
| [**Failure classes**](failure-classes.md) | Six guidewords, and why keeping `VC` apart from `VS` is the single most important modelling decision here. |
| [**Operational profiles**](operational-profiles.md) | A benchmark accuracy is a claim about the benchmark. A reliability claim needs the workload. |
| [**Imprecise probability**](imprecise.md) | Why a few hundred items give an interval, not a number, and what you may legitimately do with one. |
| [**Fault trees as Bayesian networks**](faulttree-to-bn.md) | The conversion, the exact answer it buys, and where the cut-set bound is loose. |

## The shape of the whole thing

```text
   your LangGraph app
          │
          │  architecture/            read components, ports, connections;
          ▼                           materialise routers; unroll loops
   SystemModel  (acyclic)
          │
          │  faulttree/failure.py     attach local failure logic per archetype
          ▼
   FailureModel  ──────────────────── basic events, placeholder probabilities
          │                                        ▲
          │  faulttree/synthesis.py                │  reliability/calibration.py
          ▼                                        │
   FaultTree per hazard              measured intervals from HIP-LLM,
          │                          conditional on an operational profile
          ├──▶ analysis.py ─────────▶ minimal cut sets, MCUB, importance, FMEA
          │
          │  bayes/cpt.py
          ▼
   CPTSet  ─── bayes/network.py ────▶ BayesianNetwork
                                          exact P(top), posterior over causes,
                                          lower/upper pair, drawing
```

Each arrow is a public function, so you can enter or leave the pipeline
anywhere. `AgenticReliabilityStudy` is a convenience over the whole chain, not a
gate in front of it.

```{toctree}
:hidden:

hiphops
failure-classes
operational-profiles
imprecise
faulttree-to-bn
```
