# API reference

Every public class and function, grouped by layer. The top-level package
re-exports all of it, so `from HIP_HOPS_LLM import X` works for anything listed
here.

| Layer | What it does |
|---|---|
| [`pipeline`](pipeline.md) | `AgenticReliabilityStudy` — the whole chain behind one object |
| [`architecture`](architecture.md) | LangGraph → components, ports, connections; loop elimination |
| [`faulttree`](faulttree.md) | Failure logic, synthesis, cut sets, quantification, FMEA, exports |
| [`reliability`](reliability.md) | Operational profiles, HIP-LLM, evidence calibration |
| [`bayes`](bayes.md) | CPTs, Bayesian networks, learned tables, drawing |
| [`viz`](viz.md) | Matplotlib rendering of trees, architectures and rankings |
| [`report`](report.md) | `SafetyReport` and `analyse_langgraph` |
| [`io`](io.md) | Bundled example architectures and outcome data |

## The 30-second version

```python
from HIP_HOPS_LLM import AgenticReliabilityStudy

study = AgenticReliabilityStudy(graph, globals_ns=globals())
study.observe(outcomes, profile={"short": 0.3, "long": 0.7}).run()

study.summary()                  # everything, printable
study.hazard_probability("H2")   # an Envelope
study.bayesnet("H2")             # a BayesianNetwork
study.save("artifacts/")
```

```{toctree}
:hidden:

pipeline
architecture
faulttree
reliability
bayes
viz
report
io
```
