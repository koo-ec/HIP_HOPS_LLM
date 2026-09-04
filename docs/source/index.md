# HIP-HOPS-LLM

**Hierarchical imprecise reliability assessment and failure propagation for LLM-based agentic systems.**

Give it a LangGraph application and observed benchmark outcomes. Get back
synthesised fault trees, minimal cut sets, an FMEA, and a Bayesian network whose
top-event probability is an *interval* derived from measurement rather than a
number derived from judgement.

```{container} hero-actions
[Quickstart](quickstart.md)
[Explore the concepts](concepts/index.md)
[API reference](api/index.md)
```

## From system design to safety analysis

Connect a hierarchical system design to its failure behaviour and a Bayesian
network. Analyse how failures in AI agents and non-AI components propagate to
system-level hazards, then use observed outcomes and an operational profile to
calibrate the probabilities.

```{figure} _static/system-safety-workflow.png
:alt: Three levels of system design, each containing agentic AI and non-AI components, connect to safety analysis using functional failure analysis, interface-focused FMEAs and local component failure behaviour. A synthesised Bayesian network links component failures to level failures and system safety, with basic-event probabilities derived from operational-profile reliability.
:figclass: workflow-figure
:target: _static/system-safety-workflow.png

The main idea: system architecture and failure analysis provide the structure;
operational-profile reliability provides the evidence for Bayesian inference.
FFA denotes Functional Failure Analysis; IF-FMEAs denotes Interface Focused
FMEAs; IF-FMEAs\* denotes local component failure behaviour. The percentages in
this conceptual illustration are illustrative, not results produced by the package.
Select the diagram to view it at full size.
```

## Two methods, joined at the leaves

**HiP-HOPS** — Hierarchically Performed Hazard Origin and Propagation Studies —
gives compositional *structure*: annotate each component with local failure
logic, then synthesise system-level fault trees, minimal cut sets and an FMEA by
traversing the architecture. Its basic-event probabilities arrive as engineering
judgement.

**HIP-LLM** — [Aghazadeh-Chakherlou et al., *RESS* 272 (2026) 112615](https://doi.org/10.1016/j.ress.2026.112615) —
gives the *numbers*: a hierarchical imprecise-Bayesian posterior for the
probability that a language model fails on the next item drawn from an explicit
operational profile. It returns an interval, because a few hundred benchmark
items do not identify a point. But it describes one model, not a workflow.

This package runs the second per component and writes its intervals onto the
first's leaves.

```text
LangGraph app ──▶ architecture ──▶ fault trees ──▶ cut sets, FMEA
                                        ▲                │
 outcomes + operational profile ────────┘                ▼
        (HIP-LLM imprecise posterior)              CPTs ──▶ Bayesian network
                                                              exact P, diagnosis, plot
```

## Try the complete workflow

Start with a bundled agent architecture and observed outcomes. Run the analysis,
inspect the summary, and display the Bayesian network for a hazard:

```python
from hiphopsllm import AgenticReliabilityStudy, load_example, load_outcomes

study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
study.observe(load_outcomes(), profile={"short": 0.3, "medium": 0.5, "long": 0.2})
study.run()
print(study.summary())
study.bayesnet("H2").show()
```

See the [quickstart](quickstart.md) for a walkthrough, or follow the
[installation guide](install.md) to set up the package.

## Where to start

| If you are… | Go to |
|---|---|
| New here | [**Quickstart**](quickstart.md) — the whole pipeline in ten minutes, no LangGraph or GPU required. |
| After the ideas first | [**Concepts**](concepts/index.md) — what HiP-HOPS does for agent graphs, why the probabilities are intervals, and why a fault tree *is* a Bayesian network. |
| Learning by doing | [**Tutorials**](tutorials/index.md) — eight worked examples, each one runnable. |
| Looking something up | [**API reference**](api/index.md) — every public class and function. |
| Going to change the code | [**Development**](development/index.md) — the codebase map, the extension points, and the tests you must not weaken. |

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

quickstart
install
```

```{toctree}
:maxdepth: 2
:caption: Concepts
:hidden:

concepts/index
concepts/hiphops
concepts/failure-classes
concepts/imprecise
concepts/faulttree-to-bn
concepts/operational-profiles
```

```{toctree}
:maxdepth: 2
:caption: Tutorials
:hidden:

tutorials/index
tutorials/01-langgraph-to-fault-tree
tutorials/02-cut-sets-and-fmea
tutorials/03-operational-profile
tutorials/04-fault-tree-to-bayesnet
tutorials/05-end-to-end
tutorials/06-learned-cpts
tutorials/07-your-own-graph
tutorials/08-n8n-workflows
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
vendoring
faq
changelog
```

```{toctree}
:maxdepth: 2
:caption: Development
:hidden:

development/index
development/codebase
development/extending
development/testing
development/notebooks
development/ci-and-releasing
```

## References

- Aghazadeh-Chakherlou, R., Guo, Q., Khastgir, S., Popov, P., Zhang, X., &
  Zhao, X. (2026). A hierarchical imprecise probability approach to reliability
  assessment of large language models. *Reliability Engineering & System Safety*,
  *272*, 112615. <https://doi.org/10.1016/j.ress.2026.112615>
- Custers, B., & Aslansefat, K. (2026). Runtime uncertainty monitoring for
  LLM-based multi-agent systems using Bayesian networks. In *Computer Safety,
  Reliability, and Security: SAFECOMP 2026 Workshops, 9th International Workshop
  on Artificial Intelligence Safety Engineering (WAISE 2026)*, Valencia, Spain.
  Springer. (in press)
- Donaldson, L., Walker, C., Aslansefat, K., & Papadopoulos, Y. (2026). Bayesian
  uncertainty propagation for agentic RAG pipelines: A proof-of-concept study on
  multi-hop question answering. In *Proceedings of the 7th International
  Conference on Maintenance and Intelligent Asset Management (ICMIAM 2026)*,
  Huddersfield, UK, 1-3 September 2026. Springer Nature.
- Papadopoulos, Y., & McDermid, J. A. (1999). Hierarchically performed hazard
  origin and propagation studies. In *Computer Safety, Reliability and Security
  (SAFECOMP 1999)* (Lecture Notes in Computer Science, Vol. 1698, pp. 139-152).
  Springer. <https://doi.org/10.1007/3-540-48249-0_13>

## Licence and citation

MIT. If you use this package please cite both the HIP-LLM paper and this
repository — see `CITATION.cff` in the repository root.
