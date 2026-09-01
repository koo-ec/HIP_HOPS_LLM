# HIP-HOPS-LLM

**Hierarchical imprecise reliability assessment and failure propagation for LLM-based agentic systems.**

Give it a LangGraph application and observed benchmark outcomes. Get back
synthesised fault trees, minimal cut sets, an FMEA, and a Bayesian network whose
top-event probability is an *interval* derived from measurement rather than a
number derived from judgement.

```python
from HIP_HOPS_LLM import AgenticReliabilityStudy, load_example, load_outcomes

study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
study.observe(load_outcomes(), profile={"short": 0.3, "medium": 0.5, "long": 0.2})
study.run()
print(study.summary())
study.bayesnet("H2").show()
```

```
hazard     sev             P(top)    MCS  SPOF  name
----------------------------------------------------
H1         major           0.0906     21     5  No answer delivered
H2         critical        0.2632     11     3  Incorrect answer delivered and accepted as correct
H3         minor           0.0651      5     1  Malformed answer delivered
H4         minor           0.0975      2     2  Answer too late / budget exhausted

single points of failure:
  [critical    ] H2  CCF-LLM-Qwen-Qwen2-5-Math-1-5B-Instruct  (aggregator + cot_agent + react_agent)
```

That last line is the point. Two agents answering in parallel with a judge
between them *looks* redundant — most of its cut sets really are order 2. But both
agents run the same model snapshot, so the shared-snapshot common-cause event is
an order-1 cut set for the critical hazard, and the vote buys nothing against it.
The package finds that from the source code of the nodes, without being told.

---

## Why this exists

Two established methods, each weak exactly where the other is strong.

**HiP-HOPS** — Hierarchically Performed Hazard Origin and Propagation Studies —
gives compositional structure. Annotate each component with local failure logic,
then synthesise system-level fault trees, minimal cut sets and an FMEA by
traversing the architecture. Nothing is drawn by hand, so the analysis cannot
drift away from the system. Its basic-event probabilities, though, arrive as
engineering judgement.

**HIP-LLM** — [Aghazadeh-Chakherlou et al., *RESS* 272 (2026) 112615](https://doi.org/10.1016/j.ress.2026.112615) —
gives the numbers. A hierarchical imprecise-Bayesian posterior for the
probability that a language model fails on the *next* item drawn from an explicit
operational profile. It returns an interval, because a few hundred benchmark
items do not identify a point. But it describes one model, not a workflow of
agents, tools and routers.

This package joins them at the leaves: HIP-LLM's per-component posterior interval
becomes the fault tree's basic-event probability, and the tree's top event
becomes an interval you can defend.

```
LangGraph app ──▶ architecture ──▶ fault trees ──▶ cut sets, FMEA
                                        ▲                │
 outcomes + operational profile ────────┘                ▼
        (HIP-LLM imprecise posterior)              CPTs ──▶ Bayesian network
                                                              exact P, diagnosis, plot
```

## Install

```bash
pip install git+https://github.com/koo-ec/HIP_HOPS_LLM.git
```

With everything (pyAgrum for Bayesian networks, LangGraph for live graphs):

```bash
pip install "HIP-HOPS-LLM[all] @ git+https://github.com/koo-ec/HIP_HOPS_LLM.git"
```

Python 3.10+. The core needs only NumPy, pandas, matplotlib, SciPy and PyYAML;
**pyAgrum is optional** — every probability is also computable by exact
enumeration in NumPy alone, which is what the package's own cross-check uses.

## What you get

| | |
|---|---|
| **Architecture extraction** | `graph.get_graph()`, mermaid text, or a dict spec → typed components with ports. Conditional edges become explicit router components; shared model snapshots become common-cause groups. |
| **Loop elimination** | Agent graphs have feedback; fault trees cannot. Loops are unrolled to depth *k* and closed with a feedback-cut component, so the loop's contribution stays in the tree instead of being silently deleted. |
| **Failure annotation** | Six guidewords (`O`, `C`, `VC`, `VS`, `E`, `L`) and an archetype library: LLM agent, tool, router, aggregator, transform, boundary. Keeping `VC` (coarse, detectable) apart from `VS` (subtle, plausible — the hallucination case) is the single most important modelling decision. |
| **Fault tree synthesis** | Backward traversal from a hazard, substituting local logic, memoised so shared sub-trees stay shared. |
| **Analysis** | Minimal cut sets (MOCUS with absorption), MCUB quantification, Birnbaum and Fussell–Vesely importance, single points of failure, generated FMEA. |
| **Operational profiles** | The mix of work the system will actually meet. Every estimate is conditional on it, and it is never inferred silently. |
| **Imprecise calibration** | HIP-LLM's posterior interval per component, written onto the tree's leaves with recorded provenance — and a report of every component it could *not* match. |
| **CPTs and Bayesian networks** | Deterministic AND/OR/*k*-of-*n* gates, noisy-OR with leak, or tables **fitted from observed outcomes**. Exact inference, posterior over causes given evidence, lower/upper network pair for the imprecise case. |
| **Drawing** | pyAgrum where Graphviz is available, matplotlib where it is not — so `bn.show()` always produces a picture. Fault trees, architectures, importance and cut-set plots too. |
| **Export** | Mermaid, Graphviz DOT, JSON, **Open-PSA MEF XML**, Markdown report, cut-set and FMEA CSVs. |

## The whole API in ten lines

```python
from HIP_HOPS_LLM import AgenticReliabilityStudy

study = AgenticReliabilityStudy(graph, globals_ns=globals())   # 1. your LangGraph
study.observe(outcomes_table, profile={"short": .3, "long": .7})  # 2. what you measured
study.run()                                                     # 3. infer + calibrate

print(study.summary())                     # cut sets, SPOFs, calibration provenance
study.hazard_probability("H2")             # → [0.1727, 0.2915]
study.bayesnet("H2").posteriors({"BE-coder-EXECERR": "Fail"})   # diagnosis
study.bayesnet("H2").show()                # the picture
study.save("artifacts/")                   # every export
```

HIP-LLM's own API is re-exported unchanged, so it needs one import line, not
three:

```python
from HIP_HOPS_LLM import OperationalFailureProb, quick_inference_settings

estimator = OperationalFailureProb(
    profile={"short": 0.30, "long": 0.70},
    settings=quick_inference_settings(samples=1500, configurations=48),
)
result = estimator.fit(outcomes=[1, 1, 0, 1, 0, 0, 1, 0],
                       strata=["short"] * 4 + ["long"] * 4)
print(result.summary())
```

A test asserts that *every* public symbol of `HIPLLM` and `hip_llm` is reachable
this way, so the promise stays true.

## Documentation

| | |
|---|---|
| [Quickstart](docs/source/quickstart.md) | Ten minutes, no LangGraph needed |
| [Concepts](docs/source/concepts/) | HiP-HOPS for agents · imprecise probability · fault trees as Bayesian networks |
| [Tutorials](docs/source/tutorials/) | Six worked examples, from one graph to a calibrated network |
| [API reference](docs/source/api/) | Every public class and function |
| [Vendoring](docs/source/vendoring.md) | What HIP-LLM code is included, from which commit, under which licence |

Runnable scripts are in [`examples/`](examples/) and notebooks in
[`notebooks/`](notebooks/), including a Colab notebook that clones this
repository and runs the whole pipeline.

## Guarantees the tests enforce

These are invariants, not aspirations — each is a test that fails loudly.

- **Exact inference agrees with pyAgrum** to machine precision (`< 1e-12`). Two
  independent construction paths, cross-checked; if they diverge, one is wrong.
- **The minimal cut upper bound never under-estimates.** `MCUB ≥ exact` on every
  hazard of every bundled example, and strictly greater where cut sets share
  events — which is the entire reason for building the network.
- **Cut sets are minimal and sufficient.** Every cut set is verified to make the
  Boolean function true, and every proper subset of it to make it false.
- **Simplification preserves the Boolean function.** Reduced and unreduced trees
  give identical cut sets.
- **The top event is monotone in every basic event.** This is what licenses
  building lower and upper networks from interval-valued leaves.
- **Union-splitting is exact.** After calibration, `1 − ∏(1 − pᵢ)` over a
  component's basic events reproduces its measured probability to `1e-9`.
- **Nothing is calibrated silently.** Uncalibrated components are named, latency
  events keep their placeholders when only correctness was measured, and a
  mistyped override id raises instead of being ignored.
- **CPTs are never fitted on a test split.** `learn_cpt` raises.

```bash
pytest
```

## Two findings baked in as warnings

Both come from analysing the source notebook this package generalises, and
neither is visible in its output:

1. **The parallel architecture's redundancy is architectural only.** `MODEL_FAST`
   and `MODEL_DEEP` are the same snapshot, so the common-cause event is an
   order-1 cut set for the wrong-answer hazard. The package reports shared
   resources as common-cause groups and lists the resulting single points of
   failure.
2. **A network running on default priors looks exactly like a network running on
   evidence.** Every basic event carries an `evidence` string; uncalibrated ones
   say `placeholder`, `study.summary()` prints `NOT CALIBRATED` in capitals, and
   the calibration report names every component no measurement reached.

## Citing

If you use this package, please cite both the HIP-LLM paper and this repository;
see [`CITATION.cff`](CITATION.cff).

## Licence

MIT — see [`LICENSE`](LICENSE). The vendored HIP-LLM sources under `src/HIPLLM/`
and `src/hip_llm/` are also MIT and retain their own notice in
[`LICENSE.HIPLLM`](LICENSE.HIPLLM); their provenance is recorded in
[`docs/source/vendoring.md`](docs/source/vendoring.md).
