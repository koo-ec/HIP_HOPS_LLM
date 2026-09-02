# Quickstart

Ten minutes, and nothing to install beyond the package. Every example on this
page runs against a bundled architecture and a bundled outcome table, so you
need no LangGraph, no GPU, no API key and no network.

```bash
pip install "HIP-HOPS-LLM[all] @ git+https://github.com/koo-ec/HIP_HOPS_LLM.git"
```

## 1. Analyse an architecture

```python
from HIP_HOPS_LLM import AgenticReliabilityStudy, load_example

study = AgenticReliabilityStudy(
    load_example("parallel_aggregator"),
    name="parallel agents + aggregator",
)
print(study.summary())
```

```text
HiP-HOPS analysis — parallel agents + aggregator
==================================================
components: 5  connections: 5  basic events: 28
The architecture is already acyclic; no loop handling was required.
common-cause groups:
  llm=Qwen/Qwen2.5-Math-1.5B-Instruct: aggregator, cot_agent, react_agent
  runtime=cuda:0: aggregator, cot_agent, react_agent

hazard     sev             P(top)    MCS  SPOF  name
----------------------------------------------------
H1         major           0.0906     21     5  No answer delivered
H2         critical        0.2632     11     3  Incorrect answer delivered and accepted as correct
H3         minor           0.0651      5     1  Malformed answer delivered
H4         minor           0.0975      2     2  Answer too late / budget exhausted
...
NOT CALIBRATED — every probability above is engineering judgement. Call
observe() then calibrate() to replace them with measurement under an
operational profile.
```

Two things to notice before going further.

The **common-cause groups** were not declared anywhere. They were read out of the
node functions' source: both agents and the judge instantiate the same model
snapshot. The consequence shows up in the cut sets.

The **`NOT CALIBRATED` banner** is deliberate. Those probabilities are the
library's placeholders, and a report that does not say so is a report that
invites a reader to treat judgement as evidence.

## 2. Read the cut sets

A *minimal cut set* is a smallest combination of basic events that produces the
hazard. Order 1 means a single point of failure.

```python
for cs in sorted(study.cut_sets("H2"), key=lambda c: (len(c), c))[:8]:
    print(" + ".join(cs))
```

```text
BE-__start__-BADREQ
BE-aggregator-OWN
CCF-LLM-Qwen-Qwen2-5-Math-1-5B-Instruct
BE-aggregator-SELECT + BE-cot_agent-HALLUC
BE-aggregator-SELECT + BE-cot_agent-NONDET
BE-aggregator-SELECT + BE-react_agent-HALLUC
BE-aggregator-SELECT + BE-react_agent-NONDET
BE-cot_agent-HALLUC + BE-react_agent-HALLUC
```

The vote works — most cut sets are order 2, requiring *both* agents to be wrong.
But the third line is order 1. `CCF-LLM-…` is the common-cause event for the
shared model snapshot, and it defeats the redundancy on its own. Two agents that
are the same model are not two agents, for reliability purposes; the architecture
diagram cannot show you that and the cut sets can.

## 3. Add measurement

The bundled table has one row per benchmark item, a stratum label, and a `1`/`0`
correctness column per agent. `1` means the agent answered that item correctly.

```python
from HIP_HOPS_LLM import load_outcomes

outcomes = load_outcomes()
outcomes.head(3)
```

```text
     item_id  stratum  react_agent  cot_agent  aggregator        split
0  item_0000   medium            1          1           1  calibration
1  item_0001   medium            1          1           1  calibration
2  item_0002    short            1          1           1  calibration
```

Now say what workload the system will actually meet, and calibrate:

```python
study.observe(outcomes, profile={"short": 0.30, "medium": 0.50, "long": 0.20})
study.run()
print(study.calibration.summary())
```

```text
Calibration
-----------
  react_agent   n=160   empirical=0.2121  P=[0.1542, 0.2977]  (width 0.1435)
  cot_agent     n=160   empirical=0.1526  P=[0.1021, 0.2405]  (width 0.1384)
  aggregator    n=160   empirical=0.2389  P=[0.1742, 0.3372]  (width 0.1630)
  11 basic event(s) updated from measurement
  8 event(s) kept their placeholder (class L is outside what a correctness
    measurement can speak about; class O is outside what a correctness
    measurement can speak about)
```

Three things happened, and each is worth a sentence.

**`n=160`, not 240.** The table carries a `split` column, and only the
calibration rows were used. Fitting probabilities on the evaluation set would
make every downstream number optimistic and untestable, so the package does it
for you rather than trusting you to remember.

**The estimate is an interval.** `react_agent`'s empirical failure rate is
0.2121, but 160 items do not pin that down; HIP-LLM's hierarchical
imprecise-Bayesian posterior says `[0.1542, 0.2977]`. That width is not noise, it
is the honest statement of what 160 items support.

**Eight events kept their placeholders, and it said which.** A correctness
benchmark measures correctness. It says nothing about latency (`L`) or about the
agent producing no output at all (`O`), so those events were not touched — and
the report names them rather than quietly borrowing the correctness number.

## 4. Get the answer as an interval

```python
study.hazard_probability("H2")
```

```text
[0.172738, 0.291474]
```

That is P(an incorrect answer is delivered and accepted as correct) for one
request drawn from the stated operational profile — computed by exact inference
over the Bayesian network, at both ends of every basic event's interval.

Change the profile and it changes, which is the point:

```python
study.observe(outcomes, profile={"short": 0.05, "medium": 0.15, "long": 0.80})
study.run()
study.hazard_probability("H2")     # a harder workload, a worse interval
```

## 5. Ask why

```python
network = study.bayesnet("H2")

network.p_fail()                                  # exact, a point
network.compare_with_cutsets(study.report.analysis("H2"))
```

```text
{'exact_bayesnet': 0.2262, 'minimal_cut_upper_bound': 0.2307,
 'rare_event_sum': 0.2521, 'bound_overestimate': 0.0045}
```

The cut-set quantification over-estimates by 0.0045, because those cut sets share
basic events and the minimal cut upper bound cannot account for that. The network
can. Together they bracket the answer.

Now condition on something a run actually showed:

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

## 6. Draw it

```python
network.show()
```

pyAgrum renders this where the Graphviz `dot` binary is installed. Where it is
not, the package draws the same network with matplotlib instead, shaded by
posterior, and says so in the caption. `bn.show()` always produces a picture.

Whether Graphviz is there varies by environment — it ships on a current Colab
image and not on many Windows installs — so the package tests for it rather than
assuming: `graphviz_available()` runs `dot -V`, because `import pydot` succeeding
proves nothing.

```python
network.view().to_png("bn_h2.png")     # matplotlib, no Graphviz needed
study.plot("H2")                        # the fault tree
study.plot_architecture()               # the system
```

## 7. Save everything

```python
study.save("artifacts/")
```

Writes the Markdown report, mermaid/DOT/JSON/Open-PSA MEF exports per hazard, the
cut-set and FMEA tables, and the calibration provenance — one CSV saying which
basic event moved from which placeholder to which measured interval.

## Where next

- [Concepts](concepts/index.md) — why the failure classes are what they are, and
  why the probabilities are intervals.
- [Tutorial 7](tutorials/07-your-own-graph.md) — pointing this at your own
  LangGraph application.
- [Tutorial 6](tutorials/06-learned-cpts.md) — fitting a gate's table from data
  instead of assuming it is AND.
