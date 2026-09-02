"""Generate the repository's notebooks from Python sources.

Notebooks are hard to review and easy to break: a diff of a ``.ipynb`` is mostly
JSON noise, and stale outputs baked into one are indistinguishable from fresh
ones. So they are *generated* here from cell lists written in plain Python, with
no outputs, and regenerated whenever they change.

    python scripts/build_notebooks.py

Writes:
    notebooks/HIP_HOPS_LLM_Colab.ipynb        clone, install, run, plot
    notebooks/hip_hops_for_agentic_ai.ipynb   the Kaggle notebook, using the package
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

REPO = "https://github.com/koo-ec/HIP_HOPS_LLM.git"


_COUNTER = {"n": 0}


def _next_id() -> str:
    """Stable, deterministic cell ids so regenerating produces no spurious diff."""
    _COUNTER["n"] += 1
    return f"cell{_COUNTER['n']:03d}"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": text.strip().splitlines(True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def notebook(cells: list[dict], title: str) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True, "name": title},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #
# 1. The Colab notebook
# --------------------------------------------------------------------------- #
COLAB = [
    md(
        f"""
# HIP-HOPS-LLM on Colab

**Hierarchical imprecise reliability assessment and failure propagation for
LLM-based agentic systems.**

This notebook clones the repository, installs the package, and runs the whole
pipeline: a LangGraph architecture in, fault trees and a calibrated Bayesian
network out. It needs no GPU, no API key and no data of your own — everything
runs against bundled examples.

Runtime: about two minutes, most of it the install.

Repository: {REPO.removesuffix('.git')}
"""
    ),
    md("---\n## 1. Install"),
    code(
        f"""
!git clone --depth=1 {REPO} 2>&1 | tail -2
%pip install -q -e "HIP_HOPS_LLM[bayes,graph]" 2>&1 | tail -3
print("installed")
"""
    ),
    code(
        """
import importlib, platform, sys

print("python  ", sys.version.split()[0], "on", platform.system())
for name in ("numpy", "pandas", "matplotlib", "scipy", "pyagrum", "langgraph"):
    try:
        module = importlib.import_module(name)
        print(f"{name:<12} {getattr(module, '__version__', 'present')}")
    except ImportError:
        print(f"{name:<12} not installed (optional here)")

import HIP_HOPS_LLM as H

print(f"\\nHIP-HOPS-LLM {H.__version__} — {len(H.__all__)} public names")
print("Graphviz available:", H.graphviz_available())
"""
    ),
    md(
        """
That last line matters for the pictures further down. Graphviz is a native
binary, not a Python package, and whether a Colab image has it varies — so the
package tests for it (`dot -V`) rather than assuming. With it, pyAgrum renders
the Bayesian networks; without it, they are drawn with matplotlib and the caption
says so. Either way `bn.show()` produces a picture. `!apt-get -qq install
graphviz` forces the pyAgrum path.
"""
    ),
    md(
        """
---
## 2. The whole pipeline, in ten lines

A LangGraph architecture, observed benchmark outcomes, an operational profile —
and a calibrated Bayesian network at the end.
"""
    ),
    code(
        """
from HIP_HOPS_LLM import AgenticReliabilityStudy, load_example, load_outcomes

study = AgenticReliabilityStudy(
    load_example("parallel_aggregator"),
    name="parallel agents + aggregator",
)
study.observe(load_outcomes(), profile={"short": 0.30, "medium": 0.50, "long": 0.20})
study.run()

print(study.summary())
"""
    ),
    md(
        """
Two things in that output are the point of the whole package.

**The common-cause group was not declared anywhere.** It was read out of the node
functions' source: both agents and the judge instantiate the same model snapshot.

**`CCF-LLM-…` is an order-1 cut set for the critical hazard.** Two agents
answering in parallel with a judge between them *looks* redundant, and most of
its cut sets really are order 2 — but a fault they share defeats the vote on its
own. An architecture diagram cannot show you that.
"""
    ),
    md("---\n## 3. What the architecture looks like"),
    code(
        """
import pandas as pd

pd.DataFrame(study.system.architecture_table()).set_index("component")
"""
    ),
    code(
        """
import matplotlib.pyplot as plt

study.plot_architecture()
plt.show()
"""
    ),
    md("---\n## 4. The fault tree"),
    code(
        """
ax = study.plot("H2")
ax.figure.set_size_inches(14, 9)
plt.show()
"""
    ),
    code(
        """
print("minimal cut sets for H2 — incorrect answer delivered and accepted:\\n")
for cut in sorted(study.cut_sets("H2"), key=lambda c: (len(c), c)):
    print(f"  order {len(cut)}:  " + " + ".join(cut))
"""
    ),
    md("---\n## 5. What was measured, and what it licenses"),
    code(
        """
study.calibration.evidence_frame()
"""
    ),
    code(
        """
print("per-stratum failure rates — this is why a pooled accuracy is not enough:\\n")
for name, evidence in study.evidence.items():
    rates = ",  ".join(f"{k}={v:.3f}" for k, v in evidence.by_stratum.items())
    print(f"  {name:<14} {rates}")
"""
    ),
    code(
        """
study.calibration.to_frame()
"""
    ),
    md(
        """
`n = 160`, not 240: the outcome table carries a `split` column, and only the
calibration rows were used. Fitting basic-event probabilities on the evaluation
set would make every downstream number optimistic and untestable.

And the estimate is an **interval**. 160 items do not identify a point, and
HIP-LLM's hierarchical imprecise-Bayesian posterior says so rather than
pretending otherwise.
"""
    ),
    md("---\n## 6. The Bayesian network"),
    code(
        """
network = study.bayesnet("H2")
print(network.summary())

print("\\ntwo engines that share no code, cross-checked:")
for key, value in network.cross_check().items():
    print(f"  {key:<22} {value}")
"""
    ),
    code(
        """
print("exact inference vs the minimal cut upper bound:\\n")
for key, value in network.compare_with_cutsets(study.report.analysis("H2")).items():
    print(f"  {key:<28} {value:.6f}")
print("\\nThe bound is loose because those cut sets share basic events.")
print("Together they bracket the answer.")
"""
    ),
    code(
        """
network.show()
"""
    ),
    md("---\n## 7. Diagnosis: why did it fail?"),
    code(
        """
posterior = network.posteriors({"BE-aggregator-OWN": "Fail"})
pd.Series(posterior, name="P(cause | the judge hallucinated)").head(8)
"""
    ),
    code(
        """
network.view(evidence={"BE-aggregator-OWN": "Fail"}).show()
"""
    ),
    md("---\n## 8. The answer, as an interval"),
    code(
        """
imprecise = study.imprecise_bayesnet("H2")
print(imprecise.summary())

print("\\nP(H2) =", study.hazard_probability("H2"))
"""
    ),
    md(
        """
That is P(an incorrect answer is delivered and accepted as correct) for one
request drawn from the stated operational profile — computed by exact inference
at both ends of every basic event's interval.

Change the profile and it changes, which is exactly the point: a failure
probability quoted without the workload it is conditional on is not a claim about
the system at all.
"""
    ),
    code(
        """
for name, profile in {
    "as measured": {"short": 0.30, "medium": 0.50, "long": 0.20},
    "harder":      {"short": 0.05, "medium": 0.15, "long": 0.80},
    "easier":      {"short": 0.70, "medium": 0.25, "long": 0.05},
}.items():
    other = AgenticReliabilityStudy(load_example("parallel_aggregator"))
    other.observe(load_outcomes(), profile=profile).run()
    print(f"  {name:<12} P(H2) in {other.hazard_probability('H2')}")
"""
    ),
    md("---\n## 9. Does diversifying the models help?"),
    code(
        """
cases = {
    "as built (shared snapshot)": {},
    "cot_agent diversified": {
        "cot_agent": {"llm": "gpt-4o-mini", "runtime": "api"},
    },
    "agents and judge diversified": {
        "cot_agent": {"llm": "gpt-4o-mini", "runtime": "api"},
        "aggregator": {"llm": "claude-sonnet-4-5", "runtime": "api2"},
    },
}
for name, overrides in cases.items():
    s = AgenticReliabilityStudy(
        load_example("parallel_aggregator"), name=name, resource_overrides=overrides
    )
    s.analyse()
    analysis = s.report.analysis("H2")
    order_1 = sorted(next(iter(c)) for c in analysis.cuts.sets if len(c) == 1)
    print(f"{name:<30} P(H2)={analysis.quant.top_probability:.4f}")
    print(f"{'':<30} order-1: {order_1}\\n")
"""
    ),
    md(
        """
**Diversifying one agent changes nothing.** The common-cause group shrinks from
three members to two, but the judge is still one of them and the judge is on the
value path alone — so `CCF-LLM-…` remains an order-1 cut set. You have to
diversify the judge too, and then `P(H2)` drops by about a fifth.
"""
    ),
    md("---\n## 10. The FMEA, and everything saved"),
    code(
        """
study.fmea().head(10)
"""
    ),
    code(
        """
for path in study.save("artifacts"):
    print(path)
"""
    ),
    md("---\n## 11. Your own LangGraph"),
    code(
        """
# In your own notebook, replace the bundled example with your compiled graph and
# pass globals() so the extractor can interrogate the live model objects — which
# is what makes shared-snapshot (common-cause) detection reliable.
#
# study = AgenticReliabilityStudy(
#     graph,
#     name="my workflow",
#     globals_ns=globals(),
#     node_functions={"planner::router": route_fn},   # add_conditional_edges callables
# )
# study.analyse()                    # structure — needs no data at all
# print(study.summary())
#
# study.observe(my_outcomes, profile=my_profile).run()
# study.hazard_probability("H2")

from HIP_HOPS_LLM import describe_examples

print(describe_examples())
"""
    ),
    md(
        """
---
## Where next

* **Docs** — https://github.com/koo-ec/HIP_HOPS_LLM/tree/main/docs/source
* **Tutorial 7** covers pointing this at a real LangGraph application, including
  what to log and how much of it you need.
* **HIP-LLM's own API** is re-exported here, so
  `from HIP_HOPS_LLM import OperationalFailureProb` works.

If you use this, please cite both the HIP-LLM paper
([10.1016/j.ress.2026.112615](https://doi.org/10.1016/j.ress.2026.112615)) and
this repository — `CITATION.cff` has both.
"""
    ),
]


# --------------------------------------------------------------------------- #
# 2. The Kaggle notebook, rewritten against the package
# --------------------------------------------------------------------------- #
KAGGLE = [
    md(
        """
# HiP-HOPS for Agentic AI

**Static fault tree synthesis for LangGraph multi-agent systems, and the
Bayesian networks they become.**

This is the notebook that became
[HIP-HOPS-LLM](https://github.com/koo-ec/HIP_HOPS_LLM). The original wrote a
4,800-line module into a single cell; this version installs the package instead,
so every class here is tested, documented and importable in your own code.

What it does, in the HiP-HOPS phases:

1. **Model** the architecture — read components, ports and connections out of
   `graph.get_graph()`, materialising conditional edges as router components.
2. **Eliminate loops** — agent graphs have feedback, fault trees cannot; loops are
   unrolled and closed with a feedback-cut component rather than deleted.
3. **Annotate** every component with local failure logic from an archetype
   library.
4. **Synthesise** fault trees by backward traversal, never by hand.
5. **Analyse** — minimal cut sets, importance, single points of failure, FMEA.
6. **Convert** to a Bayesian network — exact probability, and diagnosis.
"""
    ),
    md("---\n## 0. Install"),
    code(
        f"""
!git clone --depth=1 {REPO} 2>&1 | tail -2
%pip install -q -e "HIP_HOPS_LLM[bayes,graph]" 2>&1 | tail -3

import importlib, os, platform, sys

print("python  ", sys.version.split()[0], "on", platform.system())
for name in ("matplotlib", "pandas", "numpy", "langgraph", "pyagrum"):
    try:
        module = importlib.import_module(name)
        print(f"{{name:<12}} {{getattr(module, '__version__', 'present')}}")
    except ImportError:
        print(f"{{name:<12}} not installed (optional here)")

ON_KAGGLE = os.path.isdir("/kaggle/working")
OUTPUT_DIR = "/kaggle/working/fault_trees" if ON_KAGGLE else "fault_trees"
print("\\noutput directory:", OUTPUT_DIR)
"""
    ),
    code(
        """
import matplotlib.pyplot as plt
import pandas as pd

import HIP_HOPS_LLM as hh
from HIP_HOPS_LLM import (
    AgenticReliabilityStudy,
    LangGraphExtractor,
    Role,
    SystemModel,
    extract_architecture,
    find_cycles,
    plot_architecture,
    plot_cutset_orders,
    plot_fault_tree,
    plot_importance,
)

print("HIP-HOPS-LLM", hh.__version__)
"""
    ),
    md(
        """
---
## 1. The two architectures

The node bodies below are never called — only their **source** is read, to
classify roles and detect shared model snapshots. That is the whole input to the
analysis, along with the graph topology.
"""
    ),
    code(
        '''
import re


# ---- Approach 1: ReAct + calculator ----------------------------------------
def generator(state):
    """ReAct step: generate up to 'Observation'."""
    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"          # MODEL_FAST
    state["prompt"] = state["prompt"] + "\\n" + state["output"]
    messages = [{"role": "user", "content": state["prompt"]}]
    outputs = model.generate(messages, max_new_tokens=1000, do_sample=True,
                             temperature=0.3, stop_strings="Observation")
    return {"output": tokenizer.decode(outputs[0]), "prompt": state["prompt"]}


def router(state):
    """Conditional edge: choose the tool, finish, or fail."""
    tool_match = re.search(r"Action: (.*?)(?=Action Input)", state["output"], re.DOTALL)
    tool = tool_match.group(1).strip() if tool_match else "no_tool"
    if "calculator" in tool:
        return "coder"
    if re.search(r"Final Answer", state["output"], re.DOTALL):
        return "end"
    return "error"


def coder(state):
    """Calculator tool: evaluate the extracted expression."""
    match = re.search(r"Action Input: (.*?)(?=Observation)", state["output"], re.DOTALL)
    text = match.group(1).strip().strip(chr(34)) if match else "FAILED"
    try:
        result = eval(text)
    except Exception:
        result = "not something a simple calculator can execute"
    return {"tool_output": str(result)}


# ---- Approach 2: parallel agents + aggregator -------------------------------
def react_agent(state):
    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"          # MODEL_FAST
    return {"output_1": model.generate(state.get("problem", ""), temperature=0.3)}


def cot_agent(state):
    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"          # MODEL_DEEP - the SAME snapshot
    return {"output_2": model_deep.generate("solve step by step: " + state["problem"],
                                            temperature=0.6)}


def aggregator(state):
    """Judge: select the better of the two candidate answers."""
    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    prompt = "Select the better answer:\\n" + state["output_1"] + "\\n" + state["output_2"]
    return {"combined_output": judge.generate(prompt)}


NODE_FUNCS_1 = {"generator": generator, "coder": coder, "generator::router": router}
NODE_FUNCS_2 = {"react_agent": react_agent, "cot_agent": cot_agent,
                "aggregator": aggregator}
print("node functions defined (never executed)")
'''
    ),
    code(
        '''
MERMAID_1 = """
graph TD;
    __start__([__start__]):::first
    generator(generator)
    coder(coder)
    __end__([__end__]):::last
    __start__ --> generator;
    coder --> generator;
    generator -. &nbsp;coder&nbsp; .-> coder;
    generator -. &nbsp;error&nbsp; .-> __end__;
    generator -. &nbsp;end&nbsp; .-> __end__;
"""

MERMAID_2 = """
graph TD;
    __start__([__start__]):::first
    react_agent(react_agent)
    cot_agent(cot_agent)
    aggregator(aggregator)
    __end__([__end__]):::last
    __start__ --> react_agent;
    __start__ --> cot_agent;
    react_agent --> aggregator;
    cot_agent --> aggregator;
    aggregator --> __end__;
"""

try:
    from typing_extensions import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State1(TypedDict, total=False):
        prompt: str
        output: str
        tool_output: str

    builder = StateGraph(State1)
    builder.add_node("generator", generator)
    builder.add_node("coder", coder)
    builder.add_edge(START, "generator")
    builder.add_edge("coder", "generator")
    builder.add_conditional_edges("generator", router,
                                  {"coder": "coder", "error": END, "end": END})
    graph_1 = builder.compile()

    class State2(TypedDict, total=False):
        problem: str
        output_1: str
        output_2: str
        combined_output: str

    builder = StateGraph(State2)
    builder.add_node("react_agent", react_agent)
    builder.add_node("cot_agent", cot_agent)
    builder.add_node("aggregator", aggregator)
    builder.add_edge(START, "react_agent")
    builder.add_edge(START, "cot_agent")
    builder.add_edge("react_agent", "aggregator")
    builder.add_edge("cot_agent", "aggregator")
    builder.add_edge("aggregator", END)
    graph_2 = builder.compile()

    SOURCE_1, SOURCE_2 = graph_1, graph_2
    print("using live LangGraph objects (graph.get_graph())")
except Exception as exc:
    SOURCE_1, SOURCE_2 = MERMAID_1, MERMAID_2
    print(f"langgraph unavailable ({type(exc).__name__}) - using recorded mermaid")
'''
    ),
    md(
        """
---
## 2. Extraction

Nodes become components with ports; edges become connections;
`add_conditional_edges` becomes an explicit router component with failure logic
of its own.
"""
    ),
    code(
        """
extractor = LangGraphExtractor(globals_ns=globals())

arch_1 = extractor.with_roles().extract(SOURCE_1, name="Approach 1")
arch_2 = extractor.extract(SOURCE_2, name="Approach 2")

display(pd.DataFrame(arch_1.architecture_table()).set_index("component"))
display(pd.DataFrame(arch_2.architecture_table()).set_index("component"))

print("Approach 1 cycles:", find_cycles(arch_1) or "none")
print("Approach 2 cycles:", find_cycles(arch_2) or "none")
print("\\nApproach 2 shared resources (common-cause candidates):")
for (kind, value), members in arch_2.common_cause_groups().items():
    print(f"  {kind} = {value}\\n    -> {', '.join(members)}")
"""
    ),
    md(
        """
`MODEL_FAST` and `MODEL_DEEP` are the same snapshot. The two agents therefore
share a common cause, and the "redundancy" of Approach 2 is architectural only —
which the cut sets will show in a moment.
"""
    ),
    code(
        """
plot_architecture(arch_1, title="Approach 1 - ReAct + calculator")
plt.show()
plot_architecture(arch_2, title="Approach 2 - parallel agents + aggregator")
plt.show()
"""
    ),
    md("---\n## 3. Approach 1: synthesis and analysis"),
    code(
        """
study_1 = AgenticReliabilityStudy(
    SOURCE_1,
    name="Approach 1 - ReAct + calculator",
    node_functions=NODE_FUNCS_1,
    globals_ns=globals(),
    unroll=1,
)
study_1.analyse()
print(study_1.summary())
"""
    ),
    code(
        """
ax = plot_fault_tree(study_1.report.tree("H2"))
ax.figure.set_size_inches(14, 9)
plt.show()
"""
    ),
    code(
        """
analysis = study_1.report.analysis("H2")
rows = [
    {"order": len(cut), "events": " + ".join(sorted(cut)),
     "P": analysis.quant.cut_set_probability.get(cut, float("nan"))}
    for cut in sorted(analysis.cuts.sets, key=lambda c: (len(c), sorted(c)))
]
display(pd.DataFrame(rows) if rows else "no cut sets: nothing here can cause H2")
print(f"P(H2) = {analysis.quant.top_probability:.4f}")
"""
    ),
    md(
        """
Every cut set is of order 1. There is no redundancy anywhere in this
architecture: any single fault reaches the system boundary.
"""
    ),
    md("---\n## 4. Approach 2: does the vote help?"),
    code(
        """
study_2 = AgenticReliabilityStudy(
    SOURCE_2,
    name="Approach 2 - parallel agents + aggregator",
    node_functions=NODE_FUNCS_2,
    globals_ns=globals(),
)
study_2.analyse()
print(study_2.summary())
"""
    ),
    code(
        """
ax = plot_fault_tree(study_2.report.tree("H2"))
ax.figure.set_size_inches(14, 9)
plt.show()
"""
    ),
    code(
        """
comparison = []
for name, study in (("Approach 1", study_1), ("Approach 2", study_2)):
    for hid in sorted(study.report.trees):
        a = study.report.analysis(hid)
        # A hazard can legitimately have no cut sets at all: nothing in this
        # architecture can cause it. That is a result, not a missing value.
        orders = [len(c) for c in a.cuts.sets]
        comparison.append({
            "approach": name, "hazard": hid,
            "P(top)": round(a.quant.top_probability, 4),
            "cut sets": len(orders),
            "min order": min(orders) if orders else None,
            "SPOFs": len(a.single_points),
        })
display(pd.DataFrame(comparison))

study_2.plot_cutset_orders()
plt.show()
"""
    ),
    md(
        """
The vote does work: most cut sets for H2 are of order 2 — **both** agents must be
wrong. But the shared model snapshot is still an order-1 cut set, so the
redundancy does not deliver the independence the diagram appears to promise.
"""
    ),
    code(
        """
study_2.plot_importance("H2")
plt.show()
"""
    ),
    md("---\n## 5. FMEA and single points of failure"),
    code(
        """
display(study_2.fmea())
display(pd.DataFrame(study_2.single_points()))
"""
    ),
    md("---\n## 6. The Bayesian network"),
    code(
        """
network = study_2.bayesnet("H2")
print(network.summary())

print("\\nexact vs the cut-set bound:")
for key, value in network.compare_with_cutsets(study_2.report.analysis("H2")).items():
    print(f"  {key:<28} {value:.6f}")

print("\\ntwo engines, cross-checked:")
for key, value in network.cross_check().items():
    print(f"  {key:<22} {value}")
"""
    ),
    code(
        """
posterior = network.posteriors({"BE-aggregator-SELECT": "Fail"})
display(pd.Series(posterior, name="P(cause | the judge mis-selected)").head(8))

network.show()
"""
    ),
    md(
        """
The original notebook hand-wired this network beside the fault tree, and nothing
kept the two consistent when the graph changed. Generating it from the tree
removes that failure mode — and buys an **exact** top-event probability, where
the cut-set bound over-estimates because those cut sets share basic events.
"""
    ),
    md(
        """
---
## 7. Calibration from measured outcomes

The original calibrated basic events from semantic-cluster entropy. That path is
still here (`entropy_by_component=`), but the package can do better: given
observed correctness outcomes and an explicit operational profile, HIP-LLM's
hierarchical imprecise posterior gives each component a measured **interval**.
"""
    ),
    code(
        """
from HIP_HOPS_LLM import load_outcomes

outcomes = load_outcomes()           # synthetic, bundled; substitute your own
study_2c = AgenticReliabilityStudy(
    SOURCE_2, name="Approach 2 (calibrated)",
    node_functions=NODE_FUNCS_2, globals_ns=globals(),
)
study_2c.observe(outcomes, profile={"short": 0.30, "medium": 0.50, "long": 0.20})
study_2c.run()

print(study_2c.calibration.summary())
display(study_2c.calibration.to_frame())

print(f"P(H2) placeholder = {study_2.report.analysis('H2').quant.top_probability:.4f}")
print(f"P(H2) calibrated  = {study_2c.hazard_probability('H2')}")
"""
    ),
    md(
        """
The entropy path, for comparison — note that a `uncertainty_summary` of the wrong
shape silently yields default priors, which is why the calibration report names
every component it could not reach:
"""
    ),
    code(
        """
uncertainty_summary = {
    "approach2_react":      {"n_records": 1, "mean_cluster_entropy": 1.33},
    "approach2_cot":        {"n_records": 1, "mean_cluster_entropy": 0.86},
    "approach2_aggregator": {"n_records": 1, "mean_cluster_entropy": 0.45},
}
study_2e = AgenticReliabilityStudy(
    SOURCE_2, name="Approach 2 (entropy-calibrated)",
    node_functions=NODE_FUNCS_2, globals_ns=globals(),
)
study_2e.analyse(uncertainty_summary=uncertainty_summary)

before = study_2.failure_model.events
after = study_2e.failure_model.events
rows = [{
    "basic event": eid,
    "placeholder P": round(before[eid].prob, 4),
    "entropy-calibrated P": round(after[eid].prob, 4),
    "evidence": after[eid].evidence,
} for eid in sorted(after) if eid in before and after[eid].prob != before[eid].prob]
display(pd.DataFrame(rows))
"""
    ),
    md("---\n## 8. Save everything"),
    code(
        """
written = []
for study in (study_1, study_2c):
    written += study.save(OUTPUT_DIR)
for path in written:
    print(path)

# PNGs too, since Kaggle has no mermaid renderer.
for study in (study_1, study_2c):
    stem = study.name.split(" - ")[0].replace(" ", "_").lower()
    for hid in study.report.trees:
        ax = plot_fault_tree(study.report.tree(hid))
        fig = ax.figure
        fig.savefig(f"{OUTPUT_DIR}/{stem}_{hid}.png", dpi=130,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
print("\\nfault tree images written")
"""
    ),
    md(
        """
---
## What changed from the original

| Then | Now |
|---|---|
| A 4,800-line module written into one cell | `pip install` a tested, documented package |
| A pyAgrum network hand-wired beside the tree | The network is *generated* from the tree, so they cannot disagree |
| Basic-event probabilities from engineering judgement | HIP-LLM's imprecise posterior under an explicit operational profile |
| A point estimate | An interval, with its provenance recorded per event |
| Graphviz required to draw the network | matplotlib fallback, so a picture always appears |

Everything here is in
[koo-ec/HIP_HOPS_LLM](https://github.com/koo-ec/HIP_HOPS_LLM) — seven tutorials,
a full API reference, and a test suite that checks the exact and pyAgrum
inference paths agree to nine significant figures.
"""
    ),
]


def main() -> int:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    written = []
    for cells, filename, title in (
        (COLAB, "HIP_HOPS_LLM_Colab.ipynb", "HIP-HOPS-LLM on Colab"),
        (KAGGLE, "hip_hops_for_agentic_ai.ipynb", "HiP-HOPS for Agentic AI"),
    ):
        path = NOTEBOOKS / filename
        path.write_text(
            json.dumps(notebook(cells, title), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)
        print(f"wrote {path.relative_to(ROOT)}  ({len(cells)} cells)")

    try:
        import nbformat

        for path in written:
            nbformat.validate(nbformat.read(path, as_version=4))
            print(f"validated {path.name}")
    except ImportError:
        print("nbformat not installed; skipped validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
