"""Cells for ``notebooks/hiphopsllm_two_cell_addon.ipynb``.

The scenario this notebook answers: someone already has a working LangGraph
notebook — a multi-agent pipeline they built, ran, and are happy with — and wants
a reliability analysis of it without restructuring anything.

The answer is two cells appended to the end. The first runs the graph over a
stratified set of inputs and calibrates from what it measures; the second draws
the Bayesian network. Everything above them is a faithful stand-in for the kind
of notebook this attaches to: a serial research → capture → classify pipeline
with conditional edges, taken from the SMILES/ECHA hazard-check tutorial.
"""

from __future__ import annotations

REPO = "https://github.com/koo-ec/HIP_HOPS_LLM.git"
CLONE_DIR = "hiphopsllm-repo"


def cells(md, code):
    """Return the notebook's cells, using the caller's ``md``/``code`` builders."""
    return [
        md(
            f"""
# Reliability analysis of an existing LangGraph notebook — in two cells

You already have a multi-agent LangGraph notebook that works. This shows how to
get a **quantified reliability analysis** of it by appending two cells:

* **Cell A** runs the graph over a stratified set of inputs, scores every node,
  and calibrates the fault trees from what it measured — the operational profile
  step, in one call.
* **Cell B** generates and draws the Bayesian network.

Sections 1–5 below are a stand-in for *your* notebook: a serial
research → capture → classify pipeline with conditional edges, in the shape of
the SMILES → ECHA hazard-check tutorial. Skip to **section 6** to see the two
cells that matter.

Repository: {REPO.removesuffix('.git')}
"""
        ),
        md("---\n## 1. Install"),
        code(
            f"""
!git clone --depth=1 {REPO} {CLONE_DIR} 2>&1 | tail -1

# A plain (non-editable) install: `pip install -e` only writes a .pth file, and
# an already-started kernel does not process one.
%pip install -q "./{CLONE_DIR}[bayes,graph]"
print("installed")
"""
        ),
        code(
            """
import os, platform, sys
from importlib.metadata import PackageNotFoundError, version

print("python  ", sys.version.split()[0], "on", platform.system())
for name, dist in (("numpy", "numpy"), ("pandas", "pandas"),
                   ("pyagrum", "pyagrum"), ("langgraph", "langgraph")):
    try:
        print(f"{name:<12} {version(dist)}")
    except PackageNotFoundError:
        print(f"{name:<12} NOT INSTALLED")

import hiphopsllm as H

print(f"\\nHIP-HOPS-LLM {H.__version__} — {len(H.__all__)} public names")
print("Graphviz available:", H.graphviz_available())
"""
        ),
        md(
            """
---
## 2. Your pipeline — the shared state

Everything from here to section 5 stands in for the notebook you already have.
"""
        ),
        code(
            """
import typing


class AgentState(typing.TypedDict, total=False):
    smiles: str
    identity: dict
    browser_status: str
    capture: dict
    capture_status: str
    risk: dict
    workflow_succeeded: bool
    error: str
"""
        ),
        md("---\n## 3. Your three agents"),
        code(
            '''
import random

# A stand-in for the real agents' behaviour, so the notebook runs anywhere.
# Replace these three bodies with your own and nothing below changes.
_rng = random.Random(20260902)


def research_agent(state: AgentState) -> AgentState:
    """Resolve the SMILES to an ECHA substance and prepare the browser."""
    resolved = _rng.random() < 0.88
    return {
        "identity": {"smiles": state["smiles"]},
        "browser_status": "ready" if resolved else "unresolved",
    }


def pixelrag_agent(state: AgentState) -> AgentState:
    """Capture the hazard table from the ECHA page with PixelRAG."""
    captured = _rng.random() < 0.82
    return {
        "capture": {"rows": 7 if captured else 0},
        "capture_status": "captured" if captured else "missed",
    }


def safety_agent(state: AgentState) -> AgentState:
    """Classify the hazard from the captured evidence with an LLM."""
    model_id = "gpt-4o-mini"
    correct = _rng.random() < 0.91
    return {
        "risk": {"ghs": "H302" if correct else "H000"},
        "workflow_succeeded": correct,
    }
'''
        ),
        md("---\n## 4. Your graph"),
        code(
            """
from langgraph.graph import END, START, StateGraph

builder = StateGraph(AgentState)
builder.add_node("research_agent", research_agent)
builder.add_node("pixelrag_agent", pixelrag_agent)
builder.add_node("safety_agent", safety_agent)

builder.add_edge(START, "research_agent")
builder.add_conditional_edges(
    "research_agent",
    lambda state: state.get("browser_status") == "ready",
    {True: "pixelrag_agent", False: END},
)
builder.add_conditional_edges(
    "pixelrag_agent",
    lambda state: state.get("capture_status") == "captured",
    {True: "safety_agent", False: END},
)
builder.add_edge("safety_agent", END)

graph = builder.compile()
print("Agents: research_agent -> pixelrag_agent -> safety_agent")
"""
        ),
        md("---\n## 5. Your inputs"),
        code(
            """
SMILES = [
    "CN1CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42",   # clozapine
    "CC(=O)Oc1ccccc1C(=O)O",                            # aspirin
    "CCO",                                              # ethanol
    "c1ccccc1",                                         # benzene
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                       # ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                     # caffeine
    "OCC1OC(O)C(O)C(O)C1O",                             # glucose
    "ClCCl",                                            # dichloromethane
] * 8

print(len(SMILES), "inputs")
"""
        ),
        md(
            """
---
## 6. Cell A — the operational profile, in one call

`run_and_observe` invokes the graph once per input, scores **each node**
separately, builds the operational profile, and calibrates the fault trees from
what it measured.

Two things it needs from you, because it cannot invent them:

* `stratum` — how to bucket an input. A reliability claim is conditional on the
  mix of work the system will meet, so the buckets should be things whose
  difficulty genuinely differs.
* `success` — how to tell, from the final state, whether each node did its job.
  Return `None` when a node was **not exercised** (a router sent the run to
  `END` first). That is recorded as a missing observation, not a failure —
  scoring an unreached node as failed would blame it for an upstream fault.
"""
        ),
        code(
            """
from hiphopsllm import AgenticReliabilityStudy

study = AgenticReliabilityStudy(
    graph,
    name="SMILES → ECHA hazard check",
    globals_ns=globals(),
    resource_overrides={"safety_agent": {"llm": "gpt-4o-mini", "runtime": "api"}},
)

outcomes = study.run_and_observe(
    inputs=[{"smiles": s, "capture_attempts": 3} for s in SMILES],
    stratum=lambda item: "complex" if len(item["smiles"]) > 25 else "simple",
    success={
        "research_agent": lambda s: s.get("browser_status") == "ready",
        "pixelrag_agent": lambda s: (
            s["capture_status"] == "captured" if "capture_status" in s else None
        ),
        "safety_agent": lambda s: (
            bool(s["workflow_succeeded"]) if "workflow_succeeded" in s else None
        ),
    },
    profile={"simple": 0.35, "complex": 0.65},   # the workload you expect
    progress=False,
)

print(study.summary())
"""
        ),
        md(
            """
Read the evidence table: `n` falls down the chain because a node that a router
never reached is not scored. That is the point — `pixelrag_agent`'s failure rate
is measured over the runs where it actually ran.
"""
        ),
        code(
            """
study.calibration.evidence_frame()
"""
        ),
        code(
            """
print("per-stratum failure rates — why the profile matters:\\n")
for name, evidence in study.evidence.items():
    rates = ",  ".join(f"{k}={v:.3f}" for k, v in evidence.by_stratum.items())
    print(f"  {name:<16} {rates}")

print()
print("minimal cut sets for H2 (incorrect answer delivered and accepted):")
for cut in sorted(study.cut_sets("H2"), key=lambda c: (len(c), c)):
    print(f"  order {len(cut)}:  " + " + ".join(cut))
"""
        ),
        md(
            """
---
## 7. Cell B — generate and visualise the Bayesian network

The network is generated *from* the fault tree, so the two cannot disagree.
"""
        ),
        code(
            """
network = study.bayesnet("H2")
network.show()
"""
        ),
        md(
            """
That is the whole two-cell addition. Everything below is optional detail.

---
## 8. What the network gives you that the tree does not
"""
        ),
        code(
            """
print("two independent inference engines, cross-checked:")
for key, value in network.cross_check().items():
    print(f"  {key:<22} {value}")

print("\\nexact vs the minimal cut upper bound:")
for key, value in network.compare_with_cutsets(study.report.analysis("H2")).items():
    print(f"  {key:<28} {value:.6f}")

print("\\nP(incorrect hazard classification accepted) =",
      study.hazard_probability("H2"))
"""
        ),
        code(
            """
# Diagnosis: condition on what a run showed and read the posterior over causes.
import pandas as pd

cause = study.report.analysis("H2").cuts.sets
first_single = next((next(iter(c)) for c in cause if len(c) == 1), None)
if first_single:
    print(f"given {first_single} occurred:\\n")
    display(pd.Series(network.posteriors({first_single: "Fail"}),
                      name="P(cause | evidence)").head(8))
    network.view(evidence={first_single: "Fail"}).show()
"""
        ),
        code(
            """
# pyAgrum's own rendering, where Graphviz is installed.
from hiphopsllm import graphviz_available

if graphviz_available():
    import pyagrum.lib.notebook as gnb

    gnb.showInference(network.net, size="12")
else:
    print("Graphviz not on PATH — the matplotlib rendering above is the same "
          "network. `!apt-get -qq install graphviz` to use pyAgrum's.")
"""
        ),
        md("---\n## 9. Keep the measurement"),
        code(
            """
outcomes.to_csv("agent_outcomes.csv", index=False)
for path in study.save("artifacts"):
    print(path)
"""
        ),
        md(
            f"""
---
## Applying this to your own notebook

Copy the two cells in sections 6 and 7, and change three things:

1. `inputs` — your own list of graph inputs.
2. `stratum` — how you bucket them.
3. `success` — how each node's outcome is read from the final state, returning
   `None` where a node was not exercised.

Everything else follows. Full documentation:
<http://koorosh-aslansefat.com/HIP_HOPS_LLM/>

Repository: {REPO.removesuffix('.git')}
"""
        ),
    ]
