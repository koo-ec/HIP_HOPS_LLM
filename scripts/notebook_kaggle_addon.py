"""Cells for ``notebooks/hiphopsllm_two_cell_addon.ipynb``.

The scenario this notebook answers: someone already has a working LangGraph
notebook — a multi-agent pipeline they built, ran, and are happy with — and wants
a reliability analysis of it without restructuring anything.

The answer is two cells appended to the end. The first runs the graph over a
stratified set of inputs and calibrates from what it measures; the second draws
the Bayesian network. Everything above them is a faithful stand-in for the kind
of notebook this attaches to: a serial research → capture → classify pipeline
with conditional edges, taken from the SMILES/ECHA hazard-check tutorial.

The presentation is deliberate. Hosted notebooks render styled ``<div>`` blocks
but sanitise much other HTML, so the headings are divs and the contents table is
ordinary anchored Markdown.
"""

from __future__ import annotations

REPO = "https://github.com/koo-ec/HIP_HOPS_LLM.git"
CLONE_DIR = "hiphopsllm-repo"
FIGURE = (
    "https://raw.githubusercontent.com/koo-ec/HIP_HOPS_LLM/main/"
    "docs/figures/hip-hops-llm-overview.png"
)

INK = "#0f2942"
ACCENT = "#2f6f4f"


def _banner(number: str, title: str, blurb: str, anchor: str) -> str:
    """A section heading: a coloured band, a step number, a title, and one line of why."""
    return f"""
<a id="{anchor}"></a>
<div style="background:linear-gradient(90deg,{INK} 0%,#1d4e6f 100%);
            border-radius:8px;padding:14px 20px;margin:8px 0 4px 0;">
  <div style="color:#9fd3b8;font-family:Georgia,serif;font-size:13px;
              letter-spacing:2px;text-transform:uppercase;">Step {number}</div>
  <div style="color:#ffffff;font-family:Georgia,serif;font-size:26px;
              font-weight:bold;line-height:1.25;">{title}</div>
</div>

{blurb}
"""


def cells(md, code):
    """Return the notebook's cells, using the caller's ``md``/``code`` builders."""
    return [
        md(
            """
<div style="background:linear-gradient(135deg,#0f2942 0%,#1d4e6f 55%,#2f6f4f 100%);
            border-radius:14px;padding:38px 30px;text-align:center;">
  <div style="color:#9fd3b8;font-family:Georgia,serif;font-size:15px;
              letter-spacing:4px;text-transform:uppercase;">
    Hierarchically Performed Hazard Origin &amp; Propagation Studies
  </div>
  <div style="color:#ffffff;font-family:Georgia,serif;font-size:44px;
              font-weight:bold;margin:10px 0 6px 0;line-height:1.15;">
    HIP-HOPS-LLM
  </div>
  <div style="color:#e8f2ec;font-family:Georgia,serif;font-size:23px;
              font-style:italic;margin-bottom:16px;">
    Reliability Engineering for Agentic AI
  </div>
  <div style="color:#cfe3d8;font-family:Helvetica,Arial,sans-serif;font-size:15px;
              max-width:760px;margin:0 auto;line-height:1.6;">
    From a LangGraph application to synthesised fault trees, minimal cut sets,
    a generated FMEA and a calibrated Bayesian network, ending in a defensible
    <b>interval</b> for the probability of failure, not a number nobody can check.
  </div>
  <div style="margin-top:20px;">
    <span style="background:#ffffff22;color:#ffffff;border-radius:14px;
                 padding:5px 13px;font-family:Helvetica,Arial,sans-serif;
                 font-size:12px;margin:0 4px;">fault tree synthesis</span>
    <span style="background:#ffffff22;color:#ffffff;border-radius:14px;
                 padding:5px 13px;font-family:Helvetica,Arial,sans-serif;
                 font-size:12px;margin:0 4px;">imprecise probability</span>
    <span style="background:#ffffff22;color:#ffffff;border-radius:14px;
                 padding:5px 13px;font-family:Helvetica,Arial,sans-serif;
                 font-size:12px;margin:0 4px;">operational profiles</span>
    <span style="background:#ffffff22;color:#ffffff;border-radius:14px;
                 padding:5px 13px;font-family:Helvetica,Arial,sans-serif;
                 font-size:12px;margin:0 4px;">Bayesian networks</span>
    <span style="background:#ffffff22;color:#ffffff;border-radius:14px;
                 padding:5px 13px;font-family:Helvetica,Arial,sans-serif;
                 font-size:12px;margin:0 4px;">LangGraph</span>
  </div>
</div>
"""
        ),
        md(
            f"""
<p align="center">
  <img src="{FIGURE}" width="94%"
       alt="HIP-HOPS-LLM: from a layered system design of agentic-AI and non-AI
       components, through functional failure analysis and interface-focused
       FMEAs, to a synthesised Bayesian network whose basic-event probabilities
       are derived from operational-profile reliability.">
</p>

<a id="abstract"></a>
<div style="border-left:5px solid {ACCENT};background:#f4f8f6;
            border-radius:0 8px 8px 0;padding:18px 24px;margin:6px 0 10px 0;">
  <div style="font-family:Georgia,serif;font-size:19px;font-weight:bold;
              color:{INK};margin-bottom:8px;">Abstract</div>
  <div style="font-family:Helvetica,Arial,sans-serif;font-size:14.5px;
              line-height:1.75;color:#20303a;text-align:justify;">
    <b>HIP-HOPS-LLM applies established system-safety engineering to agentic AI.</b>
    A multi-agent LLM application is built from components whose failure rates can
    be measured, but whose <i>system</i> reliability is not their product.
    <b>HIP-HOPS</b> supplies the structure: each component carries its local failure
    logic, and system fault trees, minimal cut sets and an FMEA are synthesised
    automatically from the architecture, so the analysis cannot drift from the code
    it describes. <b>HIP-LLM</b> supplies the numbers, defining reliability as the
    probability of failure-free operation over a stated number of future tasks under
    a given <b>operational profile</b>, and returning posterior <i>envelopes</i>
    rather than point estimates from hierarchical imprecise-Bayesian inference
    across a subdomain, domain and model hierarchy. This package joins them: it
    reads a LangGraph application, measures every component under an explicit
    profile, calibrates each basic event with those intervals, and converts the
    result into a Bayesian network for exact inference, diagnosis and visualisation.
    The quantity measured is task failure, whatever its mode.
  </div>
</div>

<div style="border-left:5px solid {ACCENT};background:#f9fbfa;
            border-radius:0 8px 8px 0;padding:11px 24px;margin:0 0 12px 0;
            font-family:Helvetica,Arial,sans-serif;font-size:14px;
            line-height:1.7;color:#20303a;">
  <b style="color:{INK};">Keywords:</b>
  reliability engineering, safety analysis, fault tree synthesis, HiP-HOPS, FMEA,
  Bayesian networks, imprecise probability, operational profile, agentic AI,
  multi-agent systems, LangGraph, large language models
</div>
"""
        ),
        md(
            f"""
<a id="contents"></a>
<div style="background:{INK};border-radius:8px;padding:12px 20px;margin:10px 0;">
  <span style="color:#ffffff;font-family:Georgia,serif;font-size:22px;
               font-weight:bold;">Contents</span>
</div>

**What this notebook is.** You already have a multi-agent LangGraph notebook that
works. This shows how to get a quantified reliability analysis of it by appending
**two cells**: [Step 6](#step6) and [Step 7](#step7). Everything before them
stands in for *your* notebook.

| | Section | What happens |
|---|---|---|
| | [Abstract](#abstract) | Why HIP-HOPS-LLM exists |
| 1 | [Install](#step1) | Clone and install the package |
| 2 | [Your shared state](#step2) | The `TypedDict` your agents pass around |
| 3 | [Your three agents](#step3) | Research → capture → classify |
| 4 | [Your graph](#step4) | The compiled LangGraph, with conditional edges |
| 5 | [Your inputs](#step5) | A stratified set of molecules |
| **6** | [**Cell A: the operational profile**](#step6) | **Run, score every node and calibrate, in one call** |
| **7** | [**Cell B: the Bayesian network**](#step7) | **Generate and visualise it** |
| 8 | [What the network adds](#step8) | Exact inference, diagnosis, pyAgrum |
| 9 | [Keep the measurement](#step9) | Save the artefacts |
| | [Applying this to your notebook](#yours) | The three things you change |

<div style="border-left:5px solid #b8860b;background:#fdf9ee;padding:12px 18px;
            border-radius:0 6px 6px 0;font-family:Helvetica,Arial,sans-serif;
            font-size:14px;line-height:1.6;">
  <b>Runtime:</b> about a minute, most of it the install. No GPU, no API key and
  no data of your own; the stand-in agents are deterministic.
</div>
"""
        ),
        md(
            _banner(
                "1",
                "Install",
                "A plain install rather than an editable one: `pip install -e` only "
                "writes a `.pth` file, and an already-started kernel never processes "
                "it, so the package would not actually be importable in this session.",
                "step1",
            )
        ),
        code(
            f"""
!git clone --depth=1 {REPO} {CLONE_DIR} 2>&1 | tail -1
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

print(f"\\nHIP-HOPS-LLM {H.__version__}: {len(H.__all__)} public names")
print("Graphviz available:", H.graphviz_available())
"""
        ),
        md(
            _banner(
                "2",
                "Your shared state",
                "Everything from here to Step 5 stands in for the notebook you "
                "already have. Replace it with yours and nothing below changes.",
                "step2",
            )
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
        md(
            _banner(
                "3",
                "Your three agents",
                "A research agent resolves the molecule and opens the ECHA page, a "
                "capture agent reads the hazard table from it, and a safety agent "
                "classifies the hazard with an LLM. Each fails in its own way, "
                "which is what the analysis is about.",
                "step3",
            )
        ),
        code(
            '''
import random

# Stand-ins for the real agents' behaviour, so the notebook runs anywhere.
# Replace these three bodies with your own; the analysis does not change.
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
        md(
            _banner(
                "4",
                "Your graph",
                "Two conditional edges make this a *serial* pipeline with early "
                "exits: a run that cannot resolve the molecule never reaches the "
                "capture agent, and one that cannot capture never reaches the "
                "classifier. Step 6 accounts for that.",
                "step4",
            )
        ),
        code(
            """
try:
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
    run_one = graph.invoke
    print("Agents: research_agent -> pixelrag_agent -> safety_agent")

except ImportError:
    # Without langgraph the *analysis* still works: it needs the architecture and
    # a way to run one item, and both can be supplied directly. The recorded
    # specification is the same graph, and `run_one` chains the same three
    # functions through the same conditions.
    graph = {
        "name": "SMILES to ECHA hazard check",
        "nodes": {
            "__start__": {"role": "source"},
            "research_agent": {"role": "tool"},
            "pixelrag_agent": {"role": "tool"},
            "safety_agent": {"role": "llm_agent",
                             "resources": {"llm": "gpt-4o-mini"}},
            "__end__": {"role": "sink"},
        },
        "edges": [
            ["__start__", "research_agent"],
            ["research_agent", "pixelrag_agent", "ready", True],
            ["research_agent", "__end__", "unresolved", True],
            ["pixelrag_agent", "safety_agent", "captured", True],
            ["pixelrag_agent", "__end__", "missed", True],
            ["safety_agent", "__end__"],
        ],
    }

    def run_one(item):
        state = dict(item)
        state.update(research_agent(state))
        if state.get("browser_status") == "ready":
            state.update(pixelrag_agent(state))
            if state.get("capture_status") == "captured":
                state.update(safety_agent(state))
        return state

    print("langgraph not installed; using the recorded architecture and a "
          "plain-Python runner. The analysis below is unchanged.")
"""
        ),
        md(
            _banner(
                "5",
                "Your inputs",
                "A reliability claim is conditional on the mix of work the system "
                "will meet, so the inputs are bucketed into strata whose difficulty "
                "genuinely differs (here, structural complexity).",
                "step5",
            )
        ),
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
            _banner(
                "6",
                "Cell A: the operational profile, in one call",
                """`run_and_observe` invokes the graph once per input, scores **each
node** separately, builds the operational profile, and calibrates the fault trees
from what it measured.

Two things it needs from you, because it cannot invent them:

* **`stratum`**: how to bucket an input.
* **`success`**: how to tell, from the final state, whether each node did its
  job. Return `None` when a node was **not exercised**, because a router sent the
  run to `END` first. That is recorded as a *missing observation*, not a failure:
  scoring an unreached node as failed would blame it for an upstream fault.""",
                "step6",
            )
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
    invoke=run_one,
    progress=False,
)

print(study.summary())
"""
        ),
        md(
            """
Read the evidence table below. **`n` falls down the chain**, because a node that
a router never reached is not scored. That is the point: `pixelrag_agent`'s
failure rate is measured over the runs where it actually ran, not diluted by the
ones it never saw.
"""
        ),
        code(
            """
study.calibration.evidence_frame()
"""
        ),
        code(
            """
print("per-stratum failure rates, and why the profile matters:\\n")
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
            _banner(
                "7",
                "Cell B: generate and visualise the Bayesian network",
                "The network is generated *from* the fault tree, so the two cannot "
                "disagree. Each node shows its posterior as a labelled bar per "
                "state, and the caption reports the inference time.",
                "step7",
            )
        ),
        code(
            """
network = study.bayesnet("H2")
network.show()
"""
        ),
        md(
            f"""
<div style="border-left:5px solid {ACCENT};background:#f4f8f6;padding:14px 20px;
            border-radius:0 6px 6px 0;font-family:Helvetica,Arial,sans-serif;
            font-size:14.5px;line-height:1.65;">
  <b>That is the whole two-cell addition.</b> Everything below is optional detail:
  what the network buys you beyond the tree, and how to keep the measurement.
</div>
"""
        ),
        md(
            _banner(
                "8",
                "What the network adds",
                "An exact top-event probability where the cut-set bound gives only "
                "an upper limit, a posterior over causes given evidence, and a "
                "second inference engine to check the first against.",
                "step8",
            )
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

singles = [next(iter(c)) for c in study.report.analysis("H2").cuts.sets if len(c) == 1]
if singles:
    print(f"given {singles[0]} occurred:\\n")
    display(pd.Series(network.posteriors({singles[0]: "Fail"}),
                      name="P(cause | evidence)").head(8))
    network.view(evidence={singles[0]: "Fail"}).show()
"""
        ),
        code(
            """
# pyAgrum's own rendering, where the Graphviz binary is installed.
from hiphopsllm import graphviz_available

if graphviz_available():
    import pyagrum.lib.notebook as gnb

    gnb.showInference(network.net, size="12")
else:
    print("Graphviz not on PATH; the rendering above is the same network. "
          "Run `!apt-get -qq install graphviz` to use pyAgrum's.")
"""
        ),
        md(
            _banner(
                "9",
                "Keep the measurement",
                "Re-running the graph is the expensive part. The outcomes table is "
                "the measurement; the artefacts hold the trees, the cut sets, the "
                "FMEA and the calibration provenance.",
                "step9",
            )
        ),
        code(
            """
outcomes.to_csv("agent_outcomes.csv", index=False)
for path in study.save("artifacts"):
    print(path)
"""
        ),
        md(
            f"""
<a id="yours"></a>
<div style="background:linear-gradient(90deg,{INK} 0%,{ACCENT} 100%);
            border-radius:8px;padding:14px 20px;margin:8px 0;">
  <span style="color:#ffffff;font-family:Georgia,serif;font-size:24px;
               font-weight:bold;">Applying this to your own notebook</span>
</div>

Copy the two cells from [Step 6](#step6) and [Step 7](#step7), and change three
things:

1. **`inputs`**: your own list of graph inputs.
2. **`stratum`**: how you bucket them into strata whose difficulty differs.
3. **`success`**: how each node's outcome is read from the final state,
   returning `None` where a node was not exercised.

Everything else follows: the architecture is read from your compiled graph, the
fault trees are synthesised from it, and the Bayesian network is generated from
those.

* **Documentation:** [koorosh-aslansefat.com/HIP_HOPS_LLM](http://koorosh-aslansefat.com/HIP_HOPS_LLM/)
* **Repository:** [{REPO.removesuffix('.git')}]({REPO.removesuffix('.git')})

<div style="background:{INK};border-radius:8px;padding:11px 20px;margin:16px 0 10px 0;">
  <span style="color:#ffffff;font-family:Georgia,serif;font-size:21px;
               font-weight:bold;">References</span>
</div>

* Aghazadeh-Chakherlou, R., Guo, Q., Khastgir, S., Popov, P., Zhang, X., &
  Zhao, X. (2026). A hierarchical imprecise probability approach to reliability
  assessment of large language models. *Reliability Engineering & System
  Safety*, *272*, 112615. https://doi.org/10.1016/j.ress.2026.112615
* Custers, B., & Aslansefat, K. (2026). Runtime uncertainty monitoring for
  LLM-based multi-agent systems using Bayesian networks. In *Computer Safety,
  Reliability, and Security: SAFECOMP 2026 Workshops, 9th International Workshop
  on Artificial Intelligence Safety Engineering (WAISE 2026)*, Valencia, Spain.
  Springer. (in press)
* Donaldson, L., Walker, C., Aslansefat, K., & Papadopoulos, Y. (2026). Bayesian
  uncertainty propagation for agentic RAG pipelines: A proof-of-concept study on
  multi-hop question answering. In *Proceedings of the 7th International
  Conference on Maintenance and Intelligent Asset Management (ICMIAM 2026)*,
  Huddersfield, UK, 1-3 September 2026. Springer Nature.
* Papadopoulos, Y., & McDermid, J. A. (1999). Hierarchically performed hazard
  origin and propagation studies. In *Computer Safety, Reliability and Security
  (SAFECOMP 1999)* (Lecture Notes in Computer Science, Vol. 1698, pp. 139-152).
  Springer. https://doi.org/10.1007/3-540-48249-0_13

If you use this, please cite both the HIP-LLM paper and the repository;
`CITATION.cff` has both.
"""
        ),
    ]
