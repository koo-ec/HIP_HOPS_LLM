"""Cells for ``notebooks/HIP_HOPS_LLM_Colab.ipynb``.

Kept separate from :mod:`build_notebooks` because this notebook is the long one:
it builds a real LangGraph application, runs it against OpenAI, collects
per-agent outcomes from those runs, and analyses them — rather than loading the
bundled synthetic table.

Everything degrades: without an API key the LLM cells are skipped and the
bundled outcomes are used instead, so the notebook runs end to end in CI and for
anyone who just wants to read it.
"""

from __future__ import annotations

REPO = "https://github.com/koo-ec/HIP_HOPS_LLM.git"

#: Where the clone lands. Deliberately *not* an importable name: a directory
#: called ``hiphopsllm`` beside the notebook would shadow the installed package
#: as an empty namespace package, and the import would succeed with no
#: attributes at all.
CLONE_DIR = "hiphopsllm-repo"


def cells(md, code):
    """Return the notebook's cells, using the caller's ``md``/``code`` builders."""
    return [
        md(
            f"""
# HIP-HOPS-LLM on Colab

**Hierarchical imprecise reliability assessment and failure propagation for
LLM-based agentic systems.**

This notebook builds a **real LangGraph application**, runs it against **OpenAI**,
collects per-agent outcomes from those runs, and turns them into fault trees and
a **pyAgrum** Bayesian network with an exact, imprecise answer.

What happens, end to end:

1. Build a two-agent-plus-judge LangGraph app with `ChatOpenAI`.
2. Run it over a small stratified benchmark, scoring **each node** separately.
3. Extract the architecture, synthesise fault trees, read the cut sets.
4. Calibrate the trees from those measured outcomes under an operational profile.
5. Convert to a Bayesian network and show it with pyAgrum.

**Cost:** about 70 `gpt-4o-mini` calls, a few cents. Without an
`OPENAI_API_KEY` every LLM step is skipped and bundled synthetic outcomes are
used instead, so the notebook still runs top to bottom.

Repository: {REPO.removesuffix('.git')}
"""
        ),
        md("---\n## 1. Install"),
        code(
            f"""
# Graphviz is a native binary, not a Python package. pyAgrum renders through it,
# so install it before the pip step to get pyAgrum's own drawing at the end.
!apt-get -qq install -y graphviz > /dev/null 2>&1

# Clone into a directory that is NOT an importable name. A folder called
# `hiphopsllm` (or `HIP_HOPS_LLM`) beside the notebook would be found before
# site-packages and imported as an empty namespace package.
!git clone --depth=1 {REPO} {CLONE_DIR} 2>&1 | tail -2

# A plain (non-editable) install. `pip install -e` only writes a .pth file, and
# an already-started Colab kernel does not process one, so the package would not
# actually be importable in this session.
%pip install -q "./{CLONE_DIR}[bayes,graph]" langchain-openai
print("installed")
"""
        ),
        code(
            """
import os, platform, sys
from importlib.metadata import PackageNotFoundError, version

# Read versions from installed metadata rather than importing. Importing heavy
# optional packages here can leave a half-initialised module in sys.modules if
# one of their native dependencies fails to load, which then breaks the *later*
# import that actually needs it.
print("python  ", sys.version.split()[0], "on", platform.system())
for name, dist in (
    ("numpy", "numpy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"),
    ("scipy", "scipy"), ("pyagrum", "pyagrum"), ("langgraph", "langgraph"),
    ("langchain_openai", "langchain-openai"), ("openai", "openai"),
):
    try:
        print(f"{name:<18} {version(dist)}")
    except PackageNotFoundError:
        print(f"{name:<18} NOT INSTALLED")

import hiphopsllm as H

print(f"\\nHIP-HOPS-LLM {H.__version__} — {len(H.__all__)} public names")
print("Graphviz available:", H.graphviz_available())
print("module file:", H.__file__)
"""
        ),
        md(
            """
`Graphviz available: True` means pyAgrum will render the Bayesian network at the
end. If it says `False`, nothing breaks — the package draws the same network with
matplotlib and says so in the caption.

`module file` should point inside `site-packages` or the clone's `src/`. If it
points at a bare directory next to the notebook, a folder is shadowing the
package; see the FAQ.
"""
        ),
        md("---\n## 2. Your OpenAI key"),
        code(
            """
import getpass
import os

# In Colab, prefer the secrets panel (key icon on the left) over pasting.
if not os.environ.get("OPENAI_API_KEY"):
    try:
        from google.colab import userdata

        os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY") or ""
    except Exception:
        # Only prompt when someone is actually there to type; this notebook is
        # also executed non-interactively by the test suite.
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                os.environ["OPENAI_API_KEY"] = getpass.getpass(
                    "OPENAI_API_KEY (blank = use bundled synthetic outcomes): "
                )
            except Exception:
                pass

LIVE = bool(os.environ.get("OPENAI_API_KEY"))
MODEL_ID = "gpt-4o-mini"

print("live LLM calls:", LIVE)
if not LIVE:
    print("no key — the graph is still built and analysed, using bundled outcomes")
"""
        ),
        md(
            """
---
## 3. A real LangGraph application

Two agents answer the same question in parallel and a judge picks between them.

Both agents deliberately use the **same model snapshot**. That is the point of
the example: the architecture *looks* redundant, and the analysis will show that
it is not.
"""
        ),
        code(
            '''
llm = None   # built below only when a key is present
if LIVE:
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=MODEL_ID, temperature=0.3, max_retries=2, timeout=60)
    except Exception as exc:
        # A key without a working client is not a live setup. Say so and carry
        # on with the bundled outcomes rather than failing several cells later.
        LIVE = False
        print(f"cannot build the LLM client ({type(exc).__name__}: {exc}) — "
              "falling back to the bundled outcomes")


def react_agent(state):
    """Answer directly, reasoning briefly before committing."""
    model_id = "gpt-4o-mini"                      # MODEL_FAST
    prompt = (
        "Answer with a single number and nothing else.\\n"
        "Think briefly first, then give only the number.\\n\\n" + state["question"]
    )
    return {"answer_react": llm.invoke(prompt).content.strip()}


def cot_agent(state):
    """Answer the same question with explicit step-by-step reasoning."""
    model_id = "gpt-4o-mini"                      # MODEL_DEEP — the SAME snapshot
    prompt = (
        "Solve step by step, then end with a line 'ANSWER: <number>'.\\n\\n"
        + state["question"]
    )
    return {"answer_cot": llm.invoke(prompt).content.strip()}


def aggregator(state):
    """Judge: choose the better of the two candidate answers."""
    model_id = "gpt-4o-mini"
    prompt = (
        "Two assistants answered the same question. Reply with ONLY the correct "
        "final number.\\n\\nQuestion: " + state["question"]
        + "\\n\\nA: " + state.get("answer_react", "")
        + "\\n\\nB: " + state.get("answer_cot", "")
    )
    return {"final": llm.invoke(prompt).content.strip()}


try:
    from typing_extensions import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict, total=False):
        question: str
        answer_react: str
        answer_cot: str
        final: str

    builder = StateGraph(State)
    builder.add_node("react_agent", react_agent)
    builder.add_node("cot_agent", cot_agent)
    builder.add_node("aggregator", aggregator)
    builder.add_edge(START, "react_agent")
    builder.add_edge(START, "cot_agent")
    builder.add_edge("react_agent", "aggregator")
    builder.add_edge("cot_agent", "aggregator")
    builder.add_edge("aggregator", END)
    graph = builder.compile()
    HAVE_LANGGRAPH = True
    print("compiled LangGraph:", type(graph).__name__)
    print("nodes:", list(graph.get_graph().nodes))
except Exception as exc:
    # The analysis only needs the architecture, and that can come from a
    # recorded specification just as well as from a live graph.
    from hiphopsllm import load_example

    graph = load_example("parallel_aggregator")
    HAVE_LANGGRAPH = False
    LIVE = False
    print(f"langgraph unavailable ({type(exc).__name__}) — using the recorded "
          "architecture, and the bundled outcomes below")
'''
        ),
        code(
            """
try:
    # IPython is present in any notebook, but not necessarily wherever this
    # notebook is *executed* as a test, so the import belongs inside the guard.
    from IPython.display import Image, display

    if not HAVE_LANGGRAPH:
        raise RuntimeError("no live graph to render")
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as exc:
    print(f"mermaid.ink unreachable ({type(exc).__name__}) — drawing locally")
    import matplotlib.pyplot as plt

    from hiphopsllm import extract_architecture, plot_architecture

    plot_architecture(extract_architecture(graph, globals_ns=globals()))
    plt.show()
"""
        ),
        md(
            """
---
## 4. Run it, and score every node

A reliability claim needs an **operational profile** — the mix of work the system
will meet — so the benchmark is stratified by how many reasoning steps a question
needs. Each node is scored separately: per-node outcomes are what make the fault
tree quantitative rather than illustrative.
"""
        ),
        code(
            '''
import re

BENCHMARK = [
    # (question, answer, stratum)
    ("What is 17 + 26?", 43, "short"),
    ("What is 8 * 7?", 56, "short"),
    ("What is 144 / 12?", 12, "short"),
    ("What is 95 - 38?", 57, "short"),
    ("What is 13 * 4?", 52, "short"),
    ("What is 200 - 77?", 123, "short"),
    ("A shop sells pens at 3 for 5 pounds. How much do 12 pens cost, in pounds?",
     20, "medium"),
    ("A train travels 60 km in 45 minutes. How many km does it travel in 2 hours?",
     160, "medium"),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines "
     "need to make 100 widgets?", 5, "medium"),
    ("A rectangle is 7 by 9. A second rectangle has twice the area and a width of "
     "6. What is its length?", 21, "medium"),
    ("Anna is 3 times as old as Ben. In 6 years she will be twice his age. How old "
     "is Anna now?", 18, "medium"),
    ("A book costs 12 pounds after a 25% discount. What was the original price in "
     "pounds?", 16, "medium"),
    ("A tank holds 240 litres. Pipe A fills it in 6 hours, pipe B in 4 hours. "
     "Working together, how many hours to fill it?", 2, "medium"),
    ("What is the sum of the first 20 positive integers?", 210, "medium"),
    ("A cyclist rides 30 km at 15 km/h, rests 30 minutes, then rides 20 km at "
     "20 km/h. What is the total journey time in minutes?", 210, "long"),
    ("Three friends split a bill. Ann pays twice what Bob pays, Bob pays 3 pounds "
     "more than Cal, and the total is 51 pounds. How many pounds does Ann pay?",
     26, "long"),
    ("A number is doubled, then 14 is subtracted, then the result is divided by 3, "
     "giving 10. What was the original number?", 22, "long"),
    ("A factory makes 480 units in 8 hours with 6 workers. Keeping the rate per "
     "worker, how many units do 9 workers make in 5 hours?", 450, "long"),
    ("In a class of 30, 18 study French, 15 study German, and 7 study both. How "
     "many study neither?", 4, "long"),
    ("A ladder 13 m long leans against a wall with its foot 5 m from the wall. "
     "It slips down 1 m. How many metres is the foot now from the wall?",
     6, "long"),
]

PROFILE = {"short": 0.30, "medium": 0.50, "long": 0.20}
print(f"{len(BENCHMARK)} questions; target workload {PROFILE}")


def extract_number(text):
    """Pull the answer out of a free-text response.

    Prefer an explicit 'ANSWER: n' line, else take the last number mentioned —
    models tend to state the result last.
    """
    if not text:
        return None
    tagged = re.search(r"ANSWER\\s*[:\\-]?\\s*(-?\\d+(?:\\.\\d+)?)", text, re.I)
    if tagged:
        return float(tagged.group(1))
    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def is_correct(text, expected):
    value = extract_number(text)
    return int(value is not None and abs(value - expected) < 1e-6)
'''
        ),
        code(
            """
import pandas as pd

if LIVE:
    rows = []
    for index, (question, expected, stratum) in enumerate(BENCHMARK):
        state = graph.invoke({"question": question})
        rows.append({
            "item_id": f"item_{index:03d}",
            "stratum": stratum,
            "react_agent": is_correct(state.get("answer_react"), expected),
            "cot_agent": is_correct(state.get("answer_cot"), expected),
            "aggregator": is_correct(state.get("final"), expected),
            "split": "calibration" if index % 4 else "test",
        })
        print(f"  {index + 1:>2}/{len(BENCHMARK)}  {stratum:<7}"
              f"  react={rows[-1]['react_agent']}"
              f"  cot={rows[-1]['cot_agent']}"
              f"  judge={rows[-1]['aggregator']}")
    outcomes = pd.DataFrame(rows)
else:
    from hiphopsllm import load_outcomes

    outcomes = load_outcomes()
    print("using the bundled synthetic outcomes (no API key)")

print()
print("accuracy per node:")
for column in ("react_agent", "cot_agent", "aggregator"):
    print(f"  {column:<14} {outcomes[column].mean():.3f}")
outcomes.head()
"""
        ),
        md(
            """
---
## 5. Extract, synthesise, analyse

`globals_ns=globals()` matters: it lets the extractor interrogate the *live*
`ChatOpenAI` objects rather than only the node source, which is what makes
shared-snapshot detection reliable.
"""
        ),
        code(
            """
from hiphopsllm import AgenticReliabilityStudy

study = AgenticReliabilityStudy(
    graph,
    name="parallel agents + judge (gpt-4o-mini)",
    globals_ns=globals(),
    resource_overrides={
        # Both agents and the judge call the same snapshot. Declared explicitly
        # so the finding does not depend on what the extractor can read.
        "react_agent": {"llm": MODEL_ID, "runtime": "openai-api"},
        "cot_agent": {"llm": MODEL_ID, "runtime": "openai-api"},
        "aggregator": {"llm": MODEL_ID, "runtime": "openai-api"},
    },
)
study.observe(outcomes, profile=PROFILE)
study.run()

print(study.summary())
"""
        ),
        md(
            """
Read the single points of failure. Most cut sets for `H2` are order 2 — the vote
works, both agents must be wrong. But `CCF-LLM-gpt-4o-mini` is **order 1**: a
fault the two agents share defeats the redundancy on its own, and the judge
shares it too.
"""
        ),
        code(
            """
for cut in sorted(study.cut_sets("H2"), key=lambda c: (len(c), c)):
    print(f"  order {len(cut)}:  " + " + ".join(cut))
"""
        ),
        code(
            """
import matplotlib.pyplot as plt

ax = study.plot("H2")
ax.figure.set_size_inches(15, 9)
plt.show()
"""
        ),
        md("---\n## 6. What the runs measured"),
        code(
            """
study.calibration.evidence_frame()
"""
        ),
        code(
            """
print("per-stratum failure rates — why a pooled accuracy is not enough:\\n")
for name, evidence in study.evidence.items():
    rates = ",  ".join(f"{k}={v:.3f}" for k, v in evidence.by_stratum.items())
    print(f"  {name:<14} {rates}")

print()
study.calibration.to_frame()
"""
        ),
        md(
            """
---
## 7. The Bayesian network, in pyAgrum

The network is **generated from the fault tree**, so the two cannot disagree. The
original notebook hand-wired a pyAgrum network beside its tree and nothing kept
them consistent when the graph changed.
"""
        ),
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
import pyagrum

from hiphopsllm import graphviz_available

bn = network.net                      # a real pyagrum.BayesNet
print(f"pyagrum {pyagrum.__version__} — {bn.size()} variables, {bn.sizeArcs()} arcs")

GRAPHVIZ = graphviz_available()
print("Graphviz on PATH:", GRAPHVIZ)

if GRAPHVIZ:
    import pyagrum.lib.notebook as gnb

    gnb.showBN(bn, size="10")
else:
    print("drawing with matplotlib instead")
    network.view().show()
"""
        ),
        code(
            """
# Posterior over every node.
if GRAPHVIZ:
    gnb.showInference(bn, size="12")
else:
    network.view().show()
"""
        ),
        code(
            """
# Conditioned on the judge having hallucinated its own answer.
if GRAPHVIZ:
    gnb.showInference(bn, evs={network.resolve("BE-aggregator-OWN"): 1}, size="12")
else:
    network.view(evidence={"BE-aggregator-OWN": "Fail"}).show()
"""
        ),
        code(
            """
if GRAPHVIZ:
    gnb.sideBySide(
        gnb.getBN(bn, size="7"),
        gnb.getInference(bn, size="7"),
        captions=["structure", "posterior"],
    )
else:
    network.view().side_by_side()
"""
        ),
        md("---\n## 8. Diagnosis and the answer"),
        code(
            """
posterior = network.posteriors({"BE-aggregator-OWN": "Fail"})
pd.Series(posterior, name="P(cause | the judge hallucinated)").head(8)
"""
        ),
        code(
            """
print("exact inference vs the minimal cut upper bound:\\n")
for key, value in network.compare_with_cutsets(study.report.analysis("H2")).items():
    print(f"  {key:<28} {value:.6f}")
print("\\nThe bound is loose because those cut sets share basic events.")

print()
print("P(incorrect answer delivered and accepted) =", study.hazard_probability("H2"))
"""
        ),
        md(
            """
That interval is conditional on the operational profile. Change the workload and
it changes — which is why a failure probability quoted without the workload is
not a claim about the system at all.
"""
        ),
        code(
            """
for name, profile in {
    "as measured": PROFILE,
    "harder": {"short": 0.05, "medium": 0.15, "long": 0.80},
    "easier": {"short": 0.70, "medium": 0.25, "long": 0.05},
}.items():
    other = AgenticReliabilityStudy(graph, globals_ns=globals())
    other.observe(outcomes, profile=profile).run()
    print(f"  {name:<12} P(H2) in {other.hazard_probability('H2')}")
"""
        ),
        md("---\n## 9. Save everything"),
        code(
            """
for path in study.save("artifacts"):
    print(path)

network.view().to_png("artifacts/bn_h2.png")
print("\\nnetwork image written")
"""
        ),
        md(
            f"""
---
## Where next

* **Docs** — <http://koorosh-aslansefat.com/HIP_HOPS_LLM/>
* **Tutorial 7** covers pointing this at your own LangGraph application: what to
  log, how to score intermediate nodes, and how much data you need.
* HIP-LLM's own API is re-exported here, so
  `from hiphopsllm import OperationalFailureProb` works.

Repository: {REPO.removesuffix('.git')}

If you use this, please cite both the HIP-LLM paper
([10.1016/j.ress.2026.112615](https://doi.org/10.1016/j.ress.2026.112615)) and
this repository — `CITATION.cff` has both.
"""
        ),
    ]
