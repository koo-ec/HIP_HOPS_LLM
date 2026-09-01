"""Analyse a live LangGraph application.

Needs langgraph installed; falls back to the recorded mermaid text if it is not,
so the example always runs.

    pip install langgraph
    python examples/06_your_own_langgraph.py
"""

from __future__ import annotations

import re

from HIP_HOPS_LLM import AgenticReliabilityStudy


# --- the node functions. Only their *source* is read; they are never called. ---
def generator(state):
    """ReAct step: generate up to 'Observation'.

    ``llm`` is deliberately undefined: this body is never executed. Only its
    *source* is read, to classify the role and find the model id — which is the
    whole point of the extraction step.
    """
    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    messages = [{"role": "user", "content": state["prompt"]}]
    return {
        "output": llm.generate(messages, model=model_id),  # noqa: F821
        "prompt": state["prompt"],
    }


def router(state):
    """Conditional edge: choose the tool, finish, or fail."""
    if "calculator" in state.get("output", ""):
        return "coder"
    if re.search(r"Final Answer", state.get("output", "")):
        return "end"
    return "error"


def coder(state):
    """Calculator tool: evaluate the extracted expression."""
    match = re.search(r"Action Input: (.*?)(?=Observation)", state["output"], re.DOTALL)
    try:
        result = eval(match.group(1).strip())          # noqa: S307 - the point
    except Exception:                                  # noqa: BLE001
        result = "not something a simple calculator can execute"
    return {"tool_output": str(result)}


MERMAID = """
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

try:
    from typing_extensions import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict, total=False):
        prompt: str
        output: str
        tool_output: str

    builder = StateGraph(State)
    builder.add_node("generator", generator)
    builder.add_node("coder", coder)
    builder.add_edge(START, "generator")
    builder.add_edge("coder", "generator")
    builder.add_conditional_edges(
        "generator", router, {"coder": "coder", "error": END, "end": END}
    )
    SOURCE = builder.compile()
    print("using a live LangGraph object")
except Exception as exc:                               # noqa: BLE001
    SOURCE = MERMAID
    print(f"langgraph unavailable ({type(exc).__name__}) - using recorded mermaid")

study = AgenticReliabilityStudy(
    SOURCE,
    name="ReAct + calculator (live)",
    globals_ns=globals(),
    node_functions={
        "generator": generator,
        "coder": coder,
        "generator::router": router,     # the add_conditional_edges callable
    },
    unroll=1,
)
study.analyse()
print(study.summary())

print("\nH5 exists because coder's source contains eval():")
for row in study.single_points():
    if row["severity"] == "catastrophic":
        print(f"  [{row['severity']}] {row['hazard']}  {row['event']}")

print("\nThe feedback loop was unrolled, not deleted:")
print(study.report.cycle_report.summary())
