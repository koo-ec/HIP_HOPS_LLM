"""Does the redundancy buy anything? Compare architectures and model diversity.

    python examples/04_compare_architectures.py
"""

from __future__ import annotations

import pandas as pd

from hiphopsllm import EXAMPLES, AgenticReliabilityStudy, load_example

print("--- three architectures, same failure library ---")
rows = []
for key in EXAMPLES:
    study = AgenticReliabilityStudy(load_example(key), name=key)
    study.analyse()
    analysis = study.report.analysis("H2")
    rows.append(
        {
            "architecture": key,
            "components": len(study.system.components),
            "P(H2)": round(analysis.quant.top_probability, 4),
            "cut sets": len(analysis.cuts.sets),
            "min order": min(len(c) for c in analysis.cuts.sets),
            "SPOFs": len(analysis.single_points),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
print(
    "\nMore components is not more reliable: supervisor_workers has the most "
    "parts and the worst P(H2). Depth without redundancy adds ways to be wrong."
)

print("\n--- does diversifying the models help? ---")
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
    study = AgenticReliabilityStudy(
        load_example("parallel_aggregator"), name=name, resource_overrides=overrides
    )
    study.analyse()
    analysis = study.report.analysis("H2")
    order_1 = sorted(next(iter(c)) for c in analysis.cuts.sets if len(c) == 1)
    print(f"\n  {name}")
    print(f"    P(H2)     {analysis.quant.top_probability:.4f}")
    print(f"    order-1   {order_1}")

print(
    "\nDiversifying one agent changes nothing: the judge still shares a snapshot\n"
    "with the other agent, and the judge is on the value path by itself."
)
