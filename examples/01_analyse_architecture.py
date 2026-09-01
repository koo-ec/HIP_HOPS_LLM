"""Structure only: cut sets, single points of failure, and the FMEA.

No outcome data, no operational profile, no API key. Run this the day you write
the graph — it costs nothing and its findings are usually the ones worth acting
on first.

    python examples/01_analyse_architecture.py
"""

from __future__ import annotations

import pandas as pd

from HIP_HOPS_LLM import AgenticReliabilityStudy, load_example

study = AgenticReliabilityStudy(
    load_example("react_calculator"), name="ReAct + calculator"
)
study.analyse()

print(study.summary())

print("\n--- architecture ---")
print(pd.DataFrame(study.system.architecture_table()).set_index("component"))

print("\n--- minimal cut sets for H2 (incorrect answer accepted) ---")
for cut in sorted(study.cut_sets("H2"), key=lambda c: (len(c), c)):
    print(f"  order {len(cut)}: " + " + ".join(cut))

print("\n--- single points of failure ---")
print(pd.DataFrame(study.single_points()).to_string(index=False))

print("\n--- FMEA (first rows) ---")
print(study.fmea().head(6).to_string(index=False, max_colwidth=28))
