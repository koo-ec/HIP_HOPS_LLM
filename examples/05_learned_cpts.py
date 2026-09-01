"""Fit an aggregator's table from data instead of assuming it is an AND gate.

    python examples/05_learned_cpts.py
"""

from __future__ import annotations

import pandas as pd

from HIP_HOPS_LLM import BayesianNetwork, load_outcomes
from HIP_HOPS_LLM.bayes.learn import fit_cpts, learn_gate

outcomes = load_outcomes()
calibration = outcomes[outcomes["split"] == "calibration"].copy()
for column in ("react_agent", "cot_agent", "aggregator"):
    calibration[column] = 1 - calibration[column]        # correct -> failed

learned, distance = learn_gate(calibration, "aggregator", ["react_agent", "cot_agent"])

print("--- how far is the real gate from AND and from OR? ---")
print(f"  distance to AND  {distance['and']:.4f}")
print(f"  distance to OR   {distance['or']:.4f}")
print(f"  nearest          {distance['nearest']}")
print(f"\n{learned.summary()}")

print("\n--- the fitted table ---")
print(pd.DataFrame(learned.cpt.rows()).to_string(index=False))
print(
    "\nRow 0 is the interesting one: both drafts right and the judge still fails\n"
    "about a fifth of the time. A deterministic AND gate says that is impossible,\n"
    "so a model built on one is optimistic by exactly that much.\n"
    "Rows 1 and 2 are also asymmetric — the judge recovers from a bad cot_agent\n"
    "draft far more often than from a bad react_agent one, which no deterministic\n"
    "gate can express."
)

print("\n--- a whole network, fitted from the same data ---")
cpts, fits = fit_cpts(
    outcomes[outcomes["split"] == "calibration"],
    {
        "react_agent": [],
        "cot_agent": [],
        "aggregator": ["react_agent", "cot_agent"],
    },
    outcomes_are_failures=False,      # the columns hold 1 = correct
)
network = BayesianNetwork(cpts=cpts, name="learned")
marginal = network.p_fail("aggregator")
conditional = network.p_fail(
    "aggregator", evidence={"react_agent": "Fail", "cot_agent": "Fail"}
)
print(f"  P(aggregator fails)                     {marginal:.4f}")
print(f"  P(aggregator fails | both drafts wrong) {conditional:.4f}")
for name, fit in fits.items():
    print(f"  {fit.summary()}")

print("\n--- the guard ---")
try:
    learn_gate(outcomes, "aggregator", ["react_agent", "cot_agent"])
except Exception as exc:  # noqa: BLE001
    print(f"  {type(exc).__name__}: {exc}")
