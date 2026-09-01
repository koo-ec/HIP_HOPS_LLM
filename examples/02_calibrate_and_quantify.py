"""Measure under an operational profile, and get the answer as an interval.

    python examples/02_calibrate_and_quantify.py
"""

from __future__ import annotations

import numpy as np

from HIP_HOPS_LLM import AgenticReliabilityStudy, load_example, load_outcomes

PROFILE = {"short": 0.30, "medium": 0.50, "long": 0.20}

study = AgenticReliabilityStudy(
    load_example("parallel_aggregator"), name="parallel agents + aggregator"
)
study.observe(load_outcomes(), profile=PROFILE)
study.run()

print(study.summary())

print("\n--- what was measured ---")
print(study.calibration.evidence_frame().to_string(index=False))

print("\n--- per-stratum failure rates ---")
print("(this is why a single pooled accuracy is not enough)")
for name, evidence in study.evidence.items():
    rates = ", ".join(f"{k}={v:.3f}" for k, v in evidence.by_stratum.items())
    print(f"  {name:<14} {rates}")

print("\n--- what landed on the tree ---")
print(study.calibration.to_frame().to_string(index=False))

print("\n--- the union-splitting rule is exact ---")
for component in ("react_agent", "cot_agent", "aggregator"):
    events = [
        e
        for e in study.failure_model.events.values()
        if e.component == component and e.prob_interval is not None
    ]
    union = 1.0 - np.prod([1.0 - e.prob for e in events])
    print(
        f"  {component:<14} OR of {len(events)} events = {union:.6f}   "
        f"measured = {study.evidence[component].point:.6f}"
    )

print(f"\nP(H2) = {study.hazard_probability('H2')}")

print("\n--- the answer is conditional on the workload ---")
for name, profile in {
    "as measured": PROFILE,
    "harder": {"short": 0.05, "medium": 0.15, "long": 0.80},
    "easier": {"short": 0.70, "medium": 0.25, "long": 0.05},
}.items():
    other = AgenticReliabilityStudy(load_example("parallel_aggregator"))
    other.observe(load_outcomes(), profile=profile).run()
    print(f"  {name:<12} P(H2) in {other.hazard_probability('H2')}")
