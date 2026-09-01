"""Fault tree to CPTs to Bayesian network: exact probability and diagnosis.

    python examples/03_bayesian_network.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from HIP_HOPS_LLM import (  # noqa: E402
    AgenticReliabilityStudy,
    graphviz_available,
    load_example,
    load_outcomes,
)

study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
study.observe(load_outcomes(), profile={"short": 0.3, "medium": 0.5, "long": 0.2})
study.run()

cpts = study.cpts("H2")
print(cpts.summary())

network = study.bayesnet("H2")
print("\n" + network.summary())

print("\n--- two engines that share no code, cross-checked ---")
for key, value in network.cross_check().items():
    print(f"  {key:<22} {value}")

print("\n--- exact vs the minimal cut upper bound ---")
for key, value in network.compare_with_cutsets(study.report.analysis("H2")).items():
    print(f"  {key:<26} {value:.6f}")
print("  (the bound is loose because those cut sets share basic events)")

print("\n--- posterior over causes, given the judge hallucinated ---")
for cause, p in list(network.posteriors({"BE-aggregator-OWN": "Fail"}).items())[:6]:
    print(f"  {cause:<45} {p:.4f}")

print("\n--- most probable explanation of the hazard ---")
for event, state in network.most_probable_explanation({"TOP": "Fail"}).items():
    if state == "Fail":
        print(f"  {event}")

imprecise = study.imprecise_bayesnet("H2")
print("\n" + imprecise.summary())

print("\n--- per-cause envelopes ---")
for cause, envelope in list(imprecise.posterior_envelopes().items())[:5]:
    print(f"  {cause:<45} {envelope}")

view = network.view()
print(f"\nGraphviz available: {graphviz_available()}   backend: {view.resolved_backend}")
print(f"network written to {view.to_png('bn_h2.png')}")
