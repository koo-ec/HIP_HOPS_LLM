# Changelog

## 0.1.0

First release.

### Architecture

- LangGraph, mermaid or specification-dict extraction into a ported component
  model; conditional edges materialised as router components with their own
  failure logic.
- Shared-resource detection from node source and from live model objects,
  producing common-cause groups.
- Feedback-loop elimination by unrolling to depth *k* plus a feedback-cut
  component, so a loop's contribution is preserved rather than deleted.
- `LangGraphExtractor` for applying one set of conventions across several graphs.

### Fault trees

- Six failure classes; an archetype library covering LLM agent, tool, router,
  aggregator, transform, boundary and feedback cut.
- Compositional synthesis with shared sub-trees; minimal cut sets by MOCUS with
  absorption; MCUB quantification; Birnbaum and Fussell–Vesely importance;
  single points of failure; a generated FMEA.
- Exports: mermaid, Graphviz DOT, JSON, Open-PSA MEF XML, Markdown, CSV.

### Reliability

- `OperationalProfile` with recorded provenance; empirical, uniform and declared
  constructors; a `stratify` helper that refuses labels outside the profile.
- `EvidenceCalibrator`: HIP-LLM's imprecise posterior per component, written onto
  basic events through an exact union-splitting rule, plus a report naming every
  component and event it did not touch.
- The whole of HIP-LLM re-exported under one namespace, with a test asserting the
  re-export stays complete.

### Bayesian networks

- `fault_tree_to_cpts`: deterministic AND/OR, *k*-of-*n*, and noisy-OR with leak.
- `BayesianNetwork`: exact inference through pyAgrum *or* NumPy enumeration,
  cross-checked against each other; posterior over causes given evidence; most
  probable explanation.
- `ImpreciseBayesianNetwork`: the lower/upper pair, licensed by a monotonicity
  test over the basic events.
- `learn_cpt` / `learn_gate` / `fit_cpts`: tables fitted from observed outcomes,
  with Dirichlet smoothing, coverage reporting, and a refusal to fit on a test
  split.
- `BayesNetView`: pyAgrum where Graphviz exists, matplotlib where it does not,
  with a barycentre layout pass and posterior shading.

### Pipeline

- `AgenticReliabilityStudy`: the whole chain behind one chainable object, which
  refuses to calibrate without outcomes or without an operational profile, and
  prints `NOT CALIBRATED` until it has both.

### Fixed in the underlying analysis code

Three defects found while packaging, each of which failed silently:

- **A router node declared explicitly lost its incoming edge.** The `node ->
  node::router` connection was emitted only when the router component was
  created, so pre-declaring one (to attach its source) disconnected the graph —
  and every feedback loop through that router vanished from the analysis with no
  warning.
- **The specification-dict extraction path discarded every option.** `role_overrides`,
  `resource_overrides`, `globals_ns`, `node_functions` and `materialise_routers`
  were all dropped, because the dict branch short-circuited to `from_spec`
  instead of going through `build_system_model`.
- **`probability_overrides` ignored an unknown basic-event id.** A typo left the
  placeholder in place while the analyst believed it replaced. It now raises and
  lists the ids the model does have, and rejects values outside `[0, 1]`.
