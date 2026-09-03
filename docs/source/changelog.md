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
- `plot_architecture` draws the extracted model rather than the source graph, so
  the picture and the analysis are the same object. Conditional edges are dashed;
  feedback edges and any edge that skips a layer — an early exit to `__end__` is
  the common case — are routed around the side in their own lane, because drawn
  straight they are a vertical line hidden behind every component they pass, with
  the branch label landing on top of one of them.

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

### Testing

736 tests, 94% line coverage, every module above 88%. CI runs them on Python
3.10–3.13 (Ubuntu) plus Windows and macOS, once more with pyAgrum uninstalled to
keep that dependency genuinely optional, and enforces a 90% overall / 80%
per-module coverage floor.

### Fixed in the underlying analysis code

Six defects found while packaging and while writing the unit tests, each of
which failed silently:

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
- **Resource detection could never match a variable named `model` or
  `tokenizer`.** The patterns required at least one character *before* the word,
  so they matched `my_model` but not `model` or `model_deep` — the names the
  source notebook actually uses. Two agents generating through the same shared
  variable therefore produced *no* common-cause group, which is precisely the
  error that turns a single point of failure into an apparently redundant
  architecture.
- **`source_of_function` did not unwrap its argument.** A partial or Runnable
  wrapper — the usual shape of a LangGraph node payload — yielded no source, and
  so no role hint and no detected resources, while looking like a node that
  simply had none. `_resolve_callable` existed for exactly this and was not
  being called.
- **Calibrating twice compounded.** The union split weighted events by their
  *current* probability, so a second `calibrate()` split already-split values and
  silently moved numbers that were already measured. Events now carry a
  `baseline_prob` recorded once.

Two smaller inconsistencies, also silent:

- `cross_check()` returned three keys without pyAgrum and six with it, so callers
  had to branch on the shape. It now always returns the same keys, with a
  `compared` flag and `agree` as NaN when the comparison could not run.
- `EvidenceCalibrator(bound=...)` was accepted and ignored on the approximate
  (Jeffreys) path, which has only one envelope. It now says so in the evidence
  string it writes.
