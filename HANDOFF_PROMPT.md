# Handoff prompt

Paste everything below the line into a fresh chat. It is self-contained: it
tells the assistant what the package is, what it may not get wrong, and the
exact API to reach for.

---

I am working with **HIP-HOPS-LLM**, a public pip-installable Python package that
joins HiP-HOPS compositional safety analysis to the HIP-LLM hierarchical
imprecise-probability reliability model, for LLM-based agentic systems.

- Repository: https://github.com/koo-ec/HIP_HOPS_LLM (MIT, Python 3.10-3.13)
- Documentation: http://koorosh-aslansefat.com/HIP_HOPS_LLM/
- Import name `hiphopsllm` (alias `HIP_HOPS_LLM`), version 0.1.0, 107 public names
- Install: `pip install "HIP-HOPS-LLM[bayes,graph] @ git+https://github.com/koo-ec/HIP_HOPS_LLM.git"`
  (extras: `bayes` = pyAgrum, `graph` = LangGraph, plus `test`, `live`, `docs`,
  `dev`, `all`)
- In a hosted notebook, clone to a directory that is **not** an importable name
  (`hiphopsllm-repo`) and install non-editable: a running kernel never processes
  the `.pth` that `pip install -e` writes, and a folder named `hiphopsllm` beside
  the notebook shadows the package as an empty namespace package.

## What it does

Give it a LangGraph application and observed per-node outcomes. It returns
synthesised fault trees, minimal cut sets, an FMEA, and a Bayesian network whose
top-event probability is an **interval** derived from measurement.

```
LangGraph app ──▶ architecture ──▶ fault trees ──▶ cut sets, FMEA
                                        ▲                │
 outcomes + operational profile ────────┘                ▼
        (HIP-LLM imprecise posterior)              CPTs ──▶ Bayesian network
```

HiP-HOPS supplies the structure: each component carries local failure logic, and
trees, cut sets and the FMEA are synthesised from the architecture, so the
analysis cannot drift from the code. HIP-LLM (Aghazadeh-Chakherlou et al., *RESS*
272 (2026) 112615, doi:10.1016/j.ress.2026.112615) supplies the numbers: a
posterior **envelope** for the probability that a component fails on the next
task drawn from an explicit operational profile.

## The whole thing in ten lines

```python
from hiphopsllm import AgenticReliabilityStudy, load_example, load_outcomes

study = AgenticReliabilityStudy(load_example("parallel_aggregator"))
study.observe(load_outcomes(), profile={"short": 0.3, "medium": 0.5, "long": 0.2})
study.run()
print(study.summary())
study.bayesnet("H2").show()
```

Against a live graph, measuring as it goes:

```python
study = AgenticReliabilityStudy(graph, name="...", globals_ns=globals(),
                                resource_overrides={"agent": {"llm": "gpt-4o-mini"}})
outcomes = study.run_and_observe(
    inputs=[...],
    stratum=lambda item: "complex" if ... else "simple",
    success={"node_a": lambda s: s.get("a_ok"),
             "node_b": lambda s: s["b_ok"] if "b_ok" in s else None},
    profile={"simple": 0.35, "complex": 0.65},
    invoke=graph.invoke,
)
print(study.summary())
```

## Rules the package enforces, which I do not want quietly broken

1. **The claim is "this component fails with this probability under this
   operational profile."** Not hallucination rate, not benchmark accuracy. The
   quantity measured is task failure, whatever its mode. The fault tree then
   decomposes that measured probability over the component's failure modes so it
   can be propagated; that split is a *modelling step*, not the measurement.
2. **The operational profile is never inferred.** `calibrate()` refuses without
   one. If `observe()` is called without a profile it falls back to
   `dataset_proportional_profile` and prints `NO OPERATIONAL PROFILE GIVEN`,
   because "my test set looks like production" is a claim, not a measurement
   (HIP-LLM Remark 7). A stratum here is a HIP-LLM **subdomain**; the package
   uses the flat single-domain case, so `{label: weight}` is `Omega`.
3. **A `success` predicate returning `None` means "not exercised"** (a router
   sent the run to `END` first) and is a *missing observation*, never a failure.
   Same for a run that crashed: `on_error` is `"skip"` or `"record"`, never
   scoring a node 0. Scoring an unreached node as failed blames it for an
   upstream fault and inflates the top event.
4. **Reliability is `R(n) = P(failure-free over n future tasks)`**, computed as
   `E[p^n]` per hyperparameter configuration and *then* enveloped, never
   `E[p]^n` (Jensen).
5. Intervals are **posterior envelopes**, not confidence intervals: they span the
   admissible hyperparameter set as well as sampling uncertainty.
6. Optional imports (`pyagrum`, `langgraph`, `IPython`, `openai`) must never sit
   at a notebook cell's top level; a test AST-parses every cell and fails if they
   do. `numpy`, `pandas`, `matplotlib`, `scipy` are hard dependencies and fine.

## API surface I use most

**Pipeline** - `AgenticReliabilityStudy` (`.analyse() .observe() .run_and_observe()
.calibrate() .run() .summary() .operational_reliability(n_tasks) .cut_sets(h)
.fmea() .single_points() .hazards_found() .hazard_probability(h) .bayesnet(h)
.imprecise_bayesnet() .cpts(h) .plot(h) .plot_architecture() .plot_importance(h)
.plot_cutset_orders() .save(dir)`), `SafetyReport` (`.tree(h) .analysis(h)
.markdown() .mermaid(h)`), `analyse_langgraph`, `StudyNotReady`.

**Architecture** - `extract_architecture`, `LangGraphExtractor`, `SystemModel`
(`.components .connections .common_cause_groups()`), `Component` (`.id .role
.resources .source_code`), `Role`, `parse_mermaid`, `find_cycles`, `make_acyclic`.
Conditional edges are materialised as `<node>::router` components with their own
failure logic. Shared model snapshots are detected from node source and become
common-cause groups.

**Fault trees** - `synthesise_fault_tree`, `synthesise_all`, `analyse_tree`,
`cut_sets`, `quantify`, `importance`, `fmea_table`, `single_points_of_failure`,
`FaultTree` (`.size() .depth() .basic_event_ids() .warnings`), `TreeAnalysis`
(`.cuts.sets .cuts.symbols .quant.top_probability .quant.top_interval
.quant.cut_set_probability .importance .single_points`), `BasicEvent`.
Exports: `to_mermaid`, `to_dot`, `to_json`, `to_openpsa_xml`, `markdown_report`.
Failure classes: `O` omission, `C` commission, `VC` value-coarse (detectable),
`VS` value-subtle (plausible, undetectable), `E` early, `L` late.
Default hazards: H1 no answer, H2 incorrect answer accepted (critical), H3
malformed, H4 too late, H5-<node> unsafe execution.

**Reliability** - `OperationalProfile` (`.summary() .provenance .expected()
.restricted_to() .to_hipllm()`), `empirical_profile`, `uniform_profile`,
`dataset_proportional_profile`, `stratify`, `decomposition_stratum`,
`load_strategyqa`, `EvidenceCalibrator`, `ComponentEvidence` (`.interval
.by_stratum .n_trials .n_failures .reliability(n) .statement(n)`),
`OperationalFailureProb`, `quick_inference_settings`, `paper_inference_settings`.

**Bayesian networks** - `fault_tree_to_bayesnet`, `fault_tree_to_cpts`,
`BayesianNetwork` (`.summary() .show() .view(evidence=) .posteriors(evidence)
.cross_check() .compare_with_cutsets(analysis) .net`), `ImpreciseBayesianNetwork`,
`BayesNetView`, `noisy_or_cpt`, `k_of_n_cpt`, `learn_cpt`, `fit_cpts`,
`graphviz_available`. Exact inference runs through pyAgrum *and* a NumPy
enumeration, cross-checked to ~1e-16.

**Plots** - `plot_fault_tree`, `plot_architecture`, `plot_importance`,
`plot_cutset_orders`, `display_fault_tree`. All return an `ax`; all have a
matplotlib path that works without Graphviz.

**Getting systems in** - `load_example`, `load_outcomes` for the three bundled
architectures, and `load_n8n` / `n8n_to_spec` / `analyse_n8n` / `n8n_study` for
an n8n JSON export. The n8n path folds a model sub-node into `resources["llm"]`
(so two agents on one model are a common-cause group), reverses `ai_tool` edges
into an invocation plus a cut loop, and adds a commission hazard for any tool
that acts on the world. `N8nWorkflow.ledger_frame()` shows every decision with
its reason.

## Notebooks (generated, never hand-edited)

`scripts/build_notebooks.py` generates all three from Python cell lists in
`scripts/notebook_colab.py` and `scripts/notebook_kaggle_addon.py`. A test
regenerates them and fails if the committed files differ, and another executes
every code cell. **Edit the generator, then run the generator.**

- `notebooks/HIP_HOPS_LLM_Colab.ipynb` - real LangGraph + `ChatOpenAI` +
  `OPENAI_API_KEY`, degrading to bundled outcomes without a key.
- `notebooks/hip_hops_for_agentic_ai.ipynb` - two architectures, synthesis, cut
  sets, FMEA, Bayesian network, calibration. Kaggle:
  `kooaslansefat/hip-hops-for-agentic-ai`.
- `notebooks/hiphopsllm_two_cell_addon.ipynb` - the one to show people. 11 steps;
  Steps 1-6 stand in for *your* existing LangGraph notebook, Step 7 (`Cell A`,
  `run_and_observe`) and Step 9 (`Cell B`, the Bayesian network) are the two
  cells you actually append. Steps 5, 8 and 10 show the working: the LangGraph
  drawing, the synthesised fault tree with its minimal cut sets and
  Fussell-Vesely ranking, and the exact-vs-bound cross-check. Kaggle:
  `kooaslansefat/hiphopsllm-two-cell-reliability-addon`.

House style in that notebook: styled `<div>` headings (hosted notebooks sanitise
most other HTML), an anchored Markdown contents table, APA references, no
em-dashes.

## State

736 tests, 94% line coverage, CI floors at 90% overall and 80% per module. CI
runs Python 3.10-3.13 on Ubuntu plus Windows and macOS, and once more with
pyAgrum uninstalled so that dependency stays genuinely optional.

## What I want you to do

<state your task here>
