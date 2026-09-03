# The codebase

About 10,900 lines of package code in eight modules, plus a vendored copy of
HIP-LLM that is never edited here. This page is the map: what each module owns,
which types cross the boundaries between them, and the order to read them in.

## Read these five, in this order

You do not need all of it to make a change. These five files, about 3,000 lines,
carry every idea in the package:

1. **`architecture/model.py`** - what a component *is*, and how one is read out
   of a LangGraph object.
2. **`faulttree/failure.py`** - what a failure *is*: deviations, Boolean
   expressions, and the archetype library that gives each role its local logic.
3. **`faulttree/synthesis.py`** - the backward traversal that turns those two
   into a tree. This is the heart of HiP-HOPS.
4. **`reliability/calibration.py`** - how a measured interval per component is
   written onto the tree's leaves without changing what the tree says.
5. **`pipeline.py`** - the façade that sequences all of it, and the guards that
   refuse to skip a step.

Everything else is analysis over those structures (`faulttree/analysis.py`),
translation of them (`bayes/`, `faulttree/export.py`, `io/n8n.py`), or drawing
them (`viz/`, `bayes/viz.py`).

## The pipeline, as data

Each arrow is a type, and each type is the contract between two modules. If you
are changing one module, this says what you must keep producing.

```text
  a LangGraph app, mermaid text, a dict spec, or an n8n export
        │
        │  extract_architecture / LangGraphExtractor / load_n8n
        ▼
  SystemModel                    components, connections, resources
        │                        (architecture/model.py)
        │  make_acyclic          loops unrolled + feedback-cut components
        ▼
  SystemModel (acyclic)
        │
        │  annotate_system       one archetype per Role
        ▼
  FailureModel                   ComponentFailureLogic per component,
        │                        BasicEvent registry, CCF groups
        │                        (faulttree/failure.py)
        │
        ├── calibrate_failure_model ◀── ComponentEvidence ◀── outcomes + OperationalProfile
        │      writes measured intervals onto BasicEvent.prob_interval
        │      (reliability/calibration.py, reliability/hipllm.py)
        │
        │  synthesise_fault_tree  backward traversal from a Hazard
        ▼
  FaultTree                      FTNode DAG with shared sub-trees
        │                        (faulttree/synthesis.py)
        │
        ├── analyse_tree ──▶ TreeAnalysis   cut sets, MCUB, importance
        │                    (faulttree/analysis.py)
        │
        │  fault_tree_to_cpts     one CPT per gate
        ▼
  CPTSet ──▶ BayesianNetwork      exact inference, diagnosis
             (bayes/cpt.py, bayes/network.py)
```

`SafetyReport` (`report.py`) bundles the middle of that chain for one system;
`AgenticReliabilityStudy` (`pipeline.py`) drives the whole of it and owns the
state machine.

## Module by module

### `architecture/` - HiP-HOPS Phase 0, "model the architecture"

| File | Owns |
|---|---|
| `model.py` (809) | `Component`, `Connection`, `SystemModel`, `Role`, `extract_architecture`, `parse_mermaid`, role classification, resource detection |
| `extract.py` (122) | `LangGraphExtractor` - the same conventions applied to several graphs |
| `acyclic.py` (359) | `find_cycles`, `is_acyclic`, `make_acyclic`, `CycleReport` |

Three things here are load-bearing and easy to break:

**Router materialisation.** `add_conditional_edges` has no node of its own, so a
routing mistake would have nowhere to live. `materialise_routers=True` (the
default) creates a `<node>::router` component and rewires the edges through it.
The `node -> node::router` feed edge must be emitted whether or not this call
created the router; a pre-declared router that loses it disconnects the graph.

**Resource detection.** Node source is scanned for model and tokenizer
assignments; two components naming the same snapshot become a common-cause
group. The matcher is a name filter, deliberately not a regex requiring
surrounding characters - `model`, `model_deep` and `my_model` must all match, and
an earlier regex that missed the bare names made shared snapshots invisible.

**Loop elimination.** A back edge is *never deleted*. It is replaced by a
feedback-cut pseudo-component that carries iteration-budget and
iteration-latency events, so the loop's contribution stays in the tree. Deleting
it would understate risk silently.

### `faulttree/` - Phases 1 to 3

| File | Owns |
|---|---|
| `failure.py` (860) | `FClass`, `Deviation`, `Expr` (`AND`/`OR`/`BasicEventRef`/`DevRef`/`Const`), `BasicEvent`, `ComponentFailureLogic`, `FailureModel`, `Hazard`, the archetype library, `annotate_system` |
| `synthesis.py` (735) | `FTNode`, `FaultTree`, `synthesise_fault_tree`, `synthesise_all`, `simplify_tree`, `expand_to_tree` |
| `analysis.py` (414) | `cut_sets` (MOCUS with absorption), `quantify` (MCUB), `importance` (Birnbaum, Fussell-Vesely), `fmea_table`, `single_points_of_failure` |
| `export.py` (456) | `to_mermaid`, `to_dot`, `to_json`, `to_openpsa_xml`, `markdown_report` |

The **six failure classes** are the vocabulary everything else is written in:

| | |
|---|---|
| `O` omission | nothing was produced |
| `C` commission | something was produced that should not have been |
| `VC` value-coarse | wrong, and detectable - malformed, truncated |
| `VS` value-subtle | wrong, and plausible - the undetectable case |
| `E` early | too soon |
| `L` late | too late, budget exhausted |

Keeping `VC` apart from `VS` is the single most consequential modelling decision
in the package. A coarse deviation can be caught by a validator downstream; a
subtle one propagates through every component that has no way to detect it, and
that transparency is what makes an apparently-checked pipeline unchecked.

A component's local logic is an **IF-FMEA table**: for each output `Deviation`, a
Boolean `Expr` over input deviations and internal basic events. Synthesis walks
backwards from a hazard, substituting each table and resolving input deviations
across connections, memoised so shared sub-trees stay shared.

### `reliability/` - the numbers

| File | Owns |
|---|---|
| `profile.py` (265) | `OperationalProfile`, `empirical_profile`, `uniform_profile`, `dataset_proportional_profile`, `stratify` |
| `hipllm.py` (137) | the thin adapter onto the vendored HIP-LLM engine |
| `calibration.py` (644) | `ComponentEvidence`, `EvidenceCalibrator`, `CalibrationReport`, `distribute_union` |

Two rules live here and are enforced by tests.

**The union split is exact.** A component's measured failure probability `P` is
distributed over its `w` basic events by `p_i = 1 - (1 - P)^{w_i}`, so that
`1 - Π(1 - p_i) = P` holds to `1e-9`. Splitting proportionally instead would
change the number the measurement produced.

**Calibrating twice must not compound.** The split weights events by their
*prior* probability, so each `BasicEvent` records `baseline_prob` once. Without
it the second `calibrate()` would use the first one's output as weights.

`hipllm.py` is an adapter, not an implementation. The model itself is the
vendored code under `src/HIPLLM/` and `src/hip_llm/`, which is a byte copy and
is never edited here - see [Vendoring](../vendoring.md).

### `bayes/` - the tree as a network

| File | Owns |
|---|---|
| `cpt.py` (651) | `CPT`, `CPTSet`, `GateType`, `CPTBuilder`, `deterministic_gate_cpt`, `k_of_n_cpt`, `noisy_or_cpt`, `fault_tree_to_cpts` |
| `network.py` (608) | `BayesianNetwork`, `Envelope`, `ImpreciseBayesianNetwork`, `exact_top_probability`, `compare_with_cutsets` |
| `learn.py` (276) | `LearnedCPT`, `learn_cpt`, `learn_gate`, `fit_cpts` - tables fitted from outcomes |
| `viz.py` (760) | `BayesNetView` - pyAgrum where Graphviz exists, matplotlib where it does not |

Every probability is computable **twice**: through pyAgrum, and through a NumPy
enumeration that needs no optional dependency. They are cross-checked against
each other to nine significant figures, and CI runs the whole suite once with
pyAgrum uninstalled. Anything added here must work on both paths or say in its
docstring that it does not.

`ImpreciseBayesianNetwork` builds a lower and an upper twin network. That is only
valid because the top event is **monotone** in every basic event, which
`test_network.py::test_monotone_in_every_basic_event` checks. A non-monotone gate
would silently invalidate it, which is why `soft_gates` records a note.

### `io/` - getting systems in

| File | Owns |
|---|---|
| `examples.py` (126) | `EXAMPLES`, `load_example`, `load_outcomes`, `describe_examples` - the three bundled architectures and a synthetic outcome table |
| `n8n.py` (994) | `N8nBlock`, `N8nWorkflow`, `load_n8n`, `n8n_to_spec`, `analyse_n8n`, `n8n_study`, `RULES` |

`n8n.py` is the largest single file and the one most likely to need extending,
because it encodes knowledge about a third-party product that changes. Its
`RULES` tuple is ordered and first-match-wins; each rule carries a `why` string
that reaches the ledger the analyst reads. See
[Analysing an n8n workflow](../tutorials/08-n8n-workflows.md).

### `pipeline.py` and `report.py` - the façades

`SafetyReport` holds one system's trees and analyses and knows how to render and
save them. `AgenticReliabilityStudy` is the object a notebook touches: it owns
the state machine (`analyse` → `observe` → `calibrate` → `run`), the guards that
refuse to run a step out of order, and `run_and_observe`, which invokes a live
graph and scores every node in one call.

The rule `run_and_observe` exists to enforce: a `success` predicate returning
`None` means *not exercised* and a crashed run means *nothing observed*. Both are
missing observations, never failures. Scoring an unreached node as failed blames
it for an upstream fault and inflates the top event - a mistake this package made
once, caught by testing against a replica of a real notebook.

### `viz/plots.py` - drawing

`plot_fault_tree`, `plot_architecture`, `plot_importance`, `plot_cutset_orders`.
All return a matplotlib `ax`; none require Graphviz. Conventional FTA symbols,
transfer triangles for shared sub-trees, and a reserved colour for anything
critical that is always paired with a text tag, so no finding rests on colour
alone.

## Naming

The package is `hiphopsllm`. `src/HIP_HOPS_LLM.py` is a compatibility alias that
registers the submodules in `sys.modules`; it exists so that `from HIP_HOPS_LLM
import ...` keeps working for people who copied it from the original notebook.
Only one of the two spellings can ship, because Windows and macOS filesystems are
case-insensitive.

If you are reading the original Kaggle notebook alongside this code, its
single-cell module was called `hipgraph` and the sections map like this:

| Notebook | Here |
|---|---|
| `hipgraph.arch` | `architecture/model.py` |
| `hipgraph.acyclic` | `architecture/acyclic.py` |
| `hipgraph.failure` | `faulttree/failure.py` |
| `hipgraph.synth` | `faulttree/synthesis.py` |
| `hipgraph.analysis` | `faulttree/analysis.py` |
| `hipgraph.plot` | `viz/plots.py` |
| `hipgraph.api` | `report.py` |

## What is deliberately not here

- **No solver.** Cut sets are computed by MOCUS with absorption, not by handing
  the tree to an external engine. `to_openpsa_xml` exists for when you want one.
- **No inference of the operational profile.** Ever. See
  [Operational profiles](../concepts/operational-profiles.md).
- **No silent fallback.** If a function cannot do what was asked it raises,
  naming what was wrong and what the valid options are.
