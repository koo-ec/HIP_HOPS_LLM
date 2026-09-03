# Testing

736 tests, 94% line coverage. The suite is the specification:
several tests encode properties the package *promises*, and weakening one to make
a change pass is how a silent defect gets in.

```bash
pip install -e ".[all]"
pytest                                        # about four minutes
pytest --cov=hiphopsllm --cov-report=term-missing
```

## Layout

| Directory | Count | What it covers |
|---|---|---|
| `tests/unit/` | 15 files | One module each, plus `*_internals.py` files for the parts a public API does not reach |
| `tests/integration/` | 4 files | The pipeline end to end, the public API surface, and the notebooks |
| `tests/vendor/` | 9 files, 157 tests | HIP-LLM's own suite, run against the vendored copy |

`tests/vendor/` exists so that re-vendoring cannot quietly change the reliability
model. It is not our code and its tests are not ours to relax.

## Fixtures

`tests/conftest.py` provides session-scoped fixtures so the expensive steps
happen once:

| Fixture | Is |
|---|---|
| `parallel_spec`, `react_spec`, `supervisor_spec` | the three bundled architectures |
| `outcomes` | the bundled synthetic outcome table |
| `study` | an **analysed, uncalibrated** study of `parallel_aggregator` |
| `calibrated` | the same study, calibrated from `outcomes` |

Use `study` when you need structure and `calibrated` when you need numbers.
Building a fresh `AgenticReliabilityStudy` in a test is fine and often clearer -
`test_pipeline_unit.py` does it for anything that mutates state.

## The invariant tests

These are the ones to leave alone. If you change gate semantics, the synthesiser
or the CPT conversion, this is what catches the mistake.

| Test | Property |
|---|---|
| `test_network.py::test_exact_and_pyagrum_agree` | Two independent inference paths agree to nine significant figures |
| `test_network.py::test_monotone_in_every_basic_event` | The top event is monotone - this is what licenses `ImpreciseBayesianNetwork` |
| `test_faulttree.py::test_every_cut_set_actually_causes_the_top_event` | Cut sets are sufficient, verified by Boolean evaluation |
| `test_faulttree.py::test_removing_any_event_breaks_a_cut_set` | Cut sets are minimal |
| `test_faulttree.py::test_simplification_preserves_the_cut_sets` | Tree reduction does not change the Boolean function |
| `test_pipeline.py::TestEveryExample` | On every hazard of every bundled example, `MCUB ≥ exact` |
| `test_reliability.py::test_the_union_of_calibrated_events_matches_the_measurement` | Union splitting is exact to `1e-9` |
| `test_pipeline_unit.py::test_recalibrating_does_not_compound` | Calibrating twice lands on the same numbers |
| `test_learn.py::TestSplitGuard` | CPTs are never fitted on a test split |
| `test_architecture_internals.py::test_two_nodes_generating_through_the_same_variable_share_a_cause` | A snapshot shared through a variable is still a common-cause group |
| `test_public_api.py::TestHIPLLMIsComplete` | Every HIP-LLM symbol stays reachable through this package |

Two of them exist because the bug happened. The shared-variable test was written
after a resource-detection regex was found never to match a variable simply
called `model` - which made every shared snapshot invisible and turned a single
point of failure into an apparently redundant architecture. The recalibration
test was written after `calibrate()` twice was found to compound.

## Tests that police the notebooks

| Test | Refuses |
|---|---|
| `test_notebooks.py::test_every_code_cell_executes` | A cell that raises |
| `test_notebooks.py::test_it_is_valid_and_has_no_baked_output` | Committed output, or a cell without a stable id |
| `test_notebooks.py::test_the_notebooks_are_up_to_date_with_their_generator` | A notebook that differs from what the generator produces |
| `test_notebooks.py::test_hosted_notebooks_use_an_immediately_importable_install` | `pip install -e`, which a running kernel never picks up |
| `test_notebook_imports.py::test_optional_imports_are_guarded` | An optional package imported at a cell's top level |

The last one AST-parses every cell and looks at where the import sits in the
tree. It exists because IPython, langgraph and pyagrum each broke every CI job in
turn by appearing as a bare `import` at the top of a cell. See
[Notebooks](notebooks.md).

## Coverage floors

CI enforces **90% overall** and **80% for any module over 20 statements**. They
are floors, not targets: they exist so a module added without tests fails there
rather than quietly dragging the number down.

Coverage is not the goal. A test that executes a line without asserting anything
about it is worse than no test, because it makes the number lie. Prefer asserting
a property - cut sets are minimal, the union rule is exact, the two inference
engines agree - over touching a branch.

## Running a subset

```bash
pytest tests/unit/test_faulttree.py -q             # one module
pytest -k "cut_set and not viz" -q                 # by name
pytest tests/unit/test_viz_plots.py::TestPlotArchitecture -q
pytest -x -q                                       # stop at the first failure
pytest --lf                                        # only what failed last time
pytest tests/vendor -q                             # HIP-LLM's own suite
```

Plot tests need a non-interactive backend. Every test module that draws sets it
at import time, before pyplot is imported:

```python
import matplotlib
matplotlib.use("Agg")
```

## Writing a test that is worth having

Name it after the property, not the function: `test_a_crashed_run_never_blames_a_node`
says what breaks if it fails; `test_run_and_observe_3` does not.

Assert on something you worked out yourself. Comparing one implementation against
another catches a typo but not a misunderstanding, and a misunderstanding is what
produces a confident wrong number.

Write the contrast case. `test_run_and_observe.py` has
`test_none_is_a_missing_observation_not_a_failure` immediately followed by
`test_scoring_an_unreached_node_as_failed_would_be_worse`, so the next reader can
see not just what the code does but which alternative was rejected and why.

Test the message, not only the raise. Most of this package's guards exist so a
user is told what to do next, and `pytest.raises(..., match="never inferred silently")`
keeps that sentence from being deleted as decoration.
