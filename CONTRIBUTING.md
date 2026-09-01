# Contributing

## Setup

```bash
git clone https://github.com/koo-ec/HIP_HOPS_LLM.git
cd HIP_HOPS_LLM
pip install -e ".[all]"
pytest
```

The suite should be green before you change anything. It includes HIP-LLM's own
157 tests, run against the vendored copy.

## The invariants

Several tests encode properties the package promises. If you change gate
semantics, the synthesiser, or the CPT conversion, these are what will catch a
mistake — please do not weaken them to make a change pass.

| Test | Property |
|---|---|
| `test_network.py::test_exact_and_pyagrum_agree` | Two independent inference paths agree to nine significant figures |
| `test_pipeline.py::TestEveryExample` | On every hazard of every bundled example, `MCUB ≥ exact` |
| `test_faulttree.py::test_every_cut_set_actually_causes_the_top_event` | Cut sets are sufficient, verified by Boolean evaluation |
| `test_faulttree.py::test_removing_any_event_breaks_a_cut_set` | Cut sets are minimal |
| `test_faulttree.py::test_simplification_preserves_the_cut_sets` | Tree reduction does not change the Boolean function |
| `test_network.py::test_monotone_in_every_basic_event` | The top event is monotone — this is what licenses the interval arithmetic |
| `test_reliability.py::test_the_union_of_calibrated_events_matches_the_measurement` | Union splitting is exact to `1e-9` |
| `test_public_api.py::TestHIPLLMIsComplete` | Every HIP-LLM symbol stays reachable |
| `test_learn.py::TestSplitGuard` | CPTs are never fitted on a test split |

## House rules

**Nothing fails silently.** This package exists partly because a Bayesian network
running on default priors looks exactly like one running on evidence. If a
function cannot do what was asked, it raises with a message naming what was wrong
and what the valid options are — it does not fall back quietly. If a value is a
placeholder, its `evidence` string says so.

**Provenance travels with numbers.** Every `BasicEvent` carries an `evidence`
string; every `OperationalProfile` carries a `provenance`; every
`ComponentEvidence` carries its `method`, `n` and interval. A number that reaches
a report without them is a bug.

**pyAgrum stays optional.** Anything you add to `bayes/` must also work through
`engine="exact"`, or be clearly documented as pyAgrum-only. Graphviz likewise:
`bn.show()` must always produce a picture.

**Vendored code is not edited.** `src/HIPLLM/` and `src/hip_llm/` are byte
copies. Fix bugs upstream in [`koo-ec/HIP_LLM`](https://github.com/koo-ec/HIP_LLM)
and re-vendor — see [`docs/source/vendoring.md`](docs/source/vendoring.md). They
are excluded from ruff for the same reason.

## Adding a component archetype

1. Write the annotation builder in `src/HIP_HOPS_LLM/faulttree/failure.py`,
   giving each basic event a `rationale` and a `mitigation` appropriate to its
   *class* — the FMEA is generated from those.
2. Add the role to `Role` in `architecture/model.py` and teach `classify_role`
   to recognise it.
3. Add default probabilities to `DEFAULT_P`, and say in the `evidence` string
   that they are placeholders.
4. Test that the archetype's logic produces the cut sets you expect on a small
   hand-checkable graph.

## Adding a gate type

1. Add a table constructor to `bayes/cpt.py` next to `k_of_n_cpt`, validating its
   fan-in through `_check_fan_in`.
2. Add it to `GateType` and to `CPTBuilder._gate_table`.
3. **Check coherence.** If the new gate is not monotone, `ImpreciseBayesianNetwork`
   is no longer valid for trees using it — say so in the docstring and in
   `cpts.notes`, as `soft_gates` does.
4. Add a unit test comparing it against a hand-computed table.

## Documentation

Every number in `docs/` was produced by running the code. If you change something
that appears in a code block's output, re-run it and paste the new output rather
than editing it by hand.

```bash
cd docs && make html
```

## Style

`ruff check .` and `ruff format --diff .`; line length 100. British spelling in
prose, US spelling where it is an API name (`analyze` never appears; `analyse`
does).

Docstrings explain *why*, not what the signature already says. A comment that
restates the line below it is noise; a comment recording a decision — why the
back edge is cut rather than deleted, why the split guard raises — is the reason
anyone can safely change this code later.
