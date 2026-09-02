## What this changes

<!-- One or two sentences. If it fixes an issue, "Fixes #123". -->

## Why

<!-- The reasoning, not the diff. What was wrong, or what could not be done
     before. If a reported number changes, say which and by how much. -->

## Checks

- [ ] `pytest` passes, including the vendored HIP-LLM tests
- [ ] `ruff check .` is clean
- [ ] New behaviour has a unit test asserting a *property*, not merely executing a line
- [ ] Coverage is still above the 90% overall / 80% per-module floor
- [ ] `cd docs && make html` builds without warnings (if docs changed)
- [ ] Notebooks regenerated with `python scripts/build_notebooks.py` (if they changed)

## If this touches the invariants

The suite encodes properties the package promises — the two inference engines
agreeing, `MCUB ≥ exact`, cut-set minimality and sufficiency, monotonicity of the
top event, exact union splitting, and the HIP-LLM re-export staying complete.
They are listed in [CONTRIBUTING.md](../CONTRIBUTING.md). If your change makes
one fail, please explain why the property no longer holds rather than relaxing
the test.

- [ ] No invariant was weakened, or the reason one had to be is explained above

## If this touches vendored code

`src/HIPLLM/` and `src/hip_llm/` are byte copies of
[koo-ec/HIP_LLM](https://github.com/koo-ec/HIP_LLM). Fix bugs upstream and
re-vendor — see [docs/source/vendoring.md](../docs/source/vendoring.md).

- [ ] I did not edit the vendored sources
