# Vendored HIP-LLM tests

These are HIP-LLM's own tests, copied verbatim alongside the vendored sources in
`src/HIPLLM/` and `src/hip_llm/`. They run here so that a change to the vendored
copy — an accidental edit, a bad merge from upstream — fails loudly rather than
silently changing a reliability number.

They need the `configs/`, `numerics/` and `data/reference/` directories, which
are vendored for the same reason.

## What is not here

`test_api_contracts.py` is **not** vendored. It is a live-API contract suite: by
design, a missing provider key is a FAILURE and not a skip, because its job is to
gate HIP-LLM's own releases against provider drift. That is HIP-LLM's release
gate, not this package's, and running it here would mean every CI run fails
without OpenAI and Anthropic credentials.

`test_notebook.py`, `test_strategyqa_colab_notebook.py`, `test_reproducibility.py`
and `test_source_inconsistencies.py` are likewise upstream-repository concerns
(they execute HIP-LLM's own notebooks and check its published numerics), so they
stay upstream too.

If you are changing the vendored sources, run HIP-LLM's full suite in its own
repository as well: <https://github.com/koo-ec/HIP_LLM>.
