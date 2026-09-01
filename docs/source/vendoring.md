# Vendoring HIP-LLM

HIP-LLM's sources are **copied into this repository**, not depended on. This page
records what, from where, and why.

## What is vendored

| Path | Origin | Contents |
|---|---|---|
| `src/HIPLLM/` | [`koo-ec/HIP_LLM`](https://github.com/koo-ec/HIP_LLM) `src/HIPLLM/` | The high-level API: `FailureProb`, `OperationalFailureProb`, the StrategyQA loader |
| `src/hip_llm/` | same repository, `src/hip_llm/` | The replication engine: hyperposteriors, imprecise envelopes, reliability transforms, baselines, benchmark evaluation, plotting, schemas |
| `tests/vendor/` | same repository, `tests/` | HIP-LLM's own tests, minus the live-API suite — see `tests/vendor/README.md` |
| `configs/`, `numerics/`, `data/reference/` | same repository | The configuration and reference numerics those tests need |

The upstream commit the copy was taken from is recorded in
`src/hip_llm/_VENDOR_COMMIT`.

```bash
cat src/hip_llm/_VENDOR_COMMIT
```

## Why vendored rather than depended on

HIP-LLM is not published on PyPI, so a dependency would have to be a git URL —
which does not survive a PyPI release of this package, does not pin reproducibly
without a commit hash, and adds a network fetch to every install, including in
Colab and Kaggle sessions where that is the slowest step.

Vendoring makes this repository a single reproducible artefact: clone it, install
it, and every number in the documentation is computable offline.

The cost is drift. Two things hold it down:

**HIP-LLM's own tests run here.** All 157 of them, against the vendored copy. An
accidental edit or a bad merge fails loudly rather than silently changing a
reliability number.

**The re-export is asserted complete.** A test walks `HIPLLM.__all__` and
`hip_llm.__all__` and fails if any symbol is not reachable through
`HIP_HOPS_LLM.reliability.hipllm`, so an upstream addition that is not carried
over is caught rather than quietly missing.

## Licence

HIP-LLM is MIT. Its notice is retained verbatim as `LICENSE.HIPLLM` in the
repository root, and both licence files are declared in `pyproject.toml`. This
package is also MIT, so the combination is unambiguous.

Authorship of the vendored code is unchanged: Robab Aghazadeh Chakherlou, Qing
Guo, Siddartha Khastgir, Peter Popov, Xiaoge Zhang and Xingyu Zhao. The paper it
implements is

> R. Aghazadeh-Chakherlou, Q. Guo, S. Khastgir, P. Popov, X. Zhang and X. Zhao,
> "A hierarchical imprecise probability approach to reliability assessment of
> large language models", *Reliability Engineering & System Safety* **272**
> (2026) 112615. <https://doi.org/10.1016/j.ress.2026.112615>

Please cite it alongside this package.

## Importing

Both packages remain importable under their own names, so existing code keeps
working:

```python
from HIPLLM import OperationalFailureProb      # unchanged
import hip_llm                                  # unchanged
```

and both are reachable through this package's namespace:

```python
from HIP_HOPS_LLM import OperationalFailureProb, quick_inference_settings
```

One name is deliberately not passed through unchanged. HIP-LLM's
`OperationalProfile` is a schema with parallel `labels`/`weights` arrays; this
package's `OperationalProfile` is a mapping-shaped class that converts to it. The
HIP-LLM one is available as `HIPLLMOperationalProfile`.

## Updating the vendored copy

```bash
git clone --depth=1 https://github.com/koo-ec/HIP_LLM /tmp/hipllm
rm -rf src/HIPLLM src/hip_llm
cp -r /tmp/hipllm/src/HIPLLM /tmp/hipllm/src/hip_llm src/
cp -r /tmp/hipllm/configs /tmp/hipllm/numerics .
cp /tmp/hipllm/tests/test_*.py tests/vendor/          # then re-remove the live-API suite
git -C /tmp/hipllm rev-parse HEAD > src/hip_llm/_VENDOR_COMMIT
pytest
```

`pytest` is the acceptance gate. If HIP-LLM's own tests still pass and the
re-export completeness test still passes, the update is safe; if the second fails
it will name the symbols to add to `src/HIP_HOPS_LLM/reliability/hipllm.py`.
