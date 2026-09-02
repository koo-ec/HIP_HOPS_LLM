# Installation

```bash
pip install git+https://github.com/koo-ec/HIP_HOPS_LLM.git
```

Python 3.10 or newer.

## Extras

```bash
pip install "HIP-HOPS-LLM[all] @ git+https://github.com/koo-ec/HIP_HOPS_LLM.git"
```

| Extra | Adds | For |
|---|---|---|
| `bayes` | `pyagrum` | Junction-tree inference and pyAgrum drawing. **Optional** — every quantity is also computable by exact enumeration in NumPy. |
| `graph` | `langgraph`, `langchain-core` | Reading a live compiled LangGraph. Not needed for mermaid text or specification dicts. |
| `test` | `pytest`, `pyagrum`, `nbformat` | Running the suite. |
| `docs` | `sphinx`, `furo`, `myst-parser`, … | Building these pages. |
| `live` | `openai`, `anthropic`, `datasets` | HIP-LLM's live-provider paths. |
| `all` | everything above | |

The core needs only NumPy, pandas, matplotlib, SciPy and PyYAML.

## Graphviz

pyAgrum draws through Graphviz, which is a separate **native binary**, not a
Python package. `pip install pydot` does not provide it.

```python
from hiphopsllm import graphviz_available

graphviz_available()      # runs `dot -V`, so it tells you the truth
```

If it returns `False`, nothing breaks: `bn.show()` draws the same network with
matplotlib and says so in the caption. Install Graphviz only if you want
pyAgrum's richer rendering.

- Ubuntu/Debian: `sudo apt-get install graphviz`
- macOS: `brew install graphviz`
- Windows: `winget install graphviz` (then reopen your terminal)
- Conda: `conda install -c conda-forge graphviz`

## Development install

```bash
git clone https://github.com/koo-ec/HIP_HOPS_LLM.git
cd HIP_HOPS_LLM
pip install -e ".[all]"
pytest
```

The suite includes HIP-LLM's own tests, run against the vendored copy — see
[Vendoring](vendoring.md).

## Colab

```text
!git clone --depth=1 https://github.com/koo-ec/HIP_HOPS_LLM.git hiphopsllm-repo
%pip install -q -e "hiphopsllm-repo[bayes,graph]"
```

:::{warning}
Do not let the clone directory share a name with an importable module. In a
notebook the working directory precedes `site-packages` on `sys.path`, so a
directory called `hiphopsllm` (or `HIP_HOPS_LLM`, if you use the alias) is found
first and imported as an **empty namespace package**. The import then succeeds
and the module has no attributes at all:

```text
AttributeError: module 'HIP_HOPS_LLM' has no attribute '__version__'
```

Cloning into `hiphopsllm-repo`, as above, avoids it entirely.
:::

`[all]` also works, but it pulls the docs and release toolchains into the
runtime, and on Colab that upgrades `rich` past what some preinstalled packages
pin. `[bayes,graph]` is everything the notebook actually uses.

`notebooks/HIP_HOPS_LLM_Colab.ipynb` in the repository does this and runs the
whole pipeline; open it directly from GitHub in Colab.
