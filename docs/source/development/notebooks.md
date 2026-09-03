# Notebooks

**The `.ipynb` files are generated. Never edit one by hand** - a test regenerates
them and fails if the committed file differs.

```bash
python scripts/build_notebooks.py
```

## Why they are generated

A notebook diff is mostly JSON noise, so a real change is invisible in review.
Worse, a stale output baked into a committed notebook is indistinguishable from a
fresh one, which is precisely the failure mode this package exists to prevent
elsewhere. So the cells are written in plain Python, the notebooks carry no
output at all, and the generator is the source of truth.

| Generator | Produces |
|---|---|
| `scripts/build_notebooks.py` | the `md()` / `code()` builders, the cell ids, and the Kaggle notebook's cell list inline |
| `scripts/notebook_colab.py` | `notebooks/HIP_HOPS_LLM_Colab.ipynb` |
| `scripts/notebook_kaggle_addon.py` | `notebooks/hiphopsllm_two_cell_addon.ipynb` |

Cell ids come from a module-level counter, so regenerating is deterministic and
produces no spurious diff. Adding a cell to one notebook shifts the ids of those
generated after it; that is expected and harmless.

## The three notebooks

**`HIP_HOPS_LLM_Colab.ipynb`** builds a real LangGraph application with
`ChatOpenAI`, runs it against OpenAI, and analyses the outcomes. Without an
`OPENAI_API_KEY` every LLM step is skipped and bundled outcomes are used instead,
so it runs end to end in CI and for anyone just reading it.

**`hip_hops_for_agentic_ai.ipynb`** is the notebook this package grew out of,
rewritten to install the package rather than `%%writefile` a 4,800-line module
into one cell. Two architectures, synthesis, cut sets, FMEA, Bayesian network,
calibration.

**`hiphopsllm_two_cell_addon.ipynb`** is the one to show people. Eleven steps:
1-6 stand in for a LangGraph notebook the reader already has, Step 7 (`Cell A`)
and Step 9 (`Cell B`) are the two cells they actually append, and Steps 5, 8 and
10 show the working - the graph drawing, the synthesised fault tree with its cut
sets, and the exact-versus-bound cross-check.

## Rules a cell must follow

**Optional imports go inside a `try` or an `if`.** `pyagrum`, `langgraph`,
`langchain_openai`, `langchain_core`, `openai`, `IPython`, `google.colab` and
`typing_extensions` may all be missing in a reader's environment.
`test_notebook_imports.py` AST-parses each cell and fails on a bare top-level
import of any of them. `numpy`, `pandas`, `matplotlib` and `scipy` are hard
dependencies and may be imported freely.

```python
# wrong: breaks every CI job and any reader without the package
from IPython.display import Image
display(Image(graph.get_graph().draw_mermaid_png()))

# right: states the fallback
try:
    from IPython.display import Image

    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as exc:
    print(f"renderer unavailable ({type(exc).__name__}); the drawing below is the same graph")
```

**Install with a plain `pip install`, not `-e`.** `pip install -e` only writes a
`.pth` file, and an already-started kernel never processes one, so the package
would not actually be importable in that session.

**Clone into a directory that is not an importable name.** A folder called
`hiphopsllm` beside the notebook is found before site-packages and imported as an
empty namespace package - the import succeeds and the module has no attributes.
Every notebook uses `hiphopsllm-repo`.

**Every cell must run headlessly.** `test_notebooks.py` executes each cell
in-process with `display` stubbed out and no API keys, skipping only `!` and `%`
lines. A cell that needs a key must degrade, not fail.

**Presentation.** Hosted notebooks render styled `<div>` blocks but sanitise much
other HTML, so headings are divs and the contents table is ordinary anchored
Markdown. No em-dashes in prose.

## Changing one

```bash
$EDITOR scripts/notebook_kaggle_addon.py
python scripts/build_notebooks.py
pytest tests/integration/test_notebooks.py tests/integration/test_notebook_imports.py -q
git add scripts/ notebooks/
```

Commit the regenerated `.ipynb` alongside the generator change. The
up-to-date test will fail otherwise, and so will anyone who runs the generator
next.

Before claiming a notebook works, run it and read the output. The cell-execution
test proves nothing raised; it does not prove the numbers say what the prose
around them claims. A `bound_overestimate` of exactly zero, for instance, quietly
contradicts a sentence about the bound over-counting.

## Publishing

Colab reads straight from GitHub, so pushing is publishing:

```text
https://colab.research.google.com/github/koo-ec/HIP_HOPS_LLM/blob/main/notebooks/HIP_HOPS_LLM_Colab.ipynb
```

Kaggle does not. It needs an explicit push through the Kaggle CLI, with a
`kernel-metadata.json` beside a copy of the notebook renamed to match the
kernel slug:

```bash
kaggle kernels push          # from the directory holding kernel-metadata.json
kaggle kernels status <owner>/<slug>
kaggle kernels pull <owner>/<slug> -p ./verify
```

**Push after every change, and verify by pulling it back.** A repository that is
correct and a Kaggle kernel that is three versions behind look identical from
here, and the reader sees only the stale one.
