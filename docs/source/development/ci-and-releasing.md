# CI, docs and releasing

Three workflows. None of them is optional reading if you are changing the build,
the dependencies, or anything that touches the vendored code.

## `ci.yml` - seven jobs

Runs on every push to `main`, every pull request, and on demand.

| Job | What it protects |
|---|---|
| **test** | The suite on Python 3.10-3.13 (Ubuntu), plus one Windows and one macOS run on 3.12. Those two exist because the Graphviz fallback and `importlib.resources` data loading are the platform-sensitive parts. |
| **no-pyagrum** | The suite again with pyAgrum uninstalled, *and verified absent*. This job is what makes "pyAgrum is optional" true rather than aspirational: every probability must still be computable by exact enumeration. |
| **coverage** | `--cov-fail-under=90` overall, then a second pass that reads `cov.json` and fails if any module of more than 20 statements is below 80%. |
| **examples** | Every script in `examples/` runs to completion. Documentation that does not execute is worse than none. |
| **lint** | `ruff check` and `ruff format --diff`. Line length 100. |
| **build** | `python -m build`, so a packaging mistake is caught before a tag, not after. |

Jobs set `MPLBACKEND=Agg` and `PYTHONIOENCODING=utf-8`. The second matters on
Windows, where the default console encoding cannot represent the arrows and
box-drawing characters the reports use.

To reproduce a CI failure locally:

```bash
MPLBACKEND=Agg PYTHONIOENCODING=utf-8 pytest -q --timeout=900
pip uninstall -y pyagrum && pytest -q          # the no-pyagrum job
ruff check . && ruff format --diff .
```

## `docs.yml` - the site

Builds Sphinx on every push to `main` and deploys to GitHub Pages at
<http://koorosh-aslansefat.com/HIP_HOPS_LLM/>. It also re-runs the documented
examples, so a code block whose output has drifted fails the build rather than
misleading a reader.

```bash
pip install -e ".[docs]"
cd docs && make html          # or: python -m sphinx -b html -W docs/source _build
```

`-W` turns warnings into errors, which is what CI uses. A broken cross-reference
is a broken promise to the reader, so it fails the build.

One exception, in `conf.py`: an **unreachable intersphinx inventory** is filtered
out. That warning means another project's documentation site is down, not that
anything here is wrong, and under `-W` it failed the build and let the published
site go stale - which is what happened when `docs.scipy.org` timed out. The
warning is untyped, so `suppress_warnings` cannot reach it and it is filtered on
the logger that emits it. The filter matches that one message only; a missing
module or a broken link still fails the build.

Numbers in documented code output are produced by running the code. If you change something
that appears in a code block's output, re-run it and paste the new output rather
than editing it by hand.

### Appearance and Ask AI

The documentation retains Sphinx and Furo, with the teal palette, Inter text,
JetBrains Mono code and horizontal navigation used on
[Koorosh Aslansefat's personal website](https://koorosh-aslansefat.com/).
Light and dark colours are defined in `conf.py` and `_static/custom.css`.
Fonts load from Google Fonts, with system-font fallbacks.

`_templates/page.html` adds the shared header and navigation. Links use Sphinx's
`pathto` helper so nested documentation pages and Read the Docs version prefixes
resolve correctly. The homepage diagram is the supplied conceptual illustration
at `_static/system-safety-workflow.png`; its percentages are labelled as
illustrative, rather than package output.

The **Ask AI** button opens a chooser for ChatGPT, Claude, Gemini, Perplexity and
Microsoft Copilot. Provider links live in `_templates/ask-ai.html`.
`_static/ask-ai.js` builds a prompt from the current page title, documentation URL
and repository URL. Choosing a provider copies that prompt and opens the service
in a new tab; the reader pastes it to begin. The prompt is also available to view
and copy manually if clipboard access is unavailable. No API key, embedded model
or backend is required, and opening the chooser sends no request to an AI service.

Hosted pages preserve their current documentation version in the prompt. Local
previews use `public_docs_url` in `conf.py` so an assistant is never given a
localhost URL. If the public documentation moves, update that setting. After
changing the chooser, check keyboard opening and closing, clipboard fallback,
mobile layout, all five destinations and a nested documentation page.

## `publish.yml` - PyPI

Triggered by **publishing a GitHub release** (not by pushing a tag), or manually
through `workflow_dispatch`. It builds the distribution, uploads it as an
artifact, then publishes through `pypa/gh-action-pypi-publish` from the `pypi`
environment using trusted publishing - there is no API token in the repository.

The release checklist:

1. Bump `version` in `pyproject.toml` **and** in `CITATION.cff`; they must match.
2. Write the changelog entry in `docs/source/changelog.md`.
3. Confirm CI is green on `main`.
4. Draft and publish a GitHub release. That is what starts the upload.

## Re-vendoring HIP-LLM

`src/HIPLLM/` and `src/hip_llm/` are **byte copies** of
[`koo-ec/HIP_LLM`](https://github.com/koo-ec/HIP_LLM). They are never edited
here, and they are excluded from ruff for that reason. Fix a bug upstream, then:

1. Copy the new sources in, unchanged.
2. Update the recorded commit and date in [Vendoring](../vendoring.md).
3. Run `pytest tests/vendor -q`. That is HIP-LLM's own suite, and it is not ours
   to relax - if it fails, the vendored copy is wrong, not the test.
4. Run `pytest tests/integration/test_public_api.py -q`. `TestHIPLLMIsComplete`
   asserts every HIP-LLM symbol is still reachable through this package, so a
   rename upstream surfaces here instead of at a user's import.

## Dependencies

`pyproject.toml` declares the core (NumPy, pandas, matplotlib, SciPy, PyYAML) and
seven extras:

| Extra | For |
|---|---|
| `bayes` | pyAgrum |
| `graph` | langgraph, langchain-core |
| `test` | the above plus pytest, nbformat, ipython |
| `live` | openai, anthropic, datasets |
| `docs` | sphinx, furo, myst-parser, linkify-it-py |
| `dev` | ruff, build, twine |
| `all` | everything |

If you add an optional dependency that a notebook may import, add it to the
`OPTIONAL` set in `tests/integration/test_notebook_imports.py` as well.
`test_the_optional_set_covers_what_the_extras_declare` checks that the two do not
drift apart.

Dependabot watches the GitHub Actions versions. Those pull requests are real -
check the action's releases before assuming a bump targets a version that does
not exist.
