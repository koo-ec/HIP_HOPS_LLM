# Development

The rest of this site explains how to *use* the package. This section explains
how it is built, so you can change it.

```bash
git clone https://github.com/koo-ec/HIP_HOPS_LLM.git
cd HIP_HOPS_LLM
pip install -e ".[all]"
pytest
```

The suite should be green before you change anything. It takes about four
minutes and includes HIP-LLM's own tests, run against the vendored copy.

## Where to start

| If you want to | Read |
|---|---|
| Understand what is where | [The codebase](codebase.md) |
| Add an archetype, a gate, an exporter, an n8n rule | [Extending the package](extending.md) |
| Know which tests you must not weaken | [Testing](testing.md) |
| Change a notebook | [Notebooks](notebooks.md) |
| Bump a dependency, cut a release, re-vendor HIP-LLM | [CI, docs and releasing](ci-and-releasing.md) |

If you have an hour, read [The codebase](codebase.md) and then the five files it
names. That is about 3,000 lines and it carries every idea in the package.

## The four house rules

Everything in this section is downstream of these. They are not style
preferences; each one exists because the opposite produced a wrong number that
nobody noticed.

**Nothing fails silently.** If a function cannot do what was asked, it raises,
naming what was wrong *and* what the valid options are. It does not fall back
quietly. This package exists partly because a Bayesian network running on default
priors looks exactly like one running on evidence.

**Provenance travels with numbers.** Every `BasicEvent` carries an `evidence`
string, every `OperationalProfile` a `provenance`, every `ComponentEvidence` its
`method`, `n` and interval. A number that reaches a report without them is a bug.

**pyAgrum stays optional.** Anything in `bayes/` must also work through
`engine="exact"`, or say in its docstring that it does not. Graphviz likewise:
`bn.show()` must always produce a picture. CI runs the whole suite once with
pyAgrum uninstalled, which is what keeps this honest.

**Vendored code is not edited.** `src/HIPLLM/` and `src/hip_llm/` are byte copies
of [`koo-ec/HIP_LLM`](https://github.com/koo-ec/HIP_LLM). Fix bugs upstream and
re-vendor - see [Vendoring](../vendoring.md).

## Style

`ruff check .` and `ruff format --diff .`; line length 100. British spelling in
prose, and in API names too - `analyse` appears, `analyze` never does.

Docstrings explain *why*, not what the signature already says. A comment that
restates the line below it is noise; a comment recording a decision - why the
back edge is cut rather than deleted, why the split guard raises - is the reason
anyone can safely change this code later.

```{toctree}
:hidden:

codebase
extending
testing
notebooks
ci-and-releasing
```
