# Notebooks

Both are **generated** from `scripts/build_notebooks.py`, with no committed
output. A notebook diff is otherwise mostly JSON noise, and baked-in outputs are
indistinguishable from fresh ones.

```bash
python scripts/build_notebooks.py
```

A test executes every code cell of both and fails if the committed files differ
from what the generator produces, so neither can quietly go stale.

## `HIP_HOPS_LLM_Colab.ipynb`

Clones this repository, installs the package, and runs the whole pipeline with
plots: architecture, fault tree, calibration tables, Bayesian network, diagnosis,
profile sensitivity, and the model-diversity counterfactual.

Open it in Colab directly from GitHub:

<https://colab.research.google.com/github/koo-ec/HIP_HOPS_LLM/blob/main/notebooks/HIP_HOPS_LLM_Colab.ipynb>

About two minutes end to end, most of it the install. No GPU, no API key, no data
of your own.

Whether Colab has the Graphviz binary varies with the image, so the notebook
prints `graphviz_available()` rather than assuming. When it is absent the
Bayesian networks are drawn with matplotlib and the caption says so; when it is
present pyAgrum renders them. Either way a picture appears. Add
`!apt-get -qq install graphviz` in the first cell to force the pyAgrum path.

## `hip_hops_for_agentic_ai.ipynb`

The Kaggle notebook this package grew out of
([kooaslansefat/hip-hops-for-agentic-ai](https://www.kaggle.com/code/kooaslansefat/hip-hops-for-agentic-ai)),
rewritten to install the package rather than `%%writefile` a 4,800-line module
into one cell.

It follows the same argument as the original — two architectures, synthesis,
cut sets, FMEA, a Bayesian network — with three differences:

| Then | Now |
|---|---|
| The pyAgrum network was hand-wired beside the fault tree | It is generated *from* the tree, so the two cannot disagree |
| Basic events came from entropy or engineering judgement | HIP-LLM's imprecise posterior under an explicit operational profile (the entropy path is still shown, for comparison) |
| Graphviz required to draw the network | matplotlib fallback, so a picture always appears |

Both approaches still work with a live compiled LangGraph, and fall back to the
recorded mermaid text when `langgraph` is not installed — so the notebook runs on
Kaggle, in Colab, or locally without changes.
