# Tutorials

Eight walkthroughs. The first seven are in order and run against bundled
data, so you can work through the sequence with nothing installed but the
package; the eighth stands alone and covers a different kind of input.

| | | |
|---|---|---|
| 1 | [From a LangGraph to a fault tree](01-langgraph-to-fault-tree.md) | Extraction, roles, common-cause groups, loop unrolling |
| 2 | [Cut sets, importance and the FMEA](02-cut-sets-and-fmea.md) | Reading the structural result, and what to do about it |
| 3 | [Measuring under an operational profile](03-operational-profile.md) | HIP-LLM, intervals, calibrating the leaves |
| 4 | [Fault tree to Bayesian network](04-fault-tree-to-bayesnet.md) | CPTs, exact inference, diagnosis, drawing |
| 5 | [End to end in ten lines](05-end-to-end.md) | The whole pipeline, and comparing two architectures |
| 6 | [Learned CPTs](06-learned-cpts.md) | Fitting a gate from data instead of assuming it |
| 7 | [Your own graph](07-your-own-graph.md) | Pointing all of this at a real LangGraph application |
| 8 | [Analysing an n8n workflow](08-n8n-workflows.md) | The same analysis, from an n8n JSON export instead |

Runnable scripts for each are in [`examples/`](https://github.com/koo-ec/HIP_HOPS_LLM/tree/main/examples).

```{toctree}
:hidden:

01-langgraph-to-fault-tree
02-cut-sets-and-fmea
03-operational-profile
04-fault-tree-to-bayesnet
05-end-to-end
06-learned-cpts
07-your-own-graph
08-n8n-workflows
```
