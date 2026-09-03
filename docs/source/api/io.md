# `hiphopsllm.io`

Getting systems and outcomes *in*: the bundled examples, and the n8n importer.

## Bundled examples

```{eval-rst}
.. automodule:: hiphopsllm.io.examples
   :members:
   :undoc-members:
```

## n8n workflows

Reads an n8n JSON export into the same architecture model a LangGraph
application produces, so everything downstream is unchanged. See
[Analysing an n8n workflow](../tutorials/08-n8n-workflows.md) for the three
modelling decisions this involves and why each one changes the fault tree.

```{eval-rst}
.. automodule:: hiphopsllm.io.n8n
   :members:
   :undoc-members:
   :show-inheritance:
```
