# data/

Four directories, with different rules.

## `examples/`

`agent_outcomes.csv` — 240 **synthetic** per-agent outcomes for the
`parallel_aggregator` architecture. Also shipped inside the package, so
`load_outcomes()` works from an installed wheel.

It is generated, not measured. Agent accuracy falls with decomposition length,
the two agents' errors are correlated at 0.55 because they share a model
snapshot, and the aggregator selects correctly 85% of the time. Use it to learn
the API; never as evidence about any real model.

Regenerate with `scripts/make_example_outcomes.py` (fixed seed 20260901).

## `reference/`

The published numerics HIP-LLM's own tests check against, vendored alongside
its sources. Do not edit: they are what makes the vendored copy verifiable.
See `docs/source/vendoring.md`.

## `raw/`, `processed/`

For your own data, and empty in the repository. `.gitignore` keeps their
contents out of version control — benchmark outputs are large, often licensed,
and sometimes contain prompts you should not publish.

The shape the package expects is one row per benchmark item:

```
item_id,stratum,<node_1>,<node_2>,...,split
item_0000,short,1,1,calibration
```

where each node column is `1` when that node answered the item **correctly**,
and `split` is `calibration` or `test`. See tutorial 7 for what to log and how
much of it you need.
