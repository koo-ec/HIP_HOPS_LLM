# `hiphopsllm.reliability`

```{eval-rst}
.. automodule:: hiphopsllm.reliability
   :no-members:
```

## Operational profiles

```{eval-rst}
.. automodule:: hiphopsllm.reliability.profile
   :members:
   :undoc-members:
   :show-inheritance:
```

## Evidence calibration

```{eval-rst}
.. automodule:: hiphopsllm.reliability.calibration
   :members:
   :undoc-members:
   :show-inheritance:
```

## HIP-LLM, re-exported

```{eval-rst}
.. automodule:: hiphopsllm.reliability.hipllm
   :no-members:
```

The HIP-LLM classes themselves are documented upstream at
<https://hipllm.readthedocs.io>. The names reachable from here are:

`FailureProb`, `FailureProbResult`, `LogprobsUnavailableError`,
`OperationalFailureProb`, `OperationalFailureResult`, `StrategyQALoadError`,
`decomposition_stratum`, `load_strategyqa`, `paper_inference_settings`,
`parse_strategyqa_answer`, `quick_inference_settings`, `HIPLLM_VERSION`,
`BenchmarkResult`, `CDFEnvelope`, `DomainData`, `GlobalSettings`,
`HIPLLMOperationalProfile`, `HyperparameterConfiguration`,
`HyperparameterInterval`, `HyperposteriorGrid`, `ModelResult`,
`PosteriorSamples`, `ReliabilityEnvelope`, `ReproductionRecord`,
`ReproductionStatus`, `RunMode`, `SourceRecord`, `SubdomainData`,
`OFFICIAL_REPOSITORY`, `PAPER_DOI`,

plus the engine modules `api_clients`, `baselines`, `benchmark_eval`,
`envelopes`, `grids`, `hyperposterior`, `numerics`, `operational_profile`,
`plotting`, `posterior`, `reliability`, `scalability`, `schemas`, `validation`.

A test asserts this list stays complete against `HIPLLM.__all__` and
`hip_llm.__all__`.
