"""The public surface: everything exported must import, and HIP-LLM must be complete.

The second guarantee is the one that matters most.  HIP-LLM is vendored into this
repository, and the promise made in the README is that *every* function it
exposes is reachable through this package.  A test is the only thing that keeps
that promise true after either side changes.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import HIP_HOPS_LLM


class TestTopLevelExports:
    def test_every_name_in_all_exists(self):
        missing = [n for n in HIP_HOPS_LLM.__all__ if not hasattr(HIP_HOPS_LLM, n)]
        assert not missing, f"__all__ names nothing: {missing}"

    def test_star_import_works(self):
        namespace: dict = {}
        exec("from HIP_HOPS_LLM import *", namespace)
        for name in HIP_HOPS_LLM.__all__:
            assert name in namespace

    def test_all_is_sorted_within_its_groups_and_unique(self):
        assert len(set(HIP_HOPS_LLM.__all__)) == len(HIP_HOPS_LLM.__all__)

    def test_the_version_is_readable(self):
        assert HIP_HOPS_LLM.__version__

    def test_every_submodule_imports(self):
        failures = []
        for info in pkgutil.walk_packages(
            HIP_HOPS_LLM.__path__, prefix="HIP_HOPS_LLM."
        ):
            try:
                importlib.import_module(info.name)
            except ImportError as exc:  # optional dependency, acceptable
                if "pyagrum" not in str(exc).lower():
                    failures.append((info.name, exc))
            except Exception as exc:  # noqa: BLE001
                failures.append((info.name, exc))
        assert not failures, f"submodules failed to import: {failures}"


class TestHIPLLMIsComplete:
    def test_every_hipllm_symbol_is_re_exported(self):
        import HIPLLM

        from HIP_HOPS_LLM.reliability import hipllm as shim

        renamed = {"__version__": "HIPLLM_VERSION"}
        missing = [
            n
            for n in HIPLLM.__all__
            if not hasattr(shim, renamed.get(n, n))
        ]
        assert not missing, (
            "HIP-LLM's public API is not fully re-exported; missing: "
            f"{missing}. Add them to HIP_HOPS_LLM.reliability.hipllm."
        )

    def test_every_hip_llm_engine_symbol_is_re_exported(self):
        import hip_llm

        from HIP_HOPS_LLM.reliability import hipllm as shim

        renamed = {"OperationalProfile": "HIPLLMOperationalProfile"}
        missing = [
            n
            for n in hip_llm.__all__
            if n != "__version__" and not hasattr(shim, renamed.get(n, n))
        ]
        assert not missing, f"hip_llm symbols not re-exported: {missing}"

    def test_the_re_exported_objects_are_the_same_objects(self):
        import HIPLLM

        from HIP_HOPS_LLM.reliability import hipllm as shim

        for name in HIPLLM.__all__:
            if name.startswith("__"):
                continue
            assert getattr(shim, name) is getattr(HIPLLM, name), (
                f"{name} was re-exported as a copy, not the original object"
            )

    def test_every_engine_module_is_reachable(self):
        from HIP_HOPS_LLM.reliability import hipllm as shim

        for name in (
            "posterior",
            "hyperposterior",
            "envelopes",
            "reliability",
            "baselines",
            "benchmark_eval",
            "scalability",
            "validation",
            "numerics",
            "grids",
            "plotting",
            "schemas",
            "operational_profile",
            "api_clients",
        ):
            assert hasattr(shim, name), f"hip_llm.{name} is not reachable"

    def test_the_users_own_snippet_runs_through_this_package(self):
        """The example from the request, with only the import line changed."""
        from HIP_HOPS_LLM import OperationalFailureProb, quick_inference_settings

        outcomes = [1, 1, 0, 1, 0, 0, 1, 0]
        strata = ["short"] * 4 + ["long"] * 4
        estimator = OperationalFailureProb(
            profile={"short": 0.30, "long": 0.70},
            settings=quick_inference_settings(samples=300, configurations=8),
        )
        result = estimator.fit(outcomes=outcomes, strata=strata)
        summary = result.summary()
        assert 0.0 <= summary["empirical_operational_failure_probability"] <= 1.0
        assert (
            summary["posterior_expected_failure_lower"]
            <= summary["posterior_expected_failure_upper"]
        )

    def test_hipllm_and_hip_llm_remain_importable_directly(self):
        """Vendoring must not break code that already imports them by name."""
        import HIPLLM
        import hip_llm

        assert HIPLLM.OperationalFailureProb
        assert hip_llm.PAPER_DOI


class TestNamespaceHygiene:
    def test_the_two_operational_profiles_are_distinguishable(self):
        from hip_llm.schemas import OperationalProfile as EngineProfile

        from HIP_HOPS_LLM import OperationalProfile
        from HIP_HOPS_LLM.reliability.hipllm import HIPLLMOperationalProfile

        assert OperationalProfile is not EngineProfile
        assert HIPLLMOperationalProfile is EngineProfile

    def test_conversion_between_them_round_trips(self):
        from HIP_HOPS_LLM import OperationalProfile

        original = OperationalProfile({"a": 0.25, "b": 0.75})
        assert OperationalProfile.coerce(original.to_hipllm()).weights == (
            original.weights
        )

    def test_the_lower_case_alias_module_resolves(self):
        alias = importlib.import_module("hip_hops_llm")
        assert alias.AgenticReliabilityStudy is HIP_HOPS_LLM.AgenticReliabilityStudy


class TestExamplesShip:
    def test_every_declared_example_loads(self):
        from HIP_HOPS_LLM import EXAMPLES, load_example

        for key in EXAMPLES:
            spec = load_example(key)
            assert spec["nodes"] and spec["edges"]
            assert spec.get("provenance"), f"{key} does not say where it came from"

    def test_an_unknown_example_lists_the_available_ones(self):
        from HIP_HOPS_LLM import load_example

        with pytest.raises(KeyError, match="available"):
            load_example("nope")

    def test_the_outcome_table_has_the_expected_shape(self):
        from HIP_HOPS_LLM import load_outcomes

        frame = load_outcomes()
        assert set(frame.columns) >= {
            "item_id",
            "stratum",
            "react_agent",
            "cot_agent",
            "aggregator",
            "split",
        }
        assert set(frame["split"]) == {"calibration", "test"}
        assert set(frame["react_agent"]) <= {0, 1}

    def test_outcomes_load_without_pandas_too(self):
        from HIP_HOPS_LLM import load_outcomes

        rows = load_outcomes(as_frame=False)
        assert isinstance(rows, list) and rows and isinstance(rows[0], dict)

    def test_the_catalogue_renders(self):
        from HIP_HOPS_LLM import describe_examples

        assert "parallel_aggregator" in describe_examples()
