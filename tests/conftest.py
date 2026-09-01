"""Shared fixtures.

The bundled examples are the fixtures: every test runs against the same
architectures the documentation uses, so a change that breaks the docs breaks the
tests too.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from HIP_HOPS_LLM import (  # noqa: E402
    AgenticReliabilityStudy,
    load_example,
    load_outcomes,
)

PROFILE = {"short": 0.30, "medium": 0.50, "long": 0.20}


@pytest.fixture(scope="session")
def parallel_spec():
    return load_example("parallel_aggregator")


@pytest.fixture(scope="session")
def react_spec():
    return load_example("react_calculator")


@pytest.fixture(scope="session")
def supervisor_spec():
    return load_example("supervisor_workers")


@pytest.fixture(scope="session")
def outcomes():
    return load_outcomes()


@pytest.fixture(scope="session")
def study(parallel_spec):
    """An analysed, uncalibrated study of the parallel-aggregator example."""
    s = AgenticReliabilityStudy(parallel_spec, name="parallel_aggregator")
    s.analyse()
    return s


@pytest.fixture(scope="session")
def calibrated(parallel_spec, outcomes):
    """The same study, calibrated from the bundled synthetic outcomes."""
    s = AgenticReliabilityStudy(parallel_spec, name="calibrated")
    s.observe(outcomes, profile=PROFILE)
    s.run()
    return s


@pytest.fixture(scope="session")
def tree(study):
    return study.report.tree("H2")


@pytest.fixture(scope="session")
def failure_model(study):
    return study.report.failure_model


def pytest_configure(config):
    try:
        import pyagrum  # noqa: F401
    except ImportError:
        config.addinivalue_line(
            "markers", "pyagrum: skipped — pyAgrum is not installed"
        )
