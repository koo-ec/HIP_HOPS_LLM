"""Shared fixtures.

Tolerances are declared here, **before** any result is examined, as the
acceptance criteria require.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Upstream these tests sit at ``tests/``, so ``parents[1]`` was the repository
# root. Vendored here they sit at ``tests/vendor/``, so the root is one level
# further up. Everything else in this file is byte-identical to upstream.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hip_llm.schemas import (  # noqa: E402
    DomainData,
    GlobalSettings,
    HyperparameterInterval,
    ModelResult,
    SubdomainData,
)

# --------------------------------------------------------------------------- #
# tolerances, fixed in advance
# --------------------------------------------------------------------------- #
TOL_MEDIAN = 0.005          # posterior median absolute difference
TOL_QUANTILE = 0.010        # 5% / 95% quantile absolute difference
TOL_CDF_SUP = 0.020         # CDF sup-norm difference
TOL_EXPECTED_RELIABILITY = 0.010
TOL_MC_MEAN = 0.01          # Monte-Carlo mean vs analytical
TOL_MC_VAR = 0.15           # relative tolerance on a Monte-Carlo variance


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def reference_dir(root: Path) -> Path:
    return root / "data" / "reference"


@pytest.fixture
def wide_interval() -> HyperparameterInterval:
    """The paper's baseline admissible set: a,b in [1,12], c,d in [1,25]."""
    return HyperparameterInterval((1, 12), (1, 12), (1, 25), (1, 25))


@pytest.fixture
def precise_interval() -> HyperparameterInterval:
    """A degenerate (precise) admissible set: every interval collapses to a point."""
    return HyperparameterInterval((3, 3), (2, 2), (5, 5), (2, 2))


@pytest.fixture
def toy_domain() -> DomainData:
    """Two subdomains, equal weights -- small enough for analytical cross-checks."""
    return DomainData(
        "D1",
        (SubdomainData("A", 38, 80, 0.475), SubdomainData("B", 39, 80, 0.4875)),
        np.array([0.5, 0.5]),
    )


@pytest.fixture
def sparse_rich_domain() -> DomainData:
    """One data-sparse and one data-rich subdomain sharing the same empirical rate.

    Used to test that pooling shrinks the sparse subdomain more strongly.
    """
    return DomainData(
        "D1",
        (SubdomainData("sparse", 4, 8, 0.5), SubdomainData("rich", 400, 800, 0.5)),
        np.array([0.5, 0.5]),
    )


@pytest.fixture
def toy_model(toy_domain: DomainData) -> ModelResult:
    d2 = DomainData(
        "D2",
        (SubdomainData("C", 72, 80, 0.90), SubdomainData("D", 74, 80, 0.925)),
        np.array([0.483, 0.517]),
    )
    return ModelResult("toy", (toy_domain, d2), np.array([0.149, 0.851]), "unit-test fixture")


@pytest.fixture
def fast_settings() -> GlobalSettings:
    """Small but structurally identical settings, so tests run in seconds."""
    return GlobalSettings(
        n_mu=24,
        n_nu=24,
        cdf_points_T=201,
        S=2000,
        K_per_domain=12,
        max_llm_configuration_pairs=48,
        seed_global=7,
        seed_configs=123,
        seed_pairs=999,
        config_sampling="uniform_random",
        nu_grid_scheme="log",
        mu_grid_scheme="midpoint",
    )


@pytest.fixture
def paper_settings() -> GlobalSettings:
    """The paper's exact baseline: 40 x 50 grid, S = 3000, K = 160, cap 512."""
    return GlobalSettings(
        n_mu=40,
        n_nu=50,
        cdf_points_T=201,
        S=3000,
        K_per_domain=160,
        max_llm_configuration_pairs=512,
        seed_global=7,
        seed_configs=123,
        seed_pairs=999,
        config_sampling="uniform_random",
        nu_grid_scheme="log",
        mu_grid_scheme="midpoint",
    )
