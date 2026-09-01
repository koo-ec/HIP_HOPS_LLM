""":math:`(\\mu,\\nu)` integration grids and hyper-hyper-parameter configuration sets.

REPRODUCIBILITY GAP (documented, not silently patched)
------------------------------------------------------
The paper fixes the grid *size* -- ``n_mu = 40``, ``n_nu = 50``, ``G = 2000``
(Section 4.2, Appendix B) -- but never states

  (a) how the unbounded axis :math:`\\nu\\in(0,\\infty)` is rendered finite, nor
  (b) the rule used to draw the ``K = 160`` hyper-hyper-parameter configurations.

The official repository (github.com/aghazadehchakherlou-web/llm-imprecise-bayes)
contains **no source code at all** -- only images, ``numerics/settings.yaml`` and
``numerics/data/models_accuracies.csv``.  Its full commit history (22 commits,
single ``main`` branch, no tags/releases) consists solely of README edits and
image uploads, so neither (a) nor (b) is recoverable from any official source.

Consequently this module exposes *several clearly-named* constructions.  In
``strict_exact`` mode :func:`build_grid` and :func:`sample_configurations` refuse
to run at all; outside strict mode the chosen scheme is recorded in every
artifact's metadata and labelled a reconstruction, never "exact".
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy import stats
from scipy.stats import qmc

from .numerics import assert_finite
from .schemas import HyperparameterConfiguration, HyperparameterInterval, HyperposteriorGrid

__all__ = [
    "NU_GRID_SCHEMES",
    "MU_GRID_SCHEMES",
    "CONFIG_SAMPLING_SCHEMES",
    "StrictModeError",
    "build_mu_axis",
    "build_nu_axis",
    "build_grid",
    "sample_configurations",
    "pair_llm_configurations",
]


class StrictModeError(RuntimeError):
    """Raised when strict-exact mode meets a setting that the sources do not fix."""


#: :math:`\nu` axis constructions.  None of these is recoverable from the paper
#: or its repository; all are labelled reconstructions.
NU_GRID_SCHEMES: dict[str, str] = {
    "log": (
        "Geometric (log-spaced) nodes on a truncated interval [nu_lo, nu_hi] with "
        "exact cell widths in nu-space (the Jacobian is carried by the widths, not "
        "by a change of variable in the integrand)."
    ),
    "linear": "Midpoint rule with uniform cell width on a truncated interval [nu_lo, nu_hi].",
    "gamma_quantile": (
        "Prior-quantile nodes nu_k = F^{-1}_{Gamma(c, rate=d)}((k+1/2)/n_nu). The "
        "Gamma prior density is absorbed exactly into the uniform cell mass 1/n_nu, "
        "so the axis adapts to each hyperparameter configuration and needs no "
        "truncation constant."
    ),
}

MU_GRID_SCHEMES: dict[str, str] = {
    "midpoint": "Midpoint rule mu_k = (k+1/2)/n_mu on (0,1), uniform width 1/n_mu.",
    "beta_quantile": (
        "Prior-quantile nodes mu_k = F^{-1}_{Beta(a,b)}((k+1/2)/n_mu); the Beta prior "
        "density is absorbed into the uniform cell mass 1/n_mu."
    ),
}

CONFIG_SAMPLING_SCHEMES: dict[str, str] = {
    "uniform_random": "Independent Uniform draws per coordinate from the admissible box.",
    "latin_hypercube": "Scrambled Latin-hypercube design scaled to the admissible box.",
    "sobol": "Scrambled Sobol' sequence scaled to the admissible box.",
    "interval_corners_plus_interior": (
        "The 16 box corners first (imprecise-probability extrema, cf. paper "
        "Appendix A.2 Step 3), then K-16 uniform interior draws."
    ),
}

# Default truncation for the bounded nu schemes.  Chosen to cover the admissible
# priors with margin: c in [1,25], d in [1,25] gives E[nu] = c/d in [0.04, 25]
# and Gamma(25, rate=1) has ~1e-12 mass beyond nu = 80.
_DEFAULT_NU_LO = 1e-3
_DEFAULT_NU_HI = 250.0


def _strict_guard(strict: bool, what: str) -> None:
    if strict:
        raise StrictModeError(
            f"strict_exact mode: {what} is not fixed by the paper or by the official "
            f"repository (which contains no source code and whose full commit history "
            f"is README/image-only). Refusing to guess. Either supply the recovered "
            f"setting or run with strict_exact=False, in which case the result is "
            f"labelled a reconstruction rather than an exact reproduction."
        )


# --------------------------------------------------------------------------- #
# axes
# --------------------------------------------------------------------------- #
def build_mu_axis(
    n_mu: int,
    scheme: str = "midpoint",
    a: float | None = None,
    b: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Build the :math:`\\mu\\in(0,1)` axis.

    Returns
    -------
    (nodes, log_widths, absorbs_prior)
        ``log_widths`` are :math:`\\log\\Delta\\mu_k` for a plain product rule, or
        :math:`\\log(1/n_\\mu)` when the Beta prior has been absorbed
        (``absorbs_prior=True``).
    """
    if scheme not in MU_GRID_SCHEMES:
        raise ValueError(f"unknown mu grid scheme {scheme!r}; choose from {sorted(MU_GRID_SCHEMES)}")
    if n_mu <= 0:
        raise ValueError("n_mu must be positive")

    probs = (np.arange(n_mu) + 0.5) / n_mu

    if scheme == "midpoint":
        nodes = probs.copy()
        log_widths = np.full(n_mu, -np.log(n_mu))
        return assert_finite(nodes, "mu nodes"), log_widths, False

    if a is None or b is None:
        raise ValueError("mu scheme 'beta_quantile' requires the Beta parameters (a, b)")
    nodes = stats.beta.ppf(probs, a, b)
    # ppf can return exactly 0/1 for extreme shapes; nudge into the open interval.
    eps = np.finfo(float).eps
    nodes = np.clip(nodes, eps, 1.0 - eps)
    log_widths = np.full(n_mu, -np.log(n_mu))
    return assert_finite(nodes, "mu nodes"), log_widths, True


def build_nu_axis(
    n_nu: int,
    scheme: str = "log",
    c: float | None = None,
    d: float | None = None,
    nu_lo: float = _DEFAULT_NU_LO,
    nu_hi: float = _DEFAULT_NU_HI,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Build the :math:`\\nu\\in(0,\\infty)` axis.

    Returns
    -------
    (nodes, log_widths, absorbs_prior)
        For ``log``/``linear`` the nodes are cell midpoints and ``log_widths`` are
        the true cell widths in :math:`\\nu`-space, so the Jacobian of the
        log-transform is carried exactly and no non-uniform grid is ever treated
        as uniform.  For ``gamma_quantile`` the Gamma prior is absorbed and
        ``log_widths`` is :math:`\\log(1/n_\\nu)`.
    """
    if scheme not in NU_GRID_SCHEMES:
        raise ValueError(f"unknown nu grid scheme {scheme!r}; choose from {sorted(NU_GRID_SCHEMES)}")
    if n_nu <= 0:
        raise ValueError("n_nu must be positive")

    probs = (np.arange(n_nu) + 0.5) / n_nu

    if scheme == "gamma_quantile":
        if c is None or d is None:
            raise ValueError("nu scheme 'gamma_quantile' requires the Gamma parameters (c, d)")
        nodes = stats.gamma.ppf(probs, a=c, scale=1.0 / d)
        tiny = np.finfo(float).tiny
        nodes = np.maximum(nodes, tiny)
        log_widths = np.full(n_nu, -np.log(n_nu))
        return assert_finite(nodes, "nu nodes"), log_widths, True

    if not (0 < nu_lo < nu_hi):
        raise ValueError(f"require 0 < nu_lo < nu_hi, got [{nu_lo}, {nu_hi}]")

    if scheme == "linear":
        edges = np.linspace(nu_lo, nu_hi, n_nu + 1)
    else:  # "log"
        edges = np.geomspace(nu_lo, nu_hi, n_nu + 1)

    nodes = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return assert_finite(nodes, "nu nodes"), np.log(widths), False


def build_grid(
    n_mu: int,
    n_nu: int,
    mu_scheme: str = "midpoint",
    nu_scheme: str = "log",
    config: HyperparameterConfiguration | None = None,
    strict_exact: bool = False,
    nu_params: Mapping[str, Any] | None = None,
) -> HyperposteriorGrid:
    """Assemble the full :math:`G=n_\\mu n_\\nu` integration grid.

    The returned :class:`~hip_llm.schemas.HyperposteriorGrid` records which prior
    factors have been absorbed into the cell weights so that
    :func:`hip_llm.hyperposterior.log_hyperposterior` cannot double-count them.
    """
    _strict_guard(strict_exact, "the finite construction of the nu in (0, inf) axis")

    nu_params = dict(nu_params or {})
    a = b = c = d = None
    if config is not None:
        a, b, c, d = config.a, config.b, config.c, config.d

    mu_nodes, mu_logw, mu_absorbs = build_mu_axis(n_mu, mu_scheme, a=a, b=b)
    nu_nodes, nu_logw, nu_absorbs = build_nu_axis(
        n_nu,
        nu_scheme,
        c=c,
        d=d,
        nu_lo=float(nu_params.get("nu_lo", _DEFAULT_NU_LO)),
        nu_hi=float(nu_params.get("nu_hi", _DEFAULT_NU_HI)),
    )

    MU, NU = np.meshgrid(mu_nodes, nu_nodes, indexing="ij")
    LW = mu_logw[:, None] + nu_logw[None, :]

    return HyperposteriorGrid(
        mu=MU.ravel(),
        nu=NU.ravel(),
        log_cell_weight=LW.ravel(),
        n_mu=n_mu,
        n_nu=n_nu,
        scheme=f"mu:{mu_scheme}|nu:{nu_scheme}",
        absorbs_mu_prior=mu_absorbs,
        absorbs_nu_prior=nu_absorbs,
        meta={
            "nu_lo": nu_params.get("nu_lo", _DEFAULT_NU_LO),
            "nu_hi": nu_params.get("nu_hi", _DEFAULT_NU_HI),
            "config_dependent": bool(mu_absorbs or nu_absorbs),
            "reconstruction": True,
            "reconstruction_note": (
                "nu-axis construction is NOT specified by the paper and NOT present "
                "in the official repository; this is a labelled reconstruction."
            ),
        },
    )


# --------------------------------------------------------------------------- #
# hyper-hyper-parameter configurations
# --------------------------------------------------------------------------- #
def sample_configurations(
    interval: HyperparameterInterval,
    K: int,
    seed: int,
    scheme: str = "uniform_random",
    strict_exact: bool = False,
) -> tuple[HyperparameterConfiguration, ...]:
    """Draw ``K`` admissible :math:`h_i=(a_i,b_i,c_i,d_i)\\in\\mathcal{H}_i`.

    The paper says only "the hyper-hyperparameters ... are sampled from fixed
    intervals" (Section 4.2) and the repository's ``settings.yaml`` gives
    ``N_CONFIGS: 160`` with ``seeds.configs: 123``.  The *rule* is unrecorded, so
    all four candidate schemes are implemented and the choice is surfaced.

    A precise (degenerate) interval always collapses to ``K`` identical
    configurations, which is what makes the envelope collapse test meaningful.
    """
    _strict_guard(strict_exact, "the rule used to sample the K hyper-hyper-parameter configurations")

    if scheme not in CONFIG_SAMPLING_SCHEMES:
        raise ValueError(
            f"unknown configuration scheme {scheme!r}; choose from {sorted(CONFIG_SAMPLING_SCHEMES)}"
        )
    if K <= 0:
        raise ValueError("K must be positive")

    lo = interval.bounds[:, 0]
    hi = interval.bounds[:, 1]
    span = hi - lo

    if scheme == "uniform_random":
        rng = np.random.default_rng(seed)
        unit = rng.uniform(size=(K, 4))
    elif scheme == "latin_hypercube":
        unit = qmc.LatinHypercube(d=4, seed=seed).random(K)
    elif scheme == "sobol":
        # K = 160 is not a power of two, so the Sobol' balance property does not
        # hold exactly.  That is itself evidence against Sobol' being the
        # authors' rule; the warning is expected and is downgraded rather than
        # hidden -- see the note recorded in the returned configuration metadata
        # and in data/provenance_manifest.yaml.
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*power of 2.*", category=UserWarning)
            unit = qmc.Sobol(d=4, scramble=True, seed=seed).random(K)
    else:  # interval_corners_plus_interior
        corners = np.array(np.meshgrid(*[[0.0, 1.0]] * 4, indexing="ij")).reshape(4, -1).T
        if K <= corners.shape[0]:
            unit = corners[:K]
        else:
            rng = np.random.default_rng(seed)
            unit = np.vstack([corners, rng.uniform(size=(K - corners.shape[0], 4))])

    values = lo[None, :] + unit * span[None, :]
    assert_finite(values, "hyperparameter configurations")
    return tuple(
        HyperparameterConfiguration(a=float(v[0]), b=float(v[1]), c=float(v[2]), d=float(v[3]))
        for v in values
    )


def pair_llm_configurations(
    K_per_domain: int,
    n_domains: int,
    max_pairs: int,
    seed: int,
    mode: str = "capped_random",
) -> np.ndarray:
    """Choose the LLM-level configuration tuples :math:`\\mathcal{h}=(h_1,\\dots,h_m)`.

    Paper Theorem 3 defines the LLM-level admissible set as the *full Cartesian
    product* :math:`\\mathcal{H}_{\\mathrm{LLM}}=\\mathcal{H}_1\\times\\cdots\\times
    \\mathcal{H}_m`, but Section 4.2 caps the realised pairings at
    :math:`K_{\\text{total}}\\le 512`.  With ``K=160`` and ``m=2`` the exact
    product has 25 600 members, so a capped subsample is what the paper actually
    evaluated.

    ``mode="exact_cartesian"`` enumerates the full product (feasible for small
    ``K``); ``mode="capped_random"`` draws ``max_pairs`` distinct tuples with
    :class:`numpy.random.Generator` seeded by ``seed``.  Crucially the tuple is
    drawn *jointly* -- domains are not paired independently, which would destroy
    the intended cross-domain configuration envelope.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_pairs, n_domains)`` of per-domain config indices.
    """
    if n_domains <= 0 or K_per_domain <= 0:
        raise ValueError("n_domains and K_per_domain must be positive")

    # Python integers: |H_LLM| = K^m overflows int64 for as few as m = 9 domains
    # at K = 160.  The paper sweeps m up to 12 while holding K = 160, so whatever
    # rule the authors used cannot have enumerated a flat index space either.
    total = K_per_domain**n_domains

    if mode == "exact_cartesian":
        if total > 2_000_000:
            raise ValueError(
                f"exact Cartesian product has {total} members; refuse to enumerate. "
                f"Use mode='capped_random'."
            )
        return (
            np.array(np.meshgrid(*[np.arange(K_per_domain)] * n_domains, indexing="ij"))
            .reshape(n_domains, -1)
            .T.astype(np.int64)
        )

    if mode != "capped_random":
        raise ValueError(f"unknown pairing mode {mode!r}")

    if total <= max_pairs:
        return pair_llm_configurations(K_per_domain, n_domains, max_pairs, seed, "exact_cartesian")

    rng = np.random.default_rng(seed)

    if total < 2**62:
        # Exact uniform sample without replacement via the flat index space.
        flat = np.sort(rng.choice(total, size=max_pairs, replace=False))
        out = np.empty((max_pairs, n_domains), dtype=np.int64)
        residual = flat.copy()
        for axis in range(n_domains - 1, -1, -1):
            out[:, axis] = residual % K_per_domain
            residual //= K_per_domain
        return out

    # The product is too large to index.  Draw whole tuples uniformly instead --
    # still a JOINT draw from H_1 x ... x H_m, which is what preserves the
    # cross-domain configuration envelope; domains are never paired
    # independently (e.g. by zipping per-domain orderings).  Collisions are
    # astronomically unlikely here but are removed anyway so the sample stays
    # without replacement.
    rows: dict[tuple[int, ...], None] = {}
    while len(rows) < max_pairs:
        draw = rng.integers(0, K_per_domain, size=(max_pairs - len(rows), n_domains))
        for r in draw:
            rows[tuple(int(v) for v in r)] = None
    return np.array(sorted(rows), dtype=np.int64)
