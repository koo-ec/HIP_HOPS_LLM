"""Fitting conditional probability tables from observed agent outcomes.

A synthesised fault tree says an aggregator's output is wrong when *both* its
inputs are wrong.  That is a modelling assumption, and in a multi-agent system it
is usually wrong in an interesting direction: a reviewer repairs some upstream
errors and introduces others, so the true table is neither AND nor OR.  When
per-node outcomes have actually been logged, the table can be estimated instead
of assumed.

The difference is not academic.  In the HIP-MAS synthetic ground-truth study,
with a reviewer repairing 55% of upstream errors, the deterministic AND-series
gate mispredicted held-out failure by ``+0.386`` while the learned-CPT model was
within ``0.005``.

Two guards are enforced in code rather than by convention:

* a CPT is never fitted from rows marked ``test`` --- :func:`learn_cpt` raises;
* rows with no observations fall back to the Dirichlet prior mean and are
  *counted*, so a report can state how many table rows were prior-dominated
  rather than implying they were measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .cpt import FAIL, OK, CPT, CPTSet

__all__ = [
    "LearnedCPT",
    "learn_cpt",
    "learn_gate",
    "fit_cpts",
    "CPTLearningError",
]


class CPTLearningError(ValueError):
    """A CPT was asked to learn from data it must not see, or cannot use."""


@dataclass(frozen=True)
class LearnedCPT:
    """A fitted table plus how much data stood behind each row."""

    cpt: CPT
    #: raw counts, shape ``(*parent_cards, 2)`` --- before smoothing
    counts: np.ndarray = field(repr=False)
    alpha: float = 1.0

    @property
    def n_observations(self) -> int:
        return int(self.counts.sum())

    @property
    def prior_dominated_rows(self) -> int:
        """Rows with no observation at all: the prior, not the data, speaks."""
        return int((self.counts.sum(axis=-1) == 0).sum())

    @property
    def total_rows(self) -> int:
        return int(np.prod(self.counts.shape[:-1])) if self.counts.ndim > 1 else 1

    @property
    def coverage(self) -> float:
        """Fraction of parent configurations that were actually observed."""
        return 1.0 - self.prior_dominated_rows / max(self.total_rows, 1)

    def summary(self) -> str:
        return (
            f"{self.cpt.variable}: {self.n_observations} observations over "
            f"{self.total_rows} rows, {self.prior_dominated_rows} prior-dominated "
            f"(coverage {self.coverage:.0%}, Dirichlet alpha={self.alpha})"
        )


def _column_states(values: Iterable[Any]) -> np.ndarray:
    """Coerce a column of outcomes to ``0 = OK`` / ``1 = Fail`` indices.

    Accepts booleans, 0/1 integers, and the strings ``"OK"``/``"Fail"``.  Note
    the polarity: these columns record *failure*, so a benchmark's ``correct``
    column must be inverted by the caller.  :func:`fit_cpts` says so in its
    signature rather than guessing.
    """
    out: List[int] = []
    for v in values:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("fail", "failed", "f", "1", "true", "yes"):
                out.append(FAIL)
            elif s in ("ok", "o", "0", "false", "no", "correct"):
                out.append(OK)
            else:
                raise CPTLearningError(f"cannot read {v!r} as OK/Fail")
        else:
            out.append(FAIL if bool(v) else OK)
    return np.asarray(out, dtype=int)


def _assert_not_test_rows(frame: Any, what: str) -> None:
    """Refuse to fit on rows a split has marked as held-out."""
    for column in ("split", "fold", "partition"):
        if column in getattr(frame, "columns", ()):
            marked = {str(v).strip().lower() for v in frame[column].dropna().unique()}
            forbidden = marked & {"test", "held_out", "heldout", "holdout", "eval"}
            if forbidden:
                raise CPTLearningError(
                    f"{what} was given rows marked {sorted(forbidden)} in column "
                    f"'{column}'. Fit CPTs on the calibration split only; a table "
                    "fitted on the evaluation set makes every downstream number "
                    "optimistic and untestable."
                )


def learn_cpt(
    frame: Any,
    child: str,
    parents: Sequence[str] = (),
    *,
    alpha: float = 1.0,
    check_split: bool = True,
) -> LearnedCPT:
    """Estimate ``P(child | parents)`` with symmetric Dirichlet(alpha) smoothing.

    Parameters
    ----------
    frame
        A pandas DataFrame with one row per observed task and one column per
        node, holding *failure* indicators (see :func:`fit_cpts`).
    child, parents
        Column names.  ``parents`` may be empty, giving a root prior.
    alpha
        Dirichlet concentration.  ``alpha = 1`` is Laplace smoothing and the
        sensible default: at pilot sample sizes several parent configurations
        ("both agents wrong *and* they agree") are seen a handful of times or not
        at all, and an unsmoothed MLE would put a hard 0 or 1 in the table and
        make the network claim a certainty it has not earned.
    check_split
        Refuse to fit if the frame carries rows marked as a test split.

    Returns
    -------
    LearnedCPT
        The table, plus the raw counts, so the report can say how much of it was
        measured and how much is prior.
    """
    if alpha <= 0:
        raise CPTLearningError("the Dirichlet alpha must be positive")
    columns = list(parents) + [child]
    available = set(getattr(frame, "columns", []))
    missing = [c for c in columns if c not in available]
    if missing:
        raise CPTLearningError(f"the frame has no column(s) {missing}")
    if check_split:
        _assert_not_test_rows(frame, f"learn_cpt({child!r})")

    sub = frame[columns].dropna()
    if len(sub) == 0:
        raise CPTLearningError(
            f"no complete rows for {child!r} given {list(parents)}; nothing to fit"
        )
    states = np.column_stack([_column_states(sub[c]) for c in columns])

    counts = np.zeros((2,) * len(columns), dtype=float)
    for row in states:
        counts[tuple(int(v) for v in row)] += 1.0

    smoothed = counts + float(alpha)
    table = smoothed / smoothed.sum(axis=-1, keepdims=True)

    cpt = CPT(
        variable=child,
        parents=tuple(parents),
        table=table,
        kind="learned",
        gate=None,
        node_id=child,
        label=child,
        evidence=(
            f"fitted from {int(counts.sum())} observations with Dirichlet "
            f"alpha={alpha}"
        ),
    )
    return LearnedCPT(cpt=cpt, counts=counts, alpha=float(alpha))


def learn_gate(
    frame: Any,
    child: str,
    parents: Sequence[str],
    *,
    alpha: float = 1.0,
    check_split: bool = True,
) -> Tuple[LearnedCPT, Dict[str, float]]:
    """Fit a gate and report how far it is from AND and from OR.

    The distances are the mean absolute difference between the fitted
    ``P(Fail | parents)`` column and the deterministic table, which is a direct
    answer to "is this aggregator really a voter?".
    """
    from .cpt import deterministic_gate_cpt

    learned = learn_cpt(
        frame, child, parents, alpha=alpha, check_split=check_split
    )
    fitted = learned.cpt.table[..., FAIL]
    distances = {
        gate.lower(): float(
            np.mean(np.abs(fitted - deterministic_gate_cpt(len(parents), gate)[..., FAIL]))
        )
        for gate in ("AND", "OR")
    }
    distances["nearest"] = min(distances, key=distances.get)  # type: ignore[arg-type]
    return learned, distances


def fit_cpts(
    frame: Any,
    structure: Mapping[str, Sequence[str]],
    *,
    name: str = "learned",
    alpha: float = 1.0,
    check_split: bool = True,
    outcomes_are_failures: bool = True,
) -> Tuple[CPTSet, Dict[str, LearnedCPT]]:
    """Fit every CPT of a network whose *structure* is already known.

    ``structure`` maps each variable to its parents, and must be given in a
    topological order (parents before children) --- the same order a
    :class:`~HIP_HOPS_LLM.bayes.cpt.CPTSet` keeps.

    ``outcomes_are_failures`` states the polarity of the columns explicitly.
    Benchmark data usually records *correctness*; pass ``False`` and the columns
    are inverted once, here, instead of silently everywhere::

        cpts, fits = fit_cpts(
            observations,
            {"react": [], "cot": [], "aggregator": ["react", "cot"]},
            outcomes_are_failures=False,      # columns hold 1 = correct
        )
        bn = BayesianNetwork(cpts)
    """
    working = frame
    if not outcomes_are_failures:
        working = frame.copy()
        for column in structure:
            if column in working.columns:
                working[column] = [
                    OK if _state_is_fail(v) else FAIL for v in working[column]
                ]

    cs = CPTSet(name=name)
    fits: Dict[str, LearnedCPT] = {}
    for child, parents in structure.items():
        learned = learn_cpt(
            working, child, list(parents), alpha=alpha, check_split=check_split
        )
        cs.add(learned.cpt)
        cs.event_variable.setdefault(child, child)
        fits[child] = learned

    cs.top = list(structure)[-1] if structure else ""
    prior_dominated = sum(f.prior_dominated_rows for f in fits.values())
    if prior_dominated:
        cs.notes.append(
            f"{prior_dominated} conditional table row(s) had no observations and "
            f"fell back to the Dirichlet(alpha={alpha}) prior mean"
        )
    return cs, fits


def _state_is_fail(value: Any) -> bool:
    return _column_states([value])[0] == FAIL
