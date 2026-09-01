"""Regenerate the synthetic outcome table shipped with the package.

The data is deliberately *generated*, not measured, so the documentation can run
anywhere without a GPU, an API key or a licensed benchmark. It is built to have
the properties the tutorials talk about — and no others:

* accuracy falls with decomposition length, so stratification visibly matters;
* the two agents' errors are correlated (rho = 0.55) because they share a model
  snapshot, so the common-cause discussion is not hypothetical;
* the aggregator selects the better draft 85% of the time and occasionally
  answers correctly when both drafts are wrong, so its fitted gate is neither AND
  nor OR;
* a calibration/test split exists, so the split guards can be demonstrated.

    python scripts/make_example_outcomes.py

Fixed seed. Re-running reproduces the committed file byte for byte.
"""

from __future__ import annotations

import collections
import csv
import pathlib
import sys

import numpy as np

SEED = 20260901
N_ITEMS = 240
N_CALIBRATION = 160

#: The StrategyQA `dev` operational profile measured in the HIP-MAS pilot.
STRATUM_WEIGHTS = {"short": 0.284, "medium": 0.537, "long": 0.179}

#: Per-agent accuracy by stratum. The fall with decomposition length is the
#: point: a pooled accuracy would average it away.
ACCURACY = {
    "react_agent": {"short": 0.88, "medium": 0.74, "long": 0.58},
    "cot_agent": {"short": 0.90, "medium": 0.77, "long": 0.61},
}

#: Correlation between the two agents, from the shared model snapshot.
RHO = 0.55

#: The judge selects a correct draft this often, and is correct this often when
#: both drafts are wrong.
P_SELECT = 0.85
P_RESCUE = 0.05

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = (
    ROOT / "data" / "examples" / "agent_outcomes.csv",
    ROOT / "src" / "HIP_HOPS_LLM" / "data" / "agent_outcomes.csv",
)
COLUMNS = ["item_id", "stratum", "react_agent", "cot_agent", "aggregator", "split"]


def generate() -> list[dict]:
    rng = np.random.default_rng(SEED)
    strata = rng.choice(
        list(STRATUM_WEIGHTS), size=N_ITEMS, p=list(STRATUM_WEIGHTS.values())
    )
    rows = []
    for index, stratum in enumerate(strata):
        common = rng.random()          # the shared-snapshot component
        record = {"item_id": f"item_{index:04d}", "stratum": str(stratum)}
        drafts = {}
        for agent, table in ACCURACY.items():
            u = RHO * common + (1 - RHO) * rng.random()
            drafts[agent] = int(u < table[str(stratum)])
            record[agent] = drafts[agent]
        if any(drafts.values()):
            record["aggregator"] = int(rng.random() < P_SELECT)
        else:
            record["aggregator"] = int(rng.random() < P_RESCUE)
        record["split"] = "calibration" if index < N_CALIBRATION else "test"
        rows.append(record)
    return rows


def main() -> int:
    rows = generate()
    for target in TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {target.relative_to(ROOT)}  ({len(rows)} rows)")

    print("\naccuracy:")
    for agent in ("react_agent", "cot_agent", "aggregator"):
        overall = sum(r[agent] for r in rows) / len(rows)
        print(f"  {agent:<14} {overall:.3f}", end="   ")
        for stratum in STRATUM_WEIGHTS:
            subset = [r[agent] for r in rows if r["stratum"] == stratum]
            print(f"{stratum}={sum(subset) / len(subset):.3f}", end="  ")
        print()
    print("\nstrata:", dict(collections.Counter(r["stratum"] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
