"""E9 -- how many replicates does a practitioner actually need? (no new compute)

The paper's critique is only half useful without an answer to "so what should I
do?". Replication is the only remedy left once seeds are unusable and batching is
ruled out, and replication costs money, so the practical question is how many
replicates buy how much precision.

Subsamples the existing E1b airline factorial at R = 2..5 and reports, for each
R, the variance components and the width of their bootstrap CIs. Replicates are
taken in trial order rather than sampled, so the subset is fixed by the data.

    python scripts/08_e9_allocation.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evalpower.variance import bootstrap_ci, decompose  # noqa: E402

RUN = "E1b"
CONDS = ("A", "B", "C", "D")


def load(run_id: str) -> dict[str, dict[str, dict[int, float]]]:
    """cond -> task -> {replicate -> reward}, replicate index from the filename."""
    out: dict[str, dict[str, dict[int, float]]] = {c: defaultdict(dict) for c in CONDS}
    for cond in CONDS:
        d = REPO / "experiments" / run_id / "raw" / run_id / f"cond_{cond}"
        for p in d.glob("*.summary.json"):
            rec = json.loads(p.read_text())
            if rec.get("reward") is None:
                continue
            rep = int(p.name.rsplit("_trial", 1)[1].split(".")[0])
            out[cond][str(rec["task_id"])][rep] = float(rec["reward"])
    return out


def at_r(data, r: int) -> list[dict]:
    """Flatten to the record list `decompose`/`bootstrap_ci` expect.

    Only tasks holding the full R replicates enter, so every R is estimated on a
    balanced design rather than a ragged one.
    """
    records = []
    for cond, tasks in data.items():
        for tid, reps in tasks.items():
            keys = sorted(reps)[:r]
            if len(keys) < r:
                continue
            for k in keys:
                records.append({"condition": cond, "task_id": tid, "reward": reps[k]})
    return records


if __name__ == "__main__":
    data = load(RUN)
    rows = []
    for r in (2, 3, 4, 5):
        sub = at_r(data, r)
        comp = decompose(sub)
        ci = bootstrap_ci(sub, n_boot=10000, seed=7)
        rows.append({
            "R": r,
            "n_tasks": comp.n_tasks.get("A"),
            "v_infra": round(comp.v_infra, 4),
            "v_sim": round(comp.v_sim, 4),
            "v_agent": round(comp.v_agent, 4),
            "ci_width_v_sim": round(ci["v_sim"][1] - ci["v_sim"][0], 4),
            "ci_width_v_agent": round(ci["v_agent"][1] - ci["v_agent"][0], 4),
            "v_sim_excludes_zero": ci["v_sim"][0] > 0,
            "cost_relative": r / 5,
        })

    print(json.dumps({"run": RUN, "domain": "airline", "allocation": rows}, indent=2))
    (REPO / "experiments" / "E9").mkdir(parents=True, exist_ok=True)
    (REPO / "experiments" / "E9" / "allocation.json").write_text(
        json.dumps({"run": RUN, "domain": "airline", "allocation": rows}, indent=2)
    )
