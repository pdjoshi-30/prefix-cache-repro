"""E10 -- the pre-registered hierarchical cross-check (no new compute).

PREREGISTRATION.md section 3.4 promises: "a mixed model with task random
intercept fitted per condition, to confirm the pooled estimates are not driven by
task heterogeneity. If the two disagree, both are reported and the discrepancy
discussed."

That promise was made before any data existed and had not been kept. In a paper
arguing that agent evaluation lacks statistical discipline, an unhonoured
pre-registration is the precise failure being criticised, so this closes it.

For each condition we fit

    reward_ij = mu + u_i + e_ij,     u_i ~ N(0, tau^2) over tasks i

by REML. `tau^2` is between-task heterogeneity; the residual variance is the
within-task variance the primary estimator pools directly. If the primary
estimator were being driven by task heterogeneity rather than within-task noise,
the residual would sit well below the pooled figure.

    python scripts/09_e10_hierarchical.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evalpower.variance import decompose  # noqa: E402

RUN = "E1b"
CONDS = ("A", "B", "C", "D")


def load_records(run_id: str) -> list[dict]:
    recs = []
    for cond in CONDS:
        d = REPO / "experiments" / run_id / "raw" / run_id / f"cond_{cond}"
        for p in d.glob("*.summary.json"):
            r = json.loads(p.read_text())
            if r.get("reward") is None or r.get("error"):
                continue
            recs.append({
                "condition": cond,
                "task_id": str(r["task_id"]),   # int/str spellings collapse
                "reward": float(r["reward"]),
            })
    return recs


if __name__ == "__main__":
    recs = load_records(RUN)
    df = pd.DataFrame(recs)

    # Pooled within-task variance, straight from the pre-registered estimator.
    pooled = {}
    for cond in CONDS:
        sub = [r for r in recs if r["condition"] == cond]
        by_task = defaultdict(list)
        for r in sub:
            by_task[r["task_id"]].append(r["reward"])
        var = [
            sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)
            for v in by_task.values() if len(v) > 1
        ]
        pooled[cond] = sum(var) / len(var) if var else None

    out = {"run": RUN, "domain": "airline", "conditions": {}}
    for cond in CONDS:
        d = df[df.condition == cond]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # REML convergence chatter on binary outcomes
            fit = smf.mixedlm("reward ~ 1", d, groups=d["task_id"]).fit(reml=True)
        tau2 = float(fit.cov_re.iloc[0, 0])
        resid = float(fit.scale)
        out["conditions"][cond] = {
            "n_obs": int(len(d)),
            "n_tasks": int(d.task_id.nunique()),
            "between_task_var_tau2": round(tau2, 5),
            "within_task_var_residual": round(resid, 5),
            "pooled_within_task_primary": round(pooled[cond], 5),
            "abs_diff_vs_primary": round(abs(resid - pooled[cond]), 5),
            "icc": round(tau2 / (tau2 + resid), 4) if (tau2 + resid) else None,
        }

    comp = decompose(recs)
    out["primary_components"] = {
        "v_infra": round(comp.v_infra, 5),
        "v_sim": round(comp.v_sim, 5),
        "v_agent": round(comp.v_agent, 5),
    }
    out["hierarchical_components"] = {
        "v_infra": round(out["conditions"]["A"]["within_task_var_residual"], 5),
        "v_sim": round(out["conditions"]["B"]["within_task_var_residual"]
                       - out["conditions"]["A"]["within_task_var_residual"], 5),
        "v_agent": round(out["conditions"]["C"]["within_task_var_residual"]
                         - out["conditions"]["A"]["within_task_var_residual"], 5),
    }

    print(json.dumps(out, indent=2))
    (REPO / "experiments" / "E10").mkdir(parents=True, exist_ok=True)
    (REPO / "experiments" / "E10" / "hierarchical.json").write_text(json.dumps(out, indent=2))
