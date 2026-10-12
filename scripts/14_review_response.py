"""Two analyses added in response to external review. No new compute.

1. Is the R=2 ordering "reversal" a property of the estimator, or of the one
   replicate pair we happened to run first? 08_e9_allocation.py subsamples in
   trial order (`sorted(reps)[:r]`), so the reported R=2 row is a single
   deterministic draw. Here we enumerate all C(5,2)=10 ways of choosing two
   replicates from five, applied globally, and report the distribution.

   Finding: the draw we reported is the only one of ten that puts the agent
   estimate below zero, so "under-replication reverses the ordering" was too
   strong. What the enumeration does support is stronger and cleaner: at R=2 the
   estimator cannot order the two components at all.

2. Does any of this reach the number a practitioner reports? Each replicate
   index is one complete single run of the benchmark, so the spread across
   replicate indices is the run-to-run movement of the headline pass rate --
   directly comparable to what Bjarnason et al. report, but measured here with
   both models decoding greedily.

    python scripts/14_review_response.py
"""

from __future__ import annotations

import collections
import itertools
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from evalpower.variance import decompose  # noqa: E402


def load(run: str, cond: str | None = None) -> dict:
    """cond -> task -> {replicate index: reward}, replicate index from the filename."""
    conds = [cond] if cond else ["A", "B", "C", "D"]
    out: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for c in conds:
        base = REPO / "experiments" / run / "raw" / run / f"cond_{c}"
        if not base.exists():
            continue
        for p in base.glob("*.summary.json"):
            rec = json.loads(p.read_text())
            if rec.get("reward") is None:
                continue
            rep = int(p.name.rsplit("_trial", 1)[1].split(".")[0])
            out[c][str(rec["task_id"])][rep] = float(rec["reward"])
    return out


def components_for(data: dict, keep) -> tuple[float, float]:
    """(V_sim, V_agent) using only the replicate indices in `keep`, globally."""
    recs = [
        {"condition": c, "task_id": t, "reward": reps[k]}
        for c, tasks in data.items()
        for t, reps in tasks.items()
        if all(k in reps for k in keep)
        for k in keep
    ]
    comp = decompose(recs)
    return comp.v_sim, comp.v_agent


def reversal_enumeration(run: str = "E1b", r: int = 2) -> None:
    data = load(run)
    reps = sorted({k for tasks in data.values() for v in tasks.values() for k in v})
    draws = []
    for combo in itertools.combinations(reps, r):
        vs, va = components_for(data, combo)
        draws.append((combo, vs, va))

    all_reps = components_for(data, reps)
    diffs = [vs - va for _, vs, va in draws]
    sim_bigger = sum(1 for d in diffs if d > 1e-12)
    agent_bigger = sum(1 for d in diffs if d < -1e-12)
    ties = len(diffs) - sim_bigger - agent_bigger
    neg_agent = sum(1 for _, _, va in draws if va < 0)

    print(f"--- ordering stability at R={r}, {run}, all C({len(reps)},{r}) global draws")
    for combo, vs, va in draws:
        print(f"    replicates {combo}:  V_sim {vs:+.4f}  V_agent {va:+.4f}  diff {vs-va:+.4f}")
    print(f"    simulator term larger in {sim_bigger}/{len(draws)}, "
          f"agent term larger in {agent_bigger}/{len(draws)}, tied in {ties}/{len(draws)}")
    print(f"    agent estimate below zero in {neg_agent}/{len(draws)} draws")
    print(f"    diff spans {min(diffs):+.4f} to {max(diffs):+.4f}")
    print(f"    at R={len(reps)}: V_sim {all_reps[0]:+.4f}  V_agent {all_reps[1]:+.4f}  "
          f"diff {all_reps[0]-all_reps[1]:+.4f}")


def single_run_pass_rates(run: str, domain: str) -> None:
    """Spread of the headline pass rate across complete single runs of the benchmark."""
    print(f"--- single-run pass rates, {domain} ({run})")
    for cond in ["A", "B", "C", "D"]:
        data = load(run, cond)
        if not data.get(cond):
            continue
        tasks = data[cond]
        reps = sorted({k for v in tasks.values() for k in v})
        rates = []
        for r in reps:
            vals = [v[r] for v in tasks.values() if r in v]
            rates.append(100 * sum(vals) / len(vals))
        print(f"    cond {cond}: {[f'{x:.0f}%' for x in rates]}  "
              f"spread {max(rates)-min(rates):.0f} pp, mean {statistics.mean(rates):.1f}%")


def aggregate_by_cache() -> None:
    """Whether the reported score itself moves between the cache-on and cache-off arms."""
    print("--- aggregate pass rate, matched cache-on / cache-off arms")
    pairs = [("airline", "E4a", "E12_vllm_nocache"), ("retail", "E4r", "E12r_retail_nocache"),
             ("telecom", "E4t2", "E12t_telecom_nocache"), ("banking", "E4b2", "E12b_banking_nocache")]
    for dom, on, off in pairs:
        def rate(run):
            tasks = load(run, "A")["A"]
            vals = [x for v in tasks.values() for x in v.values()]
            return 100 * sum(vals) / len(vals), len(vals)
        a, na = rate(on)
        b, nb = rate(off)
        print(f"    {dom:8s} on {a:5.1f}% (n={na})   off {b:5.1f}% (n={nb})   delta {b-a:+.1f} pp")


if __name__ == "__main__":
    reversal_enumeration()
    print()
    single_run_pass_rates("E1b", "airline")
    single_run_pass_rates("E3ret", "retail")
    print()
    aggregate_by_cache()
