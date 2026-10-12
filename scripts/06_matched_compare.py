"""Matched batched-vs-unbatched comparison of transcript divergence.

The batched baselines (E1b, E3ret) were collected at R=5 over 50 tasks; the E4
arms run R=3 over 20 tasks. Comparing them as-is would confound the serving
manipulation with replicate count -- more replicates mean more chances for two
transcripts to differ, so the baseline would look more divergent for a reason
that has nothing to do with batching.

This script therefore restricts the baseline to the SAME task ids and the SAME
number of replicates as the treatment arm before comparing anything. Replicates
are taken in trial order (0,1,2) rather than sampled, so the subset is fixed by
the data rather than by a draw.

Reported per domain:
    identity rate           tasks whose replicates are all identical
    first-divergence index  where the transcripts first differ
    which side diverged     simulator (user) or agent (assistant)

    python scripts/06_matched_compare.py E1b E4a airline
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import importlib.util

_spec = importlib.util.spec_from_file_location("div", REPO / "scripts" / "04_divergence.py")
div = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(div)


def load(run_id: str, cond: str = "A") -> dict[str, dict[int, Path]]:
    """task_id -> {replicate -> path}, keyed by string so int/str spellings collapse.

    The replicate index comes from the FILENAME, not from the record: each cell
    invokes tau2 with `--num-trials 1`, so tau2's own `trial` field is 0 in every
    file. Reading it from the record collapsed all replicates of a task onto one
    key and silently reduced every comparison to a single observation.
    """
    root = REPO / "experiments" / run_id / "raw" / run_id / f"cond_{cond}"
    by_task: dict[str, dict[int, Path]] = defaultdict(dict)
    for p in root.glob("*.full.json"):
        sim = json.loads(p.read_text())["simulations"][0]
        rep = int(p.name.rsplit("_trial", 1)[1].split(".")[0])
        by_task[str(sim["task_id"])][rep] = p
    return by_task


def summarise(by_task, tasks: set[str], n_reps: int, label: str) -> dict:
    identical = diverged = 0
    by_role: dict[str, list[int]] = defaultdict(list)
    positions = []
    used = 0
    for tid in sorted(tasks):
        trials = by_task.get(tid, {})
        paths = [trials[t] for t in sorted(trials)[:n_reps] if t in trials]
        if len(paths) < 2:
            continue
        seqs = [s for s in (div.transcript(p) for p in paths) if s]
        if len(seqs) < 2:
            continue
        used += 1
        idx, role = div.first_divergence(seqs)
        if idx is None:
            identical += 1
        else:
            diverged += 1
            by_role[role or "unknown"].append(idx)
            positions.append(idx / max(len(s) for s in seqs))
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    return {
        "label": label,
        "tasks_compared": used,
        "replicates_per_task": n_reps,
        "identical": identical,
        "diverged": diverged,
        "identity_rate": round(identical / used, 4) if used else None,
        "median_normalised_position": round(med(positions), 4) if positions else None,
        "first_divergence_by_role": {
            r: {"n": len(v), "median_index": med(v)} for r, v in sorted(by_role.items())
        },
    }


if __name__ == "__main__":
    baseline_id, treatment_id, domain = sys.argv[1], sys.argv[2], sys.argv[3]
    base, treat = load(baseline_id), load(treatment_id)

    # Only tasks present in BOTH, so neither arm is scored on tasks the other
    # never ran. Replicate count is set by the treatment arm, the smaller one.
    shared = set(base) & set(treat)
    n_reps = min(
        min((len(v) for v in treat.values()), default=0),
        min((len(v) for v in base.values()), default=0),
    )

    out = {
        "domain": domain,
        "shared_tasks": len(shared),
        "matched_replicates": n_reps,
        "batched": summarise(base, shared, n_reps, f"{baseline_id} (batching ON)"),
        "unbatched": summarise(treat, shared, n_reps, f"{treatment_id} (batching OFF)"),
    }
    print(json.dumps(out, indent=2))
