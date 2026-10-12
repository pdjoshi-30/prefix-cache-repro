"""Transcript-level divergence under temperature 0 (no new compute).

`V_infra` is measured on the *reward*, which is a lossy projection: two runs can
diverge wildly and still land on the same binary outcome, and a floored domain
(retail at ~5% pass) yields almost no reward signal at all. The transcript is the
raw phenomenon.

For every task in a condition we compare replicate transcripts message-by-message
and record:

  * identity rate       -- fraction of tasks whose replicates are all identical
  * first divergence    -- earliest message index where two replicates differ
  * which side diverged -- agent (assistant) or simulator (user)

The last one is the interesting column. The agent is Qwen3-30B-A3B, a *mixture of
experts*; the simulator is Mistral-7B, dense. Expert routing depends on batch
composition, so if continuous batching is the mechanism behind `V_infra`, the
agent side should diverge disproportionately. Dense-model float non-associativity
would not predict that asymmetry.

This metric also transfers to domains where reward is uninformative, which is what
makes the cross-domain generalisation affordable.

    python scripts/04_divergence.py experiments/E1b/raw/E1b
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def signature(msg: dict) -> tuple:
    """Identity of one message: role, text, and any tool call it makes.

    Tool *results* are included as-is -- a differing tool result means the
    environment itself diverged, which is a distinct finding from the model
    diverging and should not be silently normalised away.
    """
    calls = msg.get("tool_calls") or []
    return (
        msg.get("role"),
        msg.get("content"),
        json.dumps(
            [(c.get("name") or (c.get("function") or {}).get("name"),
              c.get("arguments") or (c.get("function") or {}).get("arguments"))
             for c in calls],
            sort_keys=True,
        ),
    )


def transcript(path: Path) -> list[tuple]:
    sims = json.loads(path.read_text()).get("simulations") or []
    if not sims:
        return []
    return [signature(m) for m in sims[0].get("messages", [])]


def first_divergence(seqs: list[list[tuple]]) -> tuple[int | None, str | None]:
    """Earliest index where the replicates stop agreeing, and whose turn it was.

    A shorter transcript that is a strict prefix of a longer one still counts as
    divergence at the truncation point: ending early is a behavioural difference,
    not a missing observation.
    """
    for i in range(max(len(s) for s in seqs)):
        present = [s[i] for s in seqs if i < len(s)]
        if len(present) < len(seqs) or len(set(present)) > 1:
            role = present[0][0] if present else None
            return i, role
    return None, None


def analyse(run_dir: Path) -> dict:
    out = {}
    for cond_dir in sorted(run_dir.glob("cond_*")):
        cond = cond_dir.name.replace("cond_", "")
        by_task: dict[str, list[Path]] = defaultdict(list)
        for p in cond_dir.glob("*.full.json"):
            # task_id is stored as int in some records and str in others; string
            # keys collapse the two spellings onto the same task.
            tid = str(json.loads(p.read_text())["simulations"][0]["task_id"])
            by_task[tid].append(p)

        identical = diverged = 0
        idx_by_role: dict[str, list[int]] = defaultdict(list)
        norm_positions = []
        for tid, paths in sorted(by_task.items()):
            seqs = [transcript(p) for p in sorted(paths)]
            seqs = [s for s in seqs if s]
            if len(seqs) < 2:
                continue
            idx, role = first_divergence(seqs)
            if idx is None:
                identical += 1
            else:
                diverged += 1
                idx_by_role[role or "unknown"].append(idx)
                norm_positions.append(idx / max(len(s) for s in seqs))

        n = identical + diverged
        out[cond] = {
            "n_tasks": n,
            "identical": identical,
            "diverged": diverged,
            "divergence_rate": round(diverged / n, 4) if n else None,
            "first_divergence_by_role": {
                r: {"n": len(v), "median_index": statistics.median(v)}
                for r, v in sorted(idx_by_role.items())
            },
            "median_normalised_position": (
                round(statistics.median(norm_positions), 4) if norm_positions else None
            ),
        }
    return out


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    result = analyse(run_dir)
    print(json.dumps(result, indent=2))
