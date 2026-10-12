"""Variance-components estimation for the temperature factorial.

Implements the estimator fixed in PREREGISTRATION.md §3. Written before E1 data
exists so the analysis cannot be tuned to the result.

Conditions:
    A  agent T=0, sim T=0   residual infrastructure nondeterminism
    B  agent T=0, sim T>0   infra + simulator
    C  agent T>0, sim T=0   infra + agent
    D  agent T>0, sim T>0   total

    V_infra       = pooled within-task variance in A
    V_sim         = pooled(B) - V_infra
    V_agent       = pooled(C) - V_infra
    V_interaction = pooled(D) - V_infra - V_sim - V_agent

Two deliberate choices, both from the pre-registration:

1. **Components are never truncated at zero.** A negative estimate means the
   component is near zero and the subtraction is noise-dominated. Clipping would
   bias every component upward -- the classic error in variance-components work.

2. **Bootstrap resamples tasks, not replicates.** Tasks are the unit of
   generalisation; resampling replicates would understate uncertainty about
   whether the finding holds on other tasks.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

__all__ = ["Components", "pooled_within_task_variance", "decompose", "bootstrap_ci"]


@dataclass
class Components:
    v_infra: float
    v_sim: float
    v_agent: float
    v_total: float
    v_interaction: float
    n_tasks: dict[str, int] = field(default_factory=dict)
    n_obs: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "v_infra": self.v_infra, "v_sim": self.v_sim, "v_agent": self.v_agent,
            "v_total": self.v_total, "v_interaction": self.v_interaction,
            "n_tasks": self.n_tasks, "n_obs": self.n_obs, "notes": self.notes,
        }

    def __str__(self) -> str:
        lines = [
            f"  V_infra       {self.v_infra:+.6f}   (condition A)",
            f"  V_sim         {self.v_sim:+.6f}   (B - A)",
            f"  V_agent       {self.v_agent:+.6f}   (C - A)",
            f"  V_total       {self.v_total:+.6f}   (condition D)",
            f"  V_interaction {self.v_interaction:+.6f}   (D - A - sim - agent)",
        ]
        for n in self.notes:
            lines.append(f"  ! {n}")
        return "\n".join(lines)


def _var(xs: list[float]) -> float | None:
    """Unbiased sample variance. None when undefined (fewer than 2 observations)."""
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def pooled_within_task_variance(by_task: dict, min_reps: int = 2) -> tuple[float, int, int]:
    """Mean of per-task variances, over tasks with enough replicates.

    Returns (pooled_variance, n_tasks_used, n_observations_used).

    Tasks with a single replicate are dropped rather than contributing zero --
    counting them as zero-variance would bias the estimate downward.
    """
    per_task = []
    n_obs = 0
    for _task, rewards in by_task.items():
        rewards = [r for r in rewards if r is not None]
        if len(rewards) < min_reps:
            continue
        v = _var(rewards)
        if v is not None:
            per_task.append(v)
            n_obs += len(rewards)
    if not per_task:
        return float("nan"), 0, 0
    return sum(per_task) / len(per_task), len(per_task), n_obs


def _group(records: list[dict]) -> dict[str, dict]:
    """records -> {condition: {task_id: [rewards]}}"""
    out: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("reward") is None or r.get("error"):
            continue
        out[r["condition"]][str(r["task_id"])].append(float(r["reward"]))
    return out


def decompose(records: list[dict]) -> Components:
    """Estimate variance components from cell records."""
    grouped = _group(records)
    pooled, n_tasks, n_obs, notes = {}, {}, {}, []

    for cond in ("A", "B", "C", "D"):
        if cond not in grouped:
            pooled[cond], n_tasks[cond], n_obs[cond] = float("nan"), 0, 0
            notes.append(f"condition {cond} absent")
            continue
        v, nt, no = pooled_within_task_variance(grouped[cond])
        pooled[cond], n_tasks[cond], n_obs[cond] = v, nt, no
        if nt == 0:
            notes.append(f"condition {cond}: no task had >=2 usable replicates")

    v_infra = pooled["A"]
    v_sim = pooled["B"] - v_infra
    v_agent = pooled["C"] - v_infra
    v_total = pooled["D"]
    v_inter = v_total - v_infra - v_sim - v_agent

    # Report, do not clip. See module docstring.
    for name, val in (("V_sim", v_sim), ("V_agent", v_agent), ("V_interaction", v_inter)):
        if val == val and val < 0:
            notes.append(f"{name} is negative -- component likely near zero, subtraction noise-dominated")

    if v_infra == v_infra and v_infra > 0:
        notes.append(
            f"condition A variance is {v_infra:.6f} > 0 -- infrastructure is NOT deterministic "
            "under concurrency, despite temperature 0"
        )

    return Components(v_infra, v_sim, v_agent, v_total, v_inter, n_tasks, n_obs, notes)


def bootstrap_ci(
    records: list[dict],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260731,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs, resampling TASKS (the unit of generalisation)."""
    rng = random.Random(seed)
    tasks = sorted({str(r["task_id"]) for r in records if r.get("reward") is not None})
    if len(tasks) < 2:
        return {}

    by_task: dict = defaultdict(list)
    for r in records:
        by_task[str(r["task_id"])].append(r)

    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        picked = [rng.choice(tasks) for _ in tasks]
        resampled = []
        for i, t in enumerate(picked):
            for r in by_task[t]:
                r2 = dict(r)
                r2["task_id"] = f"{t}#{i}"  # keep duplicated tasks distinct
                resampled.append(r2)
        c = decompose(resampled)
        for k, v in (("v_infra", c.v_infra), ("v_sim", c.v_sim),
                     ("v_agent", c.v_agent), ("v_total", c.v_total),
                     ("v_interaction", c.v_interaction)):
            if v == v:
                draws[k].append(v)

    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    out = {}
    for k, vals in draws.items():
        if len(vals) < 100:
            continue
        vals.sort()
        out[k] = (vals[int(lo_q * len(vals))], vals[min(int(hi_q * len(vals)), len(vals) - 1)])
    return out
