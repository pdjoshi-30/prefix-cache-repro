"""Tests for the variance-components estimator.

The estimator is the scientific core of this paper. A silent bug in it produces
confident, wrong numbers -- the exact failure mode the paper is about. So it is
tested against synthetic data where the true components are known by construction.

Run:  ./.venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalpower.variance import (  # noqa: E402
    bootstrap_ci,
    decompose,
    pooled_within_task_variance,
)


def make_records(cond_sigma: dict[str, float], n_tasks: int, n_reps: int, seed: int = 7):
    """Synthesise cells whose within-task SD is known exactly.

    Task means are spread widely to confirm the estimator measures *within*-task
    variance and is not contaminated by between-task heterogeneity.
    """
    rng = random.Random(seed)
    out = []
    for cond, sigma in cond_sigma.items():
        for task in range(n_tasks):
            task_mean = rng.uniform(0.2, 0.8)
            for trial in range(n_reps):
                out.append({
                    "condition": cond, "task_id": task, "trial": trial,
                    "reward": task_mean + rng.gauss(0, sigma), "error": None,
                })
    return out


# --- pooled_within_task_variance -------------------------------------------

def test_pooled_variance_recovers_known_sigma():
    sigma = 0.10
    recs = make_records({"A": sigma}, n_tasks=200, n_reps=30)
    by_task = {}
    for r in recs:
        by_task.setdefault(r["task_id"], []).append(r["reward"])
    v, n_tasks, n_obs = pooled_within_task_variance(by_task)
    assert n_tasks == 200
    assert n_obs == 200 * 30
    assert math.isclose(v, sigma**2, rel_tol=0.15), f"got {v}, expected ~{sigma**2}"


def test_between_task_spread_does_not_leak_in():
    """Task means vary over [0.2, 0.8] but within-task SD is tiny.

    A naive estimator that pooled all rewards together would report the large
    between-task variance. This must not happen.
    """
    recs = make_records({"A": 0.001}, n_tasks=100, n_reps=10)
    by_task = {}
    for r in recs:
        by_task.setdefault(r["task_id"], []).append(r["reward"])
    v, _, _ = pooled_within_task_variance(by_task)
    assert v < 1e-4, f"between-task variance leaked in: {v}"


def test_single_replicate_tasks_are_dropped_not_zeroed():
    """A task with one observation has undefined variance.

    Counting it as zero would bias the pooled estimate downward.
    """
    by_task = {0: [0.5], 1: [0.1, 0.9], 2: [0.4]}
    v, n_tasks, n_obs = pooled_within_task_variance(by_task)
    assert n_tasks == 1, "only task 1 has >=2 replicates"
    assert n_obs == 2
    assert math.isclose(v, 0.32, rel_tol=1e-6)


# --- decompose --------------------------------------------------------------

def test_decompose_recovers_known_components():
    """Var(B) - Var(A) should recover the simulator component."""
    s_infra, s_sim, s_agent = 0.05, 0.12, 0.08
    recs = make_records(
        {
            "A": s_infra,
            "B": math.sqrt(s_infra**2 + s_sim**2),
            "C": math.sqrt(s_infra**2 + s_agent**2),
            "D": math.sqrt(s_infra**2 + s_sim**2 + s_agent**2),
        },
        n_tasks=300, n_reps=30,
    )
    c = decompose(recs)
    assert math.isclose(c.v_infra, s_infra**2, rel_tol=0.2)
    assert math.isclose(c.v_sim, s_sim**2, rel_tol=0.2)
    assert math.isclose(c.v_agent, s_agent**2, rel_tol=0.2)
    # Constructed additively, so interaction must be ~0.
    assert abs(c.v_interaction) < 0.003, f"spurious interaction {c.v_interaction}"


def test_decompose_detects_superadditive_interaction():
    """When D exceeds the additive prediction, interaction must be positive."""
    s_infra, s_sim, s_agent = 0.05, 0.10, 0.10
    additive = math.sqrt(s_infra**2 + s_sim**2 + s_agent**2)
    recs = make_records(
        {
            "A": s_infra,
            "B": math.sqrt(s_infra**2 + s_sim**2),
            "C": math.sqrt(s_infra**2 + s_agent**2),
            "D": additive * 1.5,  # deliberately super-additive
        },
        n_tasks=300, n_reps=30,
    )
    c = decompose(recs)
    assert c.v_interaction > 0, f"failed to detect super-additivity: {c.v_interaction}"


def test_negative_components_are_reported_not_clipped():
    """A component near zero can estimate negative. That must survive.

    Clipping at zero biases every component upward -- the classic error.
    """
    recs = make_records({"A": 0.10, "B": 0.02, "C": 0.10, "D": 0.10}, n_tasks=60, n_reps=10)
    c = decompose(recs)
    assert c.v_sim < 0, "expected a negative estimate for this construction"
    assert any("negative" in n for n in c.notes), "negative component must be flagged"


def test_zero_variance_condition_A_flagged_when_positive():
    recs = make_records({"A": 0.05, "B": 0.05, "C": 0.05, "D": 0.05}, n_tasks=60, n_reps=10)
    c = decompose(recs)
    assert any("NOT deterministic" in n for n in c.notes)


def test_errored_and_null_cells_excluded():
    recs = [
        {"condition": "A", "task_id": 0, "reward": 1.0, "error": None},
        {"condition": "A", "task_id": 0, "reward": 0.0, "error": None},
        {"condition": "A", "task_id": 0, "reward": None, "error": "boom"},
        {"condition": "A", "task_id": 0, "reward": 0.5, "error": "transport"},
    ]
    c = decompose(recs)
    assert c.n_obs["A"] == 2, "errored and null cells must not enter the estimate"


def test_missing_condition_is_flagged_not_silently_nan():
    recs = make_records({"A": 0.05, "B": 0.05}, n_tasks=20, n_reps=5)
    c = decompose(recs)
    assert any("condition C absent" in n for n in c.notes)
    assert any("condition D absent" in n for n in c.notes)


# --- bootstrap --------------------------------------------------------------

def test_bootstrap_ci_brackets_truth():
    s_infra, s_sim = 0.05, 0.12
    recs = make_records(
        {
            "A": s_infra,
            "B": math.sqrt(s_infra**2 + s_sim**2),
            "C": s_infra,
            "D": math.sqrt(s_infra**2 + s_sim**2),
        },
        n_tasks=120, n_reps=20,
    )
    cis = bootstrap_ci(recs, n_boot=400)
    lo, hi = cis["v_sim"]
    assert lo < s_sim**2 < hi, f"true {s_sim**2} outside CI [{lo}, {hi}]"
    assert lo > 0, "CI should exclude zero for a real component"


def test_bootstrap_ci_includes_zero_for_absent_component():
    recs = make_records({"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08}, n_tasks=120, n_reps=20)
    cis = bootstrap_ci(recs, n_boot=400)
    lo, hi = cis["v_sim"]
    assert lo < 0 < hi, f"CI [{lo}, {hi}] should include 0 when no component exists"


def test_bootstrap_is_deterministic_given_seed():
    recs = make_records({"A": 0.05, "B": 0.10, "C": 0.05, "D": 0.10}, n_tasks=40, n_reps=8)
    a = bootstrap_ci(recs, n_boot=200, seed=123)
    b = bootstrap_ci(recs, n_boot=200, seed=123)
    assert a == b, "bootstrap must be reproducible for a fixed seed"
