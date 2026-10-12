"""Power and minimum-detectable-effect calculations for agent benchmarks.

The public API is deliberately small. Three questions practitioners actually ask:

    mde(...)     what delta can I detect with the budget I have?
    n_needed(...)  how big a budget do I need to detect the delta I care about?
    compare(...) is this observed delta real?

Design notes
------------
Agent benchmark outcomes are binary per task and clustered: repeated runs of the same
agent on the same task are correlated, and tasks differ enormously in difficulty. A naive
binomial interval understates uncertainty, sometimes badly. We model that with a design
effect derived from the intra-task correlation ``rho``.

``rho`` must be estimated from repeated-run data, not assumed. Until it is measured for a
given benchmark, callers should sweep it and report a range -- see ``experiments/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

__all__ = ["design_effect", "mde", "n_needed", "compare", "Comparison"]

# Two-sided normal quantiles for common alpha levels.
_Z = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}
_Z_POWER = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}


def _z(alpha: float) -> float:
    if alpha not in _Z:
        raise ValueError(f"alpha must be one of {sorted(_Z)}, got {alpha}")
    return _Z[alpha]


def _z_power(power: float) -> float:
    if power not in _Z_POWER:
        raise ValueError(f"power must be one of {sorted(_Z_POWER)}, got {power}")
    return _Z_POWER[power]


def design_effect(n_seeds: int, rho: float) -> float:
    """Variance inflation from clustering repeated runs within tasks.

    Standard cluster-sampling design effect: ``1 + (m - 1) * rho`` for ``m`` observations
    per cluster. With ``rho = 0`` extra seeds are fully informative; with ``rho = 1`` they
    add nothing at all, which is the regime where re-running an agent tells you nothing you
    did not already know.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    return 1.0 + (n_seeds - 1) * rho


def mde(
    n_tasks: int,
    baseline_rate: float,
    n_seeds: int = 1,
    rho: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    paired: bool = True,
) -> float:
    """Smallest true difference detectable with the given budget.

    Args:
        n_tasks: Number of distinct tasks in the benchmark.
        baseline_rate: Success rate of the reference agent, in (0, 1).
        n_seeds: Runs per task per agent.
        rho: Intra-task correlation across seeds. **Measure this.** The default of 0.5 is a
            placeholder, not an empirical finding.
        paired: True when both agents are evaluated on the same tasks, which is the usual
            benchmark setup and materially reduces the detectable delta.

    Returns:
        Absolute difference in success rate, e.g. 0.043 for 4.3 percentage points.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be in (0, 1)")
    if n_tasks < 2:
        raise ValueError("n_tasks must be >= 2")

    p = baseline_rate
    var = p * (1.0 - p)
    deff = design_effect(n_seeds, rho)
    effective_n = (n_tasks * n_seeds) / deff

    # Paired comparisons cancel task-difficulty variance; unpaired carry both arms.
    arms = 1.0 if paired else 2.0
    se = sqrt(arms * var / effective_n)
    return (_z(alpha) + _z_power(power)) * se


def n_needed(
    target_delta: float,
    baseline_rate: float,
    n_seeds: int = 1,
    rho: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    paired: bool = True,
) -> int:
    """Tasks required to detect ``target_delta``. Inverse of :func:`mde`."""
    if target_delta <= 0:
        raise ValueError("target_delta must be > 0")

    p = baseline_rate
    var = p * (1.0 - p)
    arms = 1.0 if paired else 2.0
    deff = design_effect(n_seeds, rho)

    effective_n = arms * var * ((_z(alpha) + _z_power(power)) / target_delta) ** 2
    return int(-(-(effective_n * deff / n_seeds) // 1))  # ceil


@dataclass(frozen=True)
class Comparison:
    """Result of testing one observed delta against the noise floor."""

    delta: float
    mde: float
    survives: bool
    baseline_rate: float
    n_tasks: int
    n_seeds: int
    rho: float

    def __str__(self) -> str:
        verdict = "survives" if self.survives else "WITHIN NOISE"
        return (
            f"delta={self.delta:+.4f} vs MDE={self.mde:.4f} -> {verdict} "
            f"(n_tasks={self.n_tasks}, n_seeds={self.n_seeds}, rho={self.rho})"
        )


def compare(
    score_a: float,
    score_b: float,
    n_tasks: int,
    n_seeds: int = 1,
    rho: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    paired: bool = True,
) -> Comparison:
    """Test whether an observed delta exceeds the minimum detectable effect.

    This is a screening test against the noise floor, not a substitute for a proper paired
    test on per-task outcomes. Use it when only aggregate scores are available -- which,
    for most published comparisons, is all there is.
    """
    baseline = min(score_a, score_b)
    detectable = mde(
        n_tasks=n_tasks,
        baseline_rate=baseline,
        n_seeds=n_seeds,
        rho=rho,
        alpha=alpha,
        power=power,
        paired=paired,
    )
    delta = score_a - score_b
    return Comparison(
        delta=delta,
        mde=detectable,
        survives=abs(delta) >= detectable,
        baseline_rate=baseline,
        n_tasks=n_tasks,
        n_seeds=n_seeds,
        rho=rho,
    )
