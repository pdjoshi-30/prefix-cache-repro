"""Figure set for the 8-page paper. Numbers are read from experiments/, never typed.

Supersedes 11_paper_figures.py. Three changes of substance:

1. The engine x cache result is four numbers, each 0% or 100%. A grouped bar
   chart spends a lot of ink saying that. It is drawn as a 2x2 factorial matrix
   instead, which is what the design actually is.

2. The cross-domain result is a *change* under one flag, so it is drawn as a
   slopegraph. The quantity the reader should take away is the slope, and a
   slopegraph encodes it directly rather than asking them to compare bar heights
   across a gap.

3. New: the per-task first-divergence strip. The claim "divergence begins at the
   first generated message, it does not accumulate" was previously carried by a
   median in the text. The strip shows the whole distribution, and colouring by
   role carries the banking result (every surviving divergence is at a tool
   result) in the same panel.

    python scripts/13_figures.py
"""

from __future__ import annotations

import collections
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("div", REPO / "scripts" / "04_divergence.py")
div = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(div)

# Validated categorical set, shared with the TikZ methodology diagram.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d5"
WASH, WARM = "#eef4fc", "#fdf0ea"

# Role -> (label, colour). "user" is the simulated user, "assistant" the agent,
# "tool" the environment returning a result.
ROLE = {
    "user":      ("user simulator", BLUE),
    "assistant": ("agent", AQUA),
    "tool":      ("environment (tool result)", ORANGE),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
})


# ------------------------------------------------------------------ data access
def _by_task(run: str, cond: str = "A") -> dict[str, list[Path]]:
    base = REPO / "experiments" / run / "raw" / run / f"cond_{cond}"
    out: dict[str, list[Path]] = collections.defaultdict(list)
    for p in base.glob("*.full.json"):
        tid = str(json.loads(p.read_text())["simulations"][0]["task_id"])
        out[tid].append(p)
    return out


def _seqs(paths: list[Path]) -> list[list[tuple]]:
    return [s for s in (div.transcript(p) for p in sorted(paths)) if s]


def per_task(run: str, cond: str = "A") -> list[tuple[str, int | None, str | None]]:
    """(task_id, first-divergence message index or None, role at that index)."""
    rows = []
    for tid, paths in sorted(_by_task(run, cond).items()):
        seqs = _seqs(paths)
        if len(seqs) < 2:
            continue
        idx, role = div.first_divergence(seqs)
        rows.append((tid, idx, role))
    return rows


def identity_matched(run_a: str, run_b: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """(identical, compared) for two arms over the tasks usable in BOTH.

    Arms are only comparable on a shared denominator. banking loses task_001,
    whose three cache-off cells all died with a null-error infrastructure
    failure; telecom's cache-on arm lost task ids to the filename-collision
    defect. Restricting both arms to the shared, usable ids is the comparison
    the captions claim to show.
    """
    ta, tb = _by_task(run_a), _by_task(run_b)
    common = [t for t in sorted(set(ta) & set(tb))
              if len(_seqs(ta[t])) >= 2 and len(_seqs(tb[t])) >= 2]

    def score(d):
        ok = sum(1 for t in common if div.first_divergence(_seqs(d[t]))[0] is None)
        return ok, len(common)

    return score(ta), score(tb)


def identity(run: str) -> tuple[int, int]:
    a = div.analyse(REPO / "experiments" / run / "raw" / run)["A"]
    return a["identical"], a["n_tasks"]


def reward_varying(run: str, cond: str = "A") -> tuple[int, int]:
    """(tasks whose replicate rewards are not all equal, tasks scored)."""
    base = REPO / "experiments" / run / "raw" / run / f"cond_{cond}"
    by: dict[str, list[float]] = collections.defaultdict(list)
    for p in base.glob("*.summary.json"):
        d = json.loads(p.read_text())
        if d.get("reward") is not None:
            by[str(d["task_id"])].append(d["reward"])
    flips = sum(1 for v in by.values() if len(set(v)) > 1)
    return flips, len(by)


def _style(ax, ylabel: str | None = None, xlabel: str | None = None):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5, color=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.5, color=INK2)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ figure: 2x2
def fig_mechanism_matrix() -> None:
    """The engine x cache factorial as the 2x2 it is."""
    cells = {
        ("vLLM",   "on"):  identity("E4a"),
        ("vLLM",   "off"): identity("E12_vllm_nocache"),
        ("SGLang", "on"):  identity("E13_sglang_cache"),
        ("SGLang", "off"): identity("E11_sglang_single"),
    }
    engines, caches = ["vLLM", "SGLang"], ["on", "off"]

    fig, ax = plt.subplots(figsize=(2.62, 1.98))
    for i, eng in enumerate(engines):
        for j, cache in enumerate(caches):
            ok, n = cells[(eng, cache)]
            good = ok == n
            ax.add_patch(plt.Rectangle(
                (j, 1 - i), 1, 1, facecolor=WASH if good else WARM,
                edgecolor=BLUE if good else ORANGE, linewidth=1.0, zorder=2))
            ax.text(j + 0.5, 1 - i + 0.60, f"{ok}/{n}", ha="center", va="center",
                    fontsize=12, color=BLUE if good else ORANGE, zorder=3)
            ax.text(j + 0.5, 1 - i + 0.30,
                    "every task\nreproduces" if good else "no task\nreproduces",
                    ha="center", va="center", fontsize=6.2, color=INK2, zorder=3,
                    linespacing=1.25)

    ax.set_xlim(-0.02, 2.02)
    ax.set_ylim(-0.02, 2.02)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["cache on\n(default)", "cache off"], fontsize=7.5, color=INK)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(engines, fontsize=7.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(OUT / "fig_matrix.pdf")
    fig.savefig(OUT / "fig_matrix.png")
    plt.close(fig)
    print("fig_matrix", cells)


# ------------------------------------------------------------ figure: slopegraph
def fig_domains_slope() -> None:
    """The one-flag change per domain. The slope is the finding, so encode it."""
    pairs = [
        ("airline", "E4a", "E12_vllm_nocache"),
        ("retail",  "E4r", "E12r_retail_nocache"),
        ("telecom", "E4t2", "E12t_telecom_nocache"),
        ("banking", "E4b2", "E12b_banking_nocache"),
    ]
    fig, ax = plt.subplots(figsize=(2.62, 1.98))

    left, right, results = [], [], []
    for name, a, b in pairs:
        (ok_on, n), (ok_off, _) = identity_matched(a, b)
        y0, y1 = 100.0 * ok_on / n, 100.0 * ok_off / n
        results.append((name, ok_on, ok_off, n))
        colour = BLUE if ok_off == n else ORANGE
        ax.plot([0, 1], [y0, y1], color=colour, linewidth=1.5,
                marker="o", markersize=3.6, zorder=3, solid_capstyle="round")
        left.append((y0, f"{ok_on}/{n}", colour))
        right.append((y1, f"{name}  {ok_off}/{n}", colour))

    def stack(entries, x, ha):
        """Draw labels, nudging apart only where they would collide.

        Several domains share a y value. Where the *text* also matches --- two
        domains both at 0/20 --- one label serves both and stacking would only
        print the same string twice, pushing the lower copy into the axis.
        """
        seen, used = set(), []
        for y, text, colour in sorted(entries, key=lambda r: -r[0]):
            if text in seen:
                continue
            seen.add(text)
            yy = y
            while any(abs(yy - u) < 10.5 for u in used):
                yy -= 10.5
            used.append(yy)
            ax.text(x, yy, text, ha=ha, va="center", fontsize=6.8, color=colour)

    stack(left, -0.07, "right")
    stack(right, 1.07, "left")

    ax.set_xlim(-0.42, 1.92)
    ax.set_ylim(-13, 108)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["prefix cache\non (default)", "prefix cache\noff"],
                       fontsize=7.2, color=INK)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    _style(ax, "tasks reproducing exactly (%)")
    ax.spines["bottom"].set_visible(False)
    fig.savefig(OUT / "fig_slope.pdf")
    fig.savefig(OUT / "fig_slope.png")
    plt.close(fig)
    print("fig_slope", results)


# ----------------------------------------------------------------- figure: strip
def fig_divergence_strip() -> None:
    """Where the first divergence lands, per task, coloured by whose turn it was.

    One row per arm. This is the panel that shows the effect is not accumulation:
    under a default deployment the mass sits on message index 1, the simulated
    user's opening utterance, with nothing before it to have accumulated from.
    """
    arms = [
        ("airline",           "E1b",                  "cache on"),
        ("retail",            "E3ret",                "cache on"),
        ("telecom",           "E4t2",                 "cache on"),
        ("banking",           "E4b2",                 "cache on"),
        ("banking",           "E12b_banking_nocache", "cache OFF"),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 2.05))
    summary = []

    for row, (dom, run, cache) in enumerate(arms):
        y = len(arms) - 1 - row
        rows = per_task(run)
        idxs = [(i, r) for _, i, r in rows if i is not None]
        n_ident = sum(1 for _, i, _ in rows if i is None)
        summary.append((run, len(idxs), n_ident))

        # Vertical offset within the row, so equal indices stay countable.
        seen: dict[int, int] = collections.Counter()
        for i, r in sorted(idxs):
            k = seen[i]
            seen[i] += 1
            ax.plot(i, y + ((k % 7) - 3) * 0.088, marker="o", markersize=2.9,
                    markerfacecolor=ROLE.get(r, ("?", INK2))[1],
                    markeredgecolor="white", markeredgewidth=0.35, zorder=3)

        ax.text(-1.1, y, f"{dom}", ha="right", va="center", fontsize=7.5, color=INK)
        ax.text(-1.1, y - 0.30, cache, ha="right", va="center", fontsize=6.2,
                color=ORANGE if cache == "cache on" else BLUE)
        tail = f"{len(idxs)} diverge"
        if n_ident:
            tail += f", {n_ident} identical"
        ax.text(24.2, y - 0.02, tail, ha="left", va="center", fontsize=6.4, color=INK2)

    ax.axvline(1, color=ORANGE, linewidth=0.7, linestyle=(0, (2.5, 2)), zorder=1)
    ax.text(1.35, len(arms) - 0.42, "index 1 = the simulated user's opening utterance",
            fontsize=6.4, color=ORANGE, va="center")

    ax.set_xlim(-9.5, 33)
    ax.set_ylim(-0.62, len(arms) - 0.28)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    _style(ax, xlabel="message index of the first divergence")
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=3.4,
                      markerfacecolor=c, markeredgecolor="white", label=lab)
               for lab, c in ROLE.values()]
    ax.legend(handles=handles, frameon=False, fontsize=6.4, ncol=3,
              loc="upper center", bbox_to_anchor=(0.52, -0.16), handletextpad=0.3,
              columnspacing=1.6)
    fig.savefig(OUT / "fig_strip.pdf")
    fig.savefig(OUT / "fig_strip.png")
    plt.close(fig)
    print("fig_strip", summary)


# ----------------------------------------------------------- figure: concealment
def fig_concealment() -> None:
    """What the reported score shows, against what the transcripts show.

    The two quantities are nested -- every reward flip is also a divergence --
    so they are drawn as one bar split into visible and invisible parts, rather
    than as two bars the reader has to mentally subtract.
    """
    doms = [("airline", "E1b"), ("retail", "E3ret")]
    fig, ax = plt.subplots(figsize=(2.62, 1.42))
    out = []
    for row, (dom, run) in enumerate(doms):
        y = len(doms) - 1 - row
        flips, n_scored = reward_varying(run)
        ident, n = identity(run)
        diverged = n - ident
        hidden = diverged - flips
        ax.barh(y, flips, height=0.42, color=AQUA, zorder=3)
        ax.barh(y, hidden, left=flips, height=0.42, color=ORANGE, alpha=0.55, zorder=3)
        ax.text(flips / 2, y, f"{flips}", ha="center", va="center",
                fontsize=6.8, color="white", zorder=4)
        ax.text(flips + hidden / 2, y, f"{hidden} invisible to the score",
                ha="center", va="center", fontsize=6.6, color=INK, zorder=4)
        ax.text(-1.2, y, dom, ha="right", va="center", fontsize=7.5, color=INK)
        out.append((dom, flips, diverged, n))

    ax.set_xlim(0, 56)
    ax.set_ylim(-0.55, len(doms) - 0.45)
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    _style(ax, xlabel="tasks (of 50) whose replicates diverge")
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=AQUA, label="reward changes"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.55,
                      label="reward identical, transcript differs"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.4, ncol=1,
              loc="lower left", bbox_to_anchor=(-0.16, -0.62), handlelength=1.1,
              handletextpad=0.4)
    fig.savefig(OUT / "fig_conceal.pdf")
    fig.savefig(OUT / "fig_conceal.png")
    plt.close(fig)
    print("fig_conceal", out)


if __name__ == "__main__":
    fig_mechanism_matrix()
    fig_domains_slope()
    fig_divergence_strip()
    fig_concealment()
    print("figures written to", OUT)
