# Prefix Caching Can Silently Break Reproducibility in Interactive Agent Evaluation

**Prathamesh Dinesh Joshi, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat** · Vizuara AI Labs

Accepted at the [Workshop on Evaluation of Interactive Agents](https://eval-interactive-agents-workshop.github.io/) @ NeurIPS 2026.

| | |
|---|---|
| **Camera-ready PDF** | *coming soon* |
| **OpenReview** | *coming soon* |
| **Benchmark** | [τ²-bench](https://github.com/sierra-research/tau2-bench) at commit `363133a` (`v1.0.1-17`) |
| **Engines** | vLLM `0.11.0` (V1) and SGLang `0.5.2` |

This repository has the analysis code, per-run records and figures for the paper.

---

## TL;DR

With **both** conversational models decoded greedily, τ²-bench transcripts still don't
reproduce. Most runs diverge at the **first generated message**, so this isn't small errors
building up over a long dialogue. The setting that decides it is the serving engine's
**prefix cache**, which both vLLM and SGLang turn on by default.

- **0 of 50** airline tasks and **0 of 50** retail tasks reproduce exactly across 5 greedy
  replicates on a default deployment.
- In a matched **engine × cache** 2×2 on 20 airline tasks, cache status alone predicted the
  outcome: **0/20 with the cache on, 20/20 with it off, on both vLLM and SGLang.**
- With the cache off, **airline (20/20), retail (20/20) and telecom (11/11)** reproduce exactly.
  **Banking** improves from 4/19 to 7/19. Its 12 remaining divergences all start at
  **environment tool results**, never at either model.
- The benchmark's reward metric **hides 76–84%** of these divergences (airline: 50 tasks
  diverge, 12 change reward; retail: 50 and 8).
- At **R = 2** replicates the ordering of the simulator and agent variance components depends on
  which pair you run. It doesn't settle until R = 4–5.

**Recommendation.** Report the prefix/radix-cache setting along with temperature and model
version. Also report replicates per task, the retry policy, and whether any part of the
environment or grader is a language model.

---

## Setup

An episode is a conversation between an **agent** and a **user simulator** (two language
models) over a shared **environment** with a **judge**. Both models run on inference servers
that keep a prefix cache. The cache is mutable state shared across requests, so one episode
can affect the next.

### Temperature factorial (pre-registered)

| Condition | agent T | sim T |
|---|---|---|
| A | 0 | 0 |
| B | 0 | 1 |
| C | 1 | 0 |
| D | 1 | 1 |

With $\bar V_X$ the mean unbiased within-task reward variance in condition $X$:

$$
\hat V_{\text{infra}} = \bar V_A,\quad
\hat V_{\text{sim}} = \bar V_B - \bar V_A,\quad
\hat V_{\text{agent}} = \bar V_C - \bar V_A,\quad
\hat V_{\text{int}} = \bar V_D - \bar V_A - \hat V_{\text{sim}} - \hat V_{\text{agent}}
$$

Components are never truncated at zero. CIs use a percentile bootstrap over **tasks**
(10⁴ resamples). Any third stochastic participant (an LM environment, a judge, retrieval
tie-breaking) is untouched by all four conditions, so it ends up in $\hat V_{\text{infra}}$.

### Transcript identity

Each message reduces to `(role, text, [(tool name, canonicalised args)])`. Tool-result text is
included. IDs, timestamps, token usage and latency are excluded. A task is *identical* only if
all R replicates agree at every index, and a truncated transcript counts as a divergence.

### Serving

| Factor | vLLM | SGLang | Default |
|---|---|---|---|
| prefix cache off | `--no-enable-prefix-caching` | `--disable-radix-cache` | on |
| request serialisation | `--max-num-seqs 1` | `--max-running-requests 1` | batched |

Mechanism and domain arms run with one request in flight. The primary factorial (`E1b`,
`E3ret`) used the engine defaults. τ²-bench retries are set to 0, since its default of 3
retries would hide unlucky rollouts. Agent on an H100 80 GB, simulator on an L40S 48 GB.

### Models

Reasoning is disabled throughout (`enable_thinking: false`). Conversations are capped at 30
steps and calls at 1024 tokens. Retail's NL-assertion judge and banking's environment interface
call the **simulator** model at temperature 0.

| Cfg | Agent | Simulator |
|---|---|---|
| C1 (primary) | `Qwen/Qwen3-30B-A3B` (MoE) | `mistralai/Mistral-7B-Instruct-v0.3` |
| C2 | `Qwen/Qwen3-32B` (dense) | `mistralai/Mistral-7B-Instruct-v0.3` |
| C4 | `NousResearch/Hermes-3-Llama-3.1-8B` | `Qwen/Qwen3-8B` |
| C5 | `NousResearch/Hermes-3-Llama-3.1-8B` | `unsloth/gemma-3-12b-it` |

---

## Results

### Engine × cache (Fig. 3a): 20 matched airline tasks, C1, 3 replicates

| | cache on (default) | cache off |
|---|---|---|
| **vLLM** | 0/20 (`E4a`) | 20/20 (`E12`) |
| **SGLang** | 0/20 (`E13`) | 20/20 (`E11s`) |

`E11m` (SGLang, cache off, multiple replicas) also gives 20/20.

### The same flag across domains (Fig. 3b)

| domain | cache on | cache off | note |
|---|---|---|---|
| airline | 0/20 | 20/20 | |
| retail | 0/20 | 20/20 | |
| telecom | 1/11 | 11/11 | 11 usable tasks (filename collision, App. G) |
| banking | 4/19 | 7/19 | all 12 residual first divergences at tool results |

### Variance components (airline, cache on, R = 5)

| | estimate | 95% CI |
|---|---|---|
| $\hat V_{\text{infra}}$ | 0.058 | [0.030, 0.090] |
| $\hat V_{\text{sim}}$ | 0.044 | [0.016, 0.074] |
| $\hat V_{\text{agent}}$ | 0.040 | [0.014, 0.070] |

These were measured on the default cache-on deployment, so $\hat V_{\text{infra}}$ describes
that deployment rather than interactive evaluation in general. Between-task share (ICC) falls
from 0.574 (A) to 0.202 (D). Across the five greedy runs, single-run airline pass rates were
20, 16, 16, 14 and 14%.

### Generalisation (Table 3): cache on, condition A

| Cfg | airline | retail | banking | telecom |
|---|---|---|---|---|
| C1 | 0/20 | 0/20 | 4/20 | 1/11 |
| C2 | 1/20 | 1/20 | 2/20 | 8/20 |
| C4 | 1/20 | 2/17 | 0/20 | 7/20 |
| C5 | 4/20 | 3/17 | n/a | n/a |

### Controls

- **Not sampling.** Every mechanism arm was greedy on both sides, checked per call.
- **Not seeds.** `seed` was accepted and ignored on the one endpoint tested.
- **Not continuous batching.** Serialisation on/off moves airline by one task (1/20 → 0/20).
- **Not the model family.** With the cache on, no configuration reproduces more than 40% of
  its tasks.
- **Not engine identity.** See the 2×2 above.

### Limitations

The study covers one benchmark. The cache-off fix is shown on one pairing (C1), with 20 tasks
and 3 replicates. The low-level cache mechanism is not traced. Banking's residual isn't split
between its LM interface and retrieval tie-breaking. Conditions B–D were not re-run with the
cache off. Throughput cost was not measured.

---

## Repository

```
src/evalpower/   variance.py   pre-registered estimator and task-level bootstrap
                 power.py      MDE / power helpers
scripts/         04_divergence.py       transcript identity and first divergence
                 06_matched_compare.py  matched cache / serialisation contrasts
                 08_e9_allocation.py    components by replicate count (Table 4)
                 09_e10_hierarchical.py REML cross-check and ICC (App. F, Table 5)
                 13_figures.py          Figs. 2, 3a, 3b, 4
                 14_review_response.py  R = 2 pair enumeration, single-run pass rates (§4.4–4.5)
experiments/     one directory per arm (see below)
paper/figures/   figures as they appear in the paper
tests/           estimator tests
```

```bash
pip install -e ".[analysis,dev]"
pytest tests/
python scripts/09_e10_hierarchical.py   # reproduces 0.058 / 0.044 / 0.040
python scripts/08_e9_allocation.py
python scripts/14_review_response.py
```

### Arms

| Paper name | Directory | Role |
|---|---|---|
| `E4a`, `E12`, `E13`, `E11s`, `E11m` | `E4a`, `E12_vllm_nocache`, `E13_sglang_cache`, `E11_sglang_single`, `E11_sglang_multi` | engine × cache 2×2, replica control |
| `E4r`, `E12r` | `E4r`, `E12r_retail_nocache` | retail cache on / off |
| `E4t2`, `E12t` | `E4t2`, `E12t_telecom_nocache` | telecom cache on / off |
| `E4b2`, `E12b` | `E4b2`, `E12b_banking_nocache` | banking cache on / off |
| `E4a2` | `E4a2` | serialisation control |
| `E1b`, `E3ret` | `E1b`, `E3ret` | primary factorial, airline / retail (R = 5, 4 conditions) |
| Table 3 C2 / C4 / C5 | `E5*`, `E7*`, `E8*` | pairing generalisation |
| Table 4, App. F | `E9`, `E10` | replicate allocation, hierarchical cross-check |

### Data released

- **Per-cell records** (`raw/**/*.summary.json`) for every arm: reward, message count,
  termination, temperatures and seed. Every **reward-level** number in the paper can be
  recomputed from them.
- **Transcript-identity outputs** (`divergence.json`) for `E1b`, `E3ret`, `E4a` and `E4r`.
- **Full transcripts are not released.** `04_divergence.py`, `06_matched_compare.py` and
  `13_figures.py` read `*.full.json` transcripts. To recompute transcript-level results you need
  to regenerate them with τ²-bench at the pinned commit.

---

## Citation

```bibtex
@inproceedings{joshi2026prefixcache,
  title     = {Prefix Caching Can Silently Break Reproducibility in Interactive Agent Evaluation},
  author    = {Joshi, Prathamesh Dinesh and Dandekar, Raj Abhijit and Dandekar, Rajat and Panat, Sreedath},
  booktitle = {Workshop on Evaluation of Interactive Agents at NeurIPS 2026},
  year      = {2026}
}
```

## Licenses

τ²-bench: MIT. vLLM, SGLang: Apache 2.0. Qwen3 and Mistral checkpoints: Apache 2.0.
Hermes-3-Llama-3.1-8B: Llama 3 license. `gemma-3-12b-it`: Gemma Terms of Use.

---

## Ongoing work

We are continuing this line of work. This repository may be updated as new results become
available.
