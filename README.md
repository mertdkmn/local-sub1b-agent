# sub1b-agent

How far can we push a **sub-1B parameter browser agent** under strict compute and memory constraints on **consumer hardware, locally**?

This is a research project. The goal is not to build a small browser agent, but to characterize and extend the **capability-efficiency frontier**: how much browser-agent capability we can get per parameter, per MB of memory, and per unit of compute, and which techniques move that frontier.

See [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) for the full plan, locked decisions, and phase breakdown.

## Paper

**"The Instability of Recovery-Trajectory Supervision in Sub-1B Browser Agents"** ([`paper/main.pdf`](paper/main.pdf))

A controlled study of whether adding recovery trajectories, episodes in which a
failure was injected and a teacher recovered, makes a 0.6B browser agent more
robust. The finding is that recovery supervision is unstable: the full recovery
condition is bimodal across training seeds, and a subset of runs collapses to a
policy that emits CLICK actions almost exclusively. Validation loss fails to
detect the collapse, but a simple action-distribution statistic does. For local
agent training, the practical conclusion is to select models with multiple seeds
and action-distribution monitoring rather than validation loss.

- **Composition ablation**: error-free versus error-containing recovery steps.
- **Data-mixing sweep**: the instability is non-monotonic in the recovery fraction.
- **Monitoring result**: validation loss is blind; the CLICK share detects collapse.
- **Dynamics**: the collapse is a late-training bifurcation.

The source is in [`paper/draft.md`](paper/draft.md); figures and LaTeX live under
[`paper/`](paper/).

## TL;DR

- **Student**: sub-1B pretrained Transformer (start with Qwen3-0.6B base), fine-tuned with LoRA.
- **Teacher**: Qwen3-4B (4-bit) generates trajectories + injected-failure recovery data.
- **Environment**: Playwright-controlled browser, a11y-tree observations, 8 tools.
- **Inference**: local on Apple Silicon via MLX (consumer hardware is the research constraint, not just a runtime).
- **Output**: a capability-efficiency frontier, logged to `experiments.csv`.

## Requirements

- Apple Silicon Mac (tested on M2 Pro, 16GB)
- Python 3.12
- Internet (the browser navigates the web; only the model's *thinking* is local)

## Quick start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install browser
playwright install chromium
```

## Project structure

```
benchmark/     # task definitions + static HTML fixtures
harness/       # eval harness: run models, measure metrics
data/          # teacher-generated trajectories → Dataset v1
training/      # SFT/LoRA training code
configs/       # per-experiment configs (single-variable sweeps)
experiments/   # experiments.csv + frontier plots
```

## Status

- [x] Scaffold + git init
- [x] Python environment (venv)
- [x] RESEARCH_PLAN.md
- [x] Dependencies (playwright, mlx-lm)
- [x] Benchmark schema + sample tasks
- [x] Eval harness v1 (RandomAgent + MLXAgent)
- [x] Zero-shot baseline (Qwen3-0.6B base, 33%)
- [x] Teacher (Qwen3-4B 4-bit) + 45-task benchmark (91%)
- [x] Trajectory collector + failure injection → Dataset v1 (P1)
- [x] P2 LoRA SFT: 0.6B student 35.6% → 95.6% std / 94% recovery (beats 91%/81% teacher, same-task)
- [x] P3 held-out: student 40% → 90% on unseen site (= teacher 90%); generalizes, not memorizing
- [x] P3 sweeps #1–#5 (data, loss-weight, observation format, tool space, form-state): all whack-a-mole/negative — see `experiments/P3_SWEEP6_STALEOBS.md` and RESEARCH_PLAN.md
- [x] P3 sweep #6 (stale_obs recovery): **rejected** — 27/51 std; 3-way ablation attributes the collapse to the 21 recovery-signal examples (not the missing clean data); see `experiments/P3_SWEEP6_STALEOBS.md`
- [x] P3 sweep #7 (quantization): **positive** — 8-bit free (48/51, −47% mem), 4-bit nearly free (47/51, −72% mem, +61% speed); see `experiments/P3_SWEEP7_QUANT.md`
