# sub1b-agent

How far can we push a **sub-1B parameter browser agent** under strict compute and memory constraints on **consumer hardware, locally**?

This is a research project. The goal is not to build a small browser agent, but to characterize and extend the **capability-efficiency frontier** — how much browser-agent capability we can get per parameter, per MB of memory, and per unit of compute, and which techniques move that frontier.

See [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) for the full plan, locked decisions, and phase breakdown.

## TL;DR

- **Student**: sub-1B pretrained Transformer (start with Qwen3-0.6B-Instruct), fine-tuned with LoRA.
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
- [ ] Dependencies (playwright, mlx-lm)
- [ ] Benchmark schema + sample tasks
- [ ] Eval harness v1
- [ ] Zero-shot baseline (Qwen3-0.6B-Instruct)
