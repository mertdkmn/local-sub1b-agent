"""The agent loop (RESEARCH_PLAN §8) and the experiments.csv row it produces.

One call to ``run_task`` runs a single task to completion: load the start page,
then loop OBSERVE → ACT → (execute) → check oracle, until success or the step
budget is exhausted. It returns a flat dict whose keys are exactly the CSV
columns in RESEARCH_PLAN §11.
"""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import Page

from .agents import Agent
from .observation import observe, render
from .oracle import check
from .tools import TOOL_NAMES, execute

# One row per (run × task). Metadata columns first, then measured metrics —
# mirrors RESEARCH_PLAN §11 exactly.
CSV_COLUMNS = [
    "model_name", "param_count", "quantization", "observation_type",
    "context_length", "tool_count", "teacher", "training_method", "loss",
    "recovery_training", "task_id", "task_category", "failure_type",
    "success", "steps", "latency_ms", "tokens_per_sec", "peak_ram_mb",
    "model_size_mb", "tool_errors", "recovery_success",
]


def run_task(task, agent: Agent, page: Page, base_dir: Path) -> dict:
    start = time.perf_counter()
    agent.reset()
    # Systemic guard: cap every locator action at 3s. Playwright auto-waits up
    # to 30s by default; on local fixtures 3s is plenty, and it means no tool
    # can ever stall the harness (this is the "timeout" failure type).
    page.set_default_timeout(3000)
    page.goto((base_dir / task.start_url).as_uri())

    tool_errors = 0
    success = False
    steps = 0

    for steps in range(1, task.max_steps + 1):
        elements = observe(page)
        observation = render(elements)

        action = agent.act(task.goal, observation, elements)
        error = execute(page, action, base_dir)
        if error:
            tool_errors += 1

        if check(task, page):
            success = True
            break

    # The metadata fields that would identify a *real* model are placeholder
    # values for v1's fake agent. They become meaningful in v2 (MLX model) and
    # P2 (training). peak_ram_mb is model-memory-dependent too, so None for now.
    return {
        "model_name": agent.name,
        "param_count": getattr(agent, "param_count", 0),
        "quantization": getattr(agent, "quantization", "none"),
        "observation_type": "compact_interactive",
        "context_length": None,
        "tool_count": len(TOOL_NAMES),
        "teacher": None,
        "training_method": None,
        "loss": None,
        "recovery_training": False,
        "task_id": task.task_id,
        "task_category": task.category,
        "failure_type": None,
        "success": success,
        "steps": steps,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "tokens_per_sec": getattr(agent, "tokens_per_sec", 0),
        "peak_ram_mb": getattr(agent, "peak_ram_mb", None),
        "model_size_mb": getattr(agent, "model_size_mb", 0),
        "tool_errors": tool_errors,
        "recovery_success": None,
    }
