"""Load and validate benchmark task definitions.

A task is a JSON file in ``benchmark/tasks/``. The schema is documented in
``benchmark/task_schema.md``; this module only turns those files into a typed
``Task`` object the runner can use. We keep it a plain dataclass (not a pydantic
model) so the project has zero extra dependencies for now.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    task_id: str
    category: str
    goal: str
    start_url: str        # relative to benchmark/ (e.g. "fixtures/index.html")
    max_steps: int
    difficulty: str
    oracle: dict          # {"type": ..., ...} — see oracle.py
    failure_injection: list  # P1 feature; empty for now


def load_tasks(tasks_dir: Path) -> list[Task]:
    """Read every *.json in tasks_dir, sorted by filename, into Task objects."""
    tasks: list[Task] = []
    for path in sorted(tasks_dir.glob("*.json")):
        data = json.loads(path.read_text())
        tasks.append(
            Task(
                task_id=data["task_id"],
                category=data["category"],
                goal=data["goal"],
                start_url=data["start_url"],
                max_steps=data["max_steps"],
                difficulty=data["difficulty"],
                oracle=data["oracle"],
                failure_injection=data.get("failure_injection", []),
            )
        )
    return tasks
