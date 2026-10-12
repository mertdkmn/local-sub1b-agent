"""Trajectory data model, serialization, and SFT conversion.

RESEARCH_PLAN §7: the trajectory collector records every (observation → action)
step of a teacher run; that record *is* the training data. This module defines
the on-disk shape of a trajectory and turns one into single-turn SFT examples
using the *same* prompt format MLXAgent uses at inference (§4: teacher and
student share the interface, else distillation is out-of-distribution).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .agents import SYSTEM_PROMPT, build_user_message


@dataclass
class Step:
    step: int
    observation: str          # what the model conditioned on (page + a11y + error feedback)
    action_tool: str
    action_args: dict[str, Any]
    action_call: str          # canonical parseable target, e.g. "CLICK(ref=3)"
    action_label: str         # label-resolved form for history, e.g. "CLICK(Products)"
    error: str                # "" or the error this action produced (feeds the next step)
    raw: str | None           # raw model output, for debugging
    injected: bool
    failure_type: str | None


@dataclass
class Trajectory:
    task_id: str
    category: str
    goal: str
    agent_name: str
    param_count: float
    quantization: str
    failure_type: str | None  # None = clean trace; else the injected failure type
    success: bool
    recovery_success: bool | None
    steps: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_sft_examples(self) -> list[dict]:
        """One single-turn example per step: (goal + history + observation) → action.

        History is rebuilt from prior steps' ``action_label`` exactly as MLXAgent
        maintains ``self.history``. Unparseable steps are skipped (nothing to
        teach) and do not advance history, matching the agent's real behavior.

        Only *successful* trajectories contribute examples: a failed trajectory
        (natural teacher failure, or a recovery the teacher couldn't complete)
        ends in wrong actions by definition, so it has no correct target to teach.
        Failed trajectories are still saved to trajectories.jsonl for analysis;
        they just don't reach the SFT set.
        """
        if not self.success:
            return []
        history: list[str] = []
        examples: list[dict] = []
        for s in self.steps:
            if s.action_tool == "UNPARSEABLE":
                continue
            user = build_user_message(self.goal, history, s.observation)
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": s.action_call},
                ],
                "task_id": self.task_id,
                "category": self.category,
                "step": s.step,
                "failure_type": self.failure_type,
                "success": self.success,
            })
            history.append(s.action_label)
        return examples
