"""Agent interface + the v1 fake agent.

The harness is agent-agnostic: it only needs an object with ``act(goal,
observation, elements) -> Action``. v1 ships a ``RandomAgent`` that picks a random
tool + random ref, purely to exercise the loop end-to-end before a real model is
wired in (v2 / task #5).
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod

from .observation import Element
from .tools import Action, TOOL_NAMES


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def act(self, goal: str, observation: str, elements: list[Element]) -> Action:
        """Return the next action given the goal and current observation."""


class RandomAgent(Agent):
    """v1 fake agent: random action selection. Deliberately near-useless.

    It exists to prove the plumbing (loop, oracles, CSV) works, not to solve
    tasks. Its ~0% success rate is expected and is itself a sanity check: if a
    random agent *did* pass tasks, the oracles would be too weak.
    """

    name = "random"

    def act(self, goal: str, observation: str, elements: list[Element]) -> Action:
        refs = [el.ref for el in elements if not el.disabled]
        tool = random.choice(TOOL_NAMES)

        if tool == "CLICK" and refs:
            args = {"ref": random.choice(refs)}
        elif tool == "TYPE" and refs:
            args = {"ref": random.choice(refs), "text": "hello"}
        elif tool == "SCROLL":
            args = {"dir": random.choice(["up", "down"])}
        elif tool == "GOTO":
            args = {"url": random.choice(["fixtures/index.html", "fixtures/about.html"])}
        elif tool == "SEARCH":
            args = {"query": "notebook"}
        elif tool in ("GO_BACK", "SUBMIT", "WAIT"):
            args = {}
        else:
            # No usable refs (e.g. CLICK/TYPE with an empty tree) → wait.
            tool, args = "WAIT", {}

        return Action(tool=tool, args=args)
