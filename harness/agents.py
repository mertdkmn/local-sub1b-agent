"""Agent interface + the v1 fake agent.

The harness is agent-agnostic: it only needs an object with ``act(goal,
observation, elements) -> Action``. v1 ships a ``RandomAgent`` that picks a random
tool + random ref, purely to exercise the loop end-to-end before a real model is
wired in (v2 / task #5).
"""
from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from collections.abc import Callable

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from .observation import Element, label_by_ref
from .tools import Action, TOOL_NAMES


class Agent(ABC):
    name: str = "agent"

    def reset(self) -> None:
        """Reset per-task metrics. No-op by default; MLXAgent overrides."""

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


SYSTEM_PROMPT = (
    "You are a web-browser automation agent. You are given a goal and a list of "
    "the interactive elements currently on the page. Decide on the single next "
    "action and reply with exactly one tool call, and nothing else.\n"
    "Tools:\n"
    "- CLICK(ref=N)\n"
    '- TYPE(ref=N, "text")\n'
    "- SCROLL(up) or SCROLL(down)\n"
    "- GO_BACK()\n"
    "- GOTO(url)\n"
    "- SEARCH(query)\n"
    "- SUBMIT()\n"
    "- WAIT()\n"
    "Use TYPE to fill a text field, CLICK for buttons and links. Fill every "
    "required field before submitting a form.\n"
    'Examples: CLICK(ref=3), TYPE(ref=5, "hello")'
)


def _strip_thinking(text: str) -> str:
    """Drop Qwen3's ``<think>…</think>`` reasoning block (or an unclosed one)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


# (regex, builder) pairs tried in order; first match wins. Case-insensitive so
# the model may emit "click(ref=3)" or "CLICK(ref = 3)" and still parse.
_PARSERS: list[tuple[re.Pattern, Callable]] = [
    (re.compile(r"CLICK\s*\(\s*(?:ref\s*=\s*)?(\d+)\s*\)", re.I),
     lambda m: Action("CLICK", {"ref": int(m.group(1))})),
    (re.compile(r'TYPE\s*\(\s*(?:ref\s*=\s*)?(\d+)\s*,\s*["\'](.+?)["\']\s*\)', re.I),
     lambda m: Action("TYPE", {"ref": int(m.group(1)), "text": m.group(2)})),
    (re.compile(r"SCROLL\s*\(\s*(up|down)\s*\)", re.I),
     lambda m: Action("SCROLL", {"dir": m.group(1).lower()})),
    (re.compile(r"GO_BACK\s*\(\s*\)", re.I),
     lambda m: Action("GO_BACK", {})),
    (re.compile(r"GOTO\s*\(\s*(\S+?)\s*\)", re.I),
     lambda m: Action("GOTO", {"url": m.group(1)})),
    (re.compile(r'SEARCH\s*\(\s*["\']?([^"\')]+)["\']?\s*\)', re.I),
     lambda m: Action("SEARCH", {"query": m.group(1).strip()})),
    (re.compile(r"SUBMIT\s*\(\s*\)", re.I),
     lambda m: Action("SUBMIT", {})),
    (re.compile(r"WAIT\s*\(\s*\)", re.I),
     lambda m: Action("WAIT", {})),
]


def parse_action(text: str) -> Action | None:
    """Return the first well-formed tool call found in ``text``, else None."""
    text = _strip_thinking(text)
    for rx, build in _PARSERS:
        m = rx.search(text)
        if m:
            return build(m)
    return None


def action_to_label(action: Action, labels: dict[int, str]) -> str:
    """Render an Action back into self-descriptive text for the history.

    The ref is replaced by the element's label (e.g. "CLICK(Products)" instead of
    "CLICK(ref=2)"), so the model can see *what* it acted on — refs are opaque
    and shift between pages.
    """
    a = action.args
    ref = a.get("ref")
    label = labels.get(ref, f"ref={ref}") if ref is not None else ""
    if action.tool == "CLICK":
        return f"CLICK({label})"
    if action.tool == "TYPE":
        return f"TYPE({label}, '{a.get('text')}')"
    if action.tool == "SCROLL":
        return f"SCROLL({a.get('dir')})"
    if action.tool == "GOTO":
        return f"GOTO({a.get('url')})"
    if action.tool == "SEARCH":
        return f"SEARCH({a.get('query')})"
    if action.tool in ("GO_BACK", "SUBMIT", "WAIT"):
        return f"{action.tool}()"
    return f"{action.tool}({a})"


def action_to_call(action: Action) -> str:
    """Render an Action into its canonical *parseable* tool call — the SFT target.

    This is the inverse of ``parse_action``: the exact string the student is
    trained to emit (e.g. "CLICK(ref=3)", "TYPE(ref=5, 'hello')").
    """
    a = action.args
    if action.tool == "CLICK":
        return f"CLICK(ref={a.get('ref')})"
    if action.tool == "TYPE":
        return f"TYPE(ref={a.get('ref')}, {a.get('text')!r})"
    if action.tool == "SCROLL":
        return f"SCROLL({a.get('dir')})"
    if action.tool == "GOTO":
        return f"GOTO({a.get('url')})"
    if action.tool == "SEARCH":
        return f"SEARCH({a.get('query')})"
    if action.tool in ("GO_BACK", "SUBMIT", "WAIT"):
        return f"{action.tool}()"
    return f"{action.tool}({a})"


def build_user_message(goal: str, history: list[str], observation: str) -> str:
    """Assemble the user turn the model conditions on (goal + history + observation).

    Extracted from ``MLXAgent.act`` so the trajectory collector / SFT converter
    build the *identical* prompt — otherwise training data would be
    out-of-distribution (RESEARCH_PLAN §4: teacher and student share the interface).
    """
    user = f"GOAL: {goal}\n\n"
    if history:
        user += "Previous actions:\n" + "\n".join(f"- {a}" for a in history) + "\n\n"
    user += "Current page (interactive elements only):\n" + (observation or "(no interactive elements)")
    return user


class MLXAgent(Agent):
    """v2 agent: a Qwen3 sub-1B model emitting tool calls from an a11y observation.

    The checkpoint is loaded once at construction and reused across tasks. Each
    ``act`` renders the goal + observation into a chat prompt, generates
    greedily (temp=0) with Qwen3 thinking off by default (``enable_thinking``),
    and strips any stray ``<think>`` block as a safety net. Unparseable output
    is returned as an
    ``UNPARSEABLE`` action so the runner records a ``tool_error`` rather than
    silently advancing.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-0.6B",
        name: str = "qwen3-0.6b",
        param_count: float = 0.6,
        quantization: str = "fp16",
        model_size_mb: int = 1200,
        max_tokens: int = 512,
        temp: float = 0.0,
        enable_thinking: bool = False,
    ):
        self.model, self.tokenizer = load(model_path)
        self.name = name
        self.param_count = param_count
        self.quantization = quantization
        self.model_size_mb = model_size_mb
        self.max_tokens = max_tokens
        self.temp = temp
        self.enable_thinking = enable_thinking
        self.reset()

    def reset(self) -> None:
        """Reset per-task metrics; the runner calls this at the start of each task."""
        self._gen_tokens = 0
        self._gen_seconds = 0.0
        self._peak_ram_mb = 0.0
        self.history: list[str] = []
        self.last_raw: str | None = None

    @property
    def tokens_per_sec(self) -> float:
        if self._gen_seconds <= 0:
            return 0.0
        return round(self._gen_tokens / self._gen_seconds, 1)

    @property
    def peak_ram_mb(self) -> float | None:
        return round(self._peak_ram_mb) if self._peak_ram_mb > 0 else None

    def act(self, goal: str, observation: str, elements: list[Element]) -> Action:
        user = build_user_message(goal, self.history, observation)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        raw = ""
        last = None
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temp),
        ):
            raw += resp.text
            last = resp

        if last is not None:
            self._gen_tokens += last.generation_tokens
            if last.generation_tps:
                self._gen_seconds += last.generation_tokens / last.generation_tps
            self._peak_ram_mb = max(
                self._peak_ram_mb, mx.get_peak_memory() / (1024 * 1024)
            )

        self.last_raw = raw
        action = parse_action(raw)
        if action is None:
            return Action("UNPARSEABLE", {})
        self.history.append(action_to_label(action, label_by_ref(elements)))
        return action
