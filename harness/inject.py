"""Failure injection for recovery trajectories (RESEARCH_PLAN §10).

The teacher produces a correct action; at a controlled point we intercept it and
substitute a failure *signal*. The teacher re-observes (its next observation
shows the error) and corrects. That "action → error → corrective action →
success" sequence is a *recovery trace*, tagged with a ``failure_type``.

Taxonomy (from the plan): wrong_target, stale_obs, wrong_arg, tool_error,
navigation, popup, timeout, missing_element.

P1 implements the signal-only subset — the injected action is *not* executed and
the page is left untouched, so every task stays solvable. This teaches "on
error, don't panic — retry", which is the core recovery lesson. Persistent
state-changing failures (stale_obs, navigation, popup) are a P3 single-variable
sweep and raise NotImplementedError so they can't be silently misused.
"""
from __future__ import annotations

from dataclasses import dataclass

from .tools import Action

KNOWN = {
    "wrong_target", "stale_obs", "wrong_arg", "tool_error",
    "navigation", "popup", "timeout", "missing_element",
}
# The error text each implemented type produces, appended to "<TOOL> error: ".
_SIGNAL_TEXT = {
    "tool_error": "simulated failure",
    "missing_element": "element not found",
    "timeout": "timeout",
    "stale_obs": "stale observation",
}


@dataclass
class FailureInjection:
    type: str
    step: int | None = None
    on_tool: str | None = None
    occurrence: int = 1

    def __post_init__(self) -> None:
        if self.type not in KNOWN:
            raise ValueError(f"unknown failure_type: {self.type}")
        if self.step is None and self.on_tool is None:
            raise ValueError("injection needs 'step' or 'on_tool'")


def parse_specs(raw: list) -> list[FailureInjection]:
    """Parse a task's ``failure_injection`` field (a list of dicts) into specs."""
    return [FailureInjection(**d) for d in raw]


def select(
    specs: list[FailureInjection],
    step: int,
    action: Action,
    seen: dict[int, int],
) -> FailureInjection | None:
    """Return the first spec that triggers at this step/action, or None.

    ``seen`` tracks per-spec ``on_tool`` occurrence counts across steps (keyed by
    id(spec)), so ``occurrence=1`` fires on the first matching action.
    """
    for spec in specs:
        if spec.step is not None:
            if step == spec.step:
                return spec
        elif spec.on_tool is not None and action.tool == spec.on_tool:
            n = seen.get(id(spec), 0) + 1
            seen[id(spec)] = n
            if n == spec.occurrence:
                return spec
    return None


def inject(action: Action, spec: FailureInjection, page=None) -> str | None:
    """Produce the injected failure's error signal in place of executing the action.

    ``stale_obs`` is state-changing: it removes the first element
    (``data-ref="0"``) so every later ref shifts down by one — the model's old
    observation no longer matches the page. Returns ``None`` to signal "skip the
    injection" when the action targets ref=0 (removing it would delete the target
    itself, which is a missing_element scenario, not stale_obs).
    """
    if spec.type in _SIGNAL_TEXT:
        if spec.type == "stale_obs" and page is not None:
            if action.args.get("ref") == 0:
                return None
            page.evaluate("document.querySelector('[data-ref=\"0\"]')?.remove()")
        return f"{action.tool} error: {_SIGNAL_TEXT[spec.type]}"
    raise NotImplementedError(
        f"failure_type '{spec.type}' is in the taxonomy but not implemented yet "
        f"(implemented: {sorted(_SIGNAL_TEXT)})"
    )
