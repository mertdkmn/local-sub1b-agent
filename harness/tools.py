"""The tool/action space and its Playwright executor.

RESEARCH_PLAN §8 locks an initial tool space of 8 tools. Each tool references an
element in the observation via ``ref`` (see observation.py). The executor is the
only place that touches the page — the agent never drives Playwright directly,
which is what makes failure injection possible later (P1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import Page

TOOL_NAMES = ("CLICK", "TYPE", "SCROLL", "GO_BACK", "GOTO", "SEARCH", "SUBMIT", "WAIT")


@dataclass(frozen=True)
class Action:
    tool: str
    args: dict[str, Any]


def _el(page: Page, ref: int):
    """Locator for the element the agent addressed by its data-ref index."""
    return page.locator(f'[data-ref="{ref}"]')


def _resolve(base_dir: Path, url: str) -> str:
    """Turn a fixture-relative URL (e.g. "fixtures/about.html") into a file:// URL."""
    if url.startswith(("http://", "https://", "file://")):
        return url
    return (base_dir / url).as_uri()


def execute(page: Page, action: Action, base_dir: Path) -> str:
    """Run one action against the live page.

    Returns "" on success, or an error message. The runner counts any non-empty
    return as a ``tool_error`` and feeds it back to the agent as its next
    observation (this is the "stale_obs"/"tool_error" signal the plan cares about).
    """
    a = action.args
    try:
        if action.tool == "CLICK":
            _el(page, a["ref"]).click()
        elif action.tool == "TYPE":
            _el(page, a["ref"]).fill(a["text"])
        elif action.tool == "SCROLL":
            dy = 500 if a.get("dir") == "down" else -500
            page.evaluate(f"window.scrollBy(0, {dy})")
        elif action.tool == "GO_BACK":
            # History always starts at [about:blank, start_url], so length <= 2
            # means the agent is still on the start page with nothing valid to go
            # back to — fail fast instead of navigating to about:blank (which
            # also destabilizes the frame for the next observation).
            if page.evaluate("history.length") <= 2:
                return "GO_BACK error: no history"
            page.go_back()
        elif action.tool == "GOTO":
            page.goto(_resolve(base_dir, a["url"]))
        elif action.tool == "SEARCH":
            # page.fill() auto-waits 30s for the selector; if the agent has
            # navigated away from index.html there is no search box, so check
            # existence first and fail fast with a useful message instead.
            box = page.locator("input[type='search']")
            if box.count() == 0:
                return "SEARCH error: no search box on page"
            box.fill(a["query"])
            box.press("Enter")
        elif action.tool == "SUBMIT":
            btn = page.locator("button[type='submit'], input[type='submit']")
            if btn.count() == 0:
                return "SUBMIT error: no submit button on page"
            btn.first.click()
        elif action.tool == "WAIT":
            page.wait_for_timeout(500)
        else:
            return f"unknown tool: {action.tool}"
    except Exception as exc:  # noqa: BLE001 — agent actions are untrusted input
        return f"{action.tool} error: {exc}"
    return ""
