"""Programmatic success checks — no LLM judge (RESEARCH_PLAN §7, §12).

An LLM judge would be expensive *and* circular (we'd be asking a model to tell us
whether the model did well). Instead each task ships a deterministic rule in its
``oracle`` field. This module is the only place those rules are interpreted.
"""
from __future__ import annotations

from playwright.sync_api import Page


def check(task, page: Page) -> bool:
    o = task.oracle
    t = o["type"]

    if t == "final_url":
        # Expected URL is matched as a *suffix*, so "fixtures/about.html"
        # matches the full "file://.../benchmark/fixtures/about.html".
        return page.url.endswith(o["expected"])

    if t == "form_value":
        # input_value() reads the *live* value of a text field (typed text,
        # not the HTML attribute), which is exactly what the agent produced.
        loc = page.locator(o["selector"])
        if loc.count() == 0:
            return False  # field not on page -> it can't hold the value
        return loc.input_value() == o["value"]

    if t == "dom_state":
        loc = page.locator(o["selector"])
        if o["condition"] == "visible":
            # is_visible() is False while the element has the `hidden` attribute,
            # True once the form's JS unhides it — the success signal we want.
            # A missing element is simply not visible.
            return loc.count() > 0 and loc.is_visible()
        raise ValueError(f"unknown condition: {o['condition']}")

    if t == "text":
        # The element's visible text equals ``expected`` (e.g. a cart counter
        # that has incremented to "1").
        loc = page.locator(o["selector"])
        if loc.count() == 0:
            return False
        return loc.first.inner_text().strip() == o["expected"]

    raise ValueError(f"unknown oracle type: {t}")
