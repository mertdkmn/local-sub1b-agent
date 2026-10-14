"""Generate the HELD-OUT benchmark (10 tasks on a site the student never trained on).

Run:  .venv/bin/python benchmark/generate_heldout.py
Writes to benchmark/heldout_tasks/ (separate from benchmark/tasks/, the training set).

Purpose (P3): measure generalization. The "City Library" site uses the same
interaction primitives (links, search input, form fields, cart buttons) as the
training "Mini Shop" site, but different page names, labels, ref ordering, and a
new <textarea> element. A model that only memorized training refs will fail; a
model that learned the *skill* transfers.
"""
from __future__ import annotations

import json
from pathlib import Path


def _final_url(page: str) -> dict:
    return {"type": "final_url", "expected": f"heldout_fixtures/{page}.html"}


def _form_value(selector: str, value: str) -> dict:
    return {"type": "form_value", "selector": selector, "value": value}


def _visible(selector: str) -> dict:
    return {"type": "dom_state", "selector": selector, "condition": "visible"}


def _text(selector: str, expected: str) -> dict:
    return {"type": "text", "selector": selector, "expected": expected}


def t(task_id, category, goal, start_url, oracle, difficulty="easy", max_steps=5):
    return {
        "task_id": task_id,
        "category": category,
        "goal": goal,
        "start_url": start_url,
        "max_steps": max_steps,
        "difficulty": difficulty,
        "oracle": oracle,
        "failure_injection": [],
    }


HOME = "heldout_fixtures/home.html"
CATALOG = "heldout_fixtures/catalog.html"
CONTACT = "heldout_fixtures/contact.html"

TASKS = [
    # --- navigation ---
    t("ho-nav-001", "navigation", "Navigate to the Catalog page", HOME, _final_url("catalog")),
    t("ho-nav-002", "navigation", "Go to the Contact page", HOME, _final_url("contact")),
    t("ho-nav-003", "navigation", "Return to the Home page", CATALOG, _final_url("home")),

    # --- search ---
    t("ho-search-001", "search", "Search for 'dune'", HOME, _form_value("#search", "dune")),
    t("ho-search-002", "search", "Search for 'orwell'", HOME, _form_value("#search", "orwell")),

    # --- form (incl. a textarea the student never saw) ---
    t("ho-form-001", "form", "Fill the contact form (name, email and message) and submit it", CONTACT, _visible("#sent"), "medium", 10),
    t("ho-form-002", "form", "Type 'Alice' into the name field", CONTACT, _form_value("#name", "Alice")),
    t("ho-form-003", "form", "Type 'hello' into the message field", CONTACT, _form_value("#message", "hello")),

    # --- multi-step ---
    t("ho-multi-001", "multi_step", "Go to the Catalog page and add an item to the cart", HOME, _text("#cart-count", "1"), "medium", 10),
    t("ho-multi-002", "multi_step", "Go to the Catalog page and add two items to the cart", HOME, _text("#cart-count", "2"), "hard", 12),
]


def main() -> int:
    out = Path(__file__).parent / "heldout_tasks"
    out.mkdir(exist_ok=True)
    for task in TASKS:
        path = out / f"{task['task_id']}.json"
        path.write_text(json.dumps(task, indent=2) + "\n")
    print(f"wrote {len(TASKS)} held-out tasks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
