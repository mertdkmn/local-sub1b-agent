"""Generate the benchmark task set (pilot: ~45 tasks across 4 categories).

Run:  .venv/bin/python benchmark/generate_tasks.py
Writes one JSON file per task into benchmark/tasks/ following task_schema.md.

Each task uses a deterministic (programmatic) oracle — never an LLM judge
(RESEARCH_PLAN §7, §12). Categories and counts are a pilot; we scale to 100-300
later (P3) if promising.
"""
from __future__ import annotations

import json
from pathlib import Path


def _final_url(page: str) -> dict:
    return {"type": "final_url", "expected": f"fixtures/{page}.html"}


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


TASKS = [
    # --- navigation: reach a given page from a given start page ---
    t("nav-001", "navigation", "Navigate to the About page", "fixtures/index.html", _final_url("about")),
    t("nav-002", "navigation", "Go to the Products page", "fixtures/index.html", _final_url("products")),
    t("nav-003", "navigation", "Go to the Login page", "fixtures/index.html", _final_url("login")),
    t("nav-004", "navigation", "Return to the Home page", "fixtures/about.html", _final_url("index")),
    t("nav-005", "navigation", "Go to the Products page", "fixtures/login.html", _final_url("products")),
    t("nav-006", "navigation", "Go to the About page", "fixtures/products.html", _final_url("about")),
    t("nav-007", "navigation", "Go to the Home page", "fixtures/products.html", _final_url("index")),
    t("nav-008", "navigation", "Go to the Login page", "fixtures/products.html", _final_url("login")),
    t("nav-009", "navigation", "Go to the Home page", "fixtures/login.html", _final_url("index")),
    t("nav-010", "navigation", "Go to the About page", "fixtures/login.html", _final_url("about")),

    # --- search: type a query into the home search box ---
    t("search-001", "search", "Type 'notebook' into the search box", "fixtures/index.html", _form_value("#search-box", "notebook")),
    t("search-002", "search", "Search for 'phone'", "fixtures/index.html", _form_value("#search-box", "phone")),
    t("search-003", "search", "Search for 'tablet'", "fixtures/index.html", _form_value("#search-box", "tablet")),
    t("search-004", "search", "Search for 'laptop'", "fixtures/index.html", _form_value("#search-box", "laptop")),
    t("search-005", "search", "Search for 'headphones'", "fixtures/index.html", _form_value("#search-box", "headphones")),
    t("search-006", "search", "Search for 'camera'", "fixtures/index.html", _form_value("#search-box", "camera")),
    t("search-007", "search", "Search for 'keyboard'", "fixtures/index.html", _form_value("#search-box", "keyboard")),
    t("search-008", "search", "Search for 'monitor'", "fixtures/index.html", _form_value("#search-box", "monitor")),
    t("search-009", "search", "Search for 'speaker'", "fixtures/index.html", _form_value("#search-box", "speaker")),

    # --- form: fill one field, or fill+submit (success requires all fields) ---
    t("form-001", "form", "Fill the contact form (name and email) and submit it", "fixtures/index.html", _visible("#success"), "medium", 10),
    t("form-002", "form", "Log in by entering a username and password", "fixtures/login.html", _visible("#login-success"), "medium", 10),
    t("form-003", "form", "Type 'John' into the name field", "fixtures/index.html", _form_value("#name", "John")),
    t("form-004", "form", "Type 'alice@example.com' into the email field", "fixtures/index.html", _form_value("#email", "alice@example.com")),
    t("form-005", "form", "Type 'admin' into the username field", "fixtures/login.html", _form_value("#username", "admin")),
    t("form-006", "form", "Type 'secret123' into the password field", "fixtures/login.html", _form_value("#password", "secret123")),
    t("form-007", "form", "Fill the contact form with name 'Bob' and email 'bob@example.com' and submit", "fixtures/index.html", _visible("#success"), "medium", 10),
    t("form-008", "form", "Log in with username 'admin' and password 'admin123'", "fixtures/login.html", _visible("#login-success"), "medium", 10),
    t("form-009", "form", "Fill the contact form with name 'Carol' and email 'carol@example.com' and submit", "fixtures/index.html", _visible("#success"), "medium", 10),
    t("form-010", "form", "Type 'Eve' into the name field", "fixtures/index.html", _form_value("#name", "Eve")),
    t("form-011", "form", "Type 'hunter2' into the password field", "fixtures/login.html", _form_value("#password", "hunter2")),
    t("form-012", "form", "Log in with username 'user1' and password 'pass123'", "fixtures/login.html", _visible("#login-success"), "medium", 10),
    t("form-013", "form", "Type 'guest' into the username field", "fixtures/login.html", _form_value("#username", "guest")),
    t("form-014", "form", "Fill the contact form with name 'David' and email 'david@example.com' and submit", "fixtures/index.html", _visible("#success"), "medium", 10),
    t("form-015", "form", "Fill the contact form with name 'Emma' and email 'emma@example.com' and submit", "fixtures/index.html", _visible("#success"), "medium", 10),
    t("form-016", "form", "Fill the contact form with name 'Frank' and email 'frank@example.com' and submit", "fixtures/index.html", _visible("#success"), "medium", 10),
    t("form-017", "form", "Fill the contact form with name 'Grace' and email 'grace@example.com' and submit", "fixtures/index.html", _visible("#success"), "medium", 10),
    t("form-018", "form", "Fill the contact form with name 'Henry' and email 'henry@example.com' and submit", "fixtures/index.html", _visible("#success"), "medium", 10),
    t("form-019", "form", "Fill the contact form with name 'Ivy' and email 'ivy@example.com' and submit", "fixtures/index.html", _visible("#success"), "medium", 10),

    # --- multi-step: at least two distinct action types ---
    t("multi-001", "multi_step", "Go to the Products page and add an item to the cart", "fixtures/index.html", _text("#cart-count", "1"), "medium", 10),
    t("multi-002", "multi_step", "Go to the Login page and log in", "fixtures/index.html", _visible("#login-success"), "medium", 10),
    t("multi-003", "multi_step", "Go to the Products page and add two items to the cart", "fixtures/index.html", _text("#cart-count", "2"), "hard", 12),
    t("multi-004", "multi_step", "Go to the Login page and type 'admin' as the username", "fixtures/index.html", _form_value("#username", "admin"), "medium", 8),
    t("multi-005", "multi_step", "Go back to the Home page and search for 'tablet'", "fixtures/products.html", _form_value("#search-box", "tablet"), "medium", 10),
    t("multi-006", "multi_step", "Go to the Products page and add three items to the cart", "fixtures/index.html", _text("#cart-count", "3"), "hard", 12),
    t("multi-007", "multi_step", "Go to the Login page and type 'secret123' as the password", "fixtures/index.html", _form_value("#password", "secret123"), "medium", 8),
    t("multi-008", "multi_step", "Go to the Home page and search for 'phone'", "fixtures/login.html", _form_value("#search-box", "phone"), "medium", 8),
    t("multi-009", "multi_step", "Go to the Products page and add an item to the cart", "fixtures/login.html", _text("#cart-count", "1"), "medium", 10),
    t("multi-010", "multi_step", "Go to the Login page and type 'root' as the username", "fixtures/products.html", _form_value("#username", "root"), "medium", 8),
    t("multi-011", "multi_step", "Go to the Home page and fill the name field with 'Mike'", "fixtures/login.html", _form_value("#name", "Mike"), "medium", 8),
    t("multi-012", "multi_step", "Go to the Home page and search for 'laptop'", "fixtures/about.html", _form_value("#search-box", "laptop"), "medium", 10),
    t("multi-013", "multi_step", "Go to the Products page and add two items to the cart", "fixtures/login.html", _text("#cart-count", "2"), "hard", 12),
]


def main() -> int:
    out = Path(__file__).parent / "tasks"
    out.mkdir(exist_ok=True)
    for task in TASKS:
        path = out / f"{task['task_id']}.json"
        path.write_text(json.dumps(task, indent=2) + "\n")
    print(f"wrote {len(TASKS)} tasks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
