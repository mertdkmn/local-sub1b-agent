# Benchmark Task Schema

A task describes one thing an agent must accomplish in the browser, plus how to
judge success **deterministically** (no LLM-judge).

## Fields

| Field | Type | Meaning |
|---|---|---|
| `task_id` | string | Unique id, e.g. `nav-001` |
| `category` | string | `navigation` / `search` / `form` / `multi_step` / `recovery` |
| `goal` | string | Natural-language instruction the agent is given |
| `start_url` | string | Fixture to load at the start (relative to `benchmark/`) |
| `max_steps` | int | Step budget; agent fails if it exceeds this |
| `difficulty` | string | `easy` / `medium` / `hard` |
| `oracle` | object | How to check success (see below) |
| `failure_injection` | list | P1 feature; specs for injected failures (empty for now) |

## Oracle types

### `final_url`

Task succeeds if the browser ends on a given page. `expected` is matched as a
**suffix** of the page URL, so `fixtures/about.html` matches the full
`file://...` path regardless of the absolute prefix.

```json
{ "type": "final_url", "expected": "fixtures/about.html" }
```

### `form_value`

Task succeeds if a form field holds a given value.

```json
{ "type": "form_value", "selector": "#search-box", "value": "notebook" }
```

### `dom_state`

Task succeeds if an element matches a condition.

```json
{ "type": "dom_state", "selector": "#success", "condition": "visible" }
```

`condition` is `visible` for now; more conditions (e.g. `contains`) can be added
as needed.
