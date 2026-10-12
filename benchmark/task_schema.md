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

### `text`

Task succeeds if an element's visible text equals `expected` (e.g. a cart counter
that has incremented to `"1"`).

```json
{ "type": "text", "selector": "#cart-count", "expected": "1" }
```

## Failure injection

`failure_injection` is a list of specs. Each spec tells the trajectory collector
to intercept the teacher's action at a controlled point and substitute a failure
signal, producing a *recovery trace* tagged with `failure_type` (RESEARCH_PLAN
§10). Empty list = clean trajectory only.

A spec is one of:

```json
{ "type": "tool_error", "step": 2 }
{ "type": "missing_element", "on_tool": "CLICK", "occurrence": 1 }
```

- `type` — one of the taxonomy: `wrong_target`, `stale_obs`, `wrong_arg`,
  `tool_error`, `navigation`, `popup`, `timeout`, `missing_element`. P1
  implements the signal-only subset `tool_error` / `missing_element` / `timeout`;
  the rest raise `NotImplementedError`.
- `step` — fire at the absolute step index (1-based).
- `on_tool` + `occurrence` — fire on the Nth time the teacher chooses that tool.

The injected action is **not** executed; the teacher sees the error in its next
observation and must recover.
