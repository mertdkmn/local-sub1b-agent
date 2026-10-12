"""Trajectory collector: run the teacher, record every step, inject failures (P1).

RESEARCH_PLAN §7, §10. Two artifacts per run, both JSONL in data/:
  - trajectories.jsonl : one full trajectory per line (raw, for analysis/replay)
  - dataset_v1.jsonl   : single-turn SFT examples (the training set)

Run as ``python -m harness.collect`` (mirrors ``python -m harness.main`` for eval).
The teacher is the ~4B model; default ``Qwen/Qwen3-4B-MLX-4bit`` (4-bit, base).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from .agents import MLXAgent, action_to_call, action_to_label
from .inject import FailureInjection, inject, parse_specs, select
from .observation import build_observation, label_by_ref, observe
from .oracle import check
from .tasks import load_tasks
from .tools import execute
from .trajectory import Step, Trajectory


def collect_trajectory(
    task, agent: MLXAgent, page, base_dir: Path, specs: list[FailureInjection]
) -> Trajectory:
    """Run one task to completion, recording every (observation → action) step.

    Differs from ``runner.run_task`` only in what it records: it feeds each
    step's error back into the *next* observation (the recovery signal) and
    appends a full ``Step`` per iteration. Injection intercepts an action and
    substitutes an error signal instead of executing it.
    """
    agent.reset()
    page.set_default_timeout(3000)
    page.goto((base_dir / task.start_url).as_uri())

    steps: list[Step] = []
    last_error = ""
    seen: dict[int, int] = {}
    failure_type: str | None = None
    success = False

    for i in range(1, task.max_steps + 1):
        elements = observe(page)
        page_name = page.url.rstrip("/").split("/")[-1]
        observation = build_observation(page_name, elements, last_error)

        action = agent.act(task.goal, observation, elements)

        spec = select(specs, i, action, seen)
        if spec is not None:
            error = inject(action, spec)
            injected = True
            failure_type = failure_type or spec.type
        else:
            error = execute(page, action, base_dir)
            injected = False

        unparseable = action.tool == "UNPARSEABLE"
        steps.append(
            Step(
                step=i,
                observation=observation,
                action_tool=action.tool,
                action_args=dict(action.args),
                action_call=action_to_call(action) if not unparseable else "UNPARSEABLE",
                action_label=action_to_label(action, label_by_ref(elements)) if not unparseable else "",
                error=error,
                raw=getattr(agent, "last_raw", None),
                injected=injected,
                failure_type=spec.type if spec else None,
            )
        )

        last_error = error
        if check(task, page):
            success = True
            break

    return Trajectory(
        task_id=task.task_id,
        category=task.category,
        goal=task.goal,
        agent_name=agent.name,
        param_count=getattr(agent, "param_count", 0),
        quantization=getattr(agent, "quantization", "none"),
        failure_type=failure_type,
        success=success,
        recovery_success=(success if failure_type else None),
        steps=steps,
    )


def parse_inject_arg(s: str) -> FailureInjection:
    """Parse ``--inject type@step`` (e.g. ``tool_error@2``) or ``type@tool``
    (e.g. ``missing_element@CLICK``)."""
    if "@" not in s:
        raise ValueError(f"--inject needs 'type@step' or 'type@tool', got {s!r}")
    ftype, target = s.split("@", 1)
    if target.isdigit():
        return FailureInjection(type=ftype, step=int(target))
    return FailureInjection(type=ftype, on_tool=target.upper())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sub1b-agent trajectory collector (P1)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-MLX-4bit")
    ap.add_argument("--name", default="qwen3-4b")
    ap.add_argument("--param-count", type=float, default=4.0)
    ap.add_argument("--quantization", default="4bit")
    ap.add_argument("--model-size-mb", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--enable-thinking", action="store_true", help="default off (teacher is better without)")
    ap.add_argument("--tasks-dir", default="benchmark/tasks")
    ap.add_argument("--benchmark-dir", default="benchmark")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument(
        "--inject",
        action="append",
        default=None,
        help="inject a failure into every task: 'tool_error@2' or 'missing_element@CLICK' (repeatable)",
    )
    ap.add_argument("--max-tasks", type=int, default=0, help="limit tasks (0 = all)")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    tasks = load_tasks(root / args.tasks_dir)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    benchmark_dir = root / args.benchmark_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cli_specs = [parse_inject_arg(s) for s in (args.inject or [])]

    agent = MLXAgent(
        args.model,
        name=args.name,
        param_count=args.param_count,
        quantization=args.quantization,
        model_size_mb=args.model_size_mb,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
    )

    traj_path = out_dir / "trajectories.jsonl"
    sft_path = out_dir / "dataset_v1.jsonl"
    n_traj = 0
    n_examples = 0
    by_type: dict[str, int] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for task in tasks:
            specs = parse_specs(task.failure_injection) + cli_specs
            page = browser.new_page()
            traj = collect_trajectory(task, agent, page, benchmark_dir, specs)
            page.close()

            n_traj += 1
            examples = traj.to_sft_examples()
            n_examples += len(examples)
            if traj.failure_type:
                by_type[traj.failure_type] = by_type.get(traj.failure_type, 0) + 1

            with traj_path.open("a") as f:
                f.write(json.dumps(traj.to_dict()) + "\n")
            with sft_path.open("a") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            print(
                f"{traj.task_id:12s} success={str(traj.success):5s} "
                f"steps={len(traj.steps)} sft={len(examples)} "
                f"failure={traj.failure_type or '-'} recovery={traj.recovery_success}"
            )
        browser.close()

    print(f"\nwrote {n_traj} trajectories -> {traj_path}")
    print(f"wrote {n_examples} SFT examples -> {sft_path}")
    if by_type:
        print("recovery trajectories by failure_type:", by_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
