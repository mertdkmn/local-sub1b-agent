"""Re-render existing clean trajectories with the current observation format.

P3 sweep #3 (observation format) must isolate the FORMAT from everything else.
Re-collecting with the teacher would re-run the 4B model, which could pick
different actions and change the training targets and example count. Instead we
replay the SAME successful clean trajectories already recorded in
trajectories.jsonl — executing the identical stored actions — and swap only the
observation text to the current ``render()``. Example count, action distribution
and teacher success stay byte-identical; only the observation wording changes.

Run as ``python -m harness.rerender`` (mirrors ``harness.collect``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from .agents import SYSTEM_PROMPT, build_user_message
from .observation import build_observation, observe
from .tasks import load_tasks
from .tools import Action, execute


def replay_trajectory(traj: dict, task, page, base_dir: Path) -> list[dict]:
    """Replay one stored trajectory, re-rendering observations in the new format.

    Returns one single-turn SFT example per step (goal + history + observation →
    action), identical in every field to what ``Trajectory.to_sft_examples``
    produces, except the observation text now uses the current ``render()``.
    """
    page.set_default_timeout(3000)
    page.goto((base_dir / task.start_url).as_uri())

    steps = traj["steps"]
    examples: list[dict] = []
    history: list[str] = []

    for i, s in enumerate(steps):
        if s["action_tool"] == "UNPARSEABLE":
            continue
        elements = observe(page)
        page_name = page.url.rstrip("/").split("/")[-1]
        prev_error = steps[i - 1]["error"] if i > 0 else ""
        observation = build_observation(page_name, elements, prev_error)
        user = build_user_message(traj["goal"], history, observation)
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": s["action_call"]},
            ],
            "task_id": traj["task_id"],
            "category": traj["category"],
            "step": s["step"],
            "failure_type": s["failure_type"],
            "success": traj["success"],
        })
        history.append(s["action_label"])

        # Advance the page by replaying the recorded action. Its ref resolves
        # against the same DOM order as the original run, so it should hit the
        # same element; warn if the replayed outcome diverges from the record.
        action = Action(tool=s["action_tool"], args=s["action_args"])
        err = execute(page, action, base_dir)
        if err != s["error"]:
            print(
                f"  WARN {traj['task_id']} step {s['step']}: "
                f"replayed {err!r} != recorded {s['error']!r}"
            )

    return examples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sub1b-agent trajectory re-renderer (P3 sweep #3)")
    ap.add_argument("--traj-path", default="data/trajectories.jsonl")
    ap.add_argument("--tasks-dir", default="benchmark/tasks")
    ap.add_argument("--benchmark-dir", default="benchmark")
    ap.add_argument("--out", default="data/dataset_v1_obsformat.jsonl")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    traj_path = root / args.traj_path
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = {t.task_id: t for t in load_tasks(root / args.tasks_dir)}
    benchmark_dir = root / args.benchmark_dir

    n_traj = 0
    n_examples = 0
    action_dist: dict[str, int] = {}

    with traj_path.open() as f, out_path.open("w") as out, sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for line in f:
            traj = json.loads(line)
            if traj["failure_type"] is not None or not traj["success"]:
                continue  # only clean, successful trajectories emit SFT
            task = tasks[traj["task_id"]]
            page = browser.new_page()
            examples = replay_trajectory(traj, task, page, benchmark_dir)
            page.close()

            n_traj += 1
            for ex in examples:
                out.write(json.dumps(ex) + "\n")
                n_examples += 1
                tool = ex["messages"][-1]["content"].split("(")[0]
                action_dist[tool] = action_dist.get(tool, 0) + 1
        browser.close()

    print(f"replayed {n_traj} clean trajectories -> {n_examples} SFT examples -> {out_path}")
    print("action distribution:", dict(sorted(action_dist.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
