"""CLI entry point: run an agent across every benchmark task → experiments.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from playwright.sync_api import sync_playwright

from .agents import RandomAgent
from .runner import CSV_COLUMNS, run_task
from .tasks import load_tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sub1b-agent eval harness v1")
    ap.add_argument("--agent", default="random", choices=["random"])
    ap.add_argument("--tasks-dir", default="benchmark/tasks")
    ap.add_argument("--benchmark-dir", default="benchmark")
    ap.add_argument("--out", default="experiments/experiments.csv")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    tasks = load_tasks(root / args.tasks_dir)
    benchmark_dir = root / args.benchmark_dir

    agent = {"random": RandomAgent()}[args.agent]

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for task in tasks:
            page = browser.new_page()
            row = run_task(task, agent, page, benchmark_dir)
            page.close()
            rows.append(row)
            print(
                f"{row['task_id']:12s} success={row['success']!s:5s} "
                f"steps={row['steps']} tool_errors={row['tool_errors']}"
            )
        browser.close()

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
