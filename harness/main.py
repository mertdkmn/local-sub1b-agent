"""CLI entry point: run an agent across every benchmark task → experiments.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from playwright.sync_api import sync_playwright

from .agents import MLXAgent, RandomAgent
from .runner import CSV_COLUMNS, run_task
from .tasks import load_tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sub1b-agent eval harness v1")
    ap.add_argument("--agent", default="random", choices=["random", "mlx"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="model for the mlx agent")
    ap.add_argument("--enable-thinking", action="store_true", help="enable Qwen3 thinking mode (default off)")
    ap.add_argument("--name", default="qwen3-0.6b", help="model_name recorded in the CSV")
    ap.add_argument("--param-count", type=float, default=0.6, help="params in billions")
    ap.add_argument("--quantization", default="fp16", help="quantization level")
    ap.add_argument("--model-size-mb", type=int, default=1200, help="model size on disk in MB")
    ap.add_argument("--tasks-dir", default="benchmark/tasks")
    ap.add_argument("--benchmark-dir", default="benchmark")
    ap.add_argument("--out", default="experiments/experiments.csv")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    tasks = load_tasks(root / args.tasks_dir)
    benchmark_dir = root / args.benchmark_dir

    if args.agent == "random":
        agent = RandomAgent()
    else:
        agent = MLXAgent(
            args.model,
            name=args.name,
            param_count=args.param_count,
            quantization=args.quantization,
            model_size_mb=args.model_size_mb,
            enable_thinking=args.enable_thinking,
        )

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

    # Append to the shared experiments.csv (one row per run), writing the header
    # only when the file is new/empty — matches RESEARCH_PLAN §11.
    new_file = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
