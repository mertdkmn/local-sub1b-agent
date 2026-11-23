#!/usr/bin/env python3
"""Generate the five paper figures (bimodality, mixing, actions, val-loss, dynamics)."""
import csv
import glob
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})


def success(f):
    return sum(1 for r in csv.DictReader(open(f)) if r["success"] == "True")


def seed_vals(name):
    files = [f"experiments/p3_{name}.csv"] + sorted(
        glob.glob(f"experiments/p3_{name}_seed*.csv"))
    return [success(f) for f in files if os.path.exists(f)]


# ---------------------------------------------------------------- Fig 1: bimodality
CONDS = [("A", "clean84"), ("B", "clean84_plus34"),
         ("C", "clean84_recovery21"), ("D", "staleobs")]
COND_LABELS = {"A": "A\nclean", "B": "B\n+clean-like",
               "C": "C\n+recovery", "D": "D\nboth"}
COLORS = {"A": "tab:gray", "B": "tab:blue", "C": "tab:orange", "D": "tab:red"}
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(6, 3.6))
for i, (k, name) in enumerate(CONDS):
    vals = np.array(seed_vals(name))
    x = i + rng.uniform(-0.13, 0.13, len(vals))
    ax.scatter(x, vals, s=42, alpha=0.85, edgecolors="black", linewidths=0.6,
               zorder=3, color=COLORS[k])
    m, s = vals.mean(), vals.std()
    ax.hlines(m, i - 0.28, i + 0.28, color="black", linewidth=2.2, zorder=4)
    ax.add_patch(plt.Rectangle((i - 0.28, m - s), 0.56, 2 * s,
                               color="gray", alpha=0.13, zorder=1))
ax.set_xticks(range(len(CONDS)))
ax.set_xticklabels([COND_LABELS[k] for k, _ in CONDS])
ax.set_ylabel("Task success (out of 51)")
ax.set_ylim(10, 55)
ax.set_title("Bimodal outcomes in the full recovery condition (D)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_bimodality.png", dpi=300)
fig.savefig(f"{OUT}/fig1_bimodality.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: mixing curve
# (fraction, prefix) — values loaded from CSVs
MIX = [(0.0, "clean84"), (20.0, "clean84_recovery21"), (28.8, "clean84_plus34"),
       (32.8, "mix33"), (36.4, "mix36"), (39.6, "staleobs"),
       (45.8, "mix46"), (50.0, "mix50")]
fig, ax = plt.subplots(figsize=(6, 3.4))
for frac, p in MIX:
    vals = np.array(seed_vals(p))
    m, s = vals.mean(), vals.std()
    x = frac + rng.uniform(-0.6, 0.6, len(vals))
    ax.scatter(x, vals, s=30, alpha=0.8, edgecolors="black",
               linewidths=0.4, color="tab:red" if s > 10 else "tab:blue", zorder=3)
    ax.errorbar(frac, m, yerr=s, fmt="o", color="black", markersize=0,
                capsize=3, linewidth=1.2, zorder=4)
ax.set_xlabel("Recovery-derived fraction (%)")
ax.set_ylabel("Mean task success (out of 51)")
ax.set_ylim(20, 55)
ax.set_title("Instability is non-monotonic in the recovery-derived fraction")
ax.annotate("bimodal window", xy=(38, 30), xytext=(42, 26),
            fontsize=8, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_mixing.png", dpi=300)
fig.savefig(f"{OUT}/fig2_mixing.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: actions
def action_counts(path):
    rows = [json.loads(l) for l in open(path)]
    c = Counter()
    for r in rows:
        for s in r.get("steps", []):
            c[s.get("action_tool") or "OTHER"] += 1
    return c

DIST = {
    "A": "data/dist_clean84/trajectories.jsonl",
    "B": "data/dist_clean84_plus34/trajectories.jsonl",
    "C": "data/dist_clean84_recovery21/trajectories.jsonl",
    "D (healthy)": "data/dist_staleobs_seed3b/trajectories.jsonl",
    "D (collapse)": "data/dist_staleobs_seed4/trajectories.jsonl",
}
TOOLS = ["CLICK", "TYPE", "SEARCH", "SUBMIT"]
TCOL = {"CLICK": "tab:red", "TYPE": "tab:blue",
        "SEARCH": "tab:green", "SUBMIT": "tab:purple"}
fig, ax = plt.subplots(figsize=(6.4, 3.4))
labels = list(DIST)
bottom = np.zeros(len(labels))
for t in TOOLS:
    fracs = []
    for lb in labels:
        c = action_counts(DIST[lb])
        tot = sum(c.values()) or 1
        fracs.append(100 * c.get(t, 0) / tot)
    ax.bar(labels, fracs, bottom=bottom, color=TCOL[t], label=t, alpha=0.9)
    bottom += np.array(fracs)
ax.set_ylabel("Share of actions (%)")
ax.set_ylim(0, 100)
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_actions.png", dpi=300)
fig.savefig(f"{OUT}/fig3_actions.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: val-loss blindness
# (val_loss, success) pairs extracted from training logs
VAL = [  # (cond, val, success)
    ("A", 0.263, 46), ("A", 0.676, 45), ("A", 0.575, 40), ("A", 0.318, 48),
    ("B", 0.197, 51), ("B", 0.461, 45), ("B", 0.084, 44), ("B", 0.169, 43),
    ("C", 0.685, 37), ("C", 0.229, 46), ("C", 0.670, 41), ("C", 0.161, 45), ("C", 0.184, 41),
    ("D", 0.104, 40), ("D", 0.362, 44), ("D", 0.217, 47), ("D", 0.178, 16),
]
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for cond, v, s in VAL:
    collapse = s < 32
    ax.scatter(v, s, s=55, color=COLORS[cond],
               edgecolors="black", linewidths=0.5,
               marker=("x" if collapse else "o"), zorder=3)
ax.set_xlabel("Validation loss")
ax.set_ylabel("Task success (out of 51)")
ax.set_title("Validation loss is blind to collapse")
ax.annotate("collapsed seed\n(lowest val loss)", xy=(0.178, 16), xytext=(0.28, 12),
            fontsize=7, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_valloss.png", dpi=300)
fig.savefig(f"{OUT}/fig4_valloss.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 5: dynamics
it100 = []
for s in range(5):
    rows = [json.loads(l) for l in open(f"data/iter100_seed{s}/trajectories.jsonl")]
    it100.append(sum(1 for r in rows if r.get("success")))
final = seed_vals("staleobs")[:5]
fig, ax = plt.subplots(figsize=(4.2, 3.6))
for s, (x, y) in enumerate(zip(it100, final)):
    collapse = y < 32
    ax.scatter(x, y, s=70, color=("tab:red" if collapse else "tab:green"),
               edgecolors="black", linewidths=0.6, zorder=3)
    ax.annotate(f"seed {s}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.plot([10, 45], [10, 45], ls="--", color="gray", lw=1, alpha=0.6)
ax.set_xlabel("Iteration-100 success (out of 51)")
ax.set_ylabel("Final (iteration-200) success")
ax.set_title("Late bifurcation: iter-100 does not predict collapse")
ax.set_xlim(10, 45); ax.set_ylim(10, 51)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_dynamics.png", dpi=300)
fig.savefig(f"{OUT}/fig5_dynamics.pdf")
plt.close(fig)

print("figures written to", OUT)
for f in sorted(glob.glob(f"{OUT}/*.png")):
    print(" ", os.path.basename(f))
