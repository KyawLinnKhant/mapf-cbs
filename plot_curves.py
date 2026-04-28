"""
Learning Curve Plotter
======================
Parses the training log and produces a 3-panel figure:
  - Goals reached vs steps
  - Collisions vs steps
  - Entropy vs steps
Phase boundaries (A/B/C) and curriculum transitions are annotated.

Usage
-----
  python plot_curves.py                              # auto-finds training log
  python plot_curves.py --log training_log.txt       # explicit log file
  python plot_curves.py --out results/curves.png     # custom output path
  python plot_curves.py --smooth 20                  # rolling-window smoothing
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Log parsing ───────────────────────────────────────────────────────────────

_LOG_LINE = re.compile(
    r"step=\s*([\d,]+)\s*\|.*?lvl=(\S+)\s*\|.*?ph=([ABC])\s+cbs=([\d.]+)"
    r"\s*\|.*?goals=([\d.]+)\s*\|.*?coll=([\d.]+)\s*\|.*?ent=([\d.]+)"
)
_TRANSITION = re.compile(
    r"\[Curriculum\s*[↑↓]\]\s*→\s*(\S+)\s*\(step\s*([\d,]+)"
)


def parse_log(path: str) -> Tuple[Dict[str, List], List[Tuple[int, str]]]:
    """Return (records_dict, transitions_list)."""
    records: Dict[str, List] = {
        "step": [], "level": [], "phase": [], "cbs_w": [],
        "goals": [], "collisions": [], "entropy": [],
    }
    transitions: List[Tuple[int, str]] = []  # (step, level_name)

    with open(path, "r") as f:
        for line in f:
            m = _LOG_LINE.search(line)
            if m:
                records["step"].append(int(m.group(1).replace(",", "")))
                records["level"].append(m.group(2))
                records["phase"].append(m.group(3))
                records["cbs_w"].append(float(m.group(4)))
                records["goals"].append(float(m.group(5)))
                records["collisions"].append(float(m.group(6)))
                records["entropy"].append(float(m.group(7)))
                continue
            t = _TRANSITION.search(line)
            if t:
                transitions.append((
                    int(t.group(2).replace(",", "")),
                    t.group(1),
                ))

    return records, transitions


def smooth(values: List[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.array(values)
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


# ── Plotting ──────────────────────────────────────────────────────────────────

PHASE_COLORS = {"A": "#f39c12", "B": "#3498db", "C": "#2ecc71"}
LEVEL_COLORS = {
    "easy": "#aaaaaa", "medium": "#e67e22",
    "hard": "#e74c3c", "expert": "#9b59b6",
}

BG = "#1a1a2e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#cccccc"


def plot_curves(
    records: Dict[str, List],
    transitions: List[Tuple[int, str]],
    smooth_window: int = 10,
    out_path: str = "results/learning_curves.png",
    total_steps: int = 5_000_000,
) -> None:
    steps = np.array(records["step"])
    goals = smooth(records["goals"], smooth_window)
    colls = smooth(records["collisions"], smooth_window)
    entrs = smooth(records["entropy"], smooth_window)
    phases = records["phase"]
    cbs_w = np.array(records["cbs_w"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor(BG)

    panel_data = [
        (axes[0], goals,  records["goals"],  "Goals Reached (lifelong)", "#2ecc71"),
        (axes[1], colls,  records["collisions"], "Collisions / Episode",  "#e74c3c"),
        (axes[2], entrs,  records["entropy"], "Policy Entropy",           "#3498db"),
    ]

    phase_boundaries = {"A": 0, "B": None, "C": None}
    for i, ph in enumerate(phases):
        if ph == "B" and phase_boundaries["B"] is None:
            phase_boundaries["B"] = steps[i]
        if ph == "C" and phase_boundaries["C"] is None:
            phase_boundaries["C"] = steps[i]

    for ax, smoothed, raw, ylabel, color in panel_data:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5)

        # Raw values as faint scatter
        ax.scatter(steps, raw, s=2, alpha=0.2, color=color)
        # Smoothed line
        ax.plot(steps, smoothed, color=color, linewidth=1.8, zorder=4)

        # Phase boundary shading
        pb = phase_boundaries
        regions = [
            (0,       pb["B"] or total_steps, "A"),
            (pb["B"] or total_steps, pb["C"] or total_steps, "B"),
            (pb["C"] or total_steps, total_steps, "C"),
        ]
        for x0, x1, ph in regions:
            if x0 >= x1:
                continue
            ax.axvspan(x0, x1, alpha=0.06, color=PHASE_COLORS[ph], zorder=1)
            mid = (x0 + x1) / 2
            if mid <= steps[-1]:
                ax.text(
                    mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                    f"Phase {ph}", color=PHASE_COLORS[ph],
                    fontsize=8, ha="center", va="top", alpha=0.8,
                )

        # Curriculum transition lines
        for step_t, level_name in transitions:
            ax.axvline(step_t, color=LEVEL_COLORS.get(level_name, "#fff"),
                       linewidth=1.0, linestyle="--", alpha=0.7, zorder=3)

    # Transition labels on top panel only
    ax_top = axes[0]
    for step_t, level_name in transitions:
        ax_top.text(
            step_t, ax_top.get_ylim()[1],
            f" →{level_name}", color=LEVEL_COLORS.get(level_name, "#fff"),
            fontsize=8, rotation=90, va="top", ha="left", alpha=0.9,
        )

    # CBS weight as secondary axis on entropy panel
    ax_cbs = axes[2].twinx()
    ax_cbs.set_facecolor(BG)
    ax_cbs.plot(steps, cbs_w, color="#f39c12", linewidth=1.2,
                linestyle=":", alpha=0.7, label="CBS weight")
    ax_cbs.set_ylabel("CBS weight", color="#f39c12", fontsize=9)
    ax_cbs.tick_params(colors="#f39c12", labelsize=8)
    ax_cbs.set_ylim(-0.05, 1.15)
    ax_cbs.spines["right"].set_edgecolor("#f39c12")

    axes[-1].set_xlabel("Training Steps", color=TEXT_COLOR, fontsize=10)
    axes[-1].tick_params(axis="x", colors=TEXT_COLOR)
    axes[-1].xaxis.offsetText.set_color(TEXT_COLOR)

    # Legend for phases
    legend_patches = [
        mpatches.Patch(color=PHASE_COLORS["A"], alpha=0.5, label="Phase A (CBS=1.0)"),
        mpatches.Patch(color=PHASE_COLORS["B"], alpha=0.5, label="Phase B (anneal)"),
        mpatches.Patch(color=PHASE_COLORS["C"], alpha=0.5, label="Phase C (pure RL)"),
    ]
    for lc, ln in LEVEL_COLORS.items():
        legend_patches.append(mpatches.Patch(color=ln, alpha=0.7, label=f"→ {lc}"))
    axes[0].legend(
        handles=legend_patches, loc="upper left",
        fontsize=8, facecolor=BG, labelcolor=TEXT_COLOR,
        edgecolor=GRID_COLOR, ncol=2,
    )

    fig.suptitle(
        "CBS-Bootstrapped MAPPO — Training Curves",
        color=TEXT_COLOR, fontsize=13, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out_path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log",    default=None,
                   help="Training log file (default: auto-detect from task output)")
    p.add_argument("--out",    default="results/learning_curves.png")
    p.add_argument("--smooth", type=int, default=10,
                   help="Rolling average window (1 = no smoothing)")
    p.add_argument("--total-steps", type=int, default=5_000_000)
    args = p.parse_args()

    log_path = args.log
    if log_path is None:
        # Try common locations
        candidates = [
            "training_log.txt",
            "train.log",
        ]
        for c in candidates:
            if os.path.exists(c):
                log_path = c
                break
        if log_path is None:
            print("No log file found. Pipe training output to a file:")
            print("  python train.py ... 2>&1 | tee training_log.txt")
            print("Then run:  python plot_curves.py --log training_log.txt")
            sys.exit(1)

    print(f"Parsing log: {log_path}")
    records, transitions = parse_log(log_path)

    if not records["step"]:
        print("No log lines found — check the log file format.")
        sys.exit(1)

    print(f"  Parsed {len(records['step'])} log entries")
    print(f"  Steps: {records['step'][0]:,} → {records['step'][-1]:,}")
    print(f"  Curriculum transitions: {transitions}")
    print(f"  Latest: goals={records['goals'][-1]:.1f}  "
          f"coll={records['collisions'][-1]:.1f}  "
          f"ent={records['entropy'][-1]:.3f}")

    plot_curves(
        records, transitions,
        smooth_window=args.smooth,
        out_path=args.out,
        total_steps=args.total_steps,
    )


if __name__ == "__main__":
    main()
