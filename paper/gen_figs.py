"""Generate all supplementary figures for the paper."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import os

FIG_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GRAY   = "#6B7280"
LGRAY  = "#E5E7EB"
DGRAY  = "#374151"
LBLUE  = "#DBEAFE"
LGREEN = "#DCFCE7"
LRED   = "#FEE2E2"

# ─────────────────────────────────────────────────────────────────
# Figure 1: MAPPO Architecture with Transformer Communication
# ─────────────────────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, fc, ec=DGRAY, label="", fs=8, bold=False):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.2)
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x, y, label, ha="center", va="center", fontsize=fs,
                fontweight=weight, color=DGRAY)

    def arr(x0, y0, x1, y1, color=GRAY):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))

    # Agents (observations)
    for i, (xi, label) in enumerate([(1.0, "Agent 0\nobs 150D"),
                                      (2.5, "Agent 1\nobs 150D"),
                                      (4.0, "Agent i\nobs 150D")]):
        box(xi, 5.1, 1.3, 0.65, LBLUE, BLUE, label, fs=7)
        arr(xi, 4.77, xi, 4.17)

    ax.text(3.25, 5.1, "· · ·", ha="center", va="center", fontsize=12, color=GRAY)

    # MLP Encoders
    for xi in [1.0, 2.5, 4.0]:
        box(xi, 3.8, 1.3, 0.6, LGRAY, GRAY, "MLP Encoder\n128D", fs=7)
        arr(xi, 3.5, xi, 2.98)

    ax.text(3.25, 3.8, "· · ·", ha="center", va="center", fontsize=12, color=GRAY)

    # Transformer block
    box(2.5, 2.65, 4.6, 0.72, "#EDE9FE", PURPLE,
        "Multi-Head Self-Attention  (4 heads, 2 layers, padding mask for variable N)", fs=7.5, bold=True)

    # Context embeddings
    for i, xi in enumerate([1.0, 2.5, 4.0]):
        arr(xi, 2.29, xi, 1.73)
        box(xi, 1.5, 1.3, 0.55, LGRAY, GRAY, "emb_i 128D", fs=7)

    ax.text(3.25, 1.5, "· · ·", ha="center", va="center", fontsize=12, color=GRAY)

    # Split to Actor / Critic
    arr(1.0, 1.23, 1.0, 0.82)
    arr(4.0, 1.23, 4.0, 0.82)

    # Actor
    box(1.0, 0.55, 1.5, 0.52, LGREEN, GREEN, "Shared Actor\n5-action logits", fs=7)
    # Mean pool
    box(2.5, 0.55, 1.5, 0.52, "#FEF3C7", ORANGE, "Mean-Pool\nall embs → V(s)", fs=7)
    arr(2.5, 1.23, 2.5, 0.82)
    # Copy arrow from middle agents to critic
    arr(4.0, 1.23, 2.5, 0.82)

    ax.text(5.5, 2.65, "CTDE: critic sees\nglobal state during\ntraining only",
            ha="left", va="center", fontsize=7.5, color=GRAY, style="italic")

    ax.set_title("MAPPO Architecture with Transformer Communication",
                 fontsize=10, fontweight="bold", pad=8)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "architecture.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 2: CBS Curriculum — Annealing Schedule + Phase Bands
# ─────────────────────────────────────────────────────────────────
def fig_curriculum():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: delta annealing curve
    ax = axes[0]
    steps = np.linspace(0, 700_000, 2000)
    delta = np.where(steps < 100_000, 0.3,
            np.where(steps < 500_000, 0.3 * (1 - (steps - 100_000) / 400_000), 0.0))

    # Phase bands
    ax.axvspan(0,        100_000, color=LGREEN, alpha=0.45, label="Phase A")
    ax.axvspan(100_000,  500_000, color="#FEF3C7", alpha=0.55, label="Phase B")
    ax.axvspan(500_000,  700_000, color=LRED,  alpha=0.35, label="Phase C")

    ax.plot(steps / 1_000, delta, color=BLUE, lw=2.5)
    ax.axvline(100, color=GREEN,  ls="--", lw=1.2, alpha=0.8)
    ax.axvline(500, color=ORANGE, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Training step (×10³)", fontsize=9)
    ax.set_ylabel("CBS reward bonus δ", fontsize=9)
    ax.set_title("CBS Bonus Annealing Schedule", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    for x, lbl, col in [(50, "Phase A\nCBS oracle\nδ=0.3", GREEN),
                         (300, "Phase B\nLinear anneal\nδ→0", ORANGE),
                         (600, "Phase C\nPure RL\nδ=0", RED)]:
        ax.text(x, 0.17, lbl, ha="center", va="center", fontsize=7,
                color=col, fontweight="bold")

    ax.set_ylim(-0.02, 0.38)
    ax.set_xlim(0, 700)

    # Right: difficulty curriculum timeline
    ax2 = axes[1]
    levels = ["Easy\n7×7, 2ag", "Medium\n11×11, 4ag", "Hard\n15×15, 8ag", "Expert\n20×20, 12ag"]
    transitions = [0, 121.6, 136.4, 164.4]
    colors_lvl  = [GREEN, BLUE, ORANGE, RED]

    for i in range(len(levels)):
        x0 = transitions[i]
        x1 = transitions[i + 1] if i + 1 < len(transitions) else 7000
        ax2.barh(0, x1 - x0, left=x0, height=0.5,
                 color=colors_lvl[i], alpha=0.75, edgecolor="white")
        mid = (x0 + min(x1, 7000)) / 2
        ax2.text(mid, 0, levels[i], ha="center", va="center",
                 fontsize=7, fontweight="bold", color="white")

    ax2.axvline(100, color=GREEN,  ls="--", lw=1.2, alpha=0.8, label="Phase A end")
    ax2.axvline(500, color=ORANGE, ls="--", lw=1.2, alpha=0.8, label="Phase B end")

    ax2.set_xlabel("Training step (×10³)", fontsize=9)
    ax2.set_xlim(0, 7000)
    ax2.set_yticks([])
    ax2.set_title("Difficulty Curriculum Progression", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "curriculum.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 3: Dynamic Obstacle Scenarios — CBS vs MARL
# ─────────────────────────────────────────────────────────────────
def fig_dynamic():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    def draw_grid(ax, W, agents, goals, obstacles, planned_path=None,
                  actual_path=None, title=""):
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, W - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8, fontweight="bold", pad=4)

        # Grid lines
        for i in range(W + 1):
            ax.axhline(i - 0.5, color=LGRAY, lw=0.5)
            ax.axvline(i - 0.5, color=LGRAY, lw=0.5)

        # Planned path (dashed)
        if planned_path:
            px = [p[0] for p in planned_path]
            py = [p[1] for p in planned_path]
            ax.plot(px, py, ls="--", color=BLUE, lw=1.5, alpha=0.5, label="CBS plan")

        # Actual path
        if actual_path:
            px = [p[0] for p in actual_path]
            py = [p[1] for p in actual_path]
            ax.plot(px, py, color=GREEN, lw=2.0, alpha=0.8, label="MARL path")

        # Obstacles
        for (ox, oy) in obstacles:
            rect = plt.Rectangle((ox - 0.45, oy - 0.45), 0.9, 0.9,
                                  color=RED, alpha=0.7)
            ax.add_patch(rect)
            ax.text(ox, oy, "⬛", ha="center", va="center", fontsize=6)

        # Goals
        for (gx, gy) in goals:
            ax.scatter(gx, gy, marker="*", s=120, color=ORANGE, zorder=5)

        # Agents
        for (ax_, ay) in agents:
            circ = plt.Circle((ax_, ay), 0.3, color=BLUE, zorder=6)
            ax.add_patch(circ)

    W = 7
    # Scenario A: static (no obstacles)
    agents_a   = [(1, 1), (5, 1)]
    goals_a    = [(5, 5), (1, 5)]
    plan_a     = [(1,1),(2,1),(3,1),(4,1),(5,1),(5,2),(5,3),(5,4),(5,5)]
    draw_grid(axes[0], W, agents_a, goals_a, [],
              planned_path=plan_a,
              title="Static (no obstacles)\nCBS plan valid")

    # Scenario B: obstacle blocks CBS plan
    agents_b   = [(1, 1)]
    goals_b    = [(5, 5)]
    obs_b      = [(3, 3), (4, 3)]
    plan_b     = [(1,1),(2,1),(3,1),(3,2),(3,3),(3,4),(4,4),(5,4),(5,5)]
    draw_grid(axes[1], W, agents_b, goals_b, obs_b,
              planned_path=plan_b,
              title="Dynamic obstacles\nCBS plan invalidated (⬛ on path)")

    # Scenario C: MARL reactive avoidance
    agents_c   = [(1, 1)]
    goals_c    = [(5, 5)]
    obs_c      = [(3, 3), (4, 3)]
    actual_c   = [(1,1),(2,1),(2,2),(2,3),(2,4),(3,4),(4,4),(5,4),(5,5)]
    draw_grid(axes[2], W, agents_c, goals_c, obs_c,
              actual_path=actual_c,
              title="MARL reactive\nAgent detours around obstacles")

    # Legend
    handles = [
        mpatches.Patch(color=BLUE,   alpha=0.5, label="CBS plan (dashed)"),
        mpatches.Patch(color=GREEN,  alpha=0.8, label="MARL path"),
        mpatches.Patch(color=RED,    alpha=0.7, label="Dynamic obstacle"),
        mpatches.Patch(color=ORANGE, alpha=1.0, label="Goal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Dynamic Obstacle Scenarios: CBS Plan Invalidation vs. MARL Reactive Avoidance",
                 fontsize=9, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "dynamic_viz.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 4: Language Control Pipeline
# ─────────────────────────────────────────────────────────────────
def fig_lang_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        (0.9,  "#FEF3C7", ORANGE, "Operator\nCommand",
         '"Send agents 0–5\nto loading bay,\nrest to charging"'),
        (3.0,  "#EDE9FE", PURPLE, "Qwen2.5-3B\n(Ollama, local)",
         'JSON:\n{"assignments":\n[{agent:0,zone:\n"loading_bay"}, ...]}'),
        (5.2,  LBLUE,     BLUE,   "resolve_goals()\nzone → (x,y)",
         "Agent 0 → (18,18)\nAgent 1 → (17,18)\n..."),
        (7.4,  LGREEN,    GREEN,  "MAPPO Policy\n(pretrained)",
         "Collision-free\nnavigation to\nassigned zones"),
        (9.5,  LRED,      RED,    "Result",
         "Goals reached\n≥80% success\nno retraining"),
    ]

    box_w, box_h = 1.55, 2.2
    for x, fc, ec, title, detail in stages:
        rect = FancyBboxPatch((x - box_w/2, 0.4), box_w, box_h,
                              boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=1.5)
        ax.add_patch(rect)
        ax.text(x, 0.4 + box_h - 0.22, title, ha="center", va="top",
                fontsize=7.5, fontweight="bold", color=DGRAY)
        ax.text(x, 0.4 + box_h/2 - 0.2, detail, ha="center", va="center",
                fontsize=6.2, color=DGRAY, family="monospace")

    for i in range(len(stages) - 1):
        x0 = stages[i][0]   + box_w/2
        x1 = stages[i+1][0] - box_w/2
        ax.annotate("", xy=(x1, 1.5), xytext=(x0, 1.5),
                    arrowprops=dict(arrowstyle="-|>", color=DGRAY, lw=1.4))

    ax.text(5.2, 0.18, "No retraining required · Local inference · Rule-based fallback if LLM unavailable",
            ha="center", va="bottom", fontsize=7, color=GRAY, style="italic")

    ax.set_title("Language-Conditioned Zone Assignment Pipeline",
                 fontsize=10, fontweight="bold", pad=6)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "lang_pipeline.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 5: Expert Zone Layout (20×20 warehouse map)
# ─────────────────────────────────────────────────────────────────
def fig_zone_layout():
    fig, ax = plt.subplots(figsize=(5, 5))
    W = 20
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, W - 0.5)
    ax.set_aspect("equal")

    # Background
    ax.set_facecolor("#F9FAFB")
    for i in range(W + 1):
        ax.axhline(i - 0.5, color=LGRAY, lw=0.4, zorder=1)
        ax.axvline(i - 0.5, color=LGRAY, lw=0.4, zorder=1)

    # Zone definitions (from zones.py expert level)
    zones = {
        "loading_bay":  ([(18,18),(17,18),(18,17),(17,17),(16,18),(18,16),(16,17)], "#DBEAFE", BLUE),
        "storage_a":    ([(2,2),(2,3),(3,2),(3,3),(1,2),(2,1),(4,2)],              "#DCFCE7", GREEN),
        "storage_b":    ([(18,2),(17,2),(18,3),(17,3),(16,2),(18,1),(16,3)],        "#D1FAE5", "#15803D"),
        "storage_c":    ([(2,18),(2,17),(3,18),(3,17),(1,18),(2,16),(4,18)],         "#BBF7D0", "#166534"),
        "charging":     ([(10,18),(10,17),(11,18),(9,18),(10,16),(11,17),(9,17)],    "#FEF9C3", "#92400E"),
        "inspection":   ([(10,10),(10,11),(11,10),(9,10),(10,9),(11,11),(9,9)],      "#EDE9FE", PURPLE),
        "dispatch":     ([(18,10),(17,10),(18,11),(18,9),(17,11),(16,10),(17,9)],    "#FEE2E2", RED),
        "exit":         ([(10,2),(10,3),(11,2),(9,2),(10,1),(11,3),(9,3)],           "#FFEDD5", ORANGE),
        "staging":      ([(2,10),(2,11),(3,10),(1,10),(2,9),(3,11),(1,11)],          "#F0FDF4", "#166534"),
    }

    handles = []
    for name, (cells, fc, ec) in zones.items():
        patch = mpatches.Patch(fc=fc, ec=ec, label=name)
        handles.append(patch)
        for (cx, cy) in cells:
            rect = plt.Rectangle((cx - 0.45, cy - 0.45), 0.9, 0.9,
                                  color=fc, ec=ec, lw=1.0, zorder=2)
            ax.add_patch(rect)
        # Label at centroid
        cx_mean = sum(c[0] for c in cells) / len(cells)
        cy_mean = sum(c[1] for c in cells) / len(cells)
        ax.text(cx_mean, cy_mean, name.replace("_", "\n"), ha="center",
                va="center", fontsize=5.5, fontweight="bold", color=DGRAY, zorder=3)

    ax.set_xticks(range(0, W, 5))
    ax.set_yticks(range(0, W, 5))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.set_title("Expert-Level Warehouse Zone Layout (20×20, 9 zones)", fontsize=9, fontweight="bold")
    ax.legend(handles=handles, fontsize=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "zone_layout.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 6: CBS Solve Time vs Agents (exponential scaling)
# ─────────────────────────────────────────────────────────────────
def fig_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # Left: solve time bar
    ax = axes[0]
    levels = ["Easy\n(2ag)", "Medium\n(4ag)", "Hard\n(8ag)", "Expert\n(12ag)"]
    cbs_ms  = [758, 807, 3475, 7855]
    marl_ms = [1.2, 1.2, 1.2, 1.2]  # constant

    x = np.arange(len(levels))
    w = 0.35
    bars1 = ax.bar(x - w/2, cbs_ms,  w, label="CBS",       color=RED,  alpha=0.8)
    bars2 = ax.bar(x + w/2, marl_ms, w, label="MARL (ours)", color=BLUE, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=8)
    ax.set_ylabel("Solve time (ms)", fontsize=8)
    ax.set_yscale("log")
    ax.set_title("CBS vs MARL Inference Time", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars1, cbs_ms):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.15,
                f"{v:,}ms", ha="center", fontsize=6.5)
    ax.text(x[-1] + w/2, marl_ms[0] * 2.5, "~1ms\n(const)", ha="center",
            fontsize=6.5, color=BLUE)

    # Right: success rate comparison
    ax2 = axes[1]
    marl_sr = [0.710, 0.953, 0.917, 0.806]
    cbs_sr  = [0.605, 0.570, 0.570, 0.000]

    bars3 = ax2.bar(x - w/2, marl_sr, w, label="MARL (ours)", color=BLUE, alpha=0.8)
    bars4 = ax2.bar(x + w/2, cbs_sr,  w, label="CBS",         color=RED,  alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels, fontsize=8)
    ax2.set_ylabel("Per-agent success rate", fontsize=8)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Static Evaluation: Success Rate", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars3, marl_sr):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f"{v:.1%}", ha="center", fontsize=6.5, color=BLUE)
    for bar, v in zip(bars4, cbs_sr):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f"{v:.1%}", ha="center", fontsize=6.5, color=RED)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "scaling.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 7: Dynamic evaluation degradation heatmap
# ─────────────────────────────────────────────────────────────────
def fig_dynamic_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))

    methods = ["MARL\n(ours)", "CBS\nstatic", "CBS+\ndynamic"]
    levels  = ["Expert\n20×20", "Hard\n15×15", "Medium\n11×11"]
    # success rates [level][method]
    data = np.array([
        [0.803, 0.000, 0.000],  # expert
        [0.891, 0.490, 0.070],  # hard
        [0.915, 0.570, 0.040],  # medium
    ])

    cmap = LinearSegmentedColormap.from_list("rg", [RED, "#FBBF24", GREEN])
    im = axes[0].imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, fontsize=8)
    axes[0].set_yticks(range(len(levels)))
    axes[0].set_yticklabels(levels, fontsize=8)
    axes[0].set_title("Per-Agent Success Rate\n(dynamic obstacles, K=4)", fontsize=8.5, fontweight="bold")
    for i in range(len(levels)):
        for j in range(len(methods)):
            axes[0].text(j, i, f"{data[i,j]:.1%}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold",
                        color="white" if data[i,j] < 0.4 else DGRAY)
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    # Invalidation rates
    inval = np.array([[100.0], [52.6], [51.7]])
    cmap2 = LinearSegmentedColormap.from_list("invalidation", [GREEN, "#FBBF24", RED])
    im2 = axes[1].imshow(inval, cmap=cmap2, vmin=0, vmax=100, aspect="auto")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["CBS Plan\nInvalidation %"], fontsize=8)
    axes[1].set_yticks(range(len(levels)))
    axes[1].set_yticklabels(levels, fontsize=8)
    axes[1].set_title("CBS Plan Invalidation Rate\n(K=4 dynamic obstacles)", fontsize=8.5, fontweight="bold")
    for i, v in enumerate(inval):
        axes[1].text(0, i, f"{v[0]:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if v[0] > 60 else DGRAY)
    fig.colorbar(im2, ax=axes[1], shrink=0.85)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "dynamic_heatmap.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("Generating figures...")
    fig_architecture()
    fig_curriculum()
    fig_dynamic()
    fig_lang_pipeline()
    fig_zone_layout()
    fig_scaling()
    fig_dynamic_heatmap()
    print("Done.")
