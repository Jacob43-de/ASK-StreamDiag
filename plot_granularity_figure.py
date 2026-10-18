import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

X = [0, 1, 2, 3, 4]
MEASURED = {0, 3}

SERIES = [
    ("Diagnostic Accuracy",        "#d65043", "o", [70.0, 77.0, 80.0, 81.0, 80.0]),
    ("Task Completion - Full Model", "#3d85ad", "s", [70.0, 76.2, 82.3, 88.5, 87.5]),
    ("Task Completion - No MSU/PSK", "#f29400", "D", [70.0, 73.0, 76.0, 79.0, 78.5]),
    ("Task Completion - No Routing", "#83a987", "^", [70.0, 73.8, 77.7, 81.5, 81.0]),
]

fig, ax = plt.subplots(figsize=(12.5, 7.6), dpi=200)

for name, color, marker, ys in SERIES:
    # entire line is dashed because every segment touches interpolated points
    ax.plot(X, ys, linestyle="--", linewidth=2.6, color=color,
            marker="", zorder=2)
    for x, y in zip(X, ys):
        if x in MEASURED:
            ax.plot(x, y, marker=marker, markersize=12, color=color,
                    markerfacecolor=color, markeredgecolor=color,
                    markeredgewidth=2.0, zorder=4)
        else:
            ax.plot(x, y, marker=marker, markersize=12, color=color,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=2.2, zorder=4)
    # value labels (small offsets to reduce overlap)
    for x, y in zip(X, ys):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=11, color=color)

ax.set_title("Multi-Metric Analysis of Sub-task Granularity",
             fontsize=20, fontweight="bold", pad=12)
ax.set_xlabel("Number of Atomic Sub-tasks", fontsize=17, fontweight="bold")
ax.set_ylabel("Performance (%)", fontsize=17, fontweight="bold")
ax.set_xticks(X)
ax.set_xticklabels(["0 (No Decomp.)", "1", "2", "3", "4"], fontsize=14)
ax.set_ylim(65, 95)
ax.set_yticks(range(65, 96, 5))
ax.tick_params(axis="y", labelsize=14)
ax.grid(True, linestyle=":", linewidth=1.0, alpha=0.6)
ax.axvspan(-0.25, 4.25, color="none")

# vertical shaded band over the measured K=3 point column to reinforce
ax.axvline(3, color="grey", linestyle=":", linewidth=1.0, alpha=0.5)

series_handles = [
    Line2D([], [], color=c, marker=m, linestyle="--", linewidth=2.6,
           markersize=11, label=n)
    for n, c, m, _ in SERIES
]
measured_handle = Line2D(
    [], [], color="black", marker="o", linestyle="None", markersize=12,
    markerfacecolor="black", markeredgecolor="black",
    label="Measured point (K=0, 3)")
interp_handle = Line2D(
    [], [], color="black", marker="o", linestyle="--", linewidth=2.0,
    markersize=12, markerfacecolor="white", markeredgecolor="black",
    markeredgewidth=2.0, label="Interpolated point (K=1, 2, 4)")

ax.legend(handles=series_handles + [measured_handle, interp_handle],
          loc="lower right", fontsize=12.5, framealpha=0.95)

plt.tight_layout()
plt.savefig("Figure_3.pdf", format="pdf", bbox_inches="tight")
plt.savefig("Figure_3_preview.png", format="png", dpi=150, bbox_inches="tight")
print("saved Figure_3.pdf")
