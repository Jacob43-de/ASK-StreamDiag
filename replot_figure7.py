import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "serif"

x_labels = ["DS-1000\n(Code)", "DDXPlus\n(Medical)", "ToolBench\n(Tool)"]
x = [0, 1, 2]

ds1000   = [81.33, 74.32, 76.18]
ddxplus  = [73.91, 80.87, 75.04]
toolbench= [75.26, 74.89, 81.02]

fig, ax = plt.subplots(figsize=(10, 7.5), dpi=200)

ax.plot(x, ds1000,   color="#3b5b92", marker="o",  markersize=11, linewidth=2, label="DS-1000 (Code)")
ax.plot(x, ddxplus,  color="#c85542", marker="s",  markersize=11, linewidth=2, label="DDXPlus (Medical)")
ax.plot(x, toolbench,color="#4a9954", marker="^",  markersize=12, linewidth=2, label="ToolBench (Tool)")

for xi, v in zip(x, ds1000):
    off = -0.9 if xi == 1 else 0.45
    va = "top" if xi == 1 else "bottom"
    ax.text(xi, v + off, f"{v:.2f}", ha="center", va=va, fontsize=12, color="#222")
for xi, v in zip(x, ddxplus):
    ax.text(xi, v + 0.45, f"{v:.2f}", ha="center", va="bottom", fontsize=12, color="#222")
for xi, v in zip(x, toolbench):
    off = 0.55 if xi == 1 else 0.45
    ax.text(xi, v + off, f"{v:.2f}", ha="center", va="bottom", fontsize=12, color="#222")

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=13)
ax.set_ylim(73, 83)
ax.set_yticks([74, 76, 78, 80, 82])
ax.set_yticklabels(["74", "76", "78", "80", "82"], fontsize=13)
ax.set_ylabel("Diagnostic Accuracy (%)", fontsize=15, labelpad=15)
ax.set_xlabel("Test Dataset", fontsize=15, labelpad=15)
ax.set_title("Cross-domain Generalization Performance", fontsize=17, fontweight="bold", pad=20)
ax.grid(True, linestyle="--", alpha=0.4, axis="y")
ax.legend(loc="upper left", fontsize=12, frameon=False, ncol=1, bbox_to_anchor=(0.18, 1.0))

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure7_line.pdf")
fig.tight_layout()
fig.savefig(out, format="pdf")
print("saved:", out)
