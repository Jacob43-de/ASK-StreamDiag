import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

modes = [
    "Binary\nFeedback",
    "Overall\nJudge",
    "Ours w/o\nMSU/PSK",
    "Ours\nFull",
    "Ours Full\n+ Routing",
]
costs = [1.0, 1.3, 1.6, 1.6, 2.4]          
depths = [0, 1, 2, 3, 4]                    
repair = [0, 0, 80, 82, 88]                 

x = range(len(modes))

fig, axL = plt.subplots(figsize=(8.6, 5.6), dpi=200)

bars = axL.bar(x, costs, width=0.55, color="#5b9bd5", label="Normalized Cost", zorder=2)
axL.set_ylabel("Normalized Cost", color="#41719c", fontsize=10)
axL.set_ylim(0, 3.0)
axL.set_xlabel("Evaluation Mode", fontsize=10)
axL.set_xticks(list(x))
axL.set_xticklabels(modes, fontsize=9)
axL.grid(True, linestyle="-", alpha=0.45, zorder=0)

for xi, c, d in zip(x, costs, depths):
    axL.text(xi, c + 0.06, f"Depth: {d}", ha="center", va="bottom", fontsize=8.5, color="#333")

axR = axL.twinx()
axR.plot(list(x), repair, color="#d9534f", marker="o", linewidth=1.8,
         markersize=6, label="Repair Success Rate", zorder=3)
axR.set_ylabel("Repair Success Rate (%)", color="#d9534f", fontsize=10)
axR.set_ylim(0, 100)

h1, l1 = axL.get_legend_handles_labels()
h2, l2 = axR.get_legend_handles_labels()
axL.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5, framealpha=0.9)

axL.set_title("Cost–Diagnostic Depth–Repair Effect Trade-off", fontsize=10.5, fontweight="bold")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fig_cost_depth_tradeoff_barline.pdf")
fig.tight_layout()
fig.savefig(out)
print("saved:", out)
