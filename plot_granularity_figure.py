import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = [0, 3]
full = [70.0, 87.5]
nomsu = [70.0, 79.0]
norout = [70.0, 81.5]

fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=200)

ax.plot([3], [81.0], marker="o", linestyle="", markersize=6,
        color="#1f77b4", label="Diagnostic accuracy (not defined at K=0)")
ax.annotate("81.0", xy=(3, 81.0), xytext=(6, -10),
            textcoords="offset points", fontsize=8, color="#1f77b4")

ax.plot(K, full, marker="s", markersize=6, linewidth=1.8,
        color="#d62728", label="Task completion (Full Model)")
ax.plot(K, nomsu, marker="^", markersize=6, linewidth=1.8,
        color="#2ca02c", label="Task completion (No MSU/PSK)")
ax.plot(K, norout, marker="D", markersize=6, linewidth=1.8,
        color="#ff7f0e", label="Task completion (No Routing)")

ax.set_xticks([0, 3])
ax.set_xticklabels(["$K=0$\n(No Decomp.)", "$K=3$"])
ax.set_ylim(60, 95)
ax.set_ylabel("Performance (%)")
ax.set_xlabel("Number of subtasks $K$ (measured configurations)")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=7.5, framealpha=0.9)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure_3.pdf")
fig.savefig(out)
print("saved:", out)
