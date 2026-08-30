import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

modes = [
    "Binary\nFeedback",
    "Overall\nJudge",
    "Ours w/o\nMSU/PSK",
    "Ours\nFull",
    "Ours Full\n+ Routing"
]
norm_cost = [1.0, 1.3, 1.6, 1.8, 2.4]       # 归一化成本
diagnostic_depth = [0, 1, 2, 3, 4]          # 诊断深度得分
repair_success = [0, 0, 79.0, 81.5, 88.5]   # 修复成功率

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8

fig, ax1 = plt.subplots(figsize=(9, 6), dpi=300)
x = np.arange(len(modes))
bar_color = "#5fa8d3"
line_color = "#d65f5f"

# 左轴：归一化成本（柱状图）
bars = ax1.bar(x, norm_cost, width=0.55, color=bar_color, edgecolor="white", linewidth=0.8, label="Normalized Cost")
ax1.set_xlabel("Evaluation Mode")
ax1.set_ylabel("Normalized Cost", color=bar_color)
ax1.tick_params(axis='y', labelcolor=bar_color)
ax1.set_xticks(x)
ax1.set_xticklabels(modes)
ax1.set_ylim(0, 3.0)

# 在柱子上方标注诊断深度得分
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.05,
        f"Depth: {diagnostic_depth[i]}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="medium"
    )

# 右轴：修复成功率（折线图）
ax2 = ax1.twinx()
line = ax2.plot(
    x, repair_success,
    color=line_color,
    marker="o",
    markersize=7,
    linewidth=2,
    label="Repair Success Rate"
)
ax2.set_ylabel("Repair Success Rate (%)", color=line_color)
ax2.tick_params(axis='y', labelcolor=line_color)
ax2.set_ylim(0, 100)

# 合并双轴图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

ax1.set_title("Cost–Diagnostic Depth–Repair Effect Trade-off", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("fig_cost_depth_tradeoff_barline.pdf", bbox_inches="tight")
plt.savefig("fig_cost_depth_tradeoff_barline.png", bbox_inches="tight", dpi=300)
plt.show()