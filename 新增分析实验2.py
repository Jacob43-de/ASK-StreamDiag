import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 数据
strategies = ["All-Prompt", "All-Route", "Random", "Ours"]
categories = ["MSU", "PSK", "Overall"]
repair_rates = [
    [17.2, 78.3, 47.8],   # All-Prompt
    [84.7, 62.1, 73.4],   # All-Route
    [51.0, 70.2, 60.6],   # Random
    [84.7, 78.3, 81.5]    # Ours
]

# ===================== SCI全局标准配置 =====================
sns.set_style("white")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9.5
plt.rcParams["ytick.labelsize"] = 9.5
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["pdf.fonttype"] = 42  # PDF嵌入字体，SCI必备
plt.rcParams["ps.fonttype"] = 42

# 画布创建
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
x = np.arange(len(strategies))
width = 0.25
# 学术低饱和配色，黑白打印可区分
colors = ["#405E99", "#D0604C", "#50A060"]

# 绘制分组柱状
for i, cat in enumerate(categories):
    values = [repair_rates[s][i] for s in range(len(strategies))]
    bars = ax.bar(x + (i - 1) * width, values, width=width,
                  color=colors[i], label=cat, edgecolor="white", linewidth=0.6)
    # 柱顶标注数值（SCI推荐，直观展示数据）
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f"{height:.1f}", ha="center", va="bottom", fontsize=8)

# 坐标轴与样式优化
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.set_ylabel("Repair Success Rate (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Repair Success Rate by Strategy and Failure Type", fontweight="bold")
# SCI经典去右上边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# 浅色横向网格辅助读数
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
# 输出矢量PDF+高清PNG
plt.savefig("fig_repair_success_bar.pdf", bbox_inches="tight")
plt.savefig("fig_repair_success_bar.png", bbox_inches="tight")
plt.show()