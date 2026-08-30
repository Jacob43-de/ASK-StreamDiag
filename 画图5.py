import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# ===================== SCI全局绘图规范 =====================
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False
# 嵌入字体，避免期刊PDF乱码
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
# 字号分级（期刊标准）
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
# 1. 原始数据
data = np.array([
    [81.25, 74.32, 76.18],   # Train: DS-1000 (Code)
    [73.91, 80.87, 75.04],   # Train: DDXPlus (Medical)
    [75.26, 74.89, 81.02]    # Train: ToolBench (Tool)
])
train_names = ['DS-1000 (Code)', 'DDXPlus (Medical)', 'ToolBench (Tool)']
test_x_labels = ['DS-1000\n(Code)', 'DDXPlus\n(Medical)', 'ToolBench\n(Tool)']
x = np.arange(len(test_x_labels))
# 2. 画布创建
fig, ax = plt.subplots(figsize=(8, 6))
# SCI学术配色+标记（黑白打印可区分）
colors = ["#405E99", "#D0604C", "#50A060"]
markers = ["o", "s", "^"]
# 3. 逐条绘制折线并标注数值
for i, train_data in enumerate(data):
    ax.plot(x, train_data, color=colors[i], marker=markers[i], linewidth=1.3, markersize=7, label=train_names[i])
    # 每个数据点标注准确率
    for idx, val in enumerate(train_data):
        ax.text(idx, val + 0.25, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
# 4. 坐标轴设置
ax.set_xticks(x)
ax.set_xticklabels(test_x_labels)
ax.set_xlabel("Test Dataset", labelpad=12)
ax.set_ylabel("Diagnostic Accuracy (%)", labelpad=12)
ax.set_ylim(73, 83)  # 和原图热力图值域保持一致
# 5. 图例与标题 ========== 仅微调图例位置，向下偏移 ==========
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.4, 0.96))
ax.set_title("Cross-domain Generalization Performance", pad=15, fontweight="bold")
# 6. SCI极简边框（隐藏右上）
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# 浅色横向网格辅助读数
ax.grid(axis="y", linestyle="--", alpha=0.3)
# 7. 保存图片
plt.tight_layout()
plt.savefig("Figure7_line.pdf", bbox_inches="tight")
plt.savefig("Figure7_line.png", bbox_inches="tight")
plt.show()