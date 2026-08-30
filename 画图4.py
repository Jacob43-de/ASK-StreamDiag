import matplotlib.pyplot as plt
import numpy as np

# ===== 数据准备（来自Table 11） =====
x_labels = ['0.1', '0.2', '0.3', '0.7', '0.8', '0.9']
x_pos = np.arange(len(x_labels))

# 子图(a): 阈值扫面实验
tau_low_sweep = [76.2, 80.9, 77.1, 0, 0, 0]          # τ_low sweep (τ_high=0.8)
tau_high_sweep = [0, 0, 0, 78.5, 80.9, 77.8]         # τ_high sweep (τ_low=0.2)

# 子图(b): 四个模型的跨任务ACC
qwen_acc = [52.3, 52.3, 52.3, 67.9, 67.9, 69.4]
kimi_acc = [58.5, 58.5, 58.5, 73.9, 73.9, 62.5]
zhipu_acc = [50.1, 50.1, 50.1, 65.7, 65.7, 57.8]
doubao_acc = [51.5, 51.5, 51.5, 66.3, 66.3, 63.2]

# ===== 颜色方案 =====
color_tau_low = '#2E86AB'      # 深蓝
color_tau_high = '#E67E22'     # 橙色
color_qwen = '#8E44AD'         # 紫色
color_kimi = '#E74C3C'         # 红色
color_zhipu = '#27AE60'        # 翠绿
color_doubao = '#D4A017'       # 金色

# ===== 创建 1行2列 子图 =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ============================================================
# 子图 (a): 阈值扫面实验
# ============================================================
bar_width_a = 0.30
x_pos_a = np.arange(len(x_labels))

bars1 = ax1.bar(x_pos_a - bar_width_a/2, tau_low_sweep, bar_width_a,
                color=color_tau_low, edgecolor='black', linewidth=0.8,
                label='τ_low sweep (τ_high=0.8)', zorder=3)
bars2 = ax1.bar(x_pos_a + bar_width_a/2, tau_high_sweep, bar_width_a,
                color=color_tau_high, edgecolor='black', linewidth=0.8,
                label='τ_high sweep (τ_low=0.2)', zorder=3)

# 数值标注（仅标注有值的柱子）
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax1.annotate(f'{h}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold',
                     color='#1A5276')
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax1.annotate(f'{h}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold',
                     color='#935E38')

# 标注最优阈值组合
ax1.scatter([1], [80.9], color='#FFD700', s=400, marker='*',
            edgecolor='black', linewidth=1.5, zorder=10,
            label='★ Optimal: τ_low=0.2, τ_high=0.8')

# 子图(a)格式
ax1.set_xlabel('Threshold Value', fontsize=11, fontweight='bold')
ax1.set_ylabel('Classification Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Threshold Sensitivity', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos_a)
ax1.set_xticklabels(x_labels)
ax1.set_ylim(45, 88)
ax1.axvline(x=2.5, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)
ax1.grid(True, axis='y', linestyle=':', alpha=0.3, zorder=0)
ax1.legend(loc='upper center', fontsize=8, framealpha=0.92, ncol=2)

# 添加区域文字
ax1.text(0.5, 84, 'τ_low Sweep', ha='center', fontsize=9,
         color=color_tau_low, fontweight='bold')
ax1.text(4.0, 84, 'τ_high Sweep', ha='center', fontsize=9,
         color=color_tau_high, fontweight='bold')

# ============================================================
# 子图 (b): 四个模型跨任务ACC
# ============================================================
bar_width_b = 0.18
offset = np.arange(-1.5, 2.0, 1.0) * bar_width_b

bars3 = ax2.bar(x_pos + offset[0], qwen_acc, bar_width_b,
                color=color_qwen, edgecolor='black', linewidth=0.8,
                label='Qwen-turbo')
bars4 = ax2.bar(x_pos + offset[1], kimi_acc, bar_width_b,
                color=color_kimi, edgecolor='black', linewidth=0.8,
                label='Kimi-k2.6')
bars5 = ax2.bar(x_pos + offset[2], zhipu_acc, bar_width_b,
                color=color_zhipu, edgecolor='black', linewidth=0.8,
                label='Zhipu GLM-4')
bars6 = ax2.bar(x_pos + offset[3], doubao_acc, bar_width_b,
                color=color_doubao, edgecolor='black', linewidth=0.8,
                label='DouBao')

# 标注数值（仅标注关键值，避免过密）
for bar in bars4:  # 只标注Kimi（红色，最显著）
    h = bar.get_height()
    if h == 73.9:
        bar.set_edgecolor('#FFD700')
        bar.set_linewidth(3)
        ax2.annotate(f'{h}% ★', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 6), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold',
                     color='#C0392B')
    elif h > 55:
        ax2.annotate(f'{h}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=7, color='#333')

# 子图(b)格式
ax2.set_xlabel('Threshold Value (Task Domain)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Task Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Cross-Task Model Reference', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['0.1\n(DS-1000)', '0.2\n(DS-1000)', '0.3\n(DS-1000)',
                     '0.7\n(DDXPlus)', '0.8\n(DDXPlus)', '0.9\n(ToolBench)'], fontsize=8)
ax2.set_ylim(45, 88)
ax2.axvline(x=2.5, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)
ax2.grid(True, axis='y', linestyle=':', alpha=0.3, zorder=0)
ax2.legend(loc='upper center', fontsize=8, framealpha=0.92, ncol=4)

# ===== 整体调整 =====
plt.tight_layout()
plt.savefig('figure6.pdf', dpi=300, bbox_inches='tight')
plt.show()