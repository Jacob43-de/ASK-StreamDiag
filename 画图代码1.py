import matplotlib.pyplot as plt
import numpy as np

# ===== 新 Figure 3：多模型跨任务 ACC 折线对比 =====
# 数据来源于 Table 5 Group 3 (Self-StreamICL)
tasks = ['DS-1000', 'DDXPlus', 'ToolBench']
x_pos = np.arange(len(tasks))

# 四个模型的 ACC 数据 (%)
qwen_acc = [52.3, 67.9, 69.4]
kimi_acc = [58.5, 73.9, 62.5]
zhipu_acc = [50.1, 65.7, 57.8]
doubao_acc = [51.5, 66.3, 63.2]

plt.figure(figsize=(9, 6))

# 绘制四条折线（不同颜色和标记）
plt.plot(x_pos, qwen_acc, 'o-', color='#A23B72', linewidth=2.5, markersize=9, label='Qwen-turbo')
plt.plot(x_pos, kimi_acc, 's-', color='#2E86AB', linewidth=2.5, markersize=9, label='Kimi-k2.6')
plt.plot(x_pos, zhipu_acc, '^-', color='#F18F01', linewidth=2.5, markersize=9, label='Zhipu GLM-4')
plt.plot(x_pos, doubao_acc, 'd-', color='#73AB84', linewidth=2.5, markersize=9, label='DouBao')

# 标注具体数值
for i, (x, y) in enumerate(zip(x_pos, qwen_acc)):
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#A23B72')
for i, (x, y) in enumerate(zip(x_pos, kimi_acc)):
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#2E86AB')
for i, (x, y) in enumerate(zip(x_pos, zhipu_acc)):
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#F18F01')
for i, (x, y) in enumerate(zip(x_pos, doubao_acc)):
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#73AB84')

plt.xlabel('Benchmark Tasks', fontsize=13, fontweight='bold')
plt.ylabel('Accuracy (ACC) (%)', fontsize=13, fontweight='bold')
plt.title('Multi-Model Cross-Task Accuracy Comparison',
          fontsize=14, fontweight='bold')
plt.xticks(x_pos, tasks)
plt.ylim(45, 80)
plt.grid(True, linestyle=':', alpha=0.6, zorder=0)
plt.legend(loc='best', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig('figure3.pdf', dpi=300)
plt.show()