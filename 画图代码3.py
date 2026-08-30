import matplotlib.pyplot as plt
import numpy as np
tasks = ['DS-1000', 'DDXPlus', 'ToolBench']
x_pos = np.arange(len(tasks))

# 四个模型的 EM 数据 (%)
qwen_em = [80.0, 50.0, 75.0]
kimi_em = [88.0, 60.0, 82.0]
zhipu_em = [82.0, 55.0, 78.0]
doubao_em = [78.0, 48.0, 72.0]

plt.figure(figsize=(9, 6))

# 绘制四条折线（不同颜色和标记）
plt.plot(x_pos, qwen_em, 'o-', color='#A23B72', linewidth=2.5, markersize=9, label='Qwen-turbo')
plt.plot(x_pos, kimi_em, 's-', color='#2E86AB', linewidth=2.5, markersize=9, label='Kimi-k2.6')
plt.plot(x_pos, zhipu_em, '^-', color='#F18F01', linewidth=2.5, markersize=9, label='Zhipu GLM-4')
plt.plot(x_pos, doubao_em, 'd-', color='#73AB84', linewidth=2.5, markersize=9, label='DouBao')

# 标注具体数值
for i, (x, y) in enumerate(zip(x_pos, qwen_em)):
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#A23B72')
for i, (x, y) in enumerate(zip(x_pos, kimi_em)):
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#2E86AB')
for i, (x, y) in enumerate(zip(x_pos, zhipu_em)):
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#F18F01')
for i, (x, y) in enumerate(zip(x_pos, doubao_em)):
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#73AB84')

plt.xlabel('Benchmark Tasks', fontsize=13, fontweight='bold')
plt.ylabel('Exact Match (EM) (%)', fontsize=13, fontweight='bold')
plt.title('Multi-Model Cross-Task Exact Match Comparison',
          fontsize=14, fontweight='bold')
plt.xticks(x_pos, tasks)
plt.ylim(40, 95)
plt.grid(True, linestyle=':', alpha=0.6, zorder=0)
plt.legend(loc='best', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig('figure5', dpi=300)
plt.show()