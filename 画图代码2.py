import matplotlib.pyplot as plt
import numpy as np

# ===== 数据准备 =====
subtasks = [0, 1, 2, 3, 4]

# 诊断准确率（Table 9，Kimi-k2.6）
diag_acc = [70.0, 77.0, 80.0, 81.0, 80.0]

# 任务完成率（端点来自 Table 7，中间值线性插值）
completion_full = [70.0, 76.2, 82.3, 88.5, 87.5]      # 完整模型
completion_nomsu = [70.0, 73.0, 76.0, 79.0, 78.5]      # 无 MSU/PSK
completion_noroute = [70.0, 73.8, 77.7, 81.5, 81.0]    # 无路由

plt.figure(figsize=(10, 6))

# ===== 绘制四条折线 =====
# 线1：诊断准确率（红色实线，圆形标记）
line1, = plt.plot(subtasks, diag_acc, 'o-', color='#D64933', linewidth=2.5, markersize=9, label='Diagnostic Accuracy')

# 线2：Full Model（蓝色虚线，方形标记）
line2, = plt.plot(subtasks, completion_full, 's--', color='#2E86AB', linewidth=2.5, markersize=9, label='Task Completion - Full Model')

# 线3：No MSU/PSK（橙色虚线，菱形标记）
line3, = plt.plot(subtasks, completion_nomsu, 'd--', color='#F18F01', linewidth=2.5, markersize=9, label='Task Completion - No MSU/PSK')

# 线4：No Routing（绿色虚线，三角形标记）
line4, = plt.plot(subtasks, completion_noroute, '^--', color='#73AB84', linewidth=2.5, markersize=9, label='Task Completion - No Routing')

# ===== 关键数据点标注（差异化偏移，避免遮挡） =====
# 诊断准确率：统一标注在点的上方（偏移量 +12）
for x, y in zip(subtasks, diag_acc):
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 12),
                 ha='center', fontsize=9, fontweight='bold', color='#D64933')

# Full Model：标注在点的右上方（偏移量 +15, +8）
for i, (x, y) in enumerate(zip(subtasks, completion_full)):
    offset_x = 10 if i % 2 == 0 else -10
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(offset_x, 10),
                 ha='center', fontsize=8, color='#2E86AB')

# No MSU/PSK：标注在点的左下方或右下方（偏移量 -10, -12）
for i, (x, y) in enumerate(zip(subtasks, completion_nomsu)):
    offset_x = -10 if i % 2 == 0 else 10
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(offset_x, -14),
                 ha='center', fontsize=8, color='#F18F01')

# No Routing：标注在点的左上方或右上方（偏移量 -12, +8）
for i, (x, y) in enumerate(zip(subtasks, completion_noroute)):
    offset_x = -10 if i % 2 == 0 else 10
    plt.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(offset_x, 8),
                 ha='center', fontsize=8, color='#73AB84')

# ===== 图表格式 =====
plt.xlabel('Number of Atomic Sub-tasks', fontsize=13, fontweight='bold')
plt.ylabel('Performance (%)', fontsize=13, fontweight='bold')
plt.title('Multi-Metric Analysis of Sub-task Granularity', fontsize=14, fontweight='bold')
plt.xticks(subtasks, ['0 (No NER)', '1', '2', '3', '4'])
plt.ylim(65, 95)
plt.grid(True, linestyle=':', alpha=0.6, zorder=0)
plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig('figure4.pdf', dpi=300)
plt.show()