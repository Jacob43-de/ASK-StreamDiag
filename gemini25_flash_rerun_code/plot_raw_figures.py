
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

tasks = ["DS-1000", "DDXPlus", "ToolBench"]
x = np.arange(len(tasks))

STYLES = [
    ("Qwen",        "#A23B72", "o"),
    ("Kimi-k2.6",   "#2E86AB", "s"),
    ("Zhipu GLM-4", "#F18F01", "^"),
    ("DouBao",      "#73AB84", "d"),
]

ACC = {
    "Qwen":        [59.6, 64.2, 67.8],
    "Kimi-k2.6":   [74.8, 70.4, 72.4],
    "Zhipu GLM-4": [69.9, 68.8, 69.6],
    "DouBao":      [58.4, 61.4, 63.8],
}
EM = {
    "Qwen":        [42.8, 41.2, 41.2],
    "Kimi-k2.6":   [51.2, 47.6, 45.8],
    "Zhipu GLM-4": [47.6, 44.4, 43.0],
    "DouBao":      [42.0, 39.6, 38.0],
}


def make(data, title, ylabel, ylim, out, fmt):
    plt.figure(figsize=(9, 6))
    for name, color, marker in STYLES:
        ys = data[name]
        plt.plot(x, ys, marker + "-", color=color, linewidth=2.5,
                 markersize=9, label=name)
        for xi, yi in zip(x, ys):
            plt.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9, color=color)
    plt.xlabel("Benchmark Tasks", fontsize=13, fontweight="bold")
    plt.ylabel(ylabel, fontsize=13, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(x, tasks)
    plt.ylim(*ylim)
    plt.grid(True, linestyle=":", alpha=0.6, zorder=0)
    plt.legend(loc="best", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out, format=fmt, dpi=300)
    plt.close()
    print("saved", out)


make(ACC, "Multi-Model Cross-Task Accuracy Comparison",
     "Accuracy (ACC) (%)", (50, 78), "Figure_1.pdf", "pdf")
make(EM, "Multi-Model Cross-Task Exact Match Comparison",
     "Exact Match (EM) (%)", (35, 60), "Figure_2.pdf", "pdf")
