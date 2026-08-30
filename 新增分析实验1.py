import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

datasets = ["DS-1000", "DDXPlus", "ToolBench"]
methods = ["Evo-Memory", "LifelongAgentBench", "MAM-StreamICL", "Ours"]

error_types = {
    "DS-1000": [
        "Algorithm Understanding",
        "Control Flow",
        "Variable/Data Structure",
        "Syntax/Formatting"
    ],
    "DDXPlus": [
        "Symptom Interpretation",
        "Differential Diagnosis",
        "Risk Factor Weighting",
        "Final Diagnosis Selection",
        "Justification Quality"
    ],
    "ToolBench": [
        "Intent Recognition",
        "Parameter Extraction",
        "API Call Syntax",
        "Result Interpretation"
    ]
}

error_dist = [
    [
        [0.45, 0.20, 0.15, 0.20],  # Evo-Memory
        [0.43, 0.19, 0.16, 0.22],  # LifelongAgentBench
        [0.41, 0.18, 0.17, 0.24],  # MAM-StreamICL
        [0.47, 0.21, 0.17, 0.15]   # Ours
    ],
    [
        [0.15, 0.25, 0.32, 0.18, 0.10],
        [0.14, 0.26, 0.33, 0.17, 0.10],
        [0.13, 0.25, 0.34, 0.18, 0.10],
        [0.16, 0.27, 0.31, 0.16, 0.10]
    ],
    [
        [0.10, 0.18, 0.42, 0.30],
        [0.09, 0.17, 0.44, 0.30],
        [0.08, 0.16, 0.46, 0.30],
        [0.11, 0.19, 0.29, 0.41]
    ]
]

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8

fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)
colors = sns.color_palette("muted", n_colors=5)

for i, (dataset, ax) in enumerate(zip(datasets, axes)):
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    type_list = error_types[dataset]
    
    for j in range(len(type_list)):
        values = [error_dist[i][m][j] for m in range(len(methods))]
        ax.bar(x, values, bottom=bottom, color=colors[j], label=type_list[j], edgecolor="white", linewidth=0.5)
        bottom += values
    
    ax.set_title(f"Panel {chr(65+i)}: {dataset}", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Proportion of Failed Samples")
    ax.set_ylim(0, 1.05)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 0.97])
plt.savefig("fig_error_distribution.pdf", bbox_inches="tight")
plt.savefig("fig_error_distribution.png", bbox_inches="tight", dpi=300)
plt.show()