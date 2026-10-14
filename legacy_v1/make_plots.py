import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "results" / "legacy")
COLORS = {
    "GA": "#D85A30", "PSO": "#185FA5", "GWO": "#639922",
    "Jaya": "#7F77DD", "DE": "#D4537E", "SA": "#EF9F27",
}

raw = pd.read_csv(f"{OUT}/algo_comparison_raw.csv")
summary = pd.read_csv(f"{OUT}/algo_comparison_summary.csv")
hist = np.load(f"{OUT}/convergence_histories.npz")

# ------------------------------------------------------------------
# 1. Convergence plot: mean best-accuracy-so-far vs fitness evaluation,
#    with +/-1 std shaded band across the 5 seeds
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
for algo in hist.files:
    curves = hist[algo]  # (5 seeds, 1250 evals)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)
    x = np.arange(1, curves.shape[1] + 1)
    ax.plot(x, mean_curve, label=algo, color=COLORS[algo], linewidth=1.6)
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                     color=COLORS[algo], alpha=0.12)
ax.set_xlabel("Fitness evaluation number")
ax.set_ylabel("Best surrogate (NCC) accuracy so far")
ax.set_title("Convergence behaviour: mean ± 1 std across 5 seeds\n(identical evaluation budget = 1250 per run, all algorithms)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/convergence_plot.png", dpi=150)
plt.close(fig)
print("saved convergence_plot.png")

# ------------------------------------------------------------------
# 2. Accuracy (validated KNN) vs number of bands -- scatter, all runs
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for algo in raw["algorithm"].unique():
    sub = raw[raw["algorithm"] == algo]
    ax.scatter(sub["n_bands_selected"], sub["acc_knn5"], label=algo,
               color=COLORS[algo], s=70, alpha=0.85, edgecolor="white", linewidth=0.5)
    # mean marker
    ax.scatter(sub["n_bands_selected"].mean(), sub["acc_knn5"].mean(),
               color=COLORS[algo], s=220, marker="X", edgecolor="black", linewidth=1.2, zorder=5)
ax.set_xlabel("Number of selected bands")
ax.set_ylabel("Validated KNN (k=5) test accuracy")
ax.set_title("Accuracy vs. number of bands\n(small markers = individual seeds, X = mean)")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/accuracy_vs_bands.png", dpi=150)
plt.close(fig)
print("saved accuracy_vs_bands.png")

# ------------------------------------------------------------------
# 3. Accuracy (validated KNN) vs execution time -- scatter, all runs
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for algo in raw["algorithm"].unique():
    sub = raw[raw["algorithm"] == algo]
    ax.scatter(sub["exec_time_s"], sub["acc_knn5"], label=algo,
               color=COLORS[algo], s=70, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.scatter(sub["exec_time_s"].mean(), sub["acc_knn5"].mean(),
               color=COLORS[algo], s=220, marker="X", edgecolor="black", linewidth=1.2, zorder=5)
ax.set_xlabel("Execution time (s) for 1250 fitness evaluations")
ax.set_ylabel("Validated KNN (k=5) test accuracy")
ax.set_title("Accuracy vs. execution time\n(small markers = individual seeds, X = mean)")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/accuracy_vs_time.png", dpi=150)
plt.close(fig)
print("saved accuracy_vs_time.png")

# ------------------------------------------------------------------
# 4. Summary bar chart: mean KNN accuracy with error bars (stability)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
order = summary.sort_values("acc_knn_mean", ascending=False)
ax.bar(order["algorithm"], order["acc_knn_mean"], yerr=order["acc_knn_std"],
       color=[COLORS[a] for a in order["algorithm"]], capsize=5)
ax.set_ylabel("Validated KNN (k=5) test accuracy")
ax.set_title("Mean accuracy ± std across 5 seeds (stability)")
ax.set_ylim(0.70, 0.80)
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/accuracy_stability_bars.png", dpi=150)
plt.close(fig)
print("saved accuracy_stability_bars.png")
