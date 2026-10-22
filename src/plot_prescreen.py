"""Figure 2: entropy pre-screening.   Run: python src/plot_prescreen.py   (after src/entropy_prescreen.py)
Top: entropy of every band and the thresholds compared (bands removed by Otsu shaded).
Bottom: for each threshold rule, % of bands removed vs % of total entropy kept."""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; P = ROOT / "results" / "prescreen"; F = ROOT / "results" / "final"
F.mkdir(parents=True, exist_ok=True)
ent = pd.read_csv(P / "entropy_per_band.csv"); comp = pd.read_csv(P / "prescreening_comparison.csv")
keep = set(np.load(P / "selected_band_indices.npy").tolist())
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8985"})
SHORT = {"Mean threshold": "Mean", "Median threshold": "Median", "Mean - 0.5*std threshold": "Mean - 0.5 std",
         "Otsu automatic threshold": "Otsu (chosen)", "Percentile cut (drop lowest 25%)": "Drop lowest 25 %",
         "Percentile cut (drop lowest 33%)": "Drop lowest 33 %", "Fixed top-K (K=150)": "Top 150"}
fig, axs = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1.3, 1]})
ax = axs[0]; b, h = ent.band_index.values, ent.entropy_bits.values
for i in b:
    if i not in keep: ax.axvspan(i - 0.5, i + 0.5, color="#e34948", alpha=0.12, lw=0)
ax.plot(b, h, color="#2a78d6", lw=1.6, label="Entropy of each band")
otsu = comp.loc[comp.Strategy == "Otsu automatic threshold", "Threshold (bits)"].item()
ax.axhline(otsu, color="#e34948", lw=1.6, label=f"Otsu threshold ({otsu:.2f} bits)")
for s, c in (("Mean threshold", "#eda100"), ("Median threshold", "#8a8985"), ("Mean - 0.5*std threshold", "#1baf7a")):
    v = comp.loc[comp.Strategy == s, "Threshold (bits)"].item()
    ax.axhline(v, color=c, lw=1, ls="--", label=f"{SHORT[s]} threshold ({v:.2f} bits)")
ax.fill_between([], [], color="#e34948", alpha=0.12, label="Bands removed by Otsu (36)")
ax.set_xlabel("Band number"); ax.set_ylabel("Entropy (bits)"); ax.set_xlim(-1, 200)
ax.set_title("Entropy of each band (Indian Pines, 200 bands)"); ax.legend(loc="lower left", fontsize=8, ncol=2)
ax = axs[1]; x = np.arange(len(comp)); w = 0.38
ax.bar(x - w / 2, comp["% removed"], w, color="#eb6834", label="Bands removed (%)")
ax.bar(x + w / 2, comp["Information retained (%)"], w, color="#2a78d6", label="Entropy kept (%)")
for xi, v in zip(x, comp["% removed"]): ax.text(xi - w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
for xi, v in zip(x, comp["Information retained (%)"]): ax.text(xi + w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(x, [SHORT[s] for s in comp.Strategy]); ax.set_ylabel("%"); ax.set_ylim(0, 100)
ax.set_title("Threshold rules compared"); ax.legend(loc="upper right", fontsize=8); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(F / "fig2_prescreening.png", dpi=160); print("saved fig2")
