import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "results")

entropy_df = pd.read_csv(f"{OUT}/entropy_per_band.csv")
comp_df = pd.read_csv(f"{OUT}/prescreening_comparison.csv")
with open(f"{OUT}/prescreening_report.json") as f:
    report = json.load(f)

entropy = entropy_df["entropy_bits"].values
bands = entropy_df["band_index"].values

fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [1.3, 1]})

# --- Top: entropy per band with threshold lines ---
ax = axes[0]
ax.plot(bands, entropy, color="#185FA5", linewidth=1.2, label="Per-band entropy")
colors = {
    "Mean threshold": "#EF9F27",
    "Median threshold": "#9F9F9F",
    "Mean - 0.5*std threshold": "#639922",
    "Otsu automatic threshold": "#D85A30",
    "Percentile cut (drop lowest 25%)": "#7F77DD",
    "Percentile cut (drop lowest 33%)": "#D4537E",
}
for _, row in comp_df.iterrows():
    name = row["Strategy"]
    if name in colors:
        ax.axhline(row["Threshold (bits)"], color=colors[name], linestyle="--", linewidth=1,
                   label=f"{name} ({row['Threshold (bits)']:.2f} bits)")

rec = report["recommended_strategy"]
ax.set_title(f"Per-band Shannon entropy, Indian Pines (200 bands, 256-bin histogram)\nRecommended: {rec}", fontsize=11)
ax.set_xlabel("Band index")
ax.set_ylabel("Entropy (bits)")
ax.legend(fontsize=8, loc="lower left", ncol=2)
ax.grid(alpha=0.25)

# --- Bottom: strategy comparison bars ---
ax2 = axes[1]
x = np.arange(len(comp_df))
width = 0.38
ax2.bar(x - width/2, comp_df["% removed"], width, label="% bands removed", color="#D85A30")
ax2.bar(x + width/2, comp_df["Information retained (%)"], width, label="Information retained (%)", color="#185FA5")
ax2.set_xticks(x)
ax2.set_xticklabels(comp_df["Strategy"], rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("%")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.25, axis="y")
ax2.set_title("Strategy comparison: compression vs. information retained", fontsize=11)

plt.tight_layout()
plt.savefig(f"{OUT}/prescreening_comparison.png", dpi=150)
print("saved plot")
