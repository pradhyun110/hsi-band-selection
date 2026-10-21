"""Two pictures from results/extra/extra_metrics.csv (made by extra_metrics.py)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results" / "extra"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8985"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GREY, "figure.facecolor": SURF})
df = pd.read_csv(OUT / "extra_metrics.csv"); K = 20
d = df[df.K == K].sort_values("OA", ascending=True).reset_index(drop=True)
col = [BLUE if m.startswith("Supervised") else (GREY if m == "Evenly spaced" else ORANGE) for m in d.method]
def style(ax, t, xl):
    ax.set_facecolor(SURF); ax.grid(True, axis="x", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9.5); ax.set_title(t, fontsize=11.5, color=INK, loc="left", fontweight="bold"); ax.set_xlabel(xl, color=INK2, fontsize=9.5)
panels = [("Accuracy (OA)", "OA", lambda v: v * 100, "%", "% of exam pixels right (axis starts at 50)", (50, 90)),
          ("Per-class average (AA)", "AA", lambda v: v * 100, "%", "% right, averaged over 16 classes (axis starts at 40)", (40, 90)),
          ("Information kept", "recon_R2", lambda v: v, "", "share of all 164 bands rebuildable (axis starts at 0.8)", (0.8, 1.0)),
          ("Repeated information", "redundancy", lambda v: v, "", "average correlation between kept bands (lower = better)", (0, 1.05)),
          ("Time to choose the bands", "selection_time_s", lambda v: v, "s", "seconds (log scale)", None)]
fig, axes = plt.subplots(1, 5, figsize=(21, 5.6), sharey=True)
for ax, (t, c, f, u, xl, lim) in zip(axes, panels):
    style(ax, t, xl); vals = f(d[c].values)
    ax.barh(range(len(d)), np.maximum(vals, 1e-3) if c == "selection_time_s" else vals, color=col, height=0.62)
    for i, v in enumerate(vals):
        lab = (f"{v:.0f}%" if u == "%" else (f"{v:.3f}" if c == "recon_R2" else (f"{v:.2f}" if c == "redundancy" else (f"{v*1000:.0f} ms" if v < 1 else f"{v:.1f} s"))))
        x = max(v, 1e-3) if c == "selection_time_s" else v
        ax.text(x * (1.15 if c == "selection_time_s" else 1) + (0 if c == "selection_time_s" else (lim[1] - lim[0]) * 0.012 if lim else 0), i, lab, va="center", fontsize=9.5, color=INK)
    if c == "selection_time_s": ax.set_xscale("log"); ax.set_xlim(8e-4, 40)
    else: ax.set_xlim(*lim)
axes[0].set_yticks(range(len(d))); axes[0].set_yticklabels(d.method, fontsize=10, color=INK)
fig.suptitle(f"Scorecard with {K} of 200 bands: blue = uses labels, orange = no labels, grey = evenly spaced", fontsize=14, fontweight="bold", color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "figE1_scorecard_K20.png", dpi=130); plt.close(fig)

# cost versus number of bands (supervised sets and the two references)
r = pd.concat([df[(df.method.str.startswith("Supervised")) & df.K.isin([15, 20, 30])].sort_values("K"), df[df.K >= 164]])
lab = [f"{int(k)} bands" if k < 164 else ("164 cleaned" if k == 164 else "200 raw") for k in r.K]
cc = [BLUE] * 3 + [GREY] * 2
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, (c, t, xl, fmt) in zip(axes, [("bytes_per_pixel", "Data per pixel", "bytes (16-bit values)", "{:.0f}"), ("us_per_pixel", "Classifier time per pixel", "microseconds (SVM, this computer)", "{:.0f}"),
                                      ("svm_predict_s", "Time to classify all 8,198 exam pixels", "seconds", "{:.2f}"), ("model_kb", "Size of the stored classifier", "kilobytes", "{:.0f}")]):
    style(ax, t, xl); ax.grid(True, axis="y", color=GRID); ax.grid(False, axis="x")
    ax.bar(range(len(r)), r[c].values, color=cc, width=0.62)
    for i, v in enumerate(r[c].values): ax.text(i, v * 1.02, fmt.format(v), ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(r))); ax.set_xticklabels(lab, fontsize=9.5, color=INK); ax.set_ylim(0, r[c].max() * 1.15)
fig.suptitle("What fewer bands save (supervised band sets, blue; references, grey)", fontsize=14, fontweight="bold", color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "figE2_cost_vs_bands.png", dpi=130); plt.close(fig)
print("ok")
