"""
Final figures and tables.   Run: python src/plot_final.py   (after src/final_comparison.py)

  fig3_prescreening_effect.png        accuracy with vs without the 164-band pre-screening
  fig4_classification_metrics.png     OA, AA, kappa, precision, F1, specificity vs number of bands
  fig5_detection_reconstruction.png   CEM target-detection AUC and SSIM vs number of bands
  fig6_time_vs_accuracy_K20.png       selection time vs overall accuracy at 20 bands
  fig7_per_class_sensitivity_K20.png  sensitivity of every crop at 20 bands
  fig1b_learning_exam_split.png       what a learning / exam split is
  fig8a_classification_maps_K20.png   classification maps at 20 bands, every method (split 42)
  fig8b_error_maps_K20.png            the exam pixels each method gets wrong
  fig9_selected_bands_K20.png         which bands each method keeps at 20 bands (split 42)
  fig10_base_paper_ablation_K20.png   the base paper's module-by-module table repeated on our data
  fig11_confusion_matrices_K20.png    confusion matrices at 20 bands (exam pixels of all 6 splits added up)
  fig12_band_stability_K20.png        how similar the chosen bands are from split to split (exact and within 2 bands)
  fig13_cholesky_criteria.png         Fisher vs other criteria on the same recursive-Cholesky engine
  fig14_spread_rules.png              Fisher-Cholesky with rules that force the bands to spread out
  table_K18.csv, table_K20.csv        every metric at 18 and 20 bands, mean +- std over 6 splits
  table_K20.csv                       every metric at 20 bands, mean +- std over 6 splits
  summary_all.csv                     every metric, every method, every K (mean and std)
(fig1 comes from plot_dataset.py, fig2 from plot_prescreen.py)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parent.parent; F = ROOT / "results" / "final"
df = pd.read_csv(F / "results_long.csv")

ORDER = ["Fisher-Cholesky", "GA", "Hybrid (GA + Fisher shortlist)", "Jaya", "Grey wolf (GWO)",
         "Base paper (MFE-CFSP)", "Entropy", "Variance"]
COL = dict(zip(ORDER, ["#1baf7a", "#2a78d6", "#4a3aa7", "#eb6834", "#eda100", "#e87ba4", "#008300", "#e34948"]))
LABEL = {m: m for m in ORDER}; LABEL["Fisher-Cholesky"] = "Fisher-Cholesky (chosen)"
GREY, INK = "#8a8985", "#1a1a19"
CLASSES = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
           "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
           "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
PCT = {"OA", "AA", "Precision", "Sensitivity", "Specificity", "F1"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": GREY})
KS = sorted(df[df.method == "Fisher-Cholesky"].K.unique())
g = df.groupby(["method", "K"]); MEAN, STD = g.mean(numeric_only=True), g.std(numeric_only=True)
REF = df[df.method.isin(["All 164 bands", "All 200 bands"])].groupby("method").mean(numeric_only=True)
sc = lambda col: 100.0 if col in PCT else 1.0


def lines(ax, col, methods=ORDER, labels=LABEL, colours=COL, refs=("All 200 bands",)):
    s = sc(col)
    for m in methods:
        mu, sd = MEAN.loc[m][col] * s, STD.loc[m][col] * s
        lw = 3 if m == "Fisher-Cholesky" else 1.8
        ax.plot(mu.index, mu.values, color=colours[m], lw=lw, marker="o", ms=4, label=labels[m], zorder=3 if lw == 3 else 2)
        ax.fill_between(mu.index, mu - sd, mu + sd, color=colours[m], alpha=0.12, lw=0)
    for r, ls in zip(refs, ("--", ":")):
        v = REF.loc[r, col]
        if not np.isnan(v): ax.axhline(v * s, color=GREY, ls=ls, lw=1.3, label=r)
    ax.set_xticks(KS); ax.set_xlabel("Number of bands (K)")


def legend_below(fig, ax, ncol=5, y=0.0):
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------- fig3 pre-screening effect
pm = ["Fisher-Cholesky", "Fisher-Cholesky on 200 bands"]
pl = {"Fisher-Cholesky": "Fisher-Cholesky on the 164 pre-screened bands", "Fisher-Cholesky on 200 bands": "Fisher-Cholesky on all 200 bands"}
pc = {"Fisher-Cholesky": COL["Fisher-Cholesky"], "Fisher-Cholesky on 200 bands": GREY}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, col, t in ((axs[0], "OA", "Overall accuracy (%)"), (axs[1], "AA", "Average accuracy (%)")):
    lines(ax, col, pm, pl, pc, refs=("All 164 bands", "All 200 bands")); ax.set_title(t)
legend_below(fig, axs[0], ncol=4); fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(F / "fig3_prescreening_effect.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig4 classification metrics
MET = [("OA", "Overall accuracy (%)"), ("AA", "Average accuracy (%)"), ("Kappa", "Kappa coefficient"),
       ("Precision", "Precision (%)"), ("F1", "F1 score (%)"), ("Specificity", "Specificity (%)")]
fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
for ax, (col, t) in zip(axs.flat, MET):
    lines(ax, col); ax.set_title(t)
fig.suptitle("Indian Pines, SVM classifier, mean and spread over 6 splits", fontsize=10)
legend_below(fig, axs[0, 0], ncol=5); fig.tight_layout(rect=(0, 0.07, 1, 0.97))
fig.savefig(F / "fig4_classification_metrics.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig5 detection and reconstruction
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
lines(axs[0], "CEM_AUC", refs=("All 164 bands",)); axs[0].set_title("Target detection (CEM AUC)")
lines(axs[1], "SSIM", refs=()); axs[1].set_title("Reconstruction quality (SSIM)")
legend_below(fig, axs[0], ncol=5); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(F / "fig5_detection_reconstruction.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig6 time vs accuracy
k20 = df[df.K == 20]
fig, ax = plt.subplots(figsize=(9, 5.6))
OFF = {"Fisher-Cholesky": (12, -16), "GA": (-175, -34), "Hybrid (GA + Fisher shortlist)": (40, 30), "Jaya": (40, -40),
       "Grey wolf (GWO)": (40, -8), "Base paper (MFE-CFSP)": (10, 6), "Entropy": (10, 2), "Variance": (10, -12)}
for m in ORDER:
    d = k20[k20.method == m]; t = max(d.selection_time_s.median(), 1e-3); y, e = 100 * d.OA.mean(), 100 * d.OA.std()
    ax.errorbar(t, y, yerr=e, fmt="o", ms=11 if m == "Fisher-Cholesky" else 8, color=COL[m], ecolor=COL[m], capsize=3,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    far = abs(OFF[m][0]) > 20
    ax.annotate(f"{LABEL[m]}\n{y:.1f} %, {t:.3g} s", (t, y), textcoords="offset points", xytext=OFF[m], fontsize=8, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=COL[m], lw=1) if far else None)
ax.axhline(100 * REF.loc["All 200 bands", "OA"], color=GREY, ls="--", lw=1.2); ax.text(1.2e-3, 100 * REF.loc["All 200 bands", "OA"] + 0.4, "All 200 bands", color=GREY, fontsize=8)
ax.set_xscale("log"); ax.set_xlim(1e-3, 1000); ax.set_ylim(54, 89); ax.set_xlabel("Band selection time (s, log scale)"); ax.set_ylabel("Overall accuracy (%)")
ax.set_title("Selection time vs accuracy with 20 bands (mean over 6 splits)")
fig.tight_layout(); fig.savefig(F / "fig6_time_vs_accuracy_K20.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig7 per-class sensitivity
show = ["Fisher-Cholesky", "GA", "Base paper (MFE-CFSP)"]
R = {m: np.mean([json.loads(s) for s in k20[k20.method == m].per_class_sensitivity], axis=0) * 100 for m in show}
r200 = np.mean([json.loads(s) for s in df[df.method == "All 200 bands"].per_class_sensitivity], axis=0) * 100
counts = np.bincount(np.load(ROOT / "data" / "IPgt.npy").ravel(), minlength=17)[1:]
fig, ax = plt.subplots(figsize=(9.5, 8)); y = np.arange(16); hgt = 0.26
for i, m in enumerate(show):
    ax.barh(y + (i - 1) * hgt, R[m], height=hgt - 0.03, color=COL[m], label=LABEL[m])
ax.scatter(r200, y, marker="|", s=170, color=INK, zorder=3, label="All 200 bands")
ax.set_yticks(y, [f"{CLASSES[i]} ({counts[i]})" for i in range(16)]); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("Sensitivity (%)"); ax.grid(axis="y", visible=False)
ax.set_title("Sensitivity of each crop with 20 bands (pixel count in brackets)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
fig.tight_layout(); fig.savefig(F / "fig7_per_class_sensitivity_K20.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig8a maps, fig8b error maps, fig1b split
gt = np.load(ROOT / "data" / "IPgt.npy"); Z = np.load(F / "maps_split42_K20.npz")
cols16 = plt.cm.tab20(np.linspace(0, 1, 20))[:16]; cmap = ListedColormap(np.vstack([[1, 1, 1, 1], cols16]))
mm = ORDER + ["All 200 bands"]; LAB9 = dict(LABEL, **{"All 200 bands": "All 200 bands"})
r42 = df[(df.split == 42) & ((df.K == 20) | (df.method == "All 200 bands"))].set_index("method")
ylab = gt.reshape(-1)[Z["labelled"]]; test = np.zeros(len(ylab), bool); test[Z["test_idx"]] = True
def to_img(v):
    img = np.zeros(gt.size, int); img[Z["labelled"]] = v; return img.reshape(gt.shape)
fig, axs = plt.subplots(2, 5, figsize=(19, 10.4))
axs[0, 0].imshow(gt, cmap=cmap, vmin=0, vmax=16, interpolation="nearest"); axs[0, 0].set_title("Ground truth")
for ax, m in zip(axs.flat[1:], mm):
    ax.imshow(to_img(Z[m.replace(" ", "_")]), cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
    ax.set_title(f"{LAB9[m]}\nOA {100 * r42.loc[m, 'OA']:.1f} %")
for ax in axs.flat: ax.axis("off")
fig.legend(handles=[Patch(color=cols16[i], label=CLASSES[i]) for i in range(16)], loc="lower center", ncol=6, fontsize=8, frameon=False)
fig.suptitle("Classification maps with 20 bands (split 42; OA measured on the exam pixels)", fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 0.97), h_pad=4); fig.savefig(F / "fig8a_classification_maps_K20.png", dpi=150); plt.close(fig)

ecm = ListedColormap(["#ffffff", "#e6e6e3", "#e34948"])          # not exam pixel, exam correct, exam wrong
fig, axs = plt.subplots(3, 3, figsize=(13, 13.6))
for ax, m in zip(axs.flat, mm):
    pred = Z[m.replace(" ", "_")]; code = np.where(test, np.where(pred == ylab, 1, 2), 0)
    ax.imshow(to_img(code), cmap=ecm, vmin=0, vmax=2, interpolation="nearest"); ax.axis("off")
    ax.set_title(f"{LAB9[m]}\n{(code == 2).sum()} wrong of {test.sum()} exam pixels (OA {100 * r42.loc[m, 'OA']:.1f} %)", fontsize=9)
fig.legend(handles=[Patch(color="#e6e6e3", label="Exam pixel, correct"), Patch(color="#e34948", label="Exam pixel, wrong"),
                    Patch(facecolor="#ffffff", edgecolor=GREY, label="Learning pixel or unlabelled (not graded)")],
           loc="lower center", ncol=3, frameon=False)
fig.suptitle("Where each method makes mistakes (20 bands, split 42)", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 0.97), h_pad=2); fig.savefig(F / "fig8b_error_maps_K20.png", dpi=150); plt.close(fig)

import sys; sys.path.insert(0, str(ROOT / "src")); from data_setup import make_setup
splitcm = ListedColormap(["#ffffff", "#2a78d6", "#e6e6e3"])
fig, axs = plt.subplots(1, 3, figsize=(14, 5.4))
axs[0].imshow(gt, cmap=cmap, vmin=0, vmax=16, interpolation="nearest"); axs[0].set_title("All 10,249 labelled pixels\n(16 crops; white = unlabelled)")
for ax, sd in zip(axs[1:], (42, 1)):
    S_ = make_setup(sd); code = np.full(len(ylab), 2); code[S_["train_idx"]] = 1
    ax.imshow(to_img(code), cmap=splitcm, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(f"Split {sd}: {len(S_['train_idx'])} learning pixels (blue)\n{len(S_['test_idx'])} exam pixels (grey)")
for ax in axs: ax.axis("off")
fig.suptitle("A split picks 20 % of the pixels of every crop at random for learning; the other 80 % are the exam. 6 different random splits are used.", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(F / "fig1b_learning_exam_split.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- fig9 selected bands
sets = json.load(open(F / "band_sets.json"))["split=42"]
cube = np.load(ROOT / "data" / "indianpinearray.npy").astype(float); spec = cube.reshape(-1, 200).mean(0)
cand = set(np.load(ROOT / "results" / "prescreen" / "selected_band_indices.npy").tolist())
fig, (a0, a1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
for i in range(200):
    if i not in cand:
        for a in (a0, a1): a.axvspan(i - 0.5, i + 0.5, color="#e34948", alpha=0.10, lw=0)
a0.plot(np.arange(200), spec, color=INK, lw=1.4); a0.set_ylabel("Mean reflectance"); a0.set_title("Mean spectrum of the image (shaded = removed by pre-screening)")
for r, m in enumerate(ORDER):
    b = sets[f"{m}|K=20"]; a1.scatter(b, np.full(len(b), r), marker="|", s=220, color=COL[m], linewidths=2.2)
a1.set_yticks(range(len(ORDER)), [LABEL[m] for m in ORDER]); a1.invert_yaxis(); a1.set_xlim(-1, 200)
a1.set_xlabel("Band number"); a1.set_title("Bands kept by each method (20 bands, split 42)"); a1.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(F / "fig9_selected_bands_K20.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig10 base paper ablation
ab = pd.read_csv(F / "ablation_base_paper_K20.csv")
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True); y = np.arange(len(ab))
axs[0].barh(y, 100 * ab.OA, color=COL["Base paper (MFE-CFSP)"], height=0.6)
for yi, v in zip(y, ab.OA): axs[0].text(100 * v + 0.2, yi, f"{100 * v:.1f} %", va="center")
axs[0].set_xlim(100 * ab.OA.min() - 4, 100 * ab.OA.max() + 2.5); axs[0].set_title("Overall accuracy (%)")
axs[0].set_yticks(y, ab.step); axs[0].invert_yaxis()
axs[1].barh(y, ab.time_s, color=GREY, height=0.6)
for yi, v in zip(y, ab.time_s): axs[1].text(v, yi, f" {v:.2f} s", va="center")
axs[1].set_xlim(0, ab.time_s.max() * 1.3); axs[1].set_title("Selection time (s)")
for a in axs: a.grid(axis="y", visible=False)
axs[0].text(0, -0.22, "Accuracy axis starts above 0", transform=axs[0].transAxes, fontsize=7, color=GREY)
fig.suptitle("Base paper modules added one at a time (20 bands, split 42)", fontsize=10)
fig.tight_layout(); fig.savefig(F / "fig10_base_paper_ablation_K20.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig11 confusion matrices
CM = np.load(F / "confusion_matrices_K20_sum6splits.npz")
cmm = ["Fisher-Cholesky", "GA", "Base paper (MFE-CFSP)"]
fig, axs = plt.subplots(1, 3, figsize=(21, 7.6))
for ax, m in zip(axs, cmm):
    c = CM[m.replace(" ", "_")].astype(float); P = 100 * c / c.sum(1, keepdims=True)
    ax.imshow(P, cmap="Blues", vmin=0, vmax=100)
    for i in range(16):
        for j in range(16):
            if P[i, j] >= 5 or i == j:
                ax.text(j, i, f"{P[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if P[i, j] > 55 else INK)
    ax.set_xticks(range(16), range(1, 17)); ax.set_yticks(range(16), [f"{i + 1} {CLASSES[i]}" for i in range(16)] if m == cmm[0] else range(1, 17))
    ax.set_xlabel("Predicted crop"); ax.grid(False)
    oa = 100 * np.trace(c) / c.sum(); ax.set_title(f"{LABEL[m]}  (OA {oa:.1f} %)")
axs[0].set_ylabel("True crop")
fig.suptitle("Confusion matrices with 20 bands: % of each true crop (row) given each label; exam pixels of 6 splits added up; cells under 5 % left blank", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(F / "fig11_confusion_matrices_K20.png", dpi=160); plt.close(fig)
for m in cmm + ["All 200 bands"]:
    pd.DataFrame(CM[m.replace(" ", "_")], index=CLASSES, columns=CLASSES).to_csv(F / f"confusion_matrix_K20_{m.split(' (')[0].replace(' ', '_')}.csv")

# ---------------------------------------------------------------- fig12 band stability
allsets = json.load(open(F / "band_sets.json"))
def _sets(m, K=20): return [sorted(allsets[sp][f"{m}|K={K}"]) for sp in allsets]
def jacc(m, K=20):
    S_ = [set(x) for x in _sets(m, K)]
    return np.mean([len(a & b) / len(a | b) for i, a in enumerate(S_) for b in S_[i + 1:]])
def near(m, K=20, t=2):
    S_ = [np.array(x) for x in _sets(m, K)]
    f = lambda a, b: np.mean([np.min(np.abs(b - x)) <= t for x in a])
    return np.mean([(f(a, b) + f(b, a)) / 2 for i, a in enumerate(S_) for b in S_[i + 1:]])
fig, ax = plt.subplots(figsize=(10, 5.8)); y = np.arange(len(ORDER)); h = 0.38
ex = [100 * jacc(m) for m in ORDER]; nr = [100 * near(m) for m in ORDER]
ax.barh(y - h / 2, ex, height=h - 0.04, color=[COL[m] for m in ORDER], label="Exactly the same band")
ax.barh(y + h / 2, nr, height=h - 0.04, color=[COL[m] for m in ORDER], alpha=0.45, label="Same band or a neighbour within 2 bands")
for yi, a_, b_ in zip(y, ex, nr):
    ax.text(a_ + 1, yi - h / 2, f"{a_:.0f} %", va="center", fontsize=8); ax.text(b_ + 1, yi + h / 2, f"{b_:.0f} %", va="center", fontsize=8)
ax.set_yticks(y, [LABEL[m] for m in ORDER]); ax.invert_yaxis(); ax.set_xlim(0, 112); ax.grid(axis="y", visible=False)
ax.set_xlabel("Overlap of the chosen 20 bands between splits (%)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8, frameon=False)
ax.set_title("Do the same bands get chosen on every split?")
fig.tight_layout(); fig.savefig(F / "fig12_band_stability_K20.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig13 other criteria on the Cholesky engine
cc = pd.read_csv(F / "cholesky_criteria.csv")
CC = {"Fisher (chosen)": COL["Fisher-Cholesky"], "Mean pairwise class distance": "#4a3aa7",
      "Worst-pair class distance": "#eb6834", "Maximum volume (label-free)": "#e87ba4"}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, col, t in ((axs[0], "OA", "Overall accuracy (%)"), (axs[1], "AA", "Average accuracy (%)")):
    for c, colr in CC.items():
        d = cc[cc.criterion == c].groupby("K")[col]; mu, sd = 100 * d.mean(), 100 * d.std()
        ax.plot(mu.index, mu.values, color=colr, lw=3 if c.startswith("Fisher") else 1.8, marker="o", ms=4, label=c)
        ax.fill_between(mu.index, mu - sd, mu + sd, color=colr, alpha=0.12, lw=0)
    ga = 100 * MEAN.loc["GA"][col]; ax.plot(ga.index, ga.values, color=COL["GA"], ls="--", lw=1.3, label="GA (search, for reference)")
    ax.set_title(t); ax.set_xticks(KS); ax.set_xlabel("Number of bands (K)")
fig.suptitle("Criteria tried on the same recursive-Cholesky engine (6 splits; all select in about 0.01 s)", fontsize=10)
legend_below(fig, axs[0], ncol=5); fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(F / "fig13_cholesky_criteria.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- tables
cols = ["OA", "AA", "Kappa", "MCC", "Precision", "Sensitivity", "Specificity", "F1", "CEM_AUC", "SSIM", "selection_time_s", "fitness_calls"]
s = df.groupby(["method", "K"])[cols].agg(["mean", "std"]); s.columns = [f"{a}_{b}" for a, b in s.columns]
s.round(5).to_csv(F / "summary_all.csv")
for KT in (18, 20):
    rows = []
    for m in ORDER + ["Fisher-Cholesky on 200 bands", "All 164 bands", "All 200 bands"]:
        d = df[(df.method == m) & ((df.K == KT) | df.method.str.startswith("All"))]
        r = {"Method": LABEL.get(m, m), "Bands": int(d.K.iloc[0])}
        for c, name in (("OA", "OA (%)"), ("AA", "AA (%)"), ("Kappa", "Kappa"), ("MCC", "MCC"), ("Precision", "Precision (%)"), ("F1", "F1 (%)"),
                        ("Specificity", "Specificity (%)"), ("CEM_AUC", "CEM AUC"), ("SSIM", "SSIM")):
            k = sc(c); fmt = "{:.1f} ± {:.1f}" if k == 100 else "{:.3f} ± {:.3f}"
            r[name] = fmt.format(k * d[c].mean(), k * d[c].std())
        r["Selection time (s)"] = f"{d.selection_time_s.median():.3g}"
        sel = m in ORDER or m == "Fisher-Cholesky on 200 bands"
        r["Band overlap across splits (%)"] = f"{100 * jacc(m, KT):.0f}" if sel else ""
        r["Overlap within 2 bands (%)"] = f"{100 * near(m, KT):.0f}" if sel else ""
        rows.append(r)
    pd.DataFrame(rows).to_csv(F / f"table_K{KT}.csv", index=False)
    print(f"--- K = {KT}"); print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- fig14 spread rules
sr = pd.read_csv(F / "spread_rules.csv")
SR = {"Fisher (no rule)": COL["Fisher-Cholesky"], "Min gap 3 bands": "#2a78d6", "Min gap 6 bands": "#eb6834", "Max per 40-band region": "#4a3aa7"}
SL = {"Fisher (no rule)": "Fisher-Cholesky, no rule (chosen)", "Min gap 3 bands": "No two bands closer than 3",
      "Min gap 6 bands": "No two bands closer than 6", "Max per 40-band region": "At most ceil(K/5)+1 bands per 40-band region"}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, col, t in ((axs[0], "OA", "Overall accuracy (%)"), (axs[1], "AA", "Average accuracy (%)")):
    for rule, c in SR.items():
        g_ = sr[sr.rule == rule].groupby("K")[col]; mu, sd = 100 * g_.mean(), 100 * g_.std()
        ax.plot(mu.index, mu.values, color=c, lw=3 if rule.startswith("Fisher") else 1.8, marker="o", ms=4, label=SL[rule])
        ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
    ax.set_title(t); ax.set_xticks(sorted(sr.K.unique())); ax.set_xlabel("Number of bands (K)")
fig.suptitle("Forcing the bands to spread across the spectrum (Fisher-Cholesky, 6 splits)", fontsize=10)
legend_below(fig, axs[0], ncol=2); fig.tight_layout(rect=(0, 0.13, 1, 0.95))
fig.savefig(F / "fig14_spread_rules.png", dpi=160); plt.close(fig)
