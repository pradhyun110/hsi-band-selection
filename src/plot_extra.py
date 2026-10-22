"""
Figures and tables for the three extra studies (run src/extra_studies.py first).
  fig15_learning_set_size.png              accuracy vs share of pixels used for learning (10 ... 80 %)
  fig16_prescreening_all_methods.png       every method on 200 bands vs on the 164 pre-screened bands
  fig17_flatline_overall_accuracy.png      overall accuracy out to 100 bands (where the curve bends)
  table_split_ratio.csv, table_prescreening_all_methods.csv, table_flatline.csv
Run: python src/plot_extra.py
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; F = ROOT / "results" / "final"
GREY, INK = "#8a8985", "#1a1a19"
ORDER = ["Fisher-Cholesky", "GA", "Hybrid (GA + Fisher shortlist)", "Jaya", "Grey wolf (GWO)",
         "Base paper (MFE-CFSP)", "Entropy", "Variance"]
COL = dict(zip(ORDER, ["#1baf7a", "#2a78d6", "#4a3aa7", "#eb6834", "#eda100", "#e87ba4", "#008300", "#e34948"]))
LABEL = {m: m for m in ORDER}; LABEL["Fisher-Cholesky"] = "Fisher-Cholesky (chosen)"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": GREY})
main = pd.read_csv(F / "results_long.csv")

# ------------------------------------------------------------ learning-set size
sp = pd.read_csv(F / "split_ratio.csv"); fr = sorted(sp.learn_frac.unique())
g = sp.groupby(["method", "learn_frac"]); M, S = g.mean(numeric_only=True), g.std(numeric_only=True)
npx = sp.groupby("learn_frac").n_learn.first()
rows = []
for f in fr:
    r = {"Learning share (%)": int(round(100 * f)), "Learning pixels": int(npx[f]), "Exam pixels": int(sp[sp.learn_frac == f].n_exam.iloc[0]),
         "Fewest learning pixels in one crop": int(sp[sp.learn_frac == f].smallest_class_learn.iloc[0])}
    for m, nm in (("Fisher-Cholesky", "Fisher-Cholesky"), ("GA", "GA"), ("All 164 bands", "All 164 bands"), ("All 200 bands", "All 200 bands")):
        r[f"{nm} OA (%)"] = f"{100 * M.loc[(m, f)].OA:.1f} ± {100 * S.loc[(m, f)].OA:.1f}"
    r["Fisher-Cholesky AA (%)"] = f"{100 * M.loc[('Fisher-Cholesky', f)].AA:.1f} ± {100 * S.loc[('Fisher-Cholesky', f)].AA:.1f}"
    r["GA AA (%)"] = f"{100 * M.loc[('GA', f)].AA:.1f} ± {100 * S.loc[('GA', f)].AA:.1f}"
    r["Fisher-Cholesky accuracy on its own learning pixels (%)"] = f"{100 * M.loc[('Fisher-Cholesky', f)].train_OA:.1f}"
    r["GA minus Fisher-Cholesky OA (points)"] = f"{100 * (M.loc[('GA', f)].OA - M.loc[('Fisher-Cholesky', f)].OA):.1f}"
    r["GA selection time (s)"] = f"{sp[(sp.method == 'GA') & (sp.learn_frac == f)].selection_time_s.median():.1f}"
    r["Fisher-Cholesky selection time (s)"] = f"{sp[(sp.method == 'Fisher-Cholesky') & (sp.learn_frac == f)].selection_time_s.median():.3f}"
    rows.append(r)
pd.DataFrame(rows).to_csv(F / "table_split_ratio.csv", index=False); print(pd.DataFrame(rows).T.to_string())

SM = {"Fisher-Cholesky": (COL["Fisher-Cholesky"], "Fisher-Cholesky (chosen), 18 bands", 3, "-"), "GA": (COL["GA"], "GA, 18 bands", 1.8, "-"),
      "All 164 bands": (GREY, "All 164 pre-screened bands", 1.3, ":"), "All 200 bands": (GREY, "All 200 bands", 1.3, "--")}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8)); x = [100 * f for f in fr]
for ax, col, t in ((axs[0], "OA", "Overall accuracy (%)"), (axs[1], "AA", "Average accuracy (%)")):
    for m, (c, lab, lw, ls) in SM.items():
        mu = np.array([100 * M.loc[(m, f)][col] for f in fr]); sd = np.array([100 * S.loc[(m, f)][col] for f in fr])
        ax.plot(x, mu, color=c, lw=lw, ls=ls, marker="o" if ls == "-" else None, ms=4, label=lab)
        if ls == "-": ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
    ax.axvline(20, color="#1baf7a", lw=1, ls="--", alpha=0.6)
    ax.set_xticks(x, [f"{int(v)} %\n{npx[f]:,} px" for v, f in zip(x, fr)]); ax.set_title(t)
    ax.set_xlabel("Share of labelled pixels used for learning (and number of pixels)")
fig.suptitle("Effect of the learning-set size (6 random splits each; the rest of the labelled pixels form the exam)", fontsize=10)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig(F / "fig15_learning_set_size.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------ pre-screening, all methods
p2 = pd.read_csv(F / "prescreen_200_vs_164.csv")
chk = main[main.method == "Fisher-Cholesky on 200 bands"].merge(p2[p2.method == "Fisher-Cholesky"], on=["split", "K"], suffixes=("_a", "_b"))
assert len(chk) and np.allclose(chk.OA_a, chk.OA_b), "Fisher-Cholesky on 200 bands differs from the main run"
tab = []
for K in (18, 20):
    for m in ORDER:
        a = main[(main.method == m) & (main.K == K)].sort_values("split"); b = p2[(p2.method == m) & (p2.K == K)].sort_values("split")
        tab.append({"K": K, "Method": LABEL[m],
                    "OA on 164 bands (%)": f"{100 * a.OA.mean():.1f} ± {100 * a.OA.std():.1f}", "OA on 200 bands (%)": f"{100 * b.OA.mean():.1f} ± {100 * b.OA.std():.1f}",
                    "OA change from using 200 (points)": f"{100 * (b.OA.mean() - a.OA.mean()):+.1f}",
                    "AA on 164 bands (%)": f"{100 * a.AA.mean():.1f}", "AA on 200 bands (%)": f"{100 * b.AA.mean():.1f}",
                    # K = 20 on 200 bands was timed while another job shared the CPU, so only K = 18 times are quoted
                    "Time on 164 (s)": f"{a.selection_time_s.median():.3g}" if K == 18 else "n/a",
                    "Time on 200 (s)": f"{b.selection_time_s.median():.3g}" if K == 18 else "n/a"})
pd.DataFrame(tab).to_csv(F / "table_prescreening_all_methods.csv", index=False); print(pd.DataFrame(tab).to_string(index=False))
allb = {k: main[main.method == k] for k in ("All 164 bands", "All 200 bands")}
print("SVM on every band:", {k: f"{100 * v.OA.mean():.1f}" for k, v in allb.items()})

fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0)); K = 18; y = np.arange(len(ORDER)); h = 0.38
for ax, col, t in ((axs[0], "OA", "Overall accuracy (%)"), (axs[1], "AA", "Average accuracy (%)")):
    a = [main[(main.method == m) & (main.K == K)][col] for m in ORDER]; b = [p2[(p2.method == m) & (p2.K == K)][col] for m in ORDER]
    ax.barh(y - h / 2, [100 * v.mean() for v in a], h, xerr=[100 * v.std() for v in a], color=[COL[m] for m in ORDER], error_kw=dict(lw=0.8, ecolor=INK), label="164 pre-screened bands")
    ax.barh(y + h / 2, [100 * v.mean() for v in b], h, xerr=[100 * v.std() for v in b], color=[COL[m] for m in ORDER], alpha=0.35, error_kw=dict(lw=0.8, ecolor=INK), label="all 200 bands")
    for i, (u, v) in enumerate(zip(a, b)):
        ax.text(100 * (u.mean() + u.std()) + 1.0, i - h / 2, f"{100 * u.mean():.1f}", va="center", fontsize=8)
        ax.text(100 * (v.mean() + v.std()) + 1.0, i + h / 2, f"{100 * v.mean():.1f}", va="center", fontsize=8, color=GREY)
    ax.set_yticks(y, [LABEL[m] for m in ORDER]); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_title(t); ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"{t.split(' (')[0]} with {K} bands (%)")
from matplotlib.patches import Patch
fig.suptitle(f"Choosing {K} bands from 164 pre-screened candidates (solid) or from all 200 (faded); 6 splits, same search budget", fontsize=10)
fig.legend(handles=[Patch(color=INK, label="164 pre-screened bands (solid)"), Patch(color=INK, alpha=0.35, label="all 200 bands (faded)")], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.06, 1, 0.95)); fig.savefig(F / "fig16_prescreening_all_methods.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------ where the curve goes flat
ek = F / "results_long_extraK.csv"
if ek.exists():
    d = pd.concat([main, pd.read_csv(ek)], ignore_index=True); ks = [5, 10, 15, 18, 20, 25, 30, 40, 50, 60, 80, 100]
    t = []
    for m in ORDER:
        mu = d[d.method == m].groupby("K").OA.mean() * 100
        top = mu.loc[[k for k in ks if k in mu.index]].max(); plateau = mu.loc[[k for k in (60, 80, 100) if k in mu.index]].mean()
        row = {"Method": LABEL[m], **{f"K={k}": f"{mu[k]:.1f}" for k in ks if k in mu.index}, "Plateau (mean of K=60, 80, 100)": f"{plateau:.1f}"}
        for tol in (1, 2, 3):
            row[f"Smallest K within {tol} point{'s' if tol > 1 else ''} of plateau"] = int(next(k for k in ks if k in mu.index and mu[k] >= plateau - tol))
        row["OA at K=18 as % of plateau"] = f"{100 * mu[18] / plateau:.1f}"
        row["Gain K=10 to 20 (points)"] = f"{mu[20] - mu[10]:+.1f}"
        row["Gain K=20 to 100 (points)"] = f"{mu[100] - mu[20]:+.1f}"
        t.append(row)
    pd.DataFrame(t).to_csv(F / "table_flatline.csv", index=False); print(pd.DataFrame(t).T.to_string())

    # one clean panel: the methods that matter, OA only, so the bend at 15-20 bands is easy to see
    show = ["Fisher-Cholesky", "GA", "Hybrid (GA + Fisher shortlist)"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for m in show:
        g = d[d.method == m].groupby("K").OA; mu, sd = 100 * g.mean(), 100 * g.std()
        ax.plot(mu.index, mu.values, color=COL[m], lw=3 if m == "Fisher-Cholesky" else 1.8, marker="o", ms=4, label=LABEL[m])
        ax.fill_between(mu.index, mu - sd, mu + sd, color=COL[m], alpha=0.12, lw=0)
    ax.axhline(100 * main[main.method == "All 164 bands"].OA.mean(), color=GREY, ls=":", lw=1.3, label="All 164 pre-screened bands")
    ax.axhline(100 * main[main.method == "All 200 bands"].OA.mean(), color=GREY, ls="--", lw=1.3, label="All 200 bands")
    ax.axvspan(15, 20, color="#1baf7a", alpha=0.10, lw=0)
    fm = 100 * d[d.method == "Fisher-Cholesky"].groupby("K").OA.mean()
    for k, off in ((18, (6, -22)), (100, (-112, 16))):
        ax.annotate(f"{fm[k]:.1f} % at {k} bands", (k, fm[k]), textcoords="offset points", xytext=off, fontsize=8.5, color=COL["Fisher-Cholesky"])
    ax.set_xticks([5, 10, 15, 20, 30, 40, 50, 60, 80, 100]); ax.set_ylim(68, 87)
    ax.set_xlabel("Number of bands (K)"); ax.set_ylabel("Overall accuracy (%)")
    ax.set_title("Overall accuracy vs number of bands (green band = 15-20 bands we chose)")
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(F / "fig17_flatline_overall_accuracy.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------ small comparison across splits (run src/split_comparison.py first)
sc_path = F / "split_comparison.csv"
if sc_path.exists():
    sc = pd.read_csv(sc_path); sc["learn_frac"] = sc.learn_frac.round(2); fr5 = sorted(sc.learn_frac.unique())
    meths = ["Fisher-Cholesky", "GA", "Hybrid (GA + Fisher shortlist)", "Base paper (MFE-CFSP)", "Entropy"]
    g = sc.groupby(["method", "learn_frac"]); M2, S2 = g.mean(numeric_only=True), g.std(numeric_only=True)
    lab = lambda f: f"{int(round(100 * f))}/{int(round(100 * (1 - f)))}"
    rows = []
    for m in meths + ["All 164 bands"]:
        r = {"Method": LABEL.get(m, m)}
        for f in fr5: r[f"OA {lab(f)} (%)"] = f"{100 * M2.loc[(m, f)].OA:.1f} ± {100 * S2.loc[(m, f)].OA:.1f}"
        rows.append(r)
    r = {"Method": "GA minus Fisher-Cholesky (OA points)"}
    for f in fr5: r[f"OA {lab(f)} (%)"] = f"{100 * (M2.loc[('GA', f)].OA - M2.loc[('Fisher-Cholesky', f)].OA):+.1f}"
    rows.append(r)
    pd.DataFrame(rows).to_csv(F / "table_split_comparison.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8)); x = [100 * f for f in fr5]
    for ax, col, t in ((axs[0], "OA", "Overall accuracy (%)"), (axs[1], "AA", "Average accuracy (%)")):
        for m in meths:
            mu = np.array([100 * M2.loc[(m, f)][col] for f in fr5]); sd = np.array([100 * S2.loc[(m, f)][col] for f in fr5])
            ax.plot(x, mu, color=COL[m], lw=3 if m == "Fisher-Cholesky" else 1.8, marker="o", ms=4, label=LABEL[m])
            ax.fill_between(x, mu - sd, mu + sd, color=COL[m], alpha=0.12, lw=0)
        ax.plot(x, [100 * M2.loc[("All 164 bands", f)][col] for f in fr5], color=GREY, ls=":", lw=1.4, label="All 164 pre-screened bands")
        ax.set_xticks(x, [lab(f) for f in fr5]); ax.set_title(t); ax.set_xlabel("Learning / exam split (%)")
    fig.suptitle("Selected methods at 18 bands for five learning/exam splits (6 random splits each)", fontsize=10)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 0.95)); fig.savefig(F / "fig18_split_comparison.png", dpi=160); plt.close(fig)
