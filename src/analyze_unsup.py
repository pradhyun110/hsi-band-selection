"""Supervised wrapper vs label-free search objectives, same pool / budget / seeds / test pixels."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import *

KS = [10, 20, 30, 40]; ALG = ["SA", "GA", "GWO", "Jaya", "PSO", "DE"]; CLS = ["SVM-RBF", "KNN5", "LDA", "LogReg", "RF"]
SURF, INK, INK2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8a8985"
sup = pd.read_csv(OUT / "eval_long.csv"); uns = pd.read_csv(OUT / "eval_unsup_long.csv"); repo = pd.read_csv(OUT / "repo_comparison.csv")
S_ = sup[(sup.group == "algorithm") & (sup.fitness_kind == "lda") & sup.method.isin(ALG)]
U1 = uns[(uns.group == "algorithm") & (uns.fitness_kind == "u1") & uns.method.isin(ALG)]
U2 = uns[(uns.group == "algorithm") & (uns.fitness_kind == "u2") & uns.method.isin(ALG)]
RS1 = uns[(uns.group == "algorithm") & (uns.fitness_kind == "u1") & (uns.method == "RS")]
G = uns[(uns.group == "baseline") & (uns.method == "greedy_R2")]
UNI = sup[(sup.group == "baseline") & (sup.method == "uniform")]; RND = sup[(sup.group == "baseline") & (sup.method == "random")]
ref164 = sup[(sup.group == "reference") & (sup.method == "all164")].set_index("classifier"); ref200 = sup[(sup.group == "reference") & (sup.method == "all200")].set_index("classifier")
KM = repo[(repo.method == "kmeans") & (repo.pool.str.contains("pre-screened"))].set_index("K")

def oa(df, K, c="SVM-RBF", col="OA"):
    s = df[(df.K == K) & (df.classifier == c)][col]; return s.mean(), s.std()

rows = []
for K in KS:
    r = dict(K=K)
    for name, d in (("supervised wrapper (LDA)", S_), ("unsup U1 reconstruction R2", U1), ("unsup U2 entropy-redundancy", U2),
                    ("U1 random search", RS1), ("greedy R2", G), ("uniform spacing", UNI), ("random subsets", RND)):
        m, s = oa(d, K); r[name] = m; r[name + "_sd"] = s
    r["k-means (repo, pre-screened)"] = KM.loc[K, "OA"]
    rows.append(r)
t = pd.DataFrame(rows); t.to_csv(OUT / "tbl_unsup_vs_sup_svm.csv", index=False)
show = t[["K", "supervised wrapper (LDA)", "unsup U1 reconstruction R2", "unsup U2 entropy-redundancy", "U1 random search", "greedy R2", "uniform spacing", "k-means (repo, pre-screened)", "random subsets"]].round(4)
print("== SVM-RBF test OA, mean of 90 runs (6 algs x 15 seeds) ==\n", show.to_string(index=False))
print("all164 %.4f, all200 %.4f" % (ref164.loc["SVM-RBF", "OA"], ref200.loc["SVM-RBF", "OA"]))
print("\nSDs of run-level OA:\n", t[["K", "supervised wrapper (LDA)_sd", "unsup U1 reconstruction R2_sd", "unsup U2 entropy-redundancy_sd"]].round(4).to_string(index=False))

# paired-ish significance: supervised vs U1 / U2 (independent runs, Mann-Whitney on run-level SVM OA)
pr = []
for K in KS:
    a = S_[(S_.K == K) & (S_.classifier == "SVM-RBF")].OA.values
    for nm, d in (("U1", U1), ("U2", U2)):
        b = d[(d.K == K) & (d.classifier == "SVM-RBF")].OA.values
        pr.append(dict(K=K, vs=nm, sup_mean=a.mean(), other_mean=b.mean(), diff=a.mean() - b.mean(), p=stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))
pr = pd.DataFrame(pr); pr.to_csv(OUT / "tbl_unsup_stats.csv", index=False); print("\n== supervised minus unsupervised (SVM OA) ==\n", pr.round(4).to_string(index=False))

# does the search algorithm matter under an unsupervised objective?
pa = []
for nm, d in (("U1", U1), ("U2", U2)):
    for K in KS:
        for a in ALG + ["RS"]:
            dd = (uns[(uns.fitness_kind == nm.lower()) & (uns.method == a)])
            m, s = oa(dd, K); pa.append(dict(fitness=nm, K=K, alg=a, OA=m, sd=s))
pa = pd.DataFrame(pa); pa.to_csv(OUT / "tbl_unsup_by_algorithm.csv", index=False)
print("\n== U1: SVM OA by algorithm ==\n", pa[pa.fitness == "U1"].pivot(index="alg", columns="K", values="OA").round(4).to_string())
print("\n== U2: SVM OA by algorithm ==\n", pa[pa.fitness == "U2"].pivot(index="alg", columns="K", values="OA").round(4).to_string())

# classifiers x fitness at K = 20, 30 ; also AA and kappa for SVM
cl = []
for K in (20, 30):
    for c in CLS:
        cl.append(dict(K=K, classifier=c, supervised=oa(S_, K, c)[0], U1=oa(U1, K, c)[0], U2=oa(U2, K, c)[0], uniform=oa(UNI, K, c)[0]))
cl = pd.DataFrame(cl); cl.to_csv(OUT / "tbl_unsup_classifiers.csv", index=False); print("\n== all classifiers (mean OA) ==\n", cl.round(4).to_string(index=False))
ak = []
for K in KS:
    for nm, d in (("supervised", S_), ("U1", U1), ("U2", U2)):
        ak.append(dict(K=K, fitness=nm, AA=oa(d, K, col="AA")[0], Kappa=oa(d, K, col="Kappa")[0], redundancy=oa(d, K, col="redundancy")[0]))
print("\n== AA / kappa / redundancy (SVM) ==\n", pd.DataFrame(ak).round(4).to_string(index=False))

# figure
def style(ax, title, xl, yl):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GREY)
    ax.tick_params(colors=INK2, labelsize=9); ax.set_xlabel(xl, color=INK2, fontsize=9.5); ax.set_ylabel(yl, color=INK2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
f, ax = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURF)
style(ax[0], "Label-free vs supervised search objective (SVM-RBF)", "bands kept (of 164 candidates)", "overall accuracy, held-out test pixels")
for col, c, mk, lab in (("supervised wrapper (LDA)", "#2a78d6", "o", "supervised wrapper (uses labels)"), ("unsup U1 reconstruction R2", "#eb6834", "s", "label-free U1: reconstruction R²"),
                        ("unsup U2 entropy-redundancy", "#1baf7a", "^", "label-free U2: entropy − redundancy")):
    ax[0].plot(KS, t[col], color=c, lw=2, marker=mk, ms=6, mec=SURF, mew=1.5, label=lab)
for col, ls, lab in (("greedy R2", "-", "greedy R² (label-free)"), ("uniform spacing", "--", "uniform spacing"), ("k-means (repo, pre-screened)", "-.", "k-means, pre-screened (repo)"), ("random subsets", ":", "random subsets")):
    ax[0].plot(KS, t[col], color=INK2 if col != "random subsets" else GREY, lw=1.5, ls=ls, label=lab)
ax[0].axhline(ref164.loc["SVM-RBF", "OA"], color=INK2, lw=1, ls="--", alpha=.6); ax[0].text(40.4, ref164.loc["SVM-RBF", "OA"], "all 164", fontsize=8.5, color=INK2, va="center")
ax[0].set_xlim(8, 46); ax[0].legend(fontsize=8, frameon=False, loc="lower right")
style(ax[1], "Same comparison, every classifier (K = 30)", "", "overall accuracy, held-out test pixels")
c30 = cl[cl.K == 30].set_index("classifier").loc[CLS]; x = np.arange(len(CLS)); w = 0.26
for i, (col, c) in enumerate((("supervised", "#2a78d6"), ("U1", "#eb6834"), ("U2", "#1baf7a"))):
    ax[1].bar(x + (i - 1) * w, c30[col], w * 0.92, color=c, label={"supervised": "supervised", "U1": "label-free U1", "U2": "label-free U2"}[col])
ax[1].set_xticks(x); ax[1].set_xticklabels(CLS); ax[1].set_ylim(0.5, 0.9); ax[1].legend(fontsize=8.5, frameon=False, loc="upper right", ncol=3)
f.tight_layout(); f.savefig(OUT / "fig6_unsupervised_vs_supervised.png", dpi=140, facecolor=SURF)
print("figure written")
