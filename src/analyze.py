"""Aggregate, test and plot.  Reads eval_long.csv + search pickles + baseline pickle."""
import sys, pickle, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import *

KS = [5, 10, 15, 20, 25, 30, 40, 50, 60]
ALG = ["SA", "GA", "GWO", "Jaya", "PSO", "DE"]          # fixed order = fixed colour slots
CLS = ["SVM-RBF", "KNN5", "LDA", "LogReg", "RF"]
SURF, INK, INK2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8a8985"
COL = dict(zip(ALG, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
CCOL = dict(zip(CLS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MK = dict(zip(ALG, ["o", "s", "^", "D", "v", "P"]))

df = pd.read_csv(OUT / "eval_long.csv")
A = df[(df.group == "algorithm") & (df.fitness_kind == "lda")]
NC = df[(df.group == "algorithm") & (df.fitness_kind == "ncc")]
BL = df[df.group == "baseline"]
REF = df[df.group == "reference"]
ref164 = REF[REF.method == "all164"].set_index("classifier")
ref200 = REF[REF.method == "all200"].set_index("classifier")


def style(ax, title, xl, yl):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GREY)
    ax.tick_params(colors=INK2, labelsize=9); ax.set_xlabel(xl, color=INK2, fontsize=9.5); ax.set_ylabel(yl, color=INK2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")


def fig_new(w, h, n=1):
    f, ax = plt.subplots(1, n, figsize=(w, h), facecolor=SURF)
    return f, ax


# ============================================================ tables
out = {}
# 1) SVM-RBF OA by method x K
rows = []
for name, sub in [(a, A[A.method == a]) for a in ALG] + [("RS", A[A.method == "RS"])] + [(b, BL[BL.method == b]) for b in ["random", "uniform", "entropy", "fisher", "sfs"]]:
    for K in KS:
        s = sub[(sub.K == K) & (sub.classifier == "SVM-RBF")]
        rows.append(dict(method=name, K=K, OA_mean=s.OA.mean(), OA_std=s.OA.std(), AA_mean=s.AA.mean(), Kappa_mean=s.Kappa.mean(), n=len(s)))
t1 = pd.DataFrame(rows); t1.to_csv(OUT / "tbl_svm_by_K.csv", index=False)
piv = t1.pivot(index="method", columns="K", values="OA_mean").round(4)
print("== SVM-RBF test OA, mean over 15 seeds ==\n", piv.loc[ALG + ["RS", "random", "uniform", "entropy", "fisher", "sfs"]].to_string())
print("refs SVM: all164 %.4f  all200 %.4f" % (ref164.loc["SVM-RBF", "OA"], ref200.loc["SVM-RBF", "OA"]))

# 2) every classifier x algorithm at each K
rows = []
for a in ALG + ["RS"]:
    for K in KS:
        for c in CLS:
            s = A[(A.method == a) & (A.K == K) & (A.classifier == c)]
            rows.append(dict(method=a, K=K, classifier=c, OA=s.OA.mean(), OA_std=s.OA.std(), AA=s.AA.mean(), Kappa=s.Kappa.mean(),
                             train_s=s.train_s.mean(), infer_s=s.infer_s.mean(), redundancy=s.redundancy.mean()))
t2 = pd.DataFrame(rows); t2.to_csv(OUT / "tbl_all_by_K_classifier.csv", index=False)

# 3) pooled-over-algorithms classifier curve (+ simple baselines)
pool = A[A.method.isin(ALG)].groupby(["K", "classifier"]).OA.mean().unstack()
print("\n== classifier OA vs K (pooled over 6 algorithms) ==\n", pool.round(4).to_string())
print("all164:", {c: round(ref164.loc[c, "OA"], 4) for c in CLS}); print("all200:", {c: round(ref200.loc[c, "OA"], 4) for c in CLS})
pool.to_csv(OUT / "tbl_classifier_pooled.csv")

# 4) average rank of algorithms (across K>=10 and classifiers)
rk = []
for K in KS:
    for c in CLS:
        s = t2[(t2.K == K) & (t2.classifier == c) & (t2.method.isin(ALG))].set_index("method").OA
        rk.append(s.rank(ascending=False).rename((K, c)))
rk = pd.concat(rk, axis=1)
avg_rank_all = rk.mean(axis=1); avg_rank_svm = rk[[k for k in rk.columns if k[1] == "SVM-RBF"]].mean(axis=1)
avg_rank_small = rk[[k for k in rk.columns if k[0] <= 30]].mean(axis=1)
rank_tbl = pd.DataFrame(dict(avg_rank_all=avg_rank_all, avg_rank_svm=avg_rank_svm, avg_rank_K_le_30=avg_rank_small)).round(2).sort_values("avg_rank_all")
rank_tbl.to_csv(OUT / "tbl_avg_rank.csv"); print("\n== average rank (1=best) ==\n", rank_tbl.to_string())

# 5) statistics: Kruskal-Wallis across algorithms; pairwise vs best at a few K (Holm)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
srows = []
for K in KS:
    g = [A[(A.method == a) & (A.K == K) & (A.classifier == "SVM-RBF")].OA.values for a in ALG]
    srows.append(dict(K=K, kruskal_p=stats.kruskal(*g).pvalue))
kw = pd.DataFrame(srows); kw.to_csv(OUT / "tbl_kruskal.csv", index=False); print("\n== Kruskal-Wallis p (SVM OA, 6 algorithms) ==\n", kw.round(4).to_string(index=False))
prow = []
for K in (10, 20, 30, 40):
    means = {a: A[(A.method == a) & (A.K == K) & (A.classifier == "SVM-RBF")].OA.mean() for a in ALG}
    best = max(means, key=means.get)
    ps = []; names = []
    for a in ALG + ["RS"]:
        if a == best: continue
        x = A[(A.method == best) & (A.K == K) & (A.classifier == "SVM-RBF")].OA.values
        y = A[(A.method == a) & (A.K == K) & (A.classifier == "SVM-RBF")].OA.values
        ps.append(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue); names.append(a)
    adj = holm(ps)
    for n_, p_, q_ in zip(names, ps, adj):
        prow.append(dict(K=K, best=best, versus=n_, mean_best=means[best], mean_other=A[(A.method == n_) & (A.K == K) & (A.classifier == "SVM-RBF")].OA.mean() if n_ != "RS" else A[(A.method == "RS") & (A.K == K) & (A.classifier == "SVM-RBF")].OA.mean(), p=p_, p_holm=q_))
pw = pd.DataFrame(prow); pw.to_csv(OUT / "tbl_pairwise.csv", index=False); print("\n== pairwise vs best (Mann-Whitney, Holm) ==\n", pw.round(4).to_string(index=False))

# 6) do metaheuristics beat simple baselines? fraction of runs above SFS / above best random
brows = []
for K in KS:
    sfs = BL[(BL.method == "sfs") & (BL.K == K) & (BL.classifier == "SVM-RBF")].OA.iloc[0]
    rnd = BL[(BL.method == "random") & (BL.K == K) & (BL.classifier == "SVM-RBF")].OA
    for a in ALG:
        s = A[(A.method == a) & (A.K == K) & (A.classifier == "SVM-RBF")].OA
        brows.append(dict(K=K, method=a, mean=s.mean(), vs_sfs=s.mean() - sfs, frac_runs_above_sfs=(s > sfs).mean(),
                          vs_random_mean=s.mean() - rnd.mean(), vs_random_best=s.mean() - rnd.max()))
bt = pd.DataFrame(brows); bt.to_csv(OUT / "tbl_vs_baselines.csv", index=False)
print("\n== algorithm minus simple baselines (SVM OA), selected K ==\n", bt[bt.K.isin([10, 20, 30, 40])].round(4).to_string(index=False))

# 7) stability: std of OA over seeds (above) + Jaccard of selected sets over seeds; optimism gap
cand = load_setup()["candidate_bands"]
jr, cur = [], {}
for K in KS:
    res = pickle.load(open(OUT / f"search_lda_K{K}.pkl", "rb"))
    for a in ALG + ["RS"]:
        sets = [set(r["idx"].tolist()) for r in res if r["alg"] == a]
        js = [len(x & y) / len(x | y) for x, y in itertools.combinations(sets, 2)]
        jr.append(dict(method=a, K=K, jaccard=float(np.mean(js))))
        if K == 20 and a in ALG:
            cur[a] = np.array([r["hist"] for r in res if r["alg"] == a])
    if K == 20:
        freq = np.zeros(200)
        for r in res:
            if r["alg"] in ALG:
                freq[cand[r["idx"].astype(int)]] += 1
        freq /= (len(ALG) * 15)
jac = pd.DataFrame(jr); jac.to_csv(OUT / "tbl_jaccard.csv", index=False)
print("\n== Jaccard stability (mean pairwise over 15 seeds) ==\n", jac.pivot(index="method", columns="K", values="jaccard").round(3).to_string())
pd.DataFrame(dict(orig_band=np.arange(200), freq_K20=freq)).to_csv(OUT / "tbl_band_frequency_K20.csv", index=False)

gap = A[A.classifier == "LDA"].assign(gap=lambda d: d.search_fitness - d.OA).groupby(["method", "K"]).gap.mean().unstack().loc[ALG + ["RS"]]
gap.round(4).to_csv(OUT / "tbl_optimism_gap.csv"); print("\n== selection optimism: train-validation fitness minus LDA test OA ==\n", gap.round(3).to_string())

# 8) search cost
cost = A[A.classifier == "SVM-RBF"].groupby("method").search_time_s.mean().round(2); print("\nmean search wall time per run (s, 2 procs on 2 cores):", cost.to_dict())
conv = []
for a in ALG:
    h = cur[a]; fin = h[:, -1]
    n99 = [int(np.argmax(r >= f - 0.002)) + 1 for r, f in zip(h, fin)]
    conv.append(dict(method=a, evals_to_within_0p002_of_final_K20=float(np.mean(n99)), final_fit_K20=float(fin.mean())))
conv = pd.DataFrame(conv); conv.to_csv(OUT / "tbl_convergence_K20.csv", index=False); print("\n", conv.round(4).to_string(index=False))

# 9) fitness ablation at K=20: NCC-wrapper vs LDA-wrapper
ab = []
for a in ALG + ["RS"]:
    for c in ["SVM-RBF", "LDA", "LogReg"]:
        x = A[(A.method == a) & (A.K == 20) & (A.classifier == c)].OA; y = NC[(NC.method == a) & (NC.classifier == c)].OA
        ab.append(dict(method=a, classifier=c, LDA_fitness=x.mean(), NCC_fitness=y.mean(), diff=x.mean() - y.mean()))
ab = pd.DataFrame(ab); ab.to_csv(OUT / "tbl_fitness_ablation_K20.csv", index=False)
print("\n== K=20: LDA-wrapper vs NCC-wrapper selected subsets ==\n", ab.round(4).to_string(index=False))

# 10) redundancy
red = A[A.classifier == "SVM-RBF"].groupby(["method", "K"]).redundancy.mean().unstack().loc[ALG + ["RS"]]
redb = BL[BL.classifier == "SVM-RBF"].groupby(["method", "K"]).redundancy.mean().unstack()
red.round(3).to_csv(OUT / "tbl_redundancy.csv"); print("\n== redundancy (mean |r|), algorithms ==\n", red.round(3).to_string(), "\n baselines\n", redb.round(3).to_string())

# ============================================================ figures
lastK = max(KS)
# fig 1: accuracy vs number of bands
f, ax = fig_new(13, 5, 2)
a0 = ax[0]
K10 = [k for k in KS if k >= 10]
style(a0, "Test accuracy vs number of bands kept (SVM-RBF, mean of 15 seeds)", "bands kept (of 164 candidates)", "overall accuracy, held-out test pixels")
for a in ALG:
    s_ = t1[t1.method == a].set_index("K")
    a0.plot(K10, s_.loc[K10, "OA_mean"], color=COL[a], lw=2, marker=MK[a], ms=6, mec=SURF, mew=1.5, label=a)
s_ = t1[t1.method == "RS"].set_index("K"); a0.plot(K10, s_.loc[K10, "OA_mean"], color=GREY, lw=2, ls=":", label="random search (same budget)")
a0.axhline(ref164.loc["SVM-RBF", "OA"], color=INK2, lw=1.2, ls="--"); a0.text(61, ref164.loc["SVM-RBF", "OA"], "all 164\nbands", color=INK2, fontsize=8.5, va="center")
a0.axhline(ref200.loc["SVM-RBF", "OA"], color=GREY, lw=1.0, ls="--"); a0.text(61, ref200.loc["SVM-RBF", "OA"], "all 200\nraw bands", color=GREY, fontsize=8.5, va="center")
a0.set_xlim(8, 72); a0.set_ylim(0.76, 0.85)
a0.legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
a1 = ax[1]
style(a1, "Do the algorithms beat simple baselines? (SVM-RBF)", "bands kept (of 164 candidates)", "overall accuracy, held-out test pixels")
best_by_K = t1[t1.method.isin(ALG)].groupby("K").OA_mean.max()
a1.plot(KS, best_by_K.loc[KS], color=COL["SA"], lw=2.4, marker="o", ms=6, mec=SURF, mew=1.5, label="best search algorithm (per K)")
for b, ls, lab in [("sfs", "-", "greedy forward selection"), ("fisher", "--", "top Fisher score"), ("entropy", "-.", "top entropy"), ("uniform", ":", "uniform spacing"), ("random", (0, (1, 3)), "random subsets (mean)")]:
    s = t1[t1.method == b].set_index("K"); a1.plot(KS, s.loc[KS, "OA_mean"], color=INK2 if b != "random" else GREY, lw=1.6, ls=ls, label=lab)
a1.axhline(ref164.loc["SVM-RBF", "OA"], color=INK2, lw=1.0, ls="--", alpha=0.6)
a1.legend(fontsize=8, frameon=False, loc="lower right")
f.tight_layout(); f.savefig(OUT / "fig1_accuracy_vs_bands.png", dpi=140, facecolor=SURF); plt.close(f)

# fig 2: classifier panel vs K (pooled over algorithms)
f, ax = fig_new(7.5, 5)
style(ax, "Classifier accuracy vs bands kept (mean of 6 algorithms)", "bands kept (of 164 candidates)", "overall accuracy, held-out test pixels")
for c in CLS:
    ax.plot(KS, pool.loc[KS, c], color=CCOL[c], lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=c)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
f.tight_layout(); f.savefig(OUT / "fig2_classifiers_vs_bands.png", dpi=140, facecolor=SURF); plt.close(f)

# fig 3: convergence at K=20
f, ax = fig_new(7.5, 5)
style(ax, "Convergence at K = 20 (mean of 15 seeds)", "fitness evaluations", "best-so-far validation accuracy (LDA wrapper)")
for a in ALG:
    ax.plot(np.arange(1, 3001), cur[a].mean(0), color=COL[a], lw=2, label=a)
rs = np.array([r["hist"] for r in pickle.load(open(OUT / "search_lda_K20.pkl", "rb")) if r["alg"] == "RS"]).mean(0)
ax.plot(np.arange(1, 3001), rs, color=GREY, lw=2, ls=":", label="random search")
ax.set_xscale("log"); ax.set_ylim(0.62, 0.76); ax.legend(fontsize=8.5, frameon=False, loc="lower right")
f.tight_layout(); f.savefig(OUT / "fig3_convergence_K20.png", dpi=140, facecolor=SURF); plt.close(f)

# fig 4: stability at K=20
f, ax = fig_new(11, 4.2, 2)
s20 = A[(A.K == 20) & (A.classifier == "SVM-RBF")].groupby("method").OA.std().loc[ALG]
style(ax[0], "Spread of test accuracy across 15 seeds (K = 20)", "standard deviation of SVM-RBF accuracy", "")
ax[0].barh(ALG[::-1], s20.loc[ALG[::-1]], color=[COL[a] for a in ALG[::-1]], height=0.5)
j20 = jac[jac.K == 20].set_index("method").jaccard.loc[ALG]
style(ax[1], "Overlap of selected bands between seeds (K = 20)", "mean pairwise Jaccard index", "")
ax[1].barh(ALG[::-1], j20.loc[ALG[::-1]], color=[COL[a] for a in ALG[::-1]], height=0.5)
for a_, v in zip(ALG[::-1], j20.loc[ALG[::-1]]): ax[1].text(v + 0.003, a_, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
for a_, v in zip(ALG[::-1], s20.loc[ALG[::-1]]): ax[0].text(v + 0.0003, a_, f"{v:.4f}", va="center", fontsize=8.5, color=INK2)
f.tight_layout(); f.savefig(OUT / "fig4_stability_K20.png", dpi=140, facecolor=SURF); plt.close(f)

# fig 5: where on the spectrum do the algorithms look?  (K = 20)
f, ax = fig_new(11, 3.8)
style(ax, "How often each band is chosen at K = 20 (6 algorithms x 15 seeds)", "original band index (0-199 of the 200-band corrected cube)", "selection frequency")
removed = np.setdiff1d(np.arange(200), cand)
for r_ in removed: ax.axvspan(r_ - 0.5, r_ + 0.5, color=GRID, lw=0)
ax.bar(np.arange(200), freq, color=COL["SA"], width=0.8)
ax.text(2, ax.get_ylim()[1] * 0.92, "grey = removed by entropy pre-screening", color=INK2, fontsize=8.5)
f.tight_layout(); f.savefig(OUT / "fig5_band_frequency_K20.png", dpi=140, facecolor=SURF); plt.close(f)
print("\nfigures written")
