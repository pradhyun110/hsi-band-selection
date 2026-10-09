"""
Pilot comparison of 3 candidate fitness functions. Search uses TRAIN data only
(inner 50/50 split of the 2051 training pixels); the 8198 test pixels are touched only
for the final, independent evaluation of the selected subset.
Same candidate bands (164), same algorithms/budget (pop25 x iter50 = 1250 evals), same seeds.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_setup import load_fixed_setup, stratified_split
from algorithms import run_sa, run_jaya
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

OUT = __import__("pathlib").Path(__file__).resolve().parent.parent / "results" / "legacy"
S = load_fixed_setup(); N = S["n_candidates"]
Xtr, ytr, Xte, yte, classes = S["X_train"], S["y_train"], S["X_test"], S["y_test"], S["classes"]

# inner split of TRAIN only (fixed)
fi, vi = stratified_split(ytr, 0.5, 7)
Xf, yf, Xv, yv = Xtr[fi], ytr[fi], Xtr[vi], ytr[vi]
cent = np.vstack([Xf[yf == c].mean(0) for c in classes])

# precomputed per-band / pairwise quantities (train only)
R = np.abs(np.corrcoef(Xtr, rowvar=False)); np.fill_diagonal(R, 0)
mu = Xtr.mean(0)
sb = sum((ytr == c).sum() * (Xtr[ytr == c].mean(0) - mu) ** 2 for c in classes)
sw = sum(((Xtr[ytr == c] - Xtr[ytr == c].mean(0)) ** 2).sum(0) for c in classes)
fisher = sb / sw; fisher_n = fisher / fisher.max()          # in [0,1]

def acc_val(cols):
    C = cent[:, cols]; X = Xv[:, cols]
    d = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :]
    return float((classes[d.argmin(1)] == yv).mean())

def redund(cols):
    k = len(cols)
    return 0.0 if k < 2 else float(R[np.ix_(cols, cols)].sum() / (k * (k - 1)))

LAM, ALPHA, BETA = 0.01, 0.10, 0.05
def F1(m):
    c = np.where(m)[0]; a = acc_val(c); return a - LAM * len(c) / N, a
def F2(m):
    c = np.where(m)[0]; a = acc_val(c); return a - ALPHA * redund(c) - BETA * len(c) / N, a
def F3(m):
    c = np.where(m)[0]; f = fisher_n[c].mean() - redund(c); return f, f
FITS = {"F1 acc+size": F1, "F2 acc+red+size": F2, "F3 filter (relev-red)": F3}

# eval cost
rng = np.random.default_rng(0); masks = [rng.random(N) < 0.4 for _ in range(300)]
cost = {}
for n, f in FITS.items():
    t = time.perf_counter(); [f(m) for m in masks]; cost[n] = (time.perf_counter() - t) / 300 * 1e3
print("ms/eval", {k: round(v, 3) for k, v in cost.items()})

rows = []
for fname, f in FITS.items():
    for aname, alg in [("SA", run_sa), ("Jaya", run_jaya)]:
        for seed in range(5):
            res = alg(f, N, pop_size=25, max_iter=50, seed=seed)
            c = np.where(res["best_mask"])[0]
            Xa, Xb = Xtr[:, c], Xte[:, c]
            svm = SVC(kernel="rbf", C=10, gamma="scale").fit(Xa, ytr).score(Xb, yte)
            lr = LogisticRegression(max_iter=2000).fit(Xa, ytr).score(Xb, yte)
            knn = KNeighborsClassifier(5).fit(Xa, ytr).score(Xb, yte)
            rows.append(dict(fitness=fname, algo=aname, seed=seed, n_evals=res["n_evals"], k=len(c),
                             redundancy=redund(c), svm=svm, logreg=lr, knn=knn, search_fitness=res["best_fitness"]))
        print(fname, aname, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "fitness_comparison_raw.csv", index=False)
g = df.groupby("fitness").agg(k_mean=("k", "mean"), k_std=("k", "std"), red=("redundancy", "mean"),
    svm=("svm", "mean"), svm_std=("svm", "std"), logreg=("logreg", "mean"), knn=("knn", "mean"), evals=("n_evals", "mean"))
g["ms_per_eval"] = pd.Series(cost); g = g.round(4); g.to_csv(OUT / "fitness_comparison_summary.csv")
pd.set_option("display.width", 200); print(g.to_string())
print(df.groupby(["fitness", "algo"])[["k", "svm", "logreg"]].mean().round(4).to_string())
# reference: all 164 bands
for n, mk in [("all164", lambda: SVC(kernel='rbf', C=10, gamma='scale')), ("lr164", lambda: LogisticRegression(max_iter=2000))]:
    print(n, round(mk().fit(Xtr, ytr).score(Xte, yte), 4))
