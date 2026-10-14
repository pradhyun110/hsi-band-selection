"""
15-20 band study.  Goal: keep only 15-20 of 200 bands and still be within about 1-2 points of the
accuracy of all 200 bands -- and check that this is not luck of one train/test split.

For each of several outer splits (stratified 20 % train / 80 % test, different seeds) and each K = 15..20:
  GA-LDA        genetic algorithm, wrapper fitness = LDA accuracy on held-out TRAIN pixels (5 starts)
                 best-of-5 by train-validation, and the mean of the 5 runs
  SFS-LDA       greedy forward selection with the same fitness
  GA+SVM polish the GA-LDA best set refined by swap hill-climbing with an SVM-RBF wrapper fitness
                 (still train pixels only)
  label-free    greedy R^2, k-means band clustering, evenly spaced (references)
  references    all 200 raw bands, all 164 pre-screened bands
Every subset is graded on the test pixels with SVM-RBF (default C=10) and with an SVM whose C/gamma
were tuned by 3-fold CV on TRAIN pixels only.

Run:  python src/small_k.py            (about 15-20 min on 2 cores)   --test  for a 1-minute smoke test
"""
import sys, json, time, warnings, argparse
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_setup import stratified_split, TEST_FRACTION
from common import LDAFitness
from optimizers import run_ga
from unsupervised_selectors import sel_greedy_r2, sel_kmeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score, confusion_matrix
warnings.filterwarnings("ignore")

SPLITS = [42, 1, 2, 3, 4, 5]
KS = [15, 16, 17, 18, 19, 20]
POLISH_K = {15, 18, 20}
GA_STARTS, GA_BUDGET, GA_POP, POLISH_BUDGET = 5, 3000, 30, 300
OUT = ROOT / "results" / "small_k"; OUT.mkdir(parents=True, exist_ok=True)


def make_setup(seed):
    cube = np.load(ROOT / "data" / "indianpinearray.npy").astype(np.float64)
    gt = np.load(ROOT / "data" / "IPgt.npy"); cand = np.load(ROOT / "results" / "selected_band_indices.npy")
    m = gt.reshape(-1) != 0
    X200 = cube.reshape(-1, cube.shape[2])[m]; y = gt.reshape(-1)[m]
    tr, te = stratified_split(y, TEST_FRACTION, seed)
    mu, sd = X200[tr].mean(0), X200[tr].std(0); sd[sd == 0] = 1
    Z200 = (X200 - mu) / sd
    Xr = X200[:, cand]; mu1, sd1 = Xr[tr].mean(0), Xr[tr].std(0); sd1[sd1 == 0] = 1
    Z = (Xr - mu1) / sd1
    return dict(X_train=Z[tr], y_train=y[tr], X_test=Z[te], y_test=y[te], classes=np.unique(y),
                candidate_bands=cand, Xraw_train=Xr[tr], Z200_train=Z200[tr], Z200_test=Z200[te])


def grade(Xtr, ytr, Xte, yte, tune=True):
    clf = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr, ytr); p = clf.predict(Xte)
    cm = confusion_matrix(yte, p)
    r = dict(OA=float((p == yte).mean()), AA=float(np.nanmean(np.diag(cm) / cm.sum(1))), Kappa=float(cohen_kappa_score(yte, p)))
    if tune:
        g0 = 1.0 / (Xtr.shape[1] * Xtr.var())
        gs = GridSearchCV(SVC(kernel="rbf"), {"C": [1, 10, 100, 1000], "gamma": [g0 / 4, g0, g0 * 4]}, cv=3, n_jobs=1).fit(Xtr, ytr)
        r["OA_tuned"] = float((gs.predict(Xte) == yte).mean())
    return r


def sfs(fit, n, kmax):
    chosen, order = np.zeros(n, bool), []
    for _ in range(kmax):
        best, bj = -1, -1
        for j in np.flatnonzero(~chosen):
            chosen[j] = True; f = fit(chosen); chosen[j] = False
            if f > best: best, bj = f, j
        chosen[bj] = True; order.append(bj)
    return order


def polish(fit, mask0, budget, rng):
    cur = mask0.copy(); f = fit(cur)
    for _ in range(budget - 1):
        sel, uns = np.flatnonzero(cur), np.flatnonzero(~cur)
        c = cur.copy(); c[rng.choice(sel)] = False; c[rng.choice(uns)] = True
        fc = fit(c)
        if fc > f: cur, f = c, fc
    return cur, f


class SVMFitness:
    def __init__(self, ldafit): self.folds = ldafit.folds
    def __call__(self, mask):
        cols = np.flatnonzero(mask); a = 0.0
        for Xf, yf, Xv, yv in self.folds:
            a += (SVC(kernel="rbf", C=10, gamma="scale").fit(Xf[:, cols], yf).predict(Xv[:, cols]) == yv).mean()
        return a / len(self.folds)


def run_split(seed, ks, polish_k, ga_starts, budget):
    t0 = time.time(); S = make_setup(seed); fit = LDAFitness(S); svmfit = SVMFitness(fit)
    Xtr, ytr, Xte, yte = S["X_train"], S["y_train"], S["X_test"], S["y_test"]
    rows, subsets = [], {}
    def add(method, K, idx, tune=True, extra=None):
        idx = np.asarray(sorted(int(i) for i in idx)); assert len(set(idx)) == K, (method, K)
        r = dict(split=seed, method=method, K=K, **grade(Xtr[:, idx], ytr, Xte[:, idx], yte, tune)); r.update(extra or {})
        rows.append(r); subsets[f"{method}|K={K}"] = [int(S["candidate_bands"][i]) for i in idx]
    ref = grade(S["Z200_train"], ytr, S["Z200_test"], yte); ref164 = grade(Xtr, ytr, Xte, yte)
    rows.append(dict(split=seed, method="All 200 bands", K=200, **ref)); rows.append(dict(split=seed, method="All 164 pre-screened", K=164, **ref164))
    order = sfs(fit, 164, max(ks))
    for K in ks:
        runs = [run_ga(fit, 164, K, budget, GA_POP, 1000 * K + 97 * seed + s) for s in range(ga_starts)]
        best = max(runs, key=lambda r: r["fitness"])
        for i, r in enumerate(runs):                                    # every start, graded (to show the honest average)
            idx = np.flatnonzero(r["mask"]); g = grade(Xtr[:, idx], ytr, Xte[:, idx], yte, False)
            rows.append(dict(split=seed, method="GA-LDA (each start)", K=K, start=i, val_fitness=r["fitness"], **g))
        add("GA-LDA (best of starts)", K, np.flatnonzero(best["mask"]), extra=dict(val_fitness=best["fitness"]))
        add("Greedy forward (LDA)", K, order[:K])
        add("Greedy R2 (no labels)", K, sel_greedy_r2(S["Xraw_train"], Xtr, K))
        add("K-means (no labels)", K, sel_kmeans(S["Xraw_train"], Xtr, K))
        add("Evenly spaced", K, np.unique(np.round(np.linspace(0, 163, K)).astype(int)))
        if K in polish_k:
            m, f = polish(svmfit, best["mask"], POLISH_BUDGET, np.random.default_rng(seed * 100 + K))
            add("GA-LDA + SVM polish", K, np.flatnonzero(m), extra=dict(val_fitness=f))
        print(f"  split {seed:2d} K={K} done  ({time.time()-t0:.0f}s)", flush=True)
    return rows, subsets


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); a = ap.parse_args()
    splits, ks, pk, starts, bud = ([42], [15], {15}, 1, 600) if a.test else (SPLITS, KS, POLISH_K, GA_STARTS, GA_BUDGET)
    t = time.time()
    out = Parallel(n_jobs=2)(delayed(run_split)(s, ks, pk, starts, bud) for s in splits)
    df = pd.DataFrame([r for rows, _ in out for r in rows]); sub = {}
    for _, s in out: sub.update({f"split{r}": 0 for r in []})
    df.to_csv(OUT / ("results_long_TEST.csv" if a.test else "results_long.csv"), index=False)
    if not a.test:
        json.dump({f"split={seed}": s for seed, (_, s) in zip(splits, out)}, open(OUT / "subsets_original_band_numbers.json", "w"), indent=1)
    print(f"done in {time.time()-t:.0f}s, {len(df)} rows")
