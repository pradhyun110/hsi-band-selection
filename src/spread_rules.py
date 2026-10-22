"""
Should the chosen bands be forced to spread across the spectrum?  Fisher-Cholesky with extra rules on which
band may be added next, same 6 splits and SVM grading as the final comparison.
  Fisher (no rule)          the chosen method
  Min gap 3 / 6 bands       no two chosen bands closer than 3 (or 6) band numbers
  Max per 40-band region    at most ceil(K/5)+1 bands in each block of 40 bands
If a rule cannot be met (pre-screening leaves holes in the spectrum), it is relaxed for that step.
Writes results/final/spread_rules.csv.   Run: python src/spread_rules.py   (about 3 min)
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from data_setup import make_setup
from metrics import classification_metrics
from sklearn.svm import SVC
CAND = np.load(__import__("pathlib").Path(__file__).resolve().parent.parent / "results" / "prescreen" / "selected_band_indices.npy")

def fisher_rule(X, y, K, allowed, ridge=0.01):
    """Fisher-Cholesky (direct J, same result) with an extra rule on which bands may be added."""
    cl = np.unique(y); m = X.mean(0); n = np.array([(y == c).sum() for c in cl])
    D = np.vstack([X[y == c].mean(0) - m for c in cl]); W = X - (D + m)[np.searchsorted(cl, y)]
    Sw = W.T @ W / len(y); Sw += ridge * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    J = lambda s: sum(n[i] * D[i, s] @ np.linalg.solve(Sw[np.ix_(s, s)], D[i, s]) for i in range(len(cl)))
    ch = []
    while len(ch) < K:
        ok = [b for b in range(len(Sw)) if b not in ch and allowed(ch, b)]
        if not ok: ok = [b for b in range(len(Sw)) if b not in ch]      # rule cannot be met: relax it for this step
        ch.append(max(ok, key=lambda b: J(ch + [b])))
    return CAND[ch]

def gap(g): return lambda ch, b: all(abs(int(CAND[b]) - int(CAND[c])) > g for c in ch)
def cap(K): 
    lim = int(np.ceil(K / 5)) + 1
    return lambda ch, b: sum(int(CAND[c]) // 40 == int(CAND[b]) // 40 for c in ch) < lim

def run(seed):
    S = make_setup(seed); rows = []
    for K in [10, 15, 18, 20]:
        for name, rule in [("Fisher (no rule)", lambda ch, b: True), ("Min gap 3 bands", gap(3)),
                           ("Min gap 6 bands", gap(6)), (f"Max per 40-band region", cap(K))]:
            b = np.sort(fisher_rule(S["X_train"], S["y_train"], K, rule))
            p = SVC(C=10, gamma="scale").fit(S["Z_train"][:, b], S["y_train"]).predict(S["Z_test"][:, b])
            r = classification_metrics(S["y_test"], p, S["classes"])
            rows.append(dict(split=seed, K=K, rule=name, OA=r["OA"], AA=r["AA"], gap=np.diff(np.r_[0, b, 199]).max(),
                             low=(b < 40).sum()))
    return rows
df = pd.DataFrame([r for rows in Parallel(n_jobs=2)(delayed(run)(s) for s in [42, 1, 2, 3, 4, 5]) for r in rows])
df.to_csv(__import__("pathlib").Path(__file__).resolve().parent.parent / "results" / "final" / "spread_rules.csv", index=False)
g = df.groupby(["K", "rule"])
print((g[["OA", "AA"]].mean() * 100).round(1).join(g[["gap", "low"]].mean().round(1)).to_string())
