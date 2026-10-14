"""
Score every final subset (algorithms, baselines, all-band references) on the held-out TEST pixels
with five classifiers.  Output: eval_long.csv  (one row per subset x classifier).
Timings here are measured while 2 workers share 2 cores -> only indicative; clean timings are taken
separately (timing_clean.py).
"""
import sys, pickle, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix

warnings.filterwarnings("ignore")
PANEL = {
    "SVM-RBF": lambda: SVC(kernel="rbf", C=10, gamma="scale"),
    "KNN5": lambda: KNeighborsClassifier(n_neighbors=5),
    "LDA": lambda: LDA(),
    "LogReg": lambda: LogisticRegression(C=1.0, max_iter=500),
    "RF": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1),
}
_c = {}


def data(which):
    if which not in _c:
        _c[which] = load_setup() if which == "c164" else load_all200()
    return _c[which]


def score(y, p):
    cm = confusion_matrix(y, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        aa = np.nanmean(np.diag(cm) / cm.sum(1))
    return float((p == y).mean()), float(aa), float(cohen_kappa_score(y, p))


def eval_one(meta, idx, which="c164"):
    S = data(which); idx = np.asarray(idx, dtype=int)
    Xtr, Xte, ytr, yte = S["X_train"][:, idx], S["X_test"][:, idx], S["y_train"], S["y_test"]
    rows = []
    red = redundancy(S, idx)
    for name, mk in PANEL.items():
        m = mk(); t = time.perf_counter(); m.fit(Xtr, ytr); tr = time.perf_counter() - t
        t = time.perf_counter(); p = m.predict(Xte); ti = time.perf_counter() - t
        oa, aa, kap = score(yte, p)
        rows.append(dict(meta, classifier=name, OA=oa, AA=aa, Kappa=kap, train_s=tr, infer_s=ti, redundancy=red))
    return rows


if __name__ == "__main__":
    jobs = []
    for f in sorted(OUT.glob("search_*_K*.pkl")):
        for r in pickle.load(open(f, "rb")):
            fitkind = r["kind"]
            jobs.append((dict(group="algorithm", method=r["alg"], fitness_kind=fitkind, K=r["K"], seed=r["seed"],
                              search_fitness=r["fitness"], search_time_s=r["time_s"]), r["idx"], "c164"))
    B = pickle.load(open(OUT / "baseline_subsets.pkl", "rb"))
    for s in B["subsets"]:
        jobs.append((dict(group="baseline", method=s["method"], fitness_kind="-", K=s["K"], seed=s["seed"],
                          search_fitness=np.nan, search_time_s=np.nan), s["idx"], "c164"))
    jobs.append((dict(group="reference", method="all164", fitness_kind="-", K=164, seed=0, search_fitness=np.nan, search_time_s=np.nan),
                 np.arange(164), "c164"))
    jobs.append((dict(group="reference", method="all200", fitness_kind="-", K=200, seed=0, search_fitness=np.nan, search_time_s=np.nan),
                 np.arange(200), "c200"))
    print(len(jobs), "subsets x", len(PANEL), "classifiers", flush=True)
    t = time.perf_counter()
    out = Parallel(n_jobs=2, batch_size=4)(delayed(eval_one)(m, i, w) for m, i, w in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(OUT / "eval_long.csv", index=False)
    print(f"done in {time.perf_counter()-t:.0f}s, {len(df)} rows", flush=True)
