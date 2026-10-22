"""
Why Fisher?  Four criteria on the SAME recursive-Cholesky engine, same 6 splits and grading as the final comparison.
Writes results/final/cholesky_criteria.csv.   Run: python src/compare_cholesky_criteria.py   (about 3 min)
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_setup import make_setup
from cholesky_criteria import select
from metrics import classification_metrics
from sklearn.svm import SVC
warnings.filterwarnings("ignore")
SPLITS, KS = [42, 1, 2, 3, 4, 5], [5, 10, 15, 18, 20, 25, 30]
NAMES = {"fisher": "Fisher (chosen)", "pairwise_mean": "Mean pairwise class distance",
         "pairwise_min": "Worst-pair class distance", "volume": "Maximum volume (label-free)"}
CAND = np.load(ROOT / "results" / "prescreen" / "selected_band_indices.npy")


def run(seed):
    S = make_setup(seed); rows = []
    for K in KS:
        for c, name in NAMES.items():
            t = time.perf_counter(); pos = select(c, S["X_train"], S["y_train"], K); st = time.perf_counter() - t
            b = np.sort(CAND[pos])
            p = SVC(kernel="rbf", C=10, gamma="scale").fit(S["Z_train"][:, b], S["y_train"]).predict(S["Z_test"][:, b])
            r = classification_metrics(S["y_test"], p, S["classes"]); r.pop("per_class_sensitivity")
            rows.append(dict(split=seed, criterion=name, K=K, selection_time_s=st, **r))
    return rows


if __name__ == "__main__":
    df = pd.DataFrame([r for rows in Parallel(n_jobs=2)(delayed(run)(s) for s in SPLITS) for r in rows])
    df.to_csv(ROOT / "results" / "final" / "cholesky_criteria.csv", index=False)
    print(df[df.K.isin([15, 18, 20])].groupby(["K", "criterion"])[["OA", "AA", "Kappa", "selection_time_s"]].mean().round(4).to_string())
