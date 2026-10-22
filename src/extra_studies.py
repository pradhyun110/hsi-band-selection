"""
Three extra studies asked for after the main comparison.  Each writes one CSV into results/final/.

  python src/extra_studies.py --split       learning-set size: 10 / 20 / 40 / 50 / 80 % of the labelled pixels
  python src/extra_studies.py --prescreen   every method on all 200 bands vs on the 164 pre-screened bands
  python src/extra_studies.py --extra-k     more bands (K = 40 ... 100) so the flat part of the curve is visible

Everything else is identical to final_comparison.py: same 6 seeds, same SVM grading, same fitness, same budget
(5 restarts x 3,000 fitness calls for GA / Jaya / GWO / Hybrid).
"""
import sys, json, time, warnings, argparse
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_setup import make_setup
from common import LDAFitness
from optimizers import run_ga, run_jaya, run_gwo
from fisher_cholesky import fisher_select
from unsupervised_selectors import sel_entropy, sel_variance
from mfe_cfsp import mfe_from_raw
from metrics import classification_metrics
from sklearn.svm import SVC
import final_comparison as F
warnings.filterwarnings("ignore")

OUT = ROOT / "results" / "final"
CUBE = F.CUBE; CAND164 = F.CAND; ALL200 = np.arange(200)
SPLITS, STARTS, BUDGET, POP = F.SPLITS, F.STARTS, F.BUDGET, F.POP


def grade(S, bands):
    b = np.asarray(sorted(int(x) for x in bands))
    clf = SVC(kernel="rbf", C=10, gamma="scale").fit(S["Z_train"][:, b], S["y_train"])
    r = classification_metrics(S["y_test"], clf.predict(S["Z_test"][:, b]), S["classes"])
    r["train_OA"] = float((clf.predict(S["Z_train"][:, b]) == S["y_train"]).mean())
    return {k: r[k] for k in ("OA", "AA", "Kappa", "train_OA")}


def best_of_starts(algo, fit, n, K, seed):
    runs = [algo(fit, n, K, BUDGET, POP, 1000 * K + 97 * seed + s) for s in range(STARTS)]
    return np.flatnonzero(max(runs, key=lambda r: r["fitness"])["mask"])


class Sub:
    def __init__(self, fit, short, n): self.fit, self.short, self.n = fit, np.asarray(short), n
    def __call__(self, m):
        full = np.zeros(self.n, bool); full[self.short[m]] = True; return self.fit(full)


def select_all(S, cand, K, which):
    """name -> (function returning ORIGINAL band numbers).  cand = candidate band numbers (164 or 200)."""
    n = len(cand); Xtr, ytr, fit = S["X_train"], S["y_train"], None
    def get_fit():
        return LDAFitness(S)
    d = {"Fisher-Cholesky": lambda: cand[fisher_select(list(range(n)), Xtr, ytr, K)]}
    if "GA" in which: d["GA"] = lambda: cand[best_of_starts(run_ga, get_fit(), n, K, S["seed"])]
    if "Hybrid" in which:
        def hyb():
            f = get_fit(); short = np.array(fisher_select(list(range(n)), Xtr, ytr, 2 * K))
            return cand[short[best_of_starts(run_ga, Sub(f, short, n), len(short), K, S["seed"])]]
        d["Hybrid (GA + Fisher shortlist)"] = hyb
    if "Jaya" in which: d["Jaya"] = lambda: cand[best_of_starts(run_jaya, get_fit(), n, K, S["seed"])]
    if "GWO" in which: d["Grey wolf (GWO)"] = lambda: cand[best_of_starts(run_gwo, get_fit(), n, K, S["seed"])]
    if "Base" in which: d["Base paper (MFE-CFSP)"] = lambda: cand[mfe_from_raw(CUBE[:, :, cand], K)]
    if "Entropy" in which: d["Entropy"] = lambda: cand[sel_entropy(S["Xraw_train"][:, cand], Xtr, K)]
    if "Variance" in which: d["Variance"] = lambda: cand[sel_variance(S["Xraw_train"][:, cand], Xtr, K)]
    return d


# ---------------------------------------------------------------- 1. learning-set size
FRACS = [0.10, 0.20, 0.40, 0.50, 0.80]


def split_job(seed, frac, K=18):
    S = make_setup(seed, test_fraction=1 - frac); rows = []
    nlearn = len(S["y_train"]); smallest = int(np.bincount(S["y_train"])[1:].min())
    def add(name, bands, st=0.0, Kx=None):
        rows.append(dict(split=seed, learn_frac=frac, n_learn=nlearn, n_exam=len(S["y_test"]), smallest_class_learn=smallest,
                         method=name, K=Kx or len(bands), selection_time_s=st, **grade(S, bands)))
    add("All 200 bands", ALL200); add("All 164 bands", CAND164)
    for name, f in select_all(S, CAND164, K, {"GA"}).items():
        t = time.perf_counter(); b = f(); add(name, b, time.perf_counter() - t)
    return rows


# ---------------------------------------------------------------- 2. pre-screening
def prescreen_job(seed, K):
    S = make_setup(seed, cand=ALL200); rows = []                       # candidates = all 200 bands
    for name, f in select_all(S, ALL200, K, {"GA", "Hybrid", "Jaya", "GWO", "Base", "Entropy", "Variance"}).items():
        t = time.perf_counter(); b = np.array(sorted(int(x) for x in f())); st = time.perf_counter() - t
        assert len(set(b)) == K
        rows.append(dict(split=seed, method=name, K=K, candidates=200, selection_time_s=st, **grade(S, b)))
    return rows


# ---------------------------------------------------------------- 3. more bands
def extra_k_job(seed, ks):
    rows, sets, _ = F.run_split(seed, ks, STARTS, BUDGET)
    return [r for r in rows if not r["method"].startswith("All ")], sets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true"); ap.add_argument("--prescreen", action="store_true")
    ap.add_argument("--extra-k", action="store_true"); ap.add_argument("--test", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    seeds = [42] if a.test else SPLITS
    if a.split:
        fr = [0.1, 0.8] if a.test else FRACS
        rows = Parallel(n_jobs=2)(delayed(split_job)(s, f) for f in fr for s in seeds)
        pd.DataFrame([r for rs in rows for r in rs]).to_csv(OUT / f"split_ratio{'_TEST' if a.test else ''}.csv", index=False)
    if a.prescreen:
        ks = [20] if a.test else [18, 20]
        rows = Parallel(n_jobs=2)(delayed(prescreen_job)(s, K) for K in ks for s in seeds)
        pd.DataFrame([r for rs in rows for r in rs]).to_csv(OUT / f"prescreen_200_vs_164{'_TEST' if a.test else ''}.csv", index=False)
    if a.extra_k:
        ks = [40] if a.test else [40, 50, 60, 80, 100]
        out = Parallel(n_jobs=2)(delayed(extra_k_job)(s, ks) for s in seeds)
        tag = "_TEST" if a.test else ""
        pd.DataFrame([r for rs, _ in out for r in rs]).to_csv(OUT / f"results_long_extraK{tag}.csv", index=False)
        json.dump({f"split={s}": st for s, (_, st) in zip(seeds, out)}, open(OUT / f"band_sets_extraK{tag}.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")
