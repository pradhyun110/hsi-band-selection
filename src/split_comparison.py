"""
Small comparison of selected methods across learning/exam splits 20/80, 40/60, 50/50, 60/40, 80/20 (18 bands, 6 seeds).
Methods: Fisher-Cholesky, GA, Hybrid (GA + Fisher shortlist), Base paper (MFE-CFSP), Entropy, plus all 164 bands.
GA rows for 20/40/50/80 % are taken from split_ratio.csv (same seeds, same code); everything else is computed here.
Writes results/final/split_comparison.csv.   Run: python src/split_comparison.py   (about 10 min)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_setup import make_setup
import extra_studies as E

OUT = ROOT / "results" / "final"
FRACS, K = [0.2, 0.4, 0.5, 0.6, 0.8], 18
KEEP = ["split", "learn_frac", "method", "K", "OA", "AA", "Kappa", "train_OA", "selection_time_s"]


def job(seed, frac):
    S = make_setup(seed, test_fraction=1 - frac); rows = []
    which = {"Hybrid", "Base", "Entropy"} | ({"GA"} if abs(frac - 0.6) < 1e-9 else set())   # GA already in split_ratio.csv for the others
    for name, f in E.select_all(S, E.CAND164, K, which).items():
        t = time.perf_counter(); b = f(); st = time.perf_counter() - t
        rows.append(dict(split=seed, learn_frac=frac, method=name, K=K, selection_time_s=st, **E.grade(S, b)))
    rows.append(dict(split=seed, learn_frac=frac, method="All 164 bands", K=164, selection_time_s=0.0, **E.grade(S, E.CAND164)))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    new = pd.DataFrame([r for rs in Parallel(n_jobs=2)(delayed(job)(s, f) for f in FRACS for s in E.SPLITS) for r in rs])
    old = pd.read_csv(OUT / "split_ratio.csv"); old = old[old.method.isin(["GA"]) & old.learn_frac.round(2).isin(FRACS)]
    df = pd.concat([new, old[KEEP]], ignore_index=True)
    # Fisher-Cholesky is recomputed above for every split; it must agree with split_ratio.csv where both exist
    sr = pd.read_csv(OUT / "split_ratio.csv"); sr = sr[sr.method == "Fisher-Cholesky"]
    chk = df[df.method == "Fisher-Cholesky"].merge(sr, on=["split", "learn_frac"], suffixes=("_a", "_b"))
    assert len(chk) and np.allclose(chk.OA_a, chk.OA_b), "Fisher-Cholesky differs from split_ratio.csv"
    df.to_csv(OUT / "split_comparison.csv", index=False)
    print(df.groupby(["learn_frac", "method"]).OA.mean().unstack(0).round(4).to_string())
    print(f"done in {time.time() - t0:.0f}s")
