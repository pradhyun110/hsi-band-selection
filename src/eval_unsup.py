"""Score the unsupervised-fitness subsets (+ greedy-R2 baseline) on the held-out test pixels with the same 5 classifiers."""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from eval_subsets import eval_one

if __name__ == "__main__":
    jobs = []
    for f in sorted(OUT.glob("search_u[12]_K*.pkl")):
        for r in pickle.load(open(f, "rb")):
            jobs.append((dict(group="algorithm", method=r["alg"], fitness_kind=r["kind"], K=r["K"], seed=r["seed"],
                              search_fitness=r["fitness"], search_time_s=r["time_s"]), r["idx"], "c164"))
    order = pickle.load(open(OUT / "sfs_u1.pkl", "rb"))["order"]
    for K in (10, 20, 30, 40):
        jobs.append((dict(group="baseline", method="greedy_R2", fitness_kind="u1", K=K, seed=0, search_fitness=np.nan, search_time_s=np.nan),
                     np.sort(np.array(order[:K])), "c164"))
    print(len(jobs), "subsets", flush=True); t = time.perf_counter()
    out = Parallel(n_jobs=2, batch_size=4)(delayed(eval_one)(m, i, w) for m, i, w in jobs)
    df = pd.DataFrame([r for rows in out for r in rows]); df.to_csv(OUT / "eval_unsup_long.csv", index=False)
    print(f"done in {time.perf_counter()-t:.0f}s, {len(df)} rows", flush=True)
