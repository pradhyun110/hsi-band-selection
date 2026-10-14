"""
Search sweep.  usage:  python run_search.py main      (LDA wrapper fitness, K sweep)
                       python run_search.py ablation  (NCC wrapper fitness, K=20 only)
One pickle per (fitness, K) so the run can resume.  Every run: pop 30, budget 3000 fitness calls.
"""
import sys, pickle, time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from optimizers import ALGOS

KS = [5, 10, 15, 20, 25, 30, 40, 50, 60]
SEEDS = list(range(15))
POP, BUDGET = 30, 3000
_cache = {}


def _fit(kind):
    if kind not in _cache:
        S = load_setup()
        _cache[kind] = (S, LDAFitness(S) if kind == "lda" else NCCFitness(S))
    return _cache[kind]


def one(kind, alg, K, seed):
    S, fit = _fit(kind)
    r = ALGOS[alg](fit, S["n_candidates"], K, BUDGET, POP, 1000 * K + seed)
    assert r["mask"].sum() == K and r["n_evals"] == BUDGET
    return dict(kind=kind, alg=alg, K=K, seed=seed, idx=np.flatnonzero(r["mask"]).astype(np.int16),
                fitness=r["fitness"], time_s=r["time_s"], hist=r["hist"])


if __name__ == "__main__":
    mode = sys.argv[1]
    kind, Ks = ("lda", KS) if mode == "main" else ("ncc", [20])
    for K in Ks:
        f = OUT / f"search_{kind}_K{K}.pkl"
        if f.exists():
            print("skip", f.name, flush=True); continue
        t = time.perf_counter()
        tasks = [(a, s) for a in ALGOS for s in SEEDS]
        res = Parallel(n_jobs=2)(delayed(one)(kind, a, K, s) for a, s in tasks)
        pickle.dump(res, open(f, "wb"))
        print(f"{kind} K={K}: {len(res)} runs in {time.perf_counter()-t:.0f}s", flush=True)
