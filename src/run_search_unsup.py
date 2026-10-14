"""
usage: python run_search_unsup.py          (u1 and u2, K in 10,20,30,40; 7 algorithms; 15 seeds; pop 30; 3000 evals)
Also builds the greedy forward-selection baseline for U1 (nested prefixes up to K=40).
"""
import sys, pickle, time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from optimizers import ALGOS
from unsup import UnsupBase, UnsupFitness

KS = [10, 20, 30, 40]
SEEDS = list(range(15))
POP, BUDGET = 30, 3000
_c = {}


def get(kind, K):
    if "S" not in _c:
        _c["S"] = load_setup(); _c["base"] = UnsupBase(_c["S"])
    key = (kind, K)
    if key not in _c:
        _c[key] = UnsupFitness(_c["base"], kind, K, _c["S"])
    return _c["S"], _c[key]


def one(kind, alg, K, seed):
    S, fit = get(kind, K)
    r = ALGOS[alg](fit, S["n_candidates"], K, BUDGET, POP, 1000 * K + seed)
    assert r["mask"].sum() == K and r["n_evals"] == BUDGET
    return dict(kind=kind, alg=alg, K=K, seed=seed, idx=np.flatnonzero(r["mask"]).astype(np.int16),
                fitness=r["fitness"], raw_fitness=fit.raw(r["mask"]), time_s=r["time_s"], hist=r["hist"])


if __name__ == "__main__":
    S = load_setup(); base = UnsupBase(S); n = S["n_candidates"]
    # greedy forward selection on U1 (deterministic)
    t = time.perf_counter(); sel, rem = [], list(range(n))
    for step in range(max(KS)):
        sc = [base.r2(np.array(sel + [j])) for j in rem]
        j = rem[int(np.argmax(sc))]; sel.append(j); rem.remove(j)
    pickle.dump(dict(order=sel, time=time.perf_counter() - t), open(OUT / "sfs_u1.pkl", "wb"))
    print(f"greedy-R2 baseline in {time.perf_counter()-t:.1f}s", flush=True)
    for kind in ("u1", "u2"):
        for K in KS:
            f = OUT / f"search_{kind}_K{K}.pkl"
            if f.exists(): print("skip", f.name); continue
            t = time.perf_counter()
            res = Parallel(n_jobs=2)(delayed(one)(kind, a, K, s) for a in ALGOS for s in SEEDS)
            pickle.dump(res, open(f, "wb"))
            print(f"{kind} K={K}: {len(res)} runs in {time.perf_counter()-t:.0f}s", flush=True)
