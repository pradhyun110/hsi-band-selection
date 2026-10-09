"""
Reference subsets every algorithm must beat to be worth using.  All use TRAIN pixels only.
  uniform   : K equally spaced bands of the 164-candidate pool
  entropy   : K highest-entropy candidate bands
  fisher    : K highest Fisher-score bands (supervised filter, ignores redundancy)
  random    : 15 independent random K-subsets per K
  sfs       : greedy forward selection with the SAME LDA wrapper fitness (nested: one run to K=60)
All-band baselines (164 pre-screened, 200 raw) are evaluated in eval_subsets.py.
"""
import sys, pickle, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from optimizers import random_subset

KS = [5, 10, 15, 20, 25, 30, 40, 50, 60]


def entropy_scores(S):
    # entropy of each candidate band on TRAIN pixels (256-bin histogram), same definition as pre-screening
    X = S["X_train"]; out = []
    for j in range(X.shape[1]):
        h, _ = np.histogram(X[:, j], bins=256)
        p = h[h > 0] / h.sum(); out.append(-(p * np.log2(p)).sum())
    return np.array(out)


def fisher_scores(S):
    X, y = S["X_train"], S["y_train"]; mu = X.mean(0)
    sb = sum((y == c).sum() * (X[y == c].mean(0) - mu) ** 2 for c in S["classes"])
    sw = sum(((X[y == c] - X[y == c].mean(0)) ** 2).sum(0) for c in S["classes"])
    return sb / sw


def main():
    S = load_setup(); n = S["n_candidates"]; fit = LDAFitness(S)
    ent, fis = entropy_scores(S), fisher_scores(S)
    subsets = []
    for K in KS:
        subsets.append(dict(method="uniform", K=K, seed=0, idx=np.unique(np.round(np.linspace(0, n - 1, K)).astype(int))))
        subsets.append(dict(method="entropy", K=K, seed=0, idx=np.sort(np.argsort(-ent)[:K])))
        subsets.append(dict(method="fisher", K=K, seed=0, idx=np.sort(np.argsort(-fis)[:K])))
        for s in range(15):
            rng = np.random.default_rng(50_000 + 100 * K + s)
            subsets.append(dict(method="random", K=K, seed=s, idx=np.flatnonzero(random_subset(rng, n, K))))
    # greedy forward selection (deterministic)
    t = time.perf_counter(); sel = []; evals = 0
    rem = list(range(n)); best_seq = []
    for step in range(max(KS)):
        scores = []
        for j in rem:
            m = mask_from_idx(sel + [j], n); scores.append(fit(m)); evals += 1
        j = rem[int(np.argmax(scores))]; sel.append(j); rem.remove(j)
        best_seq.append(max(scores))
    sfs_time = time.perf_counter() - t
    for K in KS:
        subsets.append(dict(method="sfs", K=K, seed=0, idx=np.sort(np.array(sel[:K]))))
    for s in subsets:
        s["idx"] = np.asarray(s["idx"], dtype=np.int16)
        assert len(s["idx"]) == s["K"], (s["method"], s["K"], len(s["idx"]))
    pickle.dump(dict(subsets=subsets, sfs_evals=evals, sfs_time=sfs_time, sfs_curve=best_seq), open(OUT / "baseline_subsets.pkl", "wb"))
    print(f"baseline subsets: {len(subsets)}; SFS {evals} fitness calls in {sfs_time:.0f}s; SFS fitness at K=20: {best_seq[19]:.4f}")


if __name__ == "__main__":
    main()
