"""
FINAL COMPARISON -- 8 band-selection methods on Indian Pines, every metric.

Methods (all pick K bands; all except "on 200 bands" start from the same 164 Otsu-pre-screened bands)
  Fisher-Cholesky                 chosen method: greedy class-separation (Fisher/LDA) with recursive Cholesky
  GA                              genetic algorithm, LDA wrapper fitness on learning pixels
  Hybrid (GA + Fisher shortlist)  GA choosing K of the 2K bands Fisher-Cholesky ranks first
  Jaya, Grey wolf (GWO)           metaheuristics with the identical fitness and budget as the GA
  Base paper (MFE-CFSP)           our re-implementation of the base paper's selector (label-free)
  Entropy, Variance               label-free rankings (keep the K highest-scoring bands)
  references                      all 164 pre-screened bands, all 200 bands,
                                  Fisher-Cholesky run on all 200 bands (shows what the pre-screening does)
Search budget for GA / Jaya / GWO / Hybrid: 5 starts x 3,000 fitness calls, population 30, best start
chosen by learning-pixel fitness.  Labels are used only from the learning pixels.

Protocol: 6 outer splits (stratified 20 % learn / 80 % exam; seeds 42, 1, 2, 3, 4, 5),
K = 5, 10, 15, 18, 20, 25, 30.  Grading: SVM-RBF (C = 10, gamma = 'scale') trained on the learning
pixels with only the chosen bands, scored on the exam pixels (see metrics.py for every metric).

Outputs (results/final/)
  results_long.csv            one row per split x method x K
  band_sets.json              chosen band numbers (0-199), every split / method / K
  maps_split42_K20.npz        predicted labels for the classification-map figure
  ablation_base_paper_K20.csv the base paper's Table II repeated on our data

Run:  python src/final_comparison.py      (about 25 min on 2 cores)      --test  for a 1-minute check
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
from metrics import classification_metrics, cem_auc, SSIMReconstruction
from sklearn.svm import SVC
warnings.filterwarnings("ignore")

SPLITS = [42, 1, 2, 3, 4, 5]
KS = [5, 10, 15, 18, 20, 25, 30]
STARTS, BUDGET, POP = 5, 3000, 30
MAP_METHODS = ["Fisher-Cholesky", "GA", "Hybrid (GA + Fisher shortlist)", "Base paper (MFE-CFSP)", "Entropy"]
OUT = ROOT / "results" / "final"; OUT.mkdir(parents=True, exist_ok=True)
CUBE = np.load(ROOT / "data" / "indianpinearray.npy").astype(np.float64)
CAND = np.load(ROOT / "results" / "prescreen" / "selected_band_indices.npy")
SSIM = SSIMReconstruction(CUBE, CAND)


class SubFitness:
    """fitness on a shortlist of candidate positions (for the hybrid)."""
    def __init__(self, fit, short): self.fit, self.short = fit, np.asarray(short)
    def __call__(self, m):
        full = np.zeros(len(CAND), bool); full[self.short[m]] = True; return self.fit(full)


def best_of_starts(algo, fit, n, K, seed, starts, budget):
    runs = [algo(fit, n, K, budget, POP, 1000 * K + 97 * seed + s) for s in range(starts)]
    return np.flatnonzero(max(runs, key=lambda r: r["fitness"])["mask"])


def methods(S, fit, K, starts, budget):
    """yield (name, function returning ORIGINAL band numbers 0-199, fitness calls)."""
    Xtr, ytr, seed = S["X_train"], S["y_train"], S["seed"]
    n = len(CAND)
    yield "Fisher-Cholesky", lambda: CAND[fisher_select(list(range(n)), Xtr, ytr, K)], 0
    yield "GA", lambda: CAND[best_of_starts(run_ga, fit, n, K, seed, starts, budget)], starts * budget
    def hybrid():
        short = np.array(fisher_select(list(range(n)), Xtr, ytr, 2 * K))
        return CAND[short[best_of_starts(run_ga, SubFitness(fit, short), len(short), K, seed, starts, budget)]]
    yield "Hybrid (GA + Fisher shortlist)", hybrid, starts * budget
    yield "Jaya", lambda: CAND[best_of_starts(run_jaya, fit, n, K, seed, starts, budget)], starts * budget
    yield "Grey wolf (GWO)", lambda: CAND[best_of_starts(run_gwo, fit, n, K, seed, starts, budget)], starts * budget
    yield "Base paper (MFE-CFSP)", lambda: CAND[mfe_from_raw(CUBE[:, :, CAND], K)], 0
    yield "Entropy", lambda: CAND[sel_entropy(S["Xraw_train"][:, CAND], Xtr, K)], 0
    yield "Variance", lambda: CAND[sel_variance(S["Xraw_train"][:, CAND], Xtr, K)], 0
    yield "Fisher-Cholesky on 200 bands", lambda: np.array(fisher_select(list(range(200)), S["Z_train"], ytr, K)), 0


def grade(S, bands, keep_map=False):
    b = np.asarray(sorted(int(x) for x in bands))
    clf = SVC(kernel="rbf", C=10, gamma="scale").fit(S["Z_train"][:, b], S["y_train"])
    r = classification_metrics(S["y_test"], clf.predict(S["Z_test"][:, b]), S["classes"])
    r["CEM_AUC"] = cem_auc(S["Z_image"][:, b], S["Z_train"][:, b], S["y_train"], S["Z_test"][:, b], S["y_test"])
    r["SSIM"] = SSIM(b) if len(b) < 164 else 1.0
    pred_map = clf.predict(S["Z_image"][S["labelled"]][:, b]) if keep_map else None
    return r, pred_map


def run_split(seed, ks, starts, budget):
    t0 = time.time(); S = make_setup(seed); fit = LDAFitness(S); rows, sets, maps = [], {}, {}
    for name, bands in (("All 164 bands", CAND), ("All 200 bands", np.arange(200))):
        r, _ = grade(S, bands); r["SSIM"] = 1.0
        rows.append(dict(split=seed, method=name, K=len(bands), selection_time_s=0.0, fitness_calls=0, **r))
    for K in ks:
        for name, f, calls in methods(S, fit, K, starts, budget):
            t = time.perf_counter(); bands = np.array(sorted(int(x) for x in f())); st = time.perf_counter() - t
            assert len(set(bands)) == K, (name, K)
            keep_map = seed == 42 and K == 20 and name in MAP_METHODS
            r, pm = grade(S, bands, keep_map)
            rows.append(dict(split=seed, method=name, K=K, selection_time_s=st, fitness_calls=calls, **r))
            sets[f"{name}|K={K}"] = bands.tolist()
            if keep_map: maps[name] = pm
        print(f"  split {seed} K={K} done ({time.time() - t0:.0f}s)", flush=True)
    return rows, sets, maps


def ablation(K=20, seed=42, reps=3):
    """the base paper's Table II on our data: add one module at a time, time from the raw cube."""
    S = make_setup(seed); C164 = CUBE[:, :, CAND]; rows = []
    steps = [("CFSP-style (their baseline)", dict(use_gate=False, spatial=False, prune=False, recursive=False, spline=True)),
             ("+ spatial features", dict(use_gate=False, spatial=True, prune=False, recursive=False)),
             ("+ entropy gate", dict(use_gate=True, spatial=True, prune=False, recursive=False)),
             ("+ cosine pruning", dict(use_gate=True, spatial=True, prune=True, recursive=False)),
             ("+ recursive Cholesky (full MFE-CFSP)", dict(use_gate=True, spatial=True, prune=True, recursive=True))]
    for name, kw in steps:
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); idx = mfe_from_raw(C164, K, **kw); ts.append(time.perf_counter() - t)
        r, _ = grade(S, CAND[idx])
        rows.append(dict(step=name, K=K, time_s=float(np.median(ts)), OA=r["OA"], AA=r["AA"], Kappa=r["Kappa"], F1=r["F1"]))
    return pd.DataFrame(rows)


def regrade_split(seed, sets):
    """re-grade saved band sets (no new selection): adds MCC and the K = 20 confusion matrices."""
    from sklearn.metrics import confusion_matrix
    S = make_setup(seed); rows, cms = [], {}
    items = list(sets.items()) + [("All 164 bands|K=164", CAND.tolist()), ("All 200 bands|K=200", list(range(200)))]
    for key, bands in items:
        name, K = key.split("|K="); b = np.asarray(bands)
        p = SVC(kernel="rbf", C=10, gamma="scale").fit(S["Z_train"][:, b], S["y_train"]).predict(S["Z_test"][:, b])
        r = classification_metrics(S["y_test"], p, S["classes"])
        rows.append(dict(split=seed, method=name, K=int(K), MCC=r["MCC"], OA_check=r["OA"]))
        if int(K) in (20, 164, 200): cms[key] = confusion_matrix(S["y_test"], p, labels=S["classes"])
    return rows, cms


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true")
    ap.add_argument("--regrade", action="store_true", help="add MCC + confusion matrices from saved band sets")
    ap.add_argument("--maps", action="store_true", help="classification maps (split 42, K = 20) for every method")
    a = ap.parse_args()
    if a.maps:
        S = make_setup(42); sets = json.load(open(OUT / "band_sets.json"))["split=42"]; maps = {}
        for name in [k.split("|")[0] for k in sets if k.endswith("|K=20")] + ["All 200 bands"]:
            b = np.arange(200) if name == "All 200 bands" else np.asarray(sets[f"{name}|K=20"])
            clf = SVC(kernel="rbf", C=10, gamma="scale").fit(S["Z_train"][:, b], S["y_train"])
            maps[name.replace(" ", "_")] = clf.predict(S["Z_image"][S["labelled"]][:, b])
        np.savez_compressed(OUT / "maps_split42_K20.npz", labelled=S["labelled"], train_idx=S["train_idx"], test_idx=S["test_idx"], **maps)
        print("maps saved:", list(maps)); sys.exit()
    if a.regrade:
        sets = json.load(open(OUT / "band_sets.json"))
        out = Parallel(n_jobs=2)(delayed(regrade_split)(int(k.split("=")[1]), v) for k, v in sets.items())
        extra = pd.DataFrame([r for rows, _ in out for r in rows])
        df = pd.read_csv(OUT / "results_long.csv").drop(columns=["MCC"], errors="ignore").merge(extra, on=["split", "method", "K"])
        assert np.allclose(df.OA, df.OA_check), "re-graded OA differs from the saved OA"
        df.drop(columns="OA_check").to_csv(OUT / "results_long.csv", index=False)
        keys = out[0][1].keys()
        np.savez_compressed(OUT / "confusion_matrices_K20_sum6splits.npz",
                            **{k.split("|")[0].replace(" ", "_"): sum(c[k] for _, c in out) for k in keys})
        print("regrade done, OA identical, MCC added"); sys.exit()
    splits, ks, starts, bud = ([42], [20], 1, 300) if a.test else (SPLITS, KS, STARTS, BUDGET)
    t = time.time()
    out = Parallel(n_jobs=2)(delayed(run_split)(s, ks, starts, bud) for s in splits)
    tag = "_TEST" if a.test else ""
    pd.DataFrame([r for rows, _, _ in out for r in rows]).to_csv(OUT / f"results_long{tag}.csv", index=False)
    json.dump({f"split={s}": sets for s, (_, sets, _) in zip(splits, out)}, open(OUT / f"band_sets{tag}.json", "w"), indent=1)
    maps = {k: v for _, _, m in out for k, v in m.items()}
    if maps:
        S = make_setup(42)
        np.savez_compressed(OUT / f"maps_split42_K20{tag}.npz", labelled=S["labelled"], **{k.replace(" ", "_"): v for k, v in maps.items()})
    if not a.test:
        ablation().to_csv(OUT / "ablation_base_paper_K20.csv", index=False)
    print(f"done in {time.time() - t:.0f}s")
