"""
Full fair comparison: GA, PSO, GWO, Jaya, DE, SA on the Indian Pines
164-band pre-screened candidate pool.

SHARED across every algorithm (fairness):
  - candidate bands (164, from entropy pre-screening)
  - train/test split (fixed, seed=42, stratified, 20%/80%)
  - fitness function (nearest-centroid accuracy - band-count penalty)
  - population size = 25, max iterations = 50  => budget = 1250 evals
  - stopping criterion = fixed evaluation budget (no early stopping)
  - evaluation procedure (same test split, same final KNN/SVM re-check)

VARIED across runs (stability measurement):
  - algorithm-internal random seed: 0, 1, 2, 3, 4 (5 independent runs each)
"""
import sys, time, json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_setup import load_fixed_setup, NearestCentroidFitness, validate_subset_knn_svm
from algorithms import ALGORITHMS

OUT_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "results" / "legacy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POP_SIZE = 25
MAX_ITER = 50
BUDGET = POP_SIZE * MAX_ITER
SEEDS = [0, 1, 2, 3, 4]


def band_redundancy(setup, mask):
    """Mean absolute pairwise Pearson correlation among the selected bands,
    computed on the training split. Lower = less redundant subset."""
    cols = np.where(mask)[0]
    if len(cols) < 2:
        return 0.0
    X = setup["X_train"][:, cols]
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    return float(np.mean(np.abs(C[iu])))


def main():
    print("Loading fixed experimental setup (shared by all algorithms) ...")
    setup = load_fixed_setup()
    n_bands = setup["n_candidates"]
    print(f"  candidate bands: {n_bands}, train: {setup['X_train'].shape}, test: {setup['X_test'].shape}")
    print(f"  budget per run: {POP_SIZE} x {MAX_ITER} = {BUDGET} fitness evaluations")
    print(f"  seeds per algorithm: {SEEDS}\n")

    all_rows = []
    all_histories = {}  # (algorithm) -> list of acc history arrays (one per seed)
    best_overall = {}   # algorithm -> (mask, validated_knn_acc) across seeds, for reporting

    for algo_name, algo_fn in ALGORITHMS.items():
        print(f"=== {algo_name} ===")
        histories = []
        for seed in SEEDS:
            fit = NearestCentroidFitness(setup)  # fresh counter, same precomputed centroids/data
            t0 = time.perf_counter()
            res = algo_fn(fit, n_bands, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=seed)
            wall_time = time.perf_counter() - t0

            knn_acc, svm_acc = validate_subset_knn_svm(setup, res["best_mask"])
            redundancy = band_redundancy(setup, res["best_mask"])
            n_selected = int(res["best_mask"].sum())

            row = {
                "algorithm": algo_name,
                "seed": seed,
                "n_evals": res["n_evals"],
                "n_bands_selected": n_selected,
                "fitness_surrogate": res["best_fitness"],
                "acc_surrogate_ncc": res["best_acc"],
                "acc_knn5": knn_acc,
                "acc_svm_rbf": svm_acc,
                "band_redundancy": redundancy,
                "exec_time_s": wall_time,
            }
            all_rows.append(row)
            histories.append(res["history_acc"])

            if (algo_name not in best_overall) or (knn_acc > best_overall[algo_name][1]):
                best_overall[algo_name] = (res["best_mask"].copy(), knn_acc, svm_acc, n_selected)

            print(f"  seed={seed}  evals={res['n_evals']:4d}  bands={n_selected:3d}  "
                  f"NCC_acc={res['best_acc']:.4f}  KNN_acc={knn_acc:.4f}  SVM_acc={svm_acc:.4f}  "
                  f"redund={redundancy:.4f}  time={wall_time:.2f}s")

        all_histories[algo_name] = np.vstack(histories)  # (n_seeds, budget)

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(OUT_DIR / "algo_comparison_raw.csv", index=False)
    print(f"\nSaved raw results: {OUT_DIR/'algo_comparison_raw.csv'}")

    # ------------------------------------------------------------
    # Summary statistics (mean/std across the 5 seeds)
    # ------------------------------------------------------------
    summary = raw_df.groupby("algorithm").agg(
        n_bands_mean=("n_bands_selected", "mean"),
        n_bands_std=("n_bands_selected", "std"),
        fitness_mean=("fitness_surrogate", "mean"),
        fitness_std=("fitness_surrogate", "std"),
        acc_ncc_mean=("acc_surrogate_ncc", "mean"),
        acc_ncc_std=("acc_surrogate_ncc", "std"),
        acc_knn_mean=("acc_knn5", "mean"),
        acc_knn_std=("acc_knn5", "std"),
        acc_knn_best=("acc_knn5", "max"),
        acc_svm_mean=("acc_svm_rbf", "mean"),
        acc_svm_std=("acc_svm_rbf", "std"),
        acc_svm_best=("acc_svm_rbf", "max"),
        redundancy_mean=("band_redundancy", "mean"),
        redundancy_std=("band_redundancy", "std"),
        exec_time_mean=("exec_time_s", "mean"),
        exec_time_std=("exec_time_s", "std"),
        n_evals_mean=("n_evals", "mean"),
    ).reset_index()

    # Stability = std of validated KNN accuracy across seeds (lower = more stable)
    summary["stability_knn_std"] = summary["acc_knn_std"]
    summary = summary.round(4)
    summary.to_csv(OUT_DIR / "algo_comparison_summary.csv", index=False)
    print(f"Saved summary: {OUT_DIR/'algo_comparison_summary.csv'}")
    print("\n", summary.to_string(index=False))

    # ------------------------------------------------------------
    # Ranking: software performance first (explicit instruction).
    # Primary key = mean validated KNN accuracy (descending)
    # Tiebreak 1 = lower stability std (more consistent)
    # Tiebreak 2 = fewer bands selected (more compact)
    # ------------------------------------------------------------
    rank_df = summary.sort_values(
        by=["acc_knn_mean", "stability_knn_std", "n_bands_mean"],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))
    rank_df.to_csv(OUT_DIR / "algo_ranking.csv", index=False)
    print("\n=== Ranking (by mean validated KNN accuracy, software performance only) ===")
    print(rank_df[["rank", "algorithm", "acc_knn_mean", "acc_knn_std", "acc_svm_mean",
                    "n_bands_mean", "exec_time_mean", "redundancy_mean"]].to_string(index=False))

    # ------------------------------------------------------------
    # Save convergence histories + best masks for plotting / reuse
    # ------------------------------------------------------------
    np.savez(OUT_DIR / "convergence_histories.npz", **all_histories)

    best_masks_report = {}
    for algo, (mask, knn_acc, svm_acc, n_sel) in best_overall.items():
        idx_in_candidates = np.where(mask)[0]
        actual_band_indices = setup["candidate_bands"][idx_in_candidates]
        best_masks_report[algo] = {
            "n_bands": int(n_sel),
            "knn_acc": float(knn_acc),
            "svm_acc": float(svm_acc),
            "band_indices_in_original_200": actual_band_indices.tolist(),
        }
    with open(OUT_DIR / "best_band_subsets.json", "w") as f:
        json.dump(best_masks_report, f, indent=2)
    print(f"\nSaved convergence histories: {OUT_DIR/'convergence_histories.npz'}")
    print(f"Saved best band subsets (original 200-band indices): {OUT_DIR/'best_band_subsets.json'}")

    return raw_df, summary, rank_df, all_histories


if __name__ == "__main__":
    main()
