"""
Entropy-based band pre-screening for the Indian Pines hyperspectral cube.

Purpose
-------
Reduce the 200-band cube to a smaller, reproducible candidate set BEFORE
any metaheuristic optimizer (GWO / Jaya / PSO / GA / DE) is run on it.
This script only screens bands by per-band information content; it does
not run any optimization or deep-learning method.

Strategies compared (all derived deterministically from the same
per-band Shannon entropy vector -- no manual/arbitrary threshold is
hand-picked unless explicitly flagged as such):

  1. Mean threshold          : keep bands with H >= mean(H)
  2. Median threshold        : keep bands with H >= median(H)
  3. Mean - 0.5*std threshold: a slightly more permissive statistical cut
  4. Otsu threshold          : automatic bimodal split of the entropy
                                histogram (Otsu, 1979) -- fully data-driven
  5. Percentile cut (25%)    : drop the lowest-entropy 25% of bands
  6. Percentile cut (33%)    : drop the lowest-entropy 33% of bands
  7. Fixed top-K (150)       : keep the 150 highest-entropy bands (the
                                "150" is an a-priori arbitrary choice --
                                included only as a non-reproducible
                                baseline for comparison)

Entropy definition
-------------------
For each band b, pixel intensities are discretized into 256 bins
(matching the 256-bin histogram unit already specified in the FPGA
block diagram / BRAM budget for this project, so the software reference
and the eventual hardware entropy unit use the same bin count).
Shannon entropy is:  H(b) = - sum_i p_i * log2(p_i),  p_i = histogram
probability of bin i.  Max possible value = log2(256) = 8 bits.

Outputs
-------
- outputs/entropy_per_band.csv            : raw per-band entropy values
- outputs/prescreening_comparison.csv      : the comparison table
- outputs/prescreening_comparison.png      : visual comparison
- outputs/selected_band_indices.npy        : indices retained by the
                                              RECOMMENDED strategy (int32)
- outputs/selected_band_indices.csv        : same, human-readable
- outputs/prescreening_report.json         : machine-readable summary

The original indianpinearray.npy / IPgt.npy are opened read-only and
are never written to.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CUBE_PATH = DATA_DIR / "indianpinearray.npy"
GT_PATH = DATA_DIR / "IPgt.npy"

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 256          # matches the FPGA histogram-unit bin count from the synopsis
TOPK_FIXED = 150      # arbitrary baseline, included only to show non-reproducibility


# ---------------------------------------------------------------------
# Core entropy computation
# ---------------------------------------------------------------------
def compute_band_entropy(cube: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Per-band Shannon entropy (bits), histogram-based, read-only on cube."""
    n_bands = cube.shape[2]
    entropy = np.zeros(n_bands, dtype=np.float64)
    for b in range(n_bands):
        band = cube[:, :, b].astype(np.float64).ravel()
        hist, _ = np.histogram(band, bins=n_bins)
        p = hist.astype(np.float64)
        p = p[p > 0]
        p = p / p.sum()
        entropy[b] = -np.sum(p * np.log2(p))
    return entropy


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """
    Otsu's method (1979) applied to the 1-D entropy distribution itself:
    finds the threshold that minimizes intra-class variance / maximizes
    inter-class variance between a 'low-entropy' and 'high-entropy'
    group of bands. Fully deterministic given `values` -- no free
    parameter to hand-tune.
    """
    hist, bin_edges = np.histogram(values, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist = hist.astype(np.float64)
    total = hist.sum()

    best_thresh = bin_centers[0]
    best_between_var = -1.0
    w0 = 0.0
    sum_total = np.sum(hist * bin_centers)
    sum0 = 0.0

    for i in range(n_bins):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * bin_centers[i]
        mean0 = sum0 / w0
        mean1 = (sum_total - sum0) / w1
        between_var = w0 * w1 * (mean0 - mean1) ** 2
        if between_var > best_between_var:
            best_between_var = between_var
            best_thresh = bin_centers[i]

    return float(best_thresh)


# ---------------------------------------------------------------------
# Screening strategies -- each returns (indices_kept, threshold_used, reproducible_flag, reason)
# ---------------------------------------------------------------------
def strategy_mean(entropy):
    t0 = time.perf_counter()
    thr = entropy.mean()
    idx = np.where(entropy >= thr)[0]
    dt = time.perf_counter() - t0
    return idx, thr, True, "Threshold = mean(H); no free parameter.", dt


def strategy_median(entropy):
    t0 = time.perf_counter()
    thr = np.median(entropy)
    idx = np.where(entropy >= thr)[0]
    dt = time.perf_counter() - t0
    return idx, thr, True, "Threshold = median(H); no free parameter.", dt


def strategy_mean_minus_half_std(entropy):
    t0 = time.perf_counter()
    thr = entropy.mean() - 0.5 * entropy.std()
    idx = np.where(entropy >= thr)[0]
    dt = time.perf_counter() - t0
    return idx, thr, True, "Threshold = mean(H) - 0.5*std(H); fixed multiplier, no dataset-specific tuning.", dt


def strategy_otsu(entropy):
    t0 = time.perf_counter()
    thr = otsu_threshold(entropy, n_bins=256)
    idx = np.where(entropy >= thr)[0]
    dt = time.perf_counter() - t0
    return idx, thr, True, "Otsu automatic bimodal threshold; fully data-driven, no manual parameter.", dt


def strategy_percentile(entropy, pct_drop):
    t0 = time.perf_counter()
    thr = np.percentile(entropy, pct_drop)
    idx = np.where(entropy >= thr)[0]
    dt = time.perf_counter() - t0
    return idx, thr, True, f"Drop lowest {pct_drop}% by entropy; rule fixed but cut-fraction is an a-priori design choice.", dt


def strategy_topk(entropy, k):
    t0 = time.perf_counter()
    order = np.argsort(entropy)[::-1]
    idx = np.sort(order[:k])
    thr = entropy[order[k - 1]]
    dt = time.perf_counter() - t0
    return idx, thr, False, f"Keep top-{k} bands; k chosen arbitrarily a priori (not reproducible across datasets).", dt


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading cube (read-only) ...")
    cube = np.load(CUBE_PATH, mmap_mode="r")
    cube = np.asarray(cube)  # load into memory for repeated passes; original file untouched
    n_bands = cube.shape[2]
    print(f"Cube shape: {cube.shape}, dtype on disk: {np.load(CUBE_PATH, mmap_mode='r').dtype}")

    print(f"Computing per-band Shannon entropy ({N_BINS} bins) ...")
    t0 = time.perf_counter()
    entropy = compute_band_entropy(cube, N_BINS)
    entropy_time = time.perf_counter() - t0
    print(f"  done in {entropy_time:.4f} s  (shared cost for all strategies below)")

    total_entropy = entropy.sum()
    max_possible = n_bands * np.log2(N_BINS)
    print(f"Total entropy: {total_entropy:.2f} bits  (max possible = {max_possible:.2f} bits)")

    # save raw per-band entropy
    pd.DataFrame({"band_index": np.arange(n_bands), "entropy_bits": entropy}).to_csv(
        OUT_DIR / "entropy_per_band.csv", index=False
    )

    strategies = {}

    idx, thr, repro, reason, dt = strategy_mean(entropy)
    strategies["Mean threshold"] = (idx, thr, repro, reason, dt)

    idx, thr, repro, reason, dt = strategy_median(entropy)
    strategies["Median threshold"] = (idx, thr, repro, reason, dt)

    idx, thr, repro, reason, dt = strategy_mean_minus_half_std(entropy)
    strategies["Mean - 0.5*std threshold"] = (idx, thr, repro, reason, dt)

    idx, thr, repro, reason, dt = strategy_otsu(entropy)
    strategies["Otsu automatic threshold"] = (idx, thr, repro, reason, dt)

    idx, thr, repro, reason, dt = strategy_percentile(entropy, 25)
    strategies["Percentile cut (drop lowest 25%)"] = (idx, thr, repro, reason, dt)

    idx, thr, repro, reason, dt = strategy_percentile(entropy, 33)
    strategies["Percentile cut (drop lowest 33%)"] = (idx, thr, repro, reason, dt)

    idx, thr, repro, reason, dt = strategy_topk(entropy, TOPK_FIXED)
    strategies[f"Fixed top-K (K={TOPK_FIXED})"] = (idx, thr, repro, reason, dt)

    # ------------------------------------------------------------
    # Build comparison table
    # ------------------------------------------------------------
    rows = []
    for name, (idx, thr, repro, reason, dt) in strategies.items():
        n_kept = len(idx)
        pct_removed = 100.0 * (n_bands - n_kept) / n_bands
        info_retained = 100.0 * entropy[idx].sum() / total_entropy
        rows.append(
            {
                "Strategy": name,
                "Threshold (bits)": round(thr, 4),
                "Bands retained": n_kept,
                "Bands removed": n_bands - n_kept,
                "% removed": round(pct_removed, 2),
                "Information retained (%)": round(info_retained, 2),
                "Screening time (s)": round(dt, 6),
                "Reproducible": "Yes" if repro else "No",
                "Reason": reason,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "prescreening_comparison.csv", index=False)
    print("\n=== Pre-screening strategy comparison ===")
    print(df.to_string(index=False))

    # ------------------------------------------------------------
    # Recommendation logic (explicit, auditable rule -- not a guess):
    # Prefer a strategy that is (a) reproducible (no hand-tuned
    # dataset-specific constant), (b) removes a meaningful fraction of
    # bands (> 10%) so the optimizer's search space actually shrinks,
    # and (c) retains the highest information among the reproducible
    # candidates that satisfy (b).
    # ------------------------------------------------------------
    candidates = df[(df["Reproducible"] == "Yes") & (df["% removed"] > 10.0)]
    best_row = candidates.sort_values("Information retained (%)", ascending=False).iloc[0]
    recommended = best_row["Strategy"]
    rec_idx = strategies[recommended][0]

    print(f"\nRecommended strategy: {recommended}")
    print(f"  Bands retained: {len(rec_idx)} / {n_bands}")
    print(f"  Information retained: {best_row['Information retained (%)']}%")

    # ------------------------------------------------------------
    # Save selected indices (for reuse by every optimizer later)
    # ------------------------------------------------------------
    rec_idx_sorted = np.sort(rec_idx).astype(np.int32)
    np.save(OUT_DIR / "selected_band_indices.npy", rec_idx_sorted)
    pd.DataFrame({"retained_band_index": rec_idx_sorted}).to_csv(
        OUT_DIR / "selected_band_indices.csv", index=False
    )

    report = {
        "n_bands_original": int(n_bands),
        "n_bins_histogram": N_BINS,
        "total_entropy_bits": float(total_entropy),
        "max_possible_entropy_bits": float(max_possible),
        "entropy_computation_time_s": float(entropy_time),
        "recommended_strategy": recommended,
        "recommended_threshold_bits": float(best_row["Threshold (bits)"]),
        "recommended_bands_retained": int(len(rec_idx_sorted)),
        "recommended_bands_removed": int(n_bands - len(rec_idx_sorted)),
        "recommended_pct_removed": float(best_row["% removed"]),
        "recommended_information_retained_pct": float(best_row["Information retained (%)"]),
        "selected_band_indices_file": "selected_band_indices.npy",
    }
    with open(OUT_DIR / "prescreening_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: {OUT_DIR/'entropy_per_band.csv'}")
    print(f"Saved: {OUT_DIR/'prescreening_comparison.csv'}")
    print(f"Saved: {OUT_DIR/'selected_band_indices.npy'} (int32, sorted, {len(rec_idx_sorted)} indices)")
    print(f"Saved: {OUT_DIR/'selected_band_indices.csv'}")
    print(f"Saved: {OUT_DIR/'prescreening_report.json'}")

    return df, entropy, strategies, recommended


if __name__ == "__main__":
    main()
