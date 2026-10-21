"""
Metrics BEYOND accuracy, for the band sets of demo.py (split seed 42) at K = 15, 20, 30:

  accuracy side   OA (overall), AA (average per class), kappa
  information     reconstruction R2  : how much of ALL 164 bands can be rebuilt from the kept bands
                                       (1.0 = nothing lost; label-free, learning pixels only)
                  mean entropy (bits): average information content of the kept bands
                  redundancy         : average |correlation| between kept bands (lower = less repeated information)
  cost            selection time     : seconds the method needs to choose the bands
                  SVM train time, SVM predict time for the 8,198 exam pixels, support vectors, model size
  data size       bands kept, % of the 200 bands, bytes per pixel (16-bit values)

Run:  python src/extra_metrics.py       (needs results/demo/selected_bands_original_numbers.json from demo.py)
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import load_setup, LDAFitness
from optimizers import run_ga
from unsupervised_selectors import SELECTORS
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, cohen_kappa_score
warnings.filterwarnings("ignore")

OUT = ROOT / "results" / "extra"; OUT.mkdir(parents=True, exist_ok=True)
KS = [15, 20, 30]
S = load_setup(); cand = list(map(int, S["candidate_bands"]))
Xtr, ytr, Xte, yte = S["X_train"], S["y_train"], S["X_test"], S["y_test"]
cube = np.load(ROOT / "data" / "indianpinearray.npy").astype(float); gt = np.load(ROOT / "data" / "IPgt.npy")
Xraw_all = cube.reshape(-1, 200)[gt.reshape(-1) > 0]
from data_setup import stratified_split, RNG_SPLIT_SEED, TEST_FRACTION
tr_idx, _ = stratified_split(gt.reshape(-1)[gt.reshape(-1) > 0], TEST_FRACTION, RNG_SPLIT_SEED)
Xraw_train = Xraw_all[tr_idx][:, cand]
sets = json.load(open(ROOT / "results" / "demo" / "selected_bands_original_numbers.json"))
pos = {b: i for i, b in enumerate(cand)}

C_full = np.corrcoef(Xtr, rowvar=False)
def entropy_bits(col, bins=64):
    p, _ = np.histogram(col, bins=bins); p = p[p > 0] / p.sum(); return float(-(p * np.log2(p)).sum())
ENT = np.array([entropy_bits(Xraw_train[:, j]) for j in range(164)])

def recon_r2(idx):
    """share of the variance of all 164 bands that a linear model built on the kept bands explains."""
    idx = np.asarray(idx); Cs = C_full[np.ix_(idx, idx)] + 1e-6 * np.eye(len(idx))
    B = np.linalg.solve(Cs, C_full[idx, :])                      # regression coefficients (standardised bands)
    explained = np.einsum("ij,ij->j", C_full[idx, :], B)         # R2 of each band
    return float(np.clip(explained, 0, 1).mean())

def redundancy(idx):
    idx = np.asarray(idx)
    if len(idx) < 2: return 0.0
    c = np.abs(C_full[np.ix_(idx, idx)]); return float(c[np.triu_indices(len(idx), 1)].mean())

def svm_costs(idx, Xtr_, Xte_):
    """median of 3 timings; single process, nothing else running in this script."""
    tt, tp = [], []
    for _ in range(3):
        t = time.perf_counter(); m = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr_[:, idx], ytr); tt.append(time.perf_counter() - t)
        t = time.perf_counter(); p = m.predict(Xte_[:, idx]); tp.append(time.perf_counter() - t)
    cm = confusion_matrix(yte, p)
    return dict(OA=float((p == yte).mean()), AA=float(np.nanmean(np.diag(cm) / cm.sum(1))), Kappa=float(cohen_kappa_score(yte, p)),
                svm_train_s=float(np.median(tt)), svm_predict_s=float(np.median(tp)), us_per_pixel=float(np.median(tp) / len(yte) * 1e6),
                support_vectors=int(m.n_support_.sum()), model_kb=float(m.support_vectors_.size * 2 / 1024))   # 16-bit values

rows = []
fit = LDAFitness(S)
for K in KS:
    for name in ["Supervised (GA + labels)"] + list(SELECTORS) + ["Evenly spaced"]:
        if name.startswith("Supervised"):
            t = time.perf_counter(); r = run_ga(fit, 164, K, 3000, 30, 1000 * K); sel_t = time.perf_counter() - t   # one GA run, 3,000 fitness calls
            idx = [pos[b] for b in sets[f"{name} | K={K}"]]                                                         # the set used in the demo
        elif name == "Evenly spaced":
            sel_t = 0.0; idx = np.unique(np.round(np.linspace(0, 163, K)).astype(int))
        else:
            t = time.perf_counter(); idx = SELECTORS[name](Xraw_train, Xtr, K); sel_t = time.perf_counter() - t
        idx = np.array(sorted(idx)); assert len(idx) == K
        rows.append(dict(method=name, K=K, bands_pct=100 * K / 200, bytes_per_pixel=2 * K, recon_R2=recon_r2(idx), mean_entropy_bits=float(ENT[idx].mean()),
                         redundancy=redundancy(idx), selection_time_s=sel_t, **svm_costs(idx, Xtr, Xte)))
        print(f"K={K} {name:26s} OA={rows[-1]['OA']:.3f} R2={rows[-1]['recon_R2']:.3f} sel={sel_t:.2f}s", flush=True)

# references: all 164 and all 200 bands
for name, idx in (("All 164 cleaned bands", np.arange(164)),):
    rows.append(dict(method=name, K=164, bands_pct=82, bytes_per_pixel=2 * 164, recon_R2=1.0, mean_entropy_bits=float(ENT.mean()), redundancy=redundancy(idx), selection_time_s=0.0, **svm_costs(idx, Xtr, Xte)))
# all 200 raw bands (z-scored with learning stats)
mu, sd = Xraw_all[tr_idx].mean(0), Xraw_all[tr_idx].std(0); sd[sd == 0] = 1
Z = (Xraw_all - mu) / sd
_, te_idx = stratified_split(gt.reshape(-1)[gt.reshape(-1) > 0], TEST_FRACTION, RNG_SPLIT_SEED)
ytr200, yte200 = gt.reshape(-1)[gt.reshape(-1) > 0][tr_idx], gt.reshape(-1)[gt.reshape(-1) > 0][te_idx]
assert np.array_equal(ytr200, ytr) and np.array_equal(yte200, yte)
tt, tp = [], []
for _ in range(3):
    t = time.perf_counter(); m = SVC(kernel="rbf", C=10, gamma="scale").fit(Z[tr_idx], ytr); tt.append(time.perf_counter() - t)
    t = time.perf_counter(); p = m.predict(Z[te_idx]); tp.append(time.perf_counter() - t)
cm = confusion_matrix(yte, p)
rows.append(dict(method="All 200 raw bands", K=200, bands_pct=100, bytes_per_pixel=400, recon_R2=1.0, mean_entropy_bits=np.nan, redundancy=np.nan, selection_time_s=0.0,
                 OA=float((p == yte).mean()), AA=float(np.nanmean(np.diag(cm) / cm.sum(1))), Kappa=float(cohen_kappa_score(yte, p)), svm_train_s=float(np.median(tt)),
                 svm_predict_s=float(np.median(tp)), us_per_pixel=float(np.median(tp) / len(yte) * 1e6), support_vectors=int(m.n_support_.sum()), model_kb=float(m.support_vectors_.size * 2 / 1024)))
df = pd.DataFrame(rows); df.to_csv(OUT / "extra_metrics.csv", index=False)
print(df.round(3).to_string(index=False))
