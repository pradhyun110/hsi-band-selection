"""
Put the three selectors from github.com/Zrahay/HyperSpectral_Image (variance, k-means band clustering, PCA loading)
through OUR clean protocol, next to our results.  The repo code was read, not executed (it has hard-coded Windows paths);
the selectors are re-implemented line-for-line.  Selection uses TRAIN pixels only (the repo uses all labelled pixels).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pickle

cube = np.load(DATA_DIR / "indianpinearray.npy").astype(np.float64); gt = np.load(DATA_DIR / "IPgt.npy")
X = cube.reshape(-1, 200); y = gt.ravel(); m = y > 0; X, y = X[m], y[m]
tr, te = stratified_split(y, TEST_FRACTION, RNG_SPLIT_SEED)
Xtr_raw = X[tr]
mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0); sd[sd == 0] = 1
Ztr, Zte, ytr, yte = (Xtr_raw - mu) / sd, (X[te] - mu) / sd, y[tr], y[te]
cand = load_setup()["candidate_bands"]


def sel_variance(Xs, k): return np.sort(np.argsort(Xs.var(0))[::-1][:k])
def sel_kmeans(Xs, k):
    Z = StandardScaler().fit_transform(Xs).T
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z); ch = []
    for c in range(k):
        for i in np.argsort(np.linalg.norm(Z - km.cluster_centers_[c], axis=1)):
            if i not in ch: ch.append(i); break
    return np.sort(np.array(ch))
def sel_pca(Xs, k):
    Z = StandardScaler().fit_transform(Xs); p = PCA(n_components=k, random_state=0).fit(Z); ch = []
    for comp in p.components_:
        for i in np.argsort(np.abs(comp))[::-1]:
            if i not in ch: ch.append(i); break
    return np.sort(np.array(ch))
SEL = {"variance": sel_variance, "kmeans": sel_kmeans, "pca": sel_pca}


def score(bands):
    s = SVC(kernel="rbf", C=10, gamma="scale").fit(Ztr[:, bands], ytr); p = s.predict(Zte[:, bands])
    cm = confusion_matrix(yte, p); aa = np.nanmean(np.diag(cm) / cm.sum(1))
    return float((p == yte).mean()), float(aa), float(cohen_kappa_score(yte, p))


def r2(bands):  # share of the 164 candidate bands' (z-scored, train) variance recoverable linearly from the chosen bands
    S = Ztr[:, bands] - Ztr[:, bands].mean(0); T = Ztr[:, cand] - Ztr[:, cand].mean(0)
    W, *_ = np.linalg.lstsq(S, T, rcond=None); return float(1 - ((T - S @ W) ** 2).sum() / (T ** 2).sum())


def red(bands):
    C = np.corrcoef(Ztr[:, bands], rowvar=False); return float(np.abs(C[np.triu_indices_from(C, 1)]).mean())


rows = []
for K in (10, 20, 30, 40):
    for name, fn in SEL.items():
        for pool, label in ((np.arange(200), "as in repo (200 raw bands)"), (cand, "on our 164 pre-screened bands")):
            b = pool[fn(Xtr_raw[:, pool], K)]
            oa, aa, kap = score(b); rows.append(dict(K=K, method=name, pool=label, OA=oa, AA=aa, Kappa=kap, R2=r2(b), mean_abs_corr=red(b), bands=" ".join(map(str, b))))
    # ours, for context
    res = [r for r in pickle.load(open(OUT / f"search_lda_K{K}.pkl", "rb")) if r["alg"] in ("SA", "GA", "GWO", "Jaya", "PSO", "DE")]
    o = [(score(cand[r["idx"].astype(int)]), r2(cand[r["idx"].astype(int)]), red(cand[r["idx"].astype(int)])) for r in res[:0]]  # placeholder (kept cheap)
    fin = json.load(open(OUT / "final_band_sets.json"))
    uni = cand[np.unique(np.round(np.linspace(0, 163, K)).astype(int))]
    for label, b in (("uniform spacing (ours, trivial baseline)", uni),):
        oa, aa, kap = score(b); rows.append(dict(K=K, method="uniform", pool=label, OA=oa, AA=aa, Kappa=kap, R2=r2(b), mean_abs_corr=red(b), bands=""))
    for key in (f"K{K}_best-validation run", f"K{K}_consensus (most frequent)"):
        if key in fin:
            b = np.array(fin[key]); oa, aa, kap = score(b); rows.append(dict(K=K, method="ours: " + key.split("_", 1)[1], pool="our wrapper (LDA fitness)", OA=oa, AA=aa, Kappa=kap, R2=r2(b), mean_abs_corr=red(b), bands=""))
    # mean over all 90 algorithm runs
    vals = [(score(cand[r["idx"].astype(int)]), r2(cand[r["idx"].astype(int)]), red(cand[r["idx"].astype(int)])) for r in res]
    rows.append(dict(K=K, method="ours: mean of 90 runs (6 algs x 15 seeds)", pool="our wrapper (LDA fitness)", OA=np.mean([v[0][0] for v in vals]), AA=np.mean([v[0][1] for v in vals]),
                     Kappa=np.mean([v[0][2] for v in vals]), R2=np.mean([v[1] for v in vals]), mean_abs_corr=np.mean([v[2] for v in vals]), bands=""))
    print("K", K, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "repo_comparison.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(df.drop(columns="bands").round(4).to_string(index=False))

# reproduce the repo's own headline number (all 200 bands, 10% train, SVM C=100, 5 stratified splits) with our code
sc = []
for s in range(5):
    a, b_, c, d = train_test_split(X, y, train_size=0.10, stratify=y, random_state=s)
    ss = StandardScaler().fit(a); sc.append(SVC(kernel="rbf", C=100, gamma="scale").fit(ss.transform(a), c).score(ss.transform(b_), d))
print("repo protocol (10%% train, C=100, all 200 bands): OA %.4f +/- %.4f  (repo reports 0.8097 +/- 0.0053)" % (np.mean(sc), np.std(sc)))
