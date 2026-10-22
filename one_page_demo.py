"""one_page_demo.py -- the whole idea on one page.   Run:  python one_page_demo.py   (about 1 minute)
Keeps K of the 200 bands in 4 ways and grades each choice on pixels it never saw."""
import time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import cohen_kappa_score, recall_score

K = 20                                   # how many bands to keep
rng = np.random.default_rng(0)

# 1. LOAD the 200-band image and the crop labels (0 = no label, so those pixels are dropped)
cube, gt = np.load("data/indianpinearray.npy"), np.load("data/IPgt.npy")
keep = gt.reshape(-1) > 0
X, y = cube.reshape(-1, 200)[keep].astype(float), gt.reshape(-1)[keep]
cand = np.load("results/prescreen/selected_band_indices.npy")   # the 164 bands left after entropy (Otsu) screening

# 2. SPLIT: 20 % of pixels to learn from, 80 % hidden as the exam (same share of every crop)
learn = np.zeros(len(y), bool)
for c in np.unique(y):
    idx = rng.permutation(np.flatnonzero(y == c)); learn[idx[: round(0.2 * len(idx))]] = True
raw = X[learn][:, cand]
mu, sd = raw.mean(0), raw.std(0) + 1e-9
Xl, yl = (raw - mu) / sd, y[learn]                           # learning pixels, scaled with their own statistics
Xe, ye = (X[~learn][:, cand] - mu) / sd, y[~learn]           # exam pixels, scaled the same way

# 3. GRADE a band list: SVM trained on the learning pixels, tested on the exam pixels
def grade(bands):
    p = SVC(C=10).fit(Xl[:, bands], yl).predict(Xe[:, bands])
    return 100 * (p == ye).mean(), 100 * recall_score(ye, p, average="macro"), cohen_kappa_score(ye, p)   # OA, AA, kappa

# 4. FISHER-CHOLESKY (chosen method): add the band that most improves class separation J = trace(Sw^-1 Sb)
def fisher_cholesky():
    classes = np.unique(yl); m = Xl.mean(0)
    n = np.array([(yl == c).sum() for c in classes]); D = np.vstack([Xl[yl == c].mean(0) - m for c in classes])
    W = Xl - (D + m)[np.searchsorted(classes, yl)]; Sw = W.T @ W / len(yl); Sw += 0.01 * np.trace(Sw) / 164 * np.eye(164)
    J = lambda s: sum(n[i] * D[i, s] @ np.linalg.solve(Sw[np.ix_(s, s)], D[i, s]) for i in range(len(classes)))
    chosen = []
    for _ in range(K):                   # src/fisher_cholesky.py gives the same bands, faster, by updating a Cholesky factor
        chosen.append(max((b for b in range(164) if b not in chosen), key=lambda b: J(chosen + [b])))
    return np.array(chosen)

# 5. GENETIC ALGORITHM: breeds band lists, scored by an LDA classifier on two halves of the learning pixels
half = rng.permutation(len(yl)); A, B = half[: len(yl) // 2], half[len(yl) // 2:]
def fitness(bands):
    f = lambda tr, te: LDA(solver="lsqr", shrinkage=0.1).fit(Xl[tr][:, bands], yl[tr]).score(Xl[te][:, bands], yl[te])
    return (f(A, B) + f(B, A)) / 2
def genetic_algorithm(size=30, tries=1500):
    pop = [rng.choice(164, K, replace=False) for _ in range(size)]; score = [fitness(p) for p in pop]
    for _ in range(tries - size):
        pick = lambda: pop[max(rng.choice(size, 3), key=lambda i: score[i])]      # best of 3 random lists
        p1, p2 = pick(), pick()
        child = list(set(p1) & set(p2))                                           # bands both parents chose
        child += [int(b) for b in rng.permutation(list((set(p1) | set(p2)) - set(child)))[: K - len(child)]]
        if rng.random() < 0.5:                                                    # mutation: swap one band
            child[rng.integers(K)] = int(rng.choice([b for b in range(164) if b not in child]))
        s, worst = fitness(child), int(np.argmin(score))
        if s > score[worst]: pop[worst], score[worst] = np.array(child), s        # better child replaces the worst
    return np.array(pop[int(np.argmax(score))])

# 6. LABEL-FREE RANKINGS: keep the K bands with the highest entropy / variance
def entropy_ranking():
    h = []
    for j in range(164):
        p, _ = np.histogram(raw[:, j], bins=64); p = p[p > 0] / p.sum(); h.append(-(p * np.log2(p)).sum())
    return np.argsort(-np.array(h))[:K]
def variance_ranking(): return np.argsort(-raw.var(0))[:K]

# 7. RUN everything and print the exam scores
print(f"\nWith only {K} of 200 bands:            OA (%)  AA (%)  Kappa  Selection time")
for name, f in [("Fisher-Cholesky (chosen)", fisher_cholesky), ("Genetic algorithm", genetic_algorithm),
                ("Entropy ranking", entropy_ranking), ("Variance ranking", variance_ranking)]:
    t = time.perf_counter(); bands = f(); t = time.perf_counter() - t
    oa, aa, k = grade(bands)
    print(f"  {name:34s} {oa:6.1f}  {aa:6.1f}  {k:5.3f}  {t:6.2f} s   bands {sorted(cand[bands].tolist())}")
oa, aa, k = grade(np.arange(164)); print(f"  {'All 164 pre-screened bands':34s} {oa:6.1f}  {aa:6.1f}  {k:5.3f}")
