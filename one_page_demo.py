"""one_page_demo.py -- the whole idea on one page.   Run:  python one_page_demo.py   (about 1 minute)
Keeps K of the 200 bands in 7 different ways and grades each choice on pixels it never saw."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

K = 20                                   # how many bands to keep
rng = np.random.default_rng(0)

# 1. LOAD the 200-band image and the land-cover labels (0 = no label, so we drop those pixels)
cube, gt = np.load("data/indianpinearray.npy"), np.load("data/IPgt.npy")
keep = gt.reshape(-1) > 0
X, y = cube.reshape(-1, 200)[keep].astype(float), gt.reshape(-1)[keep]
cand = np.load("results/selected_band_indices.npy")          # the 164 bands left after noise screening

# 2. SPLIT: 20 % of pixels to learn from, 80 % hidden as the exam (same share of every class)
learn = np.zeros(len(y), bool)
for c in np.unique(y):
    idx = rng.permutation(np.flatnonzero(y == c)); learn[idx[: round(0.2 * len(idx))]] = True
raw = X[learn][:, cand]                                      # raw values of the learning pixels
mu, sd = raw.mean(0), raw.std(0) + 1e-9
Xl, yl = (raw - mu) / sd, y[learn]                           # scaled using learning pixels only
Xe, ye = (X[~learn][:, cand] - mu) / sd, y[~learn]           # exam pixels, scaled the same way

# 3. GRADE any band list: train an SVM on the learning pixels, count correct answers on the exam
def exam_accuracy(bands):
    return SVC(C=10).fit(Xl[:, bands], yl).score(Xe[:, bands], ye)

# 4. SUPERVISED choice (uses labels): a genetic algorithm that breeds good band lists
half = rng.permutation(len(yl)); A, B = half[: len(yl) // 2], half[len(yl) // 2:]
def fitness(bands):                       # score a band list using ONLY learning pixels: learn on half A, test on half B, and back
    f = lambda tr, te: LDA(solver="lsqr", shrinkage=0.1).fit(Xl[tr][:, bands], yl[tr]).score(Xl[te][:, bands], yl[te])
    return (f(A, B) + f(B, A)) / 2

def genetic_algorithm(size=30, tries=1500):
    pop = [rng.choice(164, K, replace=False) for _ in range(size)]      # start with 30 random band lists
    score = [fitness(p) for p in pop]
    for _ in range(tries - size):
        pick = lambda: pop[max(rng.choice(size, 3), key=lambda i: score[i])]   # best of 3 random lists
        p1, p2 = pick(), pick()
        child = list(set(p1) & set(p2))                                 # keep the bands both parents chose
        rest = rng.permutation(list((set(p1) | set(p2)) - set(child)))
        child += [int(b) for b in rest[: K - len(child)]]               # fill up with bands from either parent
        if rng.random() < 0.5:                                          # mutation: swap one band for a new one
            child[rng.integers(K)] = int(rng.choice([b for b in range(164) if b not in child]))
        s, worst = fitness(child), int(np.argmin(score))
        if s > score[worst]: pop[worst], score[worst] = np.array(child), s     # a better child replaces the worst
    return np.array(pop[int(np.argmax(score))])

# 5. UNSUPERVISED choices (never look at labels)
def by_variance():   return np.argsort(-raw.var(0))[:K]                 # most-varying bands
def by_entropy():                                                       # most "informative" bands
    h = []
    for j in range(164):
        p, _ = np.histogram(raw[:, j], bins=64); p = p[p > 0] / p.sum(); h.append(-(p * np.log2(p)).sum())
    return np.argsort(-np.array(h))[:K]
def by_kmeans():                                                        # group similar bands, keep one per group
    km = KMeans(K, n_init=10, random_state=0).fit(Xl.T)
    return np.array([np.flatnonzero(km.labels_ == c)[np.argmin(((Xl.T[km.labels_ == c] - km.cluster_centers_[c]) ** 2).sum(1))] for c in range(K)])
def by_pca():                                                           # one band per main direction of variation
    chosen = []
    for comp in PCA(K).fit(Xl).components_:
        chosen.append(next(int(i) for i in np.argsort(-abs(comp)) if i not in chosen))
    return np.array(chosen)
def by_greedy_r2():                                                     # add the band that best rebuilds all other bands
    R, chosen = np.corrcoef(Xl, rowvar=False), []
    for _ in range(K):
        gain = (R ** 2).sum(0) / np.maximum(np.diag(R), 1e-12); gain[chosen] = -1
        b = int(np.argmax(gain)); chosen.append(b); R = R - np.outer(R[:, b], R[b]) / R[b, b]
    return np.array(chosen)

# 6. RUN everything and print the exam scores
results = {"Supervised (genetic algorithm)": genetic_algorithm(), "Evenly spaced": np.linspace(0, 163, K).round().astype(int),
           "Variance": by_variance(), "Entropy": by_entropy(), "K-means": by_kmeans(), "PCA": by_pca(), "Greedy R2": by_greedy_r2()}
print(f"\nExam accuracy with only {K} of 200 bands:")
for name, bands in sorted(results.items(), key=lambda kv: -exam_accuracy(kv[1])):
    print(f"  {name:32s} {100 * exam_accuracy(bands):5.1f} %")
print(f"  {'All 164 cleaned bands':32s} {100 * exam_accuracy(np.arange(164)):5.1f} %")
