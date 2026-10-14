"""
Shared, fixed experimental setup used by EVERY band-selection algorithm.
Loading this module once and reusing its objects is what guarantees a
fair, apples-to-apples comparison across GA / PSO / GWO / Jaya / DE / SA.
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"

RNG_SPLIT_SEED = 42      # fixed once -- NOT varied across runs
TEST_FRACTION = 0.80     # 20% train / 80% test, stratified (matches earlier agreed protocol)
BAND_PENALTY_LAMBDA = 0.01  # fitness penalty weight per fraction of bands used


def stratified_split(y, test_fraction, seed):
    """Deterministic stratified split, no sklearn dependency, pure numpy."""
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        n_test = int(round(len(c_idx) * test_fraction))
        test_idx.append(c_idx[:n_test])
        train_idx.append(c_idx[n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def load_fixed_setup():
    cube = np.load(DATA_DIR / "indianpinearray.npy").astype(np.float64)
    gt = np.load(DATA_DIR / "IPgt.npy")
    candidate_bands = np.load(OUT_DIR / "selected_band_indices.npy")  # 164 pre-screened bands

    X_full = cube.reshape(-1, cube.shape[2])          # (21025, 200)
    y_full = gt.reshape(-1)                             # (21025,)
    mask = y_full != 0                                   # drop background
    X_lab = X_full[mask][:, candidate_bands]             # (10249, 164) -- restrict to candidates ONCE
    y_lab = y_full[mask]

    train_idx, test_idx = stratified_split(y_lab, TEST_FRACTION, RNG_SPLIT_SEED)
    X_train_raw, X_test_raw = X_lab[train_idx], X_lab[test_idx]
    y_train, y_test = y_lab[train_idx], y_lab[test_idx]

    # z-score standardize, stats fit on TRAIN ONLY, applied to both -- fixed once, shared
    mu = X_train_raw.mean(axis=0)
    sd = X_train_raw.std(axis=0)
    sd[sd == 0] = 1.0
    X_train = (X_train_raw - mu) / sd
    X_test = (X_test_raw - mu) / sd

    return {
        "candidate_bands": candidate_bands,
        "n_candidates": len(candidate_bands),
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "classes": np.unique(y_lab),
    }


class NearestCentroidFitness:
    """
    Fast surrogate classifier used as the SHARED fitness function for every
    algorithm during search. Nearest-centroid is ~2-3 orders of magnitude
    cheaper per evaluation than KNN/SVM (O(n_test * n_classes) distance
    computations instead of O(n_test * n_train)), which is what makes a
    600-evaluation budget x 6 algorithms x 5 seeds tractable. The FINAL
    best solution from each run is re-validated with KNN and SVM on the
    identical test split (see validate_subset) so reported accuracy
    numbers are not an artifact of the cheap surrogate.
    """
    def __init__(self, setup):
        self.X_train = setup["X_train"]
        self.y_train = setup["y_train"]
        self.X_test = setup["X_test"]
        self.y_test = setup["y_test"]
        self.classes = setup["classes"]
        self.n_bands = setup["n_candidates"]
        self.n_evals = 0

        # Precompute class centroids for ALL candidate bands once;
        # a subset's centroids are just a column slice -- O(1) per eval.
        self._centroids_full = np.vstack(
            [self.X_train[self.y_train == c].mean(axis=0) for c in self.classes]
        )  # (n_classes, n_bands)

    def reset_counter(self):
        self.n_evals = 0

    def __call__(self, bitmask):
        """bitmask: bool array of length n_bands. Returns scalar fitness (higher=better).
        Uses the expanded-norm form of squared Euclidean distance
        (||x||^2 - 2 x.C^T + ||c||^2) so the cross term is a BLAS matmul
        instead of a broadcasted elementwise subtraction -- ~15x faster,
        which is what makes a multi-thousand-evaluation budget tractable."""
        self.n_evals += 1
        k = int(bitmask.sum())
        if k == 0:
            return 0.0, 0.0  # (fitness, accuracy) -- empty subset is invalid
        cols = np.where(bitmask)[0]
        C = self._centroids_full[:, cols]       # (n_classes, k)
        Xte = self.X_test[:, cols]                # (n_test, k)
        x2 = np.einsum("ij,ij->i", Xte, Xte)[:, None]
        c2 = np.einsum("ij,ij->i", C, C)[None, :]
        d2 = x2 - 2.0 * (Xte @ C.T) + c2
        pred = self.classes[np.argmin(d2, axis=1)]
        acc = float((pred == self.y_test).mean())
        penalty = BAND_PENALTY_LAMBDA * (k / self.n_bands)
        fitness = acc - penalty
        return fitness, acc


def validate_subset_knn_svm(setup, bitmask, k_knn=5):
    """One-off, expensive, high-fidelity re-check of a FINAL best solution only."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    cols = np.where(bitmask)[0]
    Xtr, Xte = setup["X_train"][:, cols], setup["X_test"][:, cols]
    ytr, yte = setup["y_train"], setup["y_test"]

    knn = KNeighborsClassifier(n_neighbors=k_knn).fit(Xtr, ytr)
    knn_acc = float(knn.score(Xte, yte))

    svm = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr, ytr)
    svm_acc = float(svm.score(Xte, yte))

    return knn_acc, svm_acc
