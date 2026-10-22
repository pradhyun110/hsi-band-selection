"""
Eight label-free ("unsupervised") band selectors.  None of them looks at the class labels:
they only use the spectra of the TRAINING pixels, so they are fair to compare with the supervised search.

Every function has the same shape:   selector(Xraw, Xz, K) -> sorted array of K band positions

  Xraw : (pixels, bands) raw reflectance of the training pixels   (used by variance / entropy)
  Xz   : (pixels, bands) same pixels, each band standardised      (used by everything else)
  K    : how many bands to keep

In plain words
  variance        keep the K bands whose brightness varies the most from pixel to pixel
  entropy         keep the K bands that carry the most "information" (histogram entropy)
  kmeans          group similar bands into K clusters, keep the band nearest each cluster centre
  ward            same idea with hierarchical (Ward) clustering instead of k-means
  pca             take K principal components; keep the band each one relies on most
  greedy_r2       add one band at a time: always the band that best lets us rebuild ALL other bands
  spa             add one band at a time: always the band most different from those already chosen
                  (successive projections / orthogonal-subspace idea)
  density_peaks   keep bands that are both "typical" (many similar neighbours) and "distinct" from
                  other typical bands (E-FDPC style)
"""
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA


def _top(score, K):
    return np.sort(np.argsort(-score)[:K])


def _closest_to_centres(Xz, labels, K):
    """For every cluster keep the band whose spectrum is closest to the cluster mean."""
    B = Xz.T                                              # one row per band
    keep = []
    for c in range(K):
        members = np.flatnonzero(labels == c)
        centre = B[members].mean(0)
        keep.append(members[np.argmin(((B[members] - centre) ** 2).sum(1))])
    return np.sort(np.array(keep))


def sel_variance(Xraw, Xz, K):
    return _top(Xraw.var(0), K)


def sel_entropy(Xraw, Xz, K, bins=64):
    H = np.empty(Xraw.shape[1])
    for j in range(Xraw.shape[1]):
        p, _ = np.histogram(Xraw[:, j], bins=bins)
        p = p[p > 0] / p.sum()
        H[j] = -(p * np.log2(p)).sum()
    return _top(H, K)


def sel_kmeans(Xraw, Xz, K):
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xz.T)
    return _closest_to_centres(Xz, km.labels_, K)


def sel_ward(Xraw, Xz, K):
    lab = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Xz.T).labels_
    return _closest_to_centres(Xz, lab, K)


def sel_pca(Xraw, Xz, K):
    """One band per principal component: the band that component relies on most (not already taken)."""
    comps = PCA(n_components=K, random_state=0).fit(Xz).components_
    chosen = []
    for comp in comps:
        for i in np.argsort(-np.abs(comp)):
            if i not in chosen:
                chosen.append(int(i)); break
    return np.sort(np.array(chosen))


def sel_greedy_r2(Xraw, Xz, K):
    """Forward selection maximising how much of every band's variance the chosen bands explain."""
    R = np.corrcoef(Xz, rowvar=False)                      # residual correlation, updated each step
    chosen = []
    for _ in range(K):
        gain = (R ** 2).sum(0) / np.maximum(np.diag(R), 1e-12)
        gain[chosen] = -np.inf
        b = int(np.argmax(gain)); chosen.append(b)
        R = R - np.outer(R[:, b], R[b, :]) / R[b, b]       # remove what band b already explains
    return np.sort(np.array(chosen))


def sel_spa(Xraw, Xz, K):
    """Successive projections: start from the most unique band, then keep adding the band that is
    furthest from the span of the bands chosen so far."""
    B = Xz.T.copy()                                        # (bands, pixels)
    C = np.abs(np.corrcoef(Xz, rowvar=False))
    first = int(np.argmin(C.mean(0)))
    chosen = [first]
    P = B.copy()
    for _ in range(K - 1):
        v = P[chosen[-1]]
        P = P - np.outer(P @ v, v) / (v @ v)               # project out the last chosen direction
        norms = (P ** 2).sum(1); norms[chosen] = -np.inf
        chosen.append(int(np.argmax(norms)))
    return np.sort(np.array(chosen))


def sel_density_peaks(Xraw, Xz, K):
    D = 1.0 - np.abs(np.corrcoef(Xz, rowvar=False))       # band-to-band distance
    n = D.shape[0]
    dc = np.percentile(D[np.triu_indices(n, 1)], 2)        # neighbourhood size: closest 2 % of pairs
    rho = np.exp(-(D / max(dc, 1e-9)) ** 2).sum(1) - 1     # how many similar neighbours
    delta = np.empty(n)
    for i in range(n):
        higher = rho > rho[i]
        delta[i] = D[i, higher].min() if higher.any() else D[i].max()
    norm = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-12)
    return _top(norm(rho) * norm(delta), K)


SELECTORS = {
    "Variance": sel_variance,
    "Entropy": sel_entropy,
    "K-means": sel_kmeans,
    "Ward clustering": sel_ward,
    "PCA loadings": sel_pca,
    "Greedy R²": sel_greedy_r2,
    "Successive projections": sel_spa,
    "Density peaks": sel_density_peaks,
}
