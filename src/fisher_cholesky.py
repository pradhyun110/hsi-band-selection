"""
Fisher-Cholesky band selection -- the method chosen for this project.

Idea in one line: add bands one at a time, each time taking the band that most increases how well the
16 crop classes are separated, measured by the Fisher (LDA) criterion

    J(S) = trace( Sw^-1 Sb )      Sw = within-class scatter, Sb = between-class scatter of bands S,

computed on the LEARNING pixels only (no exam pixel is ever used).

Why "Cholesky": evaluating J for every candidate needs Sw^-1 of the chosen bands.  Instead of inverting
a k x k matrix for each candidate, we keep the Cholesky factor L of Sw (Sw = L L^T) and extend it by one
row per added band -- the same recursive update the base paper (MFE-CFSP) uses for its final step.
For a candidate band x:
    y   = L^-1 Sw[S, x]                 (forward substitution, k^2 operations)
    d^2 = Sw[x, x] - |y|^2              (new diagonal entry of L)
    J(S + x) = J(S) + sum_c n_c * ((dm_c[x] - y . u_c) / d)^2,   u_c = L^-1 dm_c[S]
so the cost per candidate is O(k^2) and the whole selection is O(n K^3) -- tiny for K = 15-20.
The result is identical to recomputing J from scratch (checked in the test of the project).

It is deterministic (same bands every run), needs no search budget, and uses only multiply-add,
one square root and one division per step: a good fit for FPGA hardware.
"""
import numpy as np


def fisher_select(cand, Xtr, ytr, K, ridge=0.01):
    """idea 1: forward selection maximising J = trace(Sw^-1 Sb) with a recursive Cholesky factor of Sw.
    J(S + x) = J(S) + sum_c n_c * ((dm_c[x] - y . u_c) / d)^2, with y = L^-1 Sw[S, x], d^2 = Sw[x,x] - |y|^2."""
    classes = np.unique(ytr); m = Xtr.mean(0)
    n = np.array([(ytr == c).sum() for c in classes]); DM = np.vstack([Xtr[ytr == c].mean(0) - m for c in classes])
    Xw = Xtr - (DM + m)[np.searchsorted(classes, ytr)]
    Sw = Xw.T @ Xw / len(ytr); Sw = Sw + ridge * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    cand = list(cand); chosen, rest = [], cand.copy()
    L = np.zeros((0, 0)); U = np.zeros((len(classes), 0))               # U[c] = L^-1 dm_c[S]
    while len(chosen) < K and rest:
        if chosen:
            Y = np.linalg.solve(L, Sw[np.ix_(chosen, rest)]); d2 = Sw[rest, rest] - (Y ** 2).sum(0)
            gain_num = DM[:, rest] - U @ Y
        else:
            Y = np.zeros((0, len(rest))); d2 = Sw[rest, rest]; gain_num = DM[:, rest]
        d = np.sqrt(np.maximum(d2, 1e-12)); gain = (n[:, None] * (gain_num / d) ** 2).sum(0)
        j = int(np.argmax(gain)); b = rest[j]
        L = np.block([[L, np.zeros((len(chosen), 1))], [Y[:, j][None, :], d[j:j + 1][None, :]]]) if chosen else d[j:j + 1][None, :]
        U = np.hstack([U, (gain_num[:, j] / d[j])[:, None]])
        chosen.append(b); rest.remove(b)
    return chosen
