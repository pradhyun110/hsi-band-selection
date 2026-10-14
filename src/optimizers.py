"""
Seven fixed-cardinality band selectors.  EVERY algorithm must return exactly K bands.

Shared for fairness: candidate pool, folds, fitness callable, budget (= number of fitness calls,
initial population included), K, decoding rule (continuous algorithms), random initialisation.

  Continuous algorithms (PSO, GWO, Jaya, DE):
      real "random-key" vector y in [0,1]^n; mask = indices of the K largest keys; keys clipped to [0,1].
  Native subset algorithms (GA, SA, RandomSearch): operate directly on K-subsets (swap moves / fixed-size
      crossover) so the cardinality constraint is satisfied by construction.
Hyper-parameters are standard literature defaults and were NOT tuned on test data.
"""
import time
import numpy as np


class Run:
    """Counts fitness calls and keeps the best-so-far curve."""
    def __init__(self, fit, budget):
        self.fit, self.budget = fit, budget
        self.n = 0
        self.best_f, self.best_mask = -np.inf, None
        self.hist = np.zeros(budget, dtype=np.float32)

    def __call__(self, mask):
        f = self.fit(mask)
        if f > self.best_f:
            self.best_f, self.best_mask = f, mask.copy()
        self.hist[self.n] = self.best_f
        self.n += 1
        return f

    def done(self):
        return self.n >= self.budget


def decode(y, K):
    m = np.zeros(y.shape[0], dtype=bool)
    m[np.argpartition(-y, K - 1)[:K]] = True
    return m


def _finish(run, name, t0):
    return dict(algorithm=name, mask=run.best_mask, fitness=float(run.best_f), n_evals=run.n,
                hist=run.hist, time_s=time.perf_counter() - t0)


def random_subset(rng, n, K):
    m = np.zeros(n, dtype=bool); m[rng.choice(n, K, replace=False)] = True
    return m


# -------------------------------------------------------------------- Random search (sanity baseline)
def run_rs(fit, n, K, budget, pop, seed):
    t0 = time.perf_counter(); rng = np.random.default_rng(seed); run = Run(fit, budget)
    while not run.done():
        run(random_subset(rng, n, K))
    return _finish(run, "RS", t0)


# -------------------------------------------------------------------- GA (fixed-size subsets)
def run_ga(fit, n, K, budget, pop, seed, pc=0.8, tk=3):
    t0 = time.perf_counter(); rng = np.random.default_rng(seed); run = Run(fit, budget)
    P = [random_subset(rng, n, K) for _ in range(pop)]
    F = np.array([run(m) for m in P])

    def tour():
        i = rng.integers(0, pop, tk)
        return P[i[np.argmax(F[i])]]

    while not run.done():
        e = int(np.argmax(F))
        newP, newF = [P[e].copy()], [F[e]]                   # elitism, fitness carried (no re-eval)
        while len(newP) < pop and not run.done():
            a, b = tour(), tour()
            if rng.random() < pc:                            # fixed-size uniform crossover
                child = a & b
                pool = np.flatnonzero((a | b) & ~child)
                need = K - child.sum()
                child = child.copy()
                child[rng.choice(pool, need, replace=False)] = True
            else:
                child = a.copy()
            # swap mutation: each selected band is swapped for an unselected one w.p. 1/K (>=0 swaps)
            sel = np.flatnonzero(child)
            swaps = sel[rng.random(K) < 1.0 / K]
            if len(swaps):
                uns = np.flatnonzero(~child)
                child[swaps] = False
                child[rng.choice(uns, len(swaps), replace=False)] = True
            newP.append(child); newF.append(run(child))
        P, F = newP, np.array(newF)
        pop = len(P) if len(P) < pop else pop
    return _finish(run, "GA", t0)


# -------------------------------------------------------------------- SA (swap moves)
def run_sa(fit, n, K, budget, pop, seed, T0=0.02, Tmin=1e-4):
    t0 = time.perf_counter(); rng = np.random.default_rng(seed); run = Run(fit, budget)
    cur = random_subset(rng, n, K); fc = run(cur)
    while not run.done():
        T = T0 * (Tmin / T0) ** (run.n / budget)
        cand = cur.copy()
        r = 1 if rng.random() < 0.7 else 2                    # swap 1 (70%) or 2 (30%) bands
        out = rng.choice(np.flatnonzero(cand), r, replace=False)
        inn = rng.choice(np.flatnonzero(~cand), r, replace=False)
        cand[out] = False; cand[inn] = True
        f = run(cand)
        d = f - fc
        if d >= 0 or rng.random() < np.exp(d / T):
            cur, fc = cand, f
    return _finish(run, "SA", t0)


# -------------------------------------------------------------------- continuous methods on random keys
def _init_keys(rng, pop, n):
    return rng.random((pop, n))


def run_pso(fit, n, K, budget, pop, seed, c1=1.5, c2=1.5, w0=0.9, w1=0.4, vmax=0.2):
    t0 = time.perf_counter(); rng = np.random.default_rng(seed); run = Run(fit, budget)
    y = _init_keys(rng, pop, n); v = rng.uniform(-vmax, vmax, (pop, n))
    f = np.array([run(decode(y[i], K)) for i in range(pop)])
    pb, pf = y.copy(), f.copy()
    g = pb[np.argmax(pf)].copy(); gf = pf.max()
    iters = max(1, budget // pop - 1); it = 0
    while not run.done():
        w = w0 - (w0 - w1) * it / iters
        r1, r2 = rng.random((pop, n)), rng.random((pop, n))
        v = np.clip(w * v + c1 * r1 * (pb - y) + c2 * r2 * (g - y), -vmax, vmax)
        y = np.clip(y + v, 0, 1)
        for i in range(pop):
            if run.done():
                break
            fi = run(decode(y[i], K))
            if fi > pf[i]:
                pf[i], pb[i] = fi, y[i].copy()
        j = int(np.argmax(pf))
        if pf[j] > gf:
            gf, g = pf[j], pb[j].copy()                       # synchronous global-best update
        it += 1
    return _finish(run, "PSO", t0)


def run_gwo(fit, n, K, budget, pop, seed):
    t0 = time.perf_counter(); rng = np.random.default_rng(seed); run = Run(fit, budget)
    y = _init_keys(rng, pop, n)
    f = np.array([run(decode(y[i], K)) for i in range(pop)])
    iters = max(1, budget // pop - 1); it = 0
    while not run.done():
        o = np.argsort(-f)
        lead = [y[o[0]].copy(), y[o[1]].copy(), y[o[2]].copy()]
        a = 2.0 - 2.0 * it / iters
        for i in range(pop):
            if run.done():
                break
            Xs = []
            for L in lead:
                A = 2 * a * rng.random(n) - a; C = 2 * rng.random(n)
                Xs.append(L - A * np.abs(C * L - y[i]))
            y[i] = np.clip(sum(Xs) / 3.0, 0, 1)
            f[i] = run(decode(y[i], K))
        it += 1
    return _finish(run, "GWO", t0)


def run_jaya(fit, n, K, budget, pop, seed):
    t0 = time.perf_counter(); rng = np.random.default_rng(seed); run = Run(fit, budget)
    y = _init_keys(rng, pop, n)
    f = np.array([run(decode(y[i], K)) for i in range(pop)])
    while not run.done():
        yb, yw = y[np.argmax(f)].copy(), y[np.argmin(f)].copy()
        for i in range(pop):
            if run.done():
                break
            r1, r2 = rng.random(n), rng.random(n)
            yn = np.clip(y[i] + r1 * (yb - np.abs(y[i])) - r2 * (yw - np.abs(y[i])), 0, 1)
            fn = run(decode(yn, K))
            if fn > f[i]:
                y[i], f[i] = yn, fn
    return _finish(run, "Jaya", t0)


def run_de(fit, n, K, budget, pop, seed, F=0.5, CR=0.9):
    t0 = time.perf_counter(); rng = np.random.default_rng(seed); run = Run(fit, budget)
    y = _init_keys(rng, pop, n)
    f = np.array([run(decode(y[i], K)) for i in range(pop)])
    while not run.done():
        for i in range(pop):
            if run.done():
                break
            a, b, c = rng.choice([j for j in range(pop) if j != i], 3, replace=False)
            mut = np.clip(y[a] + F * (y[b] - y[c]), 0, 1)
            cross = rng.random(n) < CR
            cross[rng.integers(0, n)] = True
            tr = np.where(cross, mut, y[i])
            ft = run(decode(tr, K))
            if ft > f[i]:
                y[i], f[i] = tr, ft
    return _finish(run, "DE", t0)


ALGOS = {"GA": run_ga, "PSO": run_pso, "GWO": run_gwo, "Jaya": run_jaya, "DE": run_de, "SA": run_sa, "RS": run_rs}
