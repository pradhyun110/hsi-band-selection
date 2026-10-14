"""
Six band-selection search algorithms, all sharing one contract:

    run(evaluator, n_bands, pop_size, max_iter, seed) -> result dict

Shared across every algorithm (fairness requirements):
  - identical evaluator (fitness function, dataset split) -- caller-supplied
  - identical fitness-evaluation BUDGET = pop_size * max_iter
  - identical binary search space (n_bands dimensions)
  - identical convergence-history bookkeeping (best-fitness-so-far per eval)

SA is the one single-solution method. To keep its evaluation BUDGET equal
to the population-based methods (the fair basis for comparison, not
"iterations"), it runs for pop_size*max_iter individual steps.

Continuous algorithms (PSO, GWO, Jaya, DE) maintain a real-valued working
position per dimension and binarize via a STOCHASTIC sigmoid transfer
(classic Kennedy & Eberhart BPSO-style: bit=1 if rand() < sigmoid(y) else 0)
-- the same transfer rule is reused for all four so no algorithm gets an
easier or harder binarization scheme than another. GA operates natively on
bits. SA operates natively on bits via bit-flip neighbourhoods (the
standard way SA is applied to combinatorial/binary problems).
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _binarize_stochastic(y, rng):
    return rng.random(y.shape[0]) < sigmoid(y)


def _ensure_nonempty(mask, rng):
    """Guard against the degenerate all-zero subset (undefined fitness)."""
    if not mask.any():
        mask = mask.copy()
        mask[rng.integers(0, len(mask))] = True
    return mask


class Recorder:
    """Tracks fitness-evaluation count and best-so-far history for a run."""
    def __init__(self, evaluator, budget):
        self.evaluator = evaluator
        self.budget = budget
        self.history = np.zeros(budget, dtype=np.float64)
        self.acc_history = np.zeros(budget, dtype=np.float64)
        self.best_fitness = -np.inf
        self.best_acc = 0.0
        self.best_mask = None
        self.n = 0

    def evaluate(self, mask):
        f, a = self.evaluator(mask)
        if f > self.best_fitness:
            self.best_fitness = f
            self.best_acc = a
            self.best_mask = mask.copy()
        self.history[self.n] = self.best_fitness
        self.acc_history[self.n] = self.best_acc
        self.n += 1
        return f, a

    def done(self):
        return self.n >= self.budget


# ---------------------------------------------------------------------
# Genetic Algorithm -- native binary representation
# ---------------------------------------------------------------------
def run_ga(evaluator, n_bands, pop_size, max_iter, seed,
           crossover_p=0.8, mutation_p=None, tournament_k=3):
    rng = np.random.default_rng(seed)
    if mutation_p is None:
        mutation_p = 1.0 / n_bands
    budget = pop_size * max_iter
    rec = Recorder(evaluator, budget)

    pop = rng.random((pop_size, n_bands)) < 0.3
    for i in range(pop_size):
        pop[i] = _ensure_nonempty(pop[i], rng)
    fitness = np.array([rec.evaluate(pop[i])[0] for i in range(pop_size)])

    def tournament():
        idx = rng.integers(0, pop_size, tournament_k)
        return pop[idx[np.argmax(fitness[idx])]].copy()

    while not rec.done():
        elite_idx = np.argmax(fitness)
        new_pop = [pop[elite_idx].copy()]        # elitism: keep best
        new_fitness = [fitness[elite_idx]]         # carried over, NOT re-evaluated (no extra eval cost)
        while len(new_pop) < pop_size and not rec.done():
            p1, p2 = tournament(), tournament()
            if rng.random() < crossover_p:
                cx_mask = rng.random(n_bands) < 0.5
                c1 = np.where(cx_mask, p1, p2)
            else:
                c1 = p1.copy()
            flip = rng.random(n_bands) < mutation_p
            c1 = np.where(flip, ~c1, c1)
            c1 = _ensure_nonempty(c1, rng)
            f1, _ = rec.evaluate(c1)
            new_pop.append(c1)
            new_fitness.append(f1)
        pop = np.array(new_pop)
        fitness = np.array(new_fitness)

    return _package(rec, "GA")


# ---------------------------------------------------------------------
# Binary PSO
# ---------------------------------------------------------------------
def run_pso(evaluator, n_bands, pop_size, max_iter, seed, w=0.7, c1=1.5, c2=1.5):
    rng = np.random.default_rng(seed)
    budget = pop_size * max_iter
    rec = Recorder(evaluator, budget)

    y = rng.uniform(-2, 2, (pop_size, n_bands))
    v = rng.uniform(-1, 1, (pop_size, n_bands))
    pos = np.array([_ensure_nonempty(_binarize_stochastic(y[i], rng), rng) for i in range(pop_size)])
    pbest = pos.copy()
    pbest_fit = np.array([rec.evaluate(pos[i])[0] for i in range(pop_size)])
    gbest = pbest[np.argmax(pbest_fit)].copy()
    gbest_fit = pbest_fit.max()

    while not rec.done():
        for i in range(pop_size):
            if rec.done():
                break
            r1, r2 = rng.random(n_bands), rng.random(n_bands)
            v[i] = w * v[i] + c1 * r1 * (pbest[i].astype(float) - y[i]) + c2 * r2 * (gbest.astype(float) - y[i])
            v[i] = np.clip(v[i], -4, 4)
            y[i] = y[i] + v[i]
            pos[i] = _ensure_nonempty(_binarize_stochastic(y[i], rng), rng)
            f, _ = rec.evaluate(pos[i])
            if f > pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i] = pos[i].copy()
                if f > gbest_fit:
                    gbest_fit = f
                    gbest = pos[i].copy()

    return _package(rec, "PSO")


# ---------------------------------------------------------------------
# Binary Grey Wolf Optimizer
# ---------------------------------------------------------------------
def run_gwo(evaluator, n_bands, pop_size, max_iter, seed):
    rng = np.random.default_rng(seed)
    budget = pop_size * max_iter
    rec = Recorder(evaluator, budget)

    y = rng.uniform(-2, 2, (pop_size, n_bands))
    pos = np.array([_ensure_nonempty(_binarize_stochastic(y[i], rng), rng) for i in range(pop_size)])
    fitness = np.array([rec.evaluate(pos[i])[0] for i in range(pop_size)])

    it = 0
    while not rec.done():
        order = np.argsort(-fitness)
        alpha, beta, delta = y[order[0]].copy(), y[order[1]].copy(), y[order[2]].copy()
        a = 2.0 - 2.0 * it / max_iter  # linearly decreases 2 -> 0

        for i in range(pop_size):
            if rec.done():
                break
            A1, C1 = 2 * a * rng.random(n_bands) - a, 2 * rng.random(n_bands)
            D_alpha = np.abs(C1 * alpha - y[i])
            X1 = alpha - A1 * D_alpha

            A2, C2 = 2 * a * rng.random(n_bands) - a, 2 * rng.random(n_bands)
            D_beta = np.abs(C2 * beta - y[i])
            X2 = beta - A2 * D_beta

            A3, C3 = 2 * a * rng.random(n_bands) - a, 2 * rng.random(n_bands)
            D_delta = np.abs(C3 * delta - y[i])
            X3 = delta - A3 * D_delta

            y[i] = (X1 + X2 + X3) / 3.0
            pos[i] = _ensure_nonempty(_binarize_stochastic(y[i], rng), rng)
            f, _ = rec.evaluate(pos[i])
            fitness[i] = f
        it += 1

    return _package(rec, "GWO")


# ---------------------------------------------------------------------
# Binary Jaya (parameter-free)
# ---------------------------------------------------------------------
def run_jaya(evaluator, n_bands, pop_size, max_iter, seed):
    rng = np.random.default_rng(seed)
    budget = pop_size * max_iter
    rec = Recorder(evaluator, budget)

    y = rng.uniform(-2, 2, (pop_size, n_bands))
    pos = np.array([_ensure_nonempty(_binarize_stochastic(y[i], rng), rng) for i in range(pop_size)])
    fitness = np.array([rec.evaluate(pos[i])[0] for i in range(pop_size)])

    while not rec.done():
        best_idx, worst_idx = np.argmax(fitness), np.argmin(fitness)
        y_best, y_worst = y[best_idx], y[worst_idx]
        for i in range(pop_size):
            if rec.done():
                break
            r1, r2 = rng.random(n_bands), rng.random(n_bands)
            y_new = y[i] + r1 * (y_best - np.abs(y[i])) - r2 * (y_worst - np.abs(y[i]))
            pos_new = _ensure_nonempty(_binarize_stochastic(y_new, rng), rng)
            f_new, _ = rec.evaluate(pos_new)
            if f_new > fitness[i]:  # Jaya's greedy accept
                y[i] = y_new
                pos[i] = pos_new
                fitness[i] = f_new

    return _package(rec, "Jaya")


# ---------------------------------------------------------------------
# Binary Differential Evolution
# ---------------------------------------------------------------------
def run_de(evaluator, n_bands, pop_size, max_iter, seed, F=0.5, CR=0.9):
    rng = np.random.default_rng(seed)
    budget = pop_size * max_iter
    rec = Recorder(evaluator, budget)

    y = rng.uniform(-2, 2, (pop_size, n_bands))
    pos = np.array([_ensure_nonempty(_binarize_stochastic(y[i], rng), rng) for i in range(pop_size)])
    fitness = np.array([rec.evaluate(pos[i])[0] for i in range(pop_size)])

    while not rec.done():
        for i in range(pop_size):
            if rec.done():
                break
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            mutant = y[r1] + F * (y[r2] - y[r3])
            cross = rng.random(n_bands) < CR
            if not cross.any():
                cross[rng.integers(0, n_bands)] = True
            trial_y = np.where(cross, mutant, y[i])
            trial_pos = _ensure_nonempty(_binarize_stochastic(trial_y, rng), rng)
            f_trial, _ = rec.evaluate(trial_pos)
            if f_trial > fitness[i]:  # greedy selection
                y[i] = trial_y
                pos[i] = trial_pos
                fitness[i] = f_trial

    return _package(rec, "DE")


# ---------------------------------------------------------------------
# Simulated Annealing -- single solution, native binary bit-flip moves
# ---------------------------------------------------------------------
def run_sa(evaluator, n_bands, pop_size, max_iter, seed, flip_frac=0.02):
    """Runs pop_size*max_iter individual steps so its fitness-evaluation
    BUDGET matches the population-based algorithms exactly."""
    rng = np.random.default_rng(seed)
    budget = pop_size * max_iter
    rec = Recorder(evaluator, budget)

    current = rng.random(n_bands) < 0.3
    current = _ensure_nonempty(current, rng)
    f_current, _ = rec.evaluate(current)

    T0, T_min = 1.0, 1e-3
    n_flip = max(1, int(round(n_bands * flip_frac)))

    while not rec.done():
        T = T0 * (T_min / T0) ** (rec.n / budget)
        candidate = current.copy()
        flip_idx = rng.choice(n_bands, n_flip, replace=False)
        candidate[flip_idx] = ~candidate[flip_idx]
        candidate = _ensure_nonempty(candidate, rng)
        f_cand, _ = rec.evaluate(candidate)
        delta = f_cand - f_current
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-12)):
            current, f_current = candidate, f_cand

    return _package(rec, "SA")


# ---------------------------------------------------------------------
def _package(rec, name):
    return {
        "algorithm": name,
        "best_mask": rec.best_mask,
        "best_fitness": rec.best_fitness,
        "best_acc": rec.best_acc,
        "history_fitness": rec.history,
        "history_acc": rec.acc_history,
        "n_evals": rec.n,
    }


ALGORITHMS = {
    "GA": run_ga,
    "PSO": run_pso,
    "GWO": run_gwo,
    "Jaya": run_jaya,
    "DE": run_de,
    "SA": run_sa,
}
