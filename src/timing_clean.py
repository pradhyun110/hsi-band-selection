"""
Clean (single-process, nothing else running) train / inference timing of the five classifiers on one
representative subset (the first SA seed-0 subset of the chosen K) plus the cost of one wrapper-fitness call.
Median of 5 repeats.  usage: python timing_clean.py <K>
"""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from eval_subsets import PANEL, data

K = int(sys.argv[1])
S = data("c164")
res = pickle.load(open(OUT / f"search_lda_K{K}.pkl", "rb"))
r = [x for x in res if x["alg"] == "SA" and x["seed"] == 0][0]
idx = r["idx"].astype(int)
Xtr, Xte = S["X_train"][:, idx], S["X_test"][:, idx]
rows = []
for name, mk in PANEL.items():
    tr, ti = [], []
    for _ in range(5):
        m = mk(); t = time.perf_counter(); m.fit(Xtr, S["y_train"]); tr.append(time.perf_counter() - t)
        t = time.perf_counter(); m.predict(Xte); ti.append(time.perf_counter() - t)
    rows.append(dict(classifier=name, K=K, train_s=np.median(tr), infer_s=np.median(ti),
                     infer_us_per_pixel=np.median(ti) / len(Xte) * 1e6))
fit = LDAFitness(load_setup()); fn = NCCFitness(load_setup())
masks = [mask_from_idx(np.random.default_rng(i).choice(164, K, replace=False), 164) for i in range(200)]
t = time.perf_counter(); [fit(m) for m in masks]; lda_ms = (time.perf_counter() - t) / 200 * 1e3
t = time.perf_counter(); [fn(m) for m in masks]; ncc_ms = (time.perf_counter() - t) / 200 * 1e3
df = pd.DataFrame(rows); df.to_csv(OUT / f"timing_clean_K{K}.csv", index=False)
print(df.round(5).to_string(index=False)); print(f"fitness call: LDA {lda_ms:.3f} ms, NCC {ncc_ms:.3f} ms (K={K})")
