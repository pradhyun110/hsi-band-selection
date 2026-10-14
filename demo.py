"""
demo.py  --  one command, about 5 minutes:   python demo.py

Compares, on Indian Pines, ONE supervised band-selection method (a genetic algorithm that is allowed to
see class labels of the TRAINING pixels) against EIGHT unsupervised methods (no labels at all), and
writes pictures + tables + an HTML report you can open in a browser:

        results/demo/report.html          <-- open this one
        results/demo/fig*.png             <-- the pictures
        results/demo/*.csv / *.json       <-- the numbers

Fair-test rules (same for every method)
  * the same 20 % / 80 % train / test split of the labelled pixels (seed 42)
  * every method chooses bands using TRAINING pixels only; the 80 % test pixels are used only to grade
  * the same classifier (SVM with RBF kernel) grades every chosen band set
  * every method must return exactly K bands

Options:  python demo.py --quick     (fewer K values / seeds, about 1 minute)
"""
import sys, json, time, base64, argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data_setup import stratified_split, RNG_SPLIT_SEED, TEST_FRACTION       # the agreed split
from common import load_setup, LDAFitness                                     # candidate bands + wrapper fitness
from optimizers import run_ga                                                 # the supervised search
from unsupervised_selectors import SELECTORS                                  # the 8 label-free methods
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
KS = [10, 20, 30, 40] if args.quick else [5, 10, 15, 20, 25, 30, 40, 50, 60]
GA_SEEDS = 2 if args.quick else 5
HEAD_K = 30                                  # the band count we show in the headline picture
OUTD = ROOT / "results" / "demo"; OUTD.mkdir(parents=True, exist_ok=True)

# palette (validated set used in the other figures of this project)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, GREY, SURF, LIGHT = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8985", "#fcfcfb", "#c9c8c3"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GREY, "figure.facecolor": SURF})

CLASS_NAMES = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
               "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
               "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
_tab = plt.cm.tab20(np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11]))
CLASS_COLORS = np.vstack([[0.96, 0.96, 0.94, 1.0], _tab])                    # index 0 = unlabelled background
CMAP = ListedColormap(CLASS_COLORS)

# ------------------------------------------------------------------ 1. data and split
t_start = time.time()
S = load_setup()                                   # z-scored 164 candidate bands, train / test (seed 42)
cand = S["candidate_bands"]                        # original band numbers of the 164 candidates
cube = np.load(ROOT / "data" / "indianpinearray.npy").astype(np.float64)
gt = np.load(ROOT / "data" / "IPgt.npy")
H, W, NB = cube.shape
flat_y = gt.reshape(-1); lab_pos = np.flatnonzero(flat_y != 0)
X200 = cube.reshape(-1, NB)[lab_pos]; y = flat_y[lab_pos]
train_i, test_i = stratified_split(y, TEST_FRACTION, RNG_SPLIT_SEED)
mu200, sd200 = X200[train_i].mean(0), X200[train_i].std(0); sd200[sd200 == 0] = 1
Z200 = (X200 - mu200) / sd200                                                 # all 200 bands, z-scored with train stats
Xraw_train = X200[train_i][:, cand]                                           # raw reflectance, candidates only
assert np.allclose(S["y_train"], y[train_i]) and np.allclose(S["X_test"], ((X200[test_i][:, cand] - Xraw_train.mean(0)) / np.where(Xraw_train.std(0) == 0, 1, Xraw_train.std(0))))
Xtr, Xte, ytr, yte = S["X_train"], S["X_test"], S["y_train"], S["y_test"]
Z164_all = ((X200[:, cand] - Xraw_train.mean(0)) / np.where(Xraw_train.std(0) == 0, 1, Xraw_train.std(0)))
print(f"pixels: {len(y)} labelled, {len(train_i)} train / {len(test_i)} test, {len(cand)} candidate bands")

# ------------------------------------------------------------------ 2. choose bands: supervised GA
cache = OUTD / "ga_subsets.json"
ga_cache = json.load(open(cache)) if cache.exists() else {}
fit = LDAFitness(S)                                # accuracy of a small LDA on held-out TRAIN pixels
subsets = {}                                       # (method, K) -> sorted positions in the 164 candidates
for K in KS:
    key = f"{K}_{GA_SEEDS}"
    if key not in ga_cache:
        runs = [run_ga(fit, 164, K, 3000, 30, 1000 * K + s) for s in range(GA_SEEDS)]
        best = max(runs, key=lambda r: r["fitness"])                           # best TRAIN-validation fitness, never test
        ga_cache[key] = [int(i) for i in np.flatnonzero(best["mask"])]
        print(f"  GA  K={K:2d}  best validation fitness {best['fitness']:.3f}", flush=True)
    subsets[("Supervised (GA + labels)", K)] = np.array(ga_cache[key])
json.dump(ga_cache, open(cache, "w"))

# ------------------------------------------------------------------ 3. choose bands: 8 unsupervised methods
for name, f in SELECTORS.items():
    for K in KS:
        subsets[(name, K)] = f(Xraw_train, Xtr, K)
for K in KS:                                       # a trivial reference: evenly spaced bands
    subsets[("Evenly spaced", K)] = np.unique(np.round(np.linspace(0, 163, K)).astype(int))
    assert len(subsets[("Evenly spaced", K)]) == K
for (m, K), idx in subsets.items():
    assert len(set(idx)) == K, (m, K)

# ------------------------------------------------------------------ 4. grade every band set on the test pixels
def metrics(p, t):
    cm = confusion_matrix(t, p)
    aa = np.nanmean(np.diag(cm) / cm.sum(1))
    return (p == t).mean(), aa, cohen_kappa_score(t, p)

rows = []
for (m, K), idx in subsets.items():
    clf = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr[:, idx], ytr)
    oa, aa, kp = metrics(clf.predict(Xte[:, idx]), yte)
    rows.append(dict(method=m, K=K, OA=oa, AA=aa, Kappa=kp))
clf = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr, ytr); oa164 = (clf.predict(Xte) == yte).mean()
clf = SVC(kernel="rbf", C=10, gamma="scale").fit(Z200[train_i], ytr); oa200 = (clf.predict(Z200[test_i]) == yte).mean()
res = pd.DataFrame(rows); res.to_csv(OUTD / "results_svm.csv", index=False)
print(f"all 200 bands: {oa200:.3f}   all 164 pre-screened: {oa164:.3f}")

head = res[res.K == HEAD_K].sort_values("OA", ascending=False).reset_index(drop=True)
print("\nOverall accuracy at K=%d (SVM, test pixels):" % HEAD_K); print(head.round(3).to_string(index=False))

# five different classifiers at K = HEAD_K (does the conclusion depend on the classifier?)
CLFS = {"SVM": lambda: SVC(kernel="rbf", C=10, gamma="scale"), "KNN": lambda: KNeighborsClassifier(5),
        "LDA": lambda: LDA(), "LogReg": lambda: LogisticRegression(max_iter=500),
        "RandomForest": lambda: RandomForestClassifier(100, random_state=0, n_jobs=-1)}
crow = []
for (m, K), idx in subsets.items():
    if K != HEAD_K: continue
    r = dict(method=m)
    for cn, mk in CLFS.items():
        r[cn] = (mk().fit(Xtr[:, idx], ytr).predict(Xte[:, idx]) == yte).mean()
    crow.append(r)
cls_tab = pd.DataFrame(crow).sort_values("SVM", ascending=False); cls_tab.to_csv(OUTD / f"results_K{HEAD_K}_all_classifiers.csv", index=False)

json.dump({f"{m} | K={K}": [int(cand[i]) for i in idx] for (m, K), idx in subsets.items()},
          open(OUTD / "selected_bands_original_numbers.json", "w"), indent=1)

SUP = "Supervised (GA + labels)"
unsup_names = list(SELECTORS)
best_un = head[head.method.isin(unsup_names)].iloc[0]; worst_un = head[head.method.isin(unsup_names)].iloc[-1]
sup_oa = float(head[head.method == SUP].OA.iloc[0])

# ------------------------------------------------------------------ 5. pictures
def style(ax, title=None, xl=None, yl=None):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=10)
    if xl: ax.set_xlabel(xl, color=INK2, fontsize=11)
    if yl: ax.set_ylabel(yl, color=INK2, fontsize=11)
    if title: ax.set_title(title, color=INK, fontsize=13, loc="left", fontweight="bold")

# --- figure 1: who wins at K = 30 (bar chart)
items = [(r.method, r.OA) for r in head.itertuples()]
items.append((f"All 200 bands (no selection)", oa200))
items = sorted(items, key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(10, 5.6)); style(ax, f"Keeping only {HEAD_K} of 200 bands: how well does each method do?", "overall accuracy on test pixels (SVM classifier)")
ax.grid(False, axis="y")
for i, (m, o) in enumerate(items):
    col = BLUE if m == SUP else (GREY if m.startswith("All 200") or m == "Evenly spaced" else ORANGE)
    ax.barh(i, o, color=col, height=0.62); ax.text(o + 0.004, i, f"{o*100:.1f}%", va="center", fontsize=10.5, color=INK)
ax.set_yticks(range(len(items))); ax.set_yticklabels([m for m, _ in items], fontsize=10.5, color=INK)
ax.set_xlim(0.5, 0.9); ax.axvline(oa200, color=INK2, lw=1, ls="--")
ax.legend(handles=[Patch(color=BLUE, label="uses class labels (supervised)"), Patch(color=ORANGE, label="no labels (unsupervised)"),
                   Patch(color=GREY, label="reference")], loc="lower right", frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig(OUTD / "fig1_ranking.png", dpi=140); plt.close(fig)

# --- figure 2: accuracy as the number of kept bands grows
fig, ax = plt.subplots(figsize=(10, 5.6)); style(ax, "Accuracy as we keep more bands", "number of bands kept (out of 200)", "overall accuracy on test pixels")
top3 = res[(res.K == HEAD_K) & res.method.isin(unsup_names)].sort_values("OA", ascending=False).method.tolist()[:3]
for m in unsup_names:
    if m in top3: continue
    d = res[res.method == m].sort_values("K"); ax.plot(d.K, d.OA, color=LIGHT, lw=1.5, zorder=1)
ax.plot([], [], color=LIGHT, lw=1.5, label="other label-free methods")
for m, c in zip(top3, (ORANGE, AQUA, YELLOW)):
    d = res[res.method == m].sort_values("K"); ax.plot(d.K, d.OA, color=c, lw=2.2, marker="o", ms=5, mec=SURF, mew=1.2, label=m + " (no labels)", zorder=3)
d = res[res.method == "Evenly spaced"].sort_values("K"); ax.plot(d.K, d.OA, color=INK2, lw=1.5, ls="--", label="Evenly spaced", zorder=2)
d = res[res.method == SUP].sort_values("K"); ax.plot(d.K, d.OA, color=BLUE, lw=3, marker="o", ms=6, mec=SURF, mew=1.4, label="Supervised (uses labels)", zorder=4)
ax.axhline(oa200, color=INK, lw=1, ls=":"); ax.text(KS[0] - 1.5, oa200 + 0.004, "all 200 bands", ha="left", fontsize=10, color=INK2)
ax.set_xticks(KS); ax.set_ylim(0.45, 0.9); ax.legend(loc="lower right", frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig(OUTD / "fig2_accuracy_vs_bands.png", dpi=140); plt.close(fig)

# --- figure 3: classification maps
def fit_predict_all(idx164=None):
    if idx164 is None:                                                 # all 200 bands
        clf = SVC(kernel="rbf", C=10, gamma="scale").fit(Z200[train_i], ytr); return clf.predict(Z200)
    clf = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr[:, idx164], ytr); return clf.predict(Z164_all[:, idx164])

panels = [("All 200 bands", None, oa200),
          (f"Supervised, {HEAD_K} bands", subsets[(SUP, HEAD_K)], sup_oa),
          (f"Best label-free: {best_un.method}, {HEAD_K} bands", subsets[(best_un.method, HEAD_K)], float(best_un.OA)),
          (f"Worst label-free: {worst_un.method}, {HEAD_K} bands", subsets[(worst_un.method, HEAD_K)], float(worst_un.OA))]
def to_img(values):
    img = np.zeros(H * W, dtype=int); img[lab_pos] = values; return img.reshape(H, W)
is_test = np.zeros(len(y), bool); is_test[test_i] = True
fig, axes = plt.subplots(2, 5, figsize=(19, 9.6))
for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
axes[0, 0].imshow(to_img(y), cmap=CMAP, vmin=0, vmax=16, interpolation="nearest"); axes[0, 0].set_title("True land cover\n(ground truth)", fontsize=12, color=INK, fontweight="bold")
split_img = to_img(np.where(is_test, 2, 1)); axes[1, 0].imshow(split_img, cmap=ListedColormap(["#f4f4f0", "#2a78d6", "#d9d8d4"]), vmin=0, vmax=2, interpolation="nearest")
axes[1, 0].set_title("blue = learning (20 %), grey = exam (80 %)", fontsize=10.5, color=INK2)
for j, (title, idx, oa) in enumerate(panels, start=1):
    pred = fit_predict_all(idx)
    axes[0, j].imshow(to_img(pred), cmap=CMAP, vmin=0, vmax=16, interpolation="nearest")
    axes[0, j].set_title(f"{title}\naccuracy {oa*100:.1f}%", fontsize=11, color=INK, fontweight="bold")
    err = np.where(~is_test, 1, np.where(pred == y, 2, 3))                  # 1 not graded, 2 right, 3 wrong
    axes[1, j].imshow(to_img(err), cmap=ListedColormap(["#f4f4f0", "#ecebe7", "#cfcec9", "#d03a2f"]), vmin=0, vmax=3, interpolation="nearest")
    axes[1, j].set_title(f"red = wrong ({(1-oa)*100:.1f}% of exam pixels)", fontsize=10.5, color=INK2)
fig.legend(handles=[Patch(color=CLASS_COLORS[i + 1], label=n) for i, n in enumerate(CLASS_NAMES)], loc="lower center", ncol=8, frameon=False, fontsize=9.5)
fig.suptitle("Land-cover maps made by each band set (same classifier, same learning pixels)", fontsize=15, fontweight="bold", color=INK, x=0.012, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.95), h_pad=5); fig.savefig(OUTD / "fig3_maps.png", dpi=120); plt.close(fig)

# --- figure 4: where on the spectrum are the chosen bands?
order = [SUP] + [m for m in head.method if m in unsup_names]
fig, ax = plt.subplots(figsize=(11, 5.6)); style(ax, f"Which {HEAD_K} bands does each method keep?", "band number (spectrum from blue/visible light on the left to infrared on the right)")
ax.grid(False)
dropped = np.setdiff1d(np.arange(NB), cand)
for b in dropped: ax.axvspan(b - 0.5, b + 0.5, color="#ecebe7", lw=0)
for i, m in enumerate(order[::-1]):
    bands = cand[subsets[(m, HEAD_K)]]
    ax.vlines(bands, i - 0.34, i + 0.34, color=BLUE if m == SUP else ORANGE, lw=2.4)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order[::-1], fontsize=10.5, color=INK); ax.set_xlim(-1, NB)
ax.text(0.995, 1.01, "grey strips = bands removed by the first (entropy) screening", transform=ax.transAxes, ha="right", fontsize=9.5, color=INK2)
fig.tight_layout(); fig.savefig(OUTD / "fig4_selected_bands.png", dpi=140); plt.close(fig)

# ------------------------------------------------------------------ 6. HTML report
def b64(p): return base64.b64encode(open(OUTD / p, "rb").read()).decode()
best_rows = head.copy()
tbl = "".join(f"<tr><td>{r.method}</td><td>{r.OA*100:.1f}%</td><td>{r.AA*100:.1f}%</td><td>{r.Kappa:.3f}</td></tr>" for r in best_rows.itertuples())
cmp_tbl = "".join("<tr><td>%s</td>%s</tr>" % (r["method"], "".join(f"<td>{r[c]*100:.1f}%</td>" for c in CLFS)) for _, r in cls_tab.iterrows())
by_k = res.pivot(index="method", columns="K", values="OA").loc[[SUP] + unsup_names + ["Evenly spaced"]]
k_tbl = "".join("<tr><td>%s</td>%s</tr>" % (m, "".join(f"<td>{v*100:.1f}</td>" for v in by_k.loc[m])) for m in by_k.index)
n_below = int((head[head.method.isin(unsup_names)].OA < sup_oa).sum())
gap = (sup_oa - float(best_un.OA)) * 100
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Band selection: supervised vs unsupervised</title>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb;line-height:1.5}}
h1{{font-size:26px}}h2{{margin-top:34px;font-size:20px}}img{{width:100%;border:1px solid #e6e5e1;border-radius:6px}}
table{{border-collapse:collapse;margin:10px 0;font-size:14px}}td,th{{border-bottom:1px solid #e6e5e1;padding:5px 10px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.box{{background:#eef4fc;border-left:4px solid #2a78d6;padding:10px 16px;margin:14px 0}}.small{{color:#52514e;font-size:14px}}</style></head><body>
<h1>Choosing the best spectral bands for hyperspectral images</h1>
<p class="small">Indian Pines scene, 10,249 labelled pixels, 200 bands, 16 land-cover classes. Everything below is produced by <code>python demo.py</code>.</p>
<div class="box"><b>The idea in three sentences.</b> A hyperspectral camera records 200 colours per pixel, but many neighbouring colours say nearly the same thing.
We pick a small set of bands that keeps almost all the useful information, so the hardware has less data to process. A good selection
should give about the same land-cover accuracy as using all 200 bands.</div>
<h2>1. The result</h2>
<div class="box"><b>With {HEAD_K} of 200 bands (15 %), the supervised method scores {sup_oa*100:.1f}%</b>, about the same as the {oa200*100:.1f}% from all 200 bands (the best of {GA_SEEDS} searches, picked using training pixels only).
The best label-free method ({best_un.method}) reaches {best_un.OA*100:.1f}%, which is {gap:.1f} points lower; {n_below} of the 8 label-free methods fall below the supervised one,
and the weakest ({worst_un.method}) gets {worst_un.OA*100:.1f}%.</div>
<img src="data:image/png;base64,{b64('fig1_ranking.png')}">
<p class="small">Supervised = the search is allowed to look at the class labels of the learning pixels. Unsupervised = it never sees labels, only the spectra.</p>
<h2>2. What happens when we keep more or fewer bands</h2>
<img src="data:image/png;base64,{b64('fig2_accuracy_vs_bands.png')}">
<table><tr><th>method \\ bands kept (accuracy %)</th>{''.join(f'<th>{k}</th>' for k in KS)}</tr>{k_tbl}</table>
<h2>3. What the maps look like</h2>
<img src="data:image/png;base64,{b64('fig3_maps.png')}">
<p class="small">Top row: land-cover maps. Bottom row: the red pixels are the mistakes, counted only on the 80 % of pixels the classifier never learned from.
The picture also paints the 20 % learning pixels, but the accuracy figures never count them.</p>
<h2>4. Which bands were chosen</h2>
<img src="data:image/png;base64,{b64('fig4_selected_bands.png')}">
<h2>5. Exact numbers at {HEAD_K} bands (SVM classifier)</h2>
<table><tr><th>method</th><th>overall accuracy</th><th>average per-class</th><th>kappa</th></tr>{tbl}</table>
<p class="small">Does the answer depend on the classifier? The same {HEAD_K}-band sets graded by five classifiers (overall accuracy):</p>
<table><tr><th>method</th>{''.join(f'<th>{c}</th>' for c in CLFS)}</tr>{cmp_tbl}</table>
<h2>6. How to explain the fairness</h2>
<ul><li>Same split for all: 20 % of pixels to learn from, 80 % kept hidden as the exam.</li>
<li>All methods pick bands using only the learning pixels. The exam pixels are used once, to grade.</li>
<li>Same classifier, same number of bands for every method.</li>
<li>Honest limits: one train/test split, nearby pixels can fall on both sides of the split, classifiers are not tuned, and the supervised search is one algorithm (GA) with {GA_SEEDS} random starts. The full study (90 runs per setting, six algorithms) averages 83.1% at 30 bands, so treat differences of about one point as noise.</li></ul>
</body></html>"""
(OUTD / "report.html").write_text(html, encoding="utf-8")
print(f"\nDone in {time.time()-t_start:.0f}s.  Open  {OUTD/'report.html'}")
