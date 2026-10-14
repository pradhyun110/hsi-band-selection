"""Tables and pictures for the 15-20 band study (reads results/small_k/results_long.csv)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "small_k"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, GREY, SURF, LIGHT = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8985", "#fcfcfb", "#c9c8c3"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GREY, "figure.facecolor": SURF})

df = pd.read_csv(OUT / "results_long.csv")
ref200 = df[df.method == "All 200 bands"].set_index("split"); ref164 = df[df.method == "All 164 pre-screened"].set_index("split")
SPL = sorted(df.split.unique()); KS = sorted(df[df.K < 164].K.unique())
MAIN = ["GA-LDA (best of starts)", "GA-LDA + SVM polish", "Greedy forward (LDA)", "Evenly spaced", "Greedy R2 (no labels)", "K-means (no labels)"]
LAB = {"GA-LDA (best of starts)": "Supervised GA", "GA-LDA + SVM polish": "Supervised GA + SVM polish", "Greedy forward (LDA)": "Supervised greedy forward",
       "Evenly spaced": "Evenly spaced", "Greedy R2 (no labels)": "Greedy R² (no labels)", "K-means (no labels)": "K-means (no labels)"}
COL = {"GA-LDA (best of starts)": BLUE, "GA-LDA + SVM polish": "#0b3d91", "Greedy forward (LDA)": AQUA, "Evenly spaced": INK2,
       "Greedy R2 (no labels)": ORANGE, "K-means (no labels)": YELLOW}

def style(ax, title=None, xl=None, yl=None):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=10)
    if xl: ax.set_xlabel(xl, color=INK2, fontsize=11)
    if yl: ax.set_ylabel(yl, color=INK2, fontsize=11)
    if title: ax.set_title(title, color=INK, fontsize=12.5, loc="left", fontweight="bold")

# ------------------------------------------------ tables
sub = df[df.method.isin(MAIN + ["GA-LDA (each start)"])].copy()
sub["d_def"] = sub.OA - sub.split.map(ref200.OA); sub["d_tun"] = sub.OA_tuned - sub.split.map(ref200.OA_tuned)
sub["d200_def_vs_tuned_ref"] = sub.OA - sub.split.map(ref200.OA)
g = sub.groupby(["method", "K"])
tab = g.agg(OA_mean=("OA", "mean"), OA_sd=("OA", "std"), tuned_mean=("OA_tuned", "mean"), tuned_sd=("OA_tuned", "std"),
            diff_default=("d_def", "mean"), diff_tuned=("d_tun", "mean"), AA=("AA", "mean"), Kappa=("Kappa", "mean"), n=("OA", "size")).reset_index()
tab.to_csv(OUT / "tbl_small_k_summary.csv", index=False)
refs = pd.DataFrame({"reference": ["All 200 bands", "All 164 pre-screened"],
                     "OA_default_mean": [ref200.OA.mean(), ref164.OA.mean()], "OA_default_sd": [ref200.OA.std(), ref164.OA.std()],
                     "OA_tuned_mean": [ref200.OA_tuned.mean(), ref164.OA_tuned.mean()], "OA_tuned_sd": [ref200.OA_tuned.std(), ref164.OA_tuned.std()]})
refs.to_csv(OUT / "tbl_small_k_references.csv", index=False)
print("splits:", SPL); print(refs.round(4).to_string(index=False))
t = tab[tab.method.isin(MAIN)]
for col in ("tuned_mean", "OA_mean"):
    print(f"\n== {col} (mean over {len(SPL)} splits) =="); print(t.pivot(index="method", columns="K", values=col).round(3).to_string())
print("\n== difference vs all-200 (tuned SVM, points; negative = worse) =="); print((t.pivot(index="method", columns="K", values="diff_tuned") * 100).round(1).to_string())
print("\n== difference vs all-200 (default SVM, points; negative = worse) =="); print((t.pivot(index="method", columns="K", values="diff_default") * 100).round(1).to_string())
ea = tab[tab.method == "GA-LDA (each start)"].set_index("K")
print("\nGA-LDA average of ALL starts (default SVM):", ea.OA_mean.round(3).to_dict())

# ------------------------------------------------ figure S1: accuracy vs K, two classifiers
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=False)
for ax, (col, sdcol, rcol, ttl) in zip(axes, (("tuned_mean", "tuned_sd", "OA_tuned", "Classifier with tuned settings"), ("OA_mean", "OA_sd", "OA", "Classifier with default settings"))):
    style(ax, ttl, "number of bands kept (out of 200)", "overall accuracy on test pixels (mean of %d splits)" % len(SPL))
    for m in MAIN:
        d = tab[(tab.method == m)].sort_values("K")
        lw = 3 if m.startswith("GA-LDA") else 1.8
        ax.plot(d.K.values[~d[col].isna()], d[col].dropna().values, color=COL[m], lw=lw, marker="o", ms=6, mec=SURF, mew=1.2, label=LAB[m], ls="--" if m == "Evenly spaced" else "-")
    r = ref200[rcol]; ax.axhspan(r.mean() - r.std(), r.mean() + r.std(), color=LIGHT, alpha=.5, lw=0)
    ax.axhline(r.mean(), color=INK, lw=1.2, ls=":"); ax.text(KS[0] - 0.15, r.mean() + 0.003, "all 200 bands", fontsize=10, color=INK2)
    ax.set_xticks(KS)
axes[0].legend(loc="lower right", frameon=False, fontsize=9.5)
fig.suptitle("Keeping only 15 to 20 bands: how much accuracy do we keep?", fontsize=14.5, fontweight="bold", color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "figS1_accuracy_15_to_20.png", dpi=140); plt.close(fig)

# ------------------------------------------------ figure S2: loss vs all-200 per split, K = 15 and 20
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
rng = np.random.default_rng(0)
for ax, K in zip(axes, (15, 20)):
    style(ax, f"{K} bands: accuracy compared with all 200 bands", "points vs all 200 bands (negative = worse)")
    ax.grid(False, axis="y"); ax.axvline(0, color=INK, lw=1)
    for i, m in enumerate(MAIN[::-1]):
        d = sub[(sub.method == m) & (sub.K == K)]
        if d.empty: continue
        ax.scatter(d.d_tun * 100, i + rng.uniform(-0.12, 0.12, len(d)), s=34, color=COL[m], edgecolor=SURF, lw=0.8, zorder=3)
        ax.plot([d.d_tun.mean() * 100] * 2, [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4)
        ax.text(1.0, i, f"{d.d_tun.mean()*100:+.1f}", va="center", ha="left", fontsize=10, color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(MAIN))); ax.set_yticklabels([LAB[m] for m in MAIN[::-1]], fontsize=10, color=INK)
    ax.set_ylim(-0.6, len(MAIN) - 0.4)
axes[1].set_yticklabels([])
fig.text(0.01, 0.01, "tuned classifier. each dot = one train/test split; black bar = average over splits; the polished set was only run for K = 15, 18, 20", fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 0.97, 1)); fig.savefig(OUT / "figS2_loss_vs_all200.png", dpi=140); plt.close(fig)
print("figures written")

# ------------------------------------------------ paired statistics over splits
def paired(a, b, K, col):
    A = df[(df.method == a) & (df.K == K)].set_index("split")[col]; B = df[(df.method == b) & (df.K == K)].set_index("split")[col]
    d = (A - B).dropna(); return d.mean() * 100, int((d > 0).sum()), len(d)
print("\n== paired over splits: mean difference in points, splits where first is better ==")
for K in (15, 18, 20):
    for a, b in (("GA-LDA + SVM polish", "GA-LDA (best of starts)"), ("GA-LDA (best of starts)", "Greedy forward (LDA)"), ("GA-LDA (best of starts)", "Evenly spaced"), ("GA-LDA (best of starts)", "Greedy R2 (no labels)")):
        for col in ("OA", "OA_tuned"):
            m, w, n = paired(a, b, K, col); print(f"K={K} {a} - {b} [{col}]: {m:+.2f} pts, better in {w}/{n}")
print("\nvs ALL 164 pre-screened (tuned):", {K: round((tab[(tab.method == "GA-LDA + SVM polish") & (tab.K == K)].tuned_mean.iloc[0] - ref164.OA_tuned.mean()) * 100, 1) for K in (15, 18, 20)})
print("vs ALL 164 pre-screened (default):", {K: round((tab[(tab.method == "GA-LDA + SVM polish") & (tab.K == K)].OA_mean.iloc[0] - ref164.OA.mean()) * 100, 1) for K in (15, 18, 20)})
print("AA/Kappa of polished sets:", tab[tab.method == "GA-LDA + SVM polish"][["K", "AA", "Kappa"]].round(3).to_dict("list"))
print("AA/Kappa all-200:", round(ref200.AA.mean(), 3), round(ref200.Kappa.mean(), 3))

# ------------------------------------------------ figure S3: maps for split 42, tuned SVM
from small_k import make_setup
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
S = make_setup(42); cand = S["candidate_bands"]; pos = {int(b): i for i, b in enumerate(cand)}
subs = json.load(open(OUT / "subsets_original_band_numbers.json"))["split=42"]
gt = np.load(ROOT / "data" / "IPgt.npy"); H, W = gt.shape; lab_pos = np.flatnonzero(gt.reshape(-1) != 0); y = gt.reshape(-1)[lab_pos]
from data_setup import stratified_split, TEST_FRACTION
tr, te = stratified_split(y, TEST_FRACTION, 42)
cube = np.load(ROOT / "data" / "indianpinearray.npy").astype(float).reshape(-1, 200)[lab_pos]
mu, sd = cube[tr].mean(0), cube[tr].std(0); sd[sd == 0] = 1; Z200 = (cube - mu) / sd
Xr = cube[:, cand]; m1, s1 = Xr[tr].mean(0), Xr[tr].std(0); s1[s1 == 0] = 1; Z164 = (Xr - m1) / s1
def tuned_predict(Xall):
    Xt = Xall[tr]; g0 = 1.0 / (Xt.shape[1] * Xt.var())
    gs = GridSearchCV(SVC(kernel="rbf"), {"C": [1, 10, 100, 1000], "gamma": [g0 / 4, g0, g0 * 4]}, cv=3, n_jobs=1).fit(Xt, y[tr]); return gs.predict(Xall)
def cols(name): return [pos[b] for b in subs[name]]
panels = [("All 200 bands", Z200), ("All 164 pre-screened bands", Z164),
          ("Supervised, 20 bands", Z164[:, cols("GA-LDA + SVM polish|K=20")]), ("Supervised, 15 bands", Z164[:, cols("GA-LDA + SVM polish|K=15")]),
          ("Label-free Greedy R², 15 bands", Z164[:, cols("Greedy R2 (no labels)|K=15")])]
CLASS_NAMES = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
tab20 = plt.cm.tab20(np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11])); CC = np.vstack([[0.96, 0.96, 0.94, 1.0], tab20]); CM = ListedColormap(CC)
def to_img(v): im = np.zeros(H * W, int); im[lab_pos] = v; return im.reshape(H, W)
is_test = np.zeros(len(y), bool); is_test[te] = True
fig, axes = plt.subplots(2, 6, figsize=(21, 8.4))
for a in axes.ravel(): a.set_xticks([]); a.set_yticks([]); [q.set_visible(False) for q in a.spines.values()]
axes[0, 0].imshow(to_img(y), cmap=CM, vmin=0, vmax=16, interpolation="nearest"); axes[0, 0].set_title("True land cover", fontsize=12, fontweight="bold", color=INK)
axes[1, 0].axis("off")
for j, (t, X) in enumerate(panels, start=1):
    p = tuned_predict(X); oa = (p[te] == y[te]).mean()
    axes[0, j].imshow(to_img(p), cmap=CM, vmin=0, vmax=16, interpolation="nearest"); axes[0, j].set_title(f"{t}\naccuracy {oa*100:.1f}%", fontsize=11, fontweight="bold", color=INK)
    err = np.where(~is_test, 1, np.where(p == y, 2, 3))
    axes[1, j].imshow(to_img(err), cmap=ListedColormap(["#f4f4f0", "#ecebe7", "#cfcec9", "#d03a2f"]), vmin=0, vmax=3, interpolation="nearest")
    axes[1, j].set_title(f"red = wrong ({(1-oa)*100:.1f}% of exam pixels)", fontsize=10, color=INK2)
    print(t, round(oa, 4))
fig.legend(handles=[Patch(color=CC[i + 1], label=n) for i, n in enumerate(CLASS_NAMES)], loc="lower center", ncol=8, frameon=False, fontsize=9.5)
fig.suptitle("Land-cover maps with 15 or 20 bands (one split, tuned classifier)", fontsize=15, fontweight="bold", color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.95), h_pad=5); fig.savefig(OUT / "figS3_maps_15_20.png", dpi=115); plt.close(fig)
print("maps written")

# ------------------------------------------------ recommended sets + HTML report
import base64
subs42 = json.load(open(OUT / "subsets_original_band_numbers.json"))["split=42"]
json.dump({f"{K}_bands": subs42[f"GA-LDA + SVM polish|K={K}"] for K in (15, 18, 20)}, open(OUT / "recommended_band_sets_original_numbers.json", "w"), indent=1)
def b64(n): return base64.b64encode(open(OUT / n, "rb").read()).decode()
def cell(m, K, c): 
    r = tab[(tab.method == m) & (tab.K == K)]; return "-" if r.empty or pd.isna(r[c].iloc[0]) else f"{r[c].iloc[0]*100:.1f}"
rows = "".join("<tr><td>%s</td>%s</tr>" % (LAB[m], "".join(f"<td>{cell(m,K,'tuned_mean')}</td><td>{cell(m,K,'OA_mean')}</td>" for K in (15, 18, 20))) for m in MAIN)
r2, r1 = refs.iloc[0], refs.iloc[1]
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Keeping 15-20 bands</title>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb;line-height:1.5}}h1{{font-size:26px}}h2{{margin-top:32px;font-size:20px}}
img{{width:100%;border:1px solid #e6e5e1;border-radius:6px}}table{{border-collapse:collapse;margin:10px 0;font-size:14px}}td,th{{border-bottom:1px solid #e6e5e1;padding:5px 10px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.box{{background:#eef4fc;border-left:4px solid #2a78d6;padding:10px 16px;margin:14px 0}}.small{{color:#52514e;font-size:14px}}</style></head><body>
<h1>Keeping only 15 to 20 of 200 bands</h1>
<div class="box"><b>Result.</b> With a supervised search, 15 to 20 bands (7.5 % to 10 % of the spectrum) score within about 1 point of all 200 bands,
averaged over {len(SPL)} different train/test splits. Compared with the cleaned 164-band set the loss is about 2 to 3 points.
Label-free methods lose 1.7 to 4.6 points against all 200 bands.</div>
<h2>1. Accuracy for 15 to 20 bands</h2><img src="data:image/png;base64,{b64('figS1_accuracy_15_to_20.png')}">
<p class="small">Grey band = all 200 bands (average of the splits, plus or minus one standard deviation). "Tuned" means the classifier's C and gamma were chosen by cross-validation on training pixels only.</p>
<table><tr><th rowspan=2>method</th><th colspan=2>15 bands</th><th colspan=2>18 bands</th><th colspan=2>20 bands</th></tr><tr><th>tuned</th><th>default</th><th>tuned</th><th>default</th><th>tuned</th><th>default</th></tr>{rows}
<tr><td>All 200 bands</td><td colspan=6 style="text-align:center">{r2.OA_tuned_mean*100:.1f} tuned / {r2.OA_default_mean*100:.1f} default</td></tr>
<tr><td>All 164 pre-screened bands</td><td colspan=6 style="text-align:center">{r1.OA_tuned_mean*100:.1f} tuned / {r1.OA_default_mean*100:.1f} default</td></tr></table>
<p class="small">Overall accuracy in %, mean of {len(SPL)} splits. The polished set was run for 15, 18 and 20 bands only.</p>
<h2>2. Is it luck of one split?</h2><img src="data:image/png;base64,{b64('figS2_loss_vs_all200.png')}">
<h2>3. What the maps look like</h2><img src="data:image/png;base64,{b64('figS3_maps_15_20.png')}">
<h2>4. Limits</h2><ul><li>The splits are random pixel splits, so neighbouring pixels from one field can be on both sides; this lifts all methods equally.</li>
<li>The polish step was tuned on training pixels only but adds just 0.3 to 1 point, which is small compared with split-to-split spread.</li>
<li>Band sets that were picked for the recommendation (the polished 15, 18 and 20 band sets of split 42) are in <code>recommended_band_sets_original_numbers.json</code>.</li></ul>
</body></html>"""
(OUT / "report_15_20.html").write_text(html, encoding="utf-8"); print("report written")
