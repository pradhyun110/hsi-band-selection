# Spectral band selection for Indian Pines: software simulation

Software study behind the project **Hardware Acceleration of Spectral Band Selection for Real-time Hyperspectral Image Processing** (course 22EC76, target board Nexys 4 / Artix-7 XC7A100T). This repository holds the Python simulation only; the FPGA/Verilog part is not in here yet.

## Headline result
Keeping **30 of the 200 bands (15%) matches the accuracy of all 200 bands** (about 83% overall accuracy, SVM-RBF, held-out test pixels). Twenty bands lose under 1 point against the raw cube.

| Bands kept | Supervised wrapper (mean of 90 runs) | Label-free search (U1, reconstruction R²) | Uniform spacing | Random subsets |
|---|---|---|---|---|
| 10 | 0.789 | 0.764 | 0.768 | 0.740 |
| 20 | 0.821 | 0.782 | 0.795 | 0.791 |
| 30 | 0.831 | 0.796 | 0.823 | 0.804 |
| 40 | 0.835 | 0.805 | 0.818 | 0.810 |

All 164 pre-screened bands: 0.841. All 200 raw bands: 0.830.

- The six search algorithms (GA, SA, DE, GWO, PSO, Jaya) are close to a statistical tie; the differences are mostly under 0.5 points at 20+ bands.
- Greedy forward selection (about 4 s) matches them at 15 to 25 bands.
- Two label-free search objectives did worse than the supervised wrapper by 2.5 to 7 points, and no better than uniform spacing.
- SVM-RBF is the most accurate classifier at small band counts; the linear models lose 8 to 9 points at 20 to 30 bands.
- The three label-free selectors from github.com/Zrahay/HyperSpectral_Image (variance, k-means, PCA loadings) score 0.63 to 0.69 at 30 bands on the raw cube. After our pre-screening k-means and PCA reach 0.79 and 0.77; top-variance stays at 0.66.

## Protocol (no leakage)
- 10,249 labelled pixels, 200 bands, 16 classes. Entropy + Otsu pre-screening keeps 164 candidate bands.
- Split: stratified 20% train / 80% test, seed 42.
- Search/fitness uses **train pixels only** (train cut into two halves, fitness = mean validation accuracy of a regularised LDA, both directions). Test pixels are used once, to score the final subset.
- Fixed band count K in {5, 10, 15, 20, 25, 30, 40, 50, 60}; every algorithm returns exactly K bands. Budget 3,000 fitness calls, population 30, 15 seeds per setting.
- Final scoring with SVM-RBF, KNN(5), LDA, Logistic Regression and Random Forest.

## Layout
- `data/` Indian Pines cube and ground truth (unmodified).
- `src/` current pipeline.
- `legacy_v1/` earlier experiments, **superseded** (fitness scored on test pixels, weak nearest-centroid fitness, free band count of about 80). Kept for the record; do not cite their rankings.
- `results/` tables, figures and raw search outputs (`results/sim/`).

## Run order (from the repository root)
```
pip install -r requirements.txt
python src/entropy_prescreen.py          # 200 -> 164 bands (reproducible)
python src/run_search.py main            # supervised-fitness search sweep (~12 min on 2 cores)
python src/run_search.py ablation        # nearest-centroid fitness at K=20
python src/baselines.py
python src/eval_subsets.py               # ~19 min
python src/analyze.py                    # tables + figures
python src/timing_clean.py 30
python src/run_search_unsup.py           # label-free objectives
python src/eval_unsup.py
python src/analyze_unsup.py
python src/compare_repo.py               # the three selectors from the GitHub repo above
```
Raw search outputs are already in `results/sim/`, so `analyze.py` and `analyze_unsup.py` run without repeating the searches.

## Caveats
- One outer split (seed 42). A random pixel split lets neighbouring pixels from the same field land in both train and test, which lifts every number equally.
- Hyper-parameters are literature defaults and were not tuned. Classifiers are untuned.
- The choice of K = 30 and of the consensus band set used test numbers (mild optimism); the pre-declared training-only rule gave K = 40.
- Only two label-free objectives were tried.
- FPGA cost figures discussed in the project notes are first-order estimates, not synthesis results.
