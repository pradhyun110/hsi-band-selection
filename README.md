# Spectral band selection for real-time hyperspectral image processing (Indian Pines)

Software study for the project *Hardware Acceleration of Spectral Band Selection for Real-time Hyperspectral Image Processing* (22EC76, Dayananda Sagar College of Engineering). The goal is to keep **15–20 of the 200 bands** while keeping classification accuracy close to using all bands, with a selection method cheap enough for an FPGA.

**Result:** the chosen method is **Fisher-Cholesky with K = 18 bands**: **82.1 % overall accuracy** (all 200 bands: 83.4 %), selected in about **0.01 s**. The genetic algorithm reaches 82.9–83.2 % but needs about 10 s, and the base paper's method (MFE-CFSP) reaches 74.7 %.

**Why 15–20 bands:** accuracy rises steeply up to about 15 bands and then only slowly (`fig17`). It is a bend in the curve, not a flat line: Fisher-Cholesky gains 2.7 more points going from 20 to 100 bands (82.1 % to 84.7 %; all 164 pre-screened bands score 84.8 %). At 18 bands it keeps 97 % of the plateau accuracy (mean of 60, 80, 100 bands) with 11 % of the 164 candidate bands. It is within 3 points of the plateau from 15 bands, within 2 from 30 and within 1 from 50 (`results/final/table_flatline.csv`). The 15–20 choice is a trade-off between data volume and accuracy, not a point where nothing more can be gained.

## Pipeline

```
200 bands --> entropy pre-screening (Otsu threshold) --> 164 bands --> band selection (K bands) --> SVM classifier
```

1. **Pre-screening** (`src/entropy_prescreen.py`). Compute the entropy of every band; the Otsu threshold (6.34 bits, no hand-picked parameter) removes 36 noisy, low-information bands.
2. **Band selection** (`src/final_comparison.py`). The 8 methods compared are listed below.
3. **Grading** (`src/metrics.py`). An SVM is trained on 20 % of the labelled pixels (the *learning* pixels) using only the chosen bands. It is then scored on the other 80 % (the *exam* pixels), which no method ever sees while choosing bands.

| Method | Uses labels? | Idea |
|---|---|---|
| **Fisher-Cholesky (chosen)** | yes (learning pixels) | Add bands one at a time, each time the band that most increases class separation (Fisher/LDA criterion). A recursive Cholesky update keeps every step cheap. |
| GA | yes | Genetic algorithm. Its fitness is the accuracy of an LDA classifier on learning pixels. |
| Hybrid (GA + Fisher shortlist) | yes | The GA searches only the 2K bands that Fisher-Cholesky ranks first. |
| Jaya, Grey wolf (GWO) | yes | Other metaheuristics with the same fitness and budget as the GA. |
| Base paper (MFE-CFSP) | no | Our re-implementation of Hao et al., IEEE JSTARS 2026. |
| Entropy, Variance | no | Keep the K bands with the highest entropy or variance. |

## Results at 20 bands (mean ± std over 6 splits; full tables: `results/final/table_K20.csv`, `table_K18.csv`)

At 18 bands (recommended K), Fisher-Cholesky scores 82.1 % OA, 75.6 % AA and 0.795 kappa; the GA scores 83.2 %, 75.5 % and 0.807.


| Method | OA (%) | AA (%) | Kappa | F1 (%) | CEM AUC | Selection time |
|---|---|---|---|---|---|---|
| **Fisher-Cholesky (chosen)** | **82.1 ± 0.3** | 75.2 ± 1.5 | 0.794 | 77.5 | 0.940 | **0.011 s** |
| GA | 82.9 ± 0.8 | 75.4 ± 3.2 | 0.804 | 78.4 | 0.942 | 10.5 s |
| Hybrid (GA + Fisher shortlist) | 83.2 ± 0.3 | 76.6 ± 1.3 | 0.808 | 79.3 | 0.941 | 10.2 s |
| Jaya | 82.1 ± 0.8 | 76.4 ± 3.3 | 0.795 | 79.0 | 0.938 | 9.4 s |
| Grey wolf (GWO) | 82.5 ± 0.7 | 74.9 ± 3.9 | 0.799 | 77.5 | 0.942 | 10.2 s |
| Base paper (MFE-CFSP) | 74.7 ± 0.4 | 67.8 ± 1.1 | 0.708 | 69.8 | 0.901 | 0.52 s |
| Entropy | 61.3 ± 5.9 | 45.0 ± 7.4 | 0.545 | 44.8 | 0.897 | 0.018 s |
| Variance | 62.0 ± 0.6 | 48.2 ± 0.6 | 0.552 | 48.1 | 0.867 | 0.003 s |
| *All 164 pre-screened bands* | 84.8 | 77.6 | 0.825 | 79.3 | 0.956 | – |
| *All 200 bands* | 83.4 | 77.2 | 0.810 | 79.8 | 0.955 | – |

**OA** is overall accuracy: the share of exam pixels labelled correctly. **AA** is average accuracy: the accuracy of each crop, averaged, so small crops count equally (it equals the mean sensitivity/recall). **Kappa** is accuracy corrected for chance. **F1** balances precision and recall (mean over crops). **CEM AUC** measures target detection (a base-paper metric). Precision, specificity, MCC and SSIM are in the full table. Selection time is the median over splits; GA, Jaya, GWO and Hybrid include 5 restarts of 3,000 fitness calls each.

## Figures (`results/final/`)

| Figure | Shows |
|---|---|
| `fig1_dataset.png` | The image, the 16-crop ground truth, the mean spectrum of each crop |
| `fig1b_learning_exam_split.png` | What a split is: 20 % of each crop's pixels for learning, 80 % for the exam |
| `fig2_prescreening.png` | Entropy of each band, threshold rules compared, bands removed |
| `fig3_prescreening_effect.png` | Fisher-Cholesky with vs without pre-screening |
| `fig4_classification_metrics.png` | OA, AA, kappa, precision, F1, specificity vs number of bands, all 8 methods |
| `fig5_detection_reconstruction.png` | CEM target-detection AUC and SSIM vs number of bands |
| `fig6_time_vs_accuracy_K20.png` | Selection time vs accuracy (why Fisher-Cholesky was chosen) |
| `fig7_per_class_sensitivity_K20.png` | Sensitivity of every crop |
| `fig8a_classification_maps_K20.png` | Classification maps, all 8 methods and all 200 bands |
| `fig8b_error_maps_K20.png` | The exam pixels each method gets wrong (red) |
| `fig9_selected_bands_K20.png` | Which bands each method keeps |
| `fig10_base_paper_ablation_K20.png` | The base paper's modules added one at a time, on our data |
| `fig11_confusion_matrices_K20.png` | Confusion matrices (which crops are mistaken for which) |
| `fig12_band_stability_K20.png` | How similar the chosen bands are from split to split (exactly, and within 2 bands) |
| `fig13_cholesky_criteria.png` | Why Fisher: other criteria on the same recursive-Cholesky engine |
| `fig14_spread_rules.png` | Forcing the bands to spread out costs accuracy; a 3-band minimum gap is free |
| `fig15_learning_set_size.png` | Accuracy with 10, 20, 40, 50 and 80 % of the pixels used for learning (18 bands) |
| `fig16_prescreening_all_methods.png` | Every method choosing 18 bands from the 164 pre-screened bands vs from all 200 |
| `fig18_split_comparison.png` | Fisher-Cholesky, GA, Hybrid, base paper and Entropy at 18 bands for 20/80, 40/60, 50/50, 60/40 and 80/20 splits |
| `fig17_flatline_overall_accuracy.png` | Overall accuracy out to 100 bands: where the curve bends (`fig4` and `fig5` also run to 100 bands) |

## How to run (from this folder)

```
pip install -r requirements.txt
python one_page_demo.py                 # 1 minute: the whole idea in one short file
python src/entropy_prescreen.py         # pre-screening -> results/prescreen/
python src/final_comparison.py          # about 20 min: all methods, 6 splits -> results/final/
python src/final_comparison.py --regrade   # adds MCC and confusion matrices (2 min)
python src/plot_dataset.py              # fig1
python src/plot_prescreen.py            # fig2
python src/final_comparison.py --maps      # maps for every method (1 min)
python src/compare_cholesky_criteria.py # Fisher vs other criteria on the Cholesky engine (3 min)
python src/spread_rules.py              # forcing spread across the spectrum (3 min)
python src/extra_studies.py --split --prescreen --extra-k   # about 40 min: learning-set size, 200 vs 164 bands, K = 40 ... 100
python src/plot_final.py                # fig1b, fig3 - fig13 and the tables
python src/split_comparison.py          # about 10 min: five methods at five learning/exam splits (needs split_ratio.csv)
python src/plot_extra.py                # fig15 - fig18 and their tables
```

## Layout

```
data/                     the Indian Pines cube (145 x 145 x 200) and its labels -- unchanged originals
one_page_demo.py          short, readable version of the method
src/data_setup.py         loading and the stratified 20 % / 80 % split
src/common.py             LDA fitness used by GA / Jaya / GWO
src/optimizers.py         GA, Jaya, GWO (and SA, PSO, DE from the earlier study)
src/fisher_cholesky.py    the chosen method
src/mfe_cfsp.py           base paper re-implementation
src/unsupervised_selectors.py   entropy, variance and other label-free rankings
src/metrics.py            every metric
src/cholesky_criteria.py  other criteria on the same engine (to justify Fisher)
src/spread_rules.py       Fisher-Cholesky with spread rules
src/extra_studies.py      learning-set size, pre-screening on/off for all methods, more bands (K up to 100)
src/split_comparison.py   five methods at five learning/exam splits
src/plot_extra.py         figures and tables for those studies
docs/                     block diagrams of this phase and the next (FPGA) phase
src/final_comparison.py, src/plot_final.py, src/plot_dataset.py, src/entropy_prescreen.py, src/plot_prescreen.py
results/prescreen/        pre-screening outputs (the 164 band numbers)
results/final/            all final figures, tables, band sets
archive/                  earlier studies kept for the record (see archive/README.md)
```

## Protocol (no leakage)

- 6 outer splits: stratified 20 % learning / 80 % exam, seeds 42, 1, 2, 3, 4, 5. K = 5, 10, 15, 18, 20, 25, 30.
- Band selection and every fitness function use learning pixels only. Exam pixels are used once, to grade the final band set.
- Every metaheuristic gets the same candidate bands, fitness, budget (5 × 3,000 calls) and seeds.

## Caveats

- **Pixel-level split.** Neighbouring pixels can fall into both the learning and the exam set, which makes every method look better than on a new scene. The comparison between methods is fair; the absolute numbers are optimistic.
- **Base paper re-implementation.** The paper does not give every setting (gate γ, block size, similarity threshold). Our choices are listed at the top of `src/mfe_cfsp.py`. The paper reports its gains mainly with CNN classifiers on image patches; we use a pixel-wise SVM.
- **Times are Python on a laptop.** The ratio matters: Fisher-Cholesky does about 20 small matrix-update steps, while the GA does 15,000 classifier evaluations.
- **Fisher-Cholesky is supervised.** It needs labelled learning pixels. For given data it is deterministic. Across splits only 38 % of the bands are exactly the same, but 90 % are the same or a neighbour within 2 bands (`fig12`). It keeps choosing the same spectral regions, and neighbouring bands carry nearly the same information.
- **Spread across the spectrum:** Fisher-Cholesky uses all five 40-band regions on every split, but takes about half its bands from bands 0–39, where the crop spectra differ most. Forcing a more even spread lowers accuracy by up to 0.8 points OA. A rule of no two bands closer than 3 costs nothing (`fig14`).
- **Why Fisher and not another criterion:** on the same Cholesky engine, Fisher beats mean pairwise class distance, worst-pair distance and label-free maximum volume (`fig13`).
- **Errors sit in the same fields for every method, including all 200 bands** (`fig8b`): mainly the corn and soybean fields and the buildings class. That is a limit of pixel-by-pixel classification on this image, not of band selection.
- **Pre-screening** barely changes any method's accuracy: choosing 18 bands from all 200 instead of the 164 pre-screened bands changes overall accuracy by less than 0.7 points for every method, in both directions (`fig3`, `fig16`). An SVM on all 164 bands scores 84.8 % against 83.4 % on all 200, so the 36 removed bands (0-5, 79-98, 102-104, 143-145, 195, 197-199) are mostly noise. Its main value is 18 % fewer bands to read and a covariance matrix a third smaller for the FPGA.
- **Learning-set size** (`fig15`, 18 bands): more learning pixels raise every method's accuracy (Fisher-Cholesky: 78.6 % at 10 %, 82.1 % at 20 %, 86.9 % at 80 %). The method ranking holds; the GA's lead over Fisher-Cholesky is 1.1 points at 20 %, 0.2 at 40 % and none at 80 %. 20 % is our main protocol because labelled pixels are scarce in practice. The study was run at 18 bands only. At 80 % the exam holds only 2,051 pixels, so per-class scores are noisy, and the neighbour-leakage optimism grows with the learning share.
- **More than 30 bands:** selection time grows with K but the gap stays: median 0.011 s (K = 18) to 0.043 s (K = 100) for Fisher-Cholesky, and 11 s to 26 s for the GA. Accuracy for K = 40-100 comes from the same 6 splits, grading and search budget as the main study.
