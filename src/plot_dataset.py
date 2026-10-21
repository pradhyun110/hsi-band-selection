"""Pictures of the Indian Pines dataset itself.  Run: python src/plot_dataset.py
Writes results/dataset/fig_dataset.png (false-colour image, ground-truth map, average spectrum of each class)."""
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parent.parent
cube = np.load(ROOT/"data"/"indianpinearray.npy").astype(float); gt = np.load(ROOT/"data"/"IPgt.npy")
NAMES = ["Alfalfa","Corn-notill","Corn-mintill","Corn","Grass-pasture","Grass-trees","Grass-pasture-mowed","Hay-windrowed",
         "Oats","Soybean-notill","Soybean-mintill","Soybean-clean","Wheat","Woods","Buildings-Grass-Trees-Drives","Stone-Steel-Towers"]
cols = plt.cm.tab20(np.linspace(0, 1, 20))[:16]
cmap = ListedColormap(np.vstack([[1, 1, 1, 1], cols]))
def stretch(b):
    lo, hi = np.percentile(b, [2, 98]); return np.clip((b-lo)/(hi-lo), 0, 1)
rgb = np.dstack([stretch(cube[:, :, i]) for i in (42, 27, 16)])   # bands chosen for a false-colour view
fig = plt.figure(figsize=(15, 6.2))
ax = fig.add_subplot(1, 3, 1); ax.imshow(rgb); ax.set_title("1. The image (false colour: red = healthy plants)\n145 x 145 pixels, 200 bands each"); ax.axis("off")
ax = fig.add_subplot(1, 3, 2); ax.imshow(gt, cmap=cmap, vmin=0, vmax=16, interpolation="nearest"); ax.axis("off")
ax.set_title("2. Ground truth: 16 crop / land types\n(white = unlabelled, 10,249 labelled pixels)")
ax.legend(handles=[Patch(color=cols[i], label=f"{i+1} {NAMES[i]}") for i in range(16)], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=6.5, handlelength=1.2, frameon=False)
ax = fig.add_subplot(1, 3, 3)
for c in range(1, 17):
    ax.plot(cube[gt == c].mean(0), color=cols[c-1], lw=1.4)
ax.set_title("3. Average spectrum of each class\n(each line = one crop; the classifier must tell them apart)")
ax.set_xlabel("band number (0-199)"); ax.set_ylabel("reflectance value"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(ROOT/"results"/"dataset"/"fig_dataset.png", dpi=170); print("saved")
