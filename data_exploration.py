import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

TRAIN_DIR = cfg["train_dir"]
PRE_DIR   = os.path.join(TRAIN_DIR, "pre-event")
POST_DIR  = os.path.join(TRAIN_DIR, "post-event")
MASK_DIR  = os.path.join(TRAIN_DIR, "target")


files = sorted(os.listdir(PRE_DIR))
print(f"Training samples: {len(files)}")

def remap_labels(mask):
    out = np.zeros_like(mask, dtype=np.uint8)
    out[(mask == 2) | (mask == 3)] = 1
    return out

print("\nComputing class distribution...")
counts = {0: 0, 1: 0}

for fname in files:
    with rasterio.open(os.path.join(MASK_DIR, fname)) as src:
        mask = src.read(1)
    mask = remap_labels(mask)
    counts[0] += int(np.sum(mask == 0))
    counts[1] += int(np.sum(mask == 1))

total = counts[0] + counts[1]
print(f"  No-Change pixels : {counts[0]:>12,}  ({100*counts[0]/total:.1f}%)")
print(f"  Change pixels    : {counts[1]:>12,}  ({100*counts[1]/total:.1f}%)")
print(f"  Imbalance ratio  : {counts[0]/max(counts[1],1):.1f}:1")

print("\nComputing EO channel statistics (first 50 samples)...")
eo_means, eo_stds = [], []

for fname in files[:50]:
    with rasterio.open(os.path.join(PRE_DIR, fname)) as src:
        eo = src.read()[:3].astype(np.float32)
    lo, hi = eo.min(), eo.max()
    eo = (eo - lo) / (hi - lo + 1e-8)
    eo_means.append(eo.mean(axis=(1, 2)))
    eo_stds.append(eo.std(axis=(1, 2)))

eo_means = np.array(eo_means).mean(axis=0)
eo_stds  = np.array(eo_stds).mean(axis=0)
print(f"  EO mean per channel: {eo_means.round(3)}")
print(f"  EO std  per channel: {eo_stds.round(3)}")

print("\nGenerating sample visualisations...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i, fname in enumerate(files[:3]):
    with rasterio.open(os.path.join(PRE_DIR,  fname)) as src:
        eo = src.read()[:3].astype(np.float32)
    with rasterio.open(os.path.join(POST_DIR, fname)) as src:
        sar = src.read(1).astype(np.float32)
    with rasterio.open(os.path.join(MASK_DIR, fname)) as src:
        raw_mask = src.read(1)

    # Normalise for display
    eo  = (eo  - eo.min())  / (eo.max()  - eo.min()  + 1e-8)
    sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-8)
    mask = remap_labels(raw_mask)

    eo_disp = np.transpose(eo, (1, 2, 0))

    axes[i][0].imshow(eo_disp)
    axes[i][0].set_title(f"EO #{i} — {fname[:20]}")

    axes[i][1].imshow(sar, cmap="gray")
    axes[i][1].set_title("SAR")

    axes[i][2].imshow(raw_mask, cmap="tab10", vmin=0, vmax=3)
    axes[i][2].set_title("Original Labels")

    axes[i][3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[i][3].set_title("Remapped (Binary)")

    for ax in axes[i]:
        ax.axis("off")

plt.suptitle("Data Exploration — Sample Visualisation", fontsize=14)
plt.tight_layout()
plt.savefig("data_exploration.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved data_exploration.png")
