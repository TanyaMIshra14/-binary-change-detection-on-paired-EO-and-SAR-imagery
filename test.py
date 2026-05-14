import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data_preprocessing import ChangeDetectionDataset
from graph_generation import create_grid_graph, get_grid_size
from metrices import compute_metrics, print_confusion_matrix
from model import SiameseChangeDetector

parser = argparse.ArgumentParser(description="Evaluate EO-SAR Change Detector")
parser.add_argument("--data_path",  type=str, required=True,
                    help="Path to test split directory")
parser.add_argument("--weights",    type=str, required=True,
                    help="Path to model checkpoint (.pth)")
parser.add_argument("--config",     type=str, default="config.yaml",
                    help="Path to YAML config (for image_size etc.)")
parser.add_argument("--threshold",  type=float, default=None,
                    help="Detection threshold (overrides config)")
parser.add_argument("--batch_size", type=int,   default=None,
                    help="Batch size (overrides config)")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

threshold  = args.threshold  if args.threshold  is not None else cfg.get("detection_threshold", 0.5)
batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 8)
image_size = cfg["image_size"]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

test_start = datetime.now()
print("\n" + "=" * 70)
print("TESTING STARTED")
print("=" * 70)
print(f"Start Time : {test_start}")
print(f"Weights    : {args.weights}")
print(f"Data       : {args.data_path}")
print(f"Threshold  : {threshold}")

test_dataset = ChangeDetectionDataset(
    root_dir=args.data_path,
    image_size=image_size,
    augment=False,
    calculate_class_weights=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

print(f"Test samples: {len(test_dataset)}")

model = SiameseChangeDetector(use_gnn=cfg.get("use_gnn", True)).to(device)

ckpt = torch.load(args.weights, map_location=device, weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)
model.eval()

grid_size  = get_grid_size(image_size, stride=32)
edge_index = create_grid_graph(H=grid_size, W=grid_size).to(device)

all_preds, all_targets = [], []
t0 = time.time()

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        masks  = masks.to(device)

        eo_img  = images[:, :3]
        sar_img = images[:, 3:4]

        outputs = model(eo_img, sar_img, edge_index)
        outputs = F.interpolate(
            outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()

        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(masks.cpu().numpy().flatten())

elapsed = time.time() - t0

all_preds_np   = np.array(all_preds)
all_targets_np = np.array(all_targets)

metrics = compute_metrics(all_preds_np, all_targets_np, verbose=True)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)
print(f"  Accuracy  : {metrics['accuracy']:.4f}")
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  F1 Score  : {metrics['f1']:.4f}")
print(f"  IoU       : {metrics['iou']:.4f}")

print_confusion_matrix(all_preds_np, all_targets_np)

# ── Confusion Matrix Visualisation ──────────────────────────────────────────

def plot_confusion_matrix(preds, targets, save_path="confusion_matrix.png"):
    cm = confusion_matrix(targets.astype(int), preds.astype(int))

    # Raw counts plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix — EO-SAR Change Detection", fontsize=14, fontweight="bold")

    labels = ["No Change", "Change"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="grey",
        ax=axes[0],
    )
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Normalised (row-wise recall) plot
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="grey",
        vmin=0,
        vmax=1,
        ax=axes[1],
    )
    axes[1].set_title("Normalised (row %)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    # Annotate key metrics below the plots
    tn, fp, fn, tp = cm.ravel()
    stats = (
        f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}  |  "
        f"F1={metrics['f1']:.4f}  IoU={metrics['iou']:.4f}  "
        f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}"
    )
    fig.text(0.5, 0.01, stats, ha="center", fontsize=10, color="dimgray")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nConfusion matrix saved → {save_path}")
    plt.show()

plot_confusion_matrix(all_preds_np, all_targets_np)

# ────────────────────────────────────────────────────────────────────────────

test_end = datetime.now()
print(f"\nTest Start Time    : {test_start}")
print(f"Test End Time      : {test_end}")
print(f"Total Testing Time : {elapsed:.1f} sec")
print("=" * 70)
