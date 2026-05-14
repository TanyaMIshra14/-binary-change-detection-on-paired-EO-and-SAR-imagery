
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from data_preprocessing import ChangeDetectionDataset, IMAGENET_MEAN, IMAGENET_STD
from graph_generation import create_grid_graph, get_grid_size
from model import SiameseChangeDetector


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,   required=True)
parser.add_argument("--weights",   type=str,   required=True)
parser.add_argument("--config",    type=str,   default="config.yaml")
parser.add_argument("--index",     type=int,   default=0,
                    help="Sample index to visualise")
parser.add_argument("--threshold", type=float, default=None)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

threshold  = args.threshold if args.threshold else cfg.get("detection_threshold", 0.5)
image_size = cfg["image_size"]

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


dataset = ChangeDetectionDataset(
    root_dir=args.data_path,
    image_size=image_size,
    augment=False,
    calculate_class_weights=False,
)

image, mask = dataset[args.index]  

model = SiameseChangeDetector(use_gnn=cfg.get("use_gnn", True)).to(device)
ckpt  = torch.load(args.weights, map_location=device)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()

grid_size  = get_grid_size(image_size, stride=32)
edge_index = create_grid_graph(H=grid_size, W=grid_size).to(device)

inp     = image.unsqueeze(0).to(device)
eo_img  = inp[:, :3]
sar_img = inp[:, 3:4]

with torch.no_grad():
    output = model(eo_img, sar_img, edge_index)
    output = F.interpolate(output, size=mask.shape[-2:],
                           mode="bilinear", align_corners=False)
    prob  = torch.sigmoid(output).squeeze().cpu().numpy()
    pred  = (prob > threshold).astype(np.float32)

eo_display = image[:3].permute(1, 2, 0).numpy()
mean = np.array(IMAGENET_MEAN)
std  = np.array(IMAGENET_STD)
eo_display = np.clip(eo_display * std + mean, 0, 1)

sar_display = image[3].numpy()
gt_display  = mask.numpy()

fig, axes = plt.subplots(1, 5, figsize=(22, 5))

axes[0].imshow(eo_display)
axes[0].set_title("EO (Pre-event)")

axes[1].imshow(sar_display, cmap="gray")
axes[1].set_title("SAR (Post-event)")

axes[2].imshow(gt_display, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Ground Truth")

axes[3].imshow(prob, cmap="hot", vmin=0, vmax=1)
axes[3].set_title("Change Probability")

axes[4].imshow(pred, cmap="gray", vmin=0, vmax=1)
axes[4].set_title(f"Prediction (t={threshold})")

for ax in axes:
    ax.axis("off")

plt.suptitle(f"Sample index {args.index}", fontsize=13)
plt.tight_layout()
plt.savefig(f"prediction_{args.index}.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to prediction_{args.index}.png")