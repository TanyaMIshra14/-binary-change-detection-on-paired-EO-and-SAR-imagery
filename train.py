import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data_preprocessing import ChangeDetectionDataset
from graph_generation import create_grid_graph, get_grid_size
from losses import hybrid_loss
from metrices import compute_metrics, print_confusion_matrix
from model import SiameseChangeDetector

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train EO-SAR Change Detector")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Seed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 80)
    print("DEVICE")
    print("=" * 80)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    training_start_time = datetime.now()
    print("\n" + "=" * 80)
    print(f"TRAINING STARTED: {training_start_time}")
    print("=" * 80)

    print("\nLoading datasets...")

    train_dataset = ChangeDetectionDataset(
        root_dir=cfg["train_dir"],
        image_size=cfg["image_size"],
        augment=True,
        calculate_class_weights=True
    )

    val_dataset = ChangeDetectionDataset(
        root_dir=cfg["val_dir"],
        image_size=cfg["image_size"],
        augment=False,
        calculate_class_weights=False
    )

    print(f"\nTraining Samples   : {len(train_dataset)}")
    print(f"Validation Samples : {len(val_dataset)}")

    # ============================================================
    # CLASS WEIGHTS
    # ============================================================

    class_weights = train_dataset.class_weights.to(device)

    # pos_weight_boost reduced from 3.0 → 1.0
    # The inverse-frequency weight (~31.85) is already sufficient;
    # an extra 3x multiplier was causing the model to predict everything as change.
    pos_boost = cfg.get("pos_weight_boost", 1.0)
    class_weights[1] = class_weights[1] * pos_boost

    print(f"\nClass weights (after boost x{pos_boost})")
    print(f"  No-Change weight : {class_weights[0]:.4f}")
    print(f"  Change weight    : {class_weights[1]:.4f}")

    # ============================================================
    # DATALOADERS
    # ============================================================
    batch_size  = cfg["batch_size"]
    num_workers = cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    # ============================================================
    # MODEL
    # ============================================================
    print("\nInitialising model...")
    model = SiameseChangeDetector(use_gnn=cfg.get("use_gnn", True)).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # ============================================================
    # LR SCHEDULER
    # ============================================================

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.get("cosine_T0", 10),
        T_mult=cfg.get("cosine_Tmult", 2),
        eta_min=cfg.get("min_lr", 1e-6),
    )

    grid_size  = get_grid_size(cfg["image_size"], stride=32)
    edge_index = create_grid_graph(H=grid_size, W=grid_size).to(device)
    print(f"\nGNN grid: {grid_size}x{grid_size}  |  edge_index: {edge_index.shape}")

    # Default threshold raised from 0.5 → 0.65 to reduce over-prediction of change class.
    # This is overridden each epoch by find_best_threshold().
    DETECTION_THRESHOLD = cfg.get("detection_threshold", 0.65)
    GRAD_CLIP = cfg.get("grad_clip", 1.0)

    # ============================================================
    # THRESHOLD SEARCH
    # Sweeps candidate thresholds on the val set and returns the
    # one that maximises F1, along with its probability outputs.
    # ============================================================

    def find_best_threshold(all_probs, all_targets,
                            thresholds=np.arange(0.3, 0.91, 0.05)):
        best_f1, best_thresh = 0.0, 0.5
        for t in thresholds:
            preds = (all_probs > t).astype(float)
            metrics = compute_metrics(preds, all_targets)
            if metrics["f1"] > best_f1:
                best_f1     = metrics["f1"]
                best_thresh = t
        print(f"  Threshold search → best threshold: {best_thresh:.2f}  "
              f"(F1 = {best_f1:.4f})")
        return best_thresh, best_f1

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate(threshold=DETECTION_THRESHOLD):
        model.eval()
        total_loss = 0
        all_probs   = []
        all_targets = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(device)
                masks  = masks.to(device)

                eo_img  = images[:, :3]
                sar_img = images[:, 3:4]

                outputs = model(eo_img, sar_img, edge_index)
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

                loss = hybrid_loss(
                    outputs.squeeze(1),
                    masks,
                    class_weights=class_weights,
                    alpha=cfg.get("focal_alpha", 0.25),
                    gamma=cfg.get("focal_gamma", 2.0),
                )
                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())

        all_probs   = np.array(all_probs)
        all_targets = np.array(all_targets)

        # Find the threshold that maximises F1 on this val pass
        best_thresh, _ = find_best_threshold(all_probs, all_targets)

        preds   = (all_probs > best_thresh).astype(float)
        metrics = compute_metrics(preds, all_targets)
        avg_loss = total_loss / len(val_loader)

        return avg_loss, metrics, best_thresh

    # ============================================================
    # TRAINING CONFIG
    # ============================================================
    EPOCHS = cfg["epochs"]
    EARLY_STOPPING_PATIENCE = cfg.get("early_stopping_patience", 10)

    best_f1          = -1.0
    best_val_loss    = float("inf")
    patience_counter = 0
    best_threshold   = DETECTION_THRESHOLD   # updated each epoch

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            leave=False
        )

        for images, masks in train_bar:
            images = images.to(device)
            masks  = masks.to(device)

            eo_img  = images[:, :3]
            sar_img = images[:, 3:4]

            optimizer.zero_grad()

            outputs = model(eo_img, sar_img, edge_index)
            outputs = F.interpolate(
                outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

            loss = hybrid_loss(
                outputs.squeeze(1),
                masks,
                class_weights=class_weights,
                alpha=cfg.get("focal_alpha", 0.25),
                gamma=cfg.get("focal_gamma", 2.0),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_metrics, epoch_best_thresh = validate()
        scheduler.step(val_loss)
        current_lr  = optimizer.param_groups[0]['lr']
        epoch_time  = time.time() - epoch_start

        print("\n" + "=" * 80)
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print("=" * 80)
        print(f"Timestamp       : {datetime.now()}")
        print(f"Epoch Time      : {epoch_time:.1f} sec")
        print(f"Learning Rate   : {current_lr:.8f}")
        print(f"Best Threshold  : {epoch_best_thresh:.2f}")
        print("\nLOSSES")
        print(f"  Train Loss      : {avg_train_loss:.4f}")
        print(f"  Validation Loss : {val_loss:.4f}")
        print("\nMETRICS (Change class)")
        print(f"  Accuracy  : {val_metrics['accuracy']:.4f}")
        print(f"  Precision : {val_metrics['precision']:.4f}")
        print(f"  Recall    : {val_metrics['recall']:.4f}")
        print(f"  F1 Score  : {val_metrics['f1']:.4f}")
        print(f"  IoU       : {val_metrics['iou']:.4f}")

        current_f1 = val_metrics["f1"]

        if current_f1 > best_f1:
            best_f1          = current_f1
            best_val_loss    = val_loss
            best_threshold   = epoch_best_thresh
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "val_loss": val_loss,
                    "best_threshold": best_threshold,
                    "config": cfg,
                },
                "hybrid_model_best.pth",
            )
            print(f"\n  ✓ BEST MODEL SAVED  "
                  f"(F1 = {best_f1:.4f}, threshold = {best_threshold:.2f})")

        else:
            patience_counter += 1
            print(f"\n  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("\nEARLY STOPPING TRIGGERED")
                break

    # ============================================================
    # FINAL VALIDATION WITH BEST MODEL
    # ============================================================

    print("\n" + "=" * 80)
    print("FINAL VALIDATION WITH BEST MODEL")
    print("=" * 80)

    checkpoint = torch.load("hybrid_model_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Restore the threshold that was optimal when this checkpoint was saved
    DETECTION_THRESHOLD = checkpoint.get("best_threshold", best_threshold)
    model.eval()

    final_val_loss, final_metrics, _ = validate(threshold=DETECTION_THRESHOLD)

    print(f"  Val Loss  : {final_val_loss:.4f}")
    print(f"  Threshold : {DETECTION_THRESHOLD:.2f}")
    print(f"  Accuracy  : {final_metrics['accuracy']:.4f}")
    print(f"  Precision : {final_metrics['precision']:.4f}")
    print(f"  Recall    : {final_metrics['recall']:.4f}")
    print(f"  F1 Score  : {final_metrics['f1']:.4f}")
    print(f"  IoU       : {final_metrics['iou']:.4f}")

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device)
            eo_img  = images[:, :3]
            sar_img = images[:, 3:4]
            outputs = model(eo_img, sar_img, edge_index)
            outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                    mode="bilinear", align_corners=False)
            probs = torch.sigmoid(outputs)
            preds = (probs > DETECTION_THRESHOLD).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    print_confusion_matrix(np.array(all_preds), np.array(all_targets))

    torch.save(model.state_dict(), "hybrid_model_final.pth")
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Start Time      : {training_start_time}")
    print(f"End Time        : {training_end_time}")
    print(f"Total Time      : {training_duration}")
    print(f"Best Val F1     : {best_f1:.4f}")
    print(f"Best Val Loss   : {best_val_loss:.4f}")
    print(f"Best Threshold  : {best_threshold:.2f}")
    print("\nFinal model saved to hybrid_model_final.pth")
    print("=" * 80)