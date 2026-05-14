import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

def compute_metrics(preds: np.ndarray, targets: np.ndarray,verbose: bool = False):
    preds = preds.flatten().astype(np.uint8)
    targets = targets.flatten().astype(np.uint8)

    if verbose:
        total      = len(preds)
        pred_pos   = int(np.sum(preds == 1))
        gt_pos     = int(np.sum(targets == 1))
        print(f"\n  Total pixels      : {total:,}")
        print(f"  Predicted positives : {pred_pos:,} ({100*pred_pos/total:.1f}%)")
        print(f"  Ground truth pos.   : {gt_pos:,}  ({100*gt_pos/total:.1f}%)")

    accuracy  = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall    = recall_score(targets,    preds, zero_division=0)
    f1        = f1_score(targets,        preds, zero_division=0)
    iou       = jaccard_score(targets,   preds, zero_division=0)

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "iou":       iou,
    }
    
def print_confusion_matrix(preds: np.ndarray, targets: np.ndarray):
    preds   = preds.flatten().astype(np.uint8)
    targets = targets.flatten().astype(np.uint8)

    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX (Change class = positive)")
    print("=" * 50)
    print(f"{'':20s}  Pred 0      Pred 1")
    print(f"  Actual 0 (No-Change): {tn:>10,}  {fp:>10,}")
    print(f"  Actual 1 (Change)   : {fn:>10,}  {tp:>10,}")
    print("=" * 50)
    total = tn + fp + fn + tp
    print(f"  True Negatives  (TN): {tn:>10,}  ({100*tn/total:.1f}%)")
    print(f"  False Positives (FP): {fp:>10,}  ({100*fp/total:.1f}%)  <- over-detection")
    print(f"  False Negatives (FN): {fn:>10,}  ({100*fn/total:.1f}%)  <- missed change")
    print(f"  True Positives  (TP): {tp:>10,}  ({100*tp/total:.1f}%)")
    print("=" * 50)