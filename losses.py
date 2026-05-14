import torch
import torch.nn.functional as F


def dice_loss(
    pred_logits,
    target,
    smooth=1
):

    pred = torch.sigmoid(pred_logits)

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (
        pred * target
    ).sum()

    dice_coeff = (
        2. * intersection + smooth
    ) / (
        pred.sum() +
        target.sum() +
        smooth
    )

    return 1 - dice_coeff


def focal_loss(
    pred_logits,
    target,
    alpha=0.40,
    gamma=2.0,
    class_weights=None
):
    
    bce = F.binary_cross_entropy_with_logits(
        pred_logits,
        target,
        reduction='none'
    )
    
    pt = torch.exp(-bce)
    focal_weight = (1 - pt) ** gamma
    
    # Apply alpha weighting
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    focal = alpha_t * focal_weight * bce
    
    # Apply class weights if provided
    if class_weights is not None:
        # Further boost minority class (damage/class 1)
        cw = class_weights.to(pred_logits.device)
        weight_map = target * cw[1] + (1.0 - target) * cw[0]
        focal = focal * weight_map
    
    return focal.mean()


def hybrid_loss(
    pred_logits,
    target,
    class_weights=None,
    focal_weight=0.5,
    dice_weight=0.5,
    alpha=0.40,
    gamma=2.0,
):

    fl = focal_loss(pred_logits, target, alpha=alpha, gamma=gamma,
                    class_weights=class_weights)
    dl = dice_loss(pred_logits, target)
    return focal_weight * fl + dice_weight * dl