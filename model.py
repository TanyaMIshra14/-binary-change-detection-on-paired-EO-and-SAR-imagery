"""
model.py — Siamese UNet Change Detector for EO-SAR Binary Change Detection

Architecture:
  - Siamese ResNet-18 encoders for pre-event (EO) and post-event (SAR)
  - Multi-scale feature differencing (abs diff) at each encoder stage
  - UNet-style decoder with skip connections from diff features
  - Lightweight GNN refinement on the bottleneck (optional, can be disabled)

Key design decisions:
  1. Separate encoders per modality (EO=3ch, SAR=1ch) — they have very different
     noise profiles and intensity ranges.
  2. Feature *differencing* (|pre - post|) is the core change signal. Concatenation
     alone forces the model to learn subtraction implicitly; explicit diff is a
     strong inductive bias for change detection.
  3. Skip connections from each diff level let the decoder recover fine-grained
     spatial details lost to pooling — critical for per-pixel accuracy.
  4. Sigmoid output — BCEWithLogits-compatible raw logits returned from forward().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    from torch_geometric.nn import SAGEConv
    HAS_GEOMETRIC = True
except ImportError:
    HAS_GEOMETRIC = False


# ============================================================
# ENCODER BLOCKS
# ============================================================

class EOEncoder(nn.Module):
    """ResNet-18 encoder for 3-channel EO images with multi-scale outputs."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Stage 0: initial conv+bn+relu+maxpool  -> /4  (64 ch)
        self.stage0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.stage1 = backbone.layer1   # /4   -> 64 ch
        self.stage2 = backbone.layer2   # /8   -> 128 ch
        self.stage3 = backbone.layer3   # /16  -> 256 ch
        self.stage4 = backbone.layer4   # /32  -> 512 ch

    def forward(self, x):
        s0 = self.stage0(x)    # [B, 64,  H/4,  W/4]
        s1 = self.stage1(s0)   # [B, 64,  H/4,  W/4]
        s2 = self.stage2(s1)   # [B, 128, H/8,  W/8]
        s3 = self.stage3(s2)   # [B, 256, H/16, W/16]
        s4 = self.stage4(s3)   # [B, 512, H/32, W/32]
        return s1, s2, s3, s4


class SAREncoder(nn.Module):
    """ResNet-18 encoder adapted for single-channel SAR images."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Replace 3→64 conv1 with 1→64 (SAR is single channel)
        # Initialise with mean of pretrained RGB weights so transfer still helps
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))

        self.stage0 = nn.Sequential(
            self.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return s1, s2, s3, s4


# ============================================================
# GRAPH MODULE (optional refinement at bottleneck)
# ============================================================

class GraphModule(nn.Module):
    """
    2-layer GraphSAGE applied to the bottleneck feature map.
    Provides non-local spatial reasoning on the change features.
    Falls back to identity if torch_geometric is not available.
    """

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        if HAS_GEOMETRIC:
            self.gnn1 = SAGEConv(in_channels, hidden_channels)
            self.gnn2 = SAGEConv(hidden_channels, hidden_channels)
        self.has_geo = HAS_GEOMETRIC

    def forward(self, x, edge_index):
        if not self.has_geo or edge_index is None:
            return x
        x = F.relu(self.gnn1(x, edge_index))
        x = self.gnn2(x, edge_index)
        return x


# ============================================================
# DECODER BLOCK
# ============================================================

class DecoderBlock(nn.Module):
    """
    Single decoder step:
      bilinear upsample x2 -> concat skip connection -> conv -> BN -> ReLU (x2)
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ============================================================
# MAIN MODEL
# ============================================================

class SiameseChangeDetector(nn.Module):
    """
    Siamese UNet for binary change detection on EO-SAR pairs.

    Input:
        eo_pre  : [B, 3, H, W] — pre-event EO image (ImageNet-normalised)
        sar_post: [B, 1, H, W] — post-event SAR image (per-image normalised)
        edge_index: graph edges for GNN refinement (or None to skip GNN)

    Output:
        logits  : [B, 1, H, W] — raw logits (apply sigmoid for probabilities)
    """

    def __init__(self, use_gnn=True):
        super().__init__()
        self.use_gnn = use_gnn and HAS_GEOMETRIC

        # Dual encoders
        self.eo_encoder  = EOEncoder(pretrained=True)
        self.sar_encoder = SAREncoder(pretrained=True)

        # Bottleneck fusion: concat 512+512=1024 -> 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Optional GNN at bottleneck
        if self.use_gnn:
            self.gnn = GraphModule(512, 512)

        # Decoder (skip channels come from abs-diff of encoder features)
        # diff channels: s3=256, s2=128, s1=64
        self.dec3 = DecoderBlock(512, 256, 256)   # /32 -> /16, skip=diff_s3
        self.dec2 = DecoderBlock(256, 128, 128)   # /16 -> /8,  skip=diff_s2
        self.dec1 = DecoderBlock(128, 64,  64)    # /8  -> /4,  skip=diff_s1
        self.dec0 = DecoderBlock(64,  0,   32)    # /4  -> /2,  no skip

        # Final output head: /2 -> /1
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, eo_pre, sar_post, edge_index=None):
        # Encode both modalities
        eo_s1,  eo_s2,  eo_s3,  eo_s4  = self.eo_encoder(eo_pre)
        sar_s1, sar_s2, sar_s3, sar_s4 = self.sar_encoder(sar_post)

        # Compute absolute feature differences as change signal at each scale
        diff_s1 = torch.abs(eo_s1 - sar_s1)   # [B, 64,  H/4,  W/4]
        diff_s2 = torch.abs(eo_s2 - sar_s2)   # [B, 128, H/8,  W/8]
        diff_s3 = torch.abs(eo_s3 - sar_s3)   # [B, 256, H/16, W/16]

        # Bottleneck: concat deepest features
        bottleneck = torch.cat([eo_s4, sar_s4], dim=1)  # [B, 1024, H/32, W/32]
        x = self.bottleneck(bottleneck)                   # [B, 512,  H/32, W/32]

        # Optional GNN refinement on bottleneck
        if self.use_gnn and edge_index is not None:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            x_flat = self.gnn(x_flat, edge_index)
            x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Decode with skip connections from diff features
        x = self.dec3(x, diff_s3)   # [B, 256, H/16, W/16]
        x = self.dec2(x, diff_s2)   # [B, 128, H/8,  W/8]
        x = self.dec1(x, diff_s1)   # [B, 64,  H/4,  W/4]
        x = self.dec0(x)            # [B, 32,  H/2,  W/2]
        x = self.head(x)            # [B, 1,   H,    W]

        return x


# Keep the old name as an alias so existing imports don't break
HybridChangeDetector = SiameseChangeDetector
