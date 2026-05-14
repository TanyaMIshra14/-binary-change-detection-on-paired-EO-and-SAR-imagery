import os
import random
import cv2
import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet statistics for EO normalisation (pretrained backbone expects these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class ChangeDetectionDataset(Dataset):

    # Class mappings
    ORIGINAL_CLASSES = {
        0: "Background",
        1: "Intact",
        2: "Damaged",
        3: "Destroyed"
    }
    
    REMAPPED_CLASSES = {
        0: "No-Change",
        1: "Change"
    }

    def __init__(
        self,
        root_dir,
        image_size=256,
        augment=False,
        denoise=True,
        calculate_class_weights=True
    ):

        self.pre_dir = os.path.join(root_dir, "pre-event")
        self.post_dir = os.path.join(root_dir, "post-event")
        self.mask_dir = os.path.join(root_dir, "target")

        self.files = sorted(os.listdir(self.pre_dir))

        self.image_size = image_size
        self.augment = augment
        self.denoise = denoise

        # ImageNet normalisation for EO images
        self.eo_normalize = transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD
        )
        
        # Calculate class weights if requested
        self.class_weights = None
        if calculate_class_weights:
            self.class_weights = self._calculate_class_weights()


    @staticmethod
    def remap_labels(mask):
        """Map 4-class labels -> binary: {Damaged, Destroyed} -> 1, rest -> 0."""
        remapped = np.zeros_like(mask, dtype=np.uint8)
        remapped[(mask == 2) | (mask == 3)] = 1
        return remapped
    
    @staticmethod
    def normalize_sar(image: np.ndarray) -> np.ndarray:
        """Per-image min-max normalisation for SAR (no global stats available)."""
        image = image.astype(np.float32)
        lo, hi = image.min(), image.max()
        return (image - lo) / (hi - lo + 1e-8)

    @staticmethod
    def normalize_eo(image: np.ndarray) -> np.ndarray:
        """
        Clip to [0,1] range and convert to float32.
        ImageNet normalisation is applied later as a tensor transform.
        """
        image = image.astype(np.float32)
        lo, hi = image.min(), image.max()
        return (image - lo) / (hi - lo + 1e-8)
    
    @staticmethod
    def _median_denoise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Fast median filter — adequate for mild SAR speckle reduction."""
        if image.ndim == 2:
            return cv2.medianBlur(image, ksize)
        # Multi-channel: apply per channel
        out = np.stack(
            [cv2.medianBlur(image[:, :, c], ksize) for c in range(image.shape[2])],
            axis=2,
        )
        return out


    def apply_augmentation(self, eo_t, sar_t, mask_t):
    # Horizontal flip
        if random.random() > 0.5:
            eo_t   = TF.hflip(eo_t)
            sar_t  = TF.hflip(sar_t)
            mask_t = TF.hflip(mask_t)

        # Vertical flip
        if random.random() > 0.5:
            eo_t   = TF.vflip(eo_t)
            sar_t  = TF.vflip(sar_t)
            mask_t = TF.vflip(mask_t)

        # Random 90° rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            eo_t   = TF.rotate(eo_t,   angle)
            sar_t  = TF.rotate(sar_t,  angle)
            mask_t = TF.rotate(mask_t, angle)

        # Random crop + resize (scale jitter)
        if random.random() > 0.5:
            H, W   = eo_t.shape[-2], eo_t.shape[-1]
            scale  = random.uniform(0.7, 1.0)
            new_h  = int(H * scale)
            new_w  = int(W * scale)
            top    = random.randint(0, H - new_h)
            left   = random.randint(0, W - new_w)
            eo_t   = TF.resized_crop(eo_t,   top, left, new_h, new_w, (H, W))
            sar_t  = TF.resized_crop(sar_t,  top, left, new_h, new_w, (H, W))
            mask_t = TF.resized_crop(mask_t, top, left, new_h, new_w, (H, W),
                                    interpolation=TF.InterpolationMode.NEAREST)

        # EO-only: colour jitter (SAR has no colour, mask must not be altered)
        if random.random() > 0.5:
            eo_t = TF.adjust_brightness(eo_t, random.uniform(0.8, 1.2))
            eo_t = TF.adjust_contrast(eo_t,   random.uniform(0.8, 1.2))

        # Gaussian blur on EO
        if random.random() > 0.5:
            eo_t = TF.gaussian_blur(eo_t, kernel_size=3)

        return eo_t, sar_t, mask_t
    
    def _calculate_class_weights(self):
        """Inverse-frequency class weights to counter severe class imbalance."""
        counts = {0: 0, 1: 0}
        print("Calculating class weights (scanning masks)...")
        for fname in self.files:
            path = os.path.join(self.mask_dir, fname)
            with rasterio.open(path) as src:
                mask = src.read(1)
            mask = self.remap_labels(mask)
            for cls in (0, 1):
                counts[cls] += int(np.sum(mask == cls))

        total = counts[0] + counts[1]
        # Inverse-frequency normalised to sum=2 (one weight per class)
        w0 = total / (2.0 * counts[0]) if counts[0] > 0 else 1.0
        w1 = total / (2.0 * counts[1]) if counts[1] > 0 else 1.0
        print(f"  No-Change weight: {w0:.4f}  |  Change weight: {w1:.4f}")
        print(f"  Change pixel ratio: {100*counts[1]/total:.2f}%")
        return torch.tensor([w0, w1], dtype=torch.float32)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname     = self.files[idx]
        pre_path  = os.path.join(self.pre_dir,  fname)
        post_path = os.path.join(self.post_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # ---- Load EO (pre-event, 3 bands) ----
        with rasterio.open(pre_path) as src:
            eo = src.read()                        # [C, H, W]
        eo = np.transpose(eo, (1, 2, 0))[:, :, :3]  # [H, W, 3]

        if self.denoise:
            eo = self._median_denoise(eo.astype(np.float32), ksize=3)

        eo = self.normalize_eo(eo)                 # [H, W, 3] float32 in [0,1]

        # ---- Load SAR (post-event, band 1) ----
        with rasterio.open(post_path) as src:
            sar = src.read(1)                      # [H, W]

        if self.denoise:
            sar = self._median_denoise(sar, ksize=3)

        sar = self.normalize_sar(sar)              # [H, W] float32 in [0,1]

        # ---- Load and remap mask ----
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        mask = self.remap_labels(mask)             # [H, W] uint8 {0,1}

        # ---- Convert to tensors ----
        eo_t   = torch.tensor(eo,   dtype=torch.float32).permute(2, 0, 1)   # [3,H,W]
        sar_t  = torch.tensor(sar,  dtype=torch.float32).unsqueeze(0)        # [1,H,W]
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)        # [1,H,W]

        # ---- Resize ----
        size = (self.image_size, self.image_size)
        eo_t   = TF.resize(eo_t,   size)
        sar_t  = TF.resize(sar_t,  size)
        mask_t = TF.resize(mask_t, size, interpolation=TF.InterpolationMode.NEAREST)

        # ---- Augmentation ----
        if self.augment:
            eo_t, sar_t, mask_t = self.apply_augmentation(eo_t, sar_t, mask_t)

        # ---- ImageNet normalise EO (after augmentation, before returning) ----
        eo_t = self.eo_normalize(eo_t)

        # ---- Concatenate EO + SAR -> [4, H, W] for backward compat ----
        image = torch.cat([eo_t, sar_t], dim=0)   # [4, H, W]

        return image, mask_t.squeeze(0)            # [4,H,W], [H,W]