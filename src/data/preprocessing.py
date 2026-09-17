"""
Image and Clinical Preprocessing Utilities for CT Radiomics.
Provides windowing (Lung Window: W=1500, L=-600), normalization, and spatial data augmentations.
"""

import numpy as np
import torch
import torch.nn.functional as F

def apply_lung_window(hu_array: np.ndarray, window_width: float = 1500.0, window_level: float = -600.0) -> np.ndarray:
    """
    Transforms raw Hounsfield Unit (HU) values to standard clinical lung window [0.0, 1.0].
    """
    lower = window_level - (window_width / 2.0)
    upper = window_level + (window_width / 2.0)
    windowed = (hu_array - lower) / (upper - lower)
    return np.clip(windowed, 0.0, 1.0).astype(np.float32)

def augment_ct_slice(img: torch.Tensor, training: bool = True) -> torch.Tensor:
    """
    Applies benign spatial transformations (random affine rotation, slight jitter) preserving tumor topology.
    """
    if not training:
        return img
    
    # 50% random horizontal flip
    if torch.rand(1).item() > 0.5:
        img = torch.flip(img, dims=[-1])

    return img
