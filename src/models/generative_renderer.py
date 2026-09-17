"""
Generative CT Disease Progression Synthesizer.
Takes baseline CT slices and simulation trajectory metrics to generate synthetic
follow-up CT slices visualizing morphological tumor shrinkage, cavitation, or progression.
"""

import numpy as np
import torch
from typing import Tuple

class GenerativeDiseaseRenderer:
    """
    Morphological and generative synthesizer reflecting treatment-induced tumor alterations
    in reconstructed CT image slices across treatment cycles.
    """
    def __init__(self, img_size: int = 64):
        self.img_size = img_size

    def render_cycle_slice(
        self,
        base_slice: np.ndarray,
        baseline_diameter_mm: float,
        current_diameter_mm: float,
        cycle_idx: int
    ) -> np.ndarray:
        """
        Renders an evolved CT slice scaled by the ratio (current_diameter / baseline_diameter).
        """
        if base_slice.ndim == 3:
            img = base_slice[0].copy()
        else:
            img = base_slice.copy()

        h, w = img.shape
        ratio = max(0.0, current_diameter_mm / max(1e-3, baseline_diameter_mm))
        
        # Center of the slice coordinates
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2.0, w / 2.0

        # Primary nodule region is in one of the lung fields
        nodule_cx = (center_x + w * 0.18)
        nodule_cy = center_y
        base_rad = w * 0.09

        current_rad = base_rad * np.sqrt(ratio)

        dist = np.sqrt((x - nodule_cx) ** 2 + (y - nodule_cy) ** 2)

        # Baseline lung background (approx 0.08 HU)
        output_slice = img.copy()

        # Erase baseline nodule location back to parenchyma
        parenchyma_mask = dist <= (base_rad * 1.35)
        output_slice[parenchyma_mask] = 0.08 + 0.02 * np.random.randn(*output_slice[parenchyma_mask].shape)

        # Re-inject current evolved nodule
        if current_diameter_mm > 2.0:
            current_nodule_mask = dist <= current_rad
            # Necrotic/cavitation center if shrinking significantly
            if ratio < 0.5:
                # Cavity core
                cavity_core = dist <= (current_rad * 0.45)
                output_slice[current_nodule_mask] = 0.65 + 0.1 * np.random.randn(*output_slice[current_nodule_mask].shape)
                output_slice[cavity_core] = 0.15
            else:
                output_slice[current_nodule_mask] = 0.75 + 0.12 * np.random.randn(*output_slice[current_nodule_mask].shape)

        # Re-apply thoracic boundary
        outer_mask = ((x - center_x) ** 2) / ((w * 0.44) ** 2) + ((y - center_y) ** 2) / ((h * 0.38) ** 2) <= 1.0
        output_slice[~outer_mask] = 0.0

        return np.clip(output_slice, 0.0, 1.0)
