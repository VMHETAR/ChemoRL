"""
Vision Encoder Backbone for Lung Cancer CT Slice Analysis.
Extracts latent spatial morphological representations (z_t) and estimates
tumor diameter proxy metrics from radiological inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CTScanVisionEncoder(nn.Module):
    """
    Convolutional Vision Encoder converting 2D/3D CT slices to low-dimensional
    radiomic tumor representations (z_t) for RL state conditioning.
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # -> (32, 32, 32)
            ConvBlock(32, 64),            # -> (64, 16, 16)
            ConvBlock(64, 128),           # -> (128, 8, 8)
            ConvBlock(128, 256),          # -> (256, 4, 4)
            nn.AdaptiveAvgPool2d((1, 1))  # -> (256, 1, 1)
        )

        self.latent_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )

        # Auxiliary head estimating visual lesion size proxy (for supervised calibration)
        self.size_estimator = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent vision vector z_t (B, latent_dim)."""
        b = x.size(0)
        feats = self.features(x).view(b, -1)
        latent = self.latent_head(feats)
        return latent

    def predict_size_proxy(self, latent: torch.Tensor) -> torch.Tensor:
        return self.size_estimator(latent)
