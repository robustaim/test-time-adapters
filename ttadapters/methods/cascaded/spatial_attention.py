import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SpatialAttentionEncoder(nn.Module):
    """
    CNN with spatial attention for extracting domain-specific visual features.

    Key insight:
    - Fog: Uniform pattern across entire image → high attention everywhere
    - Night: Local bright spots (lamps) + dark regions → selective attention

    Output: 16-dimensional feature vector
    """

    def __init__(self):
        super().__init__()
        # Feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 8x8 → 8x8, 32 channels
            nn.ReLU(),
        )

        # Spatial attention
        self.attention_conv = nn.Conv2d(32, 1, 1)  # 32 → 1 attention map

    def forward(self, img):
        """
        Args:
            img: (C, H, W) or (B, C, H, W)
        Returns:
            features: (16,) or (B, 16)
        """
        # Handle batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Aggressive downsample to 8x8
        img_tiny = F.interpolate(img, size=8, mode='bilinear', align_corners=False)

        # Extract features
        features = self.conv(img_tiny)  # (B, 32, 8, 8)

        # Compute spatial attention
        attention = torch.sigmoid(self.attention_conv(features))  # (B, 1, 8, 8)

        # Apply attention and pool
        weighted_features = features * attention  # (B, 32, 8, 8)
        pooled = weighted_features.mean(dim=[2, 3])  # (B, 32)

        # Reduce to 16-dim
        output = pooled[:, :16]  # (B, 16) - use first 16 channels

        if squeeze_output:
            output = output.squeeze(0)  # (16,)

        return output
