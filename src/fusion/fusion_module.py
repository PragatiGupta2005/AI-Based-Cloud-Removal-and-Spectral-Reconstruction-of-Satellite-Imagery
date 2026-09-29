"""
Multi-Modal Feature Fusion utilities for CloudClear AI.
"""

from typing import Dict, Any, Tuple
import numpy as np


class MultiModalFusionModule:
    """
    Combines optical, SAR, and temporal information across spectral channels.
    """

    @staticmethod
    def fuse_features(
        optical: np.ndarray,
        historical: np.ndarray,
        sar: np.ndarray,
        weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Blends multi-modal arrays according to dynamic weighting parameters.
        """
        w_hist = weights.get("historical", 0.45)
        w_sar = weights.get("sar", 0.45)
        w_opt = weights.get("optical", 0.10)

        # Normalize weights
        total_w = w_hist + w_sar + w_opt
        w_hist /= total_w
        w_sar /= total_w
        w_opt /= total_w

        sar_expanded = np.repeat(np.mean(sar, axis=-1, keepdims=True), optical.shape[-1], axis=-1)
        fused = w_opt * optical + w_hist * historical + w_sar * sar_expanded
        return np.clip(fused, 0.0, 1.0).astype(np.float32)
