"""
Confidence Estimation module for CloudClear AI.
Generates pixel-wise uncertainty maps, reliability heatmaps, and tier distributions.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from scipy import ndimage


@dataclass
class ConfidenceReport:
    mean_confidence: float
    high_pct: float     # [0.8, 1.0]
    medium_pct: float   # [0.5, 0.8)
    low_pct: float      # [0.0, 0.5)
    confidence_map: np.ndarray      # 2D float32 array [0, 1]
    colored_heatmap: np.ndarray     # (H, W, 3) uint8 RGB array

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_confidence": self.mean_confidence,
            "high_confidence_pct": self.high_pct,
            "medium_confidence_pct": self.medium_pct,
            "low_confidence_pct": self.low_pct
        }


class ConfidenceEstimator:
    """
    Computes scientific, pixel-level reliability bounds for reconstructed satellite images.
    """

    @staticmethod
    def get_colormap(name: str = 'plasma'):
        try:
            return mpl.colormaps[name]
        except AttributeError:
            return cm.get_cmap(name)

    def estimate(
        self,
        cloud_mask: np.ndarray,
        cloud_prob: np.ndarray,
        change_prob: np.ndarray,
        candidates: Optional[Dict[str, np.ndarray]] = None
    ) -> ConfidenceReport:
        """
        Calculates pixel-wise confidence score.
        """
        H, W = cloud_mask.shape[:2]
        c_mask = cloud_mask.astype(np.float32)
        c_prob = np.clip(cloud_prob.astype(np.float32), 0.0, 1.0)
        ch_prob = np.clip(change_prob.astype(np.float32), 0.0, 1.0)

        # Baseline clear-sky confidence
        conf_map = np.ones((H, W), dtype=np.float32)

        # Cloud opacity attenuation
        cloud_penalty = c_prob * 0.35

        # Change attenuation
        change_penalty = (c_mask * ch_prob) * 0.25

        # Consensus variance penalty
        variance_penalty = np.zeros((H, W), dtype=np.float32)
        if candidates and len(candidates) >= 2:
            stacked = np.stack(list(candidates.values()), axis=0)  # (N, H, W, C)
            cand_std = np.mean(np.std(stacked, axis=0), axis=-1)   # (H, W)
            variance_penalty = np.clip(cand_std * 2.0, 0.0, 0.20)

        conf_map = conf_map - cloud_penalty - change_penalty - (c_mask * variance_penalty)
        conf_map = np.clip(conf_map, 0.10, 1.0)
        conf_map = ndimage.gaussian_filter(conf_map, sigma=1.0)
        conf_map = np.clip(conf_map, 0.0, 1.0).astype(np.float32)

        total_pixels = float(H * W)
        high_mask = (conf_map >= 0.80)
        med_mask = (conf_map >= 0.50) & (conf_map < 0.80)
        low_mask = (conf_map < 0.50)

        high_pct = float(np.sum(high_mask) / total_pixels * 100.0)
        med_pct = float(np.sum(med_mask) / total_pixels * 100.0)
        low_pct = float(np.sum(low_mask) / total_pixels * 100.0)
        mean_conf = float(np.mean(conf_map))

        cmap = self.get_colormap('plasma')
        rgba = cmap(conf_map)
        colored_rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

        return ConfidenceReport(
            mean_confidence=round(mean_conf, 3),
            high_pct=round(high_pct, 2),
            medium_pct=round(med_pct, 2),
            low_pct=round(low_pct, 2),
            confidence_map=conf_map,
            colored_heatmap=colored_rgb
        )
