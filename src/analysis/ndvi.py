"""
NDVI Analysis and Comparative Vegetation Health evaluation for CloudClear AI.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm


@dataclass
class NDVIReport:
    mean_ndvi: float
    reference_mean_ndvi: float
    ndvi_mae: float
    vegetation_coverage_pct: float
    reconstructed_ndvi: np.ndarray      # 2D float32 array [-1, 1]
    reference_ndvi: Optional[np.ndarray]# 2D float32 array [-1, 1]
    diff_ndvi: np.ndarray               # 2D float32 array [-1, 1]
    colored_ndvi: np.ndarray            # (H, W, 3) uint8 RGB array

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_ndvi": self.mean_ndvi,
            "reference_mean_ndvi": self.reference_mean_ndvi,
            "ndvi_mae": self.ndvi_mae,
            "vegetation_coverage_pct": self.vegetation_coverage_pct
        }


class NDVIAnalyzer:
    """
    Computes Normalized Difference Vegetation Index (NDVI) and analyzes spectral vegetation health.
    """

    @staticmethod
    def get_colormap(name: str = 'RdYlGn'):
        try:
            return mpl.colormaps[name]
        except AttributeError:
            return cm.get_cmap(name)

    @staticmethod
    def calculate_ndvi(image_4band: np.ndarray) -> np.ndarray:
        """
        Calculates NDVI = (NIR - Red) / (NIR + Red).
        Expects 4-band array where Band 2 = Red (B4) and Band 3 = NIR (B8).
        """
        if image_4band.shape[-1] < 4:
            red = image_4band[:, :, min(0, image_4band.shape[-1]-1)].astype(np.float32)
            nir = image_4band[:, :, min(1, image_4band.shape[-1]-1)].astype(np.float32)
        else:
            red = image_4band[:, :, 2].astype(np.float32)
            nir = image_4band[:, :, 3].astype(np.float32)

        denom = nir + red
        denom[denom == 0] = 1e-5

        ndvi = (nir - red) / denom
        ndvi = np.clip(ndvi, -1.0, 1.0)
        return ndvi.astype(np.float32)

    @classmethod
    def colorize_ndvi(cls, ndvi_map: np.ndarray) -> np.ndarray:
        """
        Colorizes NDVI map using standard RdYlGn (Red-Yellow-Green) colormap.
        Values scaled from [-0.2, 0.8] to [0, 1] for visual contrast.
        """
        norm_val = np.clip((ndvi_map + 0.2) / 1.0, 0.0, 1.0)
        cmap = cls.get_colormap('RdYlGn')
        rgba = cmap(norm_val)
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    def analyze(
        self,
        reconstructed_image: np.ndarray,
        reference_image: Optional[np.ndarray] = None
    ) -> NDVIReport:
        """
        Performs comparative NDVI analysis.
        """
        rec_ndvi = self.calculate_ndvi(reconstructed_image)

        if reference_image is not None:
            ref_ndvi = self.calculate_ndvi(reference_image)
            diff_ndvi = rec_ndvi - ref_ndvi
            ndvi_mae = float(np.mean(np.abs(diff_ndvi)))
            ref_mean = float(np.mean(ref_ndvi))
        else:
            ref_ndvi = None
            diff_ndvi = np.zeros_like(rec_ndvi)
            ndvi_mae = 0.0
            ref_mean = float(np.mean(rec_ndvi))

        mean_ndvi = float(np.mean(rec_ndvi))
        veg_mask = (rec_ndvi >= 0.3)
        veg_pct = float(np.sum(veg_mask) / float(rec_ndvi.size) * 100.0)
        colored_rgb = self.colorize_ndvi(rec_ndvi)

        return NDVIReport(
            mean_ndvi=round(mean_ndvi, 3),
            reference_mean_ndvi=round(ref_mean, 3),
            ndvi_mae=round(ndvi_mae, 4),
            vegetation_coverage_pct=round(veg_pct, 2),
            reconstructed_ndvi=rec_ndvi,
            reference_ndvi=ref_ndvi,
            diff_ndvi=diff_ndvi,
            colored_ndvi=colored_rgb
        )
