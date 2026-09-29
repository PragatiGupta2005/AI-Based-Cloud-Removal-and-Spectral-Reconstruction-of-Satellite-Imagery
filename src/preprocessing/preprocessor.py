"""
Preprocessing, radiometric normalization, patch extraction, and stitching module.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy import ndimage


class ImagePreprocessor:
    """
    Handles normalization, band extraction, patch generation, and seamless blending.
    """

    def __init__(self, patch_size: int = 256, stride: int = 256):
        self.patch_size = patch_size
        self.stride = stride

    @staticmethod
    def normalize_radiometric(
        image: np.ndarray,
        p_min: float = 2.0,
        p_max: float = 98.0
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        """
        Applies robust percentile radiometric normalization per band into [0.0, 1.0].
        Returns normalized array and stats dictionary.
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        num_bands = image.shape[-1]
        img_norm = np.zeros_like(image, dtype=np.float32)
        stats = {}

        # If image is already floating-point surface reflectance in [0, 1]
        if image.max() <= 1.0 and image.min() >= 0.0:
            img_norm = np.clip(image.astype(np.float32), 0.0, 1.0)
            for b in range(num_bands):
                stats[f"band_{b}"] = (0.0, 1.0)
            return img_norm, stats

        for b in range(num_bands):
            band_data = image[:, :, b].astype(np.float32)
            valid_mask = np.isfinite(band_data) & (band_data != 0)
            if np.any(valid_mask):
                val_min = float(np.percentile(band_data[valid_mask], p_min))
                val_max = float(np.percentile(band_data[valid_mask], p_max))
            else:
                val_min, val_max = 0.0, 1.0

            if val_max - val_min < 1e-6:
                val_max = val_min + 1.0

            norm_band = np.clip((band_data - val_min) / (val_max - val_min), 0.0, 1.0)
            img_norm[:, :, b] = norm_band
            stats[f"band_{b}"] = (val_min, val_max)

        return img_norm, stats

    @staticmethod
    def extract_rgb_preview(image_4band: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """
        Extracts True Color RGB from 4-band image (Assumes: B0=Blue, B1=Green, B2=Red, B3=NIR).
        Applies GIS-standard 2%-98% percentile contrast stretching for vivid satellite visualization.
        Returns uint8 RGB array (H, W, 3).
        """
        if image_4band.ndim == 2:
            gray = np.clip(image_4band * 255, 0, 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)

        channels = image_4band.shape[-1]
        if channels >= 4:
            # Red=B2, Green=B1, Blue=B0
            rgb = image_4band[:, :, [2, 1, 0]].astype(np.float32)
        elif channels == 3:
            rgb = image_4band[:, :, [0, 1, 2]].astype(np.float32)
        elif channels == 2:
            # SAR VV, VH false-color composite
            vv = image_4band[:, :, 0].astype(np.float32)
            vh = image_4band[:, :, 1].astype(np.float32)
            ratio = np.clip(vv / (vh + 1e-4), 0.0, 1.0)
            rgb = np.stack([vv, vh, ratio], axis=-1)
        else:
            ch = image_4band[:, :, 0].astype(np.float32)
            rgb = np.stack([ch, ch, ch], axis=-1)

        # Scale to [0, 1] if not already
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        if enhance_contrast:
            stretched = np.zeros_like(rgb, dtype=np.float32)
            for c in range(3):
                c_band = rgb[:, :, c]
                valid = c_band[np.isfinite(c_band)]
                if len(valid) > 50:
                    p2 = np.percentile(valid, 2.0)
                    p98 = np.percentile(valid, 98.0)
                    if p98 - p2 > 1e-4:
                        c_band = np.clip((c_band - p2) / (p98 - p2), 0.0, 1.0)
                stretched[:, :, c] = c_band
            rgb = stretched

        rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgb_uint8

    def extract_patches(
        self,
        image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], Tuple[int, int]]:
        """
        Extracts fixed-size patches (e.g. 256x256) with padding.
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        H, W, C = image.shape
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='reflect'
            )
        else:
            padded = image.copy()

        pH, pW, _ = padded.shape
        patches = []
        coords = []

        for y in range(0, pH, self.stride):
            for x in range(0, pW, self.stride):
                y_end = min(y + self.patch_size, pH)
                x_end = min(x + self.patch_size, pW)
                y_start = y_end - self.patch_size
                x_start = x_end - self.patch_size

                patch = padded[y_start:y_end, x_start:x_end, :]
                patches.append(patch)
                coords.append((y_start, y_end, x_start, x_end))

        return patches, coords, (pH, pW)

    def stitch_patches(
        self,
        patches: List[np.ndarray],
        coords: List[Tuple[int, int, int, int]],
        padded_shape: Tuple[int, int],
        orig_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Stitches patches back into a continuous array, cropping padding.
        """
        pH, pW = padded_shape
        orig_H, orig_W = orig_shape
        C = patches[0].shape[-1] if patches[0].ndim == 3 else 1

        stitched = np.zeros((pH, pW, C), dtype=np.float32)
        weight_map = np.zeros((pH, pW, C), dtype=np.float32)

        for patch, (y1, y2, x1, x2) in zip(patches, coords):
            if patch.ndim == 2:
                patch = patch[:, :, np.newaxis]
            stitched[y1:y2, x1:x2, :] += patch
            weight_map[y1:y2, x1:x2, :] += 1.0

        weight_map[weight_map == 0] = 1.0
        result = stitched / weight_map
        result_cropped = result[:orig_H, :orig_W, :]

        return result_cropped if C > 1 else result_cropped.squeeze(-1)
