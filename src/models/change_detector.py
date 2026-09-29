"""
Siamese Change Detection Network for CloudClear AI.
Compares Current Optical, Historical Reference Optical, and Sentinel-1 SAR imagery
to identify genuine surface change and prevent temporal hallucinations.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy import ndimage


def build_siamese_change_detector(input_shape=(256, 256, 4)) -> tf.keras.Model:
    """
    Builds a Siamese CNN change detection architecture.
    """
    input_curr = layers.Input(shape=input_shape, name="curr_optical")
    input_hist = layers.Input(shape=input_shape, name="hist_optical")
    input_sar = layers.Input(shape=(input_shape[0], input_shape[1], 2), name="sar_input")

    # Shared feature branch
    def feature_branch():
        inp = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        return models.Model(inp, x)

    branch = feature_branch()
    feat_curr = branch(input_curr)
    feat_hist = branch(input_hist)

    # SAR branch
    s = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_sar)
    s = layers.MaxPooling2D((2, 2))(s)
    s = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(s)

    # Difference & Correlation features
    diff = layers.subtract([feat_curr, feat_hist])
    abs_diff = layers.Lambda(lambda t: tf.abs(t))(diff)
    fused = layers.concatenate([feat_curr, feat_hist, abs_diff, s])

    # Decoder
    d = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(fused)
    d = layers.UpSampling2D((2, 2))(d)
    d = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(d)
    out = layers.Conv2D(1, (1, 1), activation='sigmoid', name="change_prob")(d)

    model = models.Model(inputs=[input_curr, input_hist, input_sar], outputs=out, name="SiameseChangeDetector")
    return model


class ChangeDetectionModel:
    """
    Temporal Change Detection module for CloudClear AI.
    """

    def __init__(self, patch_size: int = 256):
        self.patch_size = patch_size
        self.model = build_siamese_change_detector(input_shape=(patch_size, patch_size, 4))

    def predict(
        self,
        curr_optical: np.ndarray,
        hist_optical: np.ndarray,
        sar_image: Optional[np.ndarray] = None,
        cloud_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detects surface changes between current and historical observations.

        Returns:
            {
                "change_probability": 2D float32 array [0.0, 1.0],
                "change_category_map": 2D uint8 array (0=Stable, 1=Moderate, 2=Changed),
                "stable_area": float (percentage),
                "moderate_area": float (percentage),
                "changed_area": float (percentage),
                "change_score": float (global change score 0-1)
            }
        """
        H, W = curr_optical.shape[:2]

        # Ensure 4-band opticals
        if curr_optical.shape[-1] < 4:
            curr_optical = np.pad(curr_optical, ((0, 0), (0, 0), (0, 4 - curr_optical.shape[-1])), mode='edge')
        if hist_optical.shape[-1] < 4:
            hist_optical = np.pad(hist_optical, ((0, 0), (0, 0), (0, 4 - hist_optical.shape[-1])), mode='edge')

        # Ensure 2-band SAR
        if sar_image is None:
            sar_image = np.zeros((H, W, 2), dtype=np.float32)
        elif sar_image.shape[-1] == 1:
            sar_image = np.repeat(sar_image, 2, axis=-1)
        elif sar_image.shape[-1] > 2:
            sar_image = sar_image[:, :, :2]

        # Spectral difference calculation
        spectral_diff = np.mean(np.abs(curr_optical[:, :, :4] - hist_optical[:, :, :4]), axis=-1)

        # SAR backscatter difference (structural change)
        sar_diff = np.mean(sar_image, axis=-1)

        # Neural pass
        c_in = tf.image.resize(curr_optical, (self.patch_size, self.patch_size))[np.newaxis, ...]
        h_in = tf.image.resize(hist_optical, (self.patch_size, self.patch_size))[np.newaxis, ...]
        s_in = tf.image.resize(sar_image, (self.patch_size, self.patch_size))[np.newaxis, ...]

        nn_out = self.model.predict([c_in, h_in, s_in], verbose=0)[0]
        nn_prob = tf.image.resize(nn_out, (H, W)).numpy().squeeze(-1)

        # Fuse physical difference with neural feature difference
        change_prob = np.clip(0.5 * spectral_diff * 2.0 + 0.3 * nn_prob + 0.2 * (sar_diff * 0.5), 0.0, 1.0)
        change_prob = ndimage.gaussian_filter(change_prob, sigma=1.2)

        # Categorization: 0=Stable (<=0.35), 1=Moderate (0.35-0.65), 2=Changed (>0.65)
        cat_map = np.zeros((H, W), dtype=np.uint8)
        cat_map[(change_prob > 0.35) & (change_prob <= 0.65)] = 1
        cat_map[change_prob > 0.65] = 2

        total_pixels = float(H * W)
        stable_pct = float(np.sum(cat_map == 0) / total_pixels * 100.0)
        mod_pct = float(np.sum(cat_map == 1) / total_pixels * 100.0)
        changed_pct = float(np.sum(cat_map == 2) / total_pixels * 100.0)

        global_score = float(np.mean(change_prob))

        return {
            "change_probability": change_prob.astype(np.float32),
            "change_category_map": cat_map,
            "stable_area": round(stable_pct, 2),
            "moderate_area": round(mod_pct, 2),
            "changed_area": round(changed_pct, 2),
            "change_score": round(global_score, 4)
        }
