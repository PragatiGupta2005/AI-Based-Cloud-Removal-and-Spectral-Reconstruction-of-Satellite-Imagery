"""
Multi-Hypothesis Reconstruction Network (MRR) for CloudClear AI.
Generates 3 high-definition, sharp reconstruction candidate scenes:
- C1: Historical Dominant (Multi-scale temporal texture mapping for stable terrain)
- C2: SAR Dominant (High-frequency radar edge mapping for changed or flooded areas)
- C3: Adaptive Cross-Attention Fusion (Joint optical-temporal-radar multi-scale synthesis)
Features deep multi-scale U-Net skip connections and high-frequency edge injection.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy import ndimage
from .cross_attention import CrossAttentionFusionLayer


def build_mrr_network(input_shape=(256, 256, 4)) -> tf.keras.Model:
    """
    Builds the Deep Multi-Scale Residual Skip Connection MRR Network.
    Carries spatial resolutions (256x256, 128x128, 64x64, 32x32) directly across
    encoder-decoder pathways to guarantee sharp edge preservation and eliminate blur.
    """
    inp_cloudy = layers.Input(shape=input_shape, name="cloudy_optical")
    inp_hist = layers.Input(shape=input_shape, name="historical_optical")
    inp_sar = layers.Input(shape=(input_shape[0], input_shape[1], 2), name="sar_radar")
    inp_mask = layers.Input(shape=(input_shape[0], input_shape[1], 1), name="cloud_mask")

    # =========================================================================
    # ENCODERS (Level 1: 256x256, Level 2: 128x128, Level 3: 64x64, Bottleneck: 32x32)
    # =========================================================================

    # --- Level 1 (256 x 256) ---
    c1_c = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name="c1_c")(inp_cloudy)
    c1_h = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name="c1_h")(inp_hist)
    c1_s = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name="c1_s")(inp_sar)
    p1_c = layers.MaxPooling2D((2, 2))(c1_c)
    p1_h = layers.MaxPooling2D((2, 2))(c1_h)
    p1_s = layers.MaxPooling2D((2, 2))(c1_s)

    # --- Level 2 (128 x 128) ---
    c2_c = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="c2_c")(p1_c)
    c2_h = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="c2_h")(p1_h)
    c2_s = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="c2_s")(p1_s)
    p2_c = layers.MaxPooling2D((2, 2))(c2_c)
    p2_h = layers.MaxPooling2D((2, 2))(c2_h)
    p2_s = layers.MaxPooling2D((2, 2))(c2_s)

    # --- Level 3 (64 x 64) ---
    c3_c = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="c3_c")(p2_c)
    c3_h = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="c3_h")(p2_h)
    c3_s = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="c3_s")(p2_s)
    p3_c = layers.MaxPooling2D((2, 2))(c3_c)
    p3_h = layers.MaxPooling2D((2, 2))(c3_h)
    p3_s = layers.MaxPooling2D((2, 2))(c3_s)

    # --- Bottleneck (32 x 32) ---
    b_c = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="b_c")(p3_c)
    b_h = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="b_h")(p3_h)
    b_s = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="b_s")(p3_s)
    m_pool = layers.MaxPooling2D((8, 8))(inp_mask)

    # =========================================================================
    # HEAD 1: CANDIDATE C1 (HISTORICAL DOMINANT) WITH MULTI-SCALE SKIPS
    # =========================================================================
    h1_bottleneck = layers.concatenate([b_h, b_c, m_pool])
    h1_b_conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(h1_bottleneck)

    # Upsample to 64x64 + Level 3 Skip
    h1_up3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(h1_b_conv)
    h1_skip3 = layers.concatenate([h1_up3, c3_h, c3_c])
    h1_d3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(h1_skip3)

    # Upsample to 128x128 + Level 2 Skip
    h1_up2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(h1_d3)
    h1_skip2 = layers.concatenate([h1_up2, c2_h, c2_c])
    h1_d2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(h1_skip2)

    # Upsample to 256x256 + Level 1 Skip
    h1_up1 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(h1_d2)
    h1_skip1 = layers.concatenate([h1_up1, c1_h, c1_c])
    h1_d1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(h1_skip1)
    out_c1 = layers.Conv2D(4, (1, 1), activation='sigmoid', name="candidate_c1")(h1_d1)

    # =========================================================================
    # HEAD 2: CANDIDATE C2 (SAR RADAR DOMINANT) WITH MULTI-SCALE SKIPS
    # =========================================================================
    h2_bottleneck = layers.concatenate([b_s, b_c, m_pool])
    h2_b_conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(h2_bottleneck)

    # Upsample to 64x64 + Level 3 Skip
    h2_up3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(h2_b_conv)
    h2_skip3 = layers.concatenate([h2_up3, c3_s, c3_c])
    h2_d3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(h2_skip3)

    # Upsample to 128x128 + Level 2 Skip
    h2_up2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(h2_d3)
    h2_skip2 = layers.concatenate([h2_up2, c2_s, c2_c])
    h2_d2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(h2_skip2)

    # Upsample to 256x256 + Level 1 Skip
    h2_up1 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(h2_d2)
    h2_skip1 = layers.concatenate([h2_up1, c1_s, c1_c])
    h2_d1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(h2_skip1)
    out_c2 = layers.Conv2D(4, (1, 1), activation='sigmoid', name="candidate_c2")(h2_d1)

    # =========================================================================
    # HEAD 3: CANDIDATE C3 (ADAPTIVE CROSS-ATTENTION FUSION)
    # =========================================================================
    context_latent = layers.concatenate([b_h, b_s, m_pool])
    context_proj = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(context_latent)

    cross_att = CrossAttentionFusionLayer(d_model=64, num_heads=4)(b_c, context_proj)
    h3_bottleneck = layers.concatenate([cross_att, b_h, b_s])
    h3_b_conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(h3_bottleneck)

    # Upsample to 64x64 + Cross Skip (Hist + SAR + Cloudy)
    h3_up3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(h3_b_conv)
    h3_skip3 = layers.concatenate([h3_up3, c3_h, c3_s, c3_c])
    h3_d3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(h3_skip3)

    # Upsample to 128x128 + Cross Skip
    h3_up2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(h3_d3)
    h3_skip2 = layers.concatenate([h3_up2, c2_h, c2_s, c2_c])
    h3_d2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(h3_skip2)

    # Upsample to 256x256 + Cross Skip
    h3_up1 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(h3_d2)
    h3_skip1 = layers.concatenate([h3_up1, c1_h, c1_s, c1_c])
    h3_d1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(h3_skip1)
    out_c3 = layers.Conv2D(4, (1, 1), activation='sigmoid', name="candidate_c3")(h3_d1)

    model = models.Model(
        inputs=[inp_cloudy, inp_hist, inp_sar, inp_mask],
        outputs=[out_c1, out_c2, out_c3],
        name="MRR_Reconstructor"
    )
    return model


class MRRReconstructionModel:
    """
    Multi-Hypothesis Reconstruction pipeline wrapper with High-Frequency Detail Transfer.
    """

    def __init__(self, patch_size: int = 256, weights_path: Optional[str] = None):
        self.patch_size = patch_size
        self.model = build_mrr_network(input_shape=(patch_size, patch_size, 4))
        self.weights_path = weights_path
        self._load_saved_weights_if_available()

    def _load_saved_weights_if_available(self):
        """Attempts to load trained weights from disk if available."""
        if self.weights_path and os.path.exists(self.weights_path):
            try:
                self.model.load_weights(self.weights_path)
                return
            except Exception as e:
                pass

        # Standard default weight location
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_paths = [
            os.path.join(base_dir, "models", "saved_models", "mrr_reconstructor.weights.h5"),
            os.path.join(base_dir, "models", "saved_models", "mrr_reconstructor.h5")
        ]
        for p in default_paths:
            if os.path.exists(p):
                try:
                    self.model.load_weights(p)
                    self.weights_path = p
                    break
                except Exception:
                    continue

    def reconstruct_all(
        self,
        cloudy_optical: np.ndarray,
        hist_optical: np.ndarray,
        sar_image: np.ndarray,
        cloud_mask: np.ndarray,
        cloud_prob: Optional[np.ndarray] = None,
        change_prob: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generates 3 high-definition reconstruction candidates:
        C1: Historical Dominant (Local Radiometric & Atmospheric Matched + Edge-Guided Transfer)
        C2: SAR Dominant (High-Frequency Radar Edge & Geometric Infilling)
        C3: Adaptive Cross-Attention Fusion (Joint Multi-Modal Neural Synthesis)
        """
        H, W = cloudy_optical.shape[:2]

        if cloudy_optical.shape[-1] < 4:
            cloudy_optical = np.pad(cloudy_optical, ((0, 0), (0, 0), (0, 4 - cloudy_optical.shape[-1])), mode='edge')
        if hist_optical.shape[-1] < 4:
            hist_optical = np.pad(hist_optical, ((0, 0), (0, 0), (0, 4 - hist_optical.shape[-1])), mode='edge')
        if sar_image.shape[-1] < 2:
            sar_image = np.repeat(sar_image, 2, axis=-1)

        mask_2d = cloud_mask.astype(np.float32)
        # Multi-scale smooth alpha mask for seamless boundary feathering
        smooth_mask = ndimage.gaussian_filter(mask_2d, sigma=1.2)[:, :, np.newaxis]

        # 1. High-Fidelity Local Radiometric Color & Illumination Matching
        hist_matched = hist_optical.copy()
        clear_mask = (mask_2d == 0)
        if np.sum(clear_mask) > 100:
            for b in range(4):
                c_vals = cloudy_optical[:, :, b][clear_mask]
                h_vals = hist_optical[:, :, b][clear_mask]
                c_mean, c_std = float(np.mean(c_vals)), float(np.std(c_vals)) + 1e-4
                h_mean, h_std = float(np.mean(h_vals)), float(np.std(h_vals)) + 1e-4
                matched_band = (hist_optical[:, :, b] - h_mean) * (c_std / h_std) + c_mean
                hist_matched[:, :, b] = np.clip(matched_band, 0.0, 1.0)

        # 2. High-Frequency SAR Structural & Edge Feature Extraction (Roads, Buildings, Watercourses)
        sar_vv = sar_image[:, :, 0]
        sar_sobel_x = ndimage.sobel(sar_vv, axis=0)
        sar_sobel_y = ndimage.sobel(sar_vv, axis=1)
        sar_edges = np.hypot(sar_sobel_x, sar_sobel_y)
        sar_edges_norm = np.clip(sar_edges / (np.percentile(sar_edges, 98) + 1e-5), 0.0, 1.0)[:, :, np.newaxis]

        # High-pass texture sharpening from historical baseline
        hist_blur = ndimage.gaussian_filter(hist_matched, sigma=(1.2, 1.2, 0.0))
        hist_high_pass = hist_matched - hist_blur

        # High-pass texture from SAR radar backscatter (dielectric surface roughness)
        sar_tex = np.mean(sar_image, axis=-1, keepdims=True)
        sar_tex_blur = ndimage.gaussian_filter(sar_tex, sigma=(2.0, 2.0, 0.0))
        sar_high_pass = sar_tex - sar_tex_blur

        # Sharp SAR-synthesized candidate (preserves optical radiometric mean while injecting radar structural edges)
        sar_synthesized = np.clip(
            hist_matched + 0.20 * sar_high_pass + 0.08 * (sar_edges_norm - np.mean(sar_edges_norm)) + 0.30 * hist_high_pass,
            0.0, 1.0
        )

        # 3. Candidate Infilling
        c1_phys = np.clip((1.0 - smooth_mask) * cloudy_optical + smooth_mask * (hist_matched + 0.30 * hist_high_pass), 0.0, 1.0)
        c2_phys = np.clip((1.0 - smooth_mask) * cloudy_optical + smooth_mask * sar_synthesized, 0.0, 1.0)

        # Adaptive synthesis based on detected ground change
        if change_prob is None:
            change_prob = np.zeros((H, W), dtype=np.float32)
        ch_weight = np.clip(change_prob[:, :, np.newaxis] * 0.50, 0.0, 0.60)
        adaptive_infill = (1.0 - ch_weight) * (hist_matched + 0.25 * hist_high_pass) + ch_weight * sar_synthesized
        c3_phys = np.clip((1.0 - smooth_mask) * cloudy_optical + smooth_mask * adaptive_infill, 0.0, 1.0)

        # 4. Neural Network Inference
        c_in = tf.image.resize(cloudy_optical, (self.patch_size, self.patch_size))[np.newaxis, ...]
        h_in = tf.image.resize(hist_optical, (self.patch_size, self.patch_size))[np.newaxis, ...]
        s_in = tf.image.resize(sar_image, (self.patch_size, self.patch_size))[np.newaxis, ...]
        m_in = tf.image.resize(mask_2d[:, :, np.newaxis], (self.patch_size, self.patch_size))[np.newaxis, ...]

        out_c1_nn, out_c2_nn, out_c3_nn = self.model.predict([c_in, h_in, s_in, m_in], verbose=0)
        c1_nn_res = tf.image.resize(out_c1_nn[0], (H, W)).numpy()
        c2_nn_res = tf.image.resize(out_c2_nn[0], (H, W)).numpy()
        c3_nn_res = tf.image.resize(out_c3_nn[0], (H, W)).numpy()

        # If trained weights are loaded, blend neural prediction with high-pass edge enhancement
        if self.weights_path is not None and os.path.exists(self.weights_path):
            c1_final = np.clip((1.0 - smooth_mask) * cloudy_optical + smooth_mask * (c1_nn_res * 0.75 + hist_matched * 0.25 + 0.25 * hist_high_pass), 0.0, 1.0)
            c2_final = np.clip((1.0 - smooth_mask) * cloudy_optical + smooth_mask * (c2_nn_res * 0.75 + sar_synthesized * 0.25), 0.0, 1.0)
            c3_final = np.clip((1.0 - smooth_mask) * cloudy_optical + smooth_mask * (c3_nn_res * 0.80 + adaptive_infill * 0.20 + 0.20 * sar_edges_norm), 0.0, 1.0)
        return {
            "C1": c1_final.astype(np.float32),
            "C2": c2_final.astype(np.float32),
            "C3": c3_final.astype(np.float32)
        }
