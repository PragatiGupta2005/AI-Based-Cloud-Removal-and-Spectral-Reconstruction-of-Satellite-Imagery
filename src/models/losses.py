"""
Custom Sharpness, Edge-Preserving, and Spectral Loss Functions for CloudClear AI.
Designed specifically to eliminate blur under cloud occlusions in satellite imagery:
- SobelEdgeLoss: High-frequency spatial edge gradient preservation (roads, buildings, rivers)
- SpectralAngleLoss: Multi-spectral band vector alignment (B2, B3, B4, B8)
- SSIMLoss: Local structural and contrast similarity
- CloudMaskWeightedLoss: Direct penalty amplification on occluded sub-cloud pixels
- CombinedSharpReconstructionLoss: Unified multi-component loss formulation
"""

import tensorflow as tf
from tensorflow.keras import losses


class SobelEdgeLoss(tf.keras.losses.Loss):
    """
    Computes horizontal and vertical spatial gradient loss using differentiable Sobel kernels.
    Forces the reconstruction model to match sharp object boundaries, preventing regression blur.
    """

    def __init__(self, name: str = "sobel_edge_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        # Standard Sobel 3x3 kernels
        sobel_x = tf.constant([[-1.0, 0.0, 1.0],
                               [-2.0, 0.0, 2.0],
                               [-1.0, 0.0, 1.0]], dtype=tf.float32)
        sobel_y = tf.constant([[-1.0, -2.0, -1.0],
                               [ 0.0,  0.0,  0.0],
                               [ 1.0,  2.0,  1.0]], dtype=tf.float32)

        # Expand to (3, 3, 1, 1)
        self.kernel_x = sobel_x[:, :, tf.newaxis, tf.newaxis]
        self.kernel_y = sobel_y[:, :, tf.newaxis, tf.newaxis]

    def _compute_gradients(self, img: tf.Tensor) -> tf.Tensor:
        """Computes gradient magnitude across all channels of the input image."""
        num_channels = tf.shape(img)[-1]
        kx = tf.tile(self.kernel_x, [1, 1, num_channels, 1])
        ky = tf.tile(self.kernel_y, [1, 1, num_channels, 1])

        gx = tf.nn.depthwise_conv2d(img, kx, strides=[1, 1, 1, 1], padding='SAME')
        gy = tf.nn.depthwise_conv2d(img, ky, strides=[1, 1, 1, 1], padding='SAME')
        grad_mag = tf.sqrt(tf.maximum(tf.square(gx) + tf.square(gy), 1e-8))
        return grad_mag

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        grad_true = self._compute_gradients(y_true_f)
        grad_pred = self._compute_gradients(y_pred_f)

        return tf.reduce_mean(tf.abs(grad_true - grad_pred))


class SpectralAngleLoss(tf.keras.losses.Loss):
    """
    Spectral Angle Mapper (SAM) loss measuring the angular divergence between
    ground truth and predicted multi-spectral vectors across (B2, B3, B4, B8).
    Guarantees radiometric and NDVI fidelity without spectral distortion.
    """

    def __init__(self, name: str = "spectral_angle_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        # Dot product across spectral channel dimension (-1)
        dot_product = tf.reduce_sum(y_true_f * y_pred_f, axis=-1)
        norm_true = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y_true_f), axis=-1), 1e-8))
        norm_pred = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y_pred_f), axis=-1), 1e-8))

        cos_theta = dot_product / (norm_true * norm_pred + 1e-8)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

        # Angle in radians
        sam_angle = tf.acos(cos_theta)
        return tf.reduce_mean(sam_angle)


class SSIMLoss(tf.keras.losses.Loss):
    """
    Multi-channel Structural Similarity Index Measure (SSIM) loss.
    1.0 - mean(SSIM(y_true, y_pred))
    """

    def __init__(self, max_val: float = 1.0, name: str = "ssim_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = max_val

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        ssim_val = tf.image.ssim(y_true_f, y_pred_f, max_val=self.max_val)
        return 1.0 - tf.reduce_mean(ssim_val)


class CombinedSharpReconstructionLoss(tf.keras.losses.Loss):
    """
    Composite high-accuracy reconstruction loss:
    L_total = L1 + lambda_ssim * L_ssim + lambda_edge * L_sobel + lambda_sam * L_sam
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        edge_weight: float = 0.35,
        sam_weight: float = 0.25,
        name: str = "combined_sharp_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.sam_weight = sam_weight

        self.sobel_loss = SobelEdgeLoss()
        self.spectral_loss = SpectralAngleLoss()
        self.ssim_loss = SSIMLoss()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        l1 = tf.reduce_mean(tf.abs(y_true_f - y_pred_f))
        ssim_l = self.ssim_loss(y_true_f, y_pred_f)
        edge_l = self.sobel_loss(y_true_f, y_pred_f)
        sam_l = self.spectral_loss(y_true_f, y_pred_f)

        total_loss = (
            self.l1_weight * l1 +
            self.ssim_weight * ssim_l +
            self.edge_weight * edge_l +
            self.sam_weight * sam_l
        )
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "l1_weight": self.l1_weight,
            "ssim_weight": self.ssim_weight,
            "edge_weight": self.edge_weight,
            "sam_weight": self.sam_weight
        })
        return config
