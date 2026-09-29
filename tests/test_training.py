"""
Unit Tests for CloudClear AI Training & Loss Functions.
Verifies multi-component loss calculations, dataset patch extraction, and model training.
"""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.losses import (
    SobelEdgeLoss,
    SpectralAngleLoss,
    SSIMLoss,
    CombinedSharpReconstructionLoss
)
from src.models.mrr_reconstructor import build_mrr_network, MRRReconstructionModel
from src.models.trainer import SatellitePatchDataset, ModelTrainer


class TestTrainingAndLosses(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.save_dir = os.path.join(self.base_dir, "models", "saved_models_test")
        os.makedirs(self.save_dir, exist_ok=True)

        # Synthetic test batch (B=2, H=64, W=64, C=4)
        np.random.seed(42)
        self.y_true = tf.constant(np.random.rand(2, 64, 64, 4), dtype=tf.float32)
        # Prediction with slight blur/noise
        self.y_pred = tf.clip_by_value(self.y_true + 0.05 * tf.random.normal((2, 64, 64, 4)), 0.0, 1.0)

    def test_01_sobel_edge_loss(self):
        loss_fn = SobelEdgeLoss()
        loss_val = loss_fn(self.y_true, self.y_pred)
        self.assertIsInstance(loss_val, tf.Tensor)
        self.assertGreaterEqual(float(loss_val.numpy()), 0.0)

        # Exact match should have zero edge loss
        zero_loss = loss_fn(self.y_true, self.y_true)
        self.assertAlmostEqual(float(zero_loss.numpy()), 0.0, places=4)

    def test_02_spectral_angle_loss(self):
        loss_fn = SpectralAngleLoss()
        loss_val = loss_fn(self.y_true, self.y_pred)
        self.assertIsInstance(loss_val, tf.Tensor)
        self.assertGreaterEqual(float(loss_val.numpy()), 0.0)

        # Collinear vectors should have near-zero angular loss (< 0.01 rad)
        zero_loss = loss_fn(self.y_true, self.y_true * 2.0)
        self.assertAlmostEqual(float(zero_loss.numpy()), 0.0, places=2)

    def test_03_ssim_loss(self):
        loss_fn = SSIMLoss()
        loss_val = loss_fn(self.y_true, self.y_pred)
        self.assertIsInstance(loss_val, tf.Tensor)
        self.assertGreaterEqual(float(loss_val.numpy()), 0.0)
        self.assertLessEqual(float(loss_val.numpy()), 1.0)

        # Identical images should have SSIM loss ~ 0 (SSIM = 1)
        zero_loss = loss_fn(self.y_true, self.y_true)
        self.assertAlmostEqual(float(zero_loss.numpy()), 0.0, places=3)

    def test_04_combined_sharp_loss(self):
        loss_fn = CombinedSharpReconstructionLoss(
            l1_weight=1.0,
            ssim_weight=0.5,
            edge_weight=0.35,
            sam_weight=0.25
        )
        total_loss = loss_fn(self.y_true, self.y_pred)
        self.assertIsInstance(total_loss, tf.Tensor)
        self.assertGreater(float(total_loss.numpy()), 0.0)

    def test_05_mrr_network_architecture(self):
        model = build_mrr_network(input_shape=(128, 128, 4))
        self.assertEqual(len(model.inputs), 4)
        self.assertEqual(len(model.outputs), 3)

        # Test forward pass with dummy inputs
        b = 2
        in_c = tf.random.uniform((b, 128, 128, 4))
        in_h = tf.random.uniform((b, 128, 128, 4))
        in_s = tf.random.uniform((b, 128, 128, 2))
        in_m = tf.zeros((b, 128, 128, 1))

        out_c1, out_c2, out_c3 = model([in_c, in_h, in_s, in_m])
        self.assertEqual(out_c1.shape, (b, 128, 128, 4))
        self.assertEqual(out_c2.shape, (b, 128, 128, 4))
        self.assertEqual(out_c3.shape, (b, 128, 128, 4))

    def test_06_synthetic_patch_dataset_and_trainer(self):
        # Create synthetic small dataset
        synth_patches = {
            "cloudy": np.random.rand(6, 64, 64, 4).astype(np.float32),
            "clear": np.random.rand(6, 64, 64, 4).astype(np.float32),
            "historical": np.random.rand(6, 64, 64, 4).astype(np.float32),
            "sar": np.random.rand(6, 64, 64, 2).astype(np.float32),
            "mask": np.zeros((6, 64, 64, 1), dtype=np.float32)
        }

        trainer = ModelTrainer(save_dir=self.save_dir, patch_size=64)
        summary = trainer.train_reconstruction_model(
            dataset_patches=synth_patches,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3
        )

        self.assertIn("final_val_loss", summary)
        self.assertIn("weights_path", summary)
        self.assertTrue(os.path.exists(summary["weights_path"]))


if __name__ == "__main__":
    unittest.main()
