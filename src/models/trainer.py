"""
Satellite AI Model Trainer for CloudClear AI.
Handles patch extraction, multi-modal data augmentation, and high-accuracy training for:
1. Attention U-Net Cloud & Shadow Detector (Dice + BCE Loss)
2. Multi-Scale MRR Reconstructor (L1 + SSIM + Sobel Edge + SAM Loss)
"""

import os
import glob
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

from .losses import CombinedSharpReconstructionLoss, SobelEdgeLoss, SpectralAngleLoss, SSIMLoss
from .cloud_detector import build_attention_unet, CloudDetectionModel
from .mrr_reconstructor import build_mrr_network, MRRReconstructionModel
from ..preprocessing.data_loader import GeoTIFFLoader


class SatellitePatchDataset:
    """
    Extracts multi-modal patch pairs from regional GeoTIFF scenes and applies augmentations.
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: int = 256,
        stride: int = 256,
        max_patches_per_scene: int = 6
    ):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches_per_scene

        self.cloudy_dir = os.path.join(data_dir, "cloudy")
        self.clear_dir = os.path.join(data_dir, "clear")
        self.hist_dir = os.path.join(data_dir, "historical")
        self.sar_dir = os.path.join(data_dir, "sar")

    def _augment_patch(
        self,
        cloudy: np.ndarray,
        hist: np.ndarray,
        sar: np.ndarray,
        mask: np.ndarray,
        clear: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Applies spatial and radiometric augmentations."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            cloudy = np.fliplr(cloudy)
            hist = np.fliplr(hist)
            sar = np.fliplr(sar)
            mask = np.fliplr(mask)
            clear = np.fliplr(clear)

        # Random vertical flip
        if np.random.rand() > 0.5:
            cloudy = np.flipud(cloudy)
            hist = np.flipud(hist)
            sar = np.flipud(sar)
            mask = np.flipud(mask)
            clear = np.flipud(clear)

        # Random 90-degree rotations
        k = np.random.randint(0, 4)
        if k > 0:
            cloudy = np.rot90(cloudy, k=k)
            hist = np.rot90(hist, k=k)
            sar = np.rot90(sar, k=k)
            mask = np.rot90(mask, k=k)
            clear = np.rot90(clear, k=k)

        # Random slight brightness / contrast jitter on cloudy
        jitter = float(np.random.uniform(0.95, 1.05))
        cloudy = np.clip(cloudy * jitter, 0.0, 1.0)

        return (
            cloudy.astype(np.float32),
            hist.astype(np.float32),
            sar.astype(np.float32),
            mask.astype(np.float32),
            clear.astype(np.float32)
        )

    def extract_patches(self) -> Dict[str, np.ndarray]:
        """
        Scans datasets and extracts aligned training patches.
        """
        cloudy_files = sorted(glob.glob(os.path.join(self.cloudy_dir, "*.tif")))
        if not cloudy_files:
            return {"cloudy": np.empty((0, self.patch_size, self.patch_size, 4), dtype=np.float32)}

        all_cloudy = []
        all_hist = []
        all_sar = []
        all_mask = []
        all_clear = []

        loader = GeoTIFFLoader()

        for c_path in cloudy_files:
            base_name = os.path.basename(c_path)
            # Find matching clear, hist, sar
            prefix = base_name.replace("_cloudy.tif", "")
            cl_path = os.path.join(self.clear_dir, f"{prefix}_clear.tif")
            h_path = os.path.join(self.hist_dir, f"{prefix}_historical.tif")
            s_path = os.path.join(self.sar_dir, f"{prefix}_sar.tif")

            if not os.path.exists(cl_path):
                continue

            try:
                c_img, _ = loader.load_raster(c_path)
                cl_img, _ = loader.load_raster(cl_path)
                h_img = loader.load_raster(h_path)[0] if os.path.exists(h_path) else cl_img.copy()
                s_img = loader.load_raster(s_path)[0] if os.path.exists(s_path) else np.zeros((c_img.shape[0], c_img.shape[1], 2), dtype=np.float32)
            except Exception:
                continue

            # Ensure 4 bands for optical and 2 bands for SAR
            if c_img.shape[-1] < 4:
                c_img = np.pad(c_img, ((0, 0), (0, 0), (0, 4 - c_img.shape[-1])), mode='edge')
            if cl_img.shape[-1] < 4:
                cl_img = np.pad(cl_img, ((0, 0), (0, 0), (0, 4 - cl_img.shape[-1])), mode='edge')
            if h_img.shape[-1] < 4:
                h_img = np.pad(h_img, ((0, 0), (0, 0), (0, 4 - h_img.shape[-1])), mode='edge')
            if s_img.shape[-1] < 2:
                s_img = np.repeat(s_img[:, :, :1], 2, axis=-1)

            # Generate binary cloud mask from difference between cloudy and clear
            diff = np.mean(np.abs(c_img - cl_img), axis=-1)
            c_mask = (diff > 0.12).astype(np.float32)[:, :, np.newaxis]

            H, W = c_img.shape[:2]
            patch_cnt = 0

            # Sliding window patch extraction
            for y in range(0, H - self.patch_size + 1, self.stride):
                for x in range(0, W - self.patch_size + 1, self.stride):
                    if patch_cnt >= self.max_patches:
                        break

                    p_c = c_img[y:y+self.patch_size, x:x+self.patch_size]
                    p_cl = cl_img[y:y+self.patch_size, x:x+self.patch_size]
                    p_h = h_img[y:y+self.patch_size, x:x+self.patch_size]
                    p_s = s_img[y:y+self.patch_size, x:x+self.patch_size]
                    p_m = c_mask[y:y+self.patch_size, x:x+self.patch_size]

                    # Add original patch
                    all_cloudy.append(p_c)
                    all_clear.append(p_cl)
                    all_hist.append(p_h)
                    all_sar.append(p_s)
                    all_mask.append(p_m)
                    patch_cnt += 1

                    # Add augmented patch
                    aug_c, aug_h, aug_s, aug_m, aug_cl = self._augment_patch(p_c, p_h, p_s, p_m, p_cl)
                    all_cloudy.append(aug_c)
                    all_clear.append(aug_cl)
                    all_hist.append(aug_h)
                    all_sar.append(aug_s)
                    all_mask.append(aug_m)
                    patch_cnt += 1

        return {
            "cloudy": np.array(all_cloudy, dtype=np.float32),
            "clear": np.array(all_clear, dtype=np.float32),
            "historical": np.array(all_hist, dtype=np.float32),
            "sar": np.array(all_sar, dtype=np.float32),
            "mask": np.array(all_mask, dtype=np.float32)
        }


class ModelTrainer:
    """
    Orchestrates deep learning training for CloudClear AI with high-accuracy multi-component losses.
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        patch_size: int = 256
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.save_dir = save_dir or os.path.join(base_dir, "models", "saved_models")
        os.makedirs(self.save_dir, exist_ok=True)
        self.patch_size = patch_size

    def train_reconstruction_model(
        self,
        dataset_patches: Dict[str, np.ndarray],
        epochs: int = 15,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        edge_weight: float = 0.35,
        sam_weight: float = 0.25,
        progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None
    ) -> Dict[str, Any]:
        """
        Trains the Multi-Scale MRR Reconstructor using CombinedSharpReconstructionLoss.
        """
        cloudy = dataset_patches["cloudy"]
        clear = dataset_patches["clear"]
        hist = dataset_patches["historical"]
        sar = dataset_patches["sar"]
        mask = dataset_patches["mask"]

        num_samples = len(cloudy)
        if num_samples == 0:
            raise ValueError("No training patches available to train the model.")

        # Train/Validation split (85% train, 15% val)
        indices = np.arange(num_samples)
        np.random.seed(42)
        np.random.shuffle(indices)

        val_size = max(2, int(num_samples * 0.15))
        train_idx = indices[:-val_size]
        val_idx = indices[-val_size:]

        x_train = [cloudy[train_idx], hist[train_idx], sar[train_idx], mask[train_idx]]
        y_train = [clear[train_idx], clear[train_idx], clear[train_idx]]

        x_val = [cloudy[val_idx], hist[val_idx], sar[val_idx], mask[val_idx]]
        y_val = [clear[val_idx], clear[val_idx], clear[val_idx]]

        # Build model & loss
        mrr_model = build_mrr_network(input_shape=(self.patch_size, self.patch_size, 4))
        loss_fn = CombinedSharpReconstructionLoss(
            l1_weight=1.0,
            ssim_weight=0.5,
            edge_weight=edge_weight,
            sam_weight=sam_weight
        )

        optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        mrr_model.compile(
            optimizer=optimizer,
            loss={
                "candidate_c1": loss_fn,
                "candidate_c2": loss_fn,
                "candidate_c3": loss_fn
            },
            loss_weights={
                "candidate_c1": 0.25,
                "candidate_c2": 0.25,
                "candidate_c3": 0.50
            }
        )

        # Custom Streamlit/CLI Epoch Progress Callback
        history_records = {
            "epoch": [],
            "loss": [],
            "val_loss": [],
            "c3_loss": [],
            "val_c3_loss": [],
            "val_psnr": [],
            "val_ssim": []
        }

        class CustomEpochCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                # Calculate validation PSNR & SSIM on C3
                val_preds = self.model.predict(x_val, verbose=0)
                pred_c3 = val_preds[2]
                true_img = y_val[2]

                psnr_val = float(tf.reduce_mean(tf.image.psnr(true_img, pred_c3, max_val=1.0)).numpy())
                ssim_val = float(tf.reduce_mean(tf.image.ssim(true_img, pred_c3, max_val=1.0)).numpy())

                history_records["epoch"].append(epoch + 1)
                history_records["loss"].append(float(logs.get("loss", 0.0)))
                history_records["val_loss"].append(float(logs.get("val_loss", 0.0)))
                history_records["c3_loss"].append(float(logs.get("candidate_c3_loss", 0.0)))
                history_records["val_c3_loss"].append(float(logs.get("val_candidate_c3_loss", 0.0)))
                history_records["val_psnr"].append(psnr_val)
                history_records["val_ssim"].append(ssim_val)

                if progress_callback:
                    progress_callback(epoch + 1, epochs, {
                        "loss": logs.get("loss", 0.0),
                        "val_loss": logs.get("val_loss", 0.0),
                        "psnr": psnr_val,
                        "ssim": ssim_val
                    })

        # Train model
        mrr_model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[CustomEpochCallback()],
            verbose=1
        )

        # Save trained weights
        weights_save_path = os.path.join(self.save_dir, "mrr_reconstructor.weights.h5")
        mrr_model.save_weights(weights_save_path)

        # Save training metadata
        meta_save_path = os.path.join(self.save_dir, "training_metadata.json")
        summary = {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "samples_count": num_samples,
            "final_val_loss": history_records["val_loss"][-1] if history_records["val_loss"] else 0.0,
            "final_psnr": history_records["val_psnr"][-1] if history_records["val_psnr"] else 0.0,
            "final_ssim": history_records["val_ssim"][-1] if history_records["val_ssim"] else 0.0,
            "weights_path": weights_save_path,
            "history": history_records
        }
        with open(meta_save_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary
