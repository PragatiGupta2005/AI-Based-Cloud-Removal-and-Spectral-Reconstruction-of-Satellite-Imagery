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

    # --- Audit Sec.6/7: Attention U-Net cloud/shadow training ---
    def train_cloud_detector(
        self,
        dataset_patches: Dict[str, np.ndarray],
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
    ) -> Dict[str, Any]:
        """Trains Attention U-Net on (cloudy -> cloud/shadow mask).

        Supervision: synthetic cloud masks from SatellitePatchDataset
        (diff cloudy vs clear) + shadow approximated as dark low-NIR pixels.
        Saves `cloud_detector.weights.h5` + `training_metadata_cloud.json`.
        """
        cloudy = dataset_patches["cloudy"]
        clear = dataset_patches["clear"]
        if len(cloudy) == 0:
            raise ValueError("No training patches available.")
        import tensorflow as tf
        from tensorflow.keras import optimizers as _opt
        # Build targets: ch0=cloud (mask), ch1=shadow (dark & low NIR & not cloud)
        masks = dataset_patches.get("mask", None)
        targets = []
        for i in range(len(cloudy)):
            c = cloudy[i]
            if masks is not None and len(masks) == len(cloudy):
                m = masks[i]
                cm = m[:, :, 0] if m.ndim == 3 else m.squeeze(-1)
            else:
                diff = float(np.mean(np.abs(c - clear[i])))
                cm = (np.mean(np.abs(c - clear[i]), axis=-1) > 0.12).astype(np.float32)
            b8 = c[:, :, 3]
            bright = c[:, :, :3].mean(axis=-1)
            shadow = ((bright < 0.25) & (b8 < 0.22) & (cm < 0.5)).astype(np.float32)
            targets.append(np.stack([cm, shadow], axis=-1))
        y = np.array(targets, dtype=np.float32)
        n = len(cloudy)
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        vs = max(2, int(n * 0.15))
        tr, va = idx[:-vs], idx[-vs:]
        model = build_attention_unet(input_shape=(self.patch_size, self.patch_size, 4))
        model.compile(
            optimizer=_opt.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        hist = model.fit(
            cloudy[tr], y[tr], validation_data=(cloudy[va], y[va]),
            epochs=epochs, batch_size=batch_size, verbose=1,
        )
        wpath = os.path.join(self.save_dir, "cloud_detector.weights.h5")
        model.save_weights(wpath)
        summary = {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "samples_count": n,
            "final_val_loss": float(hist.history["val_loss"][-1]),
            "weights_path": wpath,
            "data_source": "synthetic (diff cloudy-vs-clear + NIR shadow rule); "
                           "replace with real annotated clouds (e.g. 38-Cloud, SPARCS) for production",
        }
        with open(os.path.join(self.save_dir, "training_metadata_cloud.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # --- Audit Sec.8/9: Siamese change-detector training ---
    def train_change_detector(
        self,
        dataset_patches: Dict[str, np.ndarray],
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
    ) -> Dict[str, Any]:
        """Trains Siamese change detector.

        Supervision: |clear - historical| magnitude as pseudo change-probability
        (synthetic seasonal shifts). Saves `change_detector.weights.h5`.
        """
        from .change_detector import build_siamese_change_detector
        cloudy = dataset_patches["cloudy"]
        hist = dataset_patches["historical"]
        sar = dataset_patches["sar"]
        clear = dataset_patches["clear"]
        n = len(cloudy)
        if n == 0:
            raise ValueError("No training patches available.")
        # pseudo change target: normalized |clear-hist| -> 64x64 (model output res)
        import tensorflow as tf
        targets = []
        for i in range(n):
            d = np.mean(np.abs(clear[i] - hist[i]), axis=-1, keepdims=True)
            t = np.clip(d * 4.0, 0.0, 1.0).astype(np.float32)
            # model outputs 128x128 for 256 input (one pooling); resize target
            t_small = tf.image.resize(t, (128, 128)).numpy()
            targets.append(t_small)
        y = np.array(targets, dtype=np.float32)
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        vs = max(2, int(n * 0.15))
        tr, va = idx[:-vs], idx[-vs:]
        from tensorflow.keras import optimizers as _opt
        model = build_siamese_change_detector(input_shape=(self.patch_size, self.patch_size, 4))
        model.compile(optimizer=_opt.Adam(learning_rate=learning_rate), loss="binary_crossentropy")
        hist_cb = model.fit(
            [cloudy[tr], hist[tr], sar[tr]], y[tr],
            validation_data=([cloudy[va], hist[va], sar[va]], y[va]),
            epochs=epochs, batch_size=batch_size, verbose=1,
        )
        wpath = os.path.join(self.save_dir, "change_detector.weights.h5")
        model.save_weights(wpath)
        summary = {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "samples_count": n,
            "final_val_loss": float(hist_cb.history["val_loss"][-1]),
            "weights_path": wpath,
            "data_source": "synthetic (|clear-historical| pseudo change); "
                           "replace with OSCD/real bi-temporal labels for production",
        }
        with open(os.path.join(self.save_dir, "training_metadata_change.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # --- Audit Sec.10: learned DeepQAN training (torch) ---
    def train_deep_qan(
        self,
        dataset_patches: Dict[str, np.ndarray],
        epochs: int = 8,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
    ) -> Dict[str, Any]:
        """Trains DeepQAN regressor: (candidate, reference) -> composite quality.

        Labels are metric-based composite scores (distillation), so the net
        learns to predict quality without a reference at inference-adjacent use;
        training still requires references. Saves `deep_qan.pt`.
        """
        import torch
        import torch.nn as nn
        from .deep_qan import DeepQAN
        from ..qan.quality_network import QualityAssessmentNetwork as _Q
        cloudy = dataset_patches["cloudy"]
        clear = dataset_patches["clear"]
        n = len(cloudy)
        if n == 0:
            raise ValueError("No training patches available.")
        # Build pairs: (cloudy|clean) with label=metric composite; (clear|clear)=1.0
        xs, ys = [], []
        for i in range(n):
            m = _Q.calculate_metrics(cloudy[i], clear[i])
            xs.append(np.concatenate([cloudy[i], clear[i]], axis=-1))
            ys.append(m.composite_score)
            xs.append(np.concatenate([clear[i], clear[i]], axis=-1))
            ys.append(1.0)
        X = np.array(xs, dtype=np.float32)
        Y = np.array(ys, dtype=np.float32)
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X).permute(0, 3, 1, 2), torch.from_numpy(Y))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        model = DeepQAN(in_channels=8)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        wpath = os.path.join(self.save_dir, "deep_qan.pt")
        torch.save(model.state_dict(), wpath)
        summary = {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "samples_count": int(len(Y)),
            "weights_path": wpath,
            "method": "distilled metric-composite regression (torch)",
        }
        with open(os.path.join(self.save_dir, "training_metadata_qan.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # --- Audit Sec.11: land-cover UNet training on pseudo-labels ---
    def train_landcover_unet(
        self,
        dataset_patches: Dict[str, np.ndarray],
        epochs: int = 6,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
    ) -> Dict[str, Any]:
        """Trains LandCoverSegmentationUNet with rule-based pseudo-labels.

        This gives a real learned weights file (`landcover_unet.pt`) while being
        honest that supervision is synthetic; swap in real LULC labels for prod.
        """
        import torch
        import torch.nn as nn
        from .segmentation import LandCoverSegmentationUNet
        from ..analysis.landcover import LandCoverClassifier as _LC
        clear = dataset_patches["clear"]
        n = len(clear)
        if n == 0:
            raise ValueError("No training patches available.")
        pseudo = []
        rule = _LC()
        for i in range(n):
            rep = rule.classify(np.clip(clear[i], 0, 1))
            pseudo.append(rep.class_map)
        X = torch.from_numpy(np.array(clear, dtype=np.float32)).permute(0, 3, 1, 2)
        Y = torch.from_numpy(np.array(pseudo, dtype=np.int64))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )
        model = LandCoverSegmentationUNet(n_channels=4, n_classes=5)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        ce = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                logits = model(xb)
                loss = ce(logits, yb)
                loss.backward()
                opt.step()
        wpath = os.path.join(self.save_dir, "landcover_unet.pt")
        torch.save(model.state_dict(), wpath)
        summary = {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "samples_count": n,
            "weights_path": wpath,
            "method": "supervised on rule-based pseudo-labels (replace with real LULC for production)",
        }
        with open(os.path.join(self.save_dir, "training_metadata_landcover.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary
