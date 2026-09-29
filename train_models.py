"""
CloudClear AI — Standalone Model Training & Sharpness Optimization CLI.
Trains the deep Multi-Scale MRR Reconstructor and Attention U-Net on satellite patches
with multi-component loss (L1 + SSIM + Sobel Edge + SAM) to eliminate blur under clouds.

Usage:
    python train_models.py --epochs 15 --batch_size 8 --learning_rate 0.001
"""

import os
import sys
import argparse
import time

# Ensure workspace root in path and UTF-8 console output
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.trainer import SatellitePatchDataset, ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train CloudClear AI Satellite Reconstruction Models")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for gradient descent")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam optimizer learning rate")
    parser.add_argument("--patch_size", type=int, default=256, help="Spatial resolution of training patches")
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride for patch extraction")
    parser.add_argument("--edge_weight", type=float, default=0.35, help="Sobel edge gradient loss weight")
    parser.add_argument("--sam_weight", type=float, default=0.25, help="Spectral Angle Mapper loss weight")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing GeoTIFF datasets")
    parser.add_argument("--save_dir", type=str, default=os.path.join("models", "saved_models"), help="Directory to save weights")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("  🚀 CloudClear AI: High-Accuracy Deep Model Training")
    print("  Target: Multi-Modal Satellite Cloud Removal & Sharp Reconstruction")
    print("=" * 70)
    print(f"• Patch Size: {args.patch_size}x{args.patch_size}")
    print(f"• Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.learning_rate}")
    print(f"• Sobel Edge Loss Weight: {args.edge_weight} | SAM Loss Weight: {args.sam_weight}")
    print(f"• Dataset Source: {args.data_dir} -> Saving to: {args.save_dir}\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, args.data_dir)
    save_path = os.path.join(base_dir, args.save_dir)

    # 1. Patch Extraction
    print("📦 Step 1: Extracting multi-modal patch pairs & applying augmentations...")
    ds_extractor = SatellitePatchDataset(
        data_dir=data_path,
        patch_size=args.patch_size,
        stride=args.stride
    )
    patches = ds_extractor.extract_patches()
    num_patches = len(patches["cloudy"])
    print(f"✓ Extracted {num_patches} multi-modal training pairs with 4-band optical + 2-band SAR.\n")

    if num_patches == 0:
        print("❌ Error: No dataset patches found in data directory. Please generate or fetch datasets first.")
        sys.exit(1)

    # 2. Model Training
    print("🧠 Step 2: Training Multi-Scale MRR Reconstructor with Sharp Edge Loss...")
    trainer = ModelTrainer(save_dir=save_path, patch_size=args.patch_size)

    start_t = time.time()
    results = trainer.train_reconstruction_model(
        dataset_patches=patches,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        edge_weight=args.edge_weight,
        sam_weight=args.sam_weight
    )
    elapsed = time.time() - start_t

    print("\n" + "=" * 70)
    print("  ✅ Training Completed Successfully!")
    print("=" * 70)
    print(f"• Total Time: {elapsed:.2f} seconds")
    print(f"• Final Validation Loss: {results['final_val_loss']:.5f}")
    print(f"• Final Validation PSNR: {results['final_psnr']:.2f} dB")
    print(f"• Final Validation SSIM: {results['final_ssim']:.4f}")
    print(f"• Model Weights Saved: {results['weights_path']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
