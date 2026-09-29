"""
Neural Network models for CloudClear AI.
Includes Attention U-Net, Siamese Change Detector, Cross-Attention Transformer, MRR, Losses, and Trainer.
"""

from .cloud_detector import CloudDetectionModel, AttentionUNetCloudDetector, build_attention_unet
from .change_detector import ChangeDetectionModel
from .cross_attention import CrossAttentionFusionLayer
from .mrr_reconstructor import MRRReconstructionModel, build_mrr_network
from .losses import (
    SobelEdgeLoss,
    SpectralAngleLoss,
    SSIMLoss,
    CombinedSharpReconstructionLoss
)
from .trainer import ModelTrainer, SatellitePatchDataset

__all__ = [
    "CloudDetectionModel",
    "AttentionUNetCloudDetector",
    "build_attention_unet",
    "ChangeDetectionModel",
    "CrossAttentionFusionLayer",
    "MRRReconstructionModel",
    "build_mrr_network",
    "SobelEdgeLoss",
    "SpectralAngleLoss",
    "SSIMLoss",
    "CombinedSharpReconstructionLoss",
    "ModelTrainer",
    "SatellitePatchDataset"
]
