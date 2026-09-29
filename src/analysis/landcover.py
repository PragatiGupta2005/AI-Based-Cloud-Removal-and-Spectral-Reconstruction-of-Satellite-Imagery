"""
Multi-Spectral Land Cover Classification and Distribution Analysis for CloudClear AI.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional
import numpy as np
import os


@dataclass
class LandCoverReport:
    distribution: Dict[str, float]  # Class percentages
    class_map: np.ndarray          # 2D uint8 array (0-4)
    colored_map: np.ndarray        # (H, W, 3) uint8 RGB array

    def to_dict(self) -> Dict[str, Any]:
        return self.distribution


class LandCoverClassifier:
    """
    Classifies 4-band satellite imagery into 5 fundamental remote sensing classes.
    Now optionally uses a deep learning segmentation model (LandCoverSegmentationUNet) 
    to fulfill the research claim, falling back to rule-based logic if no model is provided.
    """

    CLASS_NAMES = [
        "Vegetation",
        "Agriculture",
        "Water",
        "Urban",
        "Bare Land"
    ]

    CLASS_COLORS = np.array([
        [16, 185, 129],   # 0: Vegetation (#10B981 - Emerald Green)
        [132, 204, 22],   # 1: Agriculture (#84CC16 - Lime Green)
        [59, 130, 246],   # 2: Water (#3B82F6 - Blue)
        [249, 115, 22],   # 3: Urban (#F97316 - Orange)
        [217, 119, 6]     # 4: Bare Land (#D97706 - Amber/Brown)
    ], dtype=np.uint8)

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            default = os.path.join(base, "models", "saved_models", "landcover_unet.pt")
            if os.path.exists(default):
                model_path = default
        self.model_path = model_path
        self.model = None
        self.method = "rule-based NDVI/NDWI"
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                import torch
                from src.models.segmentation import LandCoverSegmentationUNet
                self.model = LandCoverSegmentationUNet(n_channels=4, n_classes=5)
                self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                self.model.eval()
                self.method = "learned-unet (trained on pseudo-labels; needs real LULC for production)"
            except ImportError:
                pass
            except Exception as e:
                print(f"Error loading land cover model: {e}")

    def classify(self, image_4band: np.ndarray) -> LandCoverReport:
        """
        Classifies surface into 5 classes based on deep learning model or multi-spectral indices.
        """
        H, W = image_4band.shape[:2]

        if self.model is not None:
            import torch
            # Assume image_4band is [H, W, C] and values might be 0-1 or 0-255. Normalize to 0-1 if > 1.
            img_tensor = torch.from_numpy(image_4band).float()
            if img_tensor.max() > 1.0:
                img_tensor /= 255.0
            # [H, W, C] -> [1, C, H, W]
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.model(img_tensor)
                preds = torch.argmax(logits, dim=1).squeeze(0) # [H, W]
                class_map = preds.cpu().numpy().astype(np.uint8)
        else:
            if image_4band.shape[-1] >= 4:
                blue = image_4band[:, :, 0]
                green = image_4band[:, :, 1]
                red = image_4band[:, :, 2]
                nir = image_4band[:, :, 3]
            else:
                blue = green = red = image_4band[:, :, 0]
                nir = image_4band[:, :, min(1, image_4band.shape[-1]-1)]
    
            # NDVI
            denom_ndvi = nir + red + 1e-5
            ndvi = (nir - red) / denom_ndvi
    
            # NDWI (Normalized Difference Water Index = (Green - NIR) / (Green + NIR))
            denom_ndwi = green + nir + 1e-5
            ndwi = (green - nir) / denom_ndwi
    
            # Brightness / Built-up index
            brightness = (blue + green + red) / 3.0
    
            # Classification rules
            class_map = np.zeros((H, W), dtype=np.uint8)
    
            # 2: Water (NDWI > 0.1 or (NIR < 0.12 and blue > red))
            water_mask = (ndwi > 0.05) | ((nir < 0.15) & (brightness < 0.25))
    
            # 0: Dense Vegetation (NDVI >= 0.50)
            veg_mask = (ndvi >= 0.50) & ~water_mask
    
            # 1: Agriculture / Crops (0.25 <= NDVI < 0.50)
            agri_mask = (ndvi >= 0.25) & (ndvi < 0.50) & ~water_mask
    
            # 3: Urban / Built-up (High visible brightness, low NDVI)
            urban_mask = (brightness > 0.40) & (ndvi < 0.25) & ~water_mask
    
            # 4: Bare Land (Default for remaining pixels)
            bare_mask = ~water_mask & ~veg_mask & ~agri_mask & ~urban_mask
            class_map[bare_mask] = 4
            class_map[urban_mask] = 3
            class_map[water_mask] = 2
            class_map[agri_mask] = 1
            class_map[veg_mask] = 0

        # Compute percentages
        total_pixels = float(H * W)
        dist = {
            "vegetation": round(float(np.sum(class_map == 0) / total_pixels * 100.0), 1),
            "agriculture": round(float(np.sum(class_map == 1) / total_pixels * 100.0), 1),
            "water": round(float(np.sum(class_map == 2) / total_pixels * 100.0), 1),
            "urban": round(float(np.sum(class_map == 3) / total_pixels * 100.0), 1),
            "bare_land": round(float(np.sum(class_map == 4) / total_pixels * 100.0), 1)
        }

        # Colorized RGB map
        colored_rgb = self.CLASS_COLORS[class_map]

        return LandCoverReport(
            distribution=dist,
            class_map=class_map,
            colored_map=colored_rgb
        )
