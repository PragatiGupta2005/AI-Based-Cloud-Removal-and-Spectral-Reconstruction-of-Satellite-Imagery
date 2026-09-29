"""
Sub-Cloud Ground Feature Prediction & Semantic Analysis Engine for CloudClear AI.
Specifically predicts, segments, and quantifies real-world ground features
(Roads, Crop Parcels, Water Channels, Built-Up Settlements, Tree Canopies)
concealed beneath thick cloud and shadow occlusions.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy import ndimage


@dataclass
class SubCloudFeatureReport:
    total_occluded_pixels: int
    occluded_area_hectares: float
    detected_features: Dict[str, Dict[str, Any]]
    feature_map: np.ndarray           # 2D uint8 class array
    colored_feature_map: np.ndarray   # (H, W, 3) uint8 RGB array
    spectral_profile: Dict[str, List[float]]
    prominent_structures_summary: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_occluded_pixels": self.total_occluded_pixels,
            "occluded_area_hectares": self.occluded_area_hectares,
            "detected_features": self.detected_features,
            "spectral_profile": self.spectral_profile,
            "prominent_structures_summary": self.prominent_structures_summary
        }


class SubCloudFeaturePredictor:
    """
    Decodes multi-spectral optical reflectance and SAR radar dielectric properties
    to predict ground land features beneath cloud occlusions.
    """

    FEATURE_CLASSES = {
        0: ("Clear / Unobstructed Surface", [40, 50, 65], "#283241"),
        1: ("🌾 Agricultural Crops (Paddy / Wheat)", [132, 204, 22], "#84CC16"),
        2: ("🌲 Dense Tree Canopy & Forest", [16, 185, 129], "#10B981"),
        3: ("💧 Water Bodies, Dams & River Channels", [59, 130, 246], "#3B82F6"),
        4: ("🏢 Urban Built-Up & IT Complexes", [249, 115, 22], "#F97316"),
        5: ("🛣️ Highway & Road Infrastructure", [234, 179, 8], "#EAB308"),
        6: ("🏜️ Bare Fallow Land & Soil", [217, 119, 6], "#D97706")
    }

    def predict_sub_cloud_features(
        self,
        cloud_mask: np.ndarray,
        reconstructed_image: np.ndarray,
        sar_image: Optional[np.ndarray] = None,
        pixel_resolution_m: float = 10.0
    ) -> SubCloudFeatureReport:
        """
        Analyzes reconstructed ground pixels strictly inside the cloud mask.
        """
        H, W = cloud_mask.shape[:2]
        c_bool = (cloud_mask > 0)
        total_pixels = int(np.sum(c_bool))

        # Area in hectares: (pixel_count * res^2) / 10,000
        area_m2 = total_pixels * (pixel_resolution_m ** 2)
        area_ha = round(area_m2 / 10000.0, 2)

        feature_map = np.zeros((H, W), dtype=np.uint8)
        colored_map = np.zeros((H, W, 3), dtype=np.uint8)

        if total_pixels == 0:
            return SubCloudFeatureReport(
                total_occluded_pixels=0,
                occluded_area_hectares=0.0,
                detected_features={},
                feature_map=feature_map,
                colored_feature_map=colored_map,
                spectral_profile={},
                prominent_structures_summary=["No cloud obstruction detected in scene."]
            )

        # Multi-band spectral indices
        blue = reconstructed_image[:, :, 0]
        green = reconstructed_image[:, :, 1]
        red = reconstructed_image[:, :, 2]
        nir = reconstructed_image[:, :, 3] if reconstructed_image.shape[-1] >= 4 else reconstructed_image[:, :, 0]

        # NDVI & NDWI & Brightness
        ndvi = (nir - red) / (nir + red + 1e-5)
        ndwi = (green - nir) / (green + nir + 1e-5)
        bright = (blue + green + red) / 3.0

        # SAR radar backscatter intensity (surface roughness)
        sar_intensity = sar_image[:, :, 0] if sar_image is not None else np.full((H, W), 0.20)

        # 1. Water Bodies (Dams, Reservoirs, Lakes, River Channels & Irrigation Canals)
        # Characteristics: High NDWI (> -0.05), low NIR (< 0.25), blue/green hue, or specular low SAR return
        is_water = (
            (ndwi > -0.05) & (nir < 0.25)
        ) | (
            (blue >= red * 0.90) & (nir < 0.20) & (bright < 0.38)
        ) | (
            (sar_intensity < 0.12) & (ndvi < 0.12) & (bright < 0.32)
        ) | (
            (bright < 0.15) & (ndvi < 0.10)
        )
        feature_map[c_bool & is_water] = 3

        # 2. Roads / Highways (Elongated high brightness, low vegetation, moderate SAR)
        grad_mag = ndimage.generic_gradient_magnitude(bright, ndimage.sobel)
        is_road = (bright > 0.20) & (bright < 0.38) & (ndvi < 0.18) & (grad_mag > 0.035)
        feature_map[c_bool & is_road & ~is_water] = 5

        # 3. Urban Buildings (High brightness, high SAR double-bounce, low NDVI)
        is_urban = (bright > 0.25) & (ndvi < 0.22) & ~is_road & ~is_water
        feature_map[c_bool & is_urban] = 4

        # 4. Dense Tree Canopy (Very high NDVI > 0.45)
        is_forest = (ndvi >= 0.45) & ~is_water
        feature_map[c_bool & is_forest] = 2

        # 5. Agriculture / Crops (Moderate-High NDVI 0.20 - 0.45)
        is_agri = (ndvi >= 0.20) & (ndvi < 0.45) & ~is_water
        feature_map[c_bool & is_agri] = 1

        # 6. Bare Soil (Remaining unassigned pixels)
        is_bare = (feature_map == 0) & c_bool
        feature_map[is_bare] = 6

        # Build RGB colorized visualization
        for cid, (name, color, _) in self.FEATURE_CLASSES.items():
            mask = (feature_map == cid)
            colored_map[mask] = color

        # Compute statistics for each feature under the cloud
        detected_features = {}
        structures_summary = []

        for cid, (name, _, hex_col) in self.FEATURE_CLASSES.items():
            if cid == 0:
                continue
            cnt = int(np.sum(feature_map == cid))
            pct = round((cnt / total_pixels) * 100.0, 1)
            feat_ha = round((cnt * (pixel_resolution_m ** 2)) / 10000.0, 2)
            
            detected_features[name] = {
                "class_id": cid,
                "pixel_count": cnt,
                "percentage": pct,
                "area_hectares": feat_ha,
                "color": hex_col
            }

            if pct > 4.0:
                structures_summary.append(f"{name}: {pct}% ({feat_ha} ha uncovered)")

        # Spectral profile averages under clouds
        spectral_profile = {
            "Blue (B2)": [float(np.mean(blue[c_bool]))],
            "Green (B3)": [float(np.mean(green[c_bool]))],
            "Red (B4)": [float(np.mean(red[c_bool]))],
            "NIR (B8)": [float(np.mean(nir[c_bool]))],
            "Mean NDVI": [float(np.mean(ndvi[c_bool]))]
        }

        return SubCloudFeatureReport(
            total_occluded_pixels=total_pixels,
            occluded_area_hectares=area_ha,
            detected_features=detected_features,
            feature_map=feature_map,
            colored_feature_map=colored_map,
            spectral_profile=spectral_profile,
            prominent_structures_summary=structures_summary
        )
