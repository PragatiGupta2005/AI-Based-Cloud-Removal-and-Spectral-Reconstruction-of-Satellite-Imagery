"""
Synthetic & Realistic Satellite Dataset Generator for CloudClear AI.
Generates authentic 4-band GeoTIFF test sets with realistic landforms:
- Bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
- Sentinel-1 SAR: VV & VH polarimetric channels
- Cloud & shadow occlusions, historical references, and ground truth scenes.
"""

import os
import json
from typing import Tuple, Dict, Any
import numpy as np
from scipy import ndimage
from src.preprocessing.data_loader import GeoTIFFLoader, ImageMetadata


def generate_landscape(height: int = 512, width: int = 512, seed: int = 42) -> np.ndarray:
    """
    Generates a realistic 4-band multispectral landscape (B2-Blue, B3-Green, B4-Red, B8-NIR).
    Features: Water body/river, dense vegetation, agricultural plots, urban buildings, bare soil.
    """
    np.random.seed(seed)

    # Base low-frequency elevation / moisture field
    x = np.linspace(-3, 3, width)
    y = np.linspace(-3, 3, height)
    xx, yy = np.meshgrid(x, y)
    elevation = np.sin(xx) * np.cos(yy) + 0.3 * np.random.randn(height, width)
    elevation = ndimage.gaussian_filter(elevation, sigma=15.0)
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())

    # River / Water channel (meandering path)
    river_mask = np.zeros((height, width), dtype=bool)
    river_x = (width * 0.5 + 80 * np.sin(np.linspace(0, 3 * np.pi, height)) + 15 * np.sin(np.linspace(0, 10 * np.pi, height))).astype(int)
    for row, cx in enumerate(river_x):
        r_min = max(0, cx - 12)
        r_max = min(width, cx + 12)
        river_mask[row, r_min:r_max] = True
    river_mask = ndimage.binary_dilation(river_mask, iterations=3)

    # Agricultural grid / field blocks
    grid_x = (np.arange(width) // 40) % 2
    grid_y = (np.arange(height) // 40) % 2
    grid = (grid_x[np.newaxis, :] ^ grid_y[:, np.newaxis]).astype(np.float32)
    fields = ndimage.gaussian_filter(grid * 0.4 + 0.6 * np.random.rand(height, width), sigma=2.0)

    # Urban settlements (clustered blocks with high visible reflectance)
    urban_noise = ndimage.gaussian_filter((np.random.rand(height, width) > 0.985).astype(float), sigma=8.0)
    urban_mask = (urban_noise > 0.05) & ~river_mask

    # Dense forest / vegetation
    forest_mask = (elevation > 0.65) & ~river_mask & ~urban_mask

    # Initialize 4-band spectral array [0.0, 1.0]
    # Standard reflectance signatures:
    # Water: B2=0.20, B3=0.15, B4=0.08, B8=0.04 (Low NIR)
    # Dense Forest: B2=0.04, B3=0.08, B4=0.03, B8=0.75 (High NIR)
    # Agriculture: B2=0.06, B3=0.14, B4=0.08, B8=0.55 (Moderate NIR)
    # Urban: B2=0.35, B3=0.35, B4=0.38, B8=0.32 (High Visible, Flat NIR)
    # Bare Soil: B2=0.18, B3=0.22, B4=0.28, B8=0.35 (Gradual rise)

    b2 = np.full((height, width), 0.18, dtype=np.float32)
    b3 = np.full((height, width), 0.22, dtype=np.float32)
    b4 = np.full((height, width), 0.28, dtype=np.float32)
    b8 = np.full((height, width), 0.35, dtype=np.float32)

    # Apply Agriculture
    b2 = np.where(~river_mask, 0.05 + 0.03 * fields, b2)
    b3 = np.where(~river_mask, 0.12 + 0.08 * fields, b3)
    b4 = np.where(~river_mask, 0.06 + 0.05 * fields, b4)
    b8 = np.where(~river_mask, 0.45 + 0.25 * fields, b8)

    # Apply Dense Forest
    b2 = np.where(forest_mask, 0.03 + 0.02 * np.random.rand(height, width), b2)
    b3 = np.where(forest_mask, 0.08 + 0.03 * np.random.rand(height, width), b3)
    b4 = np.where(forest_mask, 0.03 + 0.02 * np.random.rand(height, width), b4)
    b8 = np.where(forest_mask, 0.70 + 0.12 * np.random.rand(height, width), b8)

    # Apply Urban
    b2 = np.where(urban_mask, 0.32 + 0.08 * np.random.rand(height, width), b2)
    b3 = np.where(urban_mask, 0.34 + 0.08 * np.random.rand(height, width), b3)
    b4 = np.where(urban_mask, 0.38 + 0.09 * np.random.rand(height, width), b4)
    b8 = np.where(urban_mask, 0.30 + 0.06 * np.random.rand(height, width), b8)

    # Apply Water
    b2 = np.where(river_mask, 0.18 + 0.02 * np.random.rand(height, width), b2)
    b3 = np.where(river_mask, 0.14 + 0.02 * np.random.rand(height, width), b3)
    b4 = np.where(river_mask, 0.08 + 0.02 * np.random.rand(height, width), b4)
    b8 = np.where(river_mask, 0.03 + 0.01 * np.random.rand(height, width), b8)

    # Add realistic texture noise
    texture = ndimage.gaussian_filter(np.random.randn(height, width) * 0.01, sigma=0.8)
    b2 = np.clip(b2 + texture, 0.01, 0.99)
    b3 = np.clip(b3 + texture, 0.01, 0.99)
    b4 = np.clip(b4 + texture, 0.01, 0.99)
    b8 = np.clip(b8 + texture, 0.01, 0.99)

    return np.stack([b2, b3, b4, b8], axis=-1)


def generate_clouds_and_shadows(height: int, width: int, seed: int = 101) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates realistic Perlin-like fractal clouds and corresponding offset ground shadows.
    Returns: (cloud_density [0, 1], binary_cloud_mask [0, 1], binary_shadow_mask [0, 1])
    """
    np.random.seed(seed)

    # Multi-octave fractal noise for cumulus cloud formation
    noise1 = ndimage.gaussian_filter(np.random.randn(height, width), sigma=35.0)
    noise2 = ndimage.gaussian_filter(np.random.randn(height, width), sigma=18.0) * 0.5
    noise3 = ndimage.gaussian_filter(np.random.randn(height, width), sigma=8.0) * 0.25

    fractal = noise1 + noise2 + noise3
    fractal = (fractal - fractal.min()) / (fractal.max() - fractal.min())

    # Concentrated cloud patch center
    cy, cx = int(height * 0.45), int(width * 0.55)
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (0.4 * width)
    patch_weight = np.clip(1.3 - dist_from_center, 0.0, 1.0)

    cloud_density = np.clip(fractal * patch_weight * 2.0 - 0.35, 0.0, 1.0)
    cloud_density = ndimage.gaussian_filter(cloud_density, sigma=2.0)

    cloud_mask = (cloud_density >= 0.40).astype(np.uint8)

    # Cloud shadow offset (e.g. sun angle casting shadow down-left: +15px Y, -15px X)
    shadow_shift = ndimage.shift(cloud_density, shift=(18, -18), mode='constant', cval=0.0)
    shadow_mask = ((shadow_shift >= 0.35) & (cloud_mask == 0)).astype(np.uint8)

    return cloud_density, cloud_mask, shadow_mask


def generate_sar_backscatter(landscape_4band: np.ndarray, seed: int = 202) -> np.ndarray:
    """
    Generates realistic Sentinel-1 C-Band SAR radar backscatter (VV and VH channels).
    SAR characteristics:
    - Water: Specular reflection -> Very dark (VV ~ 0.05, VH ~ 0.02)
    - Forest / Dense Veg: Volume scattering -> Medium-High (VV ~ 0.50, VH ~ 0.35)
    - Urban / Buildings: Double-bounce scattering -> Very Bright (VV ~ 0.85, VH ~ 0.70)
    - Agriculture: Surface + Volume scattering -> (VV ~ 0.35, VH ~ 0.20)
    """
    np.random.seed(seed)
    H, W, _ = landscape_4band.shape
    nir = landscape_4band[:, :, 3]
    visible = np.mean(landscape_4band[:, :, :3], axis=-1)

    # Base radar intensity
    vv = 0.4 * nir + 0.4 * visible
    # High urban double bounce
    urban_areas = (visible > 0.30) & (nir < 0.35)
    vv = np.where(urban_areas, 0.85 + 0.1 * np.random.randn(H, W), vv)
    # Low water backscatter
    water_areas = (nir < 0.10)
    vv = np.where(water_areas, 0.06 + 0.02 * np.random.randn(H, W), vv)

    # Cross-polarization VH (usually lower intensity than VV)
    vh = vv * 0.65 + 0.05 * np.random.randn(H, W)

    # SAR speckle noise (Rayleigh / Gamma distributed speckle)
    speckle = np.random.gamma(shape=4.0, scale=0.25, size=(H, W))
    vv = np.clip(vv * speckle, 0.01, 0.99)
    vh = np.clip(vh * speckle, 0.01, 0.99)

    return np.stack([vv, vh], axis=-1).astype(np.float32)


def generate_all_sample_datasets():
    """
    Generates 3 full sample regional satellite scenes for demonstration.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = {
        "cloudy": os.path.join(base_dir, "data", "cloudy"),
        "clear": os.path.join(base_dir, "data", "clear"),
        "historical": os.path.join(base_dir, "data", "historical"),
        "sar": os.path.join(base_dir, "data", "sar"),
        "samples": os.path.join(base_dir, "data", "samples")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    scenes = [
        {
            "id": "scene_west_bengal_nadia",
            "region": "West Bengal, Nadia District",
            "date": "2024-05-12",
            "optical_sensor": "Sentinel-2",
            "sar_sensor": "Sentinel-1",
            "crs": "EPSG:4326",
            "resolution": 10.0,
            "bounds": (88.40, 22.90, 88.55, 23.05),
            "seed": 42
        },
        {
            "id": "scene_maharashtra_pune",
            "region": "Maharashtra, Pune Mula-Mutha Basin",
            "date": "2024-04-18",
            "optical_sensor": "LISS-IV",
            "sar_sensor": "Sentinel-1",
            "crs": "EPSG:4326",
            "resolution": 5.8,
            "bounds": (73.80, 18.45, 73.95, 18.60),
            "seed": 88
        },
        {
            "id": "scene_kerala_alappuzha",
            "region": "Kerala, Alappuzha Coastal Region",
            "date": "2024-06-02",
            "optical_sensor": "Sentinel-2",
            "sar_sensor": "Sentinel-1",
            "crs": "EPSG:4326",
            "resolution": 10.0,
            "bounds": (76.30, 9.45, 76.45, 9.60),
            "seed": 123
        }
    ]

    manifest = []

    for sc in scenes:
        sid = sc["id"]
        H, W = 512, 512

        # 1. Ground Truth Clear Optical Scene
        clear_optical = generate_landscape(height=H, width=W, seed=sc["seed"])

        # 2. Historical Scene of Same AOI (Acquired 3 months earlier, seasonal variation in crop stage)
        hist_optical = clear_optical.copy()
        # Seasonal change in agricultural NIR reflectance and visible brightness
        hist_optical[:, :, 0] = np.clip(hist_optical[:, :, 0] * 1.02 + 0.01, 0.0, 1.0)
        hist_optical[:, :, 1] = np.clip(hist_optical[:, :, 1] * 0.95 + 0.01, 0.0, 1.0)
        hist_optical[:, :, 2] = np.clip(hist_optical[:, :, 2] * 1.04 - 0.01, 0.0, 1.0)
        hist_optical[:, :, 3] = np.clip(hist_optical[:, :, 3] * 0.96 + 0.02, 0.0, 1.0)
        # Minor field changes
        field_shifts = ndimage.gaussian_filter(np.random.randn(H, W) * 0.02, sigma=3.0)[:, :, np.newaxis]
        hist_optical = np.clip(hist_optical + field_shifts, 0.01, 0.99).astype(np.float32)

        # 3. Clouds and Shadows
        cloud_density, cloud_mask, shadow_mask = generate_clouds_and_shadows(H, W, seed=sc["seed"] + 55)

        # 4. Cloudy Optical Scene
        # Clouds add high brightness across all bands + atmospheric haze
        c_density_3d = cloud_density[:, :, np.newaxis]
        s_mask_3d = shadow_mask[:, :, np.newaxis]

        cloud_color = np.array([0.92, 0.95, 0.98, 0.90], dtype=np.float32)  # High across B2, B3, B4, B8
        cloudy_optical = (1.0 - c_density_3d) * clear_optical + c_density_3d * cloud_color
        # Shadow attenuates ground reflectance by 60%
        cloudy_optical = np.where(s_mask_3d > 0, cloudy_optical * 0.40, cloudy_optical)
        cloudy_optical = np.clip(cloudy_optical, 0.0, 1.0).astype(np.float32)

        # 5. Sentinel-1 SAR Scene
        sar_image = generate_sar_backscatter(clear_optical, seed=sc["seed"] + 77)

        # Save all GeoTIFF files
        meta = ImageMetadata(
            image_id=sid,
            filename=f"{sid}_cloudy.tif",
            width=W,
            height=H,
            bands=4,
            crs=sc["crs"],
            resolution=sc["resolution"],
            sensor=sc["optical_sensor"],
            acquisition_date=sc["date"],
            region=sc["region"],
            bounds=sc["bounds"]
        )

        cloudy_path = os.path.join(dirs["cloudy"], f"{sid}_cloudy.tif")
        clear_path = os.path.join(dirs["clear"], f"{sid}_clear.tif")
        hist_path = os.path.join(dirs["historical"], f"{sid}_historical.tif")
        sar_path = os.path.join(dirs["sar"], f"{sid}_sar.tif")

        GeoTIFFLoader.save_raster(cloudy_path, cloudy_optical, reference_meta=meta)
        GeoTIFFLoader.save_raster(clear_path, clear_optical, reference_meta=meta)
        GeoTIFFLoader.save_raster(hist_path, hist_optical, reference_meta=meta)
        GeoTIFFLoader.save_raster(sar_path, sar_image, reference_meta=meta)

        # Save metadata info file in samples
        sample_meta_path = os.path.join(dirs["samples"], f"{sid}_info.json")
        sample_entry = {
            "image_id": sid,
            "region": sc["region"],
            "date": sc["date"],
            "optical_sensor": sc["optical_sensor"],
            "sar_sensor": sc["sar_sensor"],
            "crs": sc["crs"],
            "resolution": sc["resolution"],
            "bounds": sc["bounds"],
            "files": {
                "cloudy": cloudy_path,
                "clear": clear_path,
                "historical": hist_path,
                "sar": sar_path
            }
        }
        with open(sample_meta_path, "w") as f:
            json.dump(sample_entry, f, indent=2)

        manifest.append(sample_entry)
        print(f"Generated realistic satellite sample scene: {sid} ({sc['region']})")

    manifest_path = os.path.join(dirs["samples"], "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll {len(scenes)} sample scenes generated successfully in data/ directory!")


if __name__ == "__main__":
    generate_all_sample_datasets()
