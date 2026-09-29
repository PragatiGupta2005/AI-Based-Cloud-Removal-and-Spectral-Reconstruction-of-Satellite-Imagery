"""
Comprehensive Dataset Collector & Generator for CloudClear AI.
Generates 30+ multi-modal satellite scenes (120+ GeoTIFFs total) spanning diverse Indian geographies:
- 4-band Optical (B2-Blue, B3-Green, B4-Red, B8-NIR) for Cloudy, Clear, and Historical scenes
- 2-band Sentinel-1 SAR (VV & VH polarimetric channels)
- Metadata CSV cataloging coordinates, sensors, resolution, cloud cover, and timestamps.
"""

import os
import csv
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import ndimage

from src.preprocessing.data_loader import GeoTIFFLoader, ImageMetadata
from generate_sample_data import generate_clouds_and_shadows, generate_sar_backscatter


def generate_regional_landscape(
    height: int,
    width: int,
    terrain_type: str,
    seed: int
) -> np.ndarray:
    """
    Generates realistic 4-band multispectral landscape for specific terrain types:
    - 'agriculture': Farm grid plots, irrigation channels, variable crop vigor
    - 'urban': Dense built-up structures, road grids, water reservoirs
    - 'coastal': Ocean / backwater bodies, sand beaches, coconut groves
    - 'forest': Dense canopy vegetation, elevation gradients, valleys
    - 'arid': Sparse scrub, bare soil, high visible reflectance, low NIR
    - 'wetland': Mangrove waterways, muddy shallows, marsh vegetation
    - 'mountain': Ridge lines, shadowed slopes, coniferous patches
    """
    np.random.seed(seed)

    x = np.linspace(-3, 3, width)
    y = np.linspace(-3, 3, height)
    xx, yy = np.meshgrid(x, y)

    b2 = np.full((height, width), 0.15, dtype=np.float32)
    b3 = np.full((height, width), 0.20, dtype=np.float32)
    b4 = np.full((height, width), 0.25, dtype=np.float32)
    b8 = np.full((height, width), 0.35, dtype=np.float32)

    if terrain_type == "agriculture":
        # Multi-crop agricultural blocks
        grid_x = (np.arange(width) // (25 + (seed % 20))) % 3
        grid_y = (np.arange(height) // (25 + (seed % 20))) % 3
        grid = (grid_x[np.newaxis, :] ^ grid_y[:, np.newaxis]).astype(np.float32)
        crop_vigor = ndimage.gaussian_filter(grid * 0.5 + 0.5 * np.random.rand(height, width), sigma=1.5)

        # Irrigation river
        river_x = (width * 0.45 + 50 * np.sin(np.linspace(0, 2.5 * np.pi, height))).astype(int)
        river_mask = np.zeros((height, width), dtype=bool)
        for r, cx in enumerate(river_x):
            river_mask[r, max(0, cx-8):min(width, cx+8)] = True

        b2 = np.where(~river_mask, 0.04 + 0.02 * crop_vigor, 0.16)
        b3 = np.where(~river_mask, 0.12 + 0.09 * crop_vigor, 0.12)
        b4 = np.where(~river_mask, 0.05 + 0.04 * crop_vigor, 0.07)
        b8 = np.where(~river_mask, 0.45 + 0.35 * crop_vigor, 0.03)

    elif terrain_type == "urban":
        # Urban settlement blocks & river
        street_grid = ((np.arange(width) % 20 < 3) | (np.arange(height) % 20 < 3)[:, np.newaxis])
        urban_density = ndimage.gaussian_filter(np.random.rand(height, width) > 0.85, sigma=6.0)
        
        river_x = (width * 0.5 + 40 * np.sin(np.linspace(0, 2 * np.pi, height))).astype(int)
        river_mask = np.zeros((height, width), dtype=bool)
        for r, cx in enumerate(river_x):
            river_mask[r, max(0, cx-10):min(width, cx+10)] = True

        b2 = np.where(urban_density > 0.03, 0.34 + 0.06 * np.random.rand(height, width), 0.15)
        b3 = np.where(urban_density > 0.03, 0.36 + 0.06 * np.random.rand(height, width), 0.20)
        b4 = np.where(urban_density > 0.03, 0.40 + 0.07 * np.random.rand(height, width), 0.18)
        b8 = np.where(urban_density > 0.03, 0.28 + 0.05 * np.random.rand(height, width), 0.45)
        # Streets
        b2 = np.where(street_grid, 0.28, b2)
        b3 = np.where(street_grid, 0.28, b3)
        b4 = np.where(street_grid, 0.30, b4)
        # Water
        b2 = np.where(river_mask, 0.18, b2)
        b3 = np.where(river_mask, 0.14, b3)
        b4 = np.where(river_mask, 0.08, b4)
        b8 = np.where(river_mask, 0.03, b8)

    elif terrain_type == "coastal":
        # Coastline (half ocean, half land with mangroves & palms)
        coast_line = (width * 0.45 + 30 * np.sin(np.linspace(0, 1.5 * np.pi, height))).astype(int)
        ocean_mask = np.zeros((height, width), dtype=bool)
        for r, cx in enumerate(coast_line):
            ocean_mask[r, :cx] = True

        land_veg = ndimage.gaussian_filter(np.random.rand(height, width), sigma=3.0)
        b2 = np.where(~ocean_mask, 0.05 + 0.03 * land_veg, 0.22)
        b3 = np.where(~ocean_mask, 0.15 + 0.08 * land_veg, 0.16)
        b4 = np.where(~ocean_mask, 0.06 + 0.04 * land_veg, 0.09)
        b8 = np.where(~ocean_mask, 0.65 + 0.20 * land_veg, 0.02)

    elif terrain_type == "forest":
        # Dense forest & topography
        topo = ndimage.gaussian_filter(np.sin(xx*2) * np.cos(yy*2) + 0.2 * np.random.randn(height, width), sigma=10.0)
        topo = (topo - topo.min()) / (topo.max() - topo.min())

        b2 = 0.03 + 0.02 * topo
        b3 = 0.08 + 0.04 * topo
        b4 = 0.03 + 0.03 * topo
        b8 = 0.72 + 0.18 * topo

    elif terrain_type == "arid":
        # Arid sandy terrain with sparse scrub
        dunes = ndimage.gaussian_filter(np.sin(xx * 5 + yy * 2) + 0.3 * np.random.randn(height, width), sigma=4.0)
        dunes = (dunes - dunes.min()) / (dunes.max() - dunes.min())

        b2 = 0.22 + 0.08 * dunes
        b3 = 0.28 + 0.09 * dunes
        b4 = 0.36 + 0.12 * dunes
        b8 = 0.38 + 0.10 * dunes

    elif terrain_type == "wetland":
        # Mangrove creeks & marsh
        creeks = ndimage.gaussian_filter(np.abs(np.sin(xx*3) * np.cos(yy*3)), sigma=3.0) < 0.15
        marsh = ndimage.gaussian_filter(np.random.rand(height, width), sigma=2.5)

        b2 = np.where(~creeks, 0.04 + 0.02 * marsh, 0.18)
        b3 = np.where(~creeks, 0.10 + 0.05 * marsh, 0.13)
        b4 = np.where(~creeks, 0.05 + 0.03 * marsh, 0.08)
        b8 = np.where(~creeks, 0.55 + 0.22 * marsh, 0.04)

    else:  # "mountain"
        ridges = ndimage.gaussian_filter(np.abs(np.sin(xx*4 + yy*3)), sigma=8.0)
        b2 = 0.08 + 0.15 * ridges
        b3 = 0.12 + 0.18 * ridges
        b4 = 0.10 + 0.20 * ridges
        b8 = 0.40 + 0.35 * ridges

    # Subtle spatial texture
    texture = ndimage.gaussian_filter(np.random.randn(height, width) * 0.01, sigma=0.8)
    b2 = np.clip(b2 + texture, 0.01, 0.99)
    b3 = np.clip(b3 + texture, 0.01, 0.99)
    b4 = np.clip(b4 + texture, 0.01, 0.99)
    b8 = np.clip(b8 + texture, 0.01, 0.99)

    return np.stack([b2, b3, b4, b8], axis=-1).astype(np.float32)


def generate_full_dataset_catalog():
    """
    Generates 30 multi-modal satellite regional scenes across India (120 GeoTIFFs total).
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

    regions_spec = [
        # Pune Specific Micro-Neighborhoods
        ("scene_pune_hadapsar", "Maharashtra, Pune Hadapsar & Magarpatta", "urban", "LISS-IV", 5.8, (73.915, 18.490, 73.965, 18.540), "2024-05-18"),
        ("scene_pune_hinjawadi", "Maharashtra, Pune Hinjawadi IT Hub", "urban", "Sentinel-2", 10.0, (73.710, 18.570, 73.760, 18.620), "2024-05-18"),
        ("scene_pune_kothrud", "Maharashtra, Pune Kothrud & Hills", "urban", "LISS-IV", 5.8, (73.790, 18.490, 73.840, 18.540), "2024-05-18"),
        ("scene_pune_shivajinagar", "Maharashtra, Pune Shivajinagar Confluence", "urban", "LISS-IV", 5.8, (73.835, 18.515, 73.885, 18.565), "2024-05-18"),
        ("scene_pune_khadakwasla", "Maharashtra, Pune Khadakwasla Lake", "forest", "Sentinel-2", 10.0, (73.740, 18.410, 73.790, 18.460), "2024-05-18"),

        # Indian Regional Geographic Diversity
        ("scene_01_west_bengal_nadia", "West Bengal, Nadia District", "agriculture", "Sentinel-2", 10.0, (88.40, 22.90, 88.55, 23.05), "2024-05-12"),
        ("scene_02_maharashtra_pune", "Maharashtra, Pune Metropolitan Basin", "urban", "LISS-IV", 5.8, (73.80, 18.45, 73.95, 18.60), "2024-04-18"),
        ("scene_03_kerala_alappuzha", "Kerala, Alappuzha Backwaters", "coastal", "Sentinel-2", 10.0, (76.30, 9.45, 76.45, 9.60), "2024-06-02"),
        ("scene_04_punjab_ludhiana", "Punjab, Ludhiana Farms", "agriculture", "Sentinel-2", 10.0, (75.80, 30.85, 75.95, 31.00), "2024-03-22"),
        ("scene_05_karnataka_bengaluru", "Karnataka, Bengaluru Urban", "urban", "LISS-IV", 5.8, (77.55, 12.90, 77.70, 13.05), "2024-04-10"),
        ("scene_06_uttarakhand_dehradun", "Uttarakhand, Dehradun Valley", "forest", "Sentinel-2", 10.0, (77.98, 30.28, 78.13, 30.43), "2024-05-04"),
        ("scene_07_rajasthan_jaisalmer", "Rajasthan, Thar Desert", "arid", "Sentinel-2", 10.0, (70.85, 26.85, 71.00, 27.00), "2024-02-15"),
        ("scene_08_odisha_cuttack", "Odisha, Mahanadi Delta", "agriculture", "Sentinel-2", 10.0, (85.80, 20.40, 85.95, 20.55), "2024-06-18"),
        ("scene_09_assam_kaziranga", "Assam, Brahmaputra Floodplain", "wetland", "Sentinel-2", 10.0, (93.30, 26.60, 93.45, 26.75), "2024-07-02"),
        ("scene_10_tamilnadu_cauvery", "Tamil Nadu, Cauvery Delta", "agriculture", "Sentinel-2", 10.0, (79.10, 10.75, 79.25, 10.90), "2024-01-28"),
        ("scene_11_telangana_hyderabad", "Telangana, Hyderabad Plateau", "urban", "LISS-IV", 5.8, (78.40, 17.30, 78.55, 17.45), "2024-03-14"),
        ("scene_12_gujarat_kutch", "Gujarat, Rann of Kutch", "arid", "Sentinel-2", 10.0, (69.80, 23.70, 69.95, 23.85), "2024-02-20"),
        ("scene_13_madhyapradesh_bhopal", "Madhya Pradesh, Upper Lake", "forest", "Sentinel-2", 10.0, (77.35, 23.20, 77.50, 23.35), "2024-04-25"),
        ("scene_14_himachal_shimla", "Himachal Pradesh, Himalayan Range", "mountain", "Sentinel-2", 10.0, (77.10, 31.05, 77.25, 31.20), "2024-05-18"),
        ("scene_15_andhra_visakhapatnam", "Andhra Pradesh, Vizag Coast", "coastal", "LISS-IV", 5.8, (83.25, 17.65, 83.40, 17.80), "2024-04-02"),
        ("scene_16_bihar_patna", "Bihar, Gangetic Plains", "agriculture", "Sentinel-2", 10.0, (85.10, 25.55, 85.25, 25.70), "2024-03-30"),
        ("scene_17_jharkhand_ranchi", "Jharkhand, Chota Nagpur Plateau", "forest", "Sentinel-2", 10.0, (85.28, 23.30, 85.43, 23.45), "2024-05-11"),
        ("scene_18_chhattisgarh_bastar", "Chhattisgarh, Indravati Forest", "forest", "Sentinel-2", 10.0, (81.80, 19.10, 81.95, 19.25), "2024-06-08"),
        ("scene_19_goa_mandovi", "Goa, Mandovi Estuary", "coastal", "Sentinel-2", 10.0, (73.80, 15.45, 73.95, 15.60), "2024-05-20"),
        ("scene_20_haryana_karnal", "Haryana, Agro Corridor", "agriculture", "Sentinel-2", 10.0, (76.90, 29.65, 77.05, 29.80), "2024-02-28"),
        ("scene_21_uttarpradesh_varanasi", "Uttar Pradesh, Varanasi Basin", "agriculture", "Sentinel-2", 10.0, (82.95, 25.25, 83.10, 25.40), "2024-04-05"),
        ("scene_22_maharashtra_nashik", "Maharashtra, Godavari River", "agriculture", "LISS-IV", 5.8, (73.75, 19.95, 73.90, 20.10), "2024-03-18"),
        ("scene_23_kerala_wayanad", "Kerala, Western Ghats Highlands", "forest", "Sentinel-2", 10.0, (76.05, 11.60, 76.20, 11.75), "2024-06-15"),
        ("scene_24_westbengal_sunderbans", "West Bengal, Sunderbans Delta", "wetland", "Sentinel-2", 10.0, (88.75, 21.80, 88.90, 21.95), "2024-05-30"),
        ("scene_25_karnataka_mysuru", "Karnataka, Mysuru Basin", "agriculture", "Sentinel-2", 10.0, (76.60, 12.25, 76.75, 12.40), "2024-01-20"),
        ("scene_26_gujarat_surat", "Gujarat, Tapi Coastal Plain", "urban", "LISS-IV", 5.8, (72.78, 21.15, 72.93, 21.30), "2024-04-12"),
        ("scene_27_tamilnadu_madurai", "Tamil Nadu, Vaigai Basin", "agriculture", "Sentinel-2", 10.0, (78.08, 9.88, 78.23, 10.03), "2024-02-10"),
        ("scene_28_rajasthan_udaipur", "Rajasthan, Aravalli Lakes", "mountain", "Sentinel-2", 10.0, (73.65, 24.55, 73.80, 24.70), "2024-03-08"),
        ("scene_29_odisha_chilika", "Odisha, Chilika Lagoon", "coastal", "Sentinel-2", 10.0, (85.25, 19.65, 85.40, 19.80), "2024-06-25"),
        ("scene_30_assam_guwahati", "Assam, Kamrup Hills", "wetland", "Sentinel-2", 10.0, (91.70, 26.10, 91.85, 26.25), "2024-07-10")
    ]

    H, W = 512, 512
    manifest = []
    csv_rows = []

    print(f"Starting collection & generation of {len(regions_spec)} multi-modal satellite scenes (120 GeoTIFFs)...")

    for idx, (sid, region, terrain, sensor, res, bounds, date) in enumerate(regions_spec, 1):
        seed = 100 + idx * 7

        # 1. Ground Truth Clear Optical Scene
        clear_optical = generate_regional_landscape(H, W, terrain, seed=seed)

        # 2. Historical Scene (Acquired earlier, matching geography with realistic seasonal variations)
        hist_optical = clear_optical.copy()
        hist_optical[:, :, 0] = np.clip(hist_optical[:, :, 0] * 1.02 + 0.01, 0.0, 1.0)
        hist_optical[:, :, 1] = np.clip(hist_optical[:, :, 1] * 0.95 + 0.01, 0.0, 1.0)
        hist_optical[:, :, 2] = np.clip(hist_optical[:, :, 2] * 1.04 - 0.01, 0.0, 1.0)
        hist_optical[:, :, 3] = np.clip(hist_optical[:, :, 3] * 0.96 + 0.02, 0.0, 1.0)
        field_shifts = ndimage.gaussian_filter(np.random.randn(H, W) * 0.015, sigma=3.0)[:, :, np.newaxis]
        hist_optical = np.clip(hist_optical + field_shifts, 0.01, 0.99).astype(np.float32)

        # 3. Realistic Clouds and Shadows
        cloud_density, cloud_mask, shadow_mask = generate_clouds_and_shadows(H, W, seed=seed + 33)
        cloud_pct = float(np.sum(cloud_mask) / float(H * W) * 100.0)

        # 4. Cloudy Scene
        c_density_3d = cloud_density[:, :, np.newaxis]
        s_mask_3d = shadow_mask[:, :, np.newaxis]
        cloud_color = np.array([0.92, 0.95, 0.98, 0.90], dtype=np.float32)
        cloudy_optical = (1.0 - c_density_3d) * clear_optical + c_density_3d * cloud_color
        cloudy_optical = np.where(s_mask_3d > 0, cloudy_optical * 0.40, cloudy_optical)
        cloudy_optical = np.clip(cloudy_optical, 0.0, 1.0).astype(np.float32)

        # 5. Sentinel-1 SAR Radar Scene
        sar_image = generate_sar_backscatter(clear_optical, seed=seed + 88)

        # Save rasters
        meta = ImageMetadata(
            image_id=sid,
            filename=f"{sid}_cloudy.tif",
            width=W,
            height=H,
            bands=4,
            crs="EPSG:4326",
            resolution=res,
            sensor=sensor,
            acquisition_date=date,
            region=region,
            bounds=bounds
        )

        cloudy_path = os.path.join(dirs["cloudy"], f"{sid}_cloudy.tif")
        clear_path = os.path.join(dirs["clear"], f"{sid}_clear.tif")
        hist_path = os.path.join(dirs["historical"], f"{sid}_historical.tif")
        sar_path = os.path.join(dirs["sar"], f"{sid}_sar.tif")

        GeoTIFFLoader.save_raster(cloudy_path, cloudy_optical, reference_meta=meta)
        GeoTIFFLoader.save_raster(clear_path, clear_optical, reference_meta=meta)
        GeoTIFFLoader.save_raster(hist_path, hist_optical, reference_meta=meta)
        GeoTIFFLoader.save_raster(sar_path, sar_image, reference_meta=meta)

        sample_entry = {
            "image_id": sid,
            "region": region,
            "terrain_type": terrain,
            "date": date,
            "optical_sensor": sensor,
            "sar_sensor": "Sentinel-1 C-SAR",
            "crs": "EPSG:4326",
            "resolution": res,
            "cloud_cover_pct": round(cloud_pct, 2),
            "bounds": bounds,
            "files": {
                "cloudy": cloudy_path,
                "clear": clear_path,
                "historical": hist_path,
                "sar": sar_path
            }
        }
        manifest.append(sample_entry)

        csv_rows.append({
            "image_id": sid,
            "region": region,
            "terrain_type": terrain,
            "optical_sensor": sensor,
            "sar_sensor": "Sentinel-1 C-SAR",
            "date": date,
            "crs": "EPSG:4326",
            "resolution_m": res,
            "cloud_cover_pct": round(cloud_pct, 2),
            "bounds_lon_min": bounds[0],
            "bounds_lat_min": bounds[1],
            "bounds_lon_max": bounds[2],
            "bounds_lat_max": bounds[3],
            "cloudy_geotiff": cloudy_path,
            "clear_geotiff": clear_path,
            "historical_geotiff": hist_path,
            "sar_geotiff": sar_path
        })

        print(f"[{idx:02d}/30] Generated {sid} ({region}) - Cloud: {cloud_pct:.1f}%")

    # Write Manifest JSON
    manifest_path = os.path.join(dirs["samples"], "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write Metadata CSV
    csv_path = os.path.join(base_dir, "data", "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDataset Collection Completed Successfully!")
    print(f"Total Scenes: {len(manifest)}")
    print(f"Total GeoTIFF Files: {len(manifest) * 4} (Cloudy, Clear, Historical, SAR)")
    print(f"Catalog Manifest: {manifest_path}")
    print(f"Metadata CSV: {csv_path}")


def generate_custom_aoi_scene(
    bounds: Tuple[float, float, float, float],
    region_name: str = "Custom Selected AOI",
    terrain: str = "urban",
    sensor: str = "LISS-IV",
    res: float = 5.8
) -> Dict[str, Any]:
    """
    Dynamically creates multi-modal satellite files (Cloudy, Clear, Historical, SAR)
    for any custom latitude/longitude bounding box clicked or drawn on the map.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = {
        "cloudy": os.path.join(base_dir, "data", "cloudy"),
        "clear": os.path.join(base_dir, "data", "clear"),
        "historical": os.path.join(base_dir, "data", "historical"),
        "sar": os.path.join(base_dir, "data", "sar")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    clean_id = "scene_custom_" + "".join(c if c.isalnum() else "_" for c in region_name.lower())[:30]
    H, W = 512, 512
    seed = int(abs(hash(region_name)) % 10000)

    clear_optical = generate_regional_landscape(H, W, terrain, seed=seed)

    hist_optical = clear_optical.copy()
    hist_optical[:, :, 0] = np.clip(hist_optical[:, :, 0] * 1.02 + 0.01, 0.0, 1.0)
    hist_optical[:, :, 1] = np.clip(hist_optical[:, :, 1] * 0.95 + 0.01, 0.0, 1.0)
    hist_optical[:, :, 2] = np.clip(hist_optical[:, :, 2] * 1.04 - 0.01, 0.0, 1.0)
    hist_optical[:, :, 3] = np.clip(hist_optical[:, :, 3] * 0.96 + 0.02, 0.0, 1.0)
    field_shifts = ndimage.gaussian_filter(np.random.randn(H, W) * 0.015, sigma=3.0)[:, :, np.newaxis]
    hist_optical = np.clip(hist_optical + field_shifts, 0.01, 0.99).astype(np.float32)

    cloud_density, cloud_mask, shadow_mask = generate_clouds_and_shadows(H, W, seed=seed + 33)
    cloud_pct = float(np.sum(cloud_mask) / float(H * W) * 100.0)

    c_density_3d = cloud_density[:, :, np.newaxis]
    s_mask_3d = shadow_mask[:, :, np.newaxis]
    cloud_color = np.array([0.92, 0.95, 0.98, 0.90], dtype=np.float32)
    cloudy_optical = (1.0 - c_density_3d) * clear_optical + c_density_3d * cloud_color
    cloudy_optical = np.where(s_mask_3d > 0, cloudy_optical * 0.40, cloudy_optical)
    cloudy_optical = np.clip(cloudy_optical, 0.0, 1.0).astype(np.float32)

    sar_image = generate_sar_backscatter(clear_optical, seed=seed + 88)

    meta = ImageMetadata(
        image_id=clean_id,
        filename=f"{clean_id}_cloudy.tif",
        width=W,
        height=H,
        bands=4,
        crs="EPSG:4326",
        resolution=res,
        sensor=sensor,
        acquisition_date="2024-05-20",
        region=region_name,
        bounds=bounds,
        dtype="float32"
    )

    cloudy_path = os.path.join(dirs["cloudy"], f"{clean_id}_cloudy.tif")
    clear_path = os.path.join(dirs["clear"], f"{clean_id}_clear.tif")
    hist_path = os.path.join(dirs["historical"], f"{clean_id}_historical.tif")
    sar_path = os.path.join(dirs["sar"], f"{clean_id}_sar.tif")

    GeoTIFFLoader.save_raster(cloudy_path, cloudy_optical, reference_meta=meta)
    GeoTIFFLoader.save_raster(clear_path, clear_optical, reference_meta=meta)
    GeoTIFFLoader.save_raster(hist_path, hist_optical, reference_meta=meta)
    GeoTIFFLoader.save_raster(sar_path, sar_image, reference_meta=meta)

    return {
        "image_id": clean_id,
        "id": clean_id,
        "region": region_name,
        "terrain_type": terrain,
        "optical_sensor": sensor,
        "sar_sensor": "Sentinel-1 C-SAR",
        "date": "2024-05-20",
        "resolution": res,
        "resolution_m": res,
        "cloud_cover_pct": round(cloud_pct, 2),
        "bounds": list(bounds),
        "files": {
            "cloudy": cloudy_path,
            "clear": clear_path,
            "historical": hist_path,
            "sar": sar_path
        },
        "paths": {
            "cloudy": cloudy_path,
            "clear": clear_path,
            "historical": hist_path,
            "sar": sar_path
        }
    }


if __name__ == "__main__":
    generate_full_dataset_catalog()
