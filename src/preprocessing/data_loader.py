"""
GeoTIFF loader, validator, and metadata extraction module for CloudClear AI.
Supports LISS-IV, Sentinel-2 Optical (B2-Blue, B3-Green, B4-Red, B8-NIR),
and Sentinel-1 SAR imagery.
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    image_id: str
    filename: str
    width: int
    height: int
    bands: int
    crs: str
    resolution: float
    sensor: str = "Sentinel-2"
    acquisition_date: str = "2024-05-12"
    region: str = "West Bengal"
    bounds: Optional[Tuple[float, float, float, float]] = None
    dtype: str = "float32"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.bounds:
            d["bounds"] = list(self.bounds)
        return d


def validate_geotiff(file_path: str) -> Tuple[bool, str, Optional[ImageMetadata]]:
    """
    Validates if a file is a valid GeoTIFF / multi-band satellite raster.
    Returns (is_valid, message, metadata).
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}", None

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 2048:
        return False, f"File exceeds maximum allowed size (2048 MB). Current: {file_size_mb:.2f} MB", None

    try:
        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    width = src.width
                    height = src.height
                    count = src.count
                    crs_str = str(src.crs) if src.crs else "EPSG:4326"
                    res = float(src.res[0]) if src.res else 10.0
                    bounds = tuple(src.bounds) if src.bounds else (88.0, 22.0, 89.0, 23.0)
                    dtype_str = str(src.dtypes[0])

                    if count < 1:
                        return False, "Image contains zero bands.", None

                    meta = ImageMetadata(
                        image_id=os.path.splitext(os.path.basename(file_path))[0],
                        filename=os.path.basename(file_path),
                        width=width,
                        height=height,
                        bands=count,
                        crs=crs_str,
                        resolution=res,
                        bounds=bounds,
                        dtype=dtype_str
                    )
                    return True, "Valid GeoTIFF", meta
            except rasterio.errors.RasterioIOError:
                pass  # Fallback to tifffile / PIL

        # Fallback reading
        if HAS_TIFFFILE:
            arr = tifffile.imread(file_path)
            shape = arr.shape
            if len(shape) == 2:
                height, width, count = shape[0], shape[1], 1
            elif len(shape) == 3:
                if shape[0] in [1, 2, 3, 4, 8, 12, 13]:
                    count, height, width = shape[0], shape[1], shape[2]
                else:
                    height, width, count = shape[0], shape[1], shape[2]
            else:
                return False, f"Unsupported array dimensions: {shape}", None

            meta = ImageMetadata(
                image_id=os.path.splitext(os.path.basename(file_path))[0],
                filename=os.path.basename(file_path),
                width=width,
                height=height,
                bands=count,
                crs="EPSG:4326",
                resolution=10.0,
                bounds=(88.0, 22.0, 89.0, 23.0),
                dtype=str(arr.dtype)
            )
            return True, "Valid TIFF (fallback reader)", meta

        with Image.open(file_path) as img:
            width, height = img.size
            bands = len(img.getbands())
            meta = ImageMetadata(
                image_id=os.path.splitext(os.path.basename(file_path))[0],
                filename=os.path.basename(file_path),
                width=width,
                height=height,
                bands=bands,
                crs="EPSG:4326",
                resolution=10.0,
                bounds=(88.0, 22.0, 89.0, 23.0),
                dtype="uint8"
            )
            return True, "Valid Image (PIL reader)", meta

    except Exception as e:
        return False, f"Failed to validate satellite image: {str(e)}", None


class GeoTIFFLoader:
    """
    Handles loading, writing, and manipulating GeoTIFF satellite files.
    """

    @staticmethod
    def load_raster(file_path: str) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Loads raster data as a numpy array with shape (Bands, Height, Width) or (Height, Width, Bands).
        Returns normalized float32 array in [0, 1] range (or raw values) + metadata.
        """
        is_valid, msg, meta = validate_geotiff(file_path)
        if not is_valid or meta is None:
            raise ValueError(f"Invalid GeoTIFF file: {msg}")

        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    data = src.read()  # Shape: (Bands, Height, Width)
                    # Convert to (H, W, Bands) for neural network compatibility
                    if data.ndim == 3:
                        data = np.transpose(data, (1, 2, 0))
                    elif data.ndim == 2:
                        data = data[:, :, np.newaxis]
                    return data.astype(np.float32), meta
            except Exception as e:
                logger.warning(f"Rasterio reading failed, falling back to tifffile: {e}")

        if HAS_TIFFFILE:
            data = tifffile.imread(file_path)
            if data.ndim == 2:
                data = data[:, :, np.newaxis]
            elif data.ndim == 3 and data.shape[0] in [1, 2, 3, 4, 8, 12, 13]:
                data = np.transpose(data, (1, 2, 0))
            return data.astype(np.float32), meta

        with Image.open(file_path) as img:
            data = np.array(img)
            if data.ndim == 2:
                data = data[:, :, np.newaxis]
            return data.astype(np.float32), meta

    @staticmethod
    def save_raster(
        output_path: str,
        data: np.ndarray,
        reference_meta: Optional[ImageMetadata] = None,
        crs: str = "EPSG:4326",
        nodata: Optional[float] = None
    ) -> str:
        """
        Saves a 2D or 3D numpy array as a GeoTIFF.
        Data expected as (Height, Width, Bands) or (Height, Width).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if data.ndim == 2:
            height, width = data.shape
            bands = 1
            data_bands = data[np.newaxis, :, :]
        elif data.ndim == 3:
            height, width, bands = data.shape
            data_bands = np.transpose(data, (2, 0, 1))  # (Bands, H, W)
        else:
            raise ValueError(f"Unsupported data shape for export: {data.shape}")

        dtype = data.dtype
        if dtype == np.float64:
            data_bands = data_bands.astype(np.float32)
            dtype = np.float32

        if HAS_RASTERIO:
            try:
                transform = None
                if reference_meta and reference_meta.bounds:
                    b = reference_meta.bounds
                    from rasterio.transform import from_bounds
                    transform = from_bounds(b[0], b[1], b[2], b[3], width, height)
                else:
                    transform = Affine.translation(88.0, 22.0) * Affine.scale(0.0001, -0.0001)

                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=bands,
                    dtype=dtype,
                    crs=reference_meta.crs if reference_meta else crs,
                    transform=transform,
                    nodata=nodata
                ) as dst:
                    dst.write(data_bands)
                return output_path
            except Exception as e:
                logger.warning(f"Rasterio save failed, using tifffile fallback: {e}")

        if HAS_TIFFFILE:
            tifffile.imwrite(output_path, data)
            return output_path

        # PIL fallback (1 or 3-4 bands)
        if bands in [1, 3, 4]:
            norm = data.copy()
            if norm.max() <= 1.0 and norm.min() >= 0.0:
                norm = (norm * 255).astype(np.uint8)
            else:
                norm = np.clip(norm, 0, 255).astype(np.uint8)
            if bands == 1:
                norm = norm.squeeze(-1)
            img = Image.fromarray(norm)
            img.save(output_path)
            return output_path

        raise RuntimeError("No suitable GeoTIFF export driver available.")
