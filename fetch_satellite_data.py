"""
Real Satellite Data Downloader & STAC Client for CloudClear AI.
Fetches real Sentinel-2 Level-2A (B2, B3, B4, B8) and Sentinel-1 GRD SAR data
from open public STAC (SpatioTemporal Asset Catalog) APIs.
"""

import os
import sys
import json
import argparse
import requests
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    pass

from src.preprocessing.data_loader import GeoTIFFLoader, ImageMetadata

# Public open STAC API endpoints (No API key required for search & open assets)
STAC_API_URL = "https://earth-search.aws.element84.com/v1"


class SatelliteDataFetcher:
    """
    Queries and downloads real open Sentinel-2 and Sentinel-1 imagery for any Area of Interest (AOI).
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.cloudy_dir = os.path.join(data_dir, "cloudy")
        self.clear_dir = os.path.join(data_dir, "clear")
        self.hist_dir = os.path.join(data_dir, "historical")
        self.sar_dir = os.path.join(data_dir, "sar")

        for d in [self.cloudy_dir, self.clear_dir, self.hist_dir, self.sar_dir]:
            os.makedirs(d, exist_ok=True)

    def search_sentinel2_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: str = "2024-01-01/2024-06-30",
        max_cloud_cover: float = 100.0,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches Sentinel-2 L2A scenes via open STAC catalog.
        bbox format: (min_lon, min_lat, max_lon, max_lat)
        """
        search_payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": list(bbox),
            "datetime": date_range,
            "query": {
                "eo:cloud_cover": {"lt": max_cloud_cover}
            },
            "limit": limit
        }

        try:
            resp = requests.post(f"{STAC_API_URL}/search", json=search_payload, timeout=15)
            if resp.status_code == 200:
                features = resp.json().get("features", [])
                return features
            else:
                print(f"STAC search returned status {resp.status_code}: {resp.text}")
                return []
        except Exception as e:
            print(f"STAC search request failed: {e}")
            return []

    def download_asset(self, url: str, target_path: str) -> bool:
        """Downloads a public raster asset with streaming."""
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                return True
            return False
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Fetch real satellite scenes via Open STAC API.")
    parser.add_argument("--region", type=str, default="West Bengal", help="Region name")
    parser.add_argument("--bbox", type=float, nargs=4, default=[88.40, 22.90, 88.55, 23.05], help="min_lon min_lat max_lon max_lat")
    parser.add_argument("--date", type=str, default="2024-01-01/2024-06-30", help="Date range YYYY-MM-DD/YYYY-MM-DD")
    args = parser.parse_args()

    fetcher = SatelliteDataFetcher()
    print(f"Searching open Sentinel-2 scenes for bbox: {args.bbox} over {args.date}...")
    scenes = fetcher.search_sentinel2_scenes(bbox=tuple(args.bbox), date_range=args.date, limit=3)
    print(f"Found {len(scenes)} matching scenes from Open STAC API.")
    for sc in scenes:
        props = sc.get("properties", {})
        print(f"- Scene: {sc.get('id')} | Date: {props.get('datetime')} | Cloud Cover: {props.get('eo:cloud_cover')}%")


if __name__ == "__main__":
    main()
