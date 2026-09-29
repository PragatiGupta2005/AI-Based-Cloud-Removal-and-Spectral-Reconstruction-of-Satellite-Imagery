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


class SensorProvenance:
    """Tracks whether a scene is real multispectral or synthetic/proxy.

    data_source values:
      - "real-sentinel2": freshly downloaded Sentinel-2 L2A B02/B03/B04/B08
      - "real-sentinel1": freshly downloaded Sentinel-1 GRD VV/VH
      - "liss-iv-proxy": resampled Sentinel-2 / Esri RGB used as LISS-IV
        geometric proxy (LISS-IV is not openly downloadable). Flagged explicitly.
      - "synthetic": fully procedural landscape (collect_datasets / generate_sample_data)
      - "pseudo-esri": live_map_fetcher Esri RGB + estimated NIR (not a real NIR band)
    """

    REAL_S2 = "real-sentinel2"
    REAL_S1 = "real-sentinel1"
    LISS_IV_PROXY = "liss-iv-proxy"
    SYNTHETIC = "synthetic"
    PSEUDO_ESRI = "pseudo-esri"


class SatelliteDataFetcher:
    """
    Queries and downloads real open Sentinel-2 and Sentinel-1 imagery for any Area of Interest (AOI).

    Real-data path (no API key): Earth-Search STAC v1 + public S3 COGs.
    LISS-IV path: ISRO LISS-IV has no open STAC; this fetcher documents that
    and produces an explicit `liss-iv-proxy` (resampled real S-2 or Esri RGB)
    instead of mislabelling proxy data as genuine LISS-IV.
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

    def search_sentinel1_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: str = "2024-01-01/2024-06-30",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Searches Sentinel-1 GRD scenes via open STAC catalog."""
        search_payload = {
            "collections": ["sentinel-1-grd"],
            "bbox": list(bbox),
            "datetime": date_range,
            "limit": limit,
        }
        try:
            resp = requests.post(f"{STAC_API_URL}/search", json=search_payload, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("features", [])
            print(f"STAC S-1 search returned status {resp.status_code}: {resp.text}")
            return []
        except Exception as e:
            print(f"STAC S-1 search request failed: {e}")
            return []

    def _band_href(self, feature: Dict[str, Any], candidates: List[str]) -> Optional[str]:
        assets = feature.get("assets", {}) or {}
        for key in candidates:
            if key in assets and isinstance(assets[key], dict) and assets[key].get("href"):
                return assets[key]["href"]
        # fallback: case-insensitive scan
        for k, v in assets.items():
            if any(c.lower() in k.lower() for c in candidates) and isinstance(v, dict) and v.get("href"):
                return v.get("href")
        return None

    def fetch_sentinel2_as_geotiff(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: str = "2024-01-01/2024-06-30",
        region_name: str = "STAC Sentinel-2 scene",
        target_size: Tuple[int, int] = (512, 512),
        max_cloud_cover: float = 100.0,
    ) -> Dict[str, Any]:
        """Downloads real Sentinel-2 L2A B02/B03/B04/B08, stacks to 4-band GeoTIFF.

        Returns dict with file paths + provenance. Raises RuntimeError if no
        scene or download fails so callers can fall back to synthetic with a flag.
        """
        features = self.search_sentinel2_scenes(bbox, date_range, max_cloud_cover, limit=3)
        if not features:
            raise RuntimeError("No Sentinel-2 scenes found for bbox/date.")
        try:
            import rasterio
            from rasterio.windows import from_bounds as _fb
            from rasterio.enums import Resampling as _RS
        except ImportError as e:
            raise RuntimeError(f"rasterio required for real S-2 fetch: {e}")

        last_err: Optional[str] = None
        for feat in features:
            try:
                href_b02 = self._band_href(feat, ["blue", "B02", "band02", "02"])
                href_b03 = self._band_href(feat, ["green", "B03", "band03", "03"])
                href_b04 = self._band_href(feat, ["red", "B04", "band04", "04"])
                href_b08 = self._band_href(feat, ["nir", "B08", "band08", "08", "nir08"])
                if not all([href_b02, href_b03, href_b04, href_b08]):
                    last_err = f"scene {feat.get('id')} missing B02/B03/B04/B08 assets"
                    continue
                bands = []
                profile = None
                for href in [href_b02, href_b03, href_b04, href_b08]:
                    with rasterio.open(f"/vsicurl/{href}") as src:
                        win = _fb(*bbox, transform=src.transform)
                        arr = src.read(
                            1,
                            window=win,
                            out_shape=target_size,
                            resampling=_RS.bilinear,
                            boundless=True,
                            fill_value=0,
                        )
                        bands.append(arr.astype(np.float32))
                        if profile is None:
                            profile = src.profile.copy()
                stack = np.stack(bands, axis=-1)  # H,W,4 in raw DN/reflectance
                # Scale Sentinel-2 L2A (0-10000) to 0-1
                if stack.max() > 1.5:
                    stack = np.clip(stack / 10000.0, 0.0, 1.0)
                clean_id = "scene_real_s2_" + "".join(
                    c if c.isalnum() else "_" for c in region_name.lower()
                )[:28]
                props = feat.get("properties", {})
                meta = ImageMetadata(
                    image_id=clean_id,
                    filename=f"{clean_id}_real_s2.tif",
                    width=target_size[1],
                    height=target_size[0],
                    bands=4,
                    crs="EPSG:4326",
                    resolution=10.0,
                    sensor="Sentinel-2 L2A (real STAC download)",
                    acquisition_date=str(props.get("datetime", date_range.split("/")[0])),
                    region=f"{region_name} [data_source=real-sentinel2, stac_id={feat.get('id')}]",
                    bounds=bbox,
                )
                out_path = os.path.join(self.clear_dir, f"{clean_id}_clear.tif")
                GeoTIFFLoader.save_raster(out_path, stack.astype(np.float32), reference_meta=meta)
                return {
                    "image_id": clean_id,
                    "region": region_name,
                    "optical_path": out_path,
                    "optical": stack.astype(np.float32),
                    "data_source": SensorProvenance.REAL_S2,
                    "stac_id": feat.get("id"),
                    "cloud_cover": props.get("eo:cloud_cover"),
                    "meta": meta.to_dict(),
                }
            except Exception as e:
                last_err = str(e)
                continue
        raise RuntimeError(f"Real Sentinel-2 download failed: {last_err}")

    def fetch_sentinel1_as_array(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: str = "2024-01-01/2024-06-30",
        target_size: Tuple[int, int] = (512, 512),
    ) -> Dict[str, Any]:
        """Searches real Sentinel-1 GRD and downloads VV/VH assets if present.

        Many public S-1 assets are full GRD products (not windowed COGs), so this
        returns file paths + provenance and raises if unusable, letting callers
        fall back to synthetic SAR with an explicit flag.
        """
        features = self.search_sentinel1_scenes(bbox, date_range, limit=3)
        if not features:
            raise RuntimeError("No Sentinel-1 scenes found for bbox/date.")
        feat = features[0]
        vv_href = self._band_href(feat, ["vv", "VH_VV", "vv-band"])
        vh_href = self._band_href(feat, ["vh", "vh-band"])
        if not vv_href:
            raise RuntimeError(
                f"S-1 scene {feat.get('id')} has no directly downloadable VV COG; "
                "use synthetic SAR fallback (flagged) or download GRD offline."
            )
        tmp_vv = os.path.join(self.sar_dir, "tmp_s1_vv.tif")
        ok = self.download_asset(vv_href, tmp_vv)
        if not ok:
            raise RuntimeError("Sentinel-1 VV download failed.")
        sar_info: Dict[str, Any] = {
            "stac_id": feat.get("id"),
            "vv_path": tmp_vv,
            "vh_href": vh_href,
            "data_source": SensorProvenance.REAL_S1,
        }
        if vh_href:
            tmp_vh = os.path.join(self.sar_dir, "tmp_s1_vh.tif")
            if self.download_asset(vh_href, tmp_vh):
                sar_info["vh_path"] = tmp_vh
        return sar_info

    def liss_iv_proxy_note(self) -> Dict[str, str]:
        """Documents LISS-IV availability (audit point Sec.4)."""
        return {
            "sensor": "LISS-IV",
            "availability": "not-open",
            "policy": (
                "ISRO LISS-IV has no open STAC download. CloudClear labels "
                "high-res geometry as 'liss-iv-proxy' (resampled real S-2 or "
                "Esri RGB) and never claims it is genuine LISS-IV L1 data."
            ),
            "data_source": SensorProvenance.LISS_IV_PROXY,
        }

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
    parser.add_argument("--download-s2", action="store_true", help="Download real S-2 B02/B03/B04/B08 and stack to GeoTIFF")
    parser.add_argument("--search-s1", action="store_true", help="Search real Sentinel-1 GRD scenes")
    args = parser.parse_args()

    fetcher = SatelliteDataFetcher()
    print(f"Searching open Sentinel-2 scenes for bbox: {args.bbox} over {args.date}...")
    scenes = fetcher.search_sentinel2_scenes(bbox=tuple(args.bbox), date_range=args.date, limit=3)
    print(f"Found {len(scenes)} matching scenes from Open STAC API.")
    for sc in scenes:
        props = sc.get("properties", {})
        print(f"- Scene: {sc.get('id')} | Date: {props.get('datetime')} | Cloud Cover: {props.get('eo:cloud_cover')}%")
    if args.download_s2:
        try:
            res = fetcher.fetch_sentinel2_as_geotiff(
                bbox=tuple(args.bbox), date_range=args.date, region_name=args.region
            )
            print(f"Downloaded real Sentinel-2 stack: {res['optical_path']} [{res['data_source']}]")
        except Exception as e:
            print(f"Real S-2 download failed (use synthetic fallback, flagged): {e}")
    if args.search_s1:
        s1 = fetcher.search_sentinel1_scenes(bbox=tuple(args.bbox), date_range=args.date, limit=3)
        print(f"Found {len(s1)} Sentinel-1 scenes.")
        for sc in s1:
            print(f"- S-1 Scene: {sc.get('id')} | Date: {sc.get('properties', {}).get('datetime')}")
    print(f"LISS-IV policy: {fetcher.liss_iv_proxy_note()['policy']}")


if __name__ == "__main__":
    main()
