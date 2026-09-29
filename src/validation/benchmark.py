"""Benchmark reconstruction vs baselines on bundled data + any real scenes.

Compares CloudClear MRR output against:
  - baseline-historical: copy historical pixels under cloud
  - baseline-sar-mean: mean-fill ( SAR-blind lower bound )
Metrics: PSNR/SSIM/SAM/ERGAS/MAE via QualityAssessmentNetwork.
Writes reports/validation_report.json with per-scene + aggregate + limitations.

This does NOT claim real-satellite validation unless data/* contains real
scenes (see provenance). It makes the validation protocol explicit and
re-runnable: `python -m src.validation.benchmark`.
"""
import glob
import json
import os
import time
import numpy as np

from ..preprocessing.data_loader import GeoTIFFLoader
from ..pipeline.cloudclear_pipeline import CloudClearPipeline
from ..qan.quality_network import QualityAssessmentNetwork


def _baseline_historical(cloudy, hist, mask):
    m = (mask > 0)[:, :, None].astype(np.float32)
    return (1 - m) * cloudy + m * hist


def run_benchmark(data_dir="data", output_dir="outputs", reports_dir="reports",
                  max_scenes=5, agent_mode="sequential"):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = data_dir if os.path.isabs(data_dir) else os.path.join(base, data_dir)
    reports_dir = reports_dir if os.path.isabs(reports_dir) else os.path.join(base, reports_dir)
    os.makedirs(reports_dir, exist_ok=True)
    cloudy_files = sorted(glob.glob(os.path.join(data_dir, "cloudy", "*_cloudy.tif")))[:max_scenes]
    pipe = CloudClearPipeline(output_dir=output_dir, reports_dir=reports_dir)
    qan = QualityAssessmentNetwork()
    loader = GeoTIFFLoader()
    scenes = []
    for cf in cloudy_files:
        prefix = os.path.basename(cf).replace("_cloudy.tif", "")
        hf = os.path.join(data_dir, "historical", f"{prefix}_historical.tif")
        sf = os.path.join(data_dir, "sar", f"{prefix}_sar.tif")
        rf = os.path.join(data_dir, "clear", f"{prefix}_clear.tif")
        if not os.path.exists(rf):
            continue
        pkt = pipe.run(cf, hf if os.path.exists(hf) else None,
                       sf if os.path.exists(sf) else None, rf,
                       image_id=f"bench_{prefix}", agent_mode=agent_mode)
        cloudy, _ = loader.load_raster(cf)
        hist, _ = loader.load_raster(hf) if os.path.exists(hf) else (cloudy.copy(), None)
        ref, _ = loader.load_raster(rf)
        occ = ((pkt.cloud_detection["cloud_mask"] > 0)).astype(np.uint8)
        base_h = _baseline_historical(
            np.clip(cloudy, 0, 1), np.clip(hist, 0, 1), occ)
        m_ours = qan.calculate_metrics(pkt.reconstructed_image, np.clip(ref, 0, 1), occ).to_dict()
        m_base = qan.calculate_metrics(base_h, np.clip(ref, 0, 1), occ).to_dict()
        scenes.append({
            "scene": prefix,
            "cloud_pct": pkt.cloud_detection["cloud_percentage"],
            "strategy": pkt.decision.get("strategy"),
            "ours": m_ours, "baseline_historical": m_base,
            "improvement_psnr_db": round(m_ours["psnr"] - m_base["psnr"], 2),
        })
    agg = {}
    if scenes:
        for k in ["psnr", "ssim", "sam", "ergas", "mae"]:
            agg[f"mean_ours_{k}"] = round(float(np.mean([s["ours"][k] for s in scenes])), 3)
            agg[f"mean_base_{k}"] = round(float(np.mean([s["baseline_historical"][k] for s in scenes])), 3)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenes": scenes,
        "aggregate": agg,
        "limitations": (
            "Bundled data/* scenes are synthetic/pseudo-esri unless provenance says "
            "real-sentinel2. This benchmark measures protocol + relative gain vs "
            "historical-copy baseline, NOT production readiness on real clouds. "
            "Add real S-2/S-1 pairs with ground truth to claim real-world validation."
        ),
    }
    out = os.path.join(reports_dir, "validation_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return report
