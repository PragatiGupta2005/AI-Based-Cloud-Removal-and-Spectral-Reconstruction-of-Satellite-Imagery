"""
End-to-End 11-Stage Intelligent Geospatial AI Pipeline for CloudClear AI.
Orchestrates:
1. Data Retrieval & Validation
2. Preprocessing & Radiometric Normalization
3. Cloud & Shadow Detection (Attention U-Net)
4. Change Detection (Siamese CNN)
5. Adaptive Decision Engine
6. Cross-Attention Multi-Modal Feature Fusion
7. Multi-Hypothesis Reconstruction (MRR - C1, C2, C3)
8. Quality Assessment Network (QAN)
9. Confidence Estimation (High, Medium, Low)
10. Analysis-Ready GeoTIFF Export (NDVI & Land Cover)
11. Delivery, PDF Reporting, and REST / UI Serialization
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np

from ..preprocessing.data_loader import GeoTIFFLoader, ImageMetadata, validate_geotiff
from ..preprocessing.preprocessor import ImagePreprocessor
from ..models.cloud_detector import CloudDetectionModel
from ..models.change_detector import ChangeDetectionModel
from ..models.mrr_reconstructor import MRRReconstructionModel
from ..fusion.decision_engine import AdaptiveDecisionEngine
from ..qan.quality_network import QualityAssessmentNetwork, QualityMetrics
from ..confidence.confidence_estimator import ConfidenceEstimator, ConfidenceReport
from ..analysis.ndvi import NDVIAnalyzer, NDVIReport
from ..analysis.landcover import LandCoverClassifier, LandCoverReport
from ..analysis.sub_cloud_predictor import SubCloudFeaturePredictor, SubCloudFeatureReport
from ..reports.report_generator import PDFReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class PredictionPacket:
    """
    Standard processing packet passed across all 11 pipeline stages.
    """
    image_id: str
    cloudy_path: str
    historical_path: Optional[str] = None
    sar_path: Optional[str] = None
    clear_reference_path: Optional[str] = None
    output_dir: str = "outputs"

    # Stage outputs
    metadata: Optional[ImageMetadata] = None
    cloudy_raw: Optional[np.ndarray] = None
    hist_raw: Optional[np.ndarray] = None
    sar_raw: Optional[np.ndarray] = None
    ref_raw: Optional[np.ndarray] = None

    cloud_detection: Optional[Dict[str, Any]] = None
    change_detection: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    candidates: Optional[Dict[str, np.ndarray]] = None
    best_candidate: str = "C3"
    reconstructed_image: Optional[np.ndarray] = None

    quality_metrics: Optional[QualityMetrics] = None
    all_candidate_metrics: Optional[Dict[str, QualityMetrics]] = None
    confidence_report: Optional[ConfidenceReport] = None
    ndvi_report: Optional[NDVIReport] = None
    landcover_report: Optional[LandCoverReport] = None
    sub_cloud_report: Optional[SubCloudFeatureReport] = None

    # File output paths
    cloud_free_geotiff_path: Optional[str] = None
    confidence_geotiff_path: Optional[str] = None
    change_geotiff_path: Optional[str] = None
    report_pdf_path: Optional[str] = None
    metadata_json_path: Optional[str] = None

    # Execution stats
    elapsed_seconds: float = 0.0
    status: str = "Initialized"

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "status": self.status,
            "best_candidate": self.best_candidate,
            "cloud_percentage": self.cloud_detection.get("cloud_percentage", 0.0) if self.cloud_detection else 0.0,
            "shadow_percentage": self.cloud_detection.get("shadow_percentage", 0.0) if self.cloud_detection else 0.0,
            "strategy": self.decision.get("strategy", "") if self.decision else "",
            "quality": self.quality_metrics.to_dict() if self.quality_metrics else {},
            "confidence": self.confidence_report.to_dict() if self.confidence_report else {},
            "ndvi": self.ndvi_report.to_dict() if self.ndvi_report else {},
            "landcover": self.landcover_report.to_dict() if self.landcover_report else {},
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "files": {
                "cloud_free_geotiff": self.cloud_free_geotiff_path,
                "confidence_geotiff": self.confidence_geotiff_path,
                "change_geotiff": self.change_geotiff_path,
                "pdf_report": self.report_pdf_path,
                "metadata_json": self.metadata_json_path
            }
        }


class CloudClearPipeline:
    """
    Unified 11-Stage Pipeline Orchestrator for CloudClear AI.
    """

    def __init__(self, output_dir: str = "outputs", reports_dir: str = "reports"):
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # Lazy-loaded singletons for memory efficiency
        self.loader = GeoTIFFLoader()
        self.preprocessor = ImagePreprocessor(patch_size=256)
        self.cloud_model = CloudDetectionModel(patch_size=256)
        self.change_model = ChangeDetectionModel(patch_size=256)
        self.mrr_model = MRRReconstructionModel(patch_size=256)
        self.decision_engine = AdaptiveDecisionEngine()
        self.qan = QualityAssessmentNetwork()
        self.confidence_estimator = ConfidenceEstimator()
        self.ndvi_analyzer = NDVIAnalyzer()
        self.landcover_classifier = LandCoverClassifier()
        self.sub_cloud_predictor = SubCloudFeaturePredictor()
        self.pdf_generator = PDFReportGenerator()

    def run(
        self,
        cloudy_path: str,
        historical_path: Optional[str] = None,
        sar_path: Optional[str] = None,
        clear_reference_path: Optional[str] = None,
        image_id: Optional[str] = None,
        strategy_override: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> PredictionPacket:
        """
        Executes all 11 stages of the pipeline.
        """
        start_time = time.time()
        if image_id is None:
            image_id = os.path.splitext(os.path.basename(cloudy_path))[0]

        packet = PredictionPacket(
            image_id=image_id,
            cloudy_path=cloudy_path,
            historical_path=historical_path,
            sar_path=sar_path,
            clear_reference_path=clear_reference_path,
            output_dir=self.output_dir
        )

        def report_step(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct:.0f}%] {msg}")

        # --- Stage 1: Data Retrieval & Validation ---
        report_step(10.0, "Stage 1: Loading & validating GeoTIFF satellite imagery...")
        cloudy_data, meta = self.loader.load_raster(cloudy_path)
        packet.metadata = meta
        packet.cloudy_raw = cloudy_data

        H, W = cloudy_data.shape[:2]

        if historical_path and os.path.exists(historical_path):
            hist_data, _ = self.loader.load_raster(historical_path)
        else:
            hist_data = cloudy_data.copy()
        packet.hist_raw = hist_data

        if sar_path and os.path.exists(sar_path):
            sar_data, _ = self.loader.load_raster(sar_path)
        else:
            # Synthetic default SAR channels if not provided
            sar_data = np.zeros((H, W, 2), dtype=np.float32)
        packet.sar_raw = sar_data

        if clear_reference_path and os.path.exists(clear_reference_path):
            ref_data, _ = self.loader.load_raster(clear_reference_path)
        else:
            ref_data = hist_data.copy()
        packet.ref_raw = ref_data

        # --- Stage 2: Preprocessing & Radiometric Normalization ---
        report_step(20.0, "Stage 2: Radiometric normalization & spatial alignment...")
        cloudy_norm, _ = self.preprocessor.normalize_radiometric(cloudy_data)
        hist_norm, _ = self.preprocessor.normalize_radiometric(hist_data)
        sar_norm, _ = self.preprocessor.normalize_radiometric(sar_data)
        ref_norm, _ = self.preprocessor.normalize_radiometric(ref_data)

        # --- Stage 3: Cloud & Shadow Detection ---
        report_step(30.0, "Stage 3: Attention U-Net cloud & shadow segmentation...")
        cloud_res = self.cloud_model.predict(cloudy_norm)
        packet.cloud_detection = cloud_res

        # --- Stage 4: Change Detection ---
        report_step(40.0, "Stage 4: Temporal change detection & SAR coherence mapping...")
        change_res = self.change_model.predict(
            curr_optical=cloudy_norm,
            hist_optical=hist_norm,
            sar_image=sar_norm,
            cloud_mask=cloud_res["cloud_mask"]
        )
        packet.change_detection = change_res

        # --- Stage 5: Adaptive Decision Engine ---
        report_step(50.0, "Stage 5: Adaptive Decision Engine strategy routing...")
        decision = self.decision_engine.evaluate_strategy(
            cloud_pct=cloud_res["cloud_percentage"],
            change_score=change_res["change_score"],
            changed_area_pct=change_res["changed_area"],
            has_sar=(sar_path is not None and os.path.exists(sar_path)),
            has_historical=(historical_path is not None and os.path.exists(historical_path))
        )
        if strategy_override in ["historical", "sar", "adaptive"]:
            decision["strategy"] = strategy_override
            cand_map = {"historical": "C1", "sar": "C2", "adaptive": "C3"}
            decision["recommended_candidate"] = cand_map.get(strategy_override, "C3")
        packet.decision = decision

        # --- Stage 6 & 7: Cross-Attention Fusion & MRR Reconstruction ---
        report_step(65.0, "Stage 6-7: Multi-Hypothesis Reconstruction (MRR Candidates C1, C2, C3)...")
        # Total occlusion mask includes both clouds and ground shadows
        total_occlusion_mask = ((cloud_res["cloud_mask"] > 0) | (cloud_res.get("shadow_mask", 0) > 0)).astype(np.uint8)

        candidates = self.mrr_model.reconstruct_all(
            cloudy_optical=cloudy_norm,
            hist_optical=hist_norm,
            sar_image=sar_norm,
            cloud_mask=total_occlusion_mask,
            cloud_prob=cloud_res["cloud_probability"],
            change_prob=change_res["change_probability"]
        )
        packet.candidates = candidates

        # --- Stage 8: Quality Assessment Network (QAN) ---
        report_step(75.0, "Stage 8: QAN evaluation (SSIM, PSNR, SAM, ERGAS)...")
        best_cand, all_metrics = self.qan.rank_candidates(
            candidates=candidates,
            reference=ref_norm,
            cloud_mask=total_occlusion_mask
        )
        # Select best candidate from QAN ranking (or strategy override)
        selected_cand = strategy_override if strategy_override in candidates else best_cand
        packet.best_candidate = selected_cand
        packet.reconstructed_image = candidates[selected_cand]
        packet.quality_metrics = all_metrics[selected_cand]
        packet.all_candidate_metrics = all_metrics

        # --- Stage 9: Confidence Estimation ---
        report_step(85.0, "Stage 9: Calibrated pixel-level confidence mapping...")
        conf_report = self.confidence_estimator.estimate(
            cloud_mask=cloud_res["cloud_mask"],
            cloud_prob=cloud_res["cloud_probability"],
            change_prob=change_res["change_probability"],
            candidates=candidates
        )
        packet.confidence_report = conf_report

        # --- Stage 10: Analysis-Ready Products (NDVI, Land Cover & Sub-Cloud Feature Decoding) ---
        report_step(92.0, "Stage 10: NDVI vegetation analytics & Sub-Cloud Ground Feature decoding...")
        packet.ndvi_report = self.ndvi_analyzer.analyze(
            reconstructed_image=packet.reconstructed_image,
            reference_image=ref_norm
        )
        packet.landcover_report = self.landcover_classifier.classify(packet.reconstructed_image)
        packet.sub_cloud_report = self.sub_cloud_predictor.predict_sub_cloud_features(
            cloud_mask=cloud_res["cloud_mask"],
            reconstructed_image=packet.reconstructed_image,
            sar_image=sar_norm,
            pixel_resolution_m=meta.resolution if meta else 10.0
        )

        # Export Analysis-Ready GeoTIFFs
        cloud_free_tif = os.path.join(self.output_dir, f"{image_id}_cloud_free.tif")
        conf_tif = os.path.join(self.output_dir, f"{image_id}_confidence.tif")
        change_tif = os.path.join(self.output_dir, f"{image_id}_change_map.tif")

        self.loader.save_raster(cloud_free_tif, packet.reconstructed_image, reference_meta=meta)
        self.loader.save_raster(conf_tif, conf_report.confidence_map, reference_meta=meta)
        self.loader.save_raster(change_tif, change_res["change_probability"], reference_meta=meta)

        packet.cloud_free_geotiff_path = cloud_free_tif
        packet.confidence_geotiff_path = conf_tif
        packet.change_geotiff_path = change_tif

        # --- Stage 11: PDF Quality Report & Delivery ---
        report_step(98.0, "Stage 11: Compiling PDF Quality Inspection Report & Metadata...")
        report_pdf = os.path.join(self.reports_dir, f"{image_id}_quality_report.pdf")
        metadata_json = os.path.join(self.output_dir, f"{image_id}_metadata.json")

        # Build preview RGBs for report
        cloudy_rgb = ImagePreprocessor.extract_rgb_preview(cloudy_norm)
        mask_rgb = np.stack([cloud_res["cloud_mask"] * 255, cloud_res["shadow_mask"] * 128, np.zeros((H, W), dtype=np.uint8)], axis=-1)
        rec_rgb = ImagePreprocessor.extract_rgb_preview(packet.reconstructed_image)
        conf_rgb = conf_report.colored_heatmap

        packet_report_data = {
            "image_id": image_id,
            "metadata": meta.to_dict() if meta else {},
            "metrics": packet.quality_metrics.to_dict(),
            "decision": decision,
            "cloud_percentage": cloud_res["cloud_percentage"],
            "confidence_stats": conf_report.to_dict(),
            "ndvi": packet.ndvi_report.to_dict(),
            "landcover": packet.landcover_report.to_dict(),
            "best_candidate": packet.best_candidate,
            "cloudy_rgb": cloudy_rgb,
            "cloud_mask_rgb": mask_rgb,
            "reconstructed_rgb": rec_rgb,
            "confidence_rgb": conf_rgb
        }

        self.pdf_generator.generate_report(report_pdf, packet_report_data)
        packet.report_pdf_path = report_pdf

        with open(metadata_json, "w") as f:
            json.dump(packet.to_summary_dict(), f, indent=2)
        packet.metadata_json_path = metadata_json

        packet.elapsed_seconds = time.time() - start_time
        packet.status = "Completed"
        report_step(100.0, f"Reconstruction pipeline completed successfully in {packet.elapsed_seconds:.2f}s!")

        return packet
