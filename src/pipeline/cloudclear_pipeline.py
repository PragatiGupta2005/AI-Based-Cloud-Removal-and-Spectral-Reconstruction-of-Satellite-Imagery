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
        progress_callback: Optional[Callable[[float, str], None]] = None,
        agent_mode: str = "adaptive",
    ) -> PredictionPacket:
        """Executes all 11 stages via the multi-agent orchestrator.

        agent_mode="sequential" reproduces the legacy fixed order;
        "adaptive" enables autonomous planner routing (audit Sec.14/15).
        """
        import os as _os
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

        from ..agents import (
            DataRetrievalAgent, PreprocessingAgent, CloudDetectionAgent,
            ChangeDetectionAgent, DecisionAgent, FusionReconstructionAgent,
            QualityAgent, ConfidenceAgent, AnalyticsAgent, DeliveryAgent,
            AgentOrchestrator,
        )
        state: Dict[str, Any] = {
            "loader": self.loader, "preprocessor": self.preprocessor,
            "cloud_model": self.cloud_model, "change_model": self.change_model,
            "mrr_model": self.mrr_model, "decision_engine": self.decision_engine,
            "qan": self.qan, "confidence_estimator": self.confidence_estimator,
            "ndvi_analyzer": self.ndvi_analyzer,
            "landcover_classifier": self.landcover_classifier,
            "sub_cloud_predictor": self.sub_cloud_predictor,
            "pdf_gen": self.pdf_generator,
            "cloudy_path": cloudy_path, "historical_path": historical_path,
            "sar_path": sar_path, "clear_reference_path": clear_reference_path,
            "image_id": image_id, "output_dir": self.output_dir,
            "reports_dir": self.reports_dir,
            "strategy_override": strategy_override,
            "has_sar": bool(sar_path and _os.path.exists(sar_path)),
            "has_hist": bool(historical_path and _os.path.exists(historical_path)),
        }
        orch = AgentOrchestrator([
            DataRetrievalAgent(), PreprocessingAgent(), CloudDetectionAgent(),
            ChangeDetectionAgent(), DecisionAgent(), FusionReconstructionAgent(),
            QualityAgent(), ConfidenceAgent(), AnalyticsAgent(), DeliveryAgent(),
        ])
        report_step(5.0, f"Agent orchestrator start (mode={agent_mode})...")
        if agent_mode == "sequential":
            orch.run_sequential(state, progress=report_step)
        else:
            orch.run_adaptive(state, progress=report_step)

        # Map agent state back onto the legacy PredictionPacket (API/tests stable)
        packet.metadata = state["meta"]
        packet.cloudy_raw = state["cloudy_raw"]
        packet.hist_raw = state["hist_raw"]
        packet.sar_raw = state["sar_raw"]
        packet.ref_raw = state["ref_raw"]
        packet.cloud_detection = state["cloud_res"]
        packet.change_detection = state["change_res"]
        packet.decision = state["decision"]
        packet.decision["agent_log"] = [
            {"agent": m.agent, "action": m.action, "detail": m.detail} for m in orch.log
        ]
        packet.decision["agent_mode"] = agent_mode
        packet.candidates = state["candidates"]
        packet.best_candidate = state["best"]
        packet.reconstructed_image = state["recon"]
        packet.quality_metrics = state["quality"]
        packet.all_candidate_metrics = state["all_metrics"]
        packet.confidence_report = state["conf"]
        packet.ndvi_report = state["ndvi"]
        packet.landcover_report = state["lc"]
        packet.sub_cloud_report = state["sub"]
        packet.cloud_free_geotiff_path = state["cf"]
        packet.confidence_geotiff_path = state["cf2"]
        packet.change_geotiff_path = state["ch"]
        packet.report_pdf_path = state["rep"]
        packet.metadata_json_path = state["mjson"]
        # Provenance + method flags (audit honesty)
        try:
            from ..models.registry import refresh_registry
            prov = {
                "qan_method": getattr(self.qan, "method", "metric"),
                "landcover_method": getattr(self.landcover_classifier, "method", "rule-based"),
                "cloud_weights": getattr(self.cloud_model, "weights_path", None),
                "change_weights": getattr(self.change_model, "weights_path", None),
                "model_registry": refresh_registry(),
            }
        except Exception:
            prov = {}
        with open(state["mjson"], "w") as f:
            d = packet.to_summary_dict()
            d["provenance"] = prov
            json.dump(d, f, indent=2)

        packet.elapsed_seconds = time.time() - start_time
        packet.status = "Completed"
        report_step(100.0, f"Reconstruction pipeline completed successfully in {packet.elapsed_seconds:.2f}s!")
        return packet
