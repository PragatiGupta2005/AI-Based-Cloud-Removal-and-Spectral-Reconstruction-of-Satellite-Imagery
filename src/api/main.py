"""
FastAPI REST API Backend for CloudClear AI.
Implements the full specification from docs/API_SPEC.md.
"""

import os
import uuid
import json
import shutil
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, status, Header, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.preprocessing.data_loader import GeoTIFFLoader, validate_geotiff
from src.pipeline.cloudclear_pipeline import CloudClearPipeline, PredictionPacket
from src.api.security import require_auth, create_token, auth_enabled
from src.hitl.feedback_store import FeedbackStore
from src.models.registry import refresh_registry

app = FastAPI(
    title="CloudClear AI REST API",
    description="Multi-Modal Geospatial AI Platform for Cloud Removal & Spectral Reconstruction",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration (restrict via CORS_ALLOW_ORIGINS env in production; audit Sec.18)
import os as _os
_cors_origins = [o.strip() for o in _os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline instance & cache
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "cloudy")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

pipeline = CloudClearPipeline(output_dir=OUTPUT_DIR, reports_dir=REPORTS_DIR)
predictions_cache: Dict[str, PredictionPacket] = {}


# --- Request & Response Schemas ---

class PredictRequest(BaseModel):
    image_id: str
    strategy: Optional[str] = Field(None, description="historical, sar, or adaptive")
    agent_mode: Optional[str] = Field("adaptive", description="sequential or adaptive")


class FeedbackRequest(BaseModel):
    image_id: str
    kind: str = Field(description="cloud_mask | strategy | quality")
    payload: Dict[str, Any] = Field(default_factory=dict)


class FetchRealRequest(BaseModel):
    bbox: List[float] = Field(description="[min_lon, min_lat, max_lon, max_lat]")
    date_range: str = "2024-01-01/2024-06-30"
    region: str = "STAC AOI"
    download_s2: bool = True


class CloudMaskRequest(BaseModel):
    image_id: str
    threshold: Optional[float] = 0.5


class ChangeMapRequest(BaseModel):
    image_id: str


class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# --- Endpoints ---

@app.get("/api/v1/health", tags=["System"])
def health_check():
    import tensorflow as tf
    return {
        "status": "online",
        "service": "CloudClear AI",
        "version": "1.1.0",
        "auth_enabled": auth_enabled(),
        "tf_version": tf.__version__,
        "model_registry": refresh_registry(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/v1/models", tags=["System"])
def list_models():
    """Pretrained-model registry with honest validation status (audit Sec.13)."""
    return {"success": True, "registry": refresh_registry()}


@app.post("/api/v1/feedback", tags=["HITL"])
def submit_feedback(req: FeedbackRequest, user: str = Depends(require_auth)):
    """Stores a human correction for later retraining (audit Sec.16)."""
    if req.kind not in ("cloud_mask", "strategy", "quality"):
        raise HTTPException(status_code=422, detail="kind must be cloud_mask|strategy|quality")
    store = FeedbackStore()
    rec = store.add({"image_id": req.image_id, "kind": req.kind,
                     "payload": req.payload, "user": user})
    return {"success": True, "message": "Feedback stored for retraining",
            "record": rec, "total_feedback": store.count()}


@app.get("/api/v1/feedback", tags=["HITL"])
def list_feedback(limit: int = 100):
    return {"success": True, "feedback": FeedbackStore().list(limit=limit)}


@app.post("/api/v1/fetch-real", tags=["Datasets"])
def fetch_real_s2(req: FetchRealRequest, background_tasks: BackgroundTasks,
                  user: str = Depends(require_auth)):
    """Fetches real Sentinel-2 via STAC (audit Sec.2/3/5). Synchronous small download."""
    if len(req.bbox) != 4:
        raise HTTPException(status_code=422, detail="bbox must have 4 floats")
    from fetch_satellite_data import SatelliteDataFetcher
    fetcher = SatelliteDataFetcher(data_dir=os.path.join(BASE_DIR, "data"))
    try:
        res = fetcher.fetch_sentinel2_as_geotiff(
            bbox=tuple(req.bbox), date_range=req.date_range, region_name=req.region)
        return {"success": True, "message": "Real Sentinel-2 downloaded",
                "data": {k: v for k, v in res.items() if k != "optical"}}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Real S-2 fetch failed: {e}")


@app.post("/api/v1/admin/retrain", tags=["Admin"])
def admin_retrain(background_tasks: BackgroundTasks, user: str = Depends(require_auth),
                  epochs: int = 3):
    """Queues lightweight retraining including HITL feedback (audit Sec.16)."""
    def _job():
        from src.models.trainer import SatellitePatchDataset, ModelTrainer
        ds = SatellitePatchDataset(data_dir=os.path.join(BASE_DIR, "data"),
                                   patch_size=128, stride=128, max_patches_per_scene=2)
        patches = ds.extract_patches()
        tr = ModelTrainer(save_dir=os.path.join(BASE_DIR, "models", "saved_models"),
                          patch_size=128)
        if len(patches.get("cloudy", [])):
            tr.train_cloud_detector(patches, epochs=epochs)
    background_tasks.add_task(_job)
    return {"success": True, "message": f"Retrain queued (epochs={epochs}) incl. feedback",
            "feedback_count": FeedbackStore().count()}


@app.post("/api/v1/admin/token", tags=["Admin"])
def admin_token(subject: str = "user"):
    if auth_enabled():
        raise HTTPException(status_code=403, detail="Token minting disabled when AUTH_ENABLED=true; use SSO")
    return {"success": True, "token": create_token(subject)}


@app.get("/api/v1/samples", tags=["Datasets"])
def list_samples():
    """Lists available pre-generated satellite sample scenes."""
    manifest_path = os.path.join(BASE_DIR, "data", "samples", "dataset_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return {"success": True, "samples": manifest}
    return {"success": False, "message": "Manifest not found. Run generate_sample_data.py first."}


@app.post("/api/v1/upload", tags=["Pipeline"])
async def upload_geotiff(
    image: UploadFile = File(...),
    region: str = Form("West Bengal"),
    sensor: str = Form("Sentinel-2"),
    user: str = Depends(require_auth),
):
    """
    Uploads a multi-band GeoTIFF satellite scene. Max 500 MB (audit Sec.18).
    """
    if not (image.filename.endswith(".tif") or image.filename.endswith(".tiff")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid format. Only .tif or .tiff GeoTIFF files are supported."
        )

    image_id = f"IMG_{uuid.uuid4().hex[:8]}"
    save_path = os.path.join(UPLOAD_DIR, f"{image_id}_{image.filename}")

    # Stream with 500 MB cap (production hardening)
    max_bytes = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
    written = 0
    with open(save_path, "wb") as buffer:
        while True:
            chunk = await image.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                buffer.close()
                if os.path.exists(save_path):
                    os.remove(save_path)
                raise HTTPException(status_code=413, detail="File exceeds MAX_UPLOAD_MB")
            buffer.write(chunk)

    is_valid, msg, meta = validate_geotiff(save_path)
    if not is_valid or meta is None:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"GeoTIFF Validation Failed: {msg}"
        )

    meta.region = region
    meta.sensor = sensor

    return {
        "success": True,
        "image_id": image_id,
        "filename": image.filename,
        "status": "Uploaded",
        "metadata": meta.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/api/v1/predict", tags=["Pipeline"])
def run_full_prediction(req: PredictRequest, user: str = Depends(require_auth)):
    """
    Runs the complete 11-stage CloudClear AI reconstruction pipeline.
    """
    img_id = req.image_id

    # Locate cloudy file
    cloudy_file = None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(img_id) or img_id in f:
            cloudy_file = os.path.join(UPLOAD_DIR, f)
            break

    if not cloudy_file:
        # Check sample directories
        sample_path = os.path.join(BASE_DIR, "data", "cloudy", f"{img_id}_cloudy.tif")
        if os.path.exists(sample_path):
            cloudy_file = sample_path
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Satellite image with ID '{img_id}' not found in upload repository."
            )

    # Locate corresponding historical, sar, and clear files
    hist_file = os.path.join(BASE_DIR, "data", "historical", f"{img_id}_historical.tif")
    sar_file = os.path.join(BASE_DIR, "data", "sar", f"{img_id}_sar.tif")
    clear_file = os.path.join(BASE_DIR, "data", "clear", f"{img_id}_clear.tif")

    packet = pipeline.run(
        cloudy_path=cloudy_file,
        historical_path=hist_file if os.path.exists(hist_file) else None,
        sar_path=sar_file if os.path.exists(sar_file) else None,
        clear_reference_path=clear_file if os.path.exists(clear_file) else None,
        image_id=img_id,
        strategy_override=req.strategy,
        agent_mode=(req.agent_mode or "adaptive"),
    )

    predictions_cache[img_id] = packet

    return {
        "success": True,
        "message": "Prediction completed successfully",
        "data": packet.to_summary_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/api/v1/cloud-mask", tags=["AI Modules"])
def get_cloud_mask(req: CloudMaskRequest):
    """
    Generates cloud and shadow masks for an uploaded scene.
    """
    if req.image_id in predictions_cache:
        packet = predictions_cache[req.image_id]
        cd = packet.cloud_detection
        return {
            "success": True,
            "image_id": req.image_id,
            "cloud_percentage": cd["cloud_percentage"],
            "shadow_percentage": cd["shadow_percentage"],
            "clear_percentage": cd["clear_percentage"]
        }

    # Run on demand
    pred = run_full_prediction(PredictRequest(image_id=req.image_id))
    packet = predictions_cache[req.image_id]
    return {
        "success": True,
        "image_id": req.image_id,
        "cloud_percentage": packet.cloud_detection["cloud_percentage"],
        "shadow_percentage": packet.cloud_detection["shadow_percentage"],
        "clear_percentage": packet.cloud_detection["clear_percentage"]
    }


@app.post("/api/v1/change-map", tags=["AI Modules"])
def get_change_map(req: ChangeMapRequest):
    """
    Returns temporal change detection analysis.
    """
    if req.image_id not in predictions_cache:
        run_full_prediction(PredictRequest(image_id=req.image_id))

    packet = predictions_cache[req.image_id]
    ch = packet.change_detection
    return {
        "success": True,
        "image_id": req.image_id,
        "stable_area": ch["stable_area"],
        "moderate_area": ch["moderate_area"],
        "changed_area": ch["changed_area"],
        "change_score": ch["change_score"]
    }


@app.get("/api/v1/quality/{prediction_id}", tags=["Analytics"])
def get_quality_metrics(prediction_id: str):
    """
    Returns PSNR, SSIM, MAE, RMSE, SAM, ERGAS, and composite quality score.
    """
    if prediction_id not in predictions_cache:
        run_full_prediction(PredictRequest(image_id=prediction_id))

    packet = predictions_cache[prediction_id]
    return {
        "success": True,
        "image_id": prediction_id,
        "best_candidate": packet.best_candidate,
        "metrics": packet.quality_metrics.to_dict()
    }


@app.get("/api/v1/ndvi/{prediction_id}", tags=["Analytics"])
def get_ndvi_analytics(prediction_id: str):
    """
    Returns NDVI vegetation metrics.
    """
    if prediction_id not in predictions_cache:
        run_full_prediction(PredictRequest(image_id=prediction_id))

    packet = predictions_cache[prediction_id]
    return {
        "success": True,
        "image_id": prediction_id,
        "ndvi": packet.ndvi_report.to_dict()
    }


@app.get("/api/v1/landcover/{prediction_id}", tags=["Analytics"])
def get_landcover_distribution(prediction_id: str):
    """
    Returns Land Cover distribution across 5 classes.
    """
    if prediction_id not in predictions_cache:
        run_full_prediction(PredictRequest(image_id=prediction_id))

    packet = predictions_cache[prediction_id]
    return {
        "success": True,
        "image_id": prediction_id,
        "landcover": packet.landcover_report.to_dict()
    }


@app.get("/api/v1/download/{prediction_id}", tags=["Download Center"])
def download_geotiff(prediction_id: str):
    """Downloads the reconstructed cloud-free GeoTIFF."""
    if prediction_id not in predictions_cache:
        run_full_prediction(PredictRequest(image_id=prediction_id))

    packet = predictions_cache[prediction_id]
    file_path = packet.cloud_free_geotiff_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="GeoTIFF file not found.")

    return FileResponse(
        file_path,
        media_type="image/tiff",
        filename=os.path.basename(file_path)
    )


@app.get("/api/v1/report/{prediction_id}", tags=["Download Center"])
def download_pdf_report(prediction_id: str):
    """
    Downloads the PDF Quality Inspection Report.
    """
    if prediction_id not in predictions_cache:
        run_full_prediction(PredictRequest(image_id=prediction_id))

    packet = predictions_cache[prediction_id]
    report_path = packet.report_pdf_path
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="PDF report not found.")

    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=os.path.basename(report_path)
    )


@app.get("/api/v1/validate", tags=["Analytics"])
def run_validation(max_scenes: int = 3, user: str = Depends(require_auth)):
    """Runs benchmark vs historical-copy baseline (audit Sec.17)."""
    from src.validation.benchmark import run_benchmark
    report = run_benchmark(
        data_dir=os.path.join(BASE_DIR, "data"),
        output_dir=os.path.join(BASE_DIR, "outputs"),
        reports_dir=os.path.join(BASE_DIR, "reports"),
        max_scenes=max(1, min(max_scenes, 10)),
    )
    return {"success": True, "report": report}
