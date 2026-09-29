# ☁️ CloudClear AI
### AI-Powered Cloud Removal & Reconstruction of Satellite Imagery

> An ISRO-inspired Multi-Modal Geospatial AI system for reconstructing cloud-free satellite imagery using LISS-IV, Sentinel-2, Sentinel-1 SAR, Cross-Attention Fusion, and Multi-Hypothesis Reconstruction.

---

## Project Overview

CloudClear AI is a deep learning based Earth Observation platform that automatically removes cloud cover from optical satellite imagery while preserving spectral and spatial information.

The system combines:

- LISS-IV / Sentinel-2 Optical Images
- Sentinel-1 SAR Images
- Historical Satellite Archive
- AI-based Cloud Detection
- Change Detection
- Cross-Attention Feature Fusion
- Multi-Hypothesis Reconstruction (MRR)
- Quality Assessment Network (QAN)
- Confidence Estimation

The final output is an **Analysis-Ready GeoTIFF** suitable for GIS, agriculture, disaster management, urban planning, and environmental monitoring.

---

# Dashboard Preview

> Main Dashboard (CloudClear AI)

![Dashboard](docs/assets/dashboard.png)

The dashboard provides:

- Region & AOI selection
- Sensor selection
- Cloud mask generation
- Cloud-free reconstruction
- Confidence heatmap
- NDVI comparison
- Land-cover analysis
- Quality metrics
- Download center

---

# Methodology

![Architecture](docs/assets/methodology.png)

The project follows an **11-stage intelligent reconstruction pipeline** inspired by ISRO remote sensing workflows.

1. Data Retrieval & Selection
2. Preprocessing
3. Cloud & Shadow Detection
4. Change Detection
5. Adaptive Decision Engine
6. Cross-Attention Fusion
7. Multi-Hypothesis Reconstruction
8. Quality Assessment Network
9. Confidence Estimation
10. Analysis-Ready Output
11. Delivery & Access

---

# Key Innovations

## Multi-Modal Intelligence

Uses both optical and SAR imagery.

Why?

- Optical provides color & spectral information.
- SAR penetrates clouds.
- Historical imagery provides temporal context.

---

## Change-Aware Reconstruction

Instead of copying historical pixels blindly, the AI first checks whether the land has changed.

This prevents temporal hallucination.

---

## Cross-Attention Fusion

Features from:

- Current Image
- Historical Image
- Sentinel-1 SAR

are fused using attention mechanisms before reconstruction.

---

## Multi-Hypothesis Reconstruction

The model generates **three candidate images**.

| Candidate | Description |
|-----------|-------------|
| C1 | Historical Dominant |
| C2 | SAR Dominant |
| C3 | Adaptive Fusion |

A Quality Assessment Network selects the best result automatically.

---

## Confidence-Aware Output

Every reconstructed pixel receives a confidence score.

Confidence levels:

- High
- Medium
- Low

This makes the system suitable for scientific analysis.

---

# Applications

- Precision Agriculture
- Flood Monitoring
- Forest Monitoring
- Urban Expansion Analysis
- Water Resource Management
- Disaster Response
- Defence Surveillance
- Climate Observation

---

# Technology Stack

## AI & ML

- TensorFlow
- Keras
- U-Net
- Cross-Attention
- CNN
- NumPy

## Geospatial

- Rasterio
- GDAL
- GeoPandas
- OpenCV

## Backend

- FastAPI
- Uvicorn
- JWT Authentication

## Frontend

- Streamlit
- Plotly
- Matplotlib

---

# Folder Structure

```text
CloudClearAI/

│── README.md
│── requirements.txt
│── app.py
│
├── docs/
│   ├── PROJECT_SPEC.md
│   ├── ARCHITECTURE.md
│   ├── FEATURES.md
│   ├── UI_SPEC.md
│   ├── API_SPEC.md
│   ├── AGENTS.md
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   └── ROADMAP.md
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── fusion/
│   ├── qan/
│   ├── confidence/
│   └── api/
│
├── data/
│   ├── cloudy/
│   ├── clear/
│   ├── historical/
│   └── sar/
│
├── outputs/
│
└── reports/
```

---

# Complete Workflow

```text
Cloudy Image
      │
      ▼
Historical Retrieval
      │
      ▼
Preprocessing
      │
      ▼
Cloud Detection
      │
      ▼
Change Detection
      │
      ▼
Decision Engine
      │
      ▼
Cross Attention Fusion
      │
      ▼
MRR Reconstruction
      │
      ▼
QAN
      │
      ▼
Confidence Map
      │
      ▼
Cloud-Free GeoTIFF
```

---

# Quality Metrics

The dashboard computes multiple reconstruction metrics.

| Metric | Purpose |
|---------|----------|
| PSNR | Peak Signal Ratio |
| SSIM | Structural Similarity |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Square Error |
| SAM | Spectral Angle Mapper |
| ERGAS | Global Spectral Error |

Example Output

| Metric | Value |
|---------|------|
| PSNR | 32.45 dB |
| SSIM | 0.912 |
| MAE | 0.018 |
| RMSE | 0.024 |
| SAM | 4.21° |
| ERGAS | 1.87 |

---

# NDVI Analysis

The reconstructed image preserves vegetation characteristics.

Outputs include:

- Reference NDVI
- Reconstructed NDVI
- Difference Map

Suitable for crop monitoring and vegetation health assessment.

---

# Land Cover Distribution

Automatically estimates land cover classes.

Supported classes:

- Vegetation
- Agriculture
- Water
- Urban
- Bare Land
- Others

Interactive donut charts are available in the dashboard.

---

# System Features

- GeoTIFF Upload
- Automatic Metadata Extraction
- Cloud Detection
- Shadow Detection
- Historical Image Matching
- Sentinel-1 SAR Integration
- Change Detection
- Adaptive Decision Engine
- Cross-Attention Fusion
- Multi-Hypothesis Reconstruction
- Quality Assessment
- Confidence Estimation
- NDVI Visualization
- Land Cover Analysis
- PDF Report Generation
- GeoTIFF Export

---

# Documentation

| File | Description |
|------|-------------|
| PROJECT_SPEC.md | Complete SRS |
| ARCHITECTURE.md | 11-module system design |
| FEATURES.md | Functional features |
| UI_SPEC.md | Frontend specification |
| API_SPEC.md | REST API documentation |
| AGENTS.md | AI agent architecture |
| INSTALLATION.md | Setup guide |
| USAGE.md | User manual |
| ROADMAP.md | Future development |

---

# Installation

```bash
git clone https://github.com/yourusername/CloudClearAI.git

cd CloudClearAI

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

---

# Future Scope

- Vision Transformer backbone
- Multi-temporal reconstruction
- LangGraph autonomous agents
- Foundation geospatial models
- Kubernetes deployment
- Real-time satellite processing

---

# Authors

**CloudClear AI**

Final Year B.Tech Major Project

Department of Computer Engineering

2026–2027

---

# License

This project is developed for academic and research purposes.

© 2026 CloudClear AI. All Rights Reserved.