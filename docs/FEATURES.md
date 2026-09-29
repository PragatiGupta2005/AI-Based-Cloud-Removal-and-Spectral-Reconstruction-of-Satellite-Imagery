# 🌟 CloudClear AI — Features & Capabilities Specification

CloudClear AI is a multi-modal, agentic geospatial AI platform designed for removing cloud obstructions and reconstructing true surface reflectance in optical satellite imagery.

---

## 1. Core Capabilities Overview

| Category | Feature | Description |
| :--- | :--- | :--- |
| **Geospatial Ingestion** | Multi-Sensor Support | Ingests LISS-IV, Sentinel-2 (B2, B3, B4, B8), and Sentinel-1 SAR imagery |
| **Pre-Processing** | Radiometric & Spatial Alignment | Automatic CRS verification (EPSG:4326/UTM), band extraction, and sliding window patch extraction |
| **Segmentation** | Attention U-Net Detection | Pixel-level binary cloud mask, shadow mask, and continuous cloud probability mapping |
| **Change Detection** | Multi-Temporal Coherence | Compares temporal optical and SAR backscatter to identify genuine land modifications |
| **Fusion Engine** | Cross-Attention Multi-Modal Fusion | Cross-attention transformer fusing optical spectral bands, temporal features, and SAR radar texture |
| **Reconstruction** | Multi-Hypothesis (MRR) | Generates Candidate 1 (Historical), Candidate 2 (SAR), and Candidate 3 (Adaptive Fusion) |
| **Quality Assessment** | QAN Ranking | Quantitative evaluation using SSIM, PSNR, MAE, RMSE, SAM, and ERGAS with automated ranking |
| **Reliability** | Calibrated Confidence Heatmap | Generates 3-tier confidence score (High, Medium, Low) for every reconstructed pixel |
| **Vegetation Analytics**| NDVI Synthesis & Tracking | Pre- and post-reconstruction NDVI delta maps and crop health metrics |
| **Land Cover** | 5-Class Categorization | Automated distribution for Vegetation, Agriculture, Water, Urban, and Bare Soil |
| **Reporting & Export** | PDF & Analysis-Ready GeoTIFF | Automated PDF quality inspection report and full GIS-ready GeoTIFF export |

---

## 2. Multi-Hypothesis Reconstruction (MRR) Strategy

```text
┌────────────────────────────────────────────────────────┐
│               Input Cloudy Satellite Scene             │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      ┌───────────────┐           ┌───────────────┐
      │ Historical    │           │ Sentinel-1    │
      │ Optical Pair  │           │ SAR Radar     │
      └───────┬───────┘           └───────┬───────┘
              │                           │
  ┌───────────┼───────────────────────────┼───────────┐
  ▼           ▼                           ▼           ▼
┌────────┐  ┌─────────────────────────────────┐  ┌────────┐
│ Candidate 1 │  Candidate 3: Adaptive Fusion   │  │ Candidate 2 │
│ Historical  │  (Cross-Attention Multi-Modal)  │  │ SAR Radar   │
│ Dominant    │                                 │  │ Dominant    │
└────┬───┘  └────────────────┬────────────────┘  └───┬────┘
     │                       │                       │
     └───────────────────────┼───────────────────────┘
                             ▼
              ┌───────────────────────────────┐
              │  Quality Assessment Network   │
              │  (SSIM, PSNR, SAM, ERGAS)     │
              └──────────────┬────────────────┘
                             ▼
              ┌───────────────────────────────┐
              │   Optimal Cloud-Free GeoTIFF  │
              └───────────────────────────────┘
```

---

## 3. Detailed Feature Breakdown

### 🛰️ 1. Multi-Sensor Data Retrieval & Verification
- **Supported Sensors**: LISS-IV (5.8m resolution), Sentinel-2 MSI (10m resolution), Sentinel-1 C-Band SAR (VV/VH polarization).
- **Validation**: Strict validation of spatial extent, Coordinate Reference System (CRS), channel count, and radiometric data ranges.

### ☁️ 2. Cloud & Shadow Detection (Attention U-Net)
- Separates thin cirrus clouds, thick cumulus clouds, and corresponding ground cloud shadows.
- Generates continuous confidence probability scores $[0.0, 1.0]$ in addition to binary masks.

### 🔄 3. Temporal Change Detection Engine
- Detects whether land cover changed between historical acquisition and current scene.
- Categorizes regions into **Stable (Green)**, **Moderate (Yellow)**, and **Changed (Red)** to prevent temporal hallucination.

### ⚡ 4. Cross-Attention Multi-Modal Fusion
- Uses attention mechanisms where SAR backscatter provides cloud-penetrating structural texture and historical imagery provides spectral reference.

### 📊 5. Quality Assessment Network (QAN)
- **PSNR**: Peak Signal-to-Noise Ratio (Target: >30 dB).
- **SSIM**: Structural Similarity Index Measure (Target: >0.90).
- **MAE & RMSE**: Mean Absolute and Root Mean Square pixel errors (Target: <0.03).
- **SAM**: Spectral Angle Mapper in degrees (Target: <5.0°).
- **ERGAS**: Relative dimensionless global error in synthesis (Target: <2.0).

### 🎯 6. Pixel-Level Confidence Heatmap
- Pixel-by-pixel reliability estimation based on cloud thickness, temporal variance, and multi-modal alignment.
- Color-coded visualization (High: Yellow/Green, Medium: Orange, Low: Purple).

### 🌿 7. NDVI & Land Cover Analytics
- Normalized Difference Vegetation Index: $\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$.
- Difference mapping showing delta between cloudy baseline, reconstructed scene, and reference.
- Automated Land Cover classification across 5 standard remote sensing classes.

### 📄 8. Automated PDF Quality Inspection Reports
- High-resolution ReportLab document generation containing metadata summaries, quality metric tables, comparative RGB visualizations, NDVI histograms, and GIS certification.
