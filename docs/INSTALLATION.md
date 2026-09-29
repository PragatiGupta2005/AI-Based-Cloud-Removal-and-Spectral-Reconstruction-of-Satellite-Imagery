# 🛠️ Installation & Setup Guide — CloudClear AI

Comprehensive installation, setup, and deployment instructions for CloudClear AI on Windows, Linux, and macOS.

---

## 1. Prerequisites

- **Operating System**: Windows 10/11, Ubuntu 22.04 LTS, or macOS (Apple Silicon / Intel).
- **Python**: Python 3.11 – 3.13.
- **Hardware (Recommended)**: 16 GB RAM, NVIDIA GPU with 8 GB+ VRAM (CUDA 12.x compatible). CPU execution is also supported via oneDNN acceleration.

---

## 2. Step-by-Step Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/CloudClearAI.git
cd CloudClearAI
```

### Step 2: Create and Activate Virtual Environment
**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Generate Demonstration Satellite Data
```bash
python generate_sample_data.py
```
This generates realistic 4-band GeoTIFF test sets (Optical B2, B3, B4, B8 + Sentinel-1 SAR + Historical Reference + Ground Truth) for West Bengal (Nadia), Maharashtra (Pune), and Kerala (Alappuzha).

---

## 3. Running the System

### Option A: Launch Streamlit Geospatial Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your default browser at:
`http://localhost:8501`

### Option B: Launch FastAPI REST Backend
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 4. Running Automated Tests
```bash
python tests/test_pipeline.py
```

---

## 5. Verification Checklist

- [x] GeoTIFF loading and 4-band radiometric validation
- [x] Attention U-Net Cloud & Shadow Detection
- [x] Siamese Temporal Change Detection
- [x] Multi-Hypothesis Reconstruction (MRR Candidates C1, C2, C3)
- [x] Quality Assessment Network (QAN metrics: PSNR, SSIM, SAM, ERGAS, MAE, RMSE)
- [x] Pixel-level Confidence Heatmap generation
- [x] NDVI & 5-Class Land Cover distribution analysis
- [x] Analysis-Ready GeoTIFF raster export
- [x] Automated PDF Quality Inspection Report generation
