Table of Contents

API Overview

System Architecture

API Standards

Authentication

Base URL

Request & Response Format

Error Handling

Data Models

Endpoints Overview

Upload API

Cloud Detection API

Change Detection API

Reconstruction API

Quality Assessment API

NDVI & Land Cover API

Reports API

Download API

Admin APIs

Security Specification

API Workflow

1. API Overview

CloudClear AI provides a RESTful backend API that powers the Streamlit dashboard and AI reconstruction pipeline.

The API allows users to:

Upload GeoTIFF satellite images

Detect clouds and shadows

Perform temporal change detection

Generate cloud-free satellite imagery

Calculate NDVI & land cover statistics

Produce quality reports

Download analysis-ready GeoTIFF outputs

The backend is implemented using FastAPI and communicates with the AI engine through TensorFlow inference services.

2. System Architecture
Components

Layer

	

Technology




Frontend

	

Streamlit




Backend

	

FastAPI




AI Service

	

TensorFlow




Storage

	

GeoTIFF




Reports

	

PDF Generator

3. API Standards
Protocol

REST

HTTPS

JSON

Multipart Upload

Supported Formats

Type

	

Purpose




JSON

	

Metadata




GeoTIFF

	

Images




TIFF

	

Confidence Maps




PDF

	

Reports




PNG

	

Preview Images

4. Authentication

All secured endpoints require JWT Bearer Authentication.

Request Header
Authorization: Bearer <ACCESS_TOKEN>
User Roles

Role

	

Permission




Analyst

	

Upload & Predict




Scientist

	

Reports & Metrics




Admin

	

Full Access

5. Base URL
Development
http://localhost:8000/api/v1
Production
https://cloudclear.ai/api/v1
6. Request & Response Format
Standard Request
{
  "image_id": "IMG1021",
  "region": "West Bengal"
}
Standard Success Response
{
  "success": true,
  "message": "Prediction completed successfully",
  "data": {},
  "timestamp": "2026-08-22T14:30:00Z"
}
Error Response
{
  "success": false,
  "error_code": 422,
  "message": "Invalid GeoTIFF format"
}
7. Error Handling

Code

	

Description




200

	

Success




201

	

Created




400

	

Bad Request




401

	

Unauthorized




403

	

Forbidden




404

	

Not Found




413

	

File Too Large




422

	

Invalid TIFF




500

	

Internal Server Error

Common Errors

Error

	

Cause




Invalid CRS

	

Wrong projection




Missing Bands

	

Less than 4 bands




Corrupted TIFF

	

Damaged image




No Historical Data

	

Archive unavailable

8. Data Models
Image Metadata
{
  "image_id": "IMG1021",
  "filename": "cloudy_scene.tif",
  "width": 10240,
  "height": 10240,
  "bands": 4,
  "crs": "EPSG:4326",
  "resolution": 5.8
}
Prediction Model
{
  "prediction_id": "P001",
  "status": "Completed",
  "ssim": 0.912,
  "psnr": 32.45
}
9. Endpoints Overview

Method

	

Endpoint

	

Purpose




POST

	

/upload

	

Upload GeoTIFF




POST

	

/predict

	

Complete AI Pipeline




POST

	

/cloud-mask

	

Cloud Detection




POST

	

/change-map

	

Change Detection




POST

	

/reconstruct

	

Reconstruction




GET

	

/quality/{id}

	

Quality Metrics




GET

	

/ndvi/{id}

	

NDVI Analysis




GET

	

/landcover/{id}

	

Land Cover




GET

	

/report/{id}

	

PDF Report




GET

	

/download/{id}

	

Download GeoTIFF

10. Upload API
POST /upload

Uploads a cloud-covered satellite image.

Request
POST /api/v1/upload
Content-Type: multipart/form-data
Form Data

Field

	

Type




image

	

GeoTIFF




region

	

String




sensor

	

String

Success Response
{
  "success": true,
  "image_id": "IMG1021",
  "status": "Uploaded"
}
Validation

GeoTIFF only

4 spectral bands

EPSG:4326

Maximum 2 GB

11. Cloud Detection API
POST /cloud-mask

Generates cloud and shadow masks.

Request
{
  "image_id": "IMG1021"
}
Response
{
  "cloud_percentage": 23.8,
  "shadow_percentage": 5.1,
  "mask": "cloud_mask.tif"
}
Outputs

Binary cloud mask

Shadow mask

Probability heatmap

12. Change Detection API
POST /change-map

Compares current and historical imagery.

Inputs

Current Image

Historical Image

Sentinel-1 SAR

Response
{
  "stable_area": 78.4,
  "changed_area": 21.6,
  "change_map": "change_map.tif"
}
Categories

Value

	

Meaning




Green

	

Stable




Yellow

	

Moderate




Red

	

Changed

13. Reconstruction API
POST /reconstruct

Runs the complete AI reconstruction pipeline.

Request
{
  "image_id": "IMG1021",
  "strategy": "adaptive"
}
AI Workflow

Cloud Detection

Change Detection

Cross-Attention Fusion

MRR Reconstruction

Quality Assessment

Response
{
  "prediction_id": "P001",
  "best_candidate": "C3",
  "output": "cloud_free.tif"
}
14. Quality Assessment API
GET /quality/{prediction_id}

Returns reconstruction quality metrics.

Example Response
{
  "psnr": 32.45,
  "ssim": 0.912,
  "mae": 0.018,
  "rmse": 0.024,
  "sam": 4.21,
  "ergas": 1.87
}
Metrics

Metric

	

Description




PSNR

	

Peak Signal Ratio




SSIM

	

Structural Similarity




MAE

	

Mean Absolute Error




RMSE

	

Root Mean Square Error




SAM

	

Spectral Angle Mapper




ERGAS

	

Global Spectral Error

15. NDVI & Land Cover API
GET /ndvi/{prediction_id}

Returns vegetation analysis.

Response
{
  "mean_ndvi": 0.63,
  "vegetation": 45.3
}
GET /landcover/{prediction_id}

Returns land cover distribution.

Response
{
  "vegetation": 45.3,
  "agriculture": 22.1,
  "water": 12.7,
  "urban": 8.3,
  "bare_land": 6.6,
  "others": 5.0
}
16. Reports API
GET /report/{prediction_id}

Generates a professional PDF report.

Includes

Original Image

Cloud Mask

Reconstruction

NDVI

Land Cover

Quality Metrics

Metadata

Response
{
  "report": "P001_report.pdf"
}
17. Download API
GET /download/{prediction_id}

Downloads the analysis-ready GeoTIFF.

Response Type
Content-Type: image/tiff
Example
GET /api/v1/download/P001

Files returned:

Cloud-Free GeoTIFF

Confidence Map

Change Map

18. Admin APIs
GET /admin/jobs

Lists processing jobs.

[
  {
    "id": "P001",
    "status": "Running"
  }
]
GET /admin/models

Returns deployed AI models.

{
  "cloud_model": "Attention U-Net",
  "fusion_model": "Cross-Attention v1",
  "qan_model": "QAN v1"
}
POST /admin/retrain

Starts model retraining.

{
  "epochs": 100,
  "batch_size": 8
}
19. Security Specification
Authentication

JWT Tokens

Role-Based Access Control

Encryption

HTTPS (TLS 1.3)

Secure GeoTIFF storage

Encrypted download URLs

Validation

CRS verification

Band validation

Metadata integrity

File size limits

Rate Limiting

Endpoint

	

Limit




Upload

	

20/hour




Predict

	

50/day




Download

	

200/day

20. Complete API Workflow
API Summary

Feature

	

Endpoint




Upload Image

	

POST /upload




Run Prediction

	

POST /predict




Cloud Mask

	

POST /cloud-mask




Change Detection

	

POST /change-map




Reconstruction

	

POST /reconstruct




Quality Metrics

	

GET /quality/{id}




NDVI

	

GET /ndvi/{id}




Land Cover

	

GET /landcover/{id}




PDF Report

	

GET /report/{id}




Download GeoTIFF

	

GET /download/{id}