Table of Contents

Introduction

System Requirements

Launching the Application

Dashboard Overview

Complete User Workflow

Step 1 – Select Area of Interest (AOI)

Step 2 – Choose Satellite Sensors

Step 3 – Upload GeoTIFF

Step 4 – Run AI Processing

Step 5 – Analyze Results

Step 6 – NDVI & Land Cover Analysis

Step 7 – Download Outputs

Reports Module

History Module

Troubleshooting

1. Introduction

CloudClear AI is an interactive geospatial dashboard that enables users to transform cloud-covered satellite imagery into analysis-ready cloud-free GeoTIFF images using Artificial Intelligence.

The application is designed for:

GIS Analysts

Remote Sensing Researchers

Agriculture Departments

Disaster Management Teams

Environmental Scientists

The complete workflow requires only 7 simple steps from image upload to report generation.

2. System Requirements
Minimum Requirements

Component

	

Requirement




OS

	

Windows 10/11




RAM

	

16 GB




CPU

	

Intel i5 / Ryzen 5




GPU

	

RTX 3060 (Recommended)




Python

	

3.11

Required Libraries

TensorFlow

Rasterio

OpenCV

NumPy

Streamlit

FastAPI

Plotly

3. Launching the Application
Step 1 – Activate Environment
venv\Scripts\activate
Step 2 – Start Backend
uvicorn api:app --reload
Step 3 – Start Dashboard
streamlit run app.py

The dashboard opens automatically at:

http://localhost:8501
4. Dashboard Overview

The interface consists of five major sections.

Dashboard Components

Component

	

Purpose




Sidebar

	

Navigation




Header

	

AOI & Sensor Selection




Image Grid

	

AI Results




Metrics

	

Quality Evaluation




Bottom Cards

	

Downloads & Reports

5. Complete User Workflow

Each prediction follows the same workflow regardless of satellite dataset.

6. Step 1 – Select Area of Interest (AOI)

The first step is choosing the geographical region.

AOI Panel

Available options:

State

District

Custom Coordinates

Draw Bounding Box

Example

Field

	

Value




State

	

West Bengal




District

	

Nadia




Latitude

	

22.98




Longitude

	

88.47

Click Change AOI to update the region.

7. Step 2 – Choose Satellite Sensors

CloudClear AI supports multiple sensors.

Optical Sensor

Options:

LISS-IV

Sentinel-2

Radar Sensor

Options:

Sentinel-1

Sensor Configuration

Sensor

	

Purpose




LISS-IV

	

Optical RGB




Sentinel-2

	

Multispectral




Sentinel-1

	

SAR

The selected sensors determine the reconstruction strategy.

8. Step 3 – Upload GeoTIFF

Click Upload Satellite Image.

Supported Formats

Format

	

Supported




TIFF

	

Yes




GeoTIFF

	

Yes




PNG

	

No




JPEG

	

No

Upload Screen
7

After upload, the metadata panel appears automatically.

Metadata Preview

Property

	

Example




CRS

	

EPSG:4326




Width

	

10240




Height

	

10240




Bands

	

4




Resolution

	

5.8 m

If validation fails, an error message is displayed.

9. Step 4 – Run AI Processing

Press Process Images.

The pipeline automatically executes:

Historical Retrieval

Preprocessing

Cloud Detection

Change Detection

Cross-Attention Fusion

MRR Reconstruction

Quality Assessment

Confidence Estimation

Progress Indicator
Uploading Image...
██████████

Cloud Detection...
██████████

Reconstruction...
██████████

Generating Report...
██████████

Completed ✓

Average processing time:

20–30 seconds

10. Step 5 – Analyze Results

After completion, four synchronized viewers appear.

Cloudy Input

Original uploaded image

Cloud Mask

AI detected cloud regions

Reconstructed Image

Final AI reconstruction

Confidence Heatmap

Pixel reliability estimation

Interpretation

Viewer

	

Description




Cloudy

	

Original image




Mask

	

Cloud segmentation




Reconstruction

	

AI output




Confidence

	

Reliability map

Users can zoom and compare all images simultaneously.

11. NDVI & Land Cover Analysis
NDVI Comparison

The dashboard computes vegetation indices before and after reconstruction.

Reference NDVI

Ground truth vegetation index

Reconstructed NDVI

Generated from cloud-free imagery

NDVI Scale

Value

	

Meaning




-1

	

Water




0

	

Bare Soil




0.3

	

Sparse Vegetation




0.6

	

Healthy Crops




1

	

Dense Forest

Land Cover Distribution

Automatically generated after reconstruction.

Example values:

Class

	

Percentage




Vegetation

	

45.3%




Agriculture

	

22.1%




Water

	

12.7%




Urban

	

8.3%




Bare Land

	

6.6%

12. Quality Metrics

The Metrics panel evaluates reconstruction quality.

Displayed Metrics

Metric

	

Description




PSNR

	

Image quality




SSIM

	

Structural similarity




MAE

	

Pixel error




RMSE

	

Reconstruction error




SAM

	

Spectral accuracy




ERGAS

	

Global spectral quality

Example Dashboard

Metric

	

Value




PSNR

	

32.45 dB




SSIM

	

0.912




MAE

	

0.018




RMSE

	

0.024




SAM

	

4.21°




ERGAS

	

1.87

Overall Quality Badge:

Excellent

13. Download Outputs

The Download Center provides multiple export options.

Available Files

File

	

Format




Cloud-Free Image

	

GeoTIFF




Confidence Map

	

TIFF




Cloud Mask

	

TIFF




NDVI Map

	

PNG




Report

	

PDF




Metadata

	

JSON

Download Panel

Users may download individually or as a ZIP package.

14. Reports Module

The Reports page generates a professional PDF.

Included Sections

Original Image

Cloud Mask

Change Map

Reconstructed Image

Confidence Heatmap

NDVI Comparison

Land Cover Statistics

Quality Metrics

Metadata

Report Generation

Click Generate Report.

The PDF is created automatically.

15. History Module

Every prediction is stored locally.

History Table

Prediction

	

Date

	

Status




P001

	

22 Aug

	

Completed




P002

	

23 Aug

	

Completed




P003

	

24 Aug

	

Running

Users can reopen previous analyses without reprocessing.

16. Troubleshooting
Upload Error

Problem

Invalid TIFF

Solution

Verify GeoTIFF format

Check CRS

Ensure 4 spectral bands

Prediction Failed

Possible Causes

Missing historical image

Corrupted SAR

GPU memory full

Resolution

Retry processing

Reduce patch size

Restart backend

Dashboard Not Loading

Run:

streamlit run app.py

Ensure FastAPI is also running.

Keyboard Shortcuts

Key

	

Action




Ctrl + U

	

Upload Image




Ctrl + P

	

Process Images




Ctrl + D

	

Download Result




Ctrl + R

	

Refresh Dashboard

Best Practices

Use cloud-covered GeoTIFF images only.

Prefer Sentinel-1 SAR for better reconstruction.

Keep CRS consistent across datasets.

Validate metadata before processing.

Use GPU for large satellite scenes.

End-to-End Usage Flow
Usage Summary

Step

	

User Action




1

	

Launch Streamlit Dashboard




2

	

Select AOI




3

	

Choose Optical & SAR Sensors




4

	

Upload Cloudy GeoTIFF




5

	

Click Process Images




6

	

Review Cloud Mask & Reconstruction




7

	

Analyze NDVI & Metrics




8

	

Download GeoTIFF & PDF Report