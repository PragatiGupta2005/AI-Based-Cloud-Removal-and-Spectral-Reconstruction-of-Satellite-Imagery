Table of Contents

Roadmap Overview

Development Strategy

Project Timeline

Phase 1 – Research & Planning

Phase 2 – Dataset Collection

Phase 3 – Data Preprocessing

Phase 4 – AI Model Development

Phase 5 – Reconstruction Pipeline

Phase 6 – Dashboard Development

Phase 7 – Testing & Validation

Phase 8 – Deployment

Future Roadmap

Success Milestones

1. Roadmap Overview

CloudClear AI follows an 8-phase development roadmap inspired by real-world AI product development. The roadmap covers everything from research and dataset preparation to deployment and future enhancements.

Total Duration

Item

	

Duration




Project Length

	

24 Weeks




Team Size

	

3 Members




AI Models

	

4




Dashboard

	

Streamlit




Backend

	

FastAPI

2. Development Strategy

The project is divided into independent milestones so each module can be developed and tested separately.

Each phase produces a usable output before moving to the next stage.

3. Project Timeline
24-Week Schedule

Phase

	

Weeks

	

Status




Research & Planning

	

1–2

	

✅




Dataset Collection

	

3–5

	

✅




Preprocessing

	

6–8

	

🔄




AI Model Training

	

9–14

	

⏳




Reconstruction Pipeline

	

15–17

	

⏳




Dashboard Development

	

18–20

	

⏳




Testing & Validation

	

21–23

	

⏳




Deployment

	

24

	

⏳

4. Phase 1 — Research & Planning

Duration: Week 1–2

Objectives

Study cloud removal techniques

Analyze ISRO methodology

Identify datasets

Design system architecture

Deliverables

Literature Review

Problem Statement

Methodology Diagram

SRS Document

Success Criteria

Architecture finalized

Tech stack selected

Dataset source confirmed

5. Phase 2 — Dataset Collection

Duration: Week 3–5

Objective

Collect paired cloudy and clear satellite imagery.

Data Sources

Dataset

	

Purpose




LISS-IV

	

Optical




Sentinel-2

	

RGB + NIR




Sentinel-1

	

SAR




Historical Archive

	

Temporal Reference

Folder Structure
data/

├── cloudy/
├── clear/
├── historical/
└── sar/
Target Dataset

Type

	

Images




Cloudy

	

30




Clear

	

30




Historical

	

30




SAR

	

30

Total: 120 GeoTIFF Images

Deliverables

Organized dataset

Metadata CSV

CRS validation

6. Phase 3 — Data Preprocessing

Duration: Week 6–8

Objectives

Prepare GeoTIFF images for deep learning.

Tasks

TIFF reading

Band extraction

Co-registration

Radiometric normalization

Patch generation

Data augmentation

Pipeline
GeoTIFF
   │
   ▼
Rasterio
   │
   ▼
Band Extraction
   │
   ▼
Normalization
   │
   ▼
256×256 Patches
Deliverables

NumPy tensors

Training patches

Validation dataset

7. Phase 4 — AI Model Development

Duration: Week 9–14

This is the core AI phase.

Model 1 — Cloud Detection

Architecture: U-Net

Output

Cloud mask

Shadow mask

Model 2 — Change Detection

Inputs

Current image

Historical image

SAR

Output

Change probability map.

Model 3 — Cross-Attention Fusion

Fuses:

RGB

NIR

SAR

Historical features

Output becomes reconstruction features.

Model 4 — Multi-Hypothesis Reconstruction

Generates three candidates.

Candidate

	

Description




C1

	

Historical




C2

	

SAR




C3

	

Adaptive

Deliverables

Trained models

Saved checkpoints

Validation results

8. Phase 5 — Reconstruction Pipeline

Duration: Week 15–17

Integrate all AI modules into one workflow.

Workflow
Deliverables

End-to-end inference

Patch stitching

Analysis-ready output

9. Phase 6 — Dashboard Development

Duration: Week 18–20

Develop the Streamlit interface shown in the UI design.

Pages

Page

	

Status




Dashboard

	

⏳




Upload

	

⏳




Cloud Detection

	

⏳




Reconstruction

	

⏳




Analysis

	

⏳




Reports

	

⏳

Components

AOI selector

Sensor selector

Image comparison

Metrics cards

NDVI visualization

Download center

Deliverables

Fully functional dashboard

Responsive UI

Dark theme

10. Phase 7 — Testing & Validation

Duration: Week 21–23

Functional Testing

Test

	

Expected




Upload TIFF

	

Success




Invalid CRS

	

Error




Prediction

	

Cloud-free image




Download

	

GeoTIFF

AI Validation

Metrics used:

Metric

	

Target




SSIM

	

>0.90




PSNR

	

>30 dB




MAE

	

<0.02




RMSE

	

<0.03




SAM

	

<5°

User Acceptance Testing

Conduct testing with:

GIS students

Researchers

Faculty

11. Phase 8 — Deployment

Duration: Week 24

Local Deployment
streamlit run app.py
Backend
uvicorn api:app
Production Stack

Component

	

Technology




Frontend

	

Streamlit




Backend

	

FastAPI




AI

	

TensorFlow




Storage

	

GeoTIFF




Server

	

Docker

Deliverables

Working application

Documentation

Final report

Source code

12. Future Roadmap
Version 2.0
Vision Transformer

Replace U-Net with ViT-based reconstruction.

Multi-Temporal Fusion

Use multiple historical dates instead of one.

Foundation Geospatial Model

Integrate large remote sensing foundation models.

Version 3.0
LangGraph AI Agents

Every processing module becomes an autonomous AI agent.

Explainable AI

Generate natural-language explanations for reconstruction decisions.

Interactive GIS Layer

Enable direct editing inside the dashboard.

Version 4.0
Cloud Deployment

Kubernetes

GPU inference

REST API service

Real-Time Processing

Connect directly to satellite acquisition streams.

13. Success Milestones

Milestone

	

Outcome




M1

	

Research completed




M2

	

Dataset collected




M3

	

Preprocessing pipeline ready




M4

	

Cloud detection trained




M5

	

Reconstruction model trained




M6

	

Dashboard completed




M7

	

Validation passed




M8

	

Final deployment

Risk Assessment

Risk

	

Mitigation




Insufficient data

	

Increase historical samples




GPU limitation

	

Patch-based training




Poor reconstruction

	

Adaptive fusion




CRS mismatch

	

Automatic reprojection




Large TIFF size

	

Tile processing

Final Development Roadmap
Research & Planning
        │
        ▼
Dataset Collection
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
Cross-Attention Fusion
        │
        ▼
MRR Reconstruction
        │
        ▼
Quality Assessment
        │
        ▼
Dashboard Development
        │
        ▼
Testing & Validation
        │
        ▼
Deployment
        │
        ▼
Version 2.0 (Future)
Project Completion Criteria

A project is considered complete when all the following are achieved:

30+ paired satellite image datasets prepared

AI models trained successfully

Cloud removal SSIM above 0.90

PSNR above 30 dB

Functional Streamlit dashboard

FastAPI backend operational

GeoTIFF export working

PDF reports generated

Complete technical documentation finalized

GitHub repository ready for submission