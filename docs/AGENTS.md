Table of Contents

Introduction

Why Multi-Agent Architecture?

Complete Agent Workflow

Agent Communication

Agent Responsibilities

Data Retrieval Agent

Preprocessing Agent

Cloud & Shadow Detection Agent

Change Detection Agent

Adaptive Decision Agent

Cross-Attention Fusion Agent

Multi-Hypothesis Reconstruction Agent

Quality Assessment Network Agent

Confidence Estimation Agent

Delivery & Visualization Agent

Agent Memory & Data Exchange

Error Recovery Strategy

Future LangGraph Integration

1. Introduction

CloudClear AI is built using a Multi-Agent Artificial Intelligence Architecture, where each processing stage is represented by an autonomous intelligent agent.

Instead of one large pipeline, specialized agents perform independent tasks such as:

Data retrieval

Image preprocessing

Cloud detection

Change analysis

Multi-modal fusion

Reconstruction

Quality validation

Confidence estimation

Output generation

This modular design improves scalability, maintainability, and explainability.

2. Why Multi-Agent Architecture?

Traditional deep learning pipelines combine every operation into a single workflow, making debugging and upgrades difficult.

Our system separates intelligence into dedicated agents.

Benefits

Feature

	

Advantage




Modular

	

Independent development




Scalable

	

Easy to extend




Explainable

	

Each decision is traceable




Fault Tolerant

	

Failure isolation




Reusable

	

Agents can be reused




Parallel

	

Multiple agents can execute simultaneously

3. Complete Agent Workflow

Every agent produces structured outputs that become inputs for the next stage.

4. Agent Communication

Agents communicate using an internal ImagePacket object.

Shared Data Structure
ImagePacket = {
    "image": GeoTIFF,
    "metadata": {},
    "cloud_mask": None,
    "change_map": None,
    "confidence_map": None,
    "quality_score": None,
    "timestamp": ""
}

This allows loose coupling between modules.

5. Agent Responsibilities

Agent

	

Primary Function




A1

	

Data Retrieval




A2

	

Preprocessing




A3

	

Cloud Detection




A4

	

Change Detection




A5

	

Adaptive Decision




A6

	

Cross-Attention Fusion




A7

	

Reconstruction




A8

	

Quality Assessment




A9

	

Confidence Estimation




A10

	

Delivery & Visualization

6. Agent 1 — Data Retrieval Agent
Objective

Retrieve all supporting datasets required for reconstruction.

Inputs

Uploaded GeoTIFF

AOI

Acquisition Date

Tasks

Read metadata

Search historical imagery

Retrieve Sentinel-1 SAR

Validate CRS

Match temporal records

Input

Data

	

Source




Current Image

	

User Upload




Historical

	

Archive




SAR

	

Sentinel-1

Output
{
  "historical":"hist_20240512.tif",
  "sar":"sar_20240512.tif"
}
Technologies

Rasterio

GDAL

GeoPandas

7. Agent 2 — Preprocessing Agent
Objective

Prepare imagery for AI inference.

Responsibilities

Band extraction

CRS alignment

Resampling

Normalization

Patch generation

Workflow
Output

256×256 patches

Normalized tensors

8. Agent 3 — Cloud & Shadow Detection Agent
Objective

Detect cloud-covered and shadow regions.

AI Model

Attention U-Net

Outputs

Cloud Mask

Shadow Mask

Probability Map

Probability Levels

Value

	

Interpretation




0–0.3

	

Clear




0.3–0.6

	

Uncertain




0.6–1.0

	

Cloud

Result

Binary segmentation suitable for reconstruction.

9. Agent 4 — Change Detection Agent
Objective

Determine whether historical imagery is still valid.

Inputs

Current Optical

Historical Optical

Sentinel-1 SAR

Analysis

Pixel difference

Texture similarity

SAR coherence

Temporal consistency

Output

Change Probability Map

Color

	

Meaning




Green

	

Stable




Yellow

	

Moderate




Red

	

Changed

10. Agent 5 — Adaptive Decision Agent
Objective

Choose the best reconstruction strategy.

Decision Parameters

Cloud Density

Change Score

SAR Availability

Historical Reliability

Decision Matrix

Condition

	

Strategy




Stable

	

Historical




Changed

	

SAR




Mixed

	

Adaptive Fusion

Output
{
 "strategy":"adaptive_fusion"
}
11. Agent 6 — Cross-Attention Fusion Agent
Objective

Fuse multi-modal features from optical and radar imagery.

Inputs

Current RGB

Historical RGB

NIR

SAR

Cloud Mask

AI Block

Cross-Attention Transformer

Output Features

Spatial tensor

Spectral tensor

Radar features

Attention maps

These features are forwarded to reconstruction.

12. Agent 7 — Multi-Hypothesis Reconstruction Agent
Objective

Generate multiple cloud-free candidates.

Candidate Images

Candidate

	

Description




C1

	

Historical Dominant




C2

	

SAR Dominant




C3

	

Adaptive Fusion

Why Multiple Outputs?

Different regions require different reconstruction strategies.

The Quality Agent selects the optimal candidate automatically.

13. Agent 8 — Quality Assessment Network (QAN)
Objective

Evaluate every reconstructed image.

Metrics

Metric

	

Purpose




SSIM

	

Structural Quality




PSNR

	

Image Fidelity




SAM

	

Spectral Accuracy




ERGAS

	

Global Error

Quality Score
Q=0.35SSIM+0.30PSNR+0.20SAM+0.15ERGAS

Highest score becomes the final output.

14. Agent 9 — Confidence Estimation Agent
Objective

Estimate pixel-level reliability.

Confidence Levels

Score

	

Label




0.8–1.0

	

High




0.5–0.8

	

Medium




0–0.5

	

Low

Output

A confidence heatmap visualized in the dashboard.

Applications

Scientific validation

Disaster assessment

Agricultural monitoring

15. Agent 10 — Delivery & Visualization Agent
Objective

Generate analysis-ready products and deliver them to the dashboard.

Outputs

Cloud-Free GeoTIFF

Confidence Map

Change Map

NDVI Map

PDF Report

Metadata JSON

Dashboard Integration

The agent updates:

Image comparison panel

Quality metrics

NDVI visualization

Download center

16. Agent Memory & Data Exchange

All agents use a shared structured object.

Processing Packet
PredictionPacket = {
    "image_id": "",
    "cloud_mask": "",
    "change_map": "",
    "candidate_images": [],
    "best_candidate": "",
    "metrics": {},
    "confidence": ""
}

Advantages:

Consistent data format

Easy debugging

Modular processing

17. Error Recovery Strategy

Agent

	

Possible Error

	

Recovery




Data

	

Missing TIFF

	

Request re-upload




Preprocessing

	

CRS mismatch

	

Reproject




Cloud Detection

	

Empty mask

	

Retry threshold




Change Detection

	

No historical image

	

Use SAR only




Fusion

	

Missing SAR

	

Optical fallback




Reconstruction

	

Low quality

	

Generate alternative




Delivery

	

Export failed

	

Retry generation

Every failure is logged for traceability.

18. Future LangGraph Integration

The current pipeline is sequential, but future versions will convert every module into a LangGraph autonomous agent.

Planned Agent Graph
Planner Agent
      │
      ├───────────────┐
      ▼               ▼
Data Agent      Metadata Agent
      │               │
      └───────┬───────┘
              ▼
      Preprocessing Agent
              │
              ▼
      Cloud Detection
              │
              ▼
      Change Detection
              │
              ▼
      Fusion Agent
              │
              ▼
      Reconstruction Agent
              │
              ▼
      Quality Agent
              │
              ▼
      Report Agent

Benefits:

Autonomous execution

Parallel reasoning

Dynamic workflow routing

Explainable AI decisions

Agent Lifecycle

Stage

	

Agent




Data Acquisition

	

A1




Image Preparation

	

A2




Segmentation

	

A3




Temporal Analysis

	

A4




Decision Making

	

A5




Feature Fusion

	

A6




Reconstruction

	

A7




Validation

	

A8




Confidence

	

A9




Delivery

	

A10

Complete Agent Workflow
Agent Summary

Agent

	

AI Technique

	

Output




A1

	

Rule-Based Retrieval

	

Historical Images




A2

	

Image Processing

	

Normalized Tensor




A3

	

Attention U-Net

	

Cloud Mask




A4

	

CNN + SAR

	

Change Map




A5

	

Rule Engine

	

Reconstruction Strategy




A6

	

Cross-Attention

	

Multi-Modal Features




A7

	

Deep Reconstruction

	

3 Candidate Images




A8

	

Quality Network

	

Best Candidate




A9

	

Probabilistic Model

	

Confidence Heatmap




A10

	

Service Layer

	

GeoTIFF & Reportss