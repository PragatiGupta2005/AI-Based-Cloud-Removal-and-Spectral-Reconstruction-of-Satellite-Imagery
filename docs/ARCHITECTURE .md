Table of Contents

Architecture Overview

Design Principles

High-Level Architecture

Complete Methodology Pipeline

System Layers

Core Processing Modules

AI Architecture

Data Flow

Infrastructure Architecture

Database Architecture

Security Architecture

Deployment Architecture

Scalability

Future Architecture

1. Architecture Overview

CloudClear AI is an ISRO-inspired geospatial AI system designed to reconstruct cloud-covered satellite imagery into analysis-ready cloud-free GeoTIFF products.

Unlike conventional image enhancement systems, CloudClear AI combines:

LISS-IV Optical Imagery

Sentinel-2 Multispectral Data

Sentinel-1 SAR

Historical Satellite Archive

Cross-Attention Feature Fusion

Multi-Hypothesis Reconstruction (MRR)

Quality Assessment Network (QAN)

Confidence Estimation

The architecture is divided into 11 intelligent modules, allowing independent development and scalable deployment.

2. Design Principles

The architecture follows modern AI engineering principles.

Principle

	

Description




Modular

	

Independent processing modules




Explainable

	

Every AI decision is traceable




Scalable

	

GPU & cloud compatible




Fault Tolerant

	

Error recovery at each stage




Multi-Modal

	

Optical + SAR fusion




GIS Ready

	

Analysis-ready GeoTIFF output

3. High-Level Architecture
Architecture Layers

Layer

	

Technology




Presentation

	

Streamlit




API Layer

	

FastAPI




AI Layer

	

TensorFlow




Data Layer

	

Rasterio + GDAL




Storage

	

GeoTIFF + Metadata

4. Complete Methodology Pipeline

The project follows the methodology shown in your architecture diagram.

This sequential workflow guarantees scientific and explainable reconstruction.

5. System Layers
Layer 1 — Presentation Layer

Responsible for user interaction.

Components

Dashboard

Sidebar Navigation

AOI Selection

Image Comparison

Metrics Panel

Download Center

Technology: Streamlit

Layer 2 — API Layer

Acts as the communication bridge.

Responsibilities

Authentication

Request Validation

GeoTIFF Upload

AI Invocation

Report Generation

Technology: FastAPI

Layer 3 — AI Layer

The intelligence of the system.

Contains:

Cloud Detection

Change Detection

Cross-Attention

Reconstruction

QAN

Confidence Model

Technology: TensorFlow / Keras

Layer 4 — Data Layer

Handles geospatial datasets.

Sources

LISS-IV

Sentinel-2

Sentinel-1

Historical Archive

Technology: Rasterio + GDAL

6. Core Processing Modules
Module 1 – Data Retrieval
Input

Cloudy GeoTIFF

Metadata

Output

Historical Image

SAR Image

Operations

Metadata parsing

Archive search

CRS validation

Module 2 – Preprocessing
Tasks

Band extraction

Co-registration

Radiometric normalization

Patch generation

Output

Normalized tensor.

Module 3 – Cloud & Shadow Detection

Model: Attention U-Net

Outputs

Cloud Mask

Shadow Mask

Probability Map

Module 4 – Change Detection

Compares:

Current image

Historical image

SAR coherence

Output

Change probability map.

Module 5 – Adaptive Decision Engine

Chooses reconstruction strategy.

Condition

	

Strategy




Stable

	

Historical




Changed

	

SAR




Mixed

	

Adaptive Fusion

Module 6 – Cross-Attention Fusion

Fuses:

RGB

NIR

Historical Features

SAR Backscatter

Produces a unified feature tensor.

Module 7 – Multi-Hypothesis Reconstruction

Generates three candidates.

Candidate

	

Description




C1

	

Historical




C2

	

SAR




C3

	

Adaptive

Module 8 – Quality Assessment Network

Evaluates every candidate.

Metrics

SSIM

PSNR

SAM

ERGAS

Selects the best reconstruction.

Module 9 – Confidence Estimation

Produces pixel-level reliability.

Confidence Levels

Level

	

Meaning




High

	

Reliable




Medium

	

Moderate




Low

	

Uncertain

Module 10 – Analysis Ready Output

Creates GIS-compatible products.

Generated files:

Cloud-Free GeoTIFF

Confidence TIFF

NDVI Map

Change Map

Module 11 – Delivery & Access

Final outputs are delivered through:

Dashboard

REST API

PDF Reports

GeoTIFF Downloads

7. AI Architecture
Primary Models

Model

	

Purpose




Attention U-Net

	

Cloud Detection




CNN

	

Change Detection




Cross-Attention Transformer

	

Feature Fusion




MRR Network

	

Reconstruction




QAN

	

Candidate Ranking

AI Workflow
8. Data Flow Architecture

All intermediate outputs remain geospatially aligned.

9. Infrastructure Architecture
Software Stack

Layer

	

Technology




UI

	

Streamlit




API

	

FastAPI




AI

	

TensorFlow




Image Processing

	

Rasterio




GIS

	

GDAL




Charts

	

Plotly

Runtime Flow
User
 │
 ▼
Dashboard
 │
 ▼
FastAPI
 │
 ▼
TensorFlow
 │
 ▼
GeoTIFF Storage
10. Database Architecture

Although imagery is stored as GeoTIFF, metadata is stored separately.

Images Table

Field

	

Type




image_id

	

UUID




filename

	

String




date

	

Date




CRS

	

String




sensor

	

String

Predictions Table

Field

	

Type




prediction_id

	

UUID




SSIM

	

Float




PSNR

	

Float




status

	

String

Reports Table

Field

	

Type




report_id

	

UUID




pdf_path

	

String




prediction_id

	

UUID

11. Security Architecture
Authentication

JWT Tokens

Role-Based Access Control

Validation

GeoTIFF verification

CRS validation

Band validation

Metadata integrity

Encryption

HTTPS (TLS 1.3)

Secure storage

Protected downloads

12. Deployment Architecture
Development Environment
Windows 11
     │
Python 3.11
     │
FastAPI
     │
TensorFlow
     │
Streamlit
Production Environment
Docker
   │
Nginx
   │
FastAPI
   │
TensorFlow Serving
   │
GPU Server
13. Scalability

The architecture supports future scaling.

Horizontal Scaling

Multiple API instances

GPU workers

Batch processing

Vertical Scaling

Larger GPU memory

Faster SSD storage

Distributed GeoTIFF archive

14. Future Architecture
Planned Enhancements
LangGraph Agent Orchestration

Convert every module into an autonomous AI agent.

Vision Transformer

Replace U-Net with transformer-based reconstruction.

Multi-Temporal Fusion

Use multiple historical images for improved accuracy.

Cloud Deployment

Deploy on Kubernetes with distributed GPU inference.

Architecture Summary

Component

	

Technology




Frontend

	

Streamlit




Backend

	

FastAPI




AI Engine

	

TensorFlow




Cloud Detection

	

Attention U-Net




Fusion

	

Cross-Attention Transformer




Reconstruction

	

MRR Network




Quality

	

QAN




Storage

	

GeoTIFF




Output

	

Analysis-Ready Products

End-to-End Architecture Workflow
Cloudy LISS-IV GeoTIFF
        │
        ▼
Data Retrieval & Historical Matching
        │
        ▼
Preprocessing & Band Normalization
        │
        ▼
Cloud & Shadow Detection
        │
        ▼
Temporal Change Detection
        │
        ▼
Adaptive Decision Engine
        │
        ▼
Cross-Attention Optical + SAR Fusion
        │
        ▼
Multi-Hypothesis Reconstruction
        │
        ▼
Quality Assessment Network
        │
        ▼
Confidence Estimation
        │
        ▼
Analysis-Ready GeoTIFF + NDVI + Report