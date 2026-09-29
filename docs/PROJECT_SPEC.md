Table of Contents

Project Introduction

Problem Statement

Proposed Solution

Project Objectives

Scope of the Project

Target Users

User Roles

System Overview

Functional Requirements

Non-Functional Requirements

AI/ML Requirements

Geospatial Requirements

Hardware & Software Requirements

Database Requirements

Input & Output Specification

Success Metrics

Constraints

Expected Deliverables

1. Project Introduction
Project Title

CloudClear AI: AI-Based Cloud Removal and Spectral Reconstruction of Satellite Imagery

CloudClear AI is an intelligent Earth Observation platform designed to reconstruct cloud-free satellite imagery using LISS-IV optical images, Sentinel-1 SAR, historical satellite archives, and deep learning.

The system automatically detects cloud-covered regions, analyzes temporal changes, reconstructs hidden land surfaces, and generates analysis-ready GeoTIFF images with confidence estimation.

Domain

Artificial Intelligence

Machine Learning

Computer Vision

Remote Sensing

Geospatial Analytics

2. Problem Statement

Clouds obstruct nearly 60% of optical satellite imagery, making Earth observation difficult.

This affects:

Crop monitoring

Flood assessment

Forest analysis

Urban planning

Disaster response

Traditional cloud removal methods:

Use interpolation

Lose spectral information

Produce unrealistic textures

Ignore temporal land changes

Therefore, an intelligent reconstruction system is required.

3. Proposed Solution

CloudClear AI introduces a multi-agent geospatial AI architecture that combines:

Current LISS-IV imagery

Historical satellite images

Sentinel-1 SAR

Cross-attention feature fusion

Multi-Hypothesis Reconstruction (MRR)

Quality Assessment Network (QAN)

The system generates three reconstruction candidates and automatically selects the most reliable result using quality-aware ranking.

4. Project Objectives
Primary Objectives

Detect cloud and shadow regions.

Retrieve historical imagery.

Analyze temporal land changes.

Fuse optical and SAR information.

Generate cloud-free imagery.

Preserve spectral consistency.

Produce GIS-ready outputs.

Secondary Objectives

Confidence-aware reconstruction

Automated quality assessment

Interactive visualization

PDF report generation

GeoTIFF export

5. Scope of the Project
Included

GeoTIFF processing

LISS-IV imagery

Sentinel-1 SAR integration

Cloud segmentation

Deep learning reconstruction

NDVI comparison

Land cover analysis

Quality metrics

Excluded

Real-time satellite streaming

Drone imagery

Hyperspectral imagery

3D terrain reconstruction

Manual cloud editing

6. Target Users

User

	

Purpose




GIS Analyst

	

Image processing




ISRO Scientist

	

Quality analysis




Agriculture Dept.

	

Crop monitoring




Disaster Authority

	

Flood assessment




Urban Planner

	

Infrastructure planning




Research Student

	

Remote sensing research

7. User Roles
GIS Analyst
Permissions

Upload imagery

Generate predictions

Download GeoTIFF

View confidence map

Scientist

Additional permissions:

Analyze NDVI

Compare land cover

Generate reports

Review quality metrics

Administrator

Full control:

Dataset management

Model retraining

Pipeline monitoring

User management

System configuration

8. System Overview
High-Level Workflow
Cloudy Image
      │
      ▼
Data Retrieval
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
Cross-Attention Fusion
      │
      ▼
MRR Reconstruction
      │
      ▼
Quality Assessment
      │
      ▼
Confidence Map
      │
      ▼
Analysis Ready GeoTIFF

The architecture follows the methodology shown in the ISRO workflow.

9. Functional Requirements
FR-01 Image Upload

The system shall allow users to upload GeoTIFF satellite images.

Supported Formats

TIFF

GeoTIFF

Validation

CRS

Resolution

Bands

Metadata

FR-02 Metadata Extraction

The system shall automatically extract:

Image size

CRS

Resolution

Acquisition date

Sensor type

FR-03 Historical Image Retrieval

The system shall retrieve the best historical image using:

Same AOI

Temporal proximity

Minimum cloud cover

Spatial resolution

FR-04 SAR Retrieval

The system shall load Sentinel-1 SAR imagery corresponding to the uploaded optical image.

FR-05 Cloud Detection

The AI shall generate:

Cloud mask

Shadow mask

Probability map

Model: U-Net

FR-06 Change Detection

The system shall compare:

Current optical

Historical optical

SAR coherence

Output:

Stable region

Changed region

Probability map

FR-07 Adaptive Decision Engine

The AI shall determine reconstruction strategy.

Condition

	

Strategy




Stable

	

Historical




Changed

	

SAR




Mixed

	

Adaptive

FR-08 Cross-Attention Fusion

The system shall fuse:

RGB

NIR

SAR

Historical texture

Output:

Feature tensor.

FR-09 Reconstruction

Generate three candidate images:

Historical

SAR

Adaptive Fusion

FR-10 Quality Assessment

The QAN shall compute:

SSIM

PSNR

SAM

ERGAS

Best candidate is selected.

FR-11 Confidence Estimation

Every pixel shall receive a confidence score.

Levels:

High

Medium

Low

FR-12 Visualization

Dashboard shall display:

Original image

Cloud mask

Reconstruction

Confidence map

NDVI

Metrics

FR-13 Download

Users shall download:

GeoTIFF

PNG

PDF

JSON metadata

10. Non-Functional Requirements
Performance

Parameter

	

Target




Upload

	

Under 10 s




Prediction

	

Under 30 s




GPU Memory

	

Under 8 GB




Patch Size

	

256×256

Reliability

Automatic validation

Error recovery

Safe storage

Logging

Scalability

Support for:

Large GeoTIFF

Batch prediction

GPU inference

Multiple users

Security

JWT authentication

HTTPS

Role-based access

Secure downloads

Usability

Modern dashboard with:

Dark theme

Interactive maps

Charts

Responsive layout

11. AI / ML Requirements
Primary Model

Attention U-Net

Purpose:

Cloud segmentation

Reconstruction

Supporting Models

Model

	

Purpose




U-Net

	

Cloud detection




CNN

	

Change detection




Cross-Attention

	

Fusion




QAN

	

Ranking

Training Configuration

Parameter

	

Value




Epochs

	

100




Batch Size

	

8




Optimizer

	

Adam




Loss

	

L1 + SSIM




Patch Size

	

256

12. Geospatial Requirements
Input Sensors

Sensor

	

Type




LISS-IV

	

Optical




Sentinel-2

	

Optical




Sentinel-1

	

SAR

Required Bands

Band

	

Description




B2

	

Blue




B3

	

Green




B4

	

Red




B8

	

Near Infrared

Coordinate System

EPSG:4326

GeoTIFF compatible

13. Hardware & Software Requirements
Hardware

Component

	

Requirement




CPU

	

Intel i5 / Ryzen 5




RAM

	

16 GB




GPU

	

RTX 3060 (8 GB+)




Storage

	

100 GB

Software

Software

	

Version




Python

	

3.11




TensorFlow

	

2.x




Rasterio

	

Latest




Streamlit

	

Latest




FastAPI

	

Latest

14. Database Requirements
Image Metadata

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

Prediction Table

Field

	

Type




prediction_id

	

UUID




SSIM

	

Float




PSNR

	

Float




Status

	

String

Reports

Stores generated PDF reports and download history.

15. Input & Output Specification
Input

Item

	

Format




Satellite Image

	

GeoTIFF




Historical Image

	

GeoTIFF




SAR Image

	

GeoTIFF

Output

Item

	

Format




Cloud-Free Image

	

GeoTIFF




Confidence Map

	

TIFF




Change Map

	

TIFF




Report

	

PDF




Metadata

	

JSON

16. Success Metrics
Reconstruction Metrics

Metric

	

Target




SSIM

	

Greater than 0.90




PSNR

	

Above 30 dB




MAE

	

Less than 0.02




RMSE

	

Less than 0.03




SAM

	

Below 5°




ERGAS

	

Below 2.0

Example Dashboard Values:

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

17. Constraints
Technical

Requires GeoTIFF input

Needs historical imagery

SAR availability improves accuracy

GPU recommended for training

Dataset

LISS-IV

Sentinel-1

Sentinel-2

Historical archive

18. Expected Deliverables
Software

Streamlit Dashboard

FastAPI Backend

TensorFlow Models

REST APIs

AI Models

Cloud Detection Model

Change Detection Model

Cross-Attention Fusion Model

QAN Model

Outputs

Cloud-Free GeoTIFF

Confidence Heatmap

Change Probability Map

NDVI Analysis

Land Cover Distribution

PDF Quality Report

Project Summary

Item

	

Description




Project Name

	

CloudClear AI




Domain

	

Geospatial AI




Architecture

	

Multi-Agent




Frontend

	

Streamlit




Backend

	

FastAPI




AI Models

	

U-Net + Cross-Attention + QAN




Input

	

LISS-IV, Sentinel-1, Sentinel-2




Output

	

Cloud-Free Analysis-Ready GeoTIFF




End Users

	

ISRO, GIS Analysts, Researchers