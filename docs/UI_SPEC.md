Table of Contents

UI Overview

Design Goals

Dashboard Layout

Navigation Structure

Color Palette

Typography

Dashboard Components

Screen Specifications

Interactive Elements

Charts & Visualizations

Download Center

Responsive Design

Accessibility Guidelines

1. UI Overview

CloudClear AI provides a professional Earth Observation dashboard for GIS analysts and researchers. The interface is inspired by modern geospatial platforms and emphasizes clarity, dark-theme visualization, and scientific analytics.

Primary Objectives

Simple satellite image workflow

One-click AI reconstruction

Scientific visualization

GIS-ready downloads

Interactive quality analysis

2. Design Goals

The UI follows five core principles:

Goal

	

Description




Minimal

	

Clean scientific interface




Dark Theme

	

Better image visualization




Interactive

	

Live metrics & charts




Explainable AI

	

Confidence & quality maps




Responsive

	

Desktop & tablet support

3. Dashboard Layout

The application uses a 3-column layout with a permanent sidebar.

Layout Sections

Area

	

Purpose




Left Sidebar

	

Navigation




Top Header

	

AOI & Sensor Selection




Image Grid

	

AI Results




Metrics Panel

	

Quality Metrics




Bottom Cards

	

Model, History, Download

4. Navigation Structure
Sidebar Menu

The sidebar remains fixed across all pages.

Menu

	

Icon

	

Function




Dashboard

	

Home

	

Main overview




Upload / Select Data

	

Upload

	

GeoTIFF upload




Cloud Detection

	

Cloud

	

View masks




Reconstruction

	

Layers

	

AI reconstruction




Analysis

	

Chart

	

Metrics & NDVI




Change Detection

	

Compare

	

Temporal analysis




Reports

	

File

	

Export PDF




History

	

Clock

	

Previous jobs




Settings

	

Gear

	

Preferences

Current AOI Card

Displays:

Latitude

Longitude

Area

Change AOI button

5. Color Palette

The UI uses a dark scientific theme.

Element

	

Color




Background

	

#08111F




Sidebar

	

#0D1726




Card

	

#111C2B




Primary Blue

	

#3B82F6




Green

	

#22C55E




Purple

	

#8B5CF6




Warning

	

#F59E0B




Text

	

#E5E7EB

Confidence Heatmap

Level

	

Color




High

	

Yellow




Medium

	

Orange




Low

	

Purple

6. Typography

Element

	

Font Size




App Title

	

30 px




Section Heading

	

22 px




Card Title

	

18 px




Labels

	

14 px




Metadata

	

12 px

Font Family: Inter / Segoe UI

7. Dashboard Components
7.1 Header

Contains:

Project title

User profile

Theme toggle

Notification icon

AOI Toolbar

Control

	

Type




Region

	

Dropdown




Date

	

Calendar




Optical Sensor

	

Dropdown




SAR Sensor

	

Dropdown




Process

	

Button

Example values:

Region: West Bengal

Date: 12 May 2024

Optical: Sentinel-2

SAR: Sentinel-1

7.2 Image Comparison Panel

Four synchronized viewers.

1. Cloudy Input

RGB
B4
B3
B2

Original optical imagery

2. Cloud Mask

Cloud
Clear

AI detected cloud & shadow

3. Reconstructed

Output

MRR + Cross-Attention result

4. Confidence Map

High

Low

Pixel reliability estimation

Each image supports zoom and pan.

8. Screen Specifications
Screen 1 — Dashboard

Purpose:

Overall project overview.

Components:

AOI selector

Image grid

Metrics

NDVI

Land cover

Downloads

Screen 2 — Upload
Components

Drag & Drop

TIFF validation

Metadata preview

Displayed metadata:

Field

	

Example




CRS

	

EPSG:4326




Width

	

10240




Height

	

10240




Bands

	

4

Screen 3 — Cloud Detection

Displays:

Cloud probability

Shadow probability

Binary mask

Cloud percentage

Example:

Metric

	

Value




Cloud

	

23.8%




Shadow

	

5.1%

Screen 4 — Reconstruction

Shows three AI candidates.

Candidate

	

Description




C1

	

Historical




C2

	

SAR




C3

	

Adaptive

QAN highlights the best candidate.

Screen 5 — Analysis

Contains:

PSNR

SSIM

MAE

RMSE

SAM

ERGAS

Quality Card

Metric

	

Value




PSNR

	

32.45




SSIM

	

0.912




MAE

	

0.018




RMSE

	

0.024

Overall quality badge:

Excellent

Screen 6 — Change Detection

Displays temporal difference.

Visualizations:

Historical

Current

Change map

Color legend:

Color

	

Meaning




Green

	

Stable




Red

	

Changed

Screen 7 — Reports

Download options:

GeoTIFF

Confidence TIFF

PDF Report

ZIP Package

9. Interactive Elements
Buttons

Button

	

Action




Process Images

	

Run AI




Change AOI

	

Select region




View Metrics

	

Open analytics




Download GeoTIFF

	

Export image




Download PDF

	

Export report

Primary button color:

Blue → Purple gradient.

Dropdowns

Four primary dropdowns:

Region

Date

Optical Sensor

SAR Sensor

All values are searchable.

Progress Indicator

During processing:

Uploading...
████████░░

Cloud Detection...
██████████

Reconstruction...
██████████

Completed
10. Charts & Visualizations
NDVI Comparison

Two maps displayed side-by-side.

Reference NDVI

Ground truth vegetation index

Reconstructed NDVI

Generated from cloud-free imagery

Color scale:

-1 → Water

0 → Soil

1 → Dense Vegetation

Land Cover Distribution

Displayed as a donut chart.

Classes:

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




Others

	

5.0%

Confidence Heatmap

Uses inferno color palette.

Interpretation:

Color

	

Confidence




Yellow

	

High




Orange

	

Medium




Purple

	

Low

11. Download Center

Three download buttons are available.

Button

	

File




Download Image

	

GeoTIFF




Download Report

	

PDF




Download All

	

ZIP

Preview image appears beside the buttons.

12. Responsive Design
Desktop (1920 px)

Full sidebar

Four image viewers

Three bottom cards

Tablet (1024 px)

Collapsible sidebar

Two-column layout

Mobile

Hidden sidebar

Vertical stacking

Swipe image gallery

13. Accessibility Guidelines
Accessibility Features

High contrast colors

Keyboard navigation

Screen-reader labels

Color-blind friendly legends

Large clickable controls

UI Component Summary

Component

	

Count




Sidebar Menus

	

9




Image Viewers

	

4




Metric Cards

	

6




Charts

	

2




Download Buttons

	

3




Dropdowns

	

4




Action Buttons

	

5

Complete User Journey
Login
   │
   ▼
Select AOI
   │
   ▼
Choose Date & Sensors
   │
   ▼
Upload GeoTIFF
   │
   ▼
Process Images
   │
   ▼
Cloud Detection
   │
   ▼
Reconstruction
   │
   ▼
Quality Analysis
   │
   ▼
Download GeoTIFF & Report
Design Conclusion

The CloudClear AI interface is designed for scientific Earth observation workflows, combining AI explainability, geospatial visualization, and professional analytics into a single dark-themed dashboard suitable for ISRO researchers, GIS analysts, and remote sensing applications.