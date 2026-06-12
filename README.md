<div align="center">

# STAR Analyzer

**Desktop analysis tool for SLEAP-based multi-animal behavioral tracking data.**  
Process pose-estimation exports, calibrate arenas, compute kinematics and social behavior metrics, and export analysis-ready Excel workbooks.

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PySide6-Desktop%20GUI-41CD52?style=for-the-badge" />
  <img src="https://img.shields.io/badge/SLEAP-Pose%20Estimation-7A3EF2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/SciPy-Signal%20Processing-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-Excel%20Export-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</p>

</div>

---

## Overview

**STAR Analyzer** is a research-oriented desktop application for analyzing multi-animal behavioral tracking data from [SLEAP](https://sleap.ai/) pose-estimation output.

It pairs a raw SLEAP `.h5` file with the original video, lets the user calibrate an arena ROI, fills and smooths pose trajectories, computes movement and social-interaction features, and exports structured workbooks for downstream analysis.

---

## Why This Project Exists

Behavioral neuroscience data is messy. Pose-estimation outputs often contain missing points, identity switches, noisy trajectories, and frame-index alignment issues. STAR Analyzer turns those raw tracking files into interpretable behavioral features without forcing every analysis step into ad hoc notebooks.

The goal is to make the workflow more reproducible, inspectable, and useful for lab-scale behavioral analysis.

---

## Analysis Pipeline

```text
Video + SLEAP .h5
        ↓
SLEAP layout detection and frame mapping
        ↓
ROI calibration and pixel-to-cm conversion
        ↓
Gap filling and smoothing
        ↓
Kinematics and body-feature extraction
        ↓
1st-order behavior and pairwise social metrics
        ↓
Time-binned aggregation
        ↓
Excel workbooks for analysis
```

---

## Features

### Input and Calibration

- Load a video alongside a SLEAP `.h5` tracking file
- Supports standard, transposed, and sparse-frame SLEAP export layouts
- Draw a square arena ROI directly on the video frame
- Convert pixel measurements into real-world units using arena size in cm

### Preprocessing

- Three-tier NaN gap filling: PCHIP spline, Kalman smoother, linear fallback
- Savitzky-Golay smoothing before kinematic computation
- Frame mapping between video indices and SLEAP data indices

### Kinematics

- Per-node velocity, speed, heading, acceleration, and jerk
- Smoothed differentiation to reduce noise amplification
- Body-scale normalization with DSR, the median inter-hip distance

### Behavior and Social Metrics

- Single-animal states: stationary, walking, running, turning, directional reversal
- Pairwise interactions: NoseNose, NoseHead, NoseBody, NoseRear, Contact, CoOriented, AntiOriented, RelPos, Engaged, Disengaged
- Engagement Index, Reciprocity Index, and Retreat Index
- Time-binned pair and animal features at 0.25 s and 1 s resolution

### Interface

- Real-time overlay during playback
- Data inspector popup for frame-by-frame review
- Non-blocking processing and export through Qt worker threads

---

## Outputs

Running an export produces two Excel workbooks:

| File | Contents |
|---|---|
| `{name}.xlsx` | Tracking Data, Zone Summary, Session Info, 1st Order Behaviors, Behavior Summary, Engagement Indices, Animal Features, Pair Features |
| `{name}_binned.xlsx` | Animal 0.25s, Pair 0.25s, Engagement Indices 0.25s, Animal 1s, Pair 1s |

---

## Tech Stack

| Layer | Technology |
|---|---|
| GUI | PySide6 |
| Data processing | NumPy, SciPy, pandas |
| Tracking input | SLEAP `.h5`, h5py |
| Video | OpenCV |
| Gap filling | PCHIP interpolation, Kalman smoothing, linear fallback |
| Export | openpyxl / Excel workbooks |

---

## Installation

### Requirements

- Python 3.8+

### Install Dependencies

```bash
pip install PySide6 numpy scipy pandas openpyxl h5py opencv-python pykalman
```

### Run

```bash
python main.py
```

---

## Usage

1. Click **Open Video** and load the paired video file.
2. Click **Load SLEAP** and select the `.h5` pose-estimation file.
3. Draw the square arena ROI on the video frame.
4. Enter arena size and border-strip settings.
5. Click **Process** to fill gaps, smooth tracks, and compute features.
6. Use **Inspect** to review per-frame values.
7. Click **Export** to write the output workbooks.

---

## Project Structure

```text
main.py             entry point
main_window.py      application window shell
run_popup.py        main analysis window: ROI, playback, export worker
preprocessing.py    gap filling, smoothing, kinematics
behaviors.py        1st-order behavior detection
features.py         animal and pair feature extraction
binned_export.py    time-binned aggregation and Excel export
sleap_loader.py     SLEAP .h5 loader with layout auto-detection
roi_view.py         interactive ROI drawing widget
```

---

## Technical Notes

- The internal `tracks` convention is `(n_frames, 2, n_nodes, n_tracks)`, where axis 1 stores x/y coordinates.
- `frame_map` links video frame numbers to SLEAP data rows so exports stay aligned to the original recording.
- DSR, the median inter-hip distance, acts as a body-scale reference for proximity thresholds.
- Heavy processing runs away from the Qt main thread through `QThread` workers.

---

## Status

Active undergraduate research software project for computational behavioral analysis.

---

## Author

Built by [Abdullah Jawwad Yousafi](https://github.com/AbdullahJawwad2005).
