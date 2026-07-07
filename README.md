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
Processing Options dialog — choose which modules to run
        ↓
Kinematics (always), behaviors, features, zones, proximity (selective)
        ↓
Analysis cache shared with Export — no double computation
        ↓
Export Options dialog — outputs gated by what was processed
        ↓
Excel workbooks + graph PDFs for analysis
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

### Selective Processing

A **Processing Options** dialog before each run lets you choose which modules to compute:

| Module | What it covers |
|---|---|
| Kinematics | Speed, acceleration, jerk, heading — always computed |
| Single-animal behaviors | Stationary, locomotion, turning, directional reversal |
| Feature arrays | Shape descriptors, path efficiency, entropy, curvature |
| Zone analysis | Center / perimeter / corner classification |
| Pair behaviors | Proximity subtypes, approach, following |
| Proximity tracking | Inter-animal distance bouts and cumulative time |

Unselected modules are skipped entirely — no wasted compute. The Export Options dialog automatically disables outputs that depend on skipped modules.

### Kinematics

- Per-node velocity, speed, heading, acceleration, and jerk
- Smoothed differentiation to reduce noise amplification
- Body-scale normalization with DSR, the median inter-hip distance

### Behavior and Social Metrics

- Single-animal states: stationary, walking, running, turning, directional reversal
- Pairwise interactions: NoseNose, NoseHead, NoseBody, NoseRear, Contact, CoOriented, AntiOriented, RelPos, Engaged, Disengaged
- Second-order compound social behaviors computed at export time
- Engagement Index, Reciprocity Index, and Retreat Index
- Time-binned pair and animal features at 0.25 s and 1 s resolution

### Interface

- Real-time overlay during playback with live metrics panel
- Data inspector popup for frame-by-frame review of all computed values
- Non-blocking processing and export through Qt worker threads
- Analysis cache shared between Process and Export — kinematics and behaviors computed once

---

## Outputs

An export produces up to five output types depending on which options are selected:

| File | Contents |
|---|---|
| `{name}.xlsx` | Tracking Data, Zone Summary, Session Info, 1st/2nd Order Behaviors, Behavior Summary, Engagement Indices, Animal Features, Pair Features |
| `{name}_binned.xlsx` | Animal 0.25s, Pair 0.25s, Engagement Indices 0.25s, Animal 1s, Pair 1s |
| `{name}_key_metrics.xlsx` | Key Metrics (session summary), Proximity & Orientation (per-second inter-animal distances and heading; 2-animal sessions only) |
| `{name}_graphs_*.pdf` | Heatmaps, cascade (speed/accel/jerk), distance, feature oncoplot, synchrony oncoplot, feature-vs-distance |
| `{name}_rf_*.csv/pdf` | Random Forest bout analysis results and plots (requires scikit-learn) |

Export options are automatically greyed out for any outputs that depend on a module that was not selected during processing.

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
| Graphs | matplotlib (optional) |
| ML analysis | scikit-learn (optional) |

---

## Installation

### Requirements

- Python 3.8+

### Install Dependencies

```bash
pip install PySide6 numpy scipy pandas openpyxl h5py opencv-python pykalman
```

Optional (for graph PDFs):
```bash
pip install matplotlib
```

Optional (for Random Forest analysis):
```bash
pip install scikit-learn
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
5. Click **Process** — a **Processing Options** dialog appears. Choose which analysis modules to run, then click **Analyze**.
6. Use **Inspect** to review per-frame values in the data popup.
7. Click **Export** — an **Export Options** dialog appears with outputs pre-gated by what was processed. Choose outputs and destination, then click **Export**.

Processing options are remembered across re-runs within the same session.

---

## Project Structure

```text
main.py             entry point
main_window.py      application window shell
run_popup.py        main analysis window: ROI, playback, processing dialog, export worker
preprocessing.py    gap filling, smoothing, kinematics
behaviors.py        1st-order and pairwise behavior detection
features.py         animal/pair feature extraction, proximity helpers, bout detection
binned_export.py    time-binned aggregation and Excel export
graph_export.py     PDF graph generation (heatmaps, cascade, oncoplots, distance)
sleap_loader.py     SLEAP .h5 loader with layout auto-detection
roi_view.py         interactive ROI drawing widget
```

---

## Technical Notes

- The internal `tracks` convention is `(n_frames, 2, n_nodes, n_tracks)`, where axis 1 stores x/y coordinates.
- `frame_map` links video frame numbers to SLEAP data rows so exports stay aligned to the original recording.
- DSR, the median inter-hip distance, acts as a body-scale reference for proximity thresholds.
- Heavy processing runs away from the Qt main thread through `QThread` workers.
- The analysis cache is computed once during processing and reused directly by the export worker — kinematics, behaviors, and feature arrays are never computed twice per session.
- The first 5 seconds of each session are trimmed at processing time to exclude hand-placement noise; the trimmed tracks and adjusted frame map are stored in the cache.

---

## Testing

A suite of headless tests verifies core pipeline correctness without requiring a display or real SLEAP file.

| Test file | What it covers |
|---|---|
| `test_key_metrics.py` | 119 ground-truth checks across 8 pipeline sections: distance/proximity, speed rolling, body heading, zone labeling, bout detection, cumulative tally, behavior states, MetricsPanel integration |
| `test_export_dialog.py` | ExportOptionsDialog init, proc\_opts gating, master/group checkbox sync, ProcessingOptionsDialog defaults |
| `test_conversions.py` | Unit conversion correctness (px/cm, speed, distance) |
| `test_oncoplot.py` | Oncoplot binning, aggregation, and normalization with hand-calculable synthetic data |

```bash
python test_key_metrics.py
python test_export_dialog.py
python test_conversions.py
python test_oncoplot.py
```

All tests exit with code `0` on success. Every check prints `PASS`/`FAIL` with `got=`/`expected=` evidence on failure so failures are self-documenting without reading the test source.

---

## Status

Active undergraduate research software project for computational behavioral analysis.

---

## Author

Built by [Abdullah Jawwad Yousafi](https://github.com/AbdullahJawwad2005).
