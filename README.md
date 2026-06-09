# STAR Analyzer

**STAR Analyzer** is a desktop application for processing and analyzing multi-animal behavioral tracking data from [SLEAP](https://sleap.ai/) pose estimation. It takes raw SLEAP `.h5` output and a paired video file, lets you calibrate an arena ROI, and exports a comprehensive set of kinematics, 1st-order behaviors, social interaction metrics, and time-binned features to Excel.

---

## Features

- **Video + SLEAP integration** — Load any video alongside its SLEAP `.h5` file. Supports both standard and transposed SLEAP export layouts, including sparse frame exports.
- **ROI calibration** — Draw a square arena ROI directly on the video frame and set the arena size in cm to convert all pixel measurements to real-world units.
- **Gap filling & smoothing** — Three-tier NaN gap filling (PCHIP spline → Kalman smoother → linear fallback) followed by Savitzky-Golay smoothing before kinematics are computed.
- **Kinematics** — Per-node velocity, speed, heading, acceleration, and jerk via Savitzky-Golay differentiation (no noise amplification from finite differences).
- **1st-order behavior detection**
  - *Single-animal:* stationary, walking, running, turning, directional reversal
  - *Pairwise:* NoseNose, NoseHead, NoseBody, NoseRear, Contact, CoOriented, AntiOriented, RelPos, Engaged, Disengaged, engagement/disengage speeds
- **Social interaction indices** — Engagement Index (EI), Reciprocity Index (RI), and Retreat Index (RTI) computed per pair, per minute, and cumulatively across the session.
- **Feature extraction** — Body shape (PCA elongation, eccentricity, convex-hull circularity/compactness), path efficiency, hourglass area, node-pair distances and angular motion, ROI distances, position entropy, visual/auditory scope, inter-animal covariance.
- **Time-binned export** — A separate `*_binned.xlsx` workbook with all features aggregated at 0.25 s and 1 s resolution using statistically appropriate rules (circular mean for angles, p90 for velocities, within-bin Pearson covariance for position synchrony).
- **Live overlay** — Real-time per-frame behavior state, speed, and heading drawn on the video during playback.
- **Data inspector popup** — Browse kinematics and behavior values frame-by-frame while the video plays.
- **Non-blocking UI** — Processing and export each run in a background QThread so the interface stays responsive throughout.

---

## Output

Running an export produces two Excel workbooks:

| File | Sheets |
|------|--------|
| `{name}.xlsx` | Tracking Data, Zone Summary, Session Info, 1st Order Behaviors, Behavior Summary, Engagement Indices, Animal Features, Pair Features |
| `{name}_binned.xlsx` | Animal 0.25s, Pair 0.25s, Eng Indices 0.25s, Animal 1s, Pair 1s |

---

## Installation

**Requirements:** Python 3.8+

```bash
pip install PySide6 numpy scipy pandas openpyxl h5py opencv-python pykalman
```

**Run:**

```bash
python main.py
```

---

## Usage

1. **Open** — Click *Open Video* and *Load SLEAP* to load your files.
2. **Draw ROI** — Click and drag on the video to draw a square arena boundary.
3. **Calibrate** — Set the arena size (cm) and border strip width in the controls.
4. **Process** — Click *Process* to fill gaps, smooth, and compute kinematics. A progress bar tracks completion.
5. **Inspect** *(optional)* — Toggle *Inspect* to open the data popup and browse values frame-by-frame.
6. **Export** — Click *Export*, choose a save location, and both workbooks are written automatically.

---

## Project Structure

```
main.py             Entry point
main_window.py      Application window shell
run_popup.py        Main analysis window — ROI, playback, export worker
preprocessing.py    Gap filling, smoothing, kinematics
behaviors.py        1st-order behavior detection
features.py         Primitive & derivative feature extraction
binned_export.py    Time-binned aggregation and Excel write
sleap_loader.py     SLEAP .h5 loader with layout auto-detection
roi_view.py         Interactive ROI drawing widget
```

---

## Architecture Notes

- All heavy computation runs off the Qt main thread via `QThread` workers — one for processing (with a live progress bar) and one for export.
- The `tracks` array convention used throughout is `(n_frames, 2, n_nodes, n_tracks)` where axis-1 index 0 = x, 1 = y.
- `frame_map` (`{video_frame_idx → sleap_data_idx}`) bridges video frame numbers and SLEAP data indices so all exports align to the correct video timestamps.
- DSR (Dynamic Sniff Range) — the median inter-hip distance — is used as a body-scale reference so all proximity thresholds are species- and camera-independent.
- See `CODE_REVIEW_GUIDE.md` in the repository for a full technical deep-dive into every module, data convention, and design decision.

---

## License

For internal/research use. Contact the repository owner for licensing information.
