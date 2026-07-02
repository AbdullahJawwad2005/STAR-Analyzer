# STAR Analyzer — Code Review Guide

**Purpose:** This document is a complete technical reference for reviewing the STAR Analyzer
codebase. It covers architecture, data conventions, every module's design decisions, known
limitations, and things to look closely at during a proofread. Intended audience: a reviewer
who has not seen this code before and needs to evaluate correctness, robustness, and clarity.

---

## 1. Project Overview

STAR Analyzer is a PySide6 desktop application that:
1. Loads a video file and a SLEAP multi-animal pose estimation `.h5` file.
2. Lets the user draw an ROI (region of interest) on the video and set arena calibration.
3. Processes the raw SLEAP tracks (gap-filling, smoothing, kinematics).
4. Computes 1st-order behaviors (locomotion states, social proximity, engagement).
5. Computes a large feature set (per-animal and pairwise).
6. Exports results to `.xlsx` — a main workbook and a separate time-binned workbook.

All computation is non-blocking: the GUI stays responsive via Qt `QThread` workers.

---

## 2. File Map

```
run_popup.py      — Qt UI, two QThread workers, export orchestration
preprocessing.py  — gap-filling, smoothing, kinematics (pure numpy/scipy)
behaviors.py      — 1st-order behavior detection (pure numpy)
features.py       — primitive & derivative feature extraction (pure numpy)
binned_export.py  — time-binned aggregation and Excel write
```

No file writes to disk or creates Qt objects except `run_popup.py` and `binned_export.py`.

---

## 3. Critical Data Conventions (read this first)

These conventions are used identically across all five files. Any confusion about axis order
is the most common source of bugs.

### 3.1 `tracks` array

```
shape: (n_frames, 2, n_nodes, n_tracks)
         axis-0: SLEAP frame index
         axis-1: coordinate  — 0 = x (horizontal), 1 = y (vertical)
         axis-2: node index  — 0..n_nodes-1
         axis-3: track index — 0..n_tracks-1
```

`NaN` means the animal was undetected at that frame/node. After `fill_and_smooth_tracks`,
NaNs are filled; the only remaining NaNs are structural (leading/trailing if the animal
was never detected, or fully unresolvable gaps).

### 3.2 `kin` dict

Output of `compute_kinematics`. Each value is shape `(n_frames, n_nodes, n_tracks)`:

| Key | Units |
|-----|-------|
| `vx`, `vy` | pixels/second |
| `speed` | pixels/second (magnitude) |
| `heading_deg` | degrees, −180..180 |
| `ax`, `ay` | pixels/second² |
| `accel` | pixels/second² (magnitude) |
| `jx`, `jy` | pixels/second³ |
| `jerk` | pixels/second³ (magnitude) |

### 3.3 `frame_map`

```python
frame_map: dict[int, int]   # video_frame_idx → sleap_data_idx
```

**Critical:** video frame indices and SLEAP data indices are NOT the same. SLEAP only stores
frames where at least one animal was detected. All indexing into `tracks` and `kin` must use
`sleap_data_idx`, not `video_frame_idx`. The export loops always do:

```python
for vid_frame, sleap_idx in sorted(frame_map.items()):
    ...
    pts = tracks[sleap_idx, ...]   # ← use sleap_idx, not vid_frame
```

### 3.4 `node_names` / `track_names`

Plain Python `list[str]`. Index `n` in `node_names` corresponds to axis-2 index `n` in
`tracks`. Index `t` in `track_names` corresponds to axis-3 index `t` in `tracks`.

### 3.5 `single_beh` dict

From `compute_single_animal`. Each value is shape `(n_frames, n_tracks)`, dtype `int8`.
Keys: `'stationary'`, `'walking'`, `'running'`, `'turning'`, `'dir_reversal'`.

### 3.6 `pair_beh` dict

From `compute_pairwise`. Keys follow the pattern `'tA_tB/BehaviorName'` (e.g.
`'t0_t1/NoseNose'`). Values are shape `(n_frames,)`. Most are `int8` (0/1); engagement speed
columns (`EngageSpeed_A`, `EngageOnsetSpeed_A`, etc.) are `float64` with `NaN` for non-engaged
frames.

### 3.7 `track_arrays` / `pair_arrays`

From `precompute_feature_arrays` in `features.py`.

```
track_arrays: dict[int, dict[str, np.ndarray]]
              track_arrays[t]['feature_name'] → (n_frames,) array

pair_arrays:  dict[str, np.ndarray]
              'tA_tB/feature_name' → (n_frames,) array
```

All arrays are indexed by **sleap frame index** (same as `tracks` axis-0).

---

## 4. `preprocessing.py`

### What it does
Gap-fills and smooths raw SLEAP coordinate traces, then computes kinematics.

### Key design decisions

**`hybrid_convergent_fill` — three-tier gap strategy:**
1. **PCHIP spline** for short gaps (≤ `fps × 0.25` frames, ~6 frames at 24 fps).
   PCHIP is chosen over standard cubic spline because it is monotonicity-preserving
   (no oscillation between control points), so it can't produce physically implausible
   position overshoots.
2. **Kalman smoother** for longer gaps. Uses a constant-velocity 2-state model
   `[position, velocity]`. The physical prior (animal keeps moving in roughly the same
   direction) is more defensible for longer occlusions than extrapolating a polynomial.
3. **Linear interpolation + constant-pad** as a last-resort safety net for any NaNs
   still remaining (e.g. at the very start or end of a trace where Kalman has no context).

**`smooth_sleap_allnodes` — median before Savitzky-Golay:**
The median filter (window 3) runs first to kill single-frame spike outliers before SG sees
the data. This is important because SG is a least-squares polynomial fit — a single spike
corrupts the fit across the entire SG window.

**`compute_kinematics` — SG differentiation, not finite differences:**
`savgol_filter(..., deriv=1, delta=dt)` analytically differentiates the local polynomial
fit. This is much better than `np.gradient` (finite differences), which divides adjacent
position differences by `dt` and amplifies noise by `1/dt` (≈ 24× at 24 fps).

**`fill_and_smooth_tracks` — axis transposition:**
The app-wide layout is `(n_frames, 2, n_nodes, n_tracks)` (axis-1 = x/y). The fill/smooth
helpers expect `(n_frames, n_nodes, 2)` (axis-2 = x/y). The conversion is:
```python
coords = processed[:, :, :, track_idx].transpose(0, 2, 1)   # → (n_frames, n_nodes, 2)
# ... process ...
processed[:, :, :, track_idx] = coords.transpose(0, 2, 1)   # back to (n_frames, 2, n_nodes)
```

### Things to check
- The Kalman `transition_covariance` (1e-3) and `observation_covariance` (1e-2) are
  hardcoded. These ratios determine how much the smoother trusts the model vs. the data.
  If animals move very fast, 1e-3 may be too tight.
- `sg_win=5, poly=2` in `fill_and_smooth_tracks` vs `sg_win=11, poly=3` in
  `compute_kinematics` — the smoothing pass uses a short window; the kinematics pass
  uses a longer window. These are different calls on different data. This is intentional
  but worth confirming no confusion between the two.

---

## 5. `behaviors.py`

### What it does
Detects five single-animal locomotor states and up to 24 pairwise social behavior arrays,
then computes session-level summary statistics and engagement indices.

### DSR (Dynamic Sniff Range)

DSR is the body-scale reference used to make all proximity thresholds scale-independent:

```
DSR = median hip-to-hip distance (filtered to 20th–80th percentile of valid frames)
      averaged across all tracks
```

If hip nodes are absent from the SLEAP model, `_fallback_dsr` returns 10% of the
bounding-box diagonal of all node positions. **This fallback is rough** (order-of-magnitude
only) — results will be less calibrated if the SLEAP model has no hip nodes.

All proximity thresholds are expressed as `multiplier × DSR`:
- `NoseNose`: `< 0.5 × DSR`
- `NoseHead_*`, `NoseBody_*`, `NoseRear_*`: `< 0.7 × DSR`
- `Contact` (any node-pair): `< 0.25 × DSR`
- `Engaged` body distance: `< 3.0 × DSR`

### Single-animal states (`compute_single_animal`)

**Speed thresholds are session-relative percentiles, not absolute values:**

| State | Threshold |
|-------|-----------|
| Stationary | CM speed < p15 of this animal's session speeds |
| Walking | p15 ≤ CM speed ≤ p75 |
| Running | CM speed > p75 |

Consequence: even a completely inactive animal will have 15% of its frames classified as
"stationary" and 25% as "running." This is intentional — it captures locomotor state relative
to the animal's own baseline rather than requiring hard-coded speed values that would need
recalibration per camera/arena/species.

**Turning:** `|angular_velocity| > 30 deg/s`

**Directional reversal:** sign flip in angular velocity within a ±2-frame window, with
`|ang_vel| > 20 deg/s` on both sides.

### Pairwise behaviors (`compute_pairwise`)

**Engagement criteria (all three must hold simultaneously):**
1. Body-centre distance < 3 × DSR
2. Face error A < 60° (animal A's heading points toward B within a 60° cone)
3. Face error B < 60° (animal B's heading points toward A within a 60° cone)

After raw frame detection, `_fill_short_gaps(..., 0.3 × fps)` merges engagement runs
separated by ≤ 0.3 s (prevents single-frame look-away flickers from fragmenting a bout).

**Disengagement criteria (all three must hold simultaneously):**
1. `engaged == 0` (not currently engaged)
2. `recent_sum > 0` (was engaged in the last 0.75 s)
3. `dist_inc == True` (inter-animal distance actively increasing by > 0.05 × DSR per frame)

The distance-increase criterion is the key directional component — it distinguishes true
spatial separation from an animal that briefly stopped facing the other but stayed nearby.

**`RelPos_A` / `RelPos_B`:** Quadrant of animal B in animal A's heading frame. Uses a
2D projection decomposition:
- `long = dot(rel_vec, heading_unit)` — longitudinal (forward/behind)
- `lat  = cross(heading_unit, rel_vec)` — lateral (left/right)
- The larger absolute component determines the quadrant.

### Engagement indices

**Engagement Index (EI_A):** Fraction of bouts initiated by animal A.
```
EI_A = n_bouts_initiated_by_A / n_total_bouts
```

**Reciprocity Index (RI_A):** Fraction of bouts where A initiated after B had just initiated
the previous bout (captures turn-taking / mutual instigation).
```
RI_A = n_bouts_A_initiated_after_previous_B_initiation / n_total_bouts
```

**Retreat Index (RTI_A):** Fraction of bouts where B initiated, the bout was short
(≤ 3 s), and A was the disengager (A retreated from B's approach).
```
RTI_A = n_short_B_initiated_bouts_where_A_disengaged / n_total_bouts
```

Bout initiator is determined by comparing approach angles in a 0.3 s pre-bout window:
the animal with the smaller approach angle (more directly facing the other) is the initiator.
Disengager is determined by the 0.5 s post-bout window: the animal with the larger
approach angle (more turned away) is the disengager.

### Things to check
- `angular_velocity` in `behaviors.py` uses `np.gradient` (finite differences) for heading
  in `_smooth_heading_deg`. **This is different from `compute_kinematics` which uses SG.**
  The behaviors module uses `np.gradient` for a fast internal estimate; kinematics uses SG
  for the exported values. This inconsistency is intentional but worth flagging.
- `_fill_short_gaps` iterates with a Python `while` loop over every frame — for long sessions
  this is O(n_frames). For 10-minute sessions at 30 fps (18,000 frames) this is fine; for
  multi-hour recordings it could be slow.
- `_mean_approach_angle_arr` returns `90.0` (neutral) when data is unavailable. This defaults
  the initiator to animal A (tie-breaking: `aa_A <= aa_B` → A initiates). The choice of A is
  arbitrary but not directionally biased.
- The directional reversal detection (`dir_reversal`) has a Python for-loop over all frames
  (`for f in range(half, n_frames - half)`). For large sessions this is the slowest part of
  `compute_single_animal`. It could be vectorised but currently isn't.
- `CoOriented` uses `rel_ang < 30` where `rel_ang` is the absolute angle difference wrapped to
  [0°, 180°]. Both 0° (same direction) and near-zero anti-parallel angles would satisfy this
  — but anti-parallel is already excluded by `rel_ang < 30` given the symmetry of the wrapping.
  Worth verifying the wrap logic: `rel_ang = np.abs(hdg_A - hdg_B) % 360; rel_ang = np.minimum(rel_ang, 360 - rel_ang)` — this correctly maps to [0°, 180°].

---

## 6. `features.py`

### What it does
Computes a large set of kinematic and geometric features for each animal and each pair,
indexed by SLEAP frame index.

### Key features

**Per-animal (`precompute_feature_arrays`, per track `t`):**

| Feature group | What is computed |
|---------------|-----------------|
| Raw coordinates | `{node}_x`, `{node}_y` per node |
| Kinematics | `{node}_speed`, `{node}_vx/vy`, `{node}_accel`, `{node}_jerk` per node |
| Cumulative displacement | `{node}_total_disp` — cumulative path length per node |
| Node-pair geometry | For every unique (i,j) node pair: `dist`, `angle`, `ang_mot` |
| Body shape (PCA) | `elongation`, `eccentricity` from 2×2 covariance of node cloud |
| Body shape (convex hull) | `circularity`, `compactness` (requires scipy) |
| Hourglass area | Area of nose–hipL–hipR triangle |
| Whole-body curvature | `|angular_velocity(CM_heading)|` |
| Path efficiency | Rolling 1-second straight-line / cumulative-path ratio (0–1) |
| ROI distances | `dist_roi_center`, `dist_roi_boundary` |
| Position entropy | Session-level Shannon entropy of spatial occupancy (single scalar broadcast to all frames) |

**Node-pair angular motion (`ang_mot`):**
`ang_mot` is the frame-to-frame circular difference of the inter-node angle. It captures
limb swings and local joint rotations at finer resolution than the whole-body CM heading.

**Shape descriptors:**
- `elongation = major_axis / minor_axis` (from PCA eigenvalues of node cloud)
- `eccentricity = sqrt(1 - (minor/major)²)` — standard ellipse eccentricity (0=circle, 1=line)
- `circularity = 4π·A / P²` — ranges 0..1; 1.0 = perfect circle
- `compactness = P² / (4π·A)` — reciprocal of circularity; minimum 1.0 for circle

**Position entropy:**
A **session-level scalar**, not time-varying. The ROI is divided into a 10×10 grid; Shannon
entropy of the occupancy histogram is normalised by `log₂(100)`. The single scalar is
broadcast to all frames. Every row in the export and every bin in the binned export will
carry the same value for a given track. This is intentional — it characterises space-use
diversity for the whole session.

**Pairwise (`_pair_features`):**

| Feature | Description |
|---------|-------------|
| `inter_animal_dist` | CM-to-CM Euclidean distance |
| `inter_animal_displacement` | Frame-to-frame change in inter-animal distance |
| `pos_covariance_x/y` | Session-centred frame products (NOT within-bin covariance) |
| `pos_correlation_x/y` | Session-centred frame products / session std product |
| `approach_angle_A/B` | Angle between heading and inter-animal vector (0° = facing directly) |
| `visual_scope_A/B` | `Binocular < 20°`, `Monocular < 120°`, `None` otherwise |
| `auditory_scope_A/B` | `Binaural < 60°`, `Monaural < 150°`, `Rear` otherwise |

**Important note on `pos_covariance_x/y` and `pos_correlation_x/y`:**
These frame-level columns are computed as `(xA - mean_xA) * (xB - mean_xB)` where the means
are session-wide. Averaged over the whole session this gives Pearson covariance, but at the
frame level it only reflects long-range positional drift (both animals spending time on the
same side of the arena), not moment-to-moment co-movement. The binned export replaces these
with proper within-bin Pearson statistics. These frame-level versions are kept for reference.

**`_precomputed` parameter:**
`build_feature_dataframes` accepts an optional `_precomputed=(track_arrays, pair_arrays)`.
If provided it skips recomputing. This is used in `run_popup.py` where both the main export
and the binned export need the same arrays — computing once saves several seconds.

**`build_key_metrics_df`:**
Produces a long-format summary DataFrame (`Category, Metric, Subject, Value, Unit`) covering
per-animal locomotion stats (distance, speed, acceleration, immobility, zone times) and
per-pair proximity/contact times and mean approach angles. Written to the dedicated
`{stem}_key_metrics.xlsx` workbook (not to the main workbook).

**`build_proximity_orientation_df`:**
Produces a per-second table for the first animal pair. Bins `frame_map` into integer-second
buckets (`int(vid_frame // fps)`) and computes:

| Column | Description |
|--------|-------------|
| `Time(s)` | Integer second bin |
| `Within_3cm` | 1 if mean centroid distance that second < 3 cm, else 0 |
| `Heading_Angle_deg` | Mean mutual facing angle — `abs(((hdgA − hdgB) + 180) % 360 − 180)`, range 0°–180° |
| `{node}_dist_cm` | Mean same-node pair distance (cm) for each canonical node present |

Canonical column order: `nose, ear_l, ear_r, body, hip_l, hip_r, tail`. Absent nodes are
omitted. Only called when `n_tracks >= 2`; written as sheet 2 of `{stem}_key_metrics.xlsx`.

### Things to check
- `_shape_features_per_track` has a Python `for f in range(n_frames)` loop doing PCA per
  frame. This is O(n_frames × n_nodes²) and is the slowest part of `features.py`. For 10,000
  frames and 12 nodes it is still manageable; for larger sessions it may need vectorising.
- `_position_entropy` being a constant broadcast to all frames may confuse analysts who see
  the same value repeating in every row. The annotation in the code explains this, but it
  should probably also be noted in the Excel column header.
- `_roi_distances`: `dist_roi_boundary` returns 0 for positions on or outside the ROI
  boundary (not negative). Worth knowing if downstream users expect it to be negative for
  out-of-arena positions.
- The `visual_scope` and `auditory_scope` thresholds (20°, 120°, 60°, 150°) are
  hardcoded with no citation. These should be verified against the behavioural literature
  for the target species.

---

## 7. `binned_export.py`

### What it does
Takes per-frame arrays from `features.py` and `behaviors.py` and aggregates them into
0.25 s and 1.0 s time bins, writing a separate `*_binned.xlsx` workbook.

### Bin-map construction (`_build_bin_map`)

```
bin_idx = floor(video_frame_idx / fps / bin_size_s)
```

Multiple SLEAP frame indices may fall into the same bin (at 30 fps, `bin_size_s=0.25`
gives ~7–8 frames per bin). The dict maps `bin_idx → [sleap_idx_0, sleap_idx_1, ...]`.

### Aggregation rules (`_agg_025`)

| Category | Output columns | Method |
|----------|---------------|--------|
| `position` | `_mean` | arithmetic mean |
| `distance` | `_mean`, `_median` | arithmetic |
| `angular` | `_cmean` | circular mean via arctan2 |
| `velocity` / `accel` | `_median`, `_p90` | percentiles |
| `jerk` | `_absmedian`, `_absp90` | absolute values of percentiles |
| `binary` | `_prop` | fraction of ALL frames in bin (denominator includes NaN) |
| `shape` / `other` | `_mean` | arithmetic mean |
| `engage_spd` | `_median` | median of non-NaN values |

For `binary`, the denominator is the **total** frames in the bin (not just detected frames).
This means bins near occlusion gaps appear deflated. The alternative (denominate by
detected-only frames) would make bins near occlusions appear artificially high. The current
choice is conservative.

### Column classification (`_classify_feat`)

Pure string-matching. Check order matters:
1. `_x`/`_y` suffix → `position` (before anything else to avoid false angular matches)
2. `_angle`, `ang_mot`, `heading`, `curvature`, `approach_angle` → `angular`
3. `jerk` → `jerk` (before `speed` to avoid false velocity matches)
4. `speed`, `_vx`, `_vy`, `displacement` → `velocity`
5. `accel` → `accel`
6. Named binary behaviors → `binary`
7. Named engagement speed patterns → `engage_spd` (**dead code** — see below)
8. Shape/geometry names → `shape`
9. `dist`, `area`, `covar`, `corr` → `distance`
10. `relpos`, `visual_scope`, `auditory_scope` → `categorical` (skipped)
11. Everything else → `other`

**Dead code in `_classify_feat`:** The `engage_spd` category (step 7) is never reached in
the current architecture. Engagement speed columns come from `pair_beh` (behaviors.py) and
are aggregated by a dedicated hardcoded loop in `build_025s_bins`, not via `_classify_feat`.
The category is retained for forward-compatibility in case engagement speeds are ever moved
into `pair_arrays`. This is annotated in the code.

### Within-bin Pearson covariance (`_bin_cov_corr`)

Replaces the session-centred frame products from `features.py` with proper within-bin
statistics using `np.cov(...)[0, 1]` (ddof=1). This captures local moment-to-moment
co-movement within each 0.25 s window rather than session-wide drift.

### Engagement at bin resolution

A bin is classified as "engaged" if `> 50%` of its frames are engaged. This conservative
threshold prevents single frames of engagement from inflating the bout count. Bouts are
then re-detected in the binary bin sequence; `_bouts_with_initiator_binned` attributes
initiator/disengager using pre/post windows measured in bins (default: 2 pre-bins, 3
post-bins) mapped back to actual SLEAP frames for the approach-angle calculation.

**Hardcoded coupling with `behaviors.py`:**
The proximity behavior list (`NoseNose`, `NoseHead_AtoB`, ..., `Contact`, `CoOriented`,
`AntiOriented`) is hardcoded in `build_025s_bins`. If `compute_pairwise` in `behaviors.py`
ever adds new proximity behavior keys, they must be manually added to this list too.
Same for the engagement speed column names (`EngageSpeed_A`, ..., `DisengageOnsetSpeed_B`).

### 1-second bins (`build_1s_from_025`)

Re-aggregates four 0.25 s sub-bins per 1 s bin:
- `*_cmean` columns → circular mean of the four sub-bin circular means
- Everything else → arithmetic mean
- `*_prop` columns and raw engagement columns → **skipped** (omitted from 1 s output)

Binary proportions are skipped at 1 s because averaging four 0.25 s proportions is not the
same as recomputing the proportion over all 1 s frames. Re-detection at 1 s resolution is
beyond the scope of this derived table.

### Output sheets (`write_binned_xlsx`)

| Sheet | Contents |
|-------|----------|
| `Animal 0.25s` | Per-track features, 0.25 s bins |
| `Pair 0.25s` | Pairwise features, 0.25 s bins |
| `Eng Indices 0.25s` | EI / RI / RTI per pair, per minute, cumulative, and full-video |
| `Animal 1s` | Per-track features, 1 s bins (no binary props) |
| `Pair 1s` | Pairwise features, 1 s bins |

### Things to check
- `_bouts_with_initiator_binned` uses the SPAN of pre/post frames (min to max+1) rather
  than the exact set. For sessions with sparse `frame_map` coverage (dropped frames),
  gap frames within that span will be passed to `_mean_approach_angle_arr` but are silently
  filtered by its NaN mask. This is correct but not immediately obvious from reading the code.
- `build_1s_from_025` uses `bins_per_s = round(1.0 / bin_size_s)` which gives 4 for 0.25 s.
  If `bin_size_s` is changed to a non-divisor of 1.0 (e.g. 0.3 s), the 1 s aggregation will
  be incorrect. There is a code comment warning about this but no runtime check.
- The engagement index export in `write_binned_xlsx` uses `fps=bin_fps` (= 4 for 0.25 s bins)
  when calling `_indices_from_bouts`. The retreat window (3.0 s default) is thereby measured
  in bin-units (`3.0 × 4 = 12 bins`). This is internally consistent but means "3 seconds"
  at bin resolution may not match "3 seconds" computed at frame resolution due to the
  50%-threshold bin classification.

---

## 8. `run_popup.py`

### What it does
The single Qt window. Manages video playback, ROI drawing, data loading, overlay rendering,
the data inspector popup, two QThread workers, and the processing/export option dialogs.

### `ProcessingOptionsDialog`

Shown immediately before each Process run. Presents checkboxes:

| Option | Key | What it gates |
|--------|-----|--------------|
| Single-animal behaviors | `proc_single_beh` | `compute_single_animal` |
| Feature arrays | `proc_features` | `precompute_feature_arrays` |
| Zone analysis | `proc_zones` | Center/Perimeter/Corner classification |
| Pair / social behaviors | `proc_pair_beh` | `compute_pairwise` + `compute_second_order` |
| Proximity tracking | `proc_proximity` | inter-animal distance bout precomputes |

The pair group is **auto-disabled** when `n_tracks < 2`. The dialog result is stored in
`self._proc_opts` and persists across re-runs within the same session. Kinematics
(`compute_kinematics`) are always computed regardless of options.

### `ExportOptionsDialog`

Receives `proc_opts` from `_analysis_cache['proc_opts']` (or `None`). Uses a `_PROC_GATES`
mapping to automatically uncheck and disable export outputs whose upstream module was
not computed. For example, if `proc_pair_beh=False`, pair behavior sheets and pair graph
PDFs are greyed out.

### Two QThread workers

**`_ProcessWorker`** — runs after the user clicks "Process":
- Takes raw `sleap_data` dict and `fps`.
- Calls `fill_and_smooth_tracks` with a progress callback (emits 0–100 via `progress` signal).
- Emits `finished(processed_data_dict)` when done.

**`_ExportWorker`** — runs after the user clicks "Export" and chooses a file path:
- Takes `analysis_cache` (the dict built by `_precompute_analysis`) plus calibration/ROI args.
- **Does NOT recompute kinematics, behaviors, or features.** Pulls everything from cache:
  `tracks`, `frame_map`, `node_names`, `track_names`, `kin`, `single_beh`, `pair_beh`,
  `track_feat`, `pair_feat`, `zone_label`, `proc_opts`.
- Only new computation in `run()` is `compute_second_order` (export-only compound behaviors,
  called only if `proc_pair_beh=True`).
- Runs in this order: pull from cache → `compute_second_order` (if applicable) →
  `compute_behavior_summary` → `build_feature_dataframes` (reusing cached arrays) →
  build main data table → build behavior DataFrame → write main `.xlsx` →
  `build_key_metrics_df` + `build_proximity_orientation_df` → write `_key_metrics.xlsx` →
  `write_binned_xlsx`.
- Emits `finished(success_msg_str)` or `error(traceback_str)`.
- Note: the `status` signal is defined but not currently connected to any UI widget.

**Important:** The Qt file-save dialog must run on the main thread (Qt restriction). In
`_run_export`, the dialog is shown before the worker is created. The path is then passed
into the worker constructor.

### Worker lifecycle (same pattern for both workers)

```python
self._worker = _SomeWorker(...)
self._thread = QThread()
self._worker.moveToThread(self._thread)
self._thread.started.connect(self._worker.run)
self._worker.finished.connect(self._on_done)
self._worker.error.connect(self._on_error)
self._worker.finished.connect(self._thread.quit)
self._worker.finished.connect(self._worker.deleteLater)
self._worker.error.connect(self._thread.quit)
self._worker.error.connect(self._worker.deleteLater)
self._thread.finished.connect(self._thread.deleteLater)
self._thread.start()
```

Note: both `finished` and `error` paths quit the thread and delete the worker. This is
the correct pattern for preventing Qt memory leaks.

### `_ExportWorker.run` computation order

1. Pull `tracks`, `kin`, `single_beh`, `pair_beh`, `track_feat`, `pair_feat`,
   `zone_label`, `proc_opts` from `self._cache`.
2. If `proc_pair_beh=True`: `compute_second_order(...)` — compound social behaviors;
   merged into `pair_beh` copy (does not mutate cache).
3. `compute_behavior_summary(...)` — session stats + windowed EI/RI/RTI.
4. If `track_feat` is non-empty and `proc_features=True`:
   `build_feature_dataframes(..., _precomputed=(track_feat, pair_feat))`.
   Otherwise sets `animal_feat_df = pair_feat_df = pd.DataFrame()`.
5. Python loop builds per-row main tracking DataFrame (kinematic values in cm units).
6. Python loop builds per-row behavior DataFrame.
7. `.xlsx` write with `pd.ExcelWriter`.
8. `build_key_metrics_df(...)` + optionally `build_proximity_orientation_df(...)` → write `_key_metrics.xlsx`.
9. `write_binned_xlsx(...)` — binned Excel workbook.
10. Emit `finished(msg)`.

### Zone classification

The ROI is divided into 9 zones:

```
C1 C2          W1
   W4 Open W2
C4 C3          W3
```

Where C = corner, W = wall strip, Open = interior. Each zone is a pixel rectangle:
```python
strip = strip_cm * px_per_cm
C1 = (rx0, ry0, rx0+strip, ry0+strip)  # top-left corner
C2 = (rx1-strip, ry0, rx1, ry0+strip)  # top-right corner
...
```

### `_precompute_analysis` vs `_run_export`

`_precompute_analysis(proc_opts=None)` is called after Process completes. It:
1. Trims the first 5 seconds from `tracks` and adjusts `frame_map`.
2. Always runs `compute_kinematics`.
3. Conditionally runs each module based on `proc_opts` flags.
4. Builds `self._analysis_cache` — a dict containing all computed results plus `proc_opts`.

`_ExportWorker` receives this cache directly and **reuses it without recomputing**.
The cache is the single source of truth for both the live overlay and the export pipeline.

`_analysis_cache` keys:

| Key | Type | Notes |
|-----|------|-------|
| `tracks` | `ndarray` | Trimmed (first 5 s removed) |
| `frame_map` | `dict` | Adjusted to trimmed indices |
| `node_names` | `list[str]` | |
| `track_names` | `list[str]` | |
| `kin` | `dict` | Output of `compute_kinematics` |
| `single_beh` | `dict` | Empty `{}` if `proc_single_beh=False` |
| `pair_beh` | `dict` | Empty `{}` if `proc_pair_beh=False` or `n_tracks < 2` |
| `track_feat` | `dict` | Empty `{}` if `proc_features=False` |
| `pair_feat` | `dict` | Empty `{}` if `proc_features=False` |
| `zone_label` | `dict` | Empty `{}` if `proc_zones=False` |
| `body_idx` | `int\|None` | Body-centre node index |
| `proc_opts` | `dict` | The resolved option flags |

### Things to check
- `_precompute_analysis` runs on the main thread (called from `_on_process_done`, a slot).
  For large sessions this could stutter the UI briefly. It is not wrapped in a QThread.
  This is a known limitation.
- The `_ExportWorker.status` signal is defined but not connected to any UI widget. The
  progress bar is in indeterminate (marquee) mode during export. If users want step-by-step
  status updates ("Computing behavior summary…", "Writing Excel…"), `status` would need to be
  connected to a `QLabel` in `_run_export`.
- `closeEvent` stops the timer and releases the cv2 capture. It does NOT explicitly stop
  or wait for a running export thread. If the window is closed mid-export, the thread will
  complete (it has no references back to the window after signal emission) but the result
  dialog will be emitted to a destroyed object. Qt's signal/slot mechanism handles
  this gracefully (deleted receiver is disconnected), but the file being written may be
  incomplete. Users should be warned not to close the window during export.
- `_analysis_cache` access is unguarded in `_draw_analysis_overlay` and `_DataPopup.refresh`.
  There is a `if self._analysis_cache is not None` guard before overlay/popup calls (covers
  the "never ran" case), but not the case where `_precompute_analysis` ran and populated
  only partial data due to an exception mid-run.
- `_DataPopup.setup()` guards `single_beh[bk]` with `if bk not in single_beh: continue`
  to handle the case where `proc_single_beh=False`. The same guard does **not** exist for
  pair behavior rows — if a pair behavior key is expected but absent (e.g. because
  `proc_pair_beh=False`), the popup will silently skip it only if `_DataPopup.setup()` also
  has a similar guard for pair keys. Verify this is consistent.
- If the user clicks Process a second time without changing `proc_opts`, the same options
  dialog re-appears. `_proc_opts` is preserved across runs so the checkboxes will be in
  the same state as the previous run. This is intentional and documented in the README.
- `_on_arena_cm_changed` calls `_precompute_analysis(proc_opts=getattr(self, '_proc_opts', None))`.
  If the user changes arena size before ever clicking Process, `_proc_opts` does not exist
  yet and `getattr` returns `None`, meaning all modules run. This is safe but means an
  arena-size-triggered recompute always runs all modules regardless of prior `_proc_opts`.

---

## 9. Cross-Cutting Issues and Design Trade-offs

### Issue 0: Selective processing and export gating (new)
`ProcessingOptionsDialog` allows skipping expensive modules. The `_PROC_GATES` dict in
`ExportOptionsDialog` maps each export checkbox to the `proc_opts` keys it depends on.
If a dependency was not computed, the export checkbox is unchecked and disabled.

**What to verify:** The `_PROC_GATES` mapping is hardcoded. If a new export option is added
to `ExportOptionsDialog`, its dependencies must be manually added to `_PROC_GATES`. There
is no runtime check that the mapping is complete.

**What changed:** Previously `_ExportWorker` recomputed kinematics + all behaviors +
all features from scratch on every export, independently from `_precompute_analysis`.
This is no longer the case — the cache is the single source of truth. Any change to
behavior or feature computation now only needs to be made in `_precompute_analysis`.

### Issue 1: Session-relative speed thresholds
Speed thresholds (`p15`, `p75` percentiles per animal) are computed from each animal's own
full-session speed distribution. A completely inactive animal will still have 15% of its
frames classified as "running." This is documented in the code and is a deliberate design
choice, but reviewers should flag whether this is appropriate for the target experiments
(e.g. if comparing across animals/sessions, session-relative thresholds make cross-animal
comparison difficult).

### Issue 2: Two heading implementations
- `preprocessing.compute_kinematics` uses Savitzky-Golay differentiation for heading.
- `behaviors._smooth_heading_deg` and `features._smooth_heading_from_pos` use `np.gradient`
  (finite differences).

The SG-based heading is used for the exported kinematic values. The np.gradient-based heading
is used internally for behavior detection (engagement face-error, RelPos, CoOriented). These
will differ slightly, especially at high-speed frames or frames near NaN gaps. The exported
"Heading (deg)" in the tracking data sheet differs from the heading implicitly used to
determine "Engaged." This is currently unaddressed.

### Issue 3: `position_entropy` as a constant per-frame column
Every frame in the `Animal Features` sheet will have the same `position_entropy` value for
a given track. This may confuse analysts. The column could instead appear only in the
`Behavior Summary` sheet as a session statistic.

### Issue 4: Hardcoded behavior list coupling
`binned_export.py` hardcodes the names of proximity behaviors and engagement speed columns
from `behaviors.py`. If `behaviors.py` adds new behaviors, `binned_export.py` must be
manually updated. There is no runtime check that the expected keys exist in `pair_beh`.
The code uses `if full_key in pair_beh:` guards, so missing keys silently produce missing
columns rather than errors.

### Issue 5: `_shape_features_per_track` Python loop
The per-frame PCA loop is the performance bottleneck in `features.py`. For sessions longer
than ~5 minutes at 30 fps (9,000+ frames) with many nodes, this loop is noticeably slow.
It could be vectorised using `np.linalg.eigvalsh` on a batch of covariance matrices.

### Issue 6: `build_1s_from_025` bin_size_s assumption
If `bin_size_s` is ever changed from 0.25, `build_1s_from_025` will produce incorrect 1 s
bins if the new value doesn't divide 1.0 evenly. There is no assertion or check.

---

## 10. Excel Output Structure

### Main workbook (`{name}.xlsx`)

| Sheet | Description | Rows |
|-------|-------------|------|
| Tracking Data | x/y positions, zone, kinematics per node per track per frame | n_video_frames × n_nodes × n_tracks |
| Zone Summary | Time in each zone per track | n_tracks × n_zones |
| Session Info | ROI, calibration, session stats | ~15 key-value rows |
| 1st Order Behaviors | Binary behavior flags per frame | n_video_frames |
| Behavior Summary | Session-level time in each state | n_tracks + n_pairs |
| Engagement Indices | EI/RI/RTI per minute, cumulative, full-video | n_pairs × n_windows × 3 |
| Animal Features | Full feature set per track per frame | n_video_frames × n_tracks |
| Pair Features | Full feature set per pair per frame | n_video_frames × n_pairs |

### Key metrics workbook (`{name}_key_metrics.xlsx`)

| Sheet | Description | Rows |
|-------|-------------|------|
| Key Metrics | Long-format summary: Category, Metric, Subject, Value, Unit | varies |
| Proximity & Orientation | Per-second inter-animal distances and heading (pairs only) | n_seconds |

`Proximity & Orientation` is only written when `n_tracks >= 2`. Single-animal sessions produce a key_metrics file with only the Key Metrics sheet.

### Binned workbook (`{name}_binned.xlsx`)

| Sheet | Rows |
|-------|------|
| Animal 0.25s | n_bins × n_tracks |
| Pair 0.25s | n_bins × n_pairs |
| Eng Indices 0.25s | n_pairs × n_windows |
| Animal 1s | (n_bins/4) × n_tracks |
| Pair 1s | (n_bins/4) × n_pairs |

---

## 11. Quick Reference: Function Signatures

```python
# preprocessing.py
hybrid_convergent_fill(trace, fps=24, pchip_time_s=0.25) → np.ndarray
smooth_sleap_allnodes(coords, med_win=3, sg_win=5, poly=3) → np.ndarray
compute_kinematics(tracks, fps, sg_win=11, sg_poly=3) → dict
fill_and_smooth_tracks(tracks, fps, med_win=3, sg_win=5, poly=2,
                        progress_callback=None) → np.ndarray

# behaviors.py
find_node_idx(node_names, *patterns) → int | None
compute_dsr(tracks, hip_l_idx, hip_r_idx) → float | None
compute_single_animal(tracks, kin, node_names, fps) → dict[str, (n_frames, n_tracks)]
compute_pairwise(tracks, node_names, fps, dsr=None) → dict[str, (n_frames,)]
compute_behavior_summary(single_beh, pair_beh, track_names, fps,
                          tracks=None, kin=None, frame_map=None,
                          node_names=None) → (pd.DataFrame, pd.DataFrame)

# features.py
precompute_feature_arrays(tracks, kin, node_names, fps, roi=None)
    → (track_arrays, pair_arrays)
build_feature_dataframes(tracks, kin, node_names, track_names, fps,
                          frame_map, roi=None, _precomputed=None)
    → (animal_df, pair_df)
build_key_metrics_df(tracks, kin, single_beh, pair_beh,
                     track_arrays, pair_arrays, frame_map,
                     zone_summary_df, node_names, track_names,
                     fps, px_per_cm)
    → pd.DataFrame
build_proximity_orientation_df(tracks, kin, frame_map, node_names,
                                track_names, fps, px_per_cm)
    → pd.DataFrame

# binned_export.py
build_025s_bins(track_arrays, pair_arrays, single_beh, pair_beh,
                tracks, node_names, track_names, fps, frame_map,
                kin=None, bin_size_s=0.25)
    → (animal_025, pair_025, eng_idx_025)
build_1s_from_025(animal_025, pair_025, bin_size_s=0.25)
    → (animal_1s, pair_1s)
write_binned_xlsx(track_arrays, pair_arrays, single_beh, pair_beh,
                  tracks, node_names, track_names, fps, frame_map,
                  kin, output_path) → None
```

---

## 12. Reviewer Checklist

Use this as a structured checklist when reading through the code:

### preprocessing.py
- [ ] Are the Kalman noise covariances (1e-3, 1e-2) reasonable for typical mouse pose data?
- [ ] Does `smooth_sleap_allnodes` get called before `hybrid_convergent_fill` or after?
      (Answer: after — fill first, then smooth)
- [ ] Can `sg_win > n_frames` ever reach `compute_kinematics`? (Guarded by min/max clamps)

### behaviors.py
- [ ] Is the `_fallback_dsr` reshape now correct? (`tracks.transpose(0, 2, 3, 1).reshape(-1, 2)`)
- [ ] Does the directional reversal detection correctly use a ±2 frame window?
- [ ] Are the face-error thresholds (60° for engagement) documented with a rationale?
- [ ] Is the 0.75 s disengagement lookback window appropriate?
- [ ] Is 3.0 s the right retreat window for the target species?
- [ ] Are EI/RI/RTI denominator choices (n_total bouts for all three) consistent?

### features.py
- [ ] Is `ang_mot` computed correctly as `_circular_diff_deg(angle)` (wraps through ±180°)?
- [ ] Is `_position_entropy` returning a correct normalised entropy? (`H / log2(n_bins²)`)
- [ ] Does `_path_efficiency` correctly handle the edge case where `path_in_win ≤ 1e-6`?
      (Returns 1.0 — "no movement = perfectly efficient" is a convention, not physical truth)
- [ ] Are the visual scope angular thresholds (20°, 120°) and auditory scope thresholds
      (60°, 150°) supported by literature for the target species?
- [ ] Does `build_feature_dataframes` correctly iterate `sorted(frame_map.items())` so that
      Animal Features rows are in chronological video-frame order?

### binned_export.py
- [ ] Does `_classify_feat` correctly classify all column names produced by `features.py`?
      (Test: are any columns falling through to `'other'` unintentionally?)
- [ ] Is the 50% engagement threshold per bin appropriate?
- [ ] Does `_bouts_with_initiator_binned` handle the edge case of a 1-bin bout correctly?
- [ ] Does `build_1s_from_025` produce the right 1 s bin start times? (`BinTime_1s(s)` is
      set from `_1s_bin` index directly, not from actual time — for bin_size_s=0.25 this
      gives 0, 1, 2, ... which are the 1 s bin *indices*, not times in seconds.
      **This may be a bug** — `BinTime_1s(s)` should probably be `_1s_bin * 1.0` to match
      the `BinTime(s)` convention in the 0.25 s sheets.)

### run_popup.py
- [ ] Are all `QThread` workers properly cleaned up (`deleteLater` on both `finished` and
      `error` paths)?
- [ ] Is there any risk of the user clicking Export twice quickly, creating two concurrent
      `_ExportWorker` instances? (The Export button is disabled during export, so no — but
      verify `_btn_export.setEnabled(False)` is called before `start()`)
- [ ] Does `closeEvent` need to explicitly stop the export thread?
- [ ] Is the `_analysis_cache` populated with valid data before `_draw_analysis_overlay`
      is ever called? (The `if self._analysis_cache is not None` guard covers the
      uncomputed case; the overlay is only drawn in `_show_frame` after `_precompute_analysis`
      has run)
- [ ] Is `self._updating = False` set in `ExportOptionsDialog.__init__` **before** any
      widget or group is created? (Checkboxes fire `stateChanged` synchronously on
      `setChecked(True)` during construction, which calls `_sync_group` before `_updating`
      would be set if it were placed at the end of `__init__`.)
- [ ] Does `_PROC_GATES` in `ExportOptionsDialog` cover every export option that depends
      on a selective processing module? Any new export option must be manually added here.
- [ ] Does `ProcessingOptionsDialog` correctly disable the pair group when `n_tracks < 2`?

---

*Document generated for code review. All line-number references reflect the state of the
codebase at the time of the last commit (see git log).*
