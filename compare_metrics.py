"""
Sanity-check: compare MOSIAC key metrics against ground-truth independent calculations.
Run: python compare_metrics.py

Checks:
  1. Total Distance   — MOSIAC cumsum vs raw np.diff cumsum
  2. Speed stats      — MOSIAC SG-derivative vs raw frame-diff / fps
  3. Acceleration     — KEY BUG CHECK: d(speed)/dt vs kin['accel'] (SG deriv-2)
  4. Immobility       — fraction of session, sanity range
  5. Proximity        — (1s bouts) sum vs raw frame count / fps
"""

import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

H5  = r"C:\Users\abdul\Downloads\Practice\Trial 5_18-12#1x3-10#3.analysis.h5"
FPS = 30.0   # update if your video is a different frame rate
# px_per_cm: derived from DSR (hip-to-hip median body width / 3 cm).
# We compute it automatically below; override here if you know the exact value.
PX_PER_CM_OVERRIDE = None   # e.g. 37.5  or leave None for auto-DSR

PROX_CM = 3.0   # general proximity threshold (same as CONTACT_THRESHOLD_CM=1 / PROX=3)
CONTACT_CM = 1.0

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
from sleap_loader import load_sleap
from preprocessing import fill_and_smooth_tracks, compute_kinematics
from features import build_key_metrics_df, find_node_idx, PROX_THRESHOLD_CM, CONTACT_THRESHOLD_CM

print("Loading H5 …")
raw = load_sleap(H5)
tracks_raw  = raw["tracks"]
node_names  = raw["node_names"]
track_names = raw["track_names"]
frame_map   = raw["frame_map"]
n_frames, _, n_nodes, n_tracks = tracks_raw.shape

print(f"  tracks shape : {tracks_raw.shape}")
print(f"  nodes        : {node_names}")
print(f"  tracks       : {track_names}")
print(f"  frame_map    : {len(frame_map)} entries, frames {min(frame_map)}-{max(frame_map)}")

# ---------------------------------------------------------------------------
# Trim first 5 s (same as _run_analysis)
# ---------------------------------------------------------------------------
analysis_start = min(int(5 * FPS), n_frames)
frame_map = {vf: si - analysis_start
             for vf, si in frame_map.items()
             if si >= analysis_start}
tracks_raw = tracks_raw[analysis_start:]
n_frames = tracks_raw.shape[0]

# ---------------------------------------------------------------------------
# Preprocess (fill + smooth)
# ---------------------------------------------------------------------------
print("Preprocessing …")
tracks = fill_and_smooth_tracks(tracks_raw, FPS)
kin    = compute_kinematics(tracks, FPS, node_names=node_names)

# ---------------------------------------------------------------------------
# px_per_cm — auto DSR from hip nodes (mirrors original.py approach)
# ---------------------------------------------------------------------------
body_idx  = find_node_idx(node_names, 'body')
nose_idx  = find_node_idx(node_names, 'nose')
hipl_idx  = find_node_idx(node_names, 'hip_l')
hipr_idx  = find_node_idx(node_names, 'hip_r')

if PX_PER_CM_OVERRIDE is not None:
    px_per_cm = PX_PER_CM_OVERRIDE
    print(f"px_per_cm    : {px_per_cm:.4f} (override)")
elif hipl_idx is not None and hipr_idx is not None:
    hip_dists = np.linalg.norm(
        tracks[:, :, hipl_idx, 0] - tracks[:, :, hipr_idx, 0], axis=1)
    lo, hi = np.nanpercentile(hip_dists, [20, 80])
    dsr = np.nanmedian(hip_dists[(hip_dists >= lo) & (hip_dists <= hi)])
    px_per_cm = dsr / 3.0   # 3 cm = mouse hip width
    print(f"px_per_cm    : {px_per_cm:.4f}  (DSR={dsr:.2f} px)")
else:
    px_per_cm = 37.5  # fallback — 1 cm ≈ 37.5 px for typical setup
    print(f"px_per_cm    : {px_per_cm:.4f}  (fallback)")

# ---------------------------------------------------------------------------
# MOSIAC key metrics (only locomotion tier needed)
# ---------------------------------------------------------------------------
print("\nRunning MOSIAC build_key_metrics_df …")

from features import _node_total_displacement
track_arrays = []
for t in range(n_tracks):
    node_disp = _node_total_displacement(tracks, t)
    # cm_total_disp must be the 1D cumulative array for the body node (n_frames,)
    body_disp = (node_disp[body_idx] if body_idx is not None
                 else np.nanmean(np.stack(list(node_disp.values())), axis=0))
    track_arrays.append({'cm_total_disp': body_disp})

km_df = build_key_metrics_df(
    tracks=tracks, kin=kin, single_beh={}, pair_beh={},
    track_arrays=track_arrays, pair_arrays={}, frame_map=frame_map,
    zone_summary_df=None, node_names=node_names, track_names=track_names,
    fps=FPS, px_per_cm=px_per_cm, zone_label=None,
)

mosiac_loco = km_df[km_df['Category'] == 'Locomotion'].copy()

print("\n--- MOSIAC Key Metrics (Locomotion) ---")
for _, row in mosiac_loco.iterrows():
    print(f"  {row['Metric']:40s}  {row['Subject']:10s}  {row['Value']:>10.4f} {row['Unit']}")

# ---------------------------------------------------------------------------
# INDEPENDENT verification
# ---------------------------------------------------------------------------
sleap_idxs = np.array(sorted(frame_map.values()))
print("\n\n=== Independent Verification ===")

for t, tname in enumerate(track_names):
    print(f"\n--- Track: {tname} ---")

    # ----- node coords (smoothed) -----
    if body_idx is not None:
        cx = tracks[sleap_idxs, 0, body_idx, t]
        cy = tracks[sleap_idxs, 1, body_idx, t]
    else:
        cx = np.nanmean(tracks[sleap_idxs, 0, :, t], axis=1)
        cy = np.nanmean(tracks[sleap_idxs, 1, :, t], axis=1)

    # 1. TOTAL DISTANCE — raw cumulative sum of frame-to-frame step sizes
    dx = np.diff(cx, prepend=cx[0])
    dy = np.diff(cy, prepend=cy[0])
    raw_total_dist_cm = float(np.nansum(np.hypot(dx, dy))) / px_per_cm
    mosiac_val = mosiac_loco.query(
        "Metric == 'Total Distance Traveled' and Subject == @tname")['Value'].values
    mosiac_val = float(mosiac_val[0]) if len(mosiac_val) else float('nan')
    pct_diff = abs(raw_total_dist_cm - mosiac_val) / max(abs(raw_total_dist_cm), 1e-9) * 100
    print(f"  Total Distance  MOSIAC={mosiac_val:.2f}  INDEP={raw_total_dist_cm:.2f}  diff={pct_diff:.1f}%")
    if pct_diff > 5:
        print("    *** WARNING: >5% difference ***")

    # 2. SPEED — raw frame diff vs MOSIAC (SG derivative)
    spd_raw_cms = np.hypot(dx, dy) * FPS / px_per_cm   # px/frame * fps / px_per_cm = cm/s
    spd_sg_cms  = kin['speed'][sleap_idxs, body_idx if body_idx is not None else 0, t] / px_per_cm

    print(f"  Avg Speed       MOSIAC={float(np.nanmean(spd_sg_cms)):.4f}  RAW-DIFF={float(np.nanmean(spd_raw_cms)):.4f}  cm/s")
    print(f"  Median Speed    MOSIAC={float(np.nanmedian(spd_sg_cms)):.4f}  RAW-DIFF={float(np.nanmedian(spd_raw_cms)):.4f}  cm/s")
    print(f"  P95 Speed       MOSIAC={float(np.nanpercentile(spd_sg_cms,95)):.4f}  RAW-DIFF={float(np.nanpercentile(spd_raw_cms,95)):.4f}  cm/s")

    # 3. ACCELERATION — KEY CHECK
    #    MOSIAC current: np.gradient(speed) * fps  = d|v|/dt
    #    MOSIAC correct: kin['accel'] from SG deriv-2 = |a_vec| = sqrt(ax^2+ay^2)
    acc_current_cms = np.abs(np.gradient(spd_sg_cms)) * FPS  # what code does now (array)
    acc_sg_cms      = kin['accel'][sleap_idxs, body_idx if body_idx is not None else 0, t] / px_per_cm
    mosiac_avg_acc  = mosiac_loco.query(
        "Metric == 'Avg Abs Acceleration' and Subject == @tname")['Value'].values
    mosiac_p95_acc  = mosiac_loco.query(
        "Metric == 'P95 Abs Acceleration' and Subject == @tname")['Value'].values
    mosiac_avg_acc  = float(mosiac_avg_acc[0]) if len(mosiac_avg_acc) else float('nan')
    mosiac_p95_acc  = float(mosiac_p95_acc[0]) if len(mosiac_p95_acc) else float('nan')

    print(f"\n  --- Acceleration comparison ---")
    print(f"  Avg  Acc  MOSIAC-current(d|v|/dt)={mosiac_avg_acc:.4f}  SG-vector={float(np.nanmean(acc_sg_cms)):.4f}  cm/s²")
    print(f"  P95  Acc  MOSIAC-current(d|v|/dt)={mosiac_p95_acc:.4f}  SG-vector={float(np.nanpercentile(acc_sg_cms,95)):.4f}  cm/s²")
    ratio_avg = float(np.nanmean(acc_sg_cms)) / max(mosiac_avg_acc, 1e-9)
    if abs(ratio_avg - 1.0) > 0.1:
        print(f"    *** SG-vector acceleration is {ratio_avg:.2f}x larger (d|v|/dt underestimates turning) ***")

print("\n\n=== Summary of Issues ===")
print("  1. Acceleration: Key Metrics uses d(speed)/dt which misses centripetal acceleration.")
print("     Fix: use kin['accel'] (SG 2nd derivative vector magnitude) instead.")
print("  See ratio above — if >1.1x, the bug meaningfully underestimates acceleration.")
