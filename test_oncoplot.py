"""
test_oncoplot.py — Verification tests for the oncoplot pipeline
================================================================
Tests the full chain: binning → aggregation → normalisation → matrix
using synthetic data with hand-calculable expected outputs.

Run: python test_oncoplot.py
"""

import numpy as np
import sys

PASS = 0
FAIL = 0


def check(name, got, expected, tol=1e-6):
    global PASS, FAIL
    ok = np.allclose(got, expected, atol=tol, equal_nan=True)
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
        print(f"  [{tag}] {name}: got {got}, expected {expected}")
    else:
        PASS += 1
        print(f"  [{tag}] {name}")


# -----------------------------------------------------------------------
# 1. Binning edge cases (_build_10s_bins)
# -----------------------------------------------------------------------

print("\n=== 1. Binning edge cases ===")

from graph_export import _build_10s_bins

# Single frame at t=0
bins = _build_10s_bins(np.array([0.0]))
check("single frame t=0 -> 1 bin", len(bins), 1)
check("single frame bin has 1 index", list(bins[0]), [0])

# Exactly 10s of data (t=0..9.99) -> 1 bin
times_10s = np.arange(0, 10.0, 0.01)
bins = _build_10s_bins(times_10s)
check("0..9.99s -> 1 bin", len(bins), 1)
check("all frames in bin 0", len(bins[0]), len(times_10s))

# Gap in time: t=0,1,2,25,26 -> bins 0 and 2 populated, bin 1 empty
times_gap = np.array([0.0, 1.0, 2.0, 25.0, 26.0])
bins = _build_10s_bins(times_gap)
check("gap: 3 bins (0,1,2)", len(bins), 3)
check("gap: bin 0 has 3 frames", len(bins[0]), 3)
check("gap: bin 1 is empty", len(bins[1]), 0)
check("gap: bin 2 has 2 frames", len(bins[2]), 2)

# All frames in one bin
times_all = np.array([3.0, 5.0, 7.0, 9.0])
bins = _build_10s_bins(times_all)
check("all in one bin -> n_bins=1", len(bins), 1)
check("all 4 frames in bin 0", len(bins[0]), 4)


# -----------------------------------------------------------------------
# 2. Per-row aggregation (bin means)
# -----------------------------------------------------------------------

print("\n=== 2. Per-row aggregation (bin means) ===")

from graph_export import (
    _compute_oncoplot_matrix, _circular_mean_360, _body_prefix
)

# 1 track, 30 frames at 1fps -> 3 bins of 10 frames each
times_30 = np.arange(30, dtype=float)  # 0..29
sleap_idxs = np.arange(30)

# Build minimal track_arrays for track 0
body_speed = np.array([10.0]*10 + [20.0]*10 + [30.0]*10)
body_jerk = np.array([-5.0]*10 + [10.0]*10 + [-15.0]*10)
heading = np.array([350.0]*5 + [10.0]*5 + [90.0]*10 + [270.0]*10)
path_eff = np.array([0.8]*10 + [0.5]*10 + [1.0]*10)
hg_area = np.array([100.0]*10 + [200.0]*10 + [50.0]*10)

track_arrays = {
    0: {
        'body_centre_speed': body_speed,
        'speed_accel': np.zeros(30),
        'body_centre_jerk': body_jerk,
        'body_heading_deg': heading,
        'path_efficiency': path_eff,
        'hourglass_area': hg_area,
        'hourglass_ratio': np.ones(30),
        'eccentricity': np.ones(30) * 0.5,
        'circularity': np.ones(30) * 0.7,
        'elongation': np.ones(30) * 0.3,
    }
}
node_names = ['body_centre']
track_names = ['Animal_0']

result = _compute_oncoplot_matrix(
    track_arrays, times_30, sleap_idxs, track_names, node_names, fps=1.0)
matrix_per_track, row_labels, section_breaks, global_stats, raw_vals = result

# Find row indices by label
label_idx = {lbl: i for i, lbl in enumerate(row_labels)}

# Speed bin means: [10, 20, 30]
r = label_idx['Speed']
check("speed bin means", list(raw_vals[r][0]), [10.0, 20.0, 30.0])

# |Jerk| bin means: |mean([-5]*10)| = 5, |mean([10]*10)| = 10, |mean([-15]*10)| = 15
r = label_idx['|Jerk|']
check("|jerk| bin means", list(raw_vals[r][0]), [5.0, 10.0, 15.0])

# Heading bin means (circular)
r = label_idx['Heading']
hdg_bin0 = _circular_mean_360(np.array([350.0]*5 + [10.0]*5))
hdg_bin1 = _circular_mean_360(np.array([90.0]*10))
hdg_bin2 = _circular_mean_360(np.array([270.0]*10))
check("heading bin 0 (wrap-around) ~0", raw_vals[r][0][0], hdg_bin0, tol=0.5)
check("heading bin 1 = 90", raw_vals[r][0][1], hdg_bin1, tol=0.5)
check("heading bin 2 = 270", raw_vals[r][0][2], hdg_bin2, tol=0.5)

# Path efficiency bin means: [0.8, 0.5, 1.0]
r = label_idx['Path Efficiency']
check("path_eff bin means", list(raw_vals[r][0]), [0.8, 0.5, 1.0])

# Hourglass area bin means: [100, 200, 50]
r = label_idx['Hourglass Area']
check("hourglass_area bin means", list(raw_vals[r][0]), [100.0, 200.0, 50.0])


# -----------------------------------------------------------------------
# 3. Normalisation functions (unit-level)
# -----------------------------------------------------------------------

print("\n=== 3. Normalisation functions ===")

from graph_export import _norm_minmax, _norm_p95, _norm_correlation

# minmax basic
check("minmax [10,20,30] min=10 max=30",
      list(_norm_minmax(np.array([10.0, 20.0, 30.0]), 10.0, 30.0)),
      [0.0, 0.5, 1.0])

# minmax with 2 tracks: global min=5, max=30
t0 = np.array([10.0, 20.0, 30.0])
t1 = np.array([5.0, 15.0, 25.0])
check("minmax 2-track t0",
      list(_norm_minmax(t0, 5.0, 30.0)),
      [0.2, 0.6, 1.0])
check("minmax 2-track t1",
      list(_norm_minmax(t1, 5.0, 30.0)),
      [0.0, 0.4, 0.8])

# p95
vals = np.array([10.0, 20.0, 30.0])
check("p95 [10,20,30] p95=25",
      list(_norm_p95(vals, 25.0)),
      [0.4, 0.8, 1.0])  # 30/25=1.2 clamped to 1.0

# p95 cross-track: max(p95_t0, p95_t1) used for both
# np.percentile([10]*10+[20]*10+[30]*10, 95) = 30.0
data_t0 = np.array([10.0]*10 + [20.0]*10 + [30.0]*10)
data_t1 = np.array([5.0]*10 + [15.0]*10 + [25.0]*10)
p95_t0 = float(np.percentile(data_t0, 95))  # 30.0
p95_t1 = float(np.percentile(data_t1, 95))  # 25.0
cross_p95 = max(p95_t0, p95_t1)  # 30.0
check("p95 cross-track divisor", cross_p95, 30.0, tol=0.1)

# heading normalisation
check("heading [0,90,270] -> [0, 0.25, 0.75]",
      list(np.clip(np.array([0.0, 90.0, 270.0]) / 360.0, 0, 1)),
      [0.0, 0.25, 0.75])

# direct
check("direct [0.8,0.5,1.0] -> unchanged",
      list(np.clip(np.array([0.8, 0.5, 1.0]), 0, 1)),
      [0.8, 0.5, 1.0])

# correlation
check("corr [-1,0,0.5,1] -> [0,0.5,0.75,1]",
      list(_norm_correlation(np.array([-1.0, 0.0, 0.5, 1.0]))),
      [0.0, 0.5, 0.75, 1.0])


# -----------------------------------------------------------------------
# 4. End-to-end matrix construction
# -----------------------------------------------------------------------

print("\n=== 4. End-to-end matrix (1 track) ===")

# Re-use the matrix_per_track from section 2
mat = matrix_per_track[0]

# Speed row (minmax): bin_means=[10,20,30], global min=10, max=30
r = label_idx['Speed']
check("matrix Speed row", list(mat[r]), [0.0, 0.5, 1.0])

# Speed p95% row: session_p95 = np.percentile([10]*10+[20]*10+[30]*10, 95) = 30.0
r = label_idx['Speed p95%']
expected_p95 = [10.0/30.0, 20.0/30.0, 1.0]  # 30/30=1.0
check("matrix Speed p95% row", list(mat[r]), expected_p95, tol=0.02)

# Heading row: circular means per bin -> /360
r = label_idx['Heading']
# bin0: ~0 deg -> 0/360=0.0, bin1: 90 -> 0.25, bin2: 270 -> 0.75
check("matrix Heading bin 0 ~0", mat[r][0], 0.0, tol=0.02)
check("matrix Heading bin 1 = 0.25", mat[r][1], 0.25, tol=0.02)
check("matrix Heading bin 2 = 0.75", mat[r][2], 0.75, tol=0.02)

# Path Efficiency row (direct): [0.8, 0.5, 1.0]
r = label_idx['Path Efficiency']
check("matrix Path Efficiency", list(mat[r]), [0.8, 0.5, 1.0])


# -----------------------------------------------------------------------
# 5. Multi-track global normalisation
# -----------------------------------------------------------------------

print("\n=== 5. Multi-track global normalisation ===")

track_arrays_2 = {
    0: {
        'body_centre_speed': np.array([5.0]*10 + [15.0]*10 + [25.0]*10),
        'speed_accel': np.zeros(30),
        'body_centre_jerk': np.zeros(30),
        'body_heading_deg': np.zeros(30),
        'path_efficiency': np.ones(30),
        'hourglass_area': np.ones(30),
        'hourglass_ratio': np.ones(30),
        'eccentricity': np.ones(30),
        'circularity': np.ones(30),
        'elongation': np.ones(30),
    },
    1: {
        'body_centre_speed': np.array([10.0]*10 + [20.0]*10 + [30.0]*10),
        'speed_accel': np.zeros(30),
        'body_centre_jerk': np.zeros(30),
        'body_heading_deg': np.zeros(30),
        'path_efficiency': np.ones(30),
        'hourglass_area': np.ones(30),
        'hourglass_ratio': np.ones(30),
        'eccentricity': np.ones(30),
        'circularity': np.ones(30),
        'elongation': np.ones(30),
    },
}
track_names_2 = ['Animal_0', 'Animal_1']

result2 = _compute_oncoplot_matrix(
    track_arrays_2, times_30, sleap_idxs, track_names_2, node_names, fps=1.0)
mpt2, rl2, sb2, gs2, rv2 = result2

r_speed = next(i for i, l in enumerate(rl2) if l == 'Speed')

# Global min=5, max=30, range=25
gmin, gmax = gs2[r_speed]
check("multi-track global min", gmin, 5.0)
check("multi-track global max", gmax, 30.0)

# Track 0: bin_means=[5,15,25] -> [(5-5)/25, (15-5)/25, (25-5)/25] = [0.0, 0.4, 0.8]
check("multi-track t0 Speed", list(mpt2[0][r_speed]), [0.0, 0.4, 0.8])

# Track 1: bin_means=[10,20,30] -> [(10-5)/25, (20-5)/25, (30-5)/25] = [0.2, 0.6, 1.0]
check("multi-track t1 Speed", list(mpt2[1][r_speed]), [0.2, 0.6, 1.0])


# -----------------------------------------------------------------------
# 6. Synchrony oncoplot matrix
# -----------------------------------------------------------------------

print("\n=== 6. Synchrony oncoplot matrix ===")

from graph_export import _compute_sync_matrix

pair_arrays = {
    't0_t1/inter_animal_dist': np.array([100.0]*10 + [50.0]*10 + [200.0]*10),
    't0_t1/pos_correlation_x': np.array([-0.5]*10 + [0.0]*10 + [0.8]*10),
    't0_t1/velocity_cos_sim':  np.array([-1.0]*10 + [0.0]*10 + [1.0]*10),
    't0_t1/pos_covariance_x':  np.zeros(30),
    't0_t1/pos_covariance_y':  np.zeros(30),
    't0_t1/pos_correlation_y': np.zeros(30),
}
sync_track_names = ['A0', 'A1']

matrix_sync, rl_sync, sb_sync, raw_sync = _compute_sync_matrix(
    pair_arrays, times_30, sleap_idxs, sync_track_names, fps=1.0)

# Build label -> row index
sl_idx = {lbl: i for i, lbl in enumerate(rl_sync)}

# Distance: bin_means=[100,50,200], minmax min=50, max=200, range=150
r = sl_idx['Distance (A0 vs A1)']
check("sync dist raw bins", list(raw_sync[r]), [100.0, 50.0, 200.0])
check("sync dist norm",
      list(np.round(matrix_sync[r], 3)),
      [round((100-50)/150, 3), 0.0, 1.0])

# Corr X: bin_means=[-0.5, 0.0, 0.8] -> corr norm: (v+1)/2
r = sl_idx['Corr X (A0 vs A1)']
check("sync corr_x norm", list(matrix_sync[r]), [0.25, 0.5, 0.9])

# Vel Cos Sim: bin_means=[-1, 0, 1] -> corr norm: [0, 0.5, 1]
r = sl_idx['Vel Cos Sim (A0 vs A1)']
check("sync vel_cos_sim norm", list(matrix_sync[r]), [0.0, 0.5, 1.0])


# -----------------------------------------------------------------------
# 7. Edge cases
# -----------------------------------------------------------------------

print("\n=== 7. Edge cases ===")

# 7a. All-NaN bin -> matrix cell = NaN
speed_nan = np.array([10.0]*10 + [np.nan]*10 + [30.0]*10)
ta_nan = {
    0: {
        'body_centre_speed': speed_nan,
        'speed_accel': np.zeros(30),
        'body_centre_jerk': np.zeros(30),
        'body_heading_deg': np.zeros(30),
        'path_efficiency': np.ones(30),
        'hourglass_area': np.ones(30),
        'hourglass_ratio': np.ones(30),
        'eccentricity': np.ones(30),
        'circularity': np.ones(30),
        'elongation': np.ones(30),
    }
}
res_nan = _compute_oncoplot_matrix(
    ta_nan, times_30, sleap_idxs, ['A0'], node_names, fps=1.0)
mat_nan = res_nan[0][0]
rl_nan = res_nan[1]
r_spd = next(i for i, l in enumerate(rl_nan) if l == 'Speed')
# bin 1 has all NaN -> nanmean = NaN -> normalized = NaN
# Actually nanmean of all-NaN returns NaN with a warning
check("all-NaN bin -> NaN cell", np.isnan(mat_nan[r_spd][1]), True)
# bins 0 and 2 should still have valid values
check("non-NaN bins still valid", np.isfinite(mat_nan[r_spd][0]), True)

# 7b. Single-frame bin: 11 frames (10 in bin0, 1 in bin1)
times_11 = np.arange(11, dtype=float)
sleap_11 = np.arange(11)
speed_11 = np.array([10.0]*10 + [42.0])
ta_11 = {
    0: {
        'body_centre_speed': speed_11,
        'speed_accel': np.zeros(11),
        'body_centre_jerk': np.zeros(11),
        'body_heading_deg': np.zeros(11),
        'path_efficiency': np.ones(11),
        'hourglass_area': np.ones(11),
        'hourglass_ratio': np.ones(11),
        'eccentricity': np.ones(11),
        'circularity': np.ones(11),
        'elongation': np.ones(11),
    }
}
res_11 = _compute_oncoplot_matrix(
    ta_11, times_11, sleap_11, ['A0'], node_names, fps=1.0)
rv_11 = res_11[4]
r_spd_11 = next(i for i, l in enumerate(res_11[1]) if l == 'Speed')
check("single-frame bin mean = that value", rv_11[r_spd_11][0][1], 42.0)

# 7c. Constant feature -> minmax range=0 -> all cells = 0.5
speed_const = np.full(30, 42.0)
ta_const = {
    0: {
        'body_centre_speed': speed_const,
        'speed_accel': np.zeros(30),
        'body_centre_jerk': np.zeros(30),
        'body_heading_deg': np.zeros(30),
        'path_efficiency': np.ones(30),
        'hourglass_area': np.ones(30),
        'hourglass_ratio': np.ones(30),
        'eccentricity': np.ones(30),
        'circularity': np.ones(30),
        'elongation': np.ones(30),
    }
}
res_const = _compute_oncoplot_matrix(
    ta_const, times_30, sleap_idxs, ['A0'], node_names, fps=1.0)
mat_const = res_const[0][0]
r_spd_c = next(i for i, l in enumerate(res_const[1]) if l == 'Speed')
check("constant feature -> all 0.5",
      list(mat_const[r_spd_c]), [0.5, 0.5, 0.5])

# 7d. Heading wrap-around: [1, 359] -> circular mean ~0 -> normalised ~0.0
hdg_wrap = np.array([1.0]*5 + [359.0]*5 + [90.0]*10 + [180.0]*10)
ta_wrap = {
    0: {
        'body_centre_speed': np.ones(30),
        'speed_accel': np.zeros(30),
        'body_centre_jerk': np.zeros(30),
        'body_heading_deg': hdg_wrap,
        'path_efficiency': np.ones(30),
        'hourglass_area': np.ones(30),
        'hourglass_ratio': np.ones(30),
        'eccentricity': np.ones(30),
        'circularity': np.ones(30),
        'elongation': np.ones(30),
    }
}
res_wrap = _compute_oncoplot_matrix(
    ta_wrap, times_30, sleap_idxs, ['A0'], node_names, fps=1.0)
mat_wrap = res_wrap[0][0]
r_hdg = next(i for i, l in enumerate(res_wrap[1]) if l == 'Heading')
# bin 0: circular_mean_360([1,1,1,1,1,359,359,359,359,359]) -> ~0 deg
# normalised = 0/360 = 0.0 (NOT 180/360 = 0.5 from arithmetic mean)
check("heading wrap [1,359] -> ~0.0 (not 0.5)", mat_wrap[r_hdg][0], 0.0, tol=0.02)


# -----------------------------------------------------------------------
# 8. Cross-check with binned_export (soft)
# -----------------------------------------------------------------------

print("\n=== 8. Cross-check with binned_export (soft, tol=5%) ===")

from graph_export import _build_10s_bins as build_10s
from binned_export import _circular_mean_deg

# For the same 30-frame data at 1fps, build 10s oncoplot bin means
# and compare with a manual 10s aggregation of the same raw data
speed_raw = np.array([10.0]*10 + [20.0]*10 + [30.0]*10)

# Oncoplot path: nanmean per 10s bin
bins_10s = build_10s(times_30)
onco_means = [np.nanmean(speed_raw[b]) for b in bins_10s]
check("cross-check speed bin 0", onco_means[0], 10.0)
check("cross-check speed bin 1", onco_means[1], 20.0)
check("cross-check speed bin 2", onco_means[2], 30.0)

# Manual sub-bin aggregation path: split into 0.25s bins, take medians,
# then aggregate to 1s, then to 10s. At 1fps each frame = 1s, so each
# 0.25s bin has at most 1 frame => median = value, 1s mean = value,
# 10s mean = same as direct nanmean. Should match exactly.
from binned_export import _build_bin_map

# Build 0.25s bins from frame_map
frame_map_30 = {i: i for i in range(30)}
bm025 = _build_bin_map(frame_map_30, fps=1.0, bin_size_s=0.25)
# Aggregate to 10s: group 0.25s bins into 10s windows
n_025_per_10s = int(10.0 / 0.25)  # 40
n_10s_bins = 3
manual_10s = []
for b10 in range(n_10s_bins):
    vals = []
    for sub_b, idxs in bm025.items():
        # sub_b is 0.25s bin index; time of this bin = sub_b * 0.25
        t_start = sub_b * 0.25
        if b10 * 10 <= t_start < (b10 + 1) * 10:
            for idx in idxs:
                vals.append(speed_raw[idx])
    manual_10s.append(np.mean(vals) if vals else np.nan)

check("cross-check manual 10s bin 0", manual_10s[0], onco_means[0], tol=0.5)
check("cross-check manual 10s bin 1", manual_10s[1], onco_means[1], tol=0.5)
check("cross-check manual 10s bin 2", manual_10s[2], onco_means[2], tol=0.5)

# Heading circular mean cross-check
heading_raw = np.array([350.0]*5 + [10.0]*5 + [90.0]*10 + [270.0]*10)
onco_hdg = [_circular_mean_360(heading_raw[b]) if len(b) else np.nan
            for b in bins_10s]
# Same values via binned_export's _circular_mean_deg (returns [-180,180])
# then mod 360 to match oncoplot convention (with same 360->0 guard)
def _to_360(deg):
    r = deg % 360.0
    return 0.0 if r >= 360.0 else r

be_hdg_bin0 = _to_360(_circular_mean_deg(heading_raw[:10]))
be_hdg_bin1 = _to_360(_circular_mean_deg(heading_raw[10:20]))
be_hdg_bin2 = _to_360(_circular_mean_deg(heading_raw[20:30]))

check("cross-check heading bin 0",
      onco_hdg[0], be_hdg_bin0, tol=1.0)
check("cross-check heading bin 1",
      onco_hdg[1], be_hdg_bin1, tol=1.0)
check("cross-check heading bin 2",
      onco_hdg[2], be_hdg_bin2, tol=1.0)


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"  PASSED: {PASS}   FAILED: {FAIL}")
print(f"{'='*60}")

if FAIL:
    print("\n  ** FAILURES DETECTED — see above **")

sys.exit(FAIL)
