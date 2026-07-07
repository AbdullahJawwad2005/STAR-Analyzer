"""
test_conversions.py — Micro-tests for unit conversion correctness
=================================================================
Run: python test_conversions.py
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
# 1. Dimensional analysis: px -> cm conversion factors
#    Given: positions in px, delta=1/fps in seconds
#    SG deriv=1 -> px/s   (velocity)
#    SG deriv=2 -> px/s²  (acceleration)
#    SG deriv=3 -> px/s³  (jerk)
#
#    To convert ANY of these from px to cm, multiply by (cm/px) ONCE:
#      px/s   × cm/px = cm/s
#      px/s²  × cm/px = cm/s²
#      px/s³  × cm/px = cm/s³
#
#    The spatial dimension is always px^1 in the numerator regardless of
#    the order of the time derivative.
# -----------------------------------------------------------------------

print("\n=== 1. Accel & Jerk conversion factor (dimensional analysis) ===")

px_per_cm = 20.0  # e.g., 20 pixels per cm
inv_ppcm  = 1.0 / px_per_cm   # cm/px = 0.05

# Simulated kinematics in px-based units
speed_px = 100.0   # px/s
accel_px = 50.0    # px/s²
jerk_px  = 25.0    # px/s³

# CORRECT conversions: always multiply by cm/px once
check("speed px/s -> cm/s",    speed_px * inv_ppcm,  5.0)    # 100 * 0.05 = 5
check("accel px/s2 -> cm/s2", accel_px * inv_ppcm, 2.5)    # 50 * 0.05 = 2.5
check("jerk  px/s3 -> cm/s3", jerk_px  * inv_ppcm, 1.25)   # 25 * 0.05 = 1.25

# WRONG (what the code currently does):
inv_ppcm2 = inv_ppcm ** 2  # (cm/px)² = cm²/px²
inv_ppcm3 = inv_ppcm ** 3  # (cm/px)³ = cm³/px³

wrong_accel = accel_px * inv_ppcm2  # 50 * 0.0025 = 0.125 (not cm/s²!)
wrong_jerk  = jerk_px  * inv_ppcm3  # 25 * 0.000125 = 0.003125 (not cm/s³!)

check("BUG: accel*inv_ppcm² gives wrong result", wrong_accel, 0.125)
check("BUG: jerk*inv_ppcm³ gives wrong result",  wrong_jerk,  0.003125)

# Show the error factor
print(f"\n  Error factor for accel: {inv_ppcm / inv_ppcm2:.1f}x too small "
      f"(inv_ppcm2 = {inv_ppcm2}, should be inv_ppcm = {inv_ppcm})")
print(f"  Error factor for jerk:  {inv_ppcm / inv_ppcm3:.1f}x too small "
      f"(inv_ppcm3 = {inv_ppcm3}, should be inv_ppcm = {inv_ppcm})")


# -----------------------------------------------------------------------
# 2. SG filter derivative units verification
# -----------------------------------------------------------------------

print("\n=== 2. SG filter derivative units ===")

from scipy.signal import savgol_filter

fps = 30.0
dt = 1.0 / fps
n = 100

# Linear motion: x(t) = 5*t  where t is in seconds
# -> velocity = 5 px/s, accel = 0, jerk = 0
t_sec = np.arange(n) * dt
x_linear = 5.0 * t_sec  # position in px

vx = savgol_filter(x_linear, 11, 3, deriv=1, delta=dt)
ax = savgol_filter(x_linear, 11, 3, deriv=2, delta=dt)

check("SG deriv=1 on linear motion -> velocity 5 px/s", vx[50], 5.0, tol=0.01)
check("SG deriv=2 on linear motion -> accel 0 px/s²",   ax[50], 0.0, tol=0.01)

# Quadratic motion: x(t) = 3*t²
# -> velocity = 6*t px/s, accel = 6 px/s², jerk = 0
x_quad = 3.0 * t_sec ** 2

vx_q = savgol_filter(x_quad, 11, 3, deriv=1, delta=dt)
ax_q = savgol_filter(x_quad, 11, 3, deriv=2, delta=dt)
jx_q = savgol_filter(x_quad, 11, 3, deriv=3, delta=dt)

check("SG deriv=1 on quadratic at t=1.67s -> ~10 px/s", vx_q[50], 6.0 * t_sec[50], tol=0.1)
check("SG deriv=2 on quadratic -> 6 px/s²",  ax_q[50], 6.0, tol=0.1)
check("SG deriv=3 on quadratic -> 0 px/s³",  jx_q[50], 0.0, tol=0.1)


# -----------------------------------------------------------------------
# 3. speed_accel units
# -----------------------------------------------------------------------

print("\n=== 3. speed_accel = d(speed)/dt units ===")

from features import _speed_accel

# Constant speed = 10 px/s -> speed_accel = 0
const_speed = np.full(100, 10.0)
sa = _speed_accel(const_speed, fps)
check("speed_accel of constant speed -> 0", sa[50], 0.0, tol=1e-10)

# Linearly increasing speed: speed(t) = 2*t (in frames, so speed increases by 2 per frame)
# d(speed)/dt = 2 per frame -> * fps = 2*30 = 60 px/s²
linear_speed = np.arange(100, dtype=float) * 2.0
sa_lin = _speed_accel(linear_speed, fps)
check("speed_accel of linear speed -> 2*fps = 60 px/s²", sa_lin[50], 60.0, tol=0.1)


# -----------------------------------------------------------------------
# 4. Position px -> cm conversion
# -----------------------------------------------------------------------

print("\n=== 4. Position px -> cm ===")

# ROI origin at (100, 200), px_per_cm = 20
rx0, ry0 = 100.0, 200.0
px_per_cm_test = 20.0

x_px, y_px = 300.0, 400.0
x_cm = (x_px - rx0) / px_per_cm_test  # (300 - 100) / 20 = 10 cm
y_cm = (y_px - ry0) / px_per_cm_test  # (400 - 200) / 20 = 10 cm

check("x px->cm", x_cm, 10.0)
check("y px->cm", y_cm, 10.0)


# -----------------------------------------------------------------------
# 5. Frame -> time conversion
# -----------------------------------------------------------------------

print("\n=== 5. Frame -> time ===")

check("frame 0 at 30fps -> 0s",   0 / 30.0, 0.0)
check("frame 30 at 30fps -> 1s",  30 / 30.0, 1.0)
check("frame 150 at 30fps -> 5s", 150 / 30.0, 5.0)


# -----------------------------------------------------------------------
# 6. Circular mean (binned_export)
# -----------------------------------------------------------------------

print("\n=== 6. Circular mean ===")

from binned_export import _circular_mean_deg

check("circular mean of [350, 10] -> 0 (wrap-around)", _circular_mean_deg(np.array([350.0, 10.0])), 0.0, tol=0.5)
check("circular mean of [90, 90] -> 90", _circular_mean_deg(np.array([90.0, 90.0])), 90.0, tol=0.01)
check("circular mean of [0, 180] -> 90 or -90 (ambiguous)",
      abs(_circular_mean_deg(np.array([0.0, 180.0]))), 90.0, tol=0.5)


# -----------------------------------------------------------------------
# 7. Oncoplot normalisation helpers (graph_export)
# -----------------------------------------------------------------------

print("\n=== 7. Oncoplot normalisation ===")

from graph_export import _norm_minmax, _norm_correlation, _norm_p95

# min-max
check("norm_minmax(5, 0, 10) -> 0.5", _norm_minmax(np.array([5.0]), 0.0, 10.0)[0], 0.5)
check("norm_minmax(0, 0, 10) -> 0.0", _norm_minmax(np.array([0.0]), 0.0, 10.0)[0], 0.0)
check("norm_minmax(10, 0, 10) -> 1.0", _norm_minmax(np.array([10.0]), 0.0, 10.0)[0], 1.0)
check("norm_minmax equal range -> 0.5", _norm_minmax(np.array([5.0]), 5.0, 5.0)[0], 0.5)

# correlation [-1,1] -> [0,1]
check("norm_corr(-1) -> 0", _norm_correlation(np.array([-1.0]))[0], 0.0)
check("norm_corr(0) -> 0.5", _norm_correlation(np.array([0.0]))[0], 0.5)
check("norm_corr(1) -> 1",  _norm_correlation(np.array([1.0]))[0], 1.0)

# p95
check("norm_p95(50, 100) -> 0.5", _norm_p95(np.array([50.0]), 100.0)[0], 0.5)
check("norm_p95(100, 100) -> 1.0", _norm_p95(np.array([100.0]), 100.0)[0], 1.0)
check("norm_p95(200, 100) -> 1.0 (clamped)", _norm_p95(np.array([200.0]), 100.0)[0], 1.0)


# -----------------------------------------------------------------------
# 8. Heading normalisation [0, 360] -> [0, 1]
# -----------------------------------------------------------------------

print("\n=== 8. Heading normalisation ===")

check("heading 0° -> 0.0", 0.0 / 360.0, 0.0)
check("heading 180° -> 0.5", 180.0 / 360.0, 0.5)
check("heading 360° -> 1.0", 360.0 / 360.0, 1.0)


# -----------------------------------------------------------------------
# 9. 10-second bin construction
# -----------------------------------------------------------------------

print("\n=== 9. 10-second binning ===")

from graph_export import _build_10s_bins

times = np.array([0, 3, 7, 10, 15, 20, 25], dtype=float)
bins = _build_10s_bins(times)
check("3 bins for 0-25s", len(bins), 3)
check("bin 0 has indices for t<10", list(bins[0]), [0, 1, 2])
check("bin 1 has indices for 10<=t<20", list(bins[1]), [3, 4])
check("bin 2 has indices for 20<=t<30", list(bins[2]), [5, 6])


# -----------------------------------------------------------------------
# 10. velocity_cos_sim range check
# -----------------------------------------------------------------------

print("\n=== 10. velocity_cos_sim ===")

# Same direction -> cos_sim = 1
vxA = np.array([1.0, 2.0, 3.0])
vyA = np.array([0.0, 0.0, 0.0])
vxB = np.array([2.0, 4.0, 6.0])
vyB = np.array([0.0, 0.0, 0.0])
dot = vxA * vxB + vyA * vyB
mag = np.hypot(vxA, vyA) * np.hypot(vxB, vyB)
cos_sim = np.where(mag > 1e-12, dot / mag, 0.0)
check("same direction -> cos_sim = 1", cos_sim[1], 1.0)

# Opposite direction -> cos_sim = -1
vxB_opp = -vxB
dot2 = vxA * vxB_opp + vyA * vyB
mag2 = np.hypot(vxA, vyA) * np.hypot(vxB_opp, vyB)
cos_sim2 = np.where(mag2 > 1e-12, dot2 / mag2, 0.0)
check("opposite direction -> cos_sim = -1", cos_sim2[1], -1.0)

# Perpendicular -> cos_sim = 0
vyB_perp = np.array([2.0, 4.0, 6.0])
vxB_perp = np.array([0.0, 0.0, 0.0])
dot3 = vxA * vxB_perp + vyA * vyB_perp
mag3 = np.hypot(vxA, vyA) * np.hypot(vxB_perp, vyB_perp)
cos_sim3 = np.where(mag3 > 1e-12, dot3 / mag3, 0.0)
check("perpendicular -> cos_sim = 0", cos_sim3[1], 0.0)


# -----------------------------------------------------------------------
# 11. Distance px -> cm in graph_export
# -----------------------------------------------------------------------

print("\n=== 11. Distance plot conversion ===")

dist_px = np.array([60.0, 120.0, 0.0])
px_per_cm_dist = 20.0
dist_cm = dist_px / max(px_per_cm_dist, 1e-9)
check("60 px at 20 px/cm -> 3 cm", dist_cm[0], 3.0)
check("120 px at 20 px/cm -> 6 cm", dist_cm[1], 6.0)
check("0 px -> 0 cm", dist_cm[2], 0.0)


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"  PASSED: {PASS}   FAILED: {FAIL}")
print(f"{'='*60}")

if FAIL:
    print("\n  ** KNOWN BUGS DETECTED — see fixes below **")
    print("  BUG 1: run_popup.py line 180: accel * inv_ppcm2 should be accel * inv_ppcm")
    print("  BUG 2: run_popup.py line 181: jerk  * inv_ppcm3 should be jerk  * inv_ppcm")
    print("  BUG 3: run_popup.py line 480: accel * ipc2 should be accel * ipc")
    print("  BUG 4: run_popup.py line 482: jerk  * ipc3 should be jerk  * ipc")
    print("  BUG 5: graph_export.py _NICE_LABEL units are wrong (px/frame should be px/s)")

sys.exit(FAIL)
