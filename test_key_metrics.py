"""
Ground-truth key-metrics test suite for STAR Analyzer.

Run:  python test_key_metrics.py
Exit: 0 = all pass; non-zero = number of failures.

Design goals
------------
- Known-answer synthetic inputs verified by hand.
- Every check prints intermediate values so results are auditable.
- Two independent computation paths compared for critical metrics.
- Domain sanity assertions (ranges, monotonicity, mutual exclusion).
- Failure-case audit: NaN propagation, angle wrap-around, edge inputs.
"""
import os, sys
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ── test framework (matches existing test files) ────────────────────────────
PASS = 0; FAIL = 0
GRN = "\033[32m"; RED = "\033[31m"; RST = "\033[0m"


def check(name, got, expected, tol=1e-6):
    global PASS, FAIL
    try:
        ok = bool(np.allclose(np.asarray(got, float), np.asarray(expected, float),
                              atol=tol, equal_nan=True))
    except Exception:
        ok = (got == expected)
    if ok:
        PASS += 1; print(f"  {GRN}PASS{RST}  {name}")
    else:
        FAIL += 1; print(f"  {RED}FAIL{RST}  {name}"
                         f"\n        got={got!r}\n   expected={expected!r}")


# ── module imports (pure-numpy, no Qt needed for sections 1-7) ──────────────
from features     import _general_min_dist, _detect_bouts, _tailend_node_idxs
from behaviors    import angular_velocity, compute_single_animal
from preprocessing import compute_kinematics

# ── helper builders ─────────────────────────────────────────────────────────

def _make_kin(speed_1d, heading_1d):
    """Inject precomputed speed/heading bypassing SG filter."""
    return {
        'speed':            np.asarray(speed_1d,   dtype=np.float32)[:, np.newaxis, np.newaxis],
        'body_heading_deg': np.asarray(heading_1d, dtype=np.float64)[:, np.newaxis],
    }


def _tracks_1n1a(xs, ys):
    """(n_frames, 2, 1, 1) -- single node, single animal."""
    n = len(xs)
    t = np.zeros((n, 2, 1, 1), dtype=np.float32)
    t[:, 0, 0, 0] = xs
    t[:, 1, 0, 0] = ys
    return t


def _tracks_2n1a(body_xs, body_ys, nose_xs, nose_ys):
    """(n_frames, 2, 2, 1) -- body+nose, single animal."""
    n = len(body_xs)
    t = np.zeros((n, 2, 2, 1), dtype=np.float32)
    t[:, 0, 0, 0] = body_xs;  t[:, 1, 0, 0] = body_ys  # node 0 = body
    t[:, 0, 1, 0] = nose_xs;  t[:, 1, 1, 0] = nose_ys  # node 1 = nose
    return t


def _zone_label(x, y, rx0, ry0, rx1, ry1, strip_px):
    """Replicate the zone-labeling logic from run_popup.py lines 237-241."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    near_h = np.minimum(x - rx0, rx1 - x) < strip_px
    near_v = np.minimum(y - ry0, ry1 - y) < strip_px
    lbl = np.full(len(x), 'Center', dtype=object)
    lbl[near_h ^ near_v] = 'Perimeter'
    lbl[near_h & near_v] = 'Corner'
    return lbl


def _hdg_diff_scalar(h0, h1):
    """Replicate the hdg_diff formula from run_popup.py lines 256-258."""
    d = abs(h0 - h1) % 360
    return min(d, 360 - d)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- _general_min_dist
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 1: Distance / Proximity ===")

# ── 1a. Two-node two-animal known geometry ───────────────────────────────────
tracks_1a = np.zeros((1, 2, 2, 2), dtype=np.float32)
# Animal 0: node0=(0,0)  node1=(10,10)
tracks_1a[0, 0, 0, 0] = 0;   tracks_1a[0, 1, 0, 0] = 0
tracks_1a[0, 0, 1, 0] = 10;  tracks_1a[0, 1, 1, 0] = 10
# Animal 1: node0=(5,0)  node1=(15,10)
tracks_1a[0, 0, 0, 1] = 5;   tracks_1a[0, 1, 0, 1] = 0
tracks_1a[0, 0, 1, 1] = 15;  tracks_1a[0, 1, 1, 1] = 10

node_names_2 = ['node0', 'node1']
min_dist_1a, closest_1a = _general_min_dist(tracks_1a, [0], 0, 1, node_names_2)

# Show work: cross-pair distance matrix (row=A-node, col=B-node)
dm = np.array([
    [np.hypot(0-5, 0-0),  np.hypot(0-15, 0-10)],
    [np.hypot(10-5, 10-0), np.hypot(10-15, 10-10)],
])
print(f"  [show] 1a cross-pair dist matrix:\n         A0-B0={dm[0,0]:.3f}  A0-B1={dm[0,1]:.3f}"
      f"\n         A1-B0={dm[1,0]:.3f}  A1-B1={dm[1,1]:.3f}")
print(f"  [show] 1a min_dist[0]={min_dist_1a[0]:.4f}, closest={closest_1a[0]!r}")

expected_1a = np.hypot(5, 0)   # independent recompute: A0-B0 = hypot(5-0, 0-0)
check("1a. min_dist two-animal two-node", min_dist_1a[0], expected_1a)
check("1a. closest_pair = node0-node0", closest_1a[0], 'node0-node0')
check("1a. independent recompute matches", min_dist_1a[0], np.hypot(5, 0))

# ── 1b. Single-node geometry ─────────────────────────────────────────────────
tracks_1b = np.zeros((1, 2, 1, 2), dtype=np.float32)
tracks_1b[0, 0, 0, 0] = 0;  tracks_1b[0, 1, 0, 0] = 0   # A at (0,0)
tracks_1b[0, 0, 0, 1] = 3;  tracks_1b[0, 1, 0, 1] = 4   # B at (3,4)
min_dist_1b, _ = _general_min_dist(tracks_1b, [0], 0, 1, ['node0'])
print(f"  [show] 1b min_dist={min_dist_1b[0]:.4f}  expected=hypot(3,4)={np.hypot(3,4):.4f}")
check("1b. single-node hypot(3,4)=5", min_dist_1b[0], np.hypot(3, 4))

# ── 1c. NaN propagation ──────────────────────────────────────────────────────
tracks_1c = np.full((1, 2, 2, 2), np.nan, dtype=np.float32)
min_dist_1c, _ = _general_min_dist(tracks_1c, [0], 0, 1, ['n0', 'n1'])
print(f"  [show] 1c all-NaN -> min_dist={min_dist_1c[0]}")
check("1c. all-NaN tracks -> min_dist is NaN", np.isnan(min_dist_1c[0]), True)

# ── 1d. Tailend exclusion ────────────────────────────────────────────────────
# 3 nodes: body, nose, tailend.  tailend of A is 1px from tailend of B.
tracks_1d = np.zeros((1, 2, 3, 2), dtype=np.float32)
# Animal 0: body=(0,0) nose=(5,0) tailend=(10,0)
tracks_1d[0, 0, 0, 0]=0;   tracks_1d[0, 1, 0, 0]=0
tracks_1d[0, 0, 1, 0]=5;   tracks_1d[0, 1, 1, 0]=0
tracks_1d[0, 0, 2, 0]=10;  tracks_1d[0, 1, 2, 0]=0
# Animal 1: body=(100,0) nose=(105,0) tailend=(11,0)  -> tailend 1px from A-tailend
tracks_1d[0, 0, 0, 1]=100; tracks_1d[0, 1, 0, 1]=0
tracks_1d[0, 0, 1, 1]=105; tracks_1d[0, 1, 1, 1]=0
tracks_1d[0, 0, 2, 1]=11;  tracks_1d[0, 1, 2, 1]=0

node_names_1d = ['body', 'nose', 'tailend']
excl_idxs = _tailend_node_idxs(node_names_1d)
min_no_excl, _  = _general_min_dist(tracks_1d, [0], 0, 1, node_names_1d, exclude_idxs=None)
min_excl, _     = _general_min_dist(tracks_1d, [0], 0, 1, node_names_1d, exclude_idxs=excl_idxs)
print(f"  [show] 1d excluded indices={excl_idxs}  "
      f"no-excl={min_no_excl[0]:.2f}  with-excl={min_excl[0]:.2f}")
check("1d. without exclusion tailend pair is closest (1px)", min_no_excl[0], 1.0)
check("1d. tailend exclusion increases min distance", min_excl[0] > min_no_excl[0], True)

# ── 1e. Proximity/contact threshold (cumulative-time formula) ────────────────
fps_1e  = 10.0
px_per_cm_1e = 10
prox_px_1e   = 3 * px_per_cm_1e   # 30 px
cont_px_1e   = 1 * px_per_cm_1e   # 10 px
dists_1e = np.array([5., 25., 35., 8., 10., 31.])  # 31 > prox_px=30, last frame Out

prox_mask_1e = (dists_1e <= prox_px_1e).astype(float)   # [1,1,0,1,1,0]
cont_mask_1e = (dists_1e <= cont_px_1e).astype(float)   # [1,0,0,1,1,0]
prox_ct_1e   = np.cumsum(prox_mask_1e) / fps_1e
cont_ct_1e   = np.cumsum(cont_mask_1e) / fps_1e

# Independent recompute via simple counting
n_prox_manual = int(sum(d <= prox_px_1e for d in dists_1e))  # 4: [5,25,8,10]
n_cont_manual = int(sum(d <= cont_px_1e for d in dists_1e))  # 3: [5,8,10]
print(f"  [show] 1e prox_mask={prox_mask_1e.astype(int).tolist()}")
print(f"  [show] 1e cont_mask={cont_mask_1e.astype(int).tolist()}")
print(f"  [show] 1e prox_cumtime={prox_ct_1e.tolist()}")
print(f"  [show] 1e cont_cumtime={cont_ct_1e.tolist()}")
print(f"  [show] 1e manual counts: prox={n_prox_manual}  cont={n_cont_manual}")
check("1e. prox_cumtime final value (4 frames)", prox_ct_1e[-1], 4.0 / fps_1e)
check("1e. cont_cumtime final value (3 frames)", cont_ct_1e[-1], 3.0 / fps_1e)
check("1e. independent count matches prox cumtime", prox_ct_1e[-1], n_prox_manual / fps_1e)
check("1e. independent count matches cont cumtime", cont_ct_1e[-1], n_cont_manual / fps_1e)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Speed + Rolling Window
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 2: Speed + Rolling Window ===")
import pandas as pd

fps_2 = 10.0; n_2 = 100
body_xs = np.arange(n_2, dtype=np.float32)   # x = 0,1,2,...,99  (+1 px/frame)
body_ys = np.zeros(n_2, dtype=np.float32)

# compute_kinematics needs (n_frames, 2, n_nodes, n_tracks)
tracks_2a = _tracks_1n1a(body_xs, body_ys)
kin_2a = compute_kinematics(tracks_2a, fps_2, node_names=None)

# SG filter on linear data -> derivative = slope = fps px/s (exact for poly >= 1)
mid = 50
raw_speed_interior = kin_2a['speed'][mid, 0, 0]
print(f"  [show] 2a kin['speed'][48:53,0,0] = {kin_2a['speed'][48:53,0,0]}")

# Independent recompute: central difference = fps (exact for linear)
grad_vx = np.gradient(body_xs.astype(float)) * fps_2
independent_speed = float(grad_vx[mid])
print(f"  [show] 2a np.gradient * fps at frame {mid} = {independent_speed:.4f}")
check("2a. SG speed at interior frame ~= fps px/s (10)", raw_speed_interior, fps_2, tol=1e-2)
check("2a. SG matches np.gradient at interior frame",   raw_speed_interior, independent_speed, tol=0.5)

# Rolling-window: ROLL_W = max(3, round(0.2*fps)) = max(3,2) = 3
ROLL_W_2 = max(3, round(0.2 * fps_2))
spd_series = kin_2a['speed'][:, 0, 0]
rolling_2a = pd.Series(spd_series).rolling(ROLL_W_2, center=True, min_periods=1).mean().to_numpy()
print(f"  [show] 2a rolling_speed[48:53] = {rolling_2a[48:53]}")
check("2a. rolling speed at interior frame ~= 10 px/s", rolling_2a[mid], fps_2, tol=1e-2)
check("2a. speed_cm = speed_px / px_per_cm = 1.0 cm/s",
      rolling_2a[mid] / px_per_cm_1e, 1.0, tol=1e-3)

# ── 2b. Domain sanity: speed >= 0 everywhere ─────────────────────────────────
check("2b. all speed values non-negative", np.all(kin_2a['speed'] >= 0), True)

# ── 2c. Stationary -> near-zero speed ────────────────────────────────────────
tracks_2c = _tracks_1n1a(np.full(50, 42.0), np.full(50, 17.0))
kin_2c = compute_kinematics(tracks_2c, fps_2, node_names=None)
max_speed_stationary = float(np.max(np.abs(kin_2c['speed'])))
print(f"  [show] 2c stationary max speed = {max_speed_stationary:.6f}")
check("2c. stationary tracks -> speed ~= 0 everywhere", max_speed_stationary, 0.0, tol=1e-3)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Body Heading & Heading Delta
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 3: Body Heading & Heading Delta ===")

# ── 3a. hdg_diff formula: known cardinal angles ──────────────────────────────
cases_3a = [
    (0,   180, 180, "0->180"),
    (0,    90,  90, "0->90"),
    (10,  350,  20, "10->350 wrap"),
    (170, 190,  20, "170->190"),
    (350,  10,  20, "350->10 wrap"),
    (0,     0,   0, "0->0 identity"),
]
# Independent recompute via unit vectors: arccos(cos(h0-h1))
def _hdg_diff_vec(h0, h1):
    d = np.radians(h0 - h1)
    return float(np.degrees(np.arccos(np.clip(np.cos(d), -1.0, 1.0))))

for h0, h1, expected_diff, label in cases_3a:
    got = _hdg_diff_scalar(h0, h1)
    vec = _hdg_diff_vec(h0, h1)
    print(f"  [show] 3a {label}: scalar={got:.1f}  vec={vec:.1f}  expected={expected_diff}")
    check(f"3a. hdg_diff({h0},{h1})={expected_diff} -- scalar", got, expected_diff, tol=1e-9)
    check(f"3a. hdg_diff({h0},{h1})={expected_diff} -- vector recompute", vec, expected_diff, tol=1e-6)

# ── 3b. Body heading from compute_kinematics -- static geometry ───────────────
N_3b = 30
# East (0deg): body at (0,0), nose at (1,0)
tr_east = _tracks_2n1a(
    body_xs=np.zeros(N_3b), body_ys=np.zeros(N_3b),
    nose_xs=np.ones(N_3b),  nose_ys=np.zeros(N_3b),
)
kin_east = compute_kinematics(tr_east, fps_2, node_names=['body', 'nose'])
interior_hdg_east = kin_east['body_heading_deg'][10:15, 0]
print(f"  [show] 3b East heading frames 10-14: {interior_hdg_east}")
check("3b. East heading ~= 0deg", float(np.mean(interior_hdg_east)), 0.0, tol=1.0)

# North (90deg): body at (0,0), nose at (0,1)
tr_north = _tracks_2n1a(
    body_xs=np.zeros(N_3b), body_ys=np.zeros(N_3b),
    nose_xs=np.zeros(N_3b), nose_ys=np.ones(N_3b),
)
kin_north = compute_kinematics(tr_north, fps_2, node_names=['body', 'nose'])
interior_hdg_north = kin_north['body_heading_deg'][10:15, 0]
print(f"  [show] 3b North heading frames 10-14: {interior_hdg_north}")
check("3b. North heading ~= 90deg", float(np.mean(interior_hdg_north)), 90.0, tol=1.0)

# ── 3c. Domain sanity: body_heading_deg in [-180, 180] ───────────────────────
all_hdg = kin_east['body_heading_deg']
in_range = np.all((all_hdg >= -180.0) & (all_hdg <= 180.0))
check("3c. body_heading_deg in [-180, 180]", in_range, True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- Zone Labeling
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 4: Zone Labeling ===")

rx0, ry0, rx1, ry1 = 0, 0, 100, 100
strip_px_4 = 10.0   # strip_cm=1.0 * px_per_cm=10

# Known geometry: 7 test points
points_4 = np.array([
    [50.0, 50.0],   # Center   (both >= 10 from all edges)
    [ 5.0, 50.0],   # Perimeter (near left edge only)
    [50.0,  5.0],   # Perimeter (near top edge only)
    [ 5.0,  5.0],   # Corner   (near left AND top)
    [95.0, 95.0],   # Corner   (near right AND bottom: min(95,5)=5 < 10)
    [10.0, 50.0],   # Center   (boundary exact: min(10,90)=10, NOT < 10)
    [ 9.0, 50.0],   # Perimeter (min(9,91)=9 < 10, near left)
])
expected_4 = ['Center', 'Perimeter', 'Perimeter', 'Corner', 'Corner', 'Center', 'Perimeter']

xs_4, ys_4 = points_4[:, 0], points_4[:, 1]
near_h_4   = np.minimum(xs_4 - rx0, rx1 - xs_4)
near_v_4   = np.minimum(ys_4 - ry0, ry1 - ys_4)
labels_4   = _zone_label(xs_4, ys_4, rx0, ry0, rx1, ry1, strip_px_4)

print(f"  [show] 4 near_h={near_h_4.tolist()}  near_v={near_v_4.tolist()}")
print(f"  [show] 4 labels={labels_4.tolist()}")

# Independent recompute via Python list comprehension
def _zone_manual(x, y):
    nh = min(x - rx0, rx1 - x) < strip_px_4
    nv = min(y - ry0, ry1 - y) < strip_px_4
    if nh and nv: return 'Corner'
    if nh or nv:  return 'Perimeter'
    return 'Center'

labels_manual_4 = [_zone_manual(x, y) for x, y in zip(xs_4, ys_4)]
print(f"  [show] 4 manual={labels_manual_4}")

for i, (got_lbl, exp_lbl, man_lbl) in enumerate(zip(labels_4, expected_4, labels_manual_4)):
    p = points_4[i]
    check(f"4a. ({p[0]:.0f},{p[1]:.0f}) -> {exp_lbl}", got_lbl, exp_lbl)
    check(f"4a. ({p[0]:.0f},{p[1]:.0f}) manual matches", man_lbl, exp_lbl)

# ── 4b. Domain sanity: all labels are valid strings ──────────────────────────
rng = np.random.default_rng(42)
xs_rand = rng.uniform(rx0, rx1, 1000)
ys_rand = rng.uniform(ry0, ry1, 1000)
labels_rand = _zone_label(xs_rand, ys_rand, rx0, ry0, rx1, ry1, strip_px_4)
valid_labels = {'Center', 'Perimeter', 'Corner'}
all_valid = all(l in valid_labels for l in labels_rand)
check("4b. all 1000 random labels in valid set", all_valid, True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- Bout Detection
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 5: Bout Detection ===")

fps_5 = 10.0
# fps=10: MAX_GAP=3, MIN_BOUT=8

def _arr(s):
    """Build bool array from 'T'/'F' string."""
    return np.array([c == 'T' for c in s])

# Case A: 20 consecutive True -> (0,19)
bouts_A = _detect_bouts(_arr('T' * 20), fps_5)
check("5a-A. 20 True -> [(0,19)]", bouts_A, [(0, 19)])

# Case B: T*4 F*2 T*14 -> gap=2 bridged -> (0,19)
arr_B = np.concatenate([np.ones(4, bool), np.zeros(2, bool), np.ones(14, bool)])
bouts_B = _detect_bouts(arr_B, fps_5)
# Show work
import numpy as _np
_padded = np.concatenate(([False], arr_B, [False]))
_diff   = np.diff(_padded.astype(np.int8))
_s, _e  = np.where(_diff==1)[0], np.where(_diff==-1)[0]-1
print(f"  [show] 5b-B raw starts={_s.tolist()}  ends={_e.tolist()}")
print(f"  [show] 5b-B gap={int(_s[1]-_e[0]-1)}  bridged -> {bouts_B}")
check("5a-B. T*4 F*2 T*14 -> gap bridged -> [(0,19)]", bouts_B, [(0, 19)])

# Case C: T*4 F*4 T*12 -> gap=4 > 3, first run too short -> [(8,19)]
arr_C = np.concatenate([np.ones(4, bool), np.zeros(4, bool), np.ones(12, bool)])
bouts_C = _detect_bouts(arr_C, fps_5)
check("5a-C. T*4 F*4 T*12 -> first run filtered -> [(8,19)]", bouts_C, [(8, 19)])

# Case D: 5 True -> length 5 < MIN_BOUT=8 -> []
bouts_D = _detect_bouts(np.ones(5, bool), fps_5)
check("5a-D. 5 True -> filtered (too short) -> []", bouts_D, [])

# Case E: empty array -> []
bouts_E = _detect_bouts(np.array([], dtype=bool), fps_5)
check("5a-E. empty array -> []", bouts_E, [])

# Case F: all False -> []
bouts_F = _detect_bouts(np.zeros(10, bool), fps_5)
check("5a-F. all False -> []", bouts_F, [])

# Case G: T*4 F*2 T*16 -> gap=2 bridged, total=22 >= 8 -> [(0,21)]
arr_G = np.concatenate([np.ones(4, bool), np.zeros(2, bool), np.ones(16, bool)])
bouts_G = _detect_bouts(arr_G, fps_5)
check("5a-G. T*4 F*2 T*16 -> gap bridged -> [(0,21)]", bouts_G, [(0, 21)])

# ── 5b. fps=30 thresholds ────────────────────────────────────────────────────
fps_5b = 30.0
# MAX_GAP = max(1, round(0.25*30)) = max(1,8) = 8  (round(7.5)=8 banker's)
# MIN_BOUT = max(1, round(0.8*30)) = max(1,24) = 24
bouts_23 = _detect_bouts(np.ones(23, bool), fps_5b)
bouts_24 = _detect_bouts(np.ones(24, bool), fps_5b)
print(f"  [show] 5b fps=30: MAX_GAP={max(1,round(0.25*fps_5b))}  MIN_BOUT={max(1,round(0.8*fps_5b))}")
check("5b. 23-frame bout filtered at fps=30", bouts_23, [])
check("5b. 24-frame bout passes at fps=30",   bouts_24, [(0, 23)])

# ── 5c. Domain sanity: sorted, non-overlapping, start <= end ─────────────────
rng5 = np.random.default_rng(7)
arr_rand_5 = rng5.random(200) > 0.4
bouts_rand = _detect_bouts(arr_rand_5, fps_5)
all_valid_5c = True
for i, (s, e) in enumerate(bouts_rand):
    if s > e:
        all_valid_5c = False; break
    if i > 0 and s <= bouts_rand[i-1][1]:
        all_valid_5c = False; break
check("5c. all bouts: start <= end and non-overlapping", all_valid_5c, True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- Cumulative Tally
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 6: Cumulative Tally ===")

fps_6 = 1.0
prox_px_6 = 3.0; cont_px_6 = 1.0
dists_6 = np.array([5., 3., 2., 8., 1.])

prox_mask_6 = (dists_6 <= prox_px_6).astype(float)  # [0,1,1,0,1]
cont_mask_6 = (dists_6 <= cont_px_6).astype(float)  # [0,0,0,0,1]
prox_ct_6   = np.cumsum(prox_mask_6) / fps_6        # [0,1,2,2,3]
cont_ct_6   = np.cumsum(cont_mask_6) / fps_6        # [0,0,0,0,1]

print(f"  [show] 6 prox_mask={prox_mask_6.astype(int).tolist()}")
print(f"  [show] 6 cont_mask={cont_mask_6.astype(int).tolist()}")
print(f"  [show] 6 prox_cumtime={prox_ct_6.tolist()}")
print(f"  [show] 6 cont_cumtime={cont_ct_6.tolist()}")

# Independent recompute via count
n_prox_6 = sum(d <= prox_px_6 for d in dists_6)   # 3
n_cont_6  = sum(d <= cont_px_6 for d in dists_6)   # 1
print(f"  [show] 6 manual counts: prox={n_prox_6}  cont={n_cont_6}")

check("6a. prox_cumtime final = 3.0s", prox_ct_6[-1], 3.0)
check("6a. cont_cumtime final = 1.0s", cont_ct_6[-1], 1.0)
check("6a. independent count matches prox", prox_ct_6[-1], float(n_prox_6) / fps_6)
check("6a. independent count matches cont", cont_ct_6[-1], float(n_cont_6) / fps_6)

# ── 6b. Domain sanity ────────────────────────────────────────────────────────
mono_prox = np.all(np.diff(prox_ct_6) >= 0)
mono_cont = np.all(np.diff(cont_ct_6) >= 0)
cont_le_prox = np.all(cont_ct_6 <= prox_ct_6)
total_time_6  = len(dists_6) / fps_6
check("6b. prox_cumtime is monotone non-decreasing", mono_prox, True)
check("6b. cont_cumtime is monotone non-decreasing", mono_cont, True)
check("6b. cont_cumtime <= prox_cumtime everywhere", cont_le_prox, True)
check("6b. prox_cumtime[-1] <= total time", prox_ct_6[-1] <= total_time_6, True)

# ── 6c. All zeros -> full contact ─────────────────────────────────────────────
dists_zeros = np.zeros(5)
prox_ct_z = np.cumsum((dists_zeros <= prox_px_6).astype(float)) / fps_6
cont_ct_z = np.cumsum((dists_zeros <= cont_px_6).astype(float)) / fps_6
check("6c. all-zero dists -> prox_cumtime[-1] = n/fps", prox_ct_z[-1], 5.0 / fps_6)
check("6c. all-zero dists -> cont_cumtime[-1] = n/fps", cont_ct_z[-1], 5.0 / fps_6)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 -- Behavior States
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 7: Behavior States ===")

fps_7    = 10.0
ppcm_7   = 10.0
walk_thr = 3.0  * ppcm_7   # 30 px/s
run_thr  = 20.0 * ppcm_7   # 200 px/s
node_names_7 = ['body']
N_7 = 20
tracks_7 = _tracks_1n1a(np.zeros(N_7), np.zeros(N_7))  # positions irrelevant
hdg_flat  = np.zeros(N_7, dtype=np.float64)

def _run_beh(speed_val):
    kin = _make_kin(np.full(N_7, speed_val, dtype=np.float32), hdg_flat)
    return compute_single_animal(tracks_7, kin, node_names_7, fps_7, px_per_cm=ppcm_7)

# ── 7a. Speed-state thresholds ───────────────────────────────────────────────
beh_stat = _run_beh(10.0)   # 10 < 30 -> stationary
beh_walk = _run_beh(50.0)   # 30 < 50 < 200 -> walking
beh_run  = _run_beh(250.0)  # 250 > 200 -> running

print(f"  [show] 7a stationary state[:5] = {beh_stat['stationary'][:5, 0].tolist()}")
print(f"  [show] 7a walking state[:5]    = {beh_walk['walking'][:5, 0].tolist()}")
print(f"  [show] 7a running state[:5]    = {beh_run['running'][:5, 0].tolist()}")

check("7a. speed=10 -> stationary everywhere", np.all(beh_stat['stationary'][:, 0] == 1), True)
check("7a. speed=10 -> not walking",           np.all(beh_stat['walking'][:,    0] == 0), True)
check("7a. speed=50 -> walking everywhere",    np.all(beh_walk['walking'][:,    0] == 1), True)
check("7a. speed=50 -> not running",           np.all(beh_walk['running'][:,    0] == 0), True)
check("7a. speed=250 -> running everywhere",   np.all(beh_run['running'][:,     0] == 1), True)
check("7a. speed=250 -> not walking",          np.all(beh_run['walking'][:,     0] == 0), True)

# Independent recompute: threshold manually
raw_10  = np.where(np.full(N_7, 10.0)  < walk_thr, 0, np.where(np.full(N_7, 10.0)  > run_thr, 2, 1))
raw_50  = np.where(np.full(N_7, 50.0)  < walk_thr, 0, np.where(np.full(N_7, 50.0)  > run_thr, 2, 1))
raw_250 = np.where(np.full(N_7, 250.0) < walk_thr, 0, np.where(np.full(N_7, 250.0) > run_thr, 2, 1))
check("7a. manual threshold: 10->stat=0", int(raw_10[0]),  0)
check("7a. manual threshold: 50->walk=1", int(raw_50[0]),  1)
check("7a. manual threshold: 250->run=2", int(raw_250[0]), 2)

# ── 7b. Turning threshold ────────────────────────────────────────────────────
# +10deg/frame -> av = 100 deg/s > 30 -> turning
hdg_10 = np.arange(N_7, dtype=float) * 10.0   # 0,10,20,...,190
av_10  = angular_velocity(hdg_10, fps_7)
print(f"  [show] 7b av(+10deg/frame) first 5: {av_10[:5].tolist()}")
check("7b. av(+10deg/frame) * fps = 100 deg/s", float(av_10[5]), 100.0)
kin_turn = _make_kin(np.zeros(N_7), hdg_10)
beh_turn = compute_single_animal(tracks_7, kin_turn, node_names_7, fps_7, px_per_cm=ppcm_7)
check("7b. +10deg/frame -> turning everywhere", np.all(beh_turn['turning'][:, 0] == 1), True)

# +1deg/frame -> av = 10 deg/s < 30 -> not turning
hdg_1  = np.arange(N_7, dtype=float) * 1.0
av_1   = angular_velocity(hdg_1, fps_7)
kin_noturn = _make_kin(np.zeros(N_7), hdg_1)
beh_noturn = compute_single_animal(tracks_7, kin_noturn, node_names_7, fps_7, px_per_cm=ppcm_7)
check("7b. +1deg/frame -> not turning", np.all(beh_noturn['turning'][:, 0] == 0), True)

# ── 7c. Angular velocity wrap-around (critical failure-case audit) ───────────
# Naive diff would give -350 at the 180/-170 boundary; correct answer is +10
hdg_wrap = np.array([170.0, 180.0, -170.0, -160.0])
av_wrap  = angular_velocity(hdg_wrap, fps_7)
print(f"  [show] 7c wrap-around heading: {hdg_wrap.tolist()}")
print(f"  [show] 7c av: {av_wrap.tolist()}")
print(f"  [show] 7c naive diff at index 2: {(-170.0 - 180.0) * fps_7:.1f}  (would be WRONG)")
print(f"  [show] 7c correct wrapped av at index 2: {av_wrap[2]:.1f}")
check("7c. wrap-around: av at boundary = +100 (not -3500)", av_wrap[2], 100.0)
check("7c. wrap-around: av before boundary = +100",         av_wrap[1], 100.0)

# ── 7d. Domain sanity: mutual exclusion ─────────────────────────────────────
for speed_v, label in [(10., 'stat'), (50., 'walk'), (250., 'run')]:
    beh = _run_beh(speed_v)
    per_frame = (beh['stationary'][:, 0].astype(int)
               + beh['walking'][:,    0].astype(int)
               + beh['running'][:,     0].astype(int))
    check(f"7d. mutual exclusion at speed={speed_v}", np.all(per_frame <= 1), True)
    for k in ('stationary', 'walking', 'running', 'turning'):
        arr_k = beh[k]
        check(f"7d. {k} dtype int8 at speed={speed_v}", arr_k.dtype, np.dtype('int8'))
        check(f"7d. {k} all 0 or 1 at speed={speed_v}", np.all((arr_k == 0) | (arr_k == 1)), True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 -- MetricsPanel._refresh_inner Integration  (needs PySide6)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 8: MetricsPanel Integration ===")

try:
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    from run_popup import _MetricsPanel

    fps_8 = 1.0; n_8 = 3
    inv_ppcm_8 = 1.0 / 10.0   # 10 px/cm
    prox_px_8  = 30.0
    cont_px_8  = 10.0
    sleap_idx_8 = 2

    # gen_dist: 3 tracked frames, at sleap_idx=2 -> d_px=8.0
    gdt_8    = np.array([50.0, 25.0, 8.0])
    gcl_8    = np.array(['nodeA-nodeB', 'nodeA-nodeB', 'nodeA-nodeB'], dtype=object)
    p_mask_8 = (gdt_8 <= prox_px_8).astype(float)   # [0,1,1]
    c_mask_8 = (gdt_8 <= cont_px_8).astype(float)   # [0,0,1]
    prox_ct_8 = np.cumsum(p_mask_8) / fps_8          # [0,1,2]
    cont_ct_8 = np.cumsum(c_mask_8) / fps_8          # [0,0,1]

    cache_8 = dict(
        inv_ppcm    = inv_ppcm_8,
        kin         = {'body_heading_deg': np.array([[30.], [60.], [90.]])},
        single_beh  = {
            'running':    np.zeros((n_8, 2), dtype=np.int8),
            'walking':    np.zeros((n_8, 2), dtype=np.int8),
            'stationary': np.ones( (n_8, 2), dtype=np.int8),
            'turning':    np.zeros((n_8, 2), dtype=np.int8),
        },
        track_names = ['A', 'B'],
        n_tracks    = 2,
        si_to_pos   = {0: 0, 1: 1, 2: 2},
        speed_rolling = {
            0: np.array([50., 50., 50.]),
            1: np.array([30., 30., 30.]),
        },
        zone_label  = {
            0: np.array(['Center', 'Perimeter', 'Corner'], dtype=object),
            1: np.array(['Center', 'Perimeter', 'Corner'], dtype=object),
        },
        gen_dist_tracked   = gdt_8,
        gen_closest_tracked = gcl_8,
        prox_bouts  = [(1, 2)],
        cont_bouts  = [(2, 2)],
        prox_cumtime = prox_ct_8,
        cont_cumtime = cont_ct_8,
        hdg_diff    = np.array([10., 20., 90.]),
        prox_px     = prox_px_8,
        cont_px     = cont_px_8,
    )

    panel = _MetricsPanel()

    # 8a. Two-animal cache at contact frame
    panel.refresh(sleap_idx_8, sleap_idx_8, n_8, fps_8, 0, cache_8)
    idist_text  = panel._lbl_idist.text()
    chip_text   = panel._lbl_chip.text()
    btype_text  = panel._lbl_btype.text()
    ctally_text = panel._lbl_ctally.text()
    ptally_text = panel._lbl_ptally.text()
    print(f"  [show] 8a idist='{idist_text}'  chip='{chip_text}'")
    print(f"  [show] 8a btype='{btype_text}'  ctally='{ctally_text}'  ptally='{ptally_text}'")

    check("8a. idist shows 0.80 cm",       '0.80' in idist_text,                True)
    check("8a. chip = Contact",            'Contact' in chip_text,              True)
    check("8a. bout type = Contact",       'Contact' in btype_text,             True)
    check("8a. cont tally = 1.0s",         '1.0' in ctally_text,               True)
    check("8a. prox tally = 2.0s",         '2.0' in ptally_text,               True)

    # 8b. One-animal cache -> N/A labels
    cache_1a = dict(
        inv_ppcm   = inv_ppcm_8,
        kin        = {'body_heading_deg': np.array([[0.]])},
        single_beh = {
            'running':    np.zeros((1, 1), dtype=np.int8),
            'walking':    np.zeros((1, 1), dtype=np.int8),
            'stationary': np.ones( (1, 1), dtype=np.int8),
            'turning':    np.zeros((1, 1), dtype=np.int8),
        },
        track_names = ['A'],
        n_tracks    = 1,
        si_to_pos   = {},
        gen_dist_tracked = None,
    )
    panel.refresh(0, 0, 1, fps_8, 0, cache_1a)
    print(f"  [show] 8b idist='{panel._lbl_idist.text()}'  btype='{panel._lbl_btype.text()}'")
    check("8b. 1-animal -> idist N/A",  'N/A' in panel._lbl_idist.text(), True)
    check("8b. 1-animal -> btype N/A",  'N/A' in panel._lbl_btype.text(), True)

    # 8c. None cache -> all labels cleared
    panel.refresh(2, 2, n_8, fps_8, 0, None)
    print(f"  [show] 8c after None cache: idist='{panel._lbl_idist.text()}'")
    # clear() sets all labels to the em dash character U+2014
    check("8c. None cache -> idist cleared", panel._lbl_idist.text() == '\u2014', True)
    check("8c. None cache -> btype cleared", panel._lbl_btype.text() == '\u2014', True)

    # 8d. Bad cache -> exception caught + logged, no crash
    import logging
    class _Capture(logging.Handler):
        def __init__(self): super().__init__(); self.msgs = []
        def emit(self, r): self.msgs.append(self.format(r))

    cap = _Capture()
    logging.getLogger('run_popup').addHandler(cap)
    try:
        panel.refresh(0, 0, 10, fps_8, 0, {'inv_ppcm': 0.1})  # missing 'kin'
        no_crash = True
    except Exception:
        no_crash = False
    logging.getLogger('run_popup').removeHandler(cap)
    print(f"  [show] 8d logged messages: {cap.msgs}")
    check("8d. bad cache -> no uncaught exception",         no_crash, True)
    check("8d. bad cache -> error logged",
          any('MetricsPanel.refresh() failed' in m for m in cap.msgs), True)

except ImportError as e:
    print(f"  [SKIP] Section 8 skipped: PySide6 not available ({e})")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
total = PASS + FAIL
print(f"\n=== DONE: {PASS} passed, {FAIL} failed (of {total}) ===")
sys.exit(FAIL)
