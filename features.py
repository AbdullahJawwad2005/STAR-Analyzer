"""
features.py — Primitive & Derivative Feature Extraction for STAR Analyzer
==========================================================================
Pure-function module. No Qt, no I/O.

Produces two DataFrames:
  Animal Features — one row per (video_frame × track)
  Pair Features   — one row per (video_frame × pair)

Data conventions:
    tracks    : (n_frames, 2, n_nodes, n_tracks)   axis-1: 0=x, 1=y
    kin       : dict of (n_frames, n_nodes, n_tracks) arrays
    frame_map : {video_frame_idx -> sleap_data_idx}
"""

from itertools import combinations

import numpy as np
import pandas as pd

try:
    from scipy.spatial import ConvexHull as _ConvexHull
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from behaviors import find_node_idx, angular_velocity


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cm_pos(tracks, body_idx, t):
    """(n_frames, 2)  centre-of-mass position for one track."""
    if body_idx is not None:
        x = tracks[:, 0, body_idx, t]
        y = tracks[:, 1, body_idx, t]
    else:
        x = np.nanmean(tracks[:, 0, :, t], axis=1)
        y = np.nanmean(tracks[:, 1, :, t], axis=1)
    return np.stack([x, y], axis=1)


def _heading_deg(tracks, kin, body_idx, t):
    """Body-axis heading in degrees for one track."""
    h = kin['body_heading_deg'][:, t].astype(np.float64)
    return np.nan_to_num(h, nan=0.0)


def _circular_diff_deg(angles):
    """Frame-by-frame circular difference → (n_frames,) in deg."""
    d = np.diff(angles)
    d = (d + 180.0) % 360.0 - 180.0
    out = np.empty_like(angles)
    out[1:] = d
    out[0]  = out[1] if len(out) > 1 else 0.0
    return out


def _smooth_heading_from_pos(pos):
    """Heading in degrees via np.gradient on (n_frames, 2) position."""
    vx = np.gradient(pos[:, 0])
    vy = np.gradient(pos[:, 1])
    return np.degrees(np.arctan2(vy, vx))


# ---------------------------------------------------------------------------
# Per-node features
# ---------------------------------------------------------------------------

def _node_total_displacement(tracks, t):
    """Cumulative path length per node. Returns dict node_idx -> (n_frames,)."""
    n_frames, _, n_nodes, _ = tracks.shape
    out = {}
    for n in range(n_nodes):
        x = tracks[:, 0, n, t]
        y = tracks[:, 1, n, t]
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        step = np.hypot(dx, dy)
        step = np.nan_to_num(step, nan=0.0)
        out[n] = np.cumsum(step)
    return out


# ---------------------------------------------------------------------------
# Node-to-node features
# ---------------------------------------------------------------------------

def _node_pair_features(tracks, node_names, t):
    """
    For every unique pair (i, j) of nodes within track t, compute:
      - distance               (pixels)
      - angle                  (degrees, angle of j relative to i)
      - angular_motion         (deg/frame)

    Note on ang_mot (angular motion):
        ang_mot is the frame-by-frame change in the inter-node angle (circular
        difference in degrees).  It captures fine-grained limb swings, head turns,
        and local joint rotations at the resolution of individual node pairs — at
        finer temporal and spatial resolution than the whole-body heading derived
        from the centre-of-mass trajectory, which only captures gross locomotor
        direction changes.

    Returns dict keyed by '{name_i}_to_{name_j}_{metric}'.
    """
    n_frames, _, n_nodes, _ = tracks.shape
    out = {}

    for i, j in combinations(range(n_nodes), 2):
        ni = node_names[i].replace(' ', '_')
        nj = node_names[j].replace(' ', '_')
        pfx = f'{ni}_to_{nj}'

        dx = tracks[:, 0, j, t] - tracks[:, 0, i, t]
        dy = tracks[:, 1, j, t] - tracks[:, 1, i, t]

        dist  = np.hypot(dx, dy)
        angle = np.degrees(np.arctan2(dy, dx))
        ang_m = _circular_diff_deg(angle)

        out[f'{pfx}_dist']    = dist
        out[f'{pfx}_angle']   = angle
        out[f'{pfx}_ang_mot'] = ang_m

    return out


# ---------------------------------------------------------------------------
# Shape features
# ---------------------------------------------------------------------------

def _shape_features_per_track(tracks, t):
    """
    Per-frame shape descriptors from the cloud of all node positions.

    Returns dict with arrays (n_frames,):
        elongation, eccentricity, compactness, circularity
    """
    n_frames, _, n_nodes, _ = tracks.shape
    elongation   = np.full(n_frames, np.nan)
    eccentricity = np.full(n_frames, np.nan)
    compactness  = np.full(n_frames, np.nan)
    circularity  = np.full(n_frames, np.nan)

    # Extract all node positions for this track: (n_frames, n_nodes, 2)
    pts_all = np.stack([tracks[:, 0, :, t], tracks[:, 1, :, t]], axis=-1)
    # Valid mask: (n_frames, n_nodes)
    valid_mask = np.all(np.isfinite(pts_all), axis=-1)
    n_valid = valid_mask.sum(axis=1)  # (n_frames,)

    # --- Elongation & eccentricity via closed-form 2x2 eigenvalues ---
    frames_ge2 = np.where(n_valid >= 2)[0]
    if len(frames_ge2) > 0:
        # For frames with >= 2 valid points, compute covariance matrix components
        # Set invalid points to NaN so they don't affect mean
        pts_masked = pts_all.copy()
        pts_masked[~valid_mask] = np.nan

        # Mean position per frame (ignoring NaN)
        mu = np.nanmean(pts_masked, axis=1)  # (n_frames, 2)
        # Centre points
        ctr = pts_masked - mu[:, np.newaxis, :]  # (n_frames, n_nodes, 2)

        # Covariance components: a = var(x), c = var(y), b = cov(x,y)
        # Using nanmean over nodes (dividing by n_valid, not n_valid-1)
        cx = ctr[:, :, 0]  # (n_frames, n_nodes)
        cy = ctr[:, :, 1]
        a = np.nanmean(cx**2, axis=1)  # (n_frames,)
        c = np.nanmean(cy**2, axis=1)
        b = np.nanmean(cx * cy, axis=1)

        # Closed-form eigenvalues of 2x2 symmetric matrix [[a,b],[b,c]]
        half_trace = (a + c) / 2.0
        disc = np.sqrt(np.maximum(((a - c) / 2.0)**2 + b**2, 0.0))
        ev0 = half_trace + disc   # larger eigenvalue
        ev1 = half_trace - disc   # smaller eigenvalue

        maj = np.sqrt(np.maximum(ev0, 1e-12))
        mn  = np.sqrt(np.maximum(ev1, 1e-12))

        elongation[frames_ge2]   = (maj / mn)[frames_ge2]
        ratio = mn / maj
        eccentricity[frames_ge2] = np.sqrt(np.maximum(0.0, 1.0 - ratio**2))[frames_ge2]

    # --- Compactness & circularity via ConvexHull (kept per-frame, fast on 7 pts) ---
    if _SCIPY_OK:
        frames_ge3 = np.where(n_valid >= 3)[0]
        for f in frames_ge3:
            pts = pts_all[f][valid_mask[f]]
            try:
                hull = _ConvexHull(pts)
                area = hull.volume    # scipy 2D: volume = area
                peri = hull.area      # scipy 2D: area   = perimeter
                if area > 1e-10:
                    circularity[f]  = 4.0 * np.pi * area / peri ** 2
                    compactness[f]  = peri ** 2 / (4.0 * np.pi * area)
            except Exception:
                pass

    return dict(elongation=elongation, eccentricity=eccentricity,
                compactness=compactness, circularity=circularity)


def _tri_area(ax, ay, bx, by, cx, cy):
    """Signed triangle area via cross-product (vectorised). Returns (n_frames,)."""
    return 0.5 * np.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))


def _hourglass_triangles(tracks, node_names, t):
    """Upper (cm-earL-earR) and lower (cm-hipL-hipR) triangle areas.

    Returns (upper, lower) each (n_frames,). NaN where required nodes missing.
    """
    n_frames = tracks.shape[0]
    cm_idx  = find_node_idx(node_names, 'body')
    el_idx  = find_node_idx(node_names, 'ear_l')
    er_idx  = find_node_idx(node_names, 'ear_r')
    hl_idx  = find_node_idx(node_names, 'hip_l')
    hr_idx  = find_node_idx(node_names, 'hip_r')

    nan_arr = np.full(n_frames, np.nan)

    # Upper triangle: cm, ear_l, ear_r
    if cm_idx is not None and el_idx is not None and er_idx is not None:
        cx = tracks[:, 0, cm_idx, t]; cy = tracks[:, 1, cm_idx, t]
        elx = tracks[:, 0, el_idx, t]; ely = tracks[:, 1, el_idx, t]
        erx = tracks[:, 0, er_idx, t]; ery = tracks[:, 1, er_idx, t]
        upper = _tri_area(cx, cy, elx, ely, erx, ery)
        bad = np.isnan(cx) | np.isnan(elx) | np.isnan(erx)
        upper[bad] = np.nan
    else:
        upper = nan_arr.copy()

    # Lower triangle: cm, hip_l, hip_r
    if cm_idx is not None and hl_idx is not None and hr_idx is not None:
        cx = tracks[:, 0, cm_idx, t]; cy = tracks[:, 1, cm_idx, t]
        hlx = tracks[:, 0, hl_idx, t]; hly = tracks[:, 1, hl_idx, t]
        hrx = tracks[:, 0, hr_idx, t]; hry = tracks[:, 1, hr_idx, t]
        lower = _tri_area(cx, cy, hlx, hly, hrx, hry)
        bad = np.isnan(cx) | np.isnan(hlx) | np.isnan(hrx)
        lower[bad] = np.nan
    else:
        lower = nan_arr.copy()

    return upper, lower


# ---------------------------------------------------------------------------
# Path efficiency
# ---------------------------------------------------------------------------

def _path_efficiency(cm_x, cm_y, fps):
    """
    Rolling 3-second path efficiency: straight-line / cumulative path.
    Values in [0, 1]; 1.0 = perfectly straight.

    Formula: efficiency[f] = straight_line_distance / cumulative_path_length,
    where both quantities are measured over the W-frame window ending at frame f
    (W = round(3 * fps), i.e. approximately 3 seconds of history).
    Straight-line distance = Euclidean distance from the position W frames ago
    to the current position.
    Cumulative path length = sum of per-frame step distances over the same window.

    Interpretation:
        1.0  — the animal walked in a perfectly straight line over the window.
        ~0   — highly tortuous movement: the animal was circling, pausing and
               reversing, or otherwise covering very little net ground relative
               to total distance travelled.
    """
    n = len(cm_x)
    W = max(1, int(round(3 * fps)))

    dx   = np.diff(cm_x, prepend=cm_x[0])
    dy   = np.diff(cm_y, prepend=cm_y[0])
    step = np.nan_to_num(np.hypot(dx, dy), nan=0.0)

    cumpath      = np.cumsum(step)
    pad_cum      = np.pad(cumpath, (W, 0), constant_values=0.0)
    path_in_win  = cumpath - pad_cum[:n]

    pad_x   = np.pad(cm_x, (W, 0), mode='edge')
    pad_y   = np.pad(cm_y, (W, 0), mode='edge')
    straight = np.hypot(cm_x - pad_x[:n], cm_y - pad_y[:n])

    with np.errstate(invalid='ignore', divide='ignore'):
        eff = np.where(path_in_win > 1e-6, straight / path_in_win, 1.0)
    return np.clip(eff, 0.0, 1.0)


def _speed_accel(speed_arr, fps):
    """d(speed)/dt — rate of change of speed magnitude. (n_frames,)"""
    clean = np.nan_to_num(speed_arr, nan=0.0)
    return np.gradient(clean) * fps


# ---------------------------------------------------------------------------
# ROI-based features
# ---------------------------------------------------------------------------

def _roi_distances(cm_x, cm_y, roi):
    """Distance to ROI centre and nearest ROI edge. Both (n_frames,)."""
    if roi is None:
        nan = np.full(len(cm_x), np.nan)
        return nan, nan.copy()

    rx0, ry0, rx1, ry1 = roi
    cx, cy = (rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0

    dist_center   = np.hypot(cm_x - cx, cm_y - cy)
    # distance to nearest edge (0 if on or outside boundary)
    d_l = cm_x - rx0
    d_r = rx1  - cm_x
    d_t = cm_y - ry0
    d_b = ry1  - cm_y
    dist_boundary = np.maximum(0.0, np.minimum(np.minimum(d_l, d_r),
                                               np.minimum(d_t, d_b)))
    return dist_center, dist_boundary


def _position_entropy(cm_x, cm_y, roi, n_bins=10):
    """
    Normalised Shannon entropy of spatial occupancy (session-level scalar).
    0 = always in one cell; 1 = perfectly uniform.

    The 2D arena ROI is divided into a 10 x 10 grid of cells (100 cells total).
    Shannon entropy of the occupancy histogram is computed once over the entire
    session and normalised by log2(100) so the result is in [0, 1].
    The single scalar is then broadcast to ALL frames via np.full(n_frames, ent),
    so every frame in the export will carry the same entropy value for a given track.
    """
    if roi is None:
        return np.nan

    rx0, ry0, rx1, ry1 = roi
    ok = np.isfinite(cm_x) & np.isfinite(cm_y)
    x  = cm_x[ok];  y = cm_y[ok]
    if len(x) == 0:
        return np.nan

    counts, _, _ = np.histogram2d(x, y, bins=n_bins,
                                   range=[[rx0, rx1], [ry0, ry1]])
    total = counts.sum()
    if total == 0:
        return np.nan
    p   = counts[counts > 0] / total
    H   = -float(np.sum(p * np.log2(p)))
    return round(H / np.log2(n_bins ** 2), 4)   # normalise to [0,1]


# ---------------------------------------------------------------------------
# Pairwise features
# ---------------------------------------------------------------------------

def _pair_features(tracks, kin, node_names, fps):
    """
    Pairwise features for every unique (tA, tB) pair.

    Returns dict keyed 'tA_tB/metric' → (n_frames,) arrays.
    """
    n_frames, _, n_nodes, n_tracks = tracks.shape
    if n_tracks < 2:
        return {}

    body_idx = find_node_idx(node_names, 'body')
    out = {}

    for tA, tB in combinations(range(n_tracks), 2):
        pfx = f't{tA}_t{tB}'

        cm_A = _cm_pos(tracks, body_idx, tA)
        cm_B = _cm_pos(tracks, body_idx, tB)

        hdg_A = kin['body_heading_deg'][:, tA]
        hdg_B = kin['body_heading_deg'][:, tB]

        # --- Inter-animal distance & displacement ---
        inter = np.hypot(cm_A[:, 0] - cm_B[:, 0], cm_A[:, 1] - cm_B[:, 1])
        out[f'{pfx}/inter_animal_dist']         = inter
        out[f'{pfx}/inter_animal_displacement'] = np.diff(inter, prepend=inter[0])

        # --- Position covariance & correlation ---
        # pos_covariance_x[f] = (xA[f] - mean_xA) * (xB[f] - mean_xB)
        # where mean_xA / mean_xB are the SESSION-WIDE means of each track's x position.
        # Averaged over all frames this product gives the Pearson covariance between
        # xA and xB across the session.  However, stored at the frame level it reflects
        # session-wide positional drift (e.g. both animals spending time on the same
        # side of the arena), not local moment-to-moment co-movement.
        # The binned export replaces these frame products with proper within-bin Pearson
        # covariance (_bin_cov_corr in binned_export.py) which captures genuine local
        # synchrony within each 0.25 s window.  The frame-level columns here are kept for
        # reference and for the non-binned feature export.
        xA = cm_A[:, 0];  yA = cm_A[:, 1]
        xB = cm_B[:, 0];  yB = cm_B[:, 1]
        cxA = xA - np.nanmean(xA);  cyA = yA - np.nanmean(yA)
        cxB = xB - np.nanmean(xB);  cyB = yB - np.nanmean(yB)

        out[f'{pfx}/pos_covariance_x'] = cxA * cxB
        out[f'{pfx}/pos_covariance_y'] = cyA * cyB

        sxA = np.nanstd(xA);  sxB = np.nanstd(xB)
        syA = np.nanstd(yA);  syB = np.nanstd(yB)
        corr_x = (cxA * cxB / (sxA * sxB)) if sxA > 0 and sxB > 0 else np.zeros(n_frames)
        corr_y = (cyA * cyB / (syA * syB)) if syA > 0 and syB > 0 else np.zeros(n_frames)
        out[f'{pfx}/pos_correlation_x'] = corr_x
        out[f'{pfx}/pos_correlation_y'] = corr_y

        # --- Approach angles ---
        vec_A2B  = cm_B - cm_A
        dir_A2B  = np.degrees(np.arctan2(vec_A2B[:, 1], vec_A2B[:, 0]))
        app_A    = np.abs(hdg_A - dir_A2B) % 360
        app_A    = np.minimum(app_A, 360 - app_A)
        out[f'{pfx}/approach_angle_A'] = app_A

        vec_B2A  = cm_A - cm_B
        dir_B2A  = np.degrees(np.arctan2(vec_B2A[:, 1], vec_B2A[:, 0]))
        app_B    = np.abs(hdg_B - dir_B2A) % 360
        app_B    = np.minimum(app_B, 360 - app_B)
        out[f'{pfx}/approach_angle_B'] = app_B

        # --- Velocity cosine similarity ---
        vxA = np.gradient(cm_A[:, 0]); vyA = np.gradient(cm_A[:, 1])
        vxB = np.gradient(cm_B[:, 0]); vyB = np.gradient(cm_B[:, 1])
        dot = vxA * vxB + vyA * vyB
        mag = np.hypot(vxA, vyA) * np.hypot(vxB, vyB)
        safe_mag = np.where(mag > 1e-12, mag, 1.0)
        out[f'{pfx}/velocity_cos_sim'] = np.where(mag > 1e-12, dot / safe_mag, 0.0)

        # --- Visual scope (A sees B, B sees A) ---
        # Binocular < 20°, Monocular < 120°, None otherwise
        vis_A = np.full(n_frames, 'None',      dtype=object)
        vis_A[app_A < 120] = 'Monocular'
        vis_A[app_A <  20] = 'Binocular'
        out[f'{pfx}/visual_scope_A'] = vis_A

        vis_B = np.full(n_frames, 'None',      dtype=object)
        vis_B[app_B < 120] = 'Monocular'
        vis_B[app_B <  20] = 'Binocular'
        out[f'{pfx}/visual_scope_B'] = vis_B

        # --- Auditory scope ---
        # Binaural < 60°, Monaural 60–150°, Rear > 150°
        aud_A = np.full(n_frames, 'Rear',      dtype=object)
        aud_A[app_A < 150] = 'Monaural'
        aud_A[app_A <  60] = 'Binaural'
        out[f'{pfx}/auditory_scope_A'] = aud_A

        aud_B = np.full(n_frames, 'Rear',      dtype=object)
        aud_B[app_B < 150] = 'Monaural'
        aud_B[app_B <  60] = 'Binaural'
        out[f'{pfx}/auditory_scope_B'] = aud_B

    return out


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def precompute_feature_arrays(tracks, kin, node_names, fps, roi=None):
    """
    Compute all feature arrays indexed by sleap frame index.

    Returns
    -------
    track_arrays : dict[int, dict[str, np.ndarray]]
        track_arrays[t][feature_name] -> (n_frames,) array
    pair_arrays  : dict[str, np.ndarray]
        't{A}_t{B}/feature_name' -> (n_frames,) array
    """
    n_frames, _, n_nodes, n_tracks = tracks.shape
    body_idx = find_node_idx(node_names, 'body')

    track_arrays = {}
    for t in range(n_tracks):
        fa = {}

        for n, nn in enumerate(node_names):
            nn_c = nn.replace(' ', '_')
            fa[f'{nn_c}_x'] = tracks[:, 0, n, t]
            fa[f'{nn_c}_y'] = tracks[:, 1, n, t]
            fa[f'{nn_c}_speed']  = kin['speed'][:, n, t]
            fa[f'{nn_c}_vx']     = kin['vx'][:, n, t]
            fa[f'{nn_c}_vy']     = kin['vy'][:, n, t]
            fa[f'{nn_c}_accel']  = kin['accel'][:, n, t]
            fa[f'{nn_c}_jerk']   = kin['jerk'][:, n, t]

        node_disp = _node_total_displacement(tracks, t)
        for n, nn in enumerate(node_names):
            nn_c = nn.replace(' ', '_')
            fa[f'{nn_c}_total_disp'] = node_disp[n]

        fa.update(_node_pair_features(tracks, node_names, t))
        fa.update(_shape_features_per_track(tracks, t))

        upper, lower = _hourglass_triangles(tracks, node_names, t)
        fa['hourglass_area'] = np.where(
            np.isnan(upper) & np.isnan(lower), np.nan,
            np.nan_to_num(upper, nan=0.0) + np.nan_to_num(lower, nan=0.0))
        fa['hourglass_ratio'] = np.where(lower > 1e-10, upper / lower, np.nan)

        hdg = _heading_deg(tracks, kin, body_idx, t)
        fa['curvature'] = np.abs(angular_velocity(hdg, fps))

        cm = _cm_pos(tracks, body_idx, t)
        cm_x = np.nan_to_num(
            cm[:, 0],
            nan=float(np.nanmean(cm[:, 0])) if np.any(np.isfinite(cm[:, 0])) else 0.0)
        cm_y = np.nan_to_num(
            cm[:, 1],
            nan=float(np.nanmean(cm[:, 1])) if np.any(np.isfinite(cm[:, 1])) else 0.0)
        fa['path_efficiency'] = _path_efficiency(cm_x, cm_y, fps)

        body_spd = (kin['speed'][:, body_idx, t] if body_idx is not None
                    else np.nanmean(kin['speed'][:, :, t], axis=1))
        fa['speed_accel'] = _speed_accel(body_spd, fps)
        fa['body_heading_deg'] = kin['body_heading_deg'][:, t]

        fa['cm_total_disp'] = (node_disp[body_idx] if body_idx is not None
                               else np.mean([node_disp[n] for n in range(n_nodes)], axis=0))
        fa['dist_roi_center'], fa['dist_roi_boundary'] = _roi_distances(
            cm[:, 0], cm[:, 1], roi)
        ent = _position_entropy(cm[:, 0], cm[:, 1], roi)
        fa['position_entropy'] = np.full(n_frames, ent)

        track_arrays[t] = fa

    pair_arrays = _pair_features(tracks, kin, node_names, fps)
    return track_arrays, pair_arrays


# ---------------------------------------------------------------------------
# Key Metrics summary (one-glance overview sheet)
# ---------------------------------------------------------------------------

def build_key_metrics_df(tracks, kin, single_beh, pair_beh,
                         track_arrays, pair_arrays, frame_map,
                         zone_summary_df, node_names, track_names,
                         fps, px_per_cm):
    """
    Build a long-format DataFrame with essential behavioural summary metrics.

    Columns: Category, Metric, Subject, Value, Unit
    """
    n_frames, _, n_nodes, n_tracks = tracks.shape
    session_frames = len(frame_map)
    session_dur = session_frames / fps  # seconds

    body_idx = find_node_idx(node_names, 'body')
    nose_idx = find_node_idx(node_names, 'nose')
    ear_l_idx = find_node_idx(node_names, 'ear_l')
    ear_r_idx = find_node_idx(node_names, 'ear_r')
    tail_idx = find_node_idx(node_names, 'tail')

    # Only use frames that are in the frame_map (valid sleap indices)
    sleap_idxs = np.array(sorted(frame_map.values()))

    rows = []

    def _add(cat, metric, subject, value, unit):
        rows.append(dict(Category=cat, Metric=metric, Subject=subject,
                         Value=round(value, 4) if np.isfinite(value) else value,
                         Unit=unit))

    # ------------------------------------------------------------------
    # Per-animal metrics
    # ------------------------------------------------------------------
    for t, tname in enumerate(track_names):
        ta = track_arrays[t]

        # Total distance (final cumulative displacement of body node, convert to cm)
        total_disp_px = ta['cm_total_disp'][sleap_idxs[-1]] if len(sleap_idxs) else 0.0
        _add('Locomotion', 'Total Distance Traveled', tname,
             total_disp_px / px_per_cm, 'cm')

        # Body speed stats (convert px/frame·fps → already in px/s via kin)
        if body_idx is not None:
            spd = kin['speed'][sleap_idxs, body_idx, t]
        else:
            spd = np.nanmean(kin['speed'][sleap_idxs, :, t], axis=1)
        spd_cm = spd / px_per_cm
        _add('Locomotion', 'Avg Speed', tname, float(np.nanmean(spd_cm)), 'cm/s')
        _add('Locomotion', 'Median Speed', tname, float(np.nanmedian(spd_cm)), 'cm/s')
        _add('Locomotion', 'P95 Speed', tname, float(np.nanpercentile(spd_cm, 95)), 'cm/s')

        # Acceleration
        if body_idx is not None:
            acc = kin['accel'][sleap_idxs, body_idx, t]
        else:
            acc = np.nanmean(kin['accel'][sleap_idxs, :, t], axis=1)
        _add('Locomotion', 'Avg Acceleration', tname,
             float(np.nanmean(acc / px_per_cm)), 'cm/s²')

        # Immobility
        if 'stationary' in single_beh:
            stat_frames = float(np.nansum(single_beh['stationary'][sleap_idxs, t]))
            imm_time = stat_frames / fps
            _add('Immobility', 'Immobility Time', tname, imm_time, 's')
            _add('Immobility', 'Immobility %', tname,
                 imm_time / session_dur * 100 if session_dur > 0 else 0.0, '%')

        # Zone times from zone_summary_df
        if zone_summary_df is not None and not zone_summary_df.empty:
            track_zones = zone_summary_df[zone_summary_df['Track'] == tname]
            # Center = Open zone
            open_row = track_zones[track_zones['Zone'] == 'Open']
            center_t = float(open_row['Time in Zone (s)'].sum()) if len(open_row) else 0.0
            _add('Zone', 'Center Zone Time', tname, center_t, 's')
            # Perimeter = W1-W4
            wall_t = float(track_zones[track_zones['Zone'].isin(
                ['W1','W2','W3','W4'])]['Time in Zone (s)'].sum())
            _add('Zone', 'Perimeter Zone Time', tname, wall_t, 's')
            # Corner = C1-C4
            corner_t = float(track_zones[track_zones['Zone'].isin(
                ['C1','C2','C3','C4'])]['Time in Zone (s)'].sum())
            _add('Zone', 'Corner Zone Time', tname, corner_t, 's')

    # ------------------------------------------------------------------
    # Per-pair metrics (proximity & contact)
    # ------------------------------------------------------------------
    if n_tracks < 2:
        return pd.DataFrame(rows, columns=['Category','Metric','Subject','Value','Unit'])

    prox_thresh_cm = 3.0
    contact_thresh_cm = 1.0
    prox_thresh_px = prox_thresh_cm * px_per_cm
    contact_thresh_px = contact_thresh_cm * px_per_cm

    for tA in range(n_tracks):
        for tB in range(tA + 1, n_tracks):
            pair_name = f'{track_names[tA]} & {track_names[tB]}'
            pfx = f't{tA}_t{tB}'

            # Body-body distance (from pair_arrays)
            inter_key = f'{pfx}/inter_animal_dist'
            if inter_key in pair_arrays:
                inter = pair_arrays[inter_key][sleap_idxs]
            else:
                # Fallback: compute from body node
                cm_A = _cm_pos(tracks, body_idx, tA)[sleap_idxs]
                cm_B = _cm_pos(tracks, body_idx, tB)[sleap_idxs]
                inter = np.hypot(cm_A[:, 0] - cm_B[:, 0], cm_A[:, 1] - cm_B[:, 1])

            def _node_pair_dist(idx_a, idx_b):
                """Euclidean distance between node idx_a on tA and idx_b on tB."""
                xa = tracks[sleap_idxs, 0, idx_a, tA]
                ya = tracks[sleap_idxs, 1, idx_a, tA]
                xb = tracks[sleap_idxs, 0, idx_b, tB]
                yb = tracks[sleap_idxs, 1, idx_b, tB]
                return np.hypot(xa - xb, ya - yb)

            def _head_centroid(t_idx):
                """Average of ear_l and ear_r positions; fall back to nose."""
                if ear_l_idx is not None and ear_r_idx is not None:
                    x = (tracks[sleap_idxs, 0, ear_l_idx, t_idx] +
                         tracks[sleap_idxs, 0, ear_r_idx, t_idx]) / 2.0
                    y = (tracks[sleap_idxs, 1, ear_l_idx, t_idx] +
                         tracks[sleap_idxs, 1, ear_r_idx, t_idx]) / 2.0
                    return x, y
                if nose_idx is not None:
                    return (tracks[sleap_idxs, 0, nose_idx, t_idx],
                            tracks[sleap_idxs, 1, nose_idx, t_idx])
                return None, None

            # Build distance arrays for each node-pair type
            dist_pairs = {}
            dist_pairs['Body-Body'] = inter

            if nose_idx is not None:
                dist_pairs['Nose-Nose'] = _node_pair_dist(nose_idx, nose_idx)

            hx_A, hy_A = _head_centroid(tA)
            hx_B, hy_B = _head_centroid(tB)
            if hx_A is not None and hx_B is not None:
                dist_pairs['Head-Head'] = np.hypot(hx_A - hx_B, hy_A - hy_B)

            if nose_idx is not None and body_idx is not None:
                # min of (noseA→bodyB, noseB→bodyA)
                d1 = _node_pair_dist(nose_idx, body_idx)
                # Reverse: nose of tB to body of tA
                xa = tracks[sleap_idxs, 0, nose_idx, tB]
                ya = tracks[sleap_idxs, 1, nose_idx, tB]
                xb = tracks[sleap_idxs, 0, body_idx, tA]
                yb = tracks[sleap_idxs, 1, body_idx, tA]
                d2 = np.hypot(xa - xb, ya - yb)
                dist_pairs['Nose-Body'] = np.minimum(d1, d2)

            if nose_idx is not None and tail_idx is not None:
                d1 = _node_pair_dist(nose_idx, tail_idx)
                xa = tracks[sleap_idxs, 0, nose_idx, tB]
                ya = tracks[sleap_idxs, 1, nose_idx, tB]
                xb = tracks[sleap_idxs, 0, tail_idx, tA]
                yb = tracks[sleap_idxs, 1, tail_idx, tA]
                d2 = np.hypot(xa - xb, ya - yb)
                dist_pairs['Nose-Tail'] = np.minimum(d1, d2)

            # Proximity and contact times
            for label, dist_arr in dist_pairs.items():
                valid = np.isfinite(dist_arr)
                prox_frames = float(np.nansum(dist_arr[valid] <= prox_thresh_px))
                contact_frames = float(np.nansum(dist_arr[valid] <= contact_thresh_px))
                _add('Proximity', f'{label} Proximity Time', pair_name,
                     prox_frames / fps, 's')
                _add('Contact', f'{label} Contact Time', pair_name,
                     contact_frames / fps, 's')

            # Mean angle when proximal (body-body ≤ 3cm)
            app_A_key = f'{pfx}/approach_angle_A'
            app_B_key = f'{pfx}/approach_angle_B'
            if app_A_key in pair_arrays and app_B_key in pair_arrays:
                prox_mask = np.isfinite(inter) & (inter <= prox_thresh_px)
                if np.any(prox_mask):
                    app_A = pair_arrays[app_A_key][sleap_idxs][prox_mask]
                    app_B = pair_arrays[app_B_key][sleap_idxs][prox_mask]
                    mean_angle = float(np.nanmean(app_A + app_B))
                else:
                    mean_angle = float('nan')
                _add('Proximity', 'Mean Angle When Proximal', pair_name,
                     mean_angle, '°')

    return pd.DataFrame(rows, columns=['Category','Metric','Subject','Value','Unit'])


def build_proximity_orientation_df(tracks, kin, frame_map, node_names, track_names, fps, px_per_cm):
    """
    Build a per-second proximity and orientation DataFrame for the first animal pair.

    Columns: Time(s), Within_3cm, Heading_Angle_deg, <node>_dist_cm for each
    canonical node present in node_names.

    Only meaningful when n_tracks >= 2; caller is responsible for that check.
    """
    n_frames, _, n_nodes, n_tracks = tracks.shape

    body_idx  = find_node_idx(node_names, 'body')
    nose_idx  = find_node_idx(node_names, 'nose')
    ear_l_idx = find_node_idx(node_names, 'ear_l')
    ear_r_idx = find_node_idx(node_names, 'ear_r')
    hip_l_idx = find_node_idx(node_names, 'hip_l')
    hip_r_idx = find_node_idx(node_names, 'hip_r')
    tail_idx  = find_node_idx(node_names, 'tail')

    # Canonical node order for distance columns
    canonical_nodes = [
        ('nose',   nose_idx,  'nose_dist_cm'),
        ('ear_l',  ear_l_idx, 'ear_l_dist_cm'),
        ('ear_r',  ear_r_idx, 'ear_r_dist_cm'),
        ('body',   body_idx,  'body_dist_cm'),
        ('hip_l',  hip_l_idx, 'hip_l_dist_cm'),
        ('hip_r',  hip_r_idx, 'hip_r_dist_cm'),
        ('tail',   tail_idx,  'tail_dist_cm'),
    ]

    # Use first pair (tA=0, tB=1)
    tA, tB = 0, 1

    heading_A = kin['body_heading_deg'][:, tA]
    heading_B = kin['body_heading_deg'][:, tB]
    delta_heading = np.abs(((heading_A - heading_B) + 180.0) % 360.0 - 180.0)  # [0, 180]

    # Centroid distance (pixels) for Within_3cm
    if body_idx is not None:
        cx_A = tracks[:, 0, body_idx, tA]; cy_A = tracks[:, 1, body_idx, tA]
        cx_B = tracks[:, 0, body_idx, tB]; cy_B = tracks[:, 1, body_idx, tB]
    else:
        cx_A = np.nanmean(tracks[:, 0, :, tA], axis=1)
        cy_A = np.nanmean(tracks[:, 1, :, tA], axis=1)
        cx_B = np.nanmean(tracks[:, 0, :, tB], axis=1)
        cy_B = np.nanmean(tracks[:, 1, :, tB], axis=1)
    centroid_dist_px = np.hypot(cx_A - cx_B, cy_A - cy_B)

    # Group frame_map into 1-second bins keyed by integer second
    bins = {}
    for vid_frame, sleap_idx in frame_map.items():
        sec = int(vid_frame // fps)
        bins.setdefault(sec, []).append(sleap_idx)

    rows = []
    for sec in sorted(bins):
        idxs = np.array(bins[sec])

        # Within_3cm: binary — mean centroid distance that second < 3 cm
        cd = centroid_dist_px[idxs]
        mean_dist_cm = float(np.nanmean(cd)) / px_per_cm
        within_3 = 1 if mean_dist_cm < 3.0 else 0

        # Heading angle
        ha = float(np.nanmean(delta_heading[idxs]))

        row = {'Time(s)': sec, 'Within_3cm': within_3, 'Heading_Angle_deg': round(ha, 2)}

        # Per-node distances
        for _cname, idx, col in canonical_nodes:
            if idx is None:
                continue
            xa = tracks[idxs, 0, idx, tA]; ya = tracks[idxs, 1, idx, tA]
            xb = tracks[idxs, 0, idx, tB]; yb = tracks[idxs, 1, idx, tB]
            d_px = np.hypot(xa - xb, ya - yb)
            row[col] = round(float(np.nanmean(d_px)) / px_per_cm, 4)

        rows.append(row)

    # Build ordered columns
    col_order = ['Time(s)', 'Within_3cm', 'Heading_Angle_deg']
    for _cname, idx, col in canonical_nodes:
        if idx is not None:
            col_order.append(col)

    return pd.DataFrame(rows, columns=col_order)


def build_feature_dataframes(tracks, kin, node_names, track_names, fps,
                             frame_map, roi=None, _precomputed=None):
    """
    Compute all primitive & derivative features and return two DataFrames.

    Parameters
    ----------
    tracks       : (n_frames, 2, n_nodes, n_tracks)
    kin          : dict from preprocessing.compute_kinematics
    node_names   : list[str]
    track_names  : list[str]
    fps          : float
    frame_map    : {video_frame_idx -> sleap_data_idx}
    roi          : (rx0, ry0, rx1, ry1) in pixels, or None
    _precomputed : optional (track_arrays, pair_arrays) tuple returned by
                   precompute_feature_arrays() — avoids recomputing when the
                   caller already has these (e.g. for the binned export).

    Returns
    -------
    animal_df : pd.DataFrame  — one row per (video_frame × track)
    pair_df   : pd.DataFrame  — one row per (video_frame × pair)
    """
    n_frames, _, n_nodes, n_tracks = tracks.shape

    # _precomputed is an optional performance shortcut.
    # This parameter was added to avoid recomputing heavy feature arrays when the
    # caller (run_popup.py _run_export) already has them available.  Both the main
    # per-frame export and the binned export (binned_export.py) require the same
    # track_arrays and pair_arrays, so the caller computes them once via
    # precompute_feature_arrays() and passes the result here and to build_025s_bins().
    # For large sessions with many nodes and tracks this avoids a second full pass
    # through all kinematics and pairwise geometry, which can be several seconds of
    # compute time.
    if _precomputed is not None:
        track_arrays, pair_arrays = _precomputed
    else:
        track_arrays, pair_arrays = precompute_feature_arrays(tracks, kin, node_names, fps, roi)

    # Build prefix -> (tA, tB) mapping for pairs
    pair_prefixes = {}
    for tA, tB in combinations(range(n_tracks), 2):
        pair_prefixes[f't{tA}_t{tB}'] = (tA, tB)

    # -----------------------------------------------------------------------
    # Assemble Animal Features DataFrame (vectorized column-first)
    # -----------------------------------------------------------------------

    sorted_items = sorted(frame_map.items())
    vid_frames = np.array([vf for vf, _ in sorted_items])
    sleap_idxs = np.array([si for _, si in sorted_items])
    n_mapped = len(vid_frames)
    n_t = len(track_names)

    # Repeat/tile for (n_mapped * n_t) rows
    col_frame = np.repeat(vid_frames, n_t)
    col_time = np.round(np.repeat(vid_frames.astype(np.float64), n_t) / fps, 4)
    track_tile = np.tile(np.arange(n_t), n_mapped)
    col_track = np.array(track_names)[track_tile]
    sleap_rep = np.repeat(sleap_idxs, n_t)

    animal_cols = {'Frame': col_frame, 'Time(s)': col_time, 'Track': col_track}

    # Get feature names from first track
    feat_names = list(track_arrays[0].keys())
    for feat_name in feat_names:
        # Stack arrays from all tracks: (n_sleap_frames, n_tracks)
        stacked = np.column_stack([track_arrays[t][feat_name] for t in range(n_t)])
        # Index: for each output row, pick sleap_rep[i] row and track_tile[i] column
        vals = stacked[sleap_rep, track_tile]
        animal_cols[feat_name] = vals

    animal_df = pd.DataFrame(animal_cols)

    # -----------------------------------------------------------------------
    # Assemble Pair Features DataFrame (vectorized column-first)
    # -----------------------------------------------------------------------

    if pair_prefixes and pair_arrays:
        n_pairs = len(pair_prefixes)
        pfx_list = list(pair_prefixes.keys())
        pair_info = list(pair_prefixes.values())  # list of (tA, tB)

        col_p_frame = np.repeat(vid_frames, n_pairs)
        col_p_time = np.round(np.repeat(vid_frames.astype(np.float64), n_pairs) / fps, 4)
        pair_tile = np.tile(np.arange(n_pairs), n_mapped)

        col_trackA = np.array([track_names[pair_info[p][0]] if pair_info[p][0] < len(track_names)
                               else f't{pair_info[p][0]}' for p in range(n_pairs)])[pair_tile]
        col_trackB = np.array([track_names[pair_info[p][1]] if pair_info[p][1] < len(track_names)
                               else f't{pair_info[p][1]}' for p in range(n_pairs)])[pair_tile]

        pair_cols = {'Frame': col_p_frame, 'Time(s)': col_p_time,
                     'Track_A': col_trackA, 'Track_B': col_trackB}

        sleap_p_rep = np.repeat(sleap_idxs, n_pairs)

        # Group pair_arrays by prefix
        feat_by_pfx = {}
        for key in pair_arrays:
            kpfx, feat = key.rsplit('/', 1)
            feat_by_pfx.setdefault(feat, {})[kpfx] = pair_arrays[key]

        for feat, pfx_dict in feat_by_pfx.items():
            # Build array: (n_sleap_frames, n_pairs) — stack in pfx_list order
            cols_list = []
            for pfx in pfx_list:
                if pfx in pfx_dict:
                    cols_list.append(pfx_dict[pfx])
                else:
                    cols_list.append(np.full(tracks.shape[0], np.nan))
            stacked = np.column_stack(cols_list)
            pair_cols[feat] = stacked[sleap_p_rep, pair_tile]

        pair_df = pd.DataFrame(pair_cols)
    else:
        pair_df = pd.DataFrame()

    return animal_df, pair_df
