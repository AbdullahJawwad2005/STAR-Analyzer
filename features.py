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
    """CM heading in degrees for one track."""
    if body_idx is not None:
        h = kin['heading_deg'][:, body_idx, t].astype(np.float64)
    else:
        h = np.nanmean(kin['heading_deg'][:, :, t], axis=1).astype(np.float64)
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

    for f in range(n_frames):
        pts = tracks[f, :, :, t].T          # (n_nodes, 2)
        ok  = np.all(np.isfinite(pts), axis=1)
        pts = pts[ok]
        nv  = len(pts)

        if nv < 2:
            continue

        # PCA applied to the (n_nodes x 2) point cloud for this frame.
        # The 2x2 covariance matrix of the centred node positions has two
        # eigenvalues: the larger one (ev[0]) is the variance along the major
        # body axis (the direction the animal is most spread out), and the
        # smaller one (ev[1]) is the variance along the minor body axis.
        # Taking the square root of each eigenvalue gives the semi-axis lengths
        # (analogous to the semi-axes of an equivalent ellipse).
        # Elongation = major / minor axis ratio: > 1 means the posture is
        # elongated / stretched; approaching 1 means compact / curled.
        # Eccentricity follows the standard ellipse definition:
        #   e = sqrt(1 - (minor/major)^2), ranging from 0 (circle) to 1 (line).
        mu  = pts.mean(axis=0)
        ctr = pts - mu
        cov = (ctr.T @ ctr) / nv
        ev  = np.sort(np.linalg.eigvalsh(cov))[::-1]   # descending
        maj = np.sqrt(max(ev[0], 1e-12))
        mn  = np.sqrt(max(ev[1], 1e-12))
        elongation[f]   = maj / mn
        eccentricity[f] = np.sqrt(max(0.0, 1.0 - (mn / maj) ** 2))

        # Compactness and circularity both measure how "round" the body shape is,
        # but they are reciprocals and interpreted from opposite directions:
        #   Circularity  = 4 * pi * A / P^2  — ranges from 0 to 1;
        #                  1.0 = perfect circle, decreases as the shape becomes
        #                  more elongated or irregular (more perimeter for the same area).
        #   Compactness  = P^2 / (4 * pi * A)  — the reciprocal of circularity;
        #                  minimum value 1.0 for a circle, increases toward infinity
        #                  for highly irregular or elongated shapes.
        # Using both in the feature set gives downstream models two complementary
        # representations of the same underlying geometric property.
        if nv >= 3 and _SCIPY_OK:
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
    cm_idx  = find_node_idx(node_names, 'center', 'body', 'cm', 'centroid')
    el_idx  = find_node_idx(node_names, 'ear_l', 'el')
    er_idx  = find_node_idx(node_names, 'ear_r', 'er')
    hl_idx  = find_node_idx(node_names, 'hip_l', 'hl')
    hr_idx  = find_node_idx(node_names, 'hip_r', 'hr')

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

    Important — this is a SESSION-LEVEL scalar, not a time-varying quantity:
    The 2D arena ROI is divided into a 10 x 10 grid of cells (100 cells total).
    Shannon entropy of the occupancy histogram is computed once over the entire
    session and normalised by log2(100) so the result is in [0, 1].
    The single scalar is then broadcast to ALL frames via np.full(n_frames, ent),
    so every frame in the export will carry the same entropy value for a given track.
    In the binned export, every bin will therefore have the same entropy column value.
    This is intentional — position_entropy is a global summary statistic that
    characterises the animal's space-use diversity for the whole session, not a
    moment-by-moment measure.  Use dist_roi_center / dist_roi_boundary for
    time-varying spatial context.
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

    body_idx = find_node_idx(node_names, 'center', 'body', 'cm', 'centroid')
    out = {}

    for tA, tB in combinations(range(n_tracks), 2):
        pfx = f't{tA}_t{tB}'

        cm_A = _cm_pos(tracks, body_idx, tA)
        cm_B = _cm_pos(tracks, body_idx, tB)

        hdg_A = _smooth_heading_from_pos(cm_A)
        hdg_B = _smooth_heading_from_pos(cm_B)

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
        out[f'{pfx}/velocity_cos_sim'] = np.where(mag > 1e-12, dot / mag, 0.0)

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
    body_idx = find_node_idx(node_names, 'center', 'body', 'cm', 'centroid')

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

        fa['cm_total_disp'] = (node_disp[body_idx] if body_idx is not None
                               else np.mean([node_disp[n] for n in range(n_nodes)], axis=0))
        fa['dist_roi_center'], fa['dist_roi_boundary'] = _roi_distances(
            cm[:, 0], cm[:, 1], roi)
        ent = _position_entropy(cm[:, 0], cm[:, 1], roi)
        fa['position_entropy'] = np.full(n_frames, ent)

        track_arrays[t] = fa

    pair_arrays = _pair_features(tracks, kin, node_names, fps)
    return track_arrays, pair_arrays


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
    # Assemble Animal Features DataFrame
    # -----------------------------------------------------------------------

    animal_rows = []
    for vid_frame, sleap_idx in sorted(frame_map.items()):
        time_s = round(vid_frame / fps, 4)
        for t, tname in enumerate(track_names):
            row = {'Frame': vid_frame, 'Time(s)': time_s, 'Track': tname}
            for feat_name, arr in track_arrays[t].items():
                val = arr[sleap_idx]
                try:
                    row[feat_name] = '' if pd.isna(val) else val
                except (TypeError, ValueError):
                    row[feat_name] = val
            animal_rows.append(row)

    animal_df = pd.DataFrame(animal_rows)

    # -----------------------------------------------------------------------
    # Assemble Pair Features DataFrame
    # -----------------------------------------------------------------------

    pair_rows = []
    for vid_frame, sleap_idx in sorted(frame_map.items()):
        time_s = round(vid_frame / fps, 4)
        for pfx, (tA, tB) in pair_prefixes.items():
            nA = track_names[tA] if tA < len(track_names) else f't{tA}'
            nB = track_names[tB] if tB < len(track_names) else f't{tB}'
            row = {'Frame': vid_frame, 'Time(s)': time_s,
                   'Track_A': nA, 'Track_B': nB}
            for key, arr in pair_arrays.items():
                kpfx, feat = key.rsplit('/', 1)
                if kpfx != pfx:
                    continue
                val = arr[sleap_idx]
                try:
                    row[feat] = '' if pd.isna(val) else val
                except (TypeError, ValueError):
                    row[feat] = val
            pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame()

    return animal_df, pair_df
