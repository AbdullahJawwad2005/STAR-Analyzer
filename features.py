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

from behaviors import find_node_idx, angular_velocity, _fill_short_gaps

PROX_THRESHOLD_CM    = 3.0   # general proximity threshold (cm)
CONTACT_THRESHOLD_CM = 1.0   # general contact threshold (cm)


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
    """d(speed)/dt — rate of change of speed magnitude. (n_frames,)
    NaN-safe: gap-boundary frames produce NaN output, not artificial spikes."""
    return np.gradient(np.asarray(speed_arr, dtype=np.float64)) * fps


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
# General minimum cross-animal distance
# ---------------------------------------------------------------------------

def _general_min_dist(tracks, idxs, tA, tB, node_names, exclude_idxs=None):
    """
    Per-frame minimum distance across ALL cross-animal node pairs.

    Parameters
    ----------
    exclude_idxs : set of int or None
        Node indices to exclude from both animals (e.g. tailend).

    Returns
    -------
    min_dist    : (len(idxs),) float — pixels; NaN where all nodes missing
    closest_pair: (len(idxs),) object — 'nodeA-nodeB' label of minimising pair
    """
    n_nodes = tracks.shape[2]
    active = [i for i in range(n_nodes) if exclude_idxs is None or i not in exclude_idxs]

    xA = tracks[idxs][:, 0, :, tA][:, active]          # (n_idxs, n_active)
    yA = tracks[idxs][:, 1, :, tA][:, active]
    xB = tracks[idxs][:, 0, :, tB][:, active]
    yB = tracks[idxs][:, 1, :, tB][:, active]

    n_active = len(active)
    dx = xA[:, :, np.newaxis] - xB[:, np.newaxis, :]  # (n_idxs, n_active, n_active)
    dy = yA[:, :, np.newaxis] - yB[:, np.newaxis, :]
    dists = np.hypot(dx, dy)
    flat = dists.reshape(len(idxs), -1)                # (n_idxs, n_active*n_active)

    with np.errstate(all='ignore'):
        min_dist = np.nanmin(flat, axis=1)

    has_data = np.any(np.isfinite(flat), axis=1)
    flat_safe = np.where(np.isfinite(flat), flat, np.inf)
    argmin_idx = np.argmin(flat_safe, axis=1)

    i_idx = argmin_idx // n_active
    j_idx = argmin_idx % n_active
    names = np.array([node_names[k].replace(' ', '_') for k in active])
    closest_pair = np.where(
        has_data,
        np.array([f'{names[i]}-{names[j]}' for i, j in zip(i_idx, j_idx)], dtype=object),
        ''
    )
    return min_dist, closest_pair


def _region_min_dist_slab(xA, yA, xB, yB, nodes_a, nodes_b):
    """Min cross-animal region distance from pre-extracted (n, n_nodes) slabs."""
    xa = xA[:, nodes_a]; ya = yA[:, nodes_a]
    xb = xB[:, nodes_b]; yb = yB[:, nodes_b]
    dx = xa[:, :, np.newaxis] - xb[:, np.newaxis, :]
    dy = ya[:, :, np.newaxis] - yb[:, np.newaxis, :]
    d = np.hypot(dx, dy).reshape(len(xA), -1)
    if set(nodes_a) == set(nodes_b):
        with np.errstate(all='ignore'):
            return np.nanmin(d, axis=1)
    # asymmetric: also compute reverse swap
    dx2 = xB[:, nodes_a][:, :, np.newaxis] - xA[:, nodes_b][:, np.newaxis, :]
    dy2 = yB[:, nodes_a][:, :, np.newaxis] - yA[:, nodes_b][:, np.newaxis, :]
    d2 = np.hypot(dx2, dy2).reshape(len(xA), -1)
    with np.errstate(all='ignore'):
        return np.nanmin(np.concatenate([d, d2], axis=1), axis=1)


def _region_min_dist(tracks, idxs, tA, tB, nodes_a, nodes_b):
    """
    Minimum cross-animal distance between two anatomical regions.

    For symmetric pairs (nodes_a == nodes_b), computes nodes_a(tA)×nodes_b(tB).
    For asymmetric pairs, also computes nodes_a(tB)×nodes_b(tA) and takes overall min.
    """
    sub = tracks[idxs]
    xA = sub[:, 0, :, tA]; yA = sub[:, 1, :, tA]
    xB = sub[:, 0, :, tB]; yB = sub[:, 1, :, tB]
    return _region_min_dist_slab(xA, yA, xB, yB, nodes_a, nodes_b)


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
        fa['speed_accel'] = np.abs(_speed_accel(body_spd, fps))
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
# Bout detection helper
# ---------------------------------------------------------------------------

def _tailend_node_idxs(node_names):
    """Return a set of node indices whose stripped name matches tailend aliases.

    If a tailstart node is present, bare 'tail' nodes are also treated as
    tailend (handles skeletons that name the end node 'tail1' etc.).
    """
    _tailend_pats = {'tailend', 'tail_end', 'te'}
    stripped = [n.lower().rstrip('0123456789') for n in node_names]
    _tailstart_pats = {'tailstart', 'tail_base', 'tailbase', 'tb'}
    if any(s in _tailstart_pats for s in stripped):
        _tailend_pats = _tailend_pats | {'tail'}
    out = set()
    for i, s in enumerate(stripped):
        if s in _tailend_pats:
            out.add(i)
    return out


def _detect_bouts(binary, fps):
    """
    Convert a boolean (or 0/1) array to a list of bout (start, end) index pairs.

    Rules
    -----
    - A bout is a contiguous run of True values.
    - Gaps of <= MAX_GAP_FRAMES consecutive False frames are bridged.
    - After bridging, bouts shorter than MIN_BOUT_FRAMES are discarded.

    Parameters
    ----------
    binary : (N,) bool array  — True = frame meets threshold
    fps    : float

    Returns
    -------
    bouts : list of (start, end) inclusive index pairs (in terms of binary's indices)
    """
    MAX_GAP_FRAMES = max(1, round(0.25 * fps))
    MIN_BOUT_FRAMES = max(1, round(0.8 * fps))

    arr = np.asarray(binary, dtype=bool)
    n = len(arr)
    if n == 0 or not np.any(arr):
        return []

    # Find run boundaries
    padded = np.concatenate(([False], arr, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]   # rising edges
    ends   = np.where(diff == -1)[0] - 1  # falling edges (inclusive)

    # Bridge gaps
    merged_starts = [starts[0]]
    merged_ends   = [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        gap = s - merged_ends[-1] - 1
        if gap <= MAX_GAP_FRAMES:
            merged_ends[-1] = e
        else:
            merged_starts.append(s)
            merged_ends.append(e)

    # Filter by minimum duration
    bouts = [(s, e) for s, e in zip(merged_starts, merged_ends)
             if (e - s + 1) >= MIN_BOUT_FRAMES]
    return bouts


def _bout_stats(dist_arr, thresh_px, fps):
    """
    Return (total_time_s, bout_count, mean_bout_s) for frames where dist <= thresh.
    dist_arr may contain NaN; NaN frames count as above-threshold (not in bout).
    """
    binary = np.isfinite(dist_arr) & (dist_arr <= thresh_px)
    bouts = _detect_bouts(binary, fps)
    if not bouts:
        return 0.0, 0, float('nan')
    durations = [(e - s + 1) / fps for s, e in bouts]
    return sum(durations), len(durations), float(np.mean(durations))


# ---------------------------------------------------------------------------
# Key Metrics bout filters (1-second minimum duration)
# ---------------------------------------------------------------------------

def _apply_state_hold_filter(labels, fps, min_dur_s=1.0):
    """
    State-hold filter for categorical zone labels.

    Short runs (< min_dur_s) are replaced by the nearest preceding confirmed
    state (one that lasted >= min_dur_s).  If no prior confirmed state exists,
    the first subsequent confirmed state is used instead.  If no run ever
    reaches min_dur_s, returns the original array unchanged.

    Parameters
    ----------
    labels  : (N,) array-like of str  — e.g. 'Center', 'Perimeter', 'Corner'
    fps     : float
    min_dur_s : float  — minimum bout duration in seconds (default 1.0)

    Returns
    -------
    filtered : (N,) np.ndarray of str (copy)
    """
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    if n == 0:
        return labels.copy()

    min_frames = round(fps * min_dur_s)

    # Run-length encode
    change_pts = np.where(np.concatenate(([True], labels[1:] != labels[:-1])))[0]
    lengths    = np.diff(np.append(change_pts, n))
    values     = labels[change_pts]

    # Identify confirmed runs
    confirmed = lengths >= min_frames
    if not np.any(confirmed):
        return labels.copy()  # fallback: no run qualifies — return unchanged

    # Build output by assigning each run its confirmed replacement
    out = labels.copy()
    last_confirmed_val = None

    for i, (start, length, val) in enumerate(zip(change_pts, lengths, values)):
        if confirmed[i]:
            last_confirmed_val = val
            # (keep original — no change needed)
        else:
            if last_confirmed_val is not None:
                # Replace with last confirmed state
                out[start:start + length] = last_confirmed_val
            else:
                # No prior confirmed state — find first subsequent one
                future = np.where(confirmed[i + 1:])[0]
                if len(future):
                    out[start:start + length] = values[i + 1 + future[0]]
                # else: leave as-is (shouldn't happen since confirmed has ≥1 True)

    return out


def _filter_short_active_bouts(arr, fps, min_dur_s=1.0):
    """
    Zero out active (True/1) runs shorter than min_dur_s seconds.

    Inactive (0) runs are never modified — no gap bridging.

    Parameters
    ----------
    arr       : (N,) array-like — binary (bool or 0/1)
    fps       : float
    min_dur_s : float  — minimum active-bout duration in seconds (default 1.0)

    Returns
    -------
    filtered : (N,) np.ndarray of float64 (copy)
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0 or not np.any(arr):
        return arr.copy()

    min_frames = round(fps * min_dur_s)

    padded = np.concatenate(([0.0], arr, [0.0]))
    diff   = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]   # rising edges (indices into arr)
    ends   = np.where(diff == -1)[0]  # exclusive end indices into arr

    out = arr.copy()
    for s, e in zip(starts, ends):
        if (e - s) < min_frames:
            out[s:e] = 0.0
    return out


def _compute_bout_second_marks(dist_seq, frame_map, fps, threshold_px):
    """
    Per-second binary marks based on ≥1-second proximity bouts.

    A second is marked 1 if qualifying-bout frames (after gap-fill and min-bout
    filter) cover >50% of that second. On a tie (exactly 50%), the second
    containing the bout start wins (earlier second takes precedence).

    Parameters
    ----------
    dist_seq     : (N,) float — distances in sorted(frame_map) video-frame order
    frame_map    : {vid_frame: sleap_idx}
    fps          : float
    threshold_px : float — distance threshold in pixels

    Returns
    -------
    marks : dict {second_index: 0 or 1}
    """
    sorted_vf = sorted(frame_map.keys())
    dist_seq  = np.asarray(dist_seq, dtype=np.float64)

    # 1. Frame-level binary
    raw = (np.isfinite(dist_seq) & (dist_seq <= threshold_px)).astype(np.int8)

    # 2. Jitter removal: fill gaps < 0.2 s
    max_gap = max(1, round(0.2 * fps))
    filled  = _fill_short_gaps(raw, max_gap)

    # 3. Minimum-bout filter: keep only runs >= 1 second
    qualified = _filter_short_active_bouts(
        filled.astype(np.float64), fps, min_dur_s=1.0).astype(bool)

    # 4. Bout-start positions -> seconds (for tie-breaking)
    padded = np.concatenate(([False], qualified, [False]))
    d = np.diff(padded.astype(np.int8))
    start_secs = {int(sorted_vf[p] // fps) for p in np.where(d == 1)[0]}

    # 5. Per-second mark
    n_exp = round(fps)
    sec_active: dict = {}
    sec_set:    set  = set()
    for pos, vf in enumerate(sorted_vf):
        sec = int(vf // fps)
        sec_set.add(sec)
        if qualified[pos]:
            sec_active[sec] = sec_active.get(sec, 0) + 1

    marks: dict = {}
    for sec in sorted(sec_set):
        n = sec_active.get(sec, 0)
        if n * 2 > n_exp or (n * 2 == n_exp and sec in start_secs):
            marks[sec] = 1
        else:
            marks[sec] = 0
    return marks


# ---------------------------------------------------------------------------
# Key Metrics summary (one-glance overview sheet)
# ---------------------------------------------------------------------------

def build_key_metrics_df(tracks, kin, single_beh, pair_beh,
                         track_arrays, pair_arrays, frame_map,
                         zone_summary_df, node_names, track_names,
                         fps, px_per_cm, zone_label=None):
    """
    Build a long-format DataFrame with essential behavioural summary metrics.

    Columns: Category, Metric, Subject, Value, Unit
    """
    n_frames, _, n_nodes, n_tracks = tracks.shape
    session_frames = len(frame_map)
    session_dur = session_frames / fps  # seconds

    body_idx  = find_node_idx(node_names, 'body')
    nose_idx  = find_node_idx(node_names, 'nose')
    ear_l_idx = find_node_idx(node_names, 'ear_l')
    ear_r_idx = find_node_idx(node_names, 'ear_r')
    hip_l_idx = find_node_idx(node_names, 'hip_l')
    hip_r_idx = find_node_idx(node_names, 'hip_r')
    tail_idx  = find_node_idx(node_names, 'tail')

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

        # Scalar absolute acceleration: |d(speed)/dt|, averaged over all frames
        abs_acc = np.abs(_speed_accel(spd_cm, fps))
        _add('Locomotion', 'Avg Abs Acceleration', tname,
             float(np.nanmean(abs_acc)), 'cm/s²')
        _add('Locomotion', 'P95 Abs Acceleration', tname,
             float(np.nanpercentile(abs_acc, 95)), 'cm/s²')

        # Immobility (1-second minimum bout filter)
        if 'stationary' in single_beh:
            stat_raw = single_beh['stationary'][sleap_idxs, t].astype(np.float64)
            stat_filtered = _filter_short_active_bouts(stat_raw, fps)
            stat_frames = float(np.nansum(stat_filtered))
            imm_time = stat_frames / fps
            _add('Immobility', 'Immobility Time', tname, imm_time, 's')
            _add('Immobility', 'Immobility %', tname,
                 imm_time / session_dur * 100 if session_dur > 0 else 0.0, '%')

        # Zone times (1-second minimum visit filter when zone_label available)
        if zone_label is not None and t in zone_label and zone_label[t] is not None:
            zl = _apply_state_hold_filter(zone_label[t][sleap_idxs], fps)
            center_t = float(np.sum(zl == 'Open')) / fps
            perim_t  = float(np.sum(np.isin(zl, ['W1','W2','W3','W4']))) / fps
            corner_t = float(np.sum(np.isin(zl, ['C1','C2','C3','C4']))) / fps
            _add('Zone', 'Center Zone Time', tname, center_t, 's')
            _add('Zone', 'Perimeter Zone Time', tname, perim_t, 's')
            _add('Zone', 'Corner Zone Time', tname, corner_t, 's')
        elif zone_summary_df is not None and not zone_summary_df.empty:
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

    prox_thresh_px    = PROX_THRESHOLD_CM    * px_per_cm
    contact_thresh_px = CONTACT_THRESHOLD_CM * px_per_cm

    # Extract indexed sub-array once; each pair loop re-uses it
    tracks_sub = tracks[sleap_idxs]          # (n_valid, 2, n_nodes, n_tracks)
    tailend_excl = _tailend_node_idxs(node_names)
    n_nodes_total = tracks.shape[2]
    active_nodes = [i for i in range(n_nodes_total) if i not in tailend_excl]

    # Anatomical region definitions (same for every pair)
    head_nodes = [i for i in [nose_idx, ear_l_idx, ear_r_idx] if i is not None]
    body_nodes = [i for i in [body_idx, hip_l_idx, hip_r_idx] if i is not None]
    tail_nodes  = [i for i in [tail_idx]                       if i is not None]

    for tA in range(n_tracks):
        for tB in range(tA + 1, n_tracks):
            pair_name = f'{track_names[tA]} & {track_names[tB]}'
            pfx = f't{tA}_t{tB}'

            # Per-pair coordinate slabs — single extraction from tracks_sub
            xA = tracks_sub[:, 0, :, tA]   # (n_valid, n_nodes)
            yA = tracks_sub[:, 1, :, tA]
            xB = tracks_sub[:, 0, :, tB]
            yB = tracks_sub[:, 1, :, tB]

            # Body-body distance (from pair_arrays)
            inter_key = f'{pfx}/inter_animal_dist'
            if inter_key in pair_arrays:
                inter = pair_arrays[inter_key][sleap_idxs]
            else:
                # Fallback: compute from body node
                cm_A = _cm_pos(tracks, body_idx, tA)[sleap_idxs]
                cm_B = _cm_pos(tracks, body_idx, tB)[sleap_idxs]
                inter = np.hypot(cm_A[:, 0] - cm_B[:, 0], cm_A[:, 1] - cm_B[:, 1])

            dist_pairs = {}
            if nose_idx is not None:
                dist_pairs['Nose-Nose'] = _region_min_dist_slab(xA, yA, xB, yB, [nose_idx], [nose_idx])
            if head_nodes:
                dist_pairs['Head-Head']  = _region_min_dist_slab(xA, yA, xB, yB, head_nodes, head_nodes)
                if body_nodes:
                    dist_pairs['Head-Body'] = _region_min_dist_slab(xA, yA, xB, yB, head_nodes, body_nodes)
                if tail_nodes:
                    dist_pairs['Head-Tail'] = _region_min_dist_slab(xA, yA, xB, yB, head_nodes, tail_nodes)
            if body_nodes:
                dist_pairs['Body-Body']  = _region_min_dist_slab(xA, yA, xB, yB, body_nodes, body_nodes)
                if tail_nodes:
                    dist_pairs['Body-Tail'] = _region_min_dist_slab(xA, yA, xB, yB, body_nodes, tail_nodes)
            if tail_nodes:
                dist_pairs['Tail-Tail']  = _region_min_dist_slab(xA, yA, xB, yB, tail_nodes, tail_nodes)

            # Proximity and contact times (1-second minimum bout filter)
            for label, dist_arr in dist_pairs.items():
                prox_bin = np.isfinite(dist_arr) & (dist_arr <= prox_thresh_px)
                prox_bin = _filter_short_active_bouts(prox_bin, fps)
                prox_frames = float(np.nansum(prox_bin))

                cont_bin = np.isfinite(dist_arr) & (dist_arr <= contact_thresh_px)
                cont_bin = _filter_short_active_bouts(cont_bin, fps)
                contact_frames = float(np.nansum(cont_bin))

                _add('Proximity', f'{label} Proximity Time', pair_name,
                     prox_frames / fps, 's')
                _add('Contact', f'{label} Contact Time', pair_name,
                     contact_frames / fps, 's')

            # General proximity/contact — minimum over all cross-animal node pairs
            # Exclude tailend nodes: only tailstart counts as a contact/proximity point
            xa_all = xA[:, active_nodes]; ya_all = yA[:, active_nodes]
            xb_all = xB[:, active_nodes]; yb_all = yB[:, active_nodes]
            dx_all = xa_all[:, :, np.newaxis] - xb_all[:, np.newaxis, :]
            dy_all = ya_all[:, :, np.newaxis] - yb_all[:, np.newaxis, :]
            with np.errstate(all='ignore'):
                gen_dist = np.nanmin(np.hypot(dx_all, dy_all).reshape(len(sleap_idxs), -1), axis=1)
            gen_prox_bin = np.isfinite(gen_dist) & (gen_dist <= prox_thresh_px)
            gen_prox_bin = _filter_short_active_bouts(gen_prox_bin, fps)
            _add('Proximity', 'General Proximity Time (2dp)', pair_name,
                 round(float(np.nansum(gen_prox_bin)) / fps, 2), 's')

            gen_cont_bin = np.isfinite(gen_dist) & (gen_dist <= contact_thresh_px)
            gen_cont_bin = _filter_short_active_bouts(gen_cont_bin, fps)
            _add('Contact', 'General Contact Time (2dp)', pair_name,
                 round(float(np.nansum(gen_cont_bin)) / fps, 2), 's')

            # Binned: whole-second counts from the P&O sheet logic (≥1 s bouts, gap-filled)
            prox_marks_km = _compute_bout_second_marks(gen_dist, frame_map, fps, prox_thresh_px)
            cont_marks_km = _compute_bout_second_marks(gen_dist, frame_map, fps, contact_thresh_px)
            _add('Proximity', 'General Proximity Time (1s bouts)', pair_name,
                 float(sum(prox_marks_km.values())), 's')
            _add('Contact', 'General Contact Time (1s bouts)', pair_name,
                 float(sum(cont_marks_km.values())), 's')

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

    # Use first pair (tA=0, tB=1)
    tA, tB = 0, 1

    heading_A = kin['body_heading_deg'][:, tA]
    heading_B = kin['body_heading_deg'][:, tB]
    delta_heading = np.abs(((heading_A - heading_B) + 180.0) % 360.0 - 180.0)  # [0, 180]

    # General min cross-animal node-pair distance for Within_3cm / Within_1cm
    # Exclude tailend nodes: only tailstart counts as a contact/proximity point
    all_idxs = np.arange(n_frames)
    gen_min_px, gen_closest = _general_min_dist(tracks, all_idxs, tA, tB, node_names,
                                                exclude_idxs=_tailend_node_idxs(node_names))
    prox_px    = PROX_THRESHOLD_CM    * px_per_cm
    contact_px = CONTACT_THRESHOLD_CM * px_per_cm

    # Anatomical region distances — precomputed for all frames
    head_nodes = [i for i in [nose_idx, ear_l_idx, ear_r_idx] if i is not None]
    body_nodes = [i for i in [body_idx, hip_l_idx, hip_r_idx] if i is not None]
    tail_nodes  = [i for i in [tail_idx]                       if i is not None]

    region_dist_arrays = {}
    if nose_idx is not None:
        region_dist_arrays['nose_nose_dist_cm'] = _region_min_dist(tracks, all_idxs, tA, tB, [nose_idx], [nose_idx])
    if head_nodes:
        region_dist_arrays['head_head_dist_cm']  = _region_min_dist(tracks, all_idxs, tA, tB, head_nodes, head_nodes)
        if body_nodes:
            region_dist_arrays['head_body_dist_cm'] = _region_min_dist(tracks, all_idxs, tA, tB, head_nodes, body_nodes)
        if tail_nodes:
            region_dist_arrays['head_tail_dist_cm'] = _region_min_dist(tracks, all_idxs, tA, tB, head_nodes, tail_nodes)
    if body_nodes:
        region_dist_arrays['body_body_dist_cm']  = _region_min_dist(tracks, all_idxs, tA, tB, body_nodes, body_nodes)
        if tail_nodes:
            region_dist_arrays['body_tail_dist_cm'] = _region_min_dist(tracks, all_idxs, tA, tB, body_nodes, tail_nodes)
    if tail_nodes:
        region_dist_arrays['tail_tail_dist_cm']  = _region_min_dist(tracks, all_idxs, tA, tB, tail_nodes, tail_nodes)

    # Group frame_map into 1-second bins keyed by integer second
    bins = {}
    for vid_frame, sleap_idx in frame_map.items():
        sec = int(vid_frame // fps)
        bins.setdefault(sec, []).append(sleap_idx)

    # Bout-aware per-second marks for Within_3cm / Within_1cm
    _sorted_vf = sorted(frame_map.keys())
    _dist_seq  = gen_min_px[np.array([frame_map[f] for f in _sorted_vf])]
    prox_marks = _compute_bout_second_marks(_dist_seq, frame_map, fps, prox_px)
    cont_marks = _compute_bout_second_marks(_dist_seq, frame_map, fps, contact_px)

    rows = []
    for sec in sorted(bins):
        idxs = np.array(bins[sec])

        gd       = gen_min_px[idxs]          # still needed for closest_pair below
        within_3 = prox_marks.get(sec, 0)
        within_1 = cont_marks.get(sec, 0)

        # Closest pair: mode among this second's frames
        cp_sec = gen_closest[idxs]
        finite_cp = cp_sec[np.isfinite(gd)]
        if len(finite_cp):
            vals, counts = np.unique(finite_cp, return_counts=True)
            closest = str(vals[np.argmax(counts)])
        else:
            closest = ''

        # Heading angle
        ha = float(np.nanmean(delta_heading[idxs]))

        row = {'Time(s)': sec, 'Within_3cm': within_3, 'Within_1cm': within_1,
               'closest_pair': closest, 'Heading_Angle_deg': round(ha, 2)}

        for col, arr_px in region_dist_arrays.items():
            row[col] = round(float(np.nanmean(arr_px[idxs])) / px_per_cm, 4)

        rows.append(row)

    col_order = ['Time(s)', 'Within_3cm', 'Within_1cm', 'closest_pair', 'Heading_Angle_deg']
    col_order += list(region_dist_arrays.keys())

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
