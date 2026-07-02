"""
behaviors.py — 1st Order Behavior Detection for STAR Analyzer
==============================================================
Pure-function module. No Qt, no I/O.

Data conventions (inherited from preprocessing / run_popup):
    tracks      : (n_frames, 2, n_nodes, n_tracks)  axis-1: 0=x, 1=y
    kin         : dict of (n_frames, n_nodes, n_tracks) arrays
    frame_map   : {video_frame_idx -> sleap_data_idx}
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Node lookup
# ---------------------------------------------------------------------------

# Canonical SLEAP node names (tried first as exact match, ignoring trailing
# digits so "centermass1" matches "centermass").  Order = priority.
_CANONICAL = {
    'body':   ('centermass', 'bodycenter', 'bodycentre', 'center_mass'),
    'nose':   ('nose',),
    'ear_l':  ('earl', 'ear_left', 'ear_l', 'left_ear', 'el'),
    'ear_r':  ('earr', 'ear_right', 'ear_r', 'right_ear', 'er'),
    'hip_l':  ('hindlegl', 'hip_left', 'hip_l', 'left_hip', 'hl'),
    'hip_r':  ('hindlegr', 'hip_right', 'hip_r', 'right_hip', 'hr'),
    'tail':   ('tailstart', 'tail_base', 'tailbase', 'tb'),
}


def _strip_digits(s):
    """Remove trailing digits: 'centermass1' → 'centermass'."""
    return s.rstrip('0123456789')


def find_node_idx(node_names, *patterns):
    """Find a node index by canonical name, then fall back to substring.

    First tries exact canonical matches (stripping trailing digits) for
    the given pattern keys.  Falls back to case-insensitive substring
    search only if no canonical match is found.
    """
    lower = [n.lower() for n in node_names]
    stripped = [_strip_digits(n) for n in lower]

    # --- Pass 1: canonical exact match (stripped of trailing digits) ---
    for pat in patterns:
        canon = _CANONICAL.get(pat)
        if canon:
            for cname in canon:
                for i, s in enumerate(stripped):
                    if s == cname:
                        return i

    # --- Pass 2: substring fallback (original behaviour) ---
    for pat in patterns:
        pat_l = pat.lower()
        for i, name in enumerate(lower):
            if pat_l in name:
                return i
    return None


# ---------------------------------------------------------------------------
# Body-scale reference (DSR)
# ---------------------------------------------------------------------------

def compute_dsr(tracks, hip_l_idx, hip_r_idx):
    """Median hip-to-hip distance (20th–80th percentile filtered), averaged
    across all tracks.

    DSR stands for Dynamic Sniff Range — a body-scale reference distance derived
    from the median inter-hip distance of each animal.  All proximity thresholds
    used in compute_pairwise() (NoseNose, NoseHead, Contact, Engagement, etc.) are
    expressed as multiples of DSR rather than absolute pixel values.  This makes
    detection robust to different arena sizes, camera zoom levels, and animal sizes:
    an animal with a larger body automatically gets proportionally larger proximity
    zones, so the same multiplier thresholds work across experimental cohorts.

    Parameters
    ----------
    tracks      : (n_frames, 2, n_nodes, n_tracks)
    hip_l_idx   : int or None
    hip_r_idx   : int or None

    Returns
    -------
    float or None  — None when hip indices are unavailable
    """
    if hip_l_idx is None or hip_r_idx is None:
        return None

    n_tracks = tracks.shape[3]
    dsrs = []
    for t in range(n_tracks):
        xl = tracks[:, 0, hip_l_idx, t]
        yl = tracks[:, 1, hip_l_idx, t]
        xr = tracks[:, 0, hip_r_idx, t]
        yr = tracks[:, 1, hip_r_idx, t]
        dists = np.hypot(xl - xr, yl - yr)
        valid = dists[np.isfinite(dists)]
        if len(valid) < 3:
            continue
        lo, hi = np.percentile(valid, [20, 80])
        filtered = valid[(valid >= lo) & (valid <= hi)]
        if len(filtered) > 0:
            dsrs.append(float(np.median(filtered)))
    return float(np.mean(dsrs)) if dsrs else None


def _fallback_dsr(tracks):
    """10 % of the bounding-box diagonal of all node positions.

    This function is only invoked when hip nodes are absent from the SLEAP model
    (i.e. compute_dsr() returns None).  It provides a rough order-of-magnitude body
    scale so that proximity thresholds remain physically meaningful.

    tracks shape: (n_frames, 2, n_nodes, n_tracks)
      axis-1: 0 = x, 1 = y

    We transpose to (n_frames, n_nodes, n_tracks, 2) first so that reshape(-1, 2)
    produces one proper [x, y] pair per node/track/frame row.
    """
    # transpose axes: (n_frames, 2, n_nodes, n_tracks) → (n_frames, n_nodes, n_tracks, 2)
    all_xy = tracks.transpose(0, 2, 3, 1).reshape(-1, 2)
    valid  = all_xy[np.all(np.isfinite(all_xy), axis=1)]
    if len(valid) == 0:
        return 50.0
    x_range = float(valid[:, 0].max() - valid[:, 0].min())
    y_range = float(valid[:, 1].max() - valid[:, 1].min())
    return 0.1 * float(np.hypot(x_range, y_range))


# ---------------------------------------------------------------------------
# Angular-velocity helper
# ---------------------------------------------------------------------------

def angular_velocity(heading_deg, fps):
    """Circular finite-difference angular velocity in deg/s.

    Parameters
    ----------
    heading_deg : (n_frames,)  values in [-180, 180]
    fps         : float

    Returns
    -------
    (n_frames,)  angular velocity in deg/s  (same length; first element
                 duplicated from index 1)
    """
    h = np.asarray(heading_deg, dtype=np.float64)
    diff = np.diff(h)
    diff = (diff + 180.0) % 360.0 - 180.0   # wrap to (-180, 180]
    av = np.empty_like(h)
    av[1:] = diff * fps
    av[0]  = av[1] if len(av) > 1 else 0.0
    return av


# ---------------------------------------------------------------------------
# Single-animal behaviors
# ---------------------------------------------------------------------------

def compute_single_animal(tracks, kin, node_names, fps, px_per_cm=1.0):
    """Per-frame locomotor and turning states for every track.

    Parameters
    ----------
    tracks     : (n_frames, 2, n_nodes, n_tracks)
    kin        : dict from preprocessing.compute_kinematics
    node_names : list[str]
    fps        : float
    px_per_cm  : float — pixels per centimetre (for absolute speed thresholds)

    Returns
    -------
    dict of int8 arrays, each shape (n_frames, n_tracks):
        'stationary', 'walking', 'running', 'turning', 'dir_reversal'
    """
    n_frames, _, n_nodes, n_tracks = tracks.shape
    cm_idx = find_node_idx(node_names, 'body')

    out = {k: np.zeros((n_frames, n_tracks), dtype=np.int8)
           for k in ('stationary', 'walking', 'running', 'turning', 'dir_reversal')}

    for t in range(n_tracks):
        if cm_idx is not None:
            speed   = kin['speed'][:, cm_idx, t].astype(np.float64)
        else:
            speed   = np.nanmean(kin['speed'][:, :, t], axis=1).astype(np.float64)
        heading = kin['body_heading_deg'][:, t].astype(np.float64)

        speed   = np.nan_to_num(speed,   nan=0.0)
        heading = np.nan_to_num(heading, nan=0.0)

        # Absolute speed thresholds (cm/s → px/s via px_per_cm).
        # stationary: < 3 cm/s, walking: 3–20 cm/s, running: > 20 cm/s
        walk_thr = 3.0 * px_per_cm   # px/s
        run_thr  = 20.0 * px_per_cm  # px/s

        raw_state = np.where(speed < walk_thr, 0,
                    np.where(speed > run_thr, 2, 1)).astype(np.int8)

        # Minimum bout duration: median-filter the state sequence to remove
        # isolated 1–2 frame flickers (kernel = max(3, ~0.1s worth of frames),
        # always odd).
        kern = max(3, int(round(0.1 * fps)) | 1)  # ensure odd
        raw_state = median_filter(raw_state, size=kern).astype(np.int8)

        out['stationary'][:, t] = (raw_state == 0).astype(np.int8)
        out['walking'][:,    t] = (raw_state == 1).astype(np.int8)
        out['running'][:,    t] = (raw_state == 2).astype(np.int8)

        av = angular_velocity(heading, fps)
        out['turning'][:, t] = (np.abs(av) > 30.0).astype(np.int8)

        # Directional reversal: sign flip in a ±2-frame window, both sides > 20 deg/s
        rev  = np.zeros(n_frames, dtype=np.int8)
        half = 2
        av_abs = np.abs(av)
        # Use convolution to detect sign changes in window
        above_thr = av_abs > 20.0
        pos_above = (av > 0) & above_thr
        neg_above = (av < 0) & above_thr
        kernel = np.ones(2 * half + 1)
        has_pos = np.convolve(pos_above.astype(np.float64), kernel, mode='same') > 0
        has_neg = np.convolve(neg_above.astype(np.float64), kernel, mode='same') > 0
        rev = (has_pos & has_neg).astype(np.int8)
        # Zero out edges
        rev[:half] = 0
        rev[-half:] = 0
        out['dir_reversal'][:, t] = rev

    return out


# ---------------------------------------------------------------------------
# Pairwise helpers
# ---------------------------------------------------------------------------

def _node_xy(tracks, node_idx, t):
    """(n_frames, 2) position for one node/track pair."""
    return np.stack([tracks[:, 0, node_idx, t],
                     tracks[:, 1, node_idx, t]], axis=1)


def _dist2d(a, b):
    """Euclidean distance between two (n_frames, 2) arrays → (n_frames,)."""
    return np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])


def _smooth_heading_deg(pos):
    """Heading in degrees from (n_frames, 2) positions via np.gradient."""
    vx = np.gradient(pos[:, 0])
    vy = np.gradient(pos[:, 1])
    return np.degrees(np.arctan2(vy, vx))


def _fill_short_gaps(arr, max_gap):
    """Fill runs of zeros ≤ max_gap frames that are flanked by ones."""
    arr = arr.copy()
    if max_gap <= 0 or len(arr) < 3:
        return arr
    # Find transitions using diff
    d = np.diff(arr.astype(np.int8))
    # Gap starts: where value goes from 1 to 0 (diff == -1)
    gap_starts = np.where(d == -1)[0] + 1  # index of first 0
    # Gap ends: where value goes from 0 to 1 (diff == 1)
    gap_ends = np.where(d == 1)[0] + 1  # index of first 1 after gap

    if len(gap_starts) == 0 or len(gap_ends) == 0:
        return arr

    # Match each gap_start with the next gap_end
    # Only consider gaps that are flanked by ones (start preceded by 1, end is 1)
    ei = 0
    for si in range(len(gap_starts)):
        s = gap_starts[si]
        # Find the next gap_end >= s
        while ei < len(gap_ends) and gap_ends[ei] <= s:
            ei += 1
        if ei >= len(gap_ends):
            break
        e = gap_ends[ei]
        gap_len = e - s
        if gap_len <= max_gap:
            arr[s:e] = 1
    return arr


def _find_bouts(arr):
    """Return list of (start, end_inclusive) for runs of ones (vectorised)."""
    d = np.diff(arr.astype(np.int8), prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _apply_onset_gate(raw_arr, fps, metric_arr, threshold_fn, onset_frames=3,
                      gap_fill_s=0.1):
    """Apply onset-based kinematic quality gate. Returns gated int8 array.

    Parameters
    ----------
    raw_arr       : (n,) int8 — raw boolean behavior array
    fps           : float
    metric_arr    : (n,) float — kinematic signal to check at bout onset
    threshold_fn  : callable(onset_slice) -> bool — True if onset passes gate
    onset_frames  : int — number of frames at bout start to evaluate
    gap_fill_s    : float — short-gap fill in seconds before gating
    """
    filled = _fill_short_gaps(raw_arr, max(1, int(round(gap_fill_s * fps))))
    bouts = _find_bouts(filled)
    gated = filled.copy()
    for s, e in bouts:
        onset_end = min(s + onset_frames, e + 1)
        if not threshold_fn(metric_arr[s:onset_end]):
            gated[s:e + 1] = 0
    return gated


def _frame_speed(pos):
    """Per-frame speed (pixels/frame) from (n_frames, 2) array."""
    dx = np.diff(pos[:, 0], prepend=pos[0, 0])
    dy = np.diff(pos[:, 1], prepend=pos[0, 1])
    return np.hypot(dx, dy)


# ---------------------------------------------------------------------------
# Projection & alignment helpers (used by pairwise + second-order)
# ---------------------------------------------------------------------------

def _face_error(hdg, pos_subj, pos_target):
    """Angle (deg) between subject's heading and direction toward target.

    Parameters
    ----------
    hdg        : (n,) heading in degrees
    pos_subj   : (n, 2) subject position
    pos_target : (n, 2) target position

    Returns
    -------
    (n,) face error in [0, 180] degrees
    """
    vec = pos_target - pos_subj
    bearing = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    err = np.abs(hdg - bearing) % 360.0
    return np.minimum(err, 360.0 - err)


def _heading_opposition(hdg_A, hdg_B):
    """Absolute angular difference between two headings → [0, 180] degrees."""
    diff = np.abs(hdg_A - hdg_B) % 360.0
    return np.minimum(diff, 360.0 - diff)


def _velocity_cos_sim(vx_A, vy_A, vx_B, vy_B):
    """Cosine similarity between velocity vectors, frame-wise → [-1, 1]."""
    dot = vx_A * vx_B + vy_A * vy_B
    mag_A = np.hypot(vx_A, vy_A)
    mag_B = np.hypot(vx_B, vy_B)
    denom = mag_A * mag_B
    out = np.where(denom > 1e-12, dot / denom, 0.0)
    return out


def _project_in_body_frame(pos_subj, pos_target, hdg_subj):
    """Project target position into subject's body frame.

    Returns (proj_long, proj_lat) where positive long = in front, positive lat = left.
    """
    rel = pos_target - pos_subj
    cos_h = np.cos(np.radians(hdg_subj))
    sin_h = np.sin(np.radians(hdg_subj))
    proj_long = rel[:, 0] * cos_h + rel[:, 1] * sin_h
    proj_lat = -rel[:, 0] * sin_h + rel[:, 1] * cos_h
    return proj_long, proj_lat


def _retreat_projection(pos_subj, pos_target, vx_subj, vy_subj):
    """Scalar projection of subject's velocity onto the away-from-target axis.

    Positive = moving away from target.
    """
    away = pos_subj - pos_target
    dist = np.hypot(away[:, 0], away[:, 1])
    dist = np.where(dist < 1e-12, 1.0, dist)
    unit_away_x = away[:, 0] / dist
    unit_away_y = away[:, 1] / dist
    return vx_subj * unit_away_x + vy_subj * unit_away_y


def _windowed_displacement(pos, half_win):
    """Displacement over a symmetric window of ±half_win frames → (n,)."""
    n = len(pos)
    # End positions (shifted forward by half_win, clamped)
    end_idx = np.minimum(np.arange(n) + half_win, n - 1)
    # Start positions (shifted back by half_win, clamped)
    start_idx = np.maximum(np.arange(n) - half_win, 0)
    dx = pos[end_idx, 0] - pos[start_idx, 0]
    dy = pos[end_idx, 1] - pos[start_idx, 1]
    return np.hypot(dx, dy)


def _rolling_std(arr, half_win):
    """Rolling standard deviation with window ±half_win frames."""
    s = pd.Series(np.nan_to_num(arr, nan=0.0))
    win = 2 * half_win + 1
    return s.rolling(window=win, min_periods=2, center=True).std().fillna(0.0).to_numpy()


def _gap_delta(cm_dist, half_win=2):
    """Smoothed change in inter-animal distance (positive = separating)."""
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(cm_dist.astype(np.float64), size=half_win, mode='nearest', origin=0)
    delta = cm_dist - smoothed
    delta[0] = 0.0
    return delta


def _had_recent_social(pair_beh, pfx, n_frames, hist_frames, social_keys=None):
    """Binary array: was any social behavior active in last hist_frames."""
    if social_keys is None:
        social_keys = ('Engaged', 'Contact', 'NoseNose', 'NoseHead_AtoB',
                       'NoseHead_BtoA', 'NoseBody_AtoB', 'NoseBody_BtoA',
                       'NoseRear_AtoB', 'NoseRear_BtoA', 'HH', 'HO',
                       'Sniff_AtoB', 'Sniff_BtoA')
    combined = np.zeros(n_frames, dtype=np.int8)
    for sk in social_keys:
        k = f'{pfx}/{sk}'
        if k in pair_beh:
            arr = pair_beh[k]
            if arr.dtype == np.int8 or np.issubdtype(arr.dtype, np.integer):
                combined |= (arr == 1).astype(np.int8)
    if np.sum(combined) == 0:
        return np.zeros(n_frames, dtype=bool)
    padded = np.pad(combined, (hist_frames, 0), constant_values=0)
    cs = np.cumsum(padded)
    f_idx = np.arange(n_frames)
    recent = cs[f_idx + hist_frames] - cs[f_idx]
    return recent > 0


def _path_efficiency(pos, half_win):
    """Path efficiency (displacement / path_length) over ±half_win → (n,)."""
    n = len(pos)
    frame_disp = np.hypot(np.diff(pos[:, 0], prepend=pos[0, 0]),
                          np.diff(pos[:, 1], prepend=pos[0, 1]))
    # Cumulative sum for windowed path length
    cumsum = np.cumsum(frame_disp)
    padded = np.pad(cumsum, (1, 0), constant_values=0.0)

    end_idx = np.minimum(np.arange(n) + half_win, n - 1) + 1  # +1 for cumsum indexing
    start_idx = np.maximum(np.arange(n) - half_win, 0)
    total = padded[end_idx] - padded[start_idx]

    # Net displacement
    ei = np.minimum(np.arange(n) + half_win, n - 1)
    si = np.maximum(np.arange(n) - half_win, 0)
    net = np.hypot(pos[ei, 0] - pos[si, 0], pos[ei, 1] - pos[si, 1])

    eff = np.zeros_like(net)
    mask = total > 1e-12
    eff[mask] = net[mask] / total[mask]
    return eff


# ---------------------------------------------------------------------------
# Pairwise social behaviors
# ---------------------------------------------------------------------------

def compute_pairwise(tracks, node_names, fps, dsr=None, kin=None):
    """Compute pairwise social behavior arrays for every unique track pair.

    Parameters
    ----------
    tracks     : (n_frames, 2, n_nodes, n_tracks)
    node_names : list[str]
    fps        : float
    dsr        : float or None — auto-computed from hip nodes if None
    kin        : dict or None — kinematics dict (must contain 'body_heading_deg')

    Returns
    -------
    dict  — keys like 'tA_tB/BehaviorName', values (n_frames,) arrays.
            Empty dict when n_tracks < 2.
    """
    n_frames, _, n_nodes, n_tracks = tracks.shape
    if n_tracks < 2:
        return {}

    # --- DSR ---
    if dsr is None:
        hl  = find_node_idx(node_names, 'hip_l')
        hr  = find_node_idx(node_names, 'hip_r')
        dsr = compute_dsr(tracks, hl, hr)
        if dsr is None:
            dsr = _fallback_dsr(tracks)

    # --- key node indices ---
    nose_idx  = find_node_idx(node_names, 'nose')
    body_idx  = find_node_idx(node_names, 'body')
    tail_idx  = find_node_idx(node_names, 'tail')
    ear_l_idx = find_node_idx(node_names, 'ear_l')
    ear_r_idx = find_node_idx(node_names, 'ear_r')

    out   = {}
    pairs = list(combinations(range(n_tracks), 2))

    _sniff_exclude_pats = ('centermass', 'bodycenter', 'bodycentre',
                           'center_mass', 'center', 'body', 'cm', 'centroid',
                           'tailmedial', 'tail_medial', 'tm',
                           'tailend', 'tail_end', 'te')
    sniff_node_idxs = [i for i, nn in enumerate(node_names)
                       if _strip_digits(nn.lower()) not in _sniff_exclude_pats]

    for tA, tB in pairs:
        pfx = f't{tA}_t{tB}'

        # Positions
        nose_A = _node_xy(tracks, nose_idx, tA) if nose_idx is not None else None
        nose_B = _node_xy(tracks, nose_idx, tB) if nose_idx is not None else None
        body_A = _node_xy(tracks, body_idx, tA) if body_idx is not None else None
        body_B = _node_xy(tracks, body_idx, tB) if body_idx is not None else None
        tail_A = _node_xy(tracks, tail_idx, tA) if tail_idx is not None else None
        tail_B = _node_xy(tracks, tail_idx, tB) if tail_idx is not None else None

        if ear_l_idx is not None and ear_r_idx is not None:
            head_A = (_node_xy(tracks, ear_l_idx, tA) +
                      _node_xy(tracks, ear_r_idx, tA)) / 2.0
            head_B = (_node_xy(tracks, ear_l_idx, tB) +
                      _node_xy(tracks, ear_r_idx, tB)) / 2.0
        else:
            head_A, head_B = nose_A, nose_B

        # --- proximity subtypes ---
        def _prox(a, b, mult):
            if a is None or b is None:
                return np.zeros(n_frames, dtype=np.int8)
            return (_dist2d(a, b) < mult * dsr).astype(np.int8)

        out[f'{pfx}/NoseNose']      = _prox(nose_A, nose_B, 0.5)
        out[f'{pfx}/NoseHead_AtoB'] = _prox(nose_A, head_B, 0.7)
        out[f'{pfx}/NoseHead_BtoA'] = _prox(nose_B, head_A, 0.7)
        out[f'{pfx}/NoseBody_AtoB'] = _prox(nose_A, body_B, 0.7)
        out[f'{pfx}/NoseBody_BtoA'] = _prox(nose_B, body_A, 0.7)
        out[f'{pfx}/NoseRear_AtoB'] = _prox(nose_A, tail_B, 0.7)
        out[f'{pfx}/NoseRear_BtoA'] = _prox(nose_B, tail_A, 0.7)

        # --- HH (Head-to-Head) and HO (Head-On / Nose-to-Nose) ---
        if (head_A is not None and head_B is not None and
                nose_A is not None and nose_B is not None and
                body_A is not None and body_B is not None):
            if kin is not None:
                hdg_A_hh = kin['body_heading_deg'][:, tA]
                hdg_B_hh = kin['body_heading_deg'][:, tB]
            else:
                hdg_A_hh = _smooth_heading_deg(body_A)
                hdg_B_hh = _smooth_heading_deg(body_B)
            head_dist = _dist2d(head_A, head_B)
            nose_to_headB = _dist2d(nose_A, head_B)
            nose_to_headA = _dist2d(nose_B, head_A)
            min_nose_head = np.minimum(nose_to_headB, nose_to_headA)
            nose_nose_dist = _dist2d(nose_A, nose_B)

            fe_A_hh = _face_error(hdg_A_hh, body_A, head_B)
            fe_B_hh = _face_error(hdg_B_hh, body_B, head_A)
            h_opp = _heading_opposition(hdg_A_hh, hdg_B_hh)

            # HH: head centroid close OR min nose-to-head close,
            #     mutual facing < 70deg, heading opposition > 60deg
            hh_raw = (((head_dist < 0.85 * dsr) | (min_nose_head < 0.90 * dsr)) &
                       (fe_A_hh < 70.0) & (fe_B_hh < 70.0) &
                       (h_opp > 60.0)).astype(np.int8)
            out[f'{pfx}/HH'] = _fill_short_gaps(hh_raw, max(1, int(round(0.1 * fps))))

            # HO: nose-nose close, strict mutual facing < 45deg,
            #     strong heading opposition > 120deg
            ho_raw = ((nose_nose_dist < 0.55 * dsr) &
                       (fe_A_hh < 45.0) & (fe_B_hh < 45.0) &
                       (h_opp > 120.0)).astype(np.int8)
            out[f'{pfx}/HO'] = _fill_short_gaps(ho_raw, max(1, int(round(0.1 * fps))))
        else:
            out[f'{pfx}/HH'] = np.zeros(n_frames, dtype=np.int8)
            out[f'{pfx}/HO'] = np.zeros(n_frames, dtype=np.int8)

        # --- Sniff (orientation-gated nose-to-body proximity) ---
        # Sniff_AtoB: A's nose near any B node (excl CM, tail-medial, tail-end),
        #             A's heading toward B's closest node < 90deg

        for subj_t, targ_t, label in [(tA, tB, 'AtoB'), (tB, tA, 'BtoA')]:
            if nose_idx is not None and body_idx is not None and sniff_node_idxs:
                nose_subj = _node_xy(tracks, nose_idx, subj_t)
                body_subj = _node_xy(tracks, body_idx, subj_t)
                if kin is not None:
                    hdg_subj = kin['body_heading_deg'][:, subj_t]
                else:
                    hdg_subj = _smooth_heading_deg(body_subj)

                # min distance from subject nose to any target sniff-eligible node
                targ_nodes = tracks[:, :, sniff_node_idxs, :][:, :, :, targ_t]  # (n_frames, 2, n_sniff)
                nose_exp = nose_subj[:, :, np.newaxis]                            # (n_frames, 2, 1)
                _sdiff = targ_nodes - nose_exp                                    # (n_frames, 2, n_sniff)
                dists = np.hypot(_sdiff[:, 0, :], _sdiff[:, 1, :])               # (n_frames, n_sniff)
                closest_ni = np.argmin(dists, axis=1)                             # (n_frames,)
                min_d = dists[np.arange(n_frames), closest_ni]                   # (n_frames,)
                closest_pos = targ_nodes[np.arange(n_frames), :, closest_ni]    # (n_frames, 2)

                fe_sniff = _face_error(hdg_subj, body_subj, closest_pos)
                sniff_raw = ((min_d < 0.7 * dsr) &
                             (fe_sniff < 90.0)).astype(np.int8)
                out[f'{pfx}/Sniff_{label}'] = _fill_short_gaps(
                    sniff_raw, max(1, int(round(0.07 * fps))))
            else:
                out[f'{pfx}/Sniff_{label}'] = np.zeros(n_frames, dtype=np.int8)

        # --- contact: any node of A within 0.25×DSR of any node of B ---
        min_dist = np.full(n_frames, np.inf)
        for _ni in range(n_nodes):
            _node_A = tracks[:, :, _ni, tA]                          # (n_frames, 2)
            _cdiff = _node_A[:, :, np.newaxis] - tracks[:, :, :, tB] # (n_frames, 2, n_nodes)
            _dists = np.hypot(_cdiff[:, 0], _cdiff[:, 1])            # (n_frames, n_nodes)
            np.minimum(min_dist, _dists.min(axis=1), out=min_dist)
        out[f'{pfx}/Contact'] = (min_dist < 0.25 * dsr).astype(np.int8)

        # --- relative position, co/anti-orientation, engagement ---
        if body_A is not None and body_B is not None:
            if kin is not None:
                hdg_A = kin['body_heading_deg'][:, tA]
                hdg_B = kin['body_heading_deg'][:, tB]
            else:
                hdg_A = _smooth_heading_deg(body_A)
                hdg_B = _smooth_heading_deg(body_B)
            cos_A = np.cos(np.radians(hdg_A))
            sin_A = np.sin(np.radians(hdg_A))
            cos_B = np.cos(np.radians(hdg_B))
            sin_B = np.sin(np.radians(hdg_B))

            # RelPos_A: quadrant of B in A's heading frame
            rel_BinA  = body_B - body_A
            long_BinA = rel_BinA[:, 0] * cos_A + rel_BinA[:, 1] * sin_A
            lat_BinA  = -rel_BinA[:, 0] * sin_A + rel_BinA[:, 1] * cos_A

            lat_dom_A = np.abs(lat_BinA) > np.abs(long_BinA)
            rp_A = np.empty(n_frames, dtype=object)
            rp_A[~lat_dom_A & (long_BinA >= 0)] = 'Front'
            rp_A[~lat_dom_A & (long_BinA <  0)] = 'Behind'
            rp_A[ lat_dom_A & (lat_BinA  >  0)] = 'Left'
            rp_A[ lat_dom_A & (lat_BinA  <= 0)] = 'Right'
            out[f'{pfx}/RelPos_A'] = rp_A

            # RelPos_B: quadrant of A in B's heading frame
            rel_AinB  = body_A - body_B
            long_AinB = rel_AinB[:, 0] * cos_B + rel_AinB[:, 1] * sin_B
            lat_AinB  = -rel_AinB[:, 0] * sin_B + rel_AinB[:, 1] * cos_B

            lat_dom_B = np.abs(lat_AinB) > np.abs(long_AinB)
            rp_B = np.empty(n_frames, dtype=object)
            rp_B[~lat_dom_B & (long_AinB >= 0)] = 'Front'
            rp_B[~lat_dom_B & (long_AinB <  0)] = 'Behind'
            rp_B[ lat_dom_B & (lat_AinB  >  0)] = 'Left'
            rp_B[ lat_dom_B & (lat_AinB  <= 0)] = 'Right'
            out[f'{pfx}/RelPos_B'] = rp_B

            # Co/Anti orientation
            rel_ang = np.abs(hdg_A - hdg_B) % 360
            rel_ang = np.minimum(rel_ang, 360 - rel_ang)
            out[f'{pfx}/CoOriented']   = (rel_ang < 30).astype(np.int8)
            out[f'{pfx}/AntiOriented'] = (rel_ang > 150).astype(np.int8)

            # --- Engagement ---
            # A frame is classified as "engaged" when all three criteria hold simultaneously:
            #   (1) Body-centre distance < 3 * DSR  — the animals are in close proximity.
            #   (2) Face error A < 60°              — animal A's heading points toward animal B
            #                                         (within a 60° cone of the A→B direction).
            #   (3) Face error B < 60°              — animal B's heading points toward animal A.
            # All three must be true at once: mutual facing in close proximity.
            # After detecting raw engaged frames, _fill_short_gaps(..., 0.3 * fps) merges
            # engagement runs separated by up to 0.3 s, preventing brief look-away flickers
            # (e.g. a single frame where one animal glances sideways) from fragmenting what
            # is behaviourally a single continuous engagement bout.
            cm_dist = _dist2d(body_A, body_B)

            vec_A2B   = body_B - body_A
            face_dir_A = np.degrees(np.arctan2(vec_A2B[:, 1], vec_A2B[:, 0]))
            face_err_A = np.abs(hdg_A - face_dir_A) % 360
            face_err_A = np.minimum(face_err_A, 360 - face_err_A)

            vec_B2A    = body_A - body_B
            face_dir_B = np.degrees(np.arctan2(vec_B2A[:, 1], vec_B2A[:, 0]))
            face_err_B = np.abs(hdg_B - face_dir_B) % 360
            face_err_B = np.minimum(face_err_B, 360 - face_err_B)

            engaged_raw = ((cm_dist    <  3  * dsr) &
                           (face_err_A < 60.0)       &
                           (face_err_B < 60.0)).astype(np.int8)
            engaged = _fill_short_gaps(engaged_raw, int(round(0.3 * fps)))
            out[f'{pfx}/Engaged'] = engaged

            # --- Speed during engagement frames ---
            spd_A = _frame_speed(body_A)
            spd_B = _frame_speed(body_B)

            def _masked(spd, mask):
                arr = np.full(n_frames, np.nan)
                arr[mask == 1] = spd[mask == 1]
                return arr

            out[f'{pfx}/EngageSpeed_A']  = _masked(spd_A, engaged)
            out[f'{pfx}/EngageSpeed_B']  = _masked(spd_B, engaged)

            # Onset speeds (first frame of each engagement bout only)
            eng_onset_A = np.full(n_frames, np.nan)
            eng_onset_B = np.full(n_frames, np.nan)
            for start, _ in _find_bouts(engaged):
                eng_onset_A[start] = spd_A[start]
                eng_onset_B[start] = spd_B[start]
            out[f'{pfx}/EngageOnsetSpeed_A'] = eng_onset_A
            out[f'{pfx}/EngageOnsetSpeed_B'] = eng_onset_B

        else:
            # No body-center node — fill with zeros / NaN / empty strings
            zeros = np.zeros(n_frames, dtype=np.int8)
            nans  = np.full(n_frames, np.nan)
            empty = np.full(n_frames, '', dtype=object)
            for key, val in [
                ('RelPos_A', empty.copy()), ('RelPos_B', empty.copy()),
                ('CoOriented', zeros.copy()), ('AntiOriented', zeros.copy()),
                ('Engaged', zeros.copy()),
                ('EngageSpeed_A', nans.copy()), ('EngageSpeed_B', nans.copy()),
                ('EngageOnsetSpeed_A', nans.copy()), ('EngageOnsetSpeed_B', nans.copy()),
            ]:
                out[f'{pfx}/{key}'] = val

    out['_dsr'] = dsr  # Store for reuse by compute_second_order
    return out


# ---------------------------------------------------------------------------
# Engagement bout initiator / retreat helpers
# ---------------------------------------------------------------------------

def _mean_approach_angle_arr(tracks, kin, body_idx, tA, tB, f0, f1):
    """
    Mean approach angle (deg) of track tA toward tB over sleap frames [f0, f1).
    Returns 90.0 when data is unavailable.
    """
    # 90.0° is a conservative neutral default — it represents "neither approaching
    # nor retreating" (exactly perpendicular to the inter-animal axis).  Returning
    # this value when data is unavailable ensures the function does not bias the
    # initiator attribution toward either animal A or animal B: both animals receive
    # the same neutral angle, so the tie-breaking rule (aa_A <= aa_B → A initiates)
    # defaults to A, which is arbitrary but at least not directionally biased.
    if body_idx is None:
        return 90.0
    n = tracks.shape[0]
    f0 = max(0, f0);  f1 = min(n, f1)
    if f0 >= f1:
        return 90.0
    hdg = kin['body_heading_deg'][f0:f1, tA].astype(np.float64)
    xA  = tracks[f0:f1, 0, body_idx, tA].astype(np.float64)
    yA  = tracks[f0:f1, 1, body_idx, tA].astype(np.float64)
    xB  = tracks[f0:f1, 0, body_idx, tB].astype(np.float64)
    yB  = tracks[f0:f1, 1, body_idx, tB].astype(np.float64)
    ok  = np.isfinite(xA) & np.isfinite(yA) & np.isfinite(xB) & np.isfinite(yB) & np.isfinite(hdg)
    if not np.any(ok):
        return 90.0
    dir_AB = np.degrees(np.arctan2(yB[ok] - yA[ok], xB[ok] - xA[ok]))
    diff   = np.abs(hdg[ok] - dir_AB) % 360.0
    return float(np.mean(np.minimum(diff, 360.0 - diff)))


def _bouts_with_initiator(engaged_arr, tracks, kin, body_idx, tA, tB, fps):
    """
    Find engagement bouts and determine which animal initiated each one and
    which animal disengaged.

    Initiator  = animal facing the other more directly in the pre-bout window.
    Disengager = animal facing away more in the post-bout window.

    Returns list of dicts:
        start      : int  sleap frame index (bout onset)
        end        : int  sleap frame index (inclusive, bout offset)
        initiator  : int  0 = tA, 1 = tB
        disengager : int  0 = tA, 1 = tB
    """
    bouts = _find_bouts(engaged_arr.astype(np.int8))
    pre   = max(1, int(round(fps * 0.3)))
    post  = max(1, int(round(fps * 0.5)))
    result = []
    for s, e in bouts:
        aa_A = _mean_approach_angle_arr(tracks, kin, body_idx, tA, tB, s - pre, s + 1)
        aa_B = _mean_approach_angle_arr(tracks, kin, body_idx, tB, tA, s - pre, s + 1)
        initiator = 0 if aa_A <= aa_B else 1

        aa_A_post = _mean_approach_angle_arr(tracks, kin, body_idx, tA, tB, e + 1, e + post + 1)
        aa_B_post = _mean_approach_angle_arr(tracks, kin, body_idx, tB, tA, e + 1, e + post + 1)
        disengager = 0 if aa_A_post >= aa_B_post else 1

        result.append(dict(start=s, end=e, initiator=initiator, disengager=disengager))
    return result


def _indices_from_bouts(bouts, fps, retreat_window_s=3.0):
    """
    Compute Engagement Index, Reciprocity Index, and Retreat Index for
    each animal from a (possibly filtered) list of bout dicts.

    Definitions (n_total = total bouts in this window for both animals):
      Engagement Index A  = n_bouts_A_initiated / n_total
      Reciprocity Index A = n_bouts_A_initiated_after_B_engaged_first / n_total
      Retreat Index A     = n_bouts_A_disengaged_within_retreat_window_after_B_initiated / n_total

    Returns dict with keys:
        n_bouts, n_init_A, n_init_B,
        EI_A, EI_B, RI_A, RI_B, RTI_A, RTI_B
    """
    n = len(bouts)
    if n == 0:
        return dict(n_bouts=0, n_init_A=0, n_init_B=0,
                    EI_A=None, EI_B=None,
                    RI_A=None, RI_B=None,
                    RTI_A=None, RTI_B=None)

    n_init_A = sum(1 for b in bouts if b['initiator'] == 0)
    n_init_B = n - n_init_A

    # Reciprocity: previous bout by B → current bout by A  (and vice versa)
    n_recip_A = n_recip_B = 0
    for i in range(1, n):
        pi, ci = bouts[i - 1]['initiator'], bouts[i]['initiator']
        if pi == 1 and ci == 0:
            n_recip_A += 1
        elif pi == 0 and ci == 1:
            n_recip_B += 1

    # Retreat is defined operationally as a bout where:
    #   — the OTHER animal initiated the bout (B approached A), AND
    #   — the bout ended quickly (duration <= retreat_window_s, default 3 s), AND
    #   — the focal animal was the one who disengaged (turned away / moved apart).
    # Logic: if B approaches and the bout ends within 3 s with A disengaging,
    # the interpretation is that A retreated from B's approach rather than the
    # two animals completing a mutual encounter.  Longer bouts (> retreat_window_s)
    # are excluded because they likely reflect genuine mutual engagement rather than
    # a retreat response.  The bout duration serves as a proxy for whether A had
    # time to stay and engage vs. quickly withdraw.
    max_dur = int(round(fps * retreat_window_s))
    n_retreat_A = n_retreat_B = 0
    for b in bouts:
        dur = b['end'] - b['start'] + 1
        if dur > max_dur:
            continue
        if b['initiator'] == 1 and b['disengager'] == 0:
            n_retreat_A += 1
        elif b['initiator'] == 0 and b['disengager'] == 1:
            n_retreat_B += 1

    _d = lambda a: round(a / n, 4) if n > 0 else None
    return dict(
        n_bouts  = n,
        n_init_A = n_init_A,
        n_init_B = n_init_B,
        EI_A     = _d(n_init_A),
        EI_B     = _d(n_init_B),
        RI_A     = _d(n_recip_A),
        RI_B     = _d(n_recip_B),
        RTI_A    = _d(n_retreat_A),
        RTI_B    = _d(n_retreat_B),
    )


# ---------------------------------------------------------------------------
# Second-order (compound) social behaviors
# ---------------------------------------------------------------------------

def compute_second_order(tracks, kin, node_names, fps, single_beh, pair_beh,
                         dsr=None):
    """Detect compound social behaviors that depend on first-order detections.

    Parameters
    ----------
    tracks      : (n_frames, 2, n_nodes, n_tracks)
    kin         : dict from compute_kinematics
    node_names  : list[str]
    fps         : float
    single_beh  : dict from compute_single_animal
    pair_beh    : dict from compute_pairwise (already contains HH/HO/Sniff/Engaged)
    dsr         : float or None — auto-computed if None

    Returns
    -------
    dict  — keys like 'tA_tB/BehaviorName', values (n_frames,) int8 arrays.
    """
    n_frames, _, n_nodes, n_tracks = tracks.shape
    if n_tracks < 2:
        return {}

    # --- DSR ---
    if dsr is None:
        dsr = pair_beh.get('_dsr')
    if dsr is None:
        hl = find_node_idx(node_names, 'hip_l')
        hr = find_node_idx(node_names, 'hip_r')
        dsr = compute_dsr(tracks, hl, hr)
        if dsr is None:
            dsr = _fallback_dsr(tracks)

    body_idx = find_node_idx(node_names, 'body')
    nose_idx = find_node_idx(node_names, 'nose')

    out = {}
    pairs = list(combinations(range(n_tracks), 2))
    hist_frames = int(round(0.75 * fps))
    half_win_05s = max(1, int(round(0.25 * fps)))  # ~0.5s window half

    for tA, tB in pairs:
        pfx = f't{tA}_t{tB}'

        # --- Shared data for this pair ---
        if body_idx is None:
            # No body-center → output all zeros for this pair
            for bname in _SECOND_ORDER_KEYS:
                out[f'{pfx}/{bname}'] = np.zeros(n_frames, dtype=np.int8)
            continue

        body_A = _node_xy(tracks, body_idx, tA)
        body_B = _node_xy(tracks, body_idx, tB)
        hdg_A = kin['body_heading_deg'][:, tA]
        hdg_B = kin['body_heading_deg'][:, tB]

        cm_dist = _dist2d(body_A, body_B)
        gap_d = _gap_delta(cm_dist, half_win=2)

        # Velocities
        vx_A = kin['vx'][:, body_idx, tA].astype(np.float64)
        vy_A = kin['vy'][:, body_idx, tA].astype(np.float64)
        vx_B = kin['vx'][:, body_idx, tB].astype(np.float64)
        vy_B = kin['vy'][:, body_idx, tB].astype(np.float64)
        spd_A = kin['speed'][:, body_idx, tA].astype(np.float64)
        spd_B = kin['speed'][:, body_idx, tB].astype(np.float64)
        spd_A = np.nan_to_num(spd_A, nan=0.0)
        spd_B = np.nan_to_num(spd_B, nan=0.0)

        accel_A = np.abs(np.nan_to_num(kin['accel'][:, body_idx, tA].astype(np.float64), nan=0.0))
        accel_B = np.abs(np.nan_to_num(kin['accel'][:, body_idx, tB].astype(np.float64), nan=0.0))
        jerk_A = np.abs(np.nan_to_num(kin['jerk'][:, body_idx, tA].astype(np.float64), nan=0.0))
        jerk_B = np.abs(np.nan_to_num(kin['jerk'][:, body_idx, tB].astype(np.float64), nan=0.0))

        # Angular velocity
        av_A = np.abs(angular_velocity(hdg_A, fps))
        av_B = np.abs(angular_velocity(hdg_B, fps))

        # Face errors: A toward B, B toward A
        fe_A = _face_error(hdg_A, body_A, body_B)
        fe_B = _face_error(hdg_B, body_B, body_A)

        # Velocity alignment
        cos_sim = _velocity_cos_sim(vx_A, vy_A, vx_B, vy_B)

        # Speed percentiles (of frames where animal is moving)
        active_mask_A = spd_A > np.percentile(spd_A, 10)
        active_mask_B = spd_B > np.percentile(spd_B, 10)
        active_A = spd_A[active_mask_A] if np.any(active_mask_A) else spd_A
        active_B = spd_B[active_mask_B] if np.any(active_mask_B) else spd_B
        p35_A, p70_A, p75_A, p95_A = np.percentile(active_A, [35, 70, 75, 95])
        p35_B, p70_B, p75_B, p95_B = np.percentile(active_B, [35, 70, 75, 95])

        med_spd_A = np.median(spd_A)
        med_spd_B = np.median(spd_B)
        mad_spd_A = np.median(np.abs(spd_A - med_spd_A))
        mad_spd_B = np.median(np.abs(spd_B - med_spd_B))

        # Retreat projections
        ret_A = _retreat_projection(body_A, body_B, vx_A, vy_A)
        ret_B = _retreat_projection(body_B, body_A, vx_B, vy_B)

        # Path efficiency
        pe_A = _path_efficiency(body_A, half_win_05s)
        pe_B = _path_efficiency(body_B, half_win_05s)

        # Displacement over 0.5s window
        disp_A = _windowed_displacement(body_A, half_win_05s)
        disp_B = _windowed_displacement(body_B, half_win_05s)

        # Rolling gap std
        gap_std = _rolling_std(cm_dist, half_win=max(2, int(round(0.3 * fps))))

        # Contact/sniff/HH/HO exclusion masks
        contact = pair_beh.get(f'{pfx}/Contact', np.zeros(n_frames, dtype=np.int8))
        hh = pair_beh.get(f'{pfx}/HH', np.zeros(n_frames, dtype=np.int8))
        ho = pair_beh.get(f'{pfx}/HO', np.zeros(n_frames, dtype=np.int8))
        sniff_ab = pair_beh.get(f'{pfx}/Sniff_AtoB', np.zeros(n_frames, dtype=np.int8))
        sniff_ba = pair_beh.get(f'{pfx}/Sniff_BtoA', np.zeros(n_frames, dtype=np.int8))
        engaged = pair_beh.get(f'{pfx}/Engaged', np.zeros(n_frames, dtype=np.int8))

        in_contact_or_sniff = ((contact == 1) | (hh == 1) | (ho == 1) |
                               (sniff_ab == 1) | (sniff_ba == 1))

        # Recent social context
        recent_social = _had_recent_social(pair_beh, pfx, n_frames, hist_frames)

        # --- B1. Follow (directional: A follows B, B follows A) ---
        for subj, targ, label, s_spd, t_spd, s_fe, s_av, s_pe, s_p35, s_vx, s_vy, t_vx, t_vy in [
            ('A', 'B', 'AtoB', spd_A, spd_B, fe_A, av_A, pe_A, p35_A, vx_A, vy_A, vx_B, vy_B),
            ('B', 'A', 'BtoA', spd_B, spd_A, fe_B, av_B, pe_B, p35_B, vx_B, vy_B, vx_A, vy_A),
        ]:
            subj_body = body_A if subj == 'A' else body_B
            targ_body = body_B if subj == 'A' else body_A
            subj_hdg = hdg_A if subj == 'A' else hdg_B
            targ_hdg = hdg_B if subj == 'A' else hdg_A

            proj_long, proj_lat = _project_in_body_frame(targ_body, subj_body, targ_hdg)

            rel_angle = _heading_opposition(subj_hdg, targ_hdg)
            # For follow, we want co-direction, so invert: heading_diff < 55
            heading_diff = np.abs(subj_hdg - targ_hdg) % 360.0
            heading_diff = np.minimum(heading_diff, 360.0 - heading_diff)

            follow_raw = (
                (cm_dist >= 0.8 * dsr) & (cm_dist <= 3.5 * dsr) &
                ((gap_d < -0.15 * dsr) | (gap_std < 0.25 * dsr)) &
                (proj_long < -0.2 * dsr) &
                (np.abs(proj_lat) < 1.0 * dsr) &
                (heading_diff < 55.0) &
                (s_fe < 50.0) &
                (cos_sim > 0.65) &
                (s_spd > s_p35) & (t_spd > s_p35) &
                (s_av < 120.0) &
                (~in_contact_or_sniff) &
                (s_pe > 0.5)
            ).astype(np.int8)
            # Speed onset gate: subject speed > median confirms genuine locomotion
            _med = np.median(s_spd)
            out[f'{pfx}/Follow_{label}'] = _apply_onset_gate(
                follow_raw, fps, s_spd,
                lambda sl, m=_med: len(sl) > 0 and np.any(sl > m),
                onset_frames=3, gap_fill_s=0.15)

        follow_ab = out[f'{pfx}/Follow_AtoB']
        follow_ba = out[f'{pfx}/Follow_BtoA']

        # --- B2. Chase (directional) ---
        accel_med_A = np.median(accel_A)
        accel_mad_A = np.median(np.abs(accel_A - accel_med_A))
        accel_med_B = np.median(accel_B)
        accel_mad_B = np.median(np.abs(accel_B - accel_med_B))

        for subj, targ, label, s_spd, t_spd, s_fe, s_av, s_pe, s_p70, s_accel, s_accel_med, s_accel_mad, s_vx, s_vy, t_vx, t_vy in [
            ('A', 'B', 'AtoB', spd_A, spd_B, fe_A, av_A, pe_A, p70_A, accel_A, accel_med_A, accel_mad_A, vx_A, vy_A, vx_B, vy_B),
            ('B', 'A', 'BtoA', spd_B, spd_A, fe_B, av_B, pe_B, p70_B, accel_B, accel_med_B, accel_mad_B, vx_B, vy_B, vx_A, vy_A),
        ]:
            subj_body = body_A if subj == 'A' else body_B
            targ_body = body_B if subj == 'A' else body_A
            subj_hdg = hdg_A if subj == 'A' else hdg_B
            targ_hdg = hdg_B if subj == 'A' else hdg_A
            subj_follow = follow_ab if subj == 'A' else follow_ba

            proj_long, proj_lat = _project_in_body_frame(targ_body, subj_body, targ_hdg)

            heading_diff = np.abs(subj_hdg - targ_hdg) % 360.0
            heading_diff = np.minimum(heading_diff, 360.0 - heading_diff)

            chase_raw = (
                (cm_dist >= 0.7 * dsr) & (cm_dist <= 3.0 * dsr) &
                ((gap_d < -0.20 * dsr) | (gap_std < 0.30 * dsr)) &
                (proj_long < -0.15 * dsr) &
                (np.abs(proj_lat) < 1.2 * dsr) &
                (heading_diff < 50.0) &
                (s_fe < 45.0) &
                (cos_sim > 0.70) &
                (s_spd > t_spd * 0.95) &
                (s_spd > s_p70) &
                (s_av < 140.0) &
                (~in_contact_or_sniff) &
                (subj_follow == 0) &  # not already classified as follow
                (s_pe > 0.7)
            ).astype(np.int8)

            # Acceleration gate at onset
            _accel_thresh = s_accel_med + 1.0 * s_accel_mad
            out[f'{pfx}/Chase_{label}'] = _apply_onset_gate(
                chase_raw, fps, s_accel,
                lambda sl: len(sl) > 0 and np.max(sl) >= _accel_thresh,
                onset_frames=3, gap_fill_s=0.1)

        chase_ab = out[f'{pfx}/Chase_AtoB']
        chase_ba = out[f'{pfx}/Chase_BtoA']

        # --- B3. Flee (directional) ---
        jerk_p90_A = np.percentile(jerk_A, 90)
        jerk_p90_B = np.percentile(jerk_B, 90)
        accel_p80_A = np.percentile(accel_A, 80)
        accel_p80_B = np.percentile(accel_B, 80)

        for subj, label, s_spd, s_fe, s_ret, s_jerk, s_accel, s_med, s_mad, s_p75, jerk_p90, accel_p80 in [
            ('A', 'AtoB', spd_A, fe_A, ret_A, jerk_A, accel_A, med_spd_A, mad_spd_A, p75_A, jerk_p90_A, accel_p80_A),
            ('B', 'BtoA', spd_B, fe_B, ret_B, jerk_B, accel_B, med_spd_B, mad_spd_B, p75_B, jerk_p90_B, accel_p80_B),
        ]:
            flee_speed_thresh = max(s_med + 2.0 * s_mad, s_p75)

            flee_raw = (
                (gap_d > 0.30 * dsr) &
                (s_fe > 130.0) &
                (s_spd > flee_speed_thresh) &
                (s_ret > 0.20 * dsr) &
                (cm_dist < 3.5 * dsr) &
                recent_social
            ).astype(np.int8)

            # Jerk/accel gate at onset — need both metrics, so use a combined array
            # threshold_fn checks if jerk OR accel exceeds thresholds
            _jp90 = jerk_p90
            _ap80 = accel_p80
            _s_jerk = s_jerk
            flee_filled = _fill_short_gaps(flee_raw, max(1, int(round(0.1 * fps))))
            flee_bouts = _find_bouts(flee_filled)
            flee_gated = flee_filled.copy()
            for s, e in flee_bouts:
                onset_end = min(s + 3, e + 1)
                if not (np.any(_s_jerk[s:onset_end] > _jp90) or
                        np.any(s_accel[s:onset_end] > _ap80)):
                    flee_gated[s:e + 1] = 0
            out[f'{pfx}/Flee_{label}'] = flee_gated

        flee_ab = out[f'{pfx}/Flee_AtoB']
        flee_ba = out[f'{pfx}/Flee_BtoA']

        # --- B4. Approach (directional) ---
        for subj, label, s_spd, s_fe, s_pe, s_p35, s_p95, s_vx, s_vy in [
            ('A', 'AtoB', spd_A, fe_A, pe_A, p35_A, p95_A, vx_A, vy_A),
            ('B', 'BtoA', spd_B, fe_B, pe_B, p35_B, p95_B, vx_B, vy_B),
        ]:
            subj_body = body_A if subj == 'A' else body_B
            targ_body = body_B if subj == 'A' else body_A
            subj_chase = chase_ab if subj == 'A' else chase_ba
            subj_follow = follow_ab if subj == 'A' else follow_ba
            subj_flee = flee_ab if subj == 'A' else flee_ba

            # Approach velocity: projection of vel toward partner
            toward = targ_body - subj_body
            toward_dist = np.hypot(toward[:, 0], toward[:, 1])
            toward_dist = np.where(toward_dist < 1e-12, 1.0, toward_dist)
            approach_vel = (s_vx * toward[:, 0] / toward_dist +
                            s_vy * toward[:, 1] / toward_dist)

            # approach_vel (px/s) vs 0.12 * dsr (px): threshold means
            # "approaching at > 12% of a body-length per second"
            approach_raw = (
                (cm_dist >= 1.0 * dsr) & (cm_dist <= 5.0 * dsr) &
                (gap_d < -0.18 * dsr) &
                (s_fe < 55.0) &
                (approach_vel > 0.12 * dsr) &
                (s_spd >= s_p35) & (s_spd <= s_p95) &
                (contact == 0) &
                (subj_follow == 0) & (subj_chase == 0) &
                (subj_flee == 0) &
                (s_pe > 0.6)
            ).astype(np.int8)
            # Approach velocity onset gate: confirms directed movement
            _av_thresh = 0.05 * dsr
            out[f'{pfx}/Approach_{label}'] = _apply_onset_gate(
                approach_raw, fps, approach_vel,
                lambda sl, th=_av_thresh: len(sl) > 0 and np.any(sl > th),
                onset_frames=3, gap_fill_s=0.15)

        approach_ab = out[f'{pfx}/Approach_AtoB']
        approach_ba = out[f'{pfx}/Approach_BtoA']

        # --- B5. Active + Passive Avoidance (directional) ---
        for subj, label, s_spd, s_fe, s_ret, s_med, s_mad, s_accel, s_disp in [
            ('A', 'AtoB', spd_A, fe_A, ret_A, med_spd_A, mad_spd_A, accel_A, disp_A),
            ('B', 'BtoA', spd_B, fe_B, ret_B, med_spd_B, mad_spd_B, accel_B, disp_B),
        ]:
            subj_flee = flee_ab if subj == 'A' else flee_ba

            # Active avoidance
            active_avoid_raw = (
                (gap_d > 0.20 * dsr) &
                (s_fe > 120.0) &
                (s_spd > s_med + 1.5 * s_mad) &
                (s_ret > 0.15 * dsr) &
                (cm_dist < 4.0 * dsr) &
                recent_social &
                (contact == 0) &
                (subj_flee == 0)
            ).astype(np.int8)
            # Accel onset gate: confirms movement initiation
            _accel_med = np.median(s_accel)
            out[f'{pfx}/ActiveAvoid_{label}'] = _apply_onset_gate(
                active_avoid_raw, fps, s_accel,
                lambda sl, m=_accel_med: len(sl) > 0 and np.any(sl > m),
                onset_frames=3, gap_fill_s=0.1)

            # Passive avoidance — low speed, facing away, not approaching
            passive_avoid_raw = (
                (s_fe > 120.0) &
                (s_disp < 0.20 * dsr) &
                (s_spd < s_med + 0.25 * s_mad) &
                (gap_d > -0.05 * dsr) &
                (cm_dist < 3.0 * dsr) &
                recent_social &
                (s_accel < np.median(s_accel))  # low accel confirms true stillness
            ).astype(np.int8)
            out[f'{pfx}/PassiveAvoid_{label}'] = _fill_short_gaps(
                passive_avoid_raw, max(1, int(round(0.15 * fps))))

        active_avoid_ab = out[f'{pfx}/ActiveAvoid_AtoB']
        active_avoid_ba = out[f'{pfx}/ActiveAvoid_BtoA']

        # --- B6. Stationary Proximity ---
        stat_prox_raw = (
            (spd_A < med_spd_A + 0.15 * mad_spd_A) &
            (spd_B < med_spd_B + 0.15 * mad_spd_B) &
            (disp_A < 0.20 * dsr) &
            (disp_B < 0.20 * dsr) &
            (cm_dist < 2.5 * dsr) &
            (accel_A < np.median(accel_A)) &
            (accel_B < np.median(accel_B)) &
            (jerk_A < np.median(jerk_A) + mad_spd_A) &
            (jerk_B < np.median(jerk_B) + mad_spd_B) &
            (contact == 0) &
            (follow_ab == 0) & (follow_ba == 0) &
            (chase_ab == 0) & (chase_ba == 0) &
            (active_avoid_ab == 0) & (active_avoid_ba == 0) &
            (flee_ab == 0) & (flee_ba == 0)
        ).astype(np.int8)
        out[f'{pfx}/StationaryProx'] = _fill_short_gaps(
            stat_prox_raw, max(1, int(round(0.2 * fps))))

        # --- B7. Social Orientation (directional) ---
        prox_flag = cm_dist < 5.0 * dsr
        for subj, label, s_fe, s_spd, s_p35 in [
            ('A', 'AtoB', fe_A, spd_A, p35_A),
            ('B', 'BtoA', fe_B, spd_B, p35_B),
        ]:
            subj_flee = flee_ab if subj == 'A' else flee_ba
            subj_avoid_active = active_avoid_ab if subj == 'A' else active_avoid_ba

            # Stronger threshold when inactive, relaxed when in active state
            is_active = s_spd > s_p35
            orient_thresh = np.where(is_active, 65.0, 45.0)

            so_raw = (
                (s_fe < orient_thresh) &
                prox_flag &
                (subj_flee == 0) &
                (subj_avoid_active == 0)
            ).astype(np.int8)
            out[f'{pfx}/SocialOrient_{label}'] = so_raw

        # --- B8. Disengaged (replaces old version) ---
        # Was in any active social state in last 0.75s
        _disengage_social_keys = (
            'Engaged', 'Contact', 'Sniff_AtoB', 'Sniff_BtoA',
            'HH', 'HO', 'NoseNose', 'NoseHead_AtoB', 'NoseHead_BtoA',
        )
        # Also check second-order behaviors computed above
        _so_disengage_keys = [
            f'Approach_AtoB', f'Approach_BtoA',
            f'Follow_AtoB', f'Follow_BtoA',
            f'Chase_AtoB', f'Chase_BtoA',
            f'SocialOrient_AtoB', f'SocialOrient_BtoA',
        ]

        # Build combined "was recently social" from both pair_beh and out
        combined_social = np.zeros(n_frames, dtype=np.int8)
        for sk in _disengage_social_keys:
            k = f'{pfx}/{sk}'
            if k in pair_beh:
                arr = pair_beh[k]
                if np.issubdtype(arr.dtype, np.integer):
                    combined_social |= (arr == 1).astype(np.int8)
        for sk in _so_disengage_keys:
            k = f'{pfx}/{sk}'
            if k in out:
                combined_social |= (out[k] == 1).astype(np.int8)

        if np.sum(combined_social) > 0:
            padded = np.pad(combined_social, (hist_frames, 0), constant_values=0)
            cs = np.cumsum(padded)
            f_idx = np.arange(n_frames)
            was_recent_social = (cs[f_idx + hist_frames] - cs[f_idx]) > 0
        else:
            was_recent_social = np.zeros(n_frames, dtype=bool)

        # Not currently in contact/approach/follow/chase
        in_active_social = (
            (contact == 1) |
            (approach_ab == 1) | (approach_ba == 1) |
            (follow_ab == 1) | (follow_ba == 1) |
            (chase_ab == 1) | (chase_ba == 1) |
            (sniff_ab == 1) | (sniff_ba == 1)
        )

        disengage_raw = (
            was_recent_social &
            ((fe_A > 95.0) | (fe_B > 95.0)) &
            (~(flee_ab == 1)) & (~(flee_ba == 1)) &
            (~in_active_social) &
            ((gap_d > 0.05 * dsr) |
             ((spd_A < med_spd_A + 0.5 * mad_spd_A) & (spd_B < med_spd_B + 0.5 * mad_spd_B)) |
             (ret_A > 0.03 * dsr) | (ret_B > 0.03 * dsr)) &
            (cm_dist < 4.0 * dsr) &
            ((accel_A < accel_med_A + 1.5 * accel_mad_A) |
             (accel_B < accel_med_B + 1.5 * accel_mad_B))
        ).astype(np.int8)
        out[f'{pfx}/Disengaged'] = _fill_short_gaps(
            disengage_raw, max(1, int(round(0.15 * fps))))

        # --- Disengage speeds (like old version but using new Disengaged) ---
        disengaged = out[f'{pfx}/Disengaged']
        frame_spd_A = _frame_speed(body_A)
        frame_spd_B = _frame_speed(body_B)

        def _masked_nan(spd, mask):
            arr = np.full(n_frames, np.nan)
            arr[mask == 1] = spd[mask == 1]
            return arr

        out[f'{pfx}/DisengageSpeed_A'] = _masked_nan(frame_spd_A, disengaged)
        out[f'{pfx}/DisengageSpeed_B'] = _masked_nan(frame_spd_B, disengaged)

        dis_onset_A = np.full(n_frames, np.nan)
        dis_onset_B = np.full(n_frames, np.nan)
        for start, _ in _find_bouts(disengaged):
            dis_onset_A[start] = frame_spd_A[start]
            dis_onset_B[start] = frame_spd_B[start]
        out[f'{pfx}/DisengageOnsetSpeed_A'] = dis_onset_A
        out[f'{pfx}/DisengageOnsetSpeed_B'] = dis_onset_B

        # --- B9. Visual & Auditory Attention ---
        # Reuse face_error as proxy for approach_angle
        for subj, label, s_fe in [
            ('A', 'AtoB', fe_A),
            ('B', 'BtoA', fe_B),
        ]:
            # VisualAttn: target in subject's visual field (approach_angle < 120deg)
            out[f'{pfx}/VisualAttn_{label}'] = (s_fe < 120.0).astype(np.int8)
            # AuditoryAttn: target in subject's auditory field (approach_angle < 150deg)
            out[f'{pfx}/AuditoryAttn_{label}'] = (s_fe < 150.0).astype(np.int8)

    return out


# Canonical list of first-order pairwise behavior key suffixes (without prefix)
_PAIRWISE_BINARY_KEYS = [
    'NoseNose', 'NoseHead_AtoB', 'NoseHead_BtoA',
    'NoseBody_AtoB', 'NoseBody_BtoA',
    'NoseRear_AtoB', 'NoseRear_BtoA',
    'Contact', 'CoOriented', 'AntiOriented',
    'HH', 'HO', 'Sniff_AtoB', 'Sniff_BtoA',
]

# Engagement-masked speed key suffixes (without pair prefix)
_ENGAGE_SPEED_KEYS = [
    'EngageSpeed_A', 'EngageSpeed_B',
    'DisengageSpeed_A', 'DisengageSpeed_B',
    'EngageOnsetSpeed_A', 'EngageOnsetSpeed_B',
    'DisengageOnsetSpeed_A', 'DisengageOnsetSpeed_B',
]

# Canonical list of second-order behavior key suffixes (without pair prefix)
_SECOND_ORDER_KEYS = [
    'Follow_AtoB', 'Follow_BtoA',
    'Chase_AtoB', 'Chase_BtoA',
    'Flee_AtoB', 'Flee_BtoA',
    'Approach_AtoB', 'Approach_BtoA',
    'ActiveAvoid_AtoB', 'ActiveAvoid_BtoA',
    'PassiveAvoid_AtoB', 'PassiveAvoid_BtoA',
    'StationaryProx',
    'SocialOrient_AtoB', 'SocialOrient_BtoA',
    'Disengaged',
    'DisengageSpeed_A', 'DisengageSpeed_B',
    'DisengageOnsetSpeed_A', 'DisengageOnsetSpeed_B',
    'VisualAttn_AtoB', 'VisualAttn_BtoA',
    'AuditoryAttn_AtoB', 'AuditoryAttn_BtoA',
]

# Binary-only subset (for behavior summary and binned export)
_SECOND_ORDER_BINARY_KEYS = [
    'Follow_AtoB', 'Follow_BtoA',
    'Chase_AtoB', 'Chase_BtoA',
    'Flee_AtoB', 'Flee_BtoA',
    'Approach_AtoB', 'Approach_BtoA',
    'ActiveAvoid_AtoB', 'ActiveAvoid_BtoA',
    'PassiveAvoid_AtoB', 'PassiveAvoid_BtoA',
    'StationaryProx',
    'SocialOrient_AtoB', 'SocialOrient_BtoA',
    'Disengaged',
    'VisualAttn_AtoB', 'VisualAttn_BtoA',
    'AuditoryAttn_AtoB', 'AuditoryAttn_BtoA',
]


# ---------------------------------------------------------------------------
# Session-level behavior summary
# ---------------------------------------------------------------------------

def compute_behavior_summary(single_beh, pair_beh, track_names, fps,
                              tracks=None, kin=None, frame_map=None,
                              node_names=None):
    """
    Session-level statistics for single-animal and pairwise behaviors,
    plus (when tracks / kin / frame_map / node_names are supplied)
    per-minute, cumulative, and full-video Engagement / Reciprocity /
    Retreat indices.

    Parameters
    ----------
    single_beh  : dict from compute_single_animal  — (n_frames, n_tracks) int8
    pair_beh    : dict from compute_pairwise        — (n_frames,) arrays
    track_names : list[str]
    fps         : float
    tracks      : (n_frames, 2, n_nodes, n_tracks) or None
    kin         : dict from compute_kinematics or None
    frame_map   : {video_frame_idx -> sleap_data_idx} or None
    node_names  : list[str] or None

    Returns
    -------
    summary_df  : pd.DataFrame  — per-track + per-pair summary stats
    indices_df  : pd.DataFrame  — windowed EI / RI / RTI per pair
                                  (empty DataFrame when optional params absent)
    """
    rows     = []
    n_tracks = len(track_names)
    n_frames = single_beh['stationary'].shape[0]

    # ---- per-track single-animal stats ------------------------------------
    for t, name in enumerate(track_names):
        n_rev_bouts = len(_find_bouts(single_beh['dir_reversal'][:, t]))
        rows.append({
            'Subject':            name,
            'Type':               'single',
            'stationary_s':       round(float(np.sum(single_beh['stationary'][:, t])) / fps, 2),
            'walking_s':          round(float(np.sum(single_beh['walking'][:,    t])) / fps, 2),
            'running_s':          round(float(np.sum(single_beh['running'][:,     t])) / fps, 2),
            'turning_s':          round(float(np.sum(single_beh['turning'][:,     t])) / fps, 2),
            'dir_reversal_bouts': n_rev_bouts,
        })

    # ---- per-pair stats ---------------------------------------------------
    seen = set()
    for key in pair_beh:
        pfx = key.rsplit('/', 1)[0]
        if pfx in seen:
            continue
        seen.add(pfx)

        parts = pfx.split('_')
        try:
            tA_i  = int(parts[0][1:]);  tB_i = int(parts[1][1:])
            name_A = track_names[tA_i] if tA_i < n_tracks else f't{tA_i}'
            name_B = track_names[tB_i] if tB_i < n_tracks else f't{tB_i}'
        except (ValueError, IndexError):
            name_A, name_B = parts[0], parts[1]
            tA_i, tB_i = 0, 1

        engaged    = pair_beh.get(f'{pfx}/Engaged',    np.zeros(n_frames, dtype=np.int8))
        disengaged = pair_beh.get(f'{pfx}/Disengaged', np.zeros(n_frames, dtype=np.int8))

        eng_frames = int(np.sum(engaged    == 1))
        dis_frames = int(np.sum(disengaged == 1))
        eng_bouts  = _find_bouts(engaged.astype(np.int8))

        eo_A  = pair_beh.get(f'{pfx}/EngageOnsetSpeed_A', np.full(n_frames, np.nan))
        eo_B  = pair_beh.get(f'{pfx}/EngageOnsetSpeed_B', np.full(n_frames, np.nan))
        va    = eo_A[np.isfinite(eo_A)];  vb = eo_B[np.isfinite(eo_B)]
        mean_ons_A = round(float(np.mean(va)), 3) if len(va) > 0 else None
        mean_ons_B = round(float(np.mean(vb)), 3) if len(vb) > 0 else None
        ed_index   = round(eng_frames / dis_frames, 3) if dis_frames > 0 else None

        pair_row = {
            'Subject':                  f'{name_A} vs {name_B}',
            'Type':                     'pair',
            'engagement_s':             round(eng_frames / fps, 2),
            'n_engagement_bouts':       len(eng_bouts),
            'mean_engage_onset_spd_A':  mean_ons_A,
            'mean_engage_onset_spd_B':  mean_ons_B,
            'E/D_index':                ed_index,
        }

        # Second-order behavior stats (seconds and bout counts)
        _so_summary_keys = [
            ('Follow_AtoB', 'follow_AtoB'),
            ('Follow_BtoA', 'follow_BtoA'),
            ('Chase_AtoB', 'chase_AtoB'),
            ('Chase_BtoA', 'chase_BtoA'),
            ('Flee_AtoB', 'flee_AtoB'),
            ('Flee_BtoA', 'flee_BtoA'),
            ('Approach_AtoB', 'approach_AtoB'),
            ('Approach_BtoA', 'approach_BtoA'),
            ('ActiveAvoid_AtoB', 'active_avoid_AtoB'),
            ('ActiveAvoid_BtoA', 'active_avoid_BtoA'),
            ('PassiveAvoid_AtoB', 'passive_avoid_AtoB'),
            ('PassiveAvoid_BtoA', 'passive_avoid_BtoA'),
            ('StationaryProx', 'stationary_prox'),
            ('SocialOrient_AtoB', 'social_orient_AtoB'),
            ('SocialOrient_BtoA', 'social_orient_BtoA'),
        ]
        for beh_key, col_prefix in _so_summary_keys:
            arr = pair_beh.get(f'{pfx}/{beh_key}', np.zeros(n_frames, dtype=np.int8))
            if np.issubdtype(arr.dtype, np.integer):
                n_beh_frames = int(np.sum(arr == 1))
                n_beh_bouts = len(_find_bouts(arr.astype(np.int8)))
            else:
                n_beh_frames = 0
                n_beh_bouts = 0
            pair_row[f'{col_prefix}_s'] = round(n_beh_frames / fps, 2)
            pair_row[f'{col_prefix}_bouts'] = n_beh_bouts

        rows.append(pair_row)

    summary_df = pd.DataFrame(rows)

    # ---- windowed engagement indices (optional) ---------------------------
    if tracks is None or kin is None or frame_map is None:
        return summary_df, pd.DataFrame()

    body_idx = (find_node_idx(node_names, 'body')
                if node_names else None)

    # sleap_idx -> video time in seconds
    si_to_vt = {}
    for vf, si in frame_map.items():
        t_s = vf / fps
        if si not in si_to_vt or t_s < si_to_vt[si]:
            si_to_vt[si] = t_s

    max_video_s = max(si_to_vt.values()) if si_to_vt else 0.0

    # Minute boundaries (0, 60, 120, ...)
    min_s = 60.0
    n_mins = max(1, int(np.ceil(max_video_s / min_s)))
    # boundaries[i] = start of minute i+1  (0-based)
    boundaries = [i * min_s for i in range(n_mins + 1)]

    idx_rows = []

    seen2 = set()
    for key in pair_beh:
        pfx = key.rsplit('/', 1)[0]
        if pfx in seen2:
            continue
        seen2.add(pfx)

        parts = pfx.split('_')
        try:
            tA_i = int(parts[0][1:]);  tB_i = int(parts[1][1:])
            name_A = track_names[tA_i] if tA_i < n_tracks else f't{tA_i}'
            name_B = track_names[tB_i] if tB_i < n_tracks else f't{tB_i}'
        except (ValueError, IndexError):
            continue

        engaged = pair_beh.get(f'{pfx}/Engaged', np.zeros(n_frames, dtype=np.int8))
        all_bouts = _bouts_with_initiator(
            engaged, tracks, kin, body_idx, tA_i, tB_i, fps)

        # Attach video-time to each bout
        for b in all_bouts:
            b['vt'] = si_to_vt.get(b['start'], None)

        valid_bouts = [b for b in all_bouts if b['vt'] is not None]
        pair_label  = f'{name_A} vs {name_B}'

        def _row(w_type, w_label, bout_list):
            idx = _indices_from_bouts(bout_list, fps)
            return {
                'Pair':        pair_label,
                'Track_A':     name_A,
                'Track_B':     name_B,
                'Window_Type': w_type,
                'Window':      w_label,
                'n_bouts':     idx['n_bouts'],
                'n_init_A':    idx['n_init_A'],
                'n_init_B':    idx['n_init_B'],
                'EI_A':        idx['EI_A'],
                'EI_B':        idx['EI_B'],
                'RI_A':        idx['RI_A'],
                'RI_B':        idx['RI_B'],
                'RTI_A':       idx['RTI_A'],
                'RTI_B':       idx['RTI_B'],
            }

        # Per-minute segments
        for i in range(n_mins):
            t0, t1 = boundaries[i], boundaries[i + 1]
            seg = [b for b in valid_bouts if t0 <= b['vt'] < t1]
            label = f'min_{i + 1}'
            idx_rows.append(_row('per_minute', label, seg))

        # Cumulative windows
        for i in range(1, n_mins + 1):
            t1 = boundaries[i]
            seg = [b for b in valid_bouts if b['vt'] < t1]
            label = f'0-{i}min'
            idx_rows.append(_row('cumulative', label, seg))

        # Full video
        idx_rows.append(_row('full_video', 'full', valid_bouts))

    indices_df = pd.DataFrame(idx_rows) if idx_rows else pd.DataFrame()
    return summary_df, indices_df
