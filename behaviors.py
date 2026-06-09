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


# ---------------------------------------------------------------------------
# Node lookup
# ---------------------------------------------------------------------------

def find_node_idx(node_names, *patterns):
    """Case-insensitive substring search through node_names.

    Returns the index of the first node whose name contains any of the
    given patterns, or None if no match.
    """
    lower = [n.lower() for n in node_names]
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

def compute_single_animal(tracks, kin, node_names, fps):
    """Per-frame locomotor and turning states for every track.

    Parameters
    ----------
    tracks     : (n_frames, 2, n_nodes, n_tracks)
    kin        : dict from preprocessing.compute_kinematics
    node_names : list[str]
    fps        : float

    Returns
    -------
    dict of int8 arrays, each shape (n_frames, n_tracks):
        'stationary', 'walking', 'running', 'turning', 'dir_reversal'
    """
    n_frames, _, n_nodes, n_tracks = tracks.shape
    cm_idx = find_node_idx(node_names, 'center', 'body', 'cm', 'centroid')

    out = {k: np.zeros((n_frames, n_tracks), dtype=np.int8)
           for k in ('stationary', 'walking', 'running', 'turning', 'dir_reversal')}

    for t in range(n_tracks):
        if cm_idx is not None:
            speed   = kin['speed'][:, cm_idx, t].astype(np.float64)
            heading = kin['heading_deg'][:, cm_idx, t].astype(np.float64)
        else:
            speed   = np.nanmean(kin['speed'][:, :, t], axis=1).astype(np.float64)
            heading = np.nanmean(kin['heading_deg'][:, :, t], axis=1).astype(np.float64)

        speed   = np.nan_to_num(speed,   nan=0.0)
        heading = np.nan_to_num(heading, nan=0.0)

        # Speed thresholds are session-relative percentiles, not absolute values.
        # "Stationary" = bottom 15% of this animal's own speed distribution for the
        # session; "running" = top 25% (above the 75th percentile); "walking" = middle.
        # Consequence: a completely inactive animal will still have 15% of its frames
        # classified as "stationary" and 25% as "running" purely based on the shape of
        # its own speed distribution — even if its absolute speeds are tiny.
        # This is a deliberate relative-classification design choice: it captures
        # locomotor state relative to the animal's own baseline rather than requiring
        # a hard-coded speed threshold that would need to be recalibrated for every
        # camera/arena/species combination.
        p15 = np.percentile(speed, 15)
        p75 = np.percentile(speed, 75)

        out['stationary'][:, t] = (speed < p15).astype(np.int8)
        out['walking'][:,    t] = ((speed >= p15) & (speed <= p75)).astype(np.int8)
        out['running'][:,    t] = (speed > p75).astype(np.int8)

        av = angular_velocity(heading, fps)
        out['turning'][:, t] = (np.abs(av) > 30.0).astype(np.int8)

        # Directional reversal: sign flip in a ±2-frame window, both sides > 20 deg/s
        rev  = np.zeros(n_frames, dtype=np.int8)
        half = 2
        av_abs = np.abs(av)
        for f in range(half, n_frames - half):
            win  = av[f - half: f + half + 1]
            mask = av_abs[f - half: f + half + 1] > 20.0
            if np.any(mask) and np.any(win[mask] > 0) and np.any(win[mask] < 0):
                rev[f] = 1
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
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == 0:
            j = i
            while j < n and arr[j] == 0:
                j += 1
            gap   = j - i
            left  = (i > 0) and (arr[i - 1] == 1)
            right = (j < n) and (arr[j]     == 1)
            if gap <= max_gap and left and right:
                arr[i:j] = 1
            i = j
        else:
            i += 1
    return arr


def _find_bouts(arr):
    """Return list of (start, end_inclusive) for runs of ones."""
    bouts, n, i = [], len(arr), 0
    while i < n:
        if arr[i] == 1:
            s = i
            while i < n and arr[i] == 1:
                i += 1
            bouts.append((s, i - 1))
        else:
            i += 1
    return bouts


def _frame_speed(pos):
    """Per-frame speed (pixels/frame) from (n_frames, 2) array."""
    dx = np.diff(pos[:, 0], prepend=pos[0, 0])
    dy = np.diff(pos[:, 1], prepend=pos[0, 1])
    return np.hypot(dx, dy)


# ---------------------------------------------------------------------------
# Pairwise social behaviors
# ---------------------------------------------------------------------------

def compute_pairwise(tracks, node_names, fps, dsr=None):
    """Compute pairwise social behavior arrays for every unique track pair.

    Parameters
    ----------
    tracks     : (n_frames, 2, n_nodes, n_tracks)
    node_names : list[str]
    fps        : float
    dsr        : float or None — auto-computed from hip nodes if None

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
        hl  = find_node_idx(node_names, 'hip_l', 'hl')
        hr  = find_node_idx(node_names, 'hip_r', 'hr')
        dsr = compute_dsr(tracks, hl, hr)
        if dsr is None:
            dsr = _fallback_dsr(tracks)

    # --- key node indices ---
    nose_idx  = find_node_idx(node_names, 'nose')
    body_idx  = find_node_idx(node_names, 'center', 'body', 'cm', 'centroid')
    tail_idx  = find_node_idx(node_names, 'tail_base', 'tb')
    ear_l_idx = find_node_idx(node_names, 'ear_l', 'el')
    ear_r_idx = find_node_idx(node_names, 'ear_r', 'er')

    out   = {}
    pairs = list(combinations(range(n_tracks), 2))

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

        # --- contact: any node of A within 0.25×DSR of any node of B ---
        pA   = tracks[:, :, :, tA]   # (n_frames, 2, n_nodes)
        pB   = tracks[:, :, :, tB]
        diff = pA[:, :, :, np.newaxis] - pB[:, :, np.newaxis, :]
        # diff shape: (n_frames, 2, n_nodes, n_nodes)
        node_dists = np.hypot(diff[:, 0], diff[:, 1])  # (n_frames, n_nodes, n_nodes)
        min_dist   = np.nanmin(node_dists, axis=(1, 2))
        out[f'{pfx}/Contact'] = (min_dist < 0.25 * dsr).astype(np.int8)

        # --- relative position, co/anti-orientation, engagement ---
        if body_A is not None and body_B is not None:
            hdg_A = _smooth_heading_deg(body_A)  # (n_frames,) deg
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

            # --- Disengagement (vectorised) ---
            # Disengagement is defined directionally — a simple "not engaged" frame does NOT
            # count.  A frame is classified as "disengaged" only when all three conditions
            # are true simultaneously:
            #   (a) engaged == 0        — not currently in an engagement bout,
            #   (b) recent_sum > 0      — was engaged at some point in the last 0.75 s
            #                             (i.e. the engagement ended recently), AND
            #   (c) dist_inc == True    — the inter-animal distance is actively increasing
            #                             (animals are spatially separating, not just pausing).
            # The dist_inc criterion is the directional component: it distinguishes an
            # animal that has just turned away and is moving apart (true disengagement)
            # from one that briefly stopped facing the other but stayed close.
            hist = int(round(0.75 * fps))
            # recent_sum[f] = sum(engaged[max(0, f-hist) : f])
            padded = np.pad(engaged, (hist, 0), constant_values=0)
            cs     = np.cumsum(padded)
            f_idx  = np.arange(n_frames)
            recent_sum = cs[f_idx + hist] - cs[f_idx]

            dist_inc = np.zeros(n_frames, dtype=bool)
            dist_inc[1:] = cm_dist[1:] > cm_dist[:-1] + 0.05 * dsr

            disengaged = ((engaged == 0) &
                          (recent_sum > 0) &
                          dist_inc).astype(np.int8)
            out[f'{pfx}/Disengaged'] = disengaged

            # --- Speed during engagement / disengagement frames ---
            spd_A = _frame_speed(body_A)
            spd_B = _frame_speed(body_B)

            def _masked(spd, mask):
                arr = np.full(n_frames, np.nan)
                arr[mask == 1] = spd[mask == 1]
                return arr

            out[f'{pfx}/EngageSpeed_A']  = _masked(spd_A, engaged)
            out[f'{pfx}/EngageSpeed_B']  = _masked(spd_B, engaged)
            out[f'{pfx}/DisengageSpeed_A'] = _masked(spd_A, disengaged)
            out[f'{pfx}/DisengageSpeed_B'] = _masked(spd_B, disengaged)

            # Onset speeds (first frame of each bout only)
            eng_onset_A = np.full(n_frames, np.nan)
            eng_onset_B = np.full(n_frames, np.nan)
            for start, _ in _find_bouts(engaged):
                eng_onset_A[start] = spd_A[start]
                eng_onset_B[start] = spd_B[start]
            out[f'{pfx}/EngageOnsetSpeed_A'] = eng_onset_A
            out[f'{pfx}/EngageOnsetSpeed_B'] = eng_onset_B

            dis_onset_A = np.full(n_frames, np.nan)
            dis_onset_B = np.full(n_frames, np.nan)
            for start, _ in _find_bouts(disengaged):
                dis_onset_A[start] = spd_A[start]
                dis_onset_B[start] = spd_B[start]
            out[f'{pfx}/DisengageOnsetSpeed_A'] = dis_onset_A
            out[f'{pfx}/DisengageOnsetSpeed_B'] = dis_onset_B

        else:
            # No body-center node — fill with zeros / NaN / empty strings
            zeros = np.zeros(n_frames, dtype=np.int8)
            nans  = np.full(n_frames, np.nan)
            empty = np.full(n_frames, '', dtype=object)
            for key, val in [
                ('RelPos_A', empty.copy()), ('RelPos_B', empty.copy()),
                ('CoOriented', zeros.copy()), ('AntiOriented', zeros.copy()),
                ('Engaged', zeros.copy()), ('Disengaged', zeros.copy()),
                ('EngageSpeed_A', nans.copy()), ('EngageSpeed_B', nans.copy()),
                ('EngageOnsetSpeed_A', nans.copy()), ('EngageOnsetSpeed_B', nans.copy()),
                ('DisengageSpeed_A', nans.copy()), ('DisengageSpeed_B', nans.copy()),
                ('DisengageOnsetSpeed_A', nans.copy()), ('DisengageOnsetSpeed_B', nans.copy()),
            ]:
                out[f'{pfx}/{key}'] = val

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
    hdg = kin['heading_deg'][f0:f1, body_idx, tA].astype(np.float64)
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

        rows.append({
            'Subject':                  f'{name_A} vs {name_B}',
            'Type':                     'pair',
            'engagement_s':             round(eng_frames / fps, 2),
            'n_engagement_bouts':       len(eng_bouts),
            'mean_engage_onset_spd_A':  mean_ons_A,
            'mean_engage_onset_spd_B':  mean_ons_B,
            'E/D_index':                ed_index,
        })

    summary_df = pd.DataFrame(rows)

    # ---- windowed engagement indices (optional) ---------------------------
    if tracks is None or kin is None or frame_map is None:
        return summary_df, pd.DataFrame()

    body_idx = (find_node_idx(node_names, 'center', 'body', 'cm', 'centroid')
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
