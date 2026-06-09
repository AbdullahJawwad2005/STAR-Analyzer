"""
binned_export.py — Time-binned feature export for STAR Analyzer
================================================================
Takes pre-computed per-frame feature arrays (from features.py and behaviors.py)
and aggregates them into fixed-duration time bins for compact temporal analysis.

Two resolution levels are produced:

  0.25 s bins — direct aggregation of per-frame data with type-specific rules:
      distance / path       : mean + median
      angular (wrapped deg) : circular mean  [arctan2(mean sin, mean cos)]
      velocity / speed      : median + 90th percentile
      acceleration          : median + 90th percentile
      jerk                  : |median| + |90th percentile|
      binary flags          : proportion  (n_true / n_frames_in_bin)
      position (x/y)        : mean
      shape scalars         : mean
      engagement speeds     : median of non-NaN values

  1.0 s bins — re-aggregation of the four 0.25 s sub-bins (not re-derived
               from raw frames).  Binary proportions and engagement columns
               are omitted at this level.

For pair features, position covariance and correlation are recomputed within
each 0.25 s bin using proper within-bin Pearson statistics rather than
averaging the session-mean-centred frame products stored in the frame-level
pair arrays.

Engagement and disengagement bouts are re-detected at bin resolution; the
Engagement Index (EI), Reciprocity Index (RI), and Retreat Index (RTI) are
then recomputed from those bin-level bout lists.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from behaviors import (
    find_node_idx,
    _find_bouts,
    _mean_approach_angle_arr,
    _indices_from_bouts,
)


# ─────────────────────────────────────────────────────────────────────────────
# Circular statistics
# ─────────────────────────────────────────────────────────────────────────────

def _circular_mean_deg(values):
    """
    Circular (von Mises) mean of an array of angles in degrees.

    A plain arithmetic mean of wrapped angles is incorrect — for example,
    averaging −170° and +170° would give 0° instead of ±180°.  The correct
    approach converts each angle to a unit vector, averages the vectors, and
    converts back via arctan2.

    Parameters
    ----------
    values : array-like  — angles in degrees (finite values only)

    Returns
    -------
    float — mean angle in [−180°, 180°], or np.nan if no finite values.
    """
    arr = np.asarray(values, dtype=np.float64)
    ok  = arr[np.isfinite(arr)]
    if len(ok) == 0:
        return np.nan
    rad = np.radians(ok)
    return float(np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


# ─────────────────────────────────────────────────────────────────────────────
# Feature-type classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_feat(name: str) -> str:
    """
    Map a feature column name to an aggregation category.

    The mapping is purely name-based (no schema introspection) so it works
    for both track-level and pair-level feature dictionaries.

    Category strings and their aggregation in _agg_025():
        'position'    raw x/y coordinates               → mean
        'distance'    Euclidean distances, areas, paths  → mean + median
        'angular'     wrapped angles in degrees          → circular mean
        'velocity'    velocity components and speeds     → median + p90
        'accel'       acceleration                       → median + p90
        'jerk'        jerk (3rd derivative of position)  → |median| + |p90|
        'binary'      0/1 flag arrays                   → proportion
        'shape'       body-shape scalars                 → mean
        'engage_spd'  engagement-masked speed columns    → median (NaN-safe)
        'categorical' string category columns            → skipped entirely
        'other'       fallback                           → mean
    """
    n = name.lower()

    # ── Raw coordinates ──────────────────────────────────────────────────────
    if n.endswith('_x') or n.endswith('_y'):
        return 'position'

    # ── Angular  (check before velocity; '_ang_mot' must not match velocity) ─
    if any(k in n for k in ('_angle', 'ang_mot', 'heading', 'curvature',
                             'approach_angle')):
        return 'angular'

    # ── Jerk  (check before speed / velocity to avoid 'jerk' matching '_vx') ─
    if 'jerk' in n:
        return 'jerk'

    # Note on check order: 'speed' appears as a substring in column names such as
    # 'EngageOnsetSpeed_A', but those names are only ever encountered through the
    # hardcoded engagement speed loop in build_025s_bins(), not through this function.
    # _classify_feat() is only called on column names from track_arrays (features.py)
    # and pair_arrays (features.py).  Behavior column names from pair_beh (behaviors.py)
    # bypass _classify_feat() entirely and are handled by dedicated hardcoded loops.
    # The check order below is therefore safe: 'speed' here only matches raw node
    # speed columns (e.g. 'nose_speed'), not engagement-masked speed columns.

    # ── Velocity / speed ─────────────────────────────────────────────────────
    if any(k in n for k in ('speed', '_vx', '_vy', 'displacement')):
        return 'velocity'

    # ── Acceleration ─────────────────────────────────────────────────────────
    if 'accel' in n:
        return 'accel'

    # ── Binary behavior flags ─────────────────────────────────────────────────
    _BINARY = ('stationary', 'walking', 'running', 'turning', 'dir_reversal',
               'nosenose', 'nosehead', 'nosebody', 'noserear',
               'contact', 'cooriented', 'antioriented',
               'engaged', 'disengaged')
    if any(k in n for k in _BINARY):
        return 'binary'

    # ── Engagement-masked speed columns (many NaNs by design) ────────────────
    # DEAD CODE NOTE: the 'engage_spd' category below is not reachable in the
    # current architecture.  Engagement speed columns (EngageSpeed_A, EngageOnsetSpeed_A,
    # etc.) come from pair_beh (behaviors.py) and are aggregated by a separate
    # hardcoded loop in build_025s_bins(), not through _classify_feat().  The column
    # names matched here would only be encountered if they were added to the pair_arrays
    # feature dictionary (features.py), which they currently are not.
    # The category is kept for forward-compatibility in case engagement speeds are
    # ever moved into pair_arrays and routed through _classify_feat().
    if any(k in n for k in ('engagespeed', 'disengagespeed',
                             'engageonsetspe', 'disengageonsetsp')):
        return 'engage_spd'

    # ── Shape / geometry scalars ──────────────────────────────────────────────
    if any(k in n for k in ('elongation', 'eccentricity', 'compactness',
                             'circularity', 'hourglass', 'path_eff',
                             'position_entropy', 'total_disp')):
        return 'shape'

    # ── Distances and related numeric measures ────────────────────────────────
    if any(k in n for k in ('dist', 'area', 'covar', 'corr')):
        return 'distance'

    # ── Categorical string fields — excluded from numeric aggregation ─────────
    if any(k in n for k in ('relpos', 'visual_scope', 'auditory_scope')):
        return 'categorical'

    return 'other'


# ─────────────────────────────────────────────────────────────────────────────
# Per-bin aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _agg_025(arr: np.ndarray, category: str) -> dict:
    """
    Aggregate a 1-D array of per-frame values from one 0.25 s bin.

    Only finite (non-NaN, non-Inf) values participate in statistics.
    For 'binary', the denominator is the full bin width (including NaN frames)
    so that bins near occlusions don't appear artificially high.

    Parameters
    ----------
    arr      : np.ndarray, dtype float64 — frame values within this bin
    category : str — from _classify_feat()

    Returns
    -------
    dict mapping column suffix → aggregated Python float (or np.nan).
    Multiple suffixes are returned for categories that produce more than
    one output (e.g. 'distance' yields '_mean' and '_median').
    """
    a  = np.asarray(arr, dtype=np.float64)
    ok = a[np.isfinite(a)]          # finite subset
    n  = len(a)                      # total frames in bin (denominator for binary)

    def _q(q):
        """Percentile, or nan if no finite values."""
        return float(np.percentile(ok, q)) if len(ok) > 0 else np.nan

    if category == 'position':
        # Mean position within the bin
        return {'_mean': float(np.mean(ok)) if len(ok) > 0 else np.nan}

    if category == 'distance':
        # Median is more robust than mean for skewed distance distributions
        return {
            '_mean':   float(np.mean(ok))   if len(ok) > 0 else np.nan,
            '_median': float(np.median(ok)) if len(ok) > 0 else np.nan,
        }

    if category == 'angular':
        # Must use circular mean — arithmetic average would be wrong for
        # angles crossing the ±180° boundary.
        return {'_cmean': _circular_mean_deg(ok)}

    if category in ('velocity', 'accel'):
        # p90 captures burst events that a median alone would miss
        return {'_median': _q(50), '_p90': _q(90)}

    if category == 'jerk':
        # Direction is noise at the jerk floor; use absolute values so that
        # large positive/negative jerks don't cancel in the bin mean.
        med = _q(50)
        p90 = _q(90)
        return {
            '_absmedian': abs(med) if np.isfinite(med) else np.nan,
            '_absp90':    abs(p90) if np.isfinite(p90) else np.nan,
        }

    if category == 'binary':
        # Proportion of ALL frames in the bin that are True (include NaN/missing
        # as 0 in denominator so bins near occlusions are not inflated).
        prop = float(np.nansum(a) / n) if n > 0 else np.nan
        return {'_prop': prop}

    if category in ('shape', 'other'):
        return {'_mean': float(np.mean(ok)) if len(ok) > 0 else np.nan}

    if category == 'engage_spd':
        # Most frames in a bin will be NaN (not engaged); take median of the
        # subset that are non-NaN, yielding NaN if the animal was never
        # engaged during this bin.
        return {'_median': float(np.median(ok)) if len(ok) > 0 else np.nan}

    # Fallback
    return {'_mean': float(np.mean(ok)) if len(ok) > 0 else np.nan}


# ─────────────────────────────────────────────────────────────────────────────
# Bin-map construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_bin_map(frame_map: dict, fps: float, bin_size_s: float = 0.25) -> dict:
    """
    Partition sleap frame indices into fixed-duration time bins.

    Bin assignment: bin_idx = floor(video_frame / fps / bin_size_s).
    Bin 0 therefore covers video frames with t ∈ [0, bin_size_s),
    bin 1 covers [bin_size_s, 2·bin_size_s), and so on.

    Parameters
    ----------
    frame_map  : {video_frame_idx -> sleap_data_idx}
    fps        : float — video frame rate
    bin_size_s : float — bin width in seconds (default 0.25)

    Returns
    -------
    dict  bin_idx (int) → sorted list of sleap_data_indices
    """
    bins: dict[int, list] = defaultdict(list)
    for vid_frame, sleap_idx in sorted(frame_map.items()):
        bin_idx = int(vid_frame / fps / bin_size_s)
        bins[bin_idx].append(sleap_idx)
    return dict(sorted(bins.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Within-bin Pearson covariance / correlation
# ─────────────────────────────────────────────────────────────────────────────

def _bin_cov_corr(a_arr: np.ndarray, b_arr: np.ndarray,
                  sleap_idxs: list) -> tuple[float, float]:
    """
    Compute Pearson covariance and correlation between two position arrays
    restricted to the frames within one bin.

    This gives a true local co-movement statistic for the bin rather than
    averaging the session-mean-centred frame products stored in the frame-level
    pair arrays (which would reflect session-level positional drift, not
    moment-to-moment synchrony).

    Parameters
    ----------
    a_arr, b_arr : (n_frames,) arrays — position values for the two animals
    sleap_idxs   : list of ints — frame indices belonging to this bin

    Returns
    -------
    (cov, corr) — both np.nan when fewer than 2 finite paired observations.
    """
    idx  = np.array(sleap_idxs, dtype=int)
    a, b = a_arr[idx], b_arr[idx]
    ok   = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return np.nan, np.nan

    a_ok, b_ok = a[ok].astype(np.float64), b[ok].astype(np.float64)
    cov  = float(np.cov(a_ok, b_ok)[0, 1])
    sa   = float(np.std(a_ok, ddof=1))
    sb   = float(np.std(b_ok, ddof=1))
    corr = (cov / (sa * sb)) if sa > 0 and sb > 0 else np.nan
    return cov, corr


# ─────────────────────────────────────────────────────────────────────────────
# Bin-level engagement bout detection
# ─────────────────────────────────────────────────────────────────────────────

def _bouts_with_initiator_binned(engaged_bin: np.ndarray,
                                  bin_list: list,
                                  tracks: np.ndarray,
                                  kin: dict,
                                  body_idx,
                                  tA: int, tB: int,
                                  fps: float,
                                  pre_bins: int = 2,
                                  post_bins: int = 3) -> list:
    """
    Detect engagement bouts in a binary bin-resolution engaged sequence and
    determine the initiator and disengager for each bout.

    Mirrors _bouts_with_initiator() in behaviors.py but works at bin
    resolution: pre/post windows are measured in bins and mapped back to the
    original sleap frame indices for the approach-angle computation.

    Parameters
    ----------
    engaged_bin : (n_bins,) int8 array — 1 if bin is engaged (>50% frames), else 0
    bin_list    : list of (bin_idx, [sleap_idxs])  — ordered bin contents
    tracks, kin : full-resolution arrays for approach-angle computation
    body_idx    : int or None — body-centre node index
    tA, tB      : track indices (A and B)
    fps         : video fps (used by _mean_approach_angle_arr)
    pre_bins    : bins before bout onset used for approach-angle measurement
    post_bins   : bins after bout end used for post-bout angle measurement

    Returns
    -------
    list of dicts — each dict has keys:
        start      : int  (index into bin_list / engaged_bin)
        end        : int  (index, inclusive)
        initiator  : int  0 = tA initiated, 1 = tB initiated
        disengager : int  0 = tA disengaged, 1 = tB disengaged
    """
    bouts  = _find_bouts(engaged_bin.astype(np.int8))
    n_bins = len(bin_list)
    result = []

    for s, e in bouts:
        # ── Collect sleap frames in the pre-bout window ───────────────────────
        pre_frames = []
        for bi in range(max(0, s - pre_bins), s + 1):
            if bi < n_bins:
                pre_frames.extend(bin_list[bi][1])

        # ── Collect sleap frames in the post-bout window ──────────────────────
        post_frames = []
        for bi in range(e + 1, min(n_bins, e + post_bins + 1)):
            post_frames.extend(bin_list[bi][1])

        # Fallback: use bout frames when windows are empty
        if not pre_frames:
            for bi in range(s, min(e + 1, n_bins)):
                pre_frames.extend(bin_list[bi][1])
        if not post_frames:
            for bi in range(max(0, e - 1), min(e + 1, n_bins)):
                post_frames.extend(bin_list[bi][1])

        # ── Determine initiator (smaller pre-bout approach angle) ─────────────
        if pre_frames:
            # This takes the SPAN of the pre-bout sleap frame indices (min to max+1)
            # rather than iterating over the individual indices.  If the pre-bout bins
            # contain non-contiguous SLEAP frame indices — which can happen when
            # frame_map has gaps in coverage (e.g. dropped frames or sparse sampling)
            # — some gap frames between min(pre_frames) and max(pre_frames) will be
            # included in the range passed to _mean_approach_angle_arr.  These extra
            # frames will have no valid position data (NaN) and are filtered out by
            # the `ok` mask inside _mean_approach_angle_arr, so the result is still
            # correct.  The span approach is simpler and faster than passing the
            # explicit index list.
            f0, f1 = min(pre_frames), max(pre_frames) + 1
            aa_A = _mean_approach_angle_arr(tracks, kin, body_idx, tA, tB, f0, f1)
            aa_B = _mean_approach_angle_arr(tracks, kin, body_idx, tB, tA, f0, f1)
        else:
            aa_A = aa_B = 90.0

        initiator = 0 if aa_A <= aa_B else 1

        # ── Determine disengager (larger post-bout approach angle = turning away) ─
        if post_frames:
            f0p, f1p = min(post_frames), max(post_frames) + 1
            aa_A_post = _mean_approach_angle_arr(tracks, kin, body_idx, tA, tB, f0p, f1p)
            aa_B_post = _mean_approach_angle_arr(tracks, kin, body_idx, tB, tA, f0p, f1p)
        else:
            aa_A_post = aa_B_post = 90.0

        disengager = 0 if aa_A_post >= aa_B_post else 1

        result.append(dict(start=s, end=e, initiator=initiator, disengager=disengager))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main 0.25 s bin builder
# ─────────────────────────────────────────────────────────────────────────────

def build_025s_bins(track_arrays: dict,
                    pair_arrays: dict,
                    single_beh: dict,
                    pair_beh: dict,
                    tracks: np.ndarray,
                    node_names: list,
                    track_names: list,
                    fps: float,
                    frame_map: dict,
                    kin: dict | None = None,
                    bin_size_s: float = 0.25):
    """
    Aggregate all per-frame feature and behavior arrays into 0.25-second bins.

    Special handling beyond simple category aggregation:
    • Position covariance/correlation — recomputed with proper within-bin
      Pearson statistics (the frame-level values in pair_arrays are
      session-centred products that reflect long-range drift, not local sync).
    • Engagement at bin level — a bin is "engaged" if > 50% of its frames
      are engaged.  Bouts are re-detected in this binary bin sequence and
      EI / RI / RTI indices are recomputed from the bin-level bout list.

    Parameters
    ----------
    track_arrays : dict[int, dict[str, np.ndarray]]  — from precompute_feature_arrays
    pair_arrays  : dict[str, np.ndarray]              — from precompute_feature_arrays
    single_beh   : dict[str, (n_frames, n_tracks)]   — from compute_single_animal
    pair_beh     : dict[str, (n_frames,)]            — from compute_pairwise
    tracks       : (n_frames, 2, n_nodes, n_tracks)
    node_names   : list[str]
    track_names  : list[str]
    fps          : float
    frame_map    : {video_frame_idx -> sleap_data_idx}
    kin          : kinematics dict or None
                   (needed for engagement-index computation; indices skipped if None)
    bin_size_s   : float (default 0.25)

    Returns
    -------
    animal_025  : pd.DataFrame — BinIdx, BinTime(s), Track, {feat}_*
    pair_025    : pd.DataFrame — BinIdx, BinTime(s), Track_A, Track_B, {feat}_*
    eng_idx_025 : pd.DataFrame — engagement / reciprocity / retreat indices
    """
    n_tracks = tracks.shape[3]
    bins     = _build_bin_map(frame_map, fps, bin_size_s)
    bin_fps  = 1.0 / bin_size_s          # effective fps at bin resolution (e.g. 4)

    # Expected number of video frames per bin (e.g. 7-8 at 30 fps / 0.25 s).
    # Used as the denominator for proportion columns instead of len(idx_arr)
    # (which is only the *detected* frames in the bin).  For sparse frame_maps
    # len(idx_arr) can be much smaller than the true window width, inflating
    # proportions: 1 detected engaged frame out of 7 total would give 1.0
    # instead of ~0.14.  Using the nominal bin width keeps proportions comparable
    # across bins regardless of tracking coverage.
    n_expected = max(1, round(fps * bin_size_s))

    # Ordered list for sequential processing; bin_list[i] = (bin_idx, [sleap_idxs])
    bin_list = list(bins.items())
    n_bins   = len(bin_list)

    # Body-centre node for covariance re-computation and approach angles
    body_idx = find_node_idx(node_names, 'center', 'body', 'cm', 'centroid')

    # ── Pre-classify track feature columns (done once, not per-bin) ───────────
    # Skip categorical columns entirely (strings don't aggregate numerically).
    track_col_cats: dict[int, list] = {}
    for t in range(n_tracks):
        cols = []
        for col in track_arrays[t]:
            cat = _classify_feat(col)
            if cat != 'categorical':
                cols.append((col, cat))
        track_col_cats[t] = cols

    # ── Pre-classify pair feature columns ────────────────────────────────────
    # The frame-level covariance/correlation columns are replaced by proper
    # within-bin statistics computed in the loop below.
    _RECOMPUTE_PAIR = frozenset({
        'pos_covariance_x', 'pos_covariance_y',
        'pos_correlation_x', 'pos_correlation_y',
    })
    pair_col_cats: dict[str, list] = {}
    for key in pair_arrays:
        pfx, col = key.rsplit('/', 1)
        if col in _RECOMPUTE_PAIR:
            continue                   # will be recomputed per-bin
        cat = _classify_feat(col)
        if cat == 'categorical':
            continue                   # strings excluded from numeric output
        pair_col_cats.setdefault(pfx, []).append((col, cat, key))

    # ── Build pair prefix → (tA, tB, name_A, name_B) mapping ─────────────────
    pair_pfx_map: dict[str, tuple] = {}
    for tA, tB in combinations(range(n_tracks), 2):
        pfx = f't{tA}_t{tB}'
        nA  = track_names[tA] if tA < len(track_names) else f't{tA}'
        nB  = track_names[tB] if tB < len(track_names) else f't{tB}'
        pair_pfx_map[pfx] = (tA, tB, nA, nB)

    # ────────────────────────────────────────────────────────────────────────
    # Animal 0.25 s bins
    # ────────────────────────────────────────────────────────────────────────
    animal_rows = []
    for bin_idx, sleap_idxs in bin_list:
        bin_time = round(bin_idx * bin_size_s, 4)
        idx_arr  = np.array(sleap_idxs, dtype=int)

        for t, tname in enumerate(track_names):
            row: dict = {'BinIdx': bin_idx, 'BinTime(s)': bin_time, 'Track': tname}

            # Primitive and derivative feature arrays
            for col, cat in track_col_cats[t]:
                vals = track_arrays[t][col][idx_arr]
                for suf, val in _agg_025(vals, cat).items():
                    row[col + suf] = val

            # Single-animal behavior proportions (stationary/walking/running/…)
            for bk in ('stationary', 'walking', 'running', 'turning', 'dir_reversal'):
                if bk in single_beh:
                    vals = single_beh[bk][idx_arr, t].astype(np.float64)
                    row[f'{bk}_prop'] = float(vals.sum() / n_expected)

            animal_rows.append(row)

    animal_025 = pd.DataFrame(animal_rows)

    # ────────────────────────────────────────────────────────────────────────
    # Pair 0.25 s bins
    # ────────────────────────────────────────────────────────────────────────
    pair_rows = []

    if pair_col_cats:
        for bin_idx, sleap_idxs in bin_list:
            bin_time = round(bin_idx * bin_size_s, 4)
            idx_arr  = np.array(sleap_idxs, dtype=int)

            for pfx, (tA, tB, nA, nB) in pair_pfx_map.items():
                if pfx not in pair_col_cats:
                    continue

                row: dict = {
                    'BinIdx': bin_idx, 'BinTime(s)': bin_time,
                    'Track_A': nA, 'Track_B': nB,
                }

                # Standard feature aggregation per column category
                for col, cat, key in pair_col_cats[pfx]:
                    vals = pair_arrays[key][idx_arr]
                    for suf, val in _agg_025(vals, cat).items():
                        row[col + suf] = val

                # ── Recomputed within-bin covariance / correlation ──────────
                # Proper Pearson statistics on the raw positions within this bin.
                if body_idx is not None:
                    xA = tracks[:, 0, body_idx, tA]
                    yA = tracks[:, 1, body_idx, tA]
                    xB = tracks[:, 0, body_idx, tB]
                    yB = tracks[:, 1, body_idx, tB]
                    cov_x, corr_x = _bin_cov_corr(xA, xB, sleap_idxs)
                    cov_y, corr_y = _bin_cov_corr(yA, yB, sleap_idxs)
                    row['pos_covariance_x_bin']  = cov_x
                    row['pos_covariance_y_bin']  = cov_y
                    row['pos_correlation_x_bin'] = corr_x
                    row['pos_correlation_y_bin'] = corr_y

                # ── Pair behavior proportions ───────────────────────────────
                # WARNING: this proximity behavior list is hardcoded here and must be
                # kept in sync with the behavior types produced by compute_pairwise() in
                # behaviors.py.  If a new proximity behavior key is added to that function,
                # it must also be added to this list for it to appear in the binned export.
                for bk in ('NoseNose', 'NoseHead_AtoB', 'NoseHead_BtoA',
                            'NoseBody_AtoB', 'NoseBody_BtoA',
                            'NoseRear_AtoB', 'NoseRear_BtoA',
                            'Contact', 'CoOriented', 'AntiOriented'):
                    full_key = f'{pfx}/{bk}'
                    if full_key in pair_beh:
                        vals = pair_beh[full_key][idx_arr].astype(np.float64)
                        row[f'{bk}_prop'] = float(vals.sum() / n_expected)

                # ── Engagement / disengagement proportions (frame-level) ────
                eng_key = f'{pfx}/Engaged'
                dis_key = f'{pfx}/Disengaged'
                eng_vals = pair_beh[eng_key][idx_arr] if eng_key in pair_beh \
                           else np.zeros(len(idx_arr))
                dis_vals = pair_beh[dis_key][idx_arr] if dis_key in pair_beh \
                           else np.zeros(len(idx_arr))
                row['Engaged_prop']    = float(eng_vals.sum() / n_expected)
                row['Disengaged_prop'] = float(dis_vals.sum() / n_expected)

                # ── Engagement-masked speeds: median of non-NaN values ──────
                # WARNING: the engagement speed column names are hardcoded here.
                # If behaviors.py adds new engagement speed variants (e.g. per-node
                # engagement speeds or additional onset/offset variants), those new
                # column names must also be added to this list to be included in the
                # binned export.
                for sk in ('EngageSpeed_A', 'EngageSpeed_B',
                            'DisengageSpeed_A', 'DisengageSpeed_B',
                            'EngageOnsetSpeed_A', 'EngageOnsetSpeed_B',
                            'DisengageOnsetSpeed_A', 'DisengageOnsetSpeed_B'):
                    full_sk = f'{pfx}/{sk}'
                    if full_sk in pair_beh:
                        v    = pair_beh[full_sk][idx_arr]
                        ok_v = v[np.isfinite(v)]
                        row[f'{sk}_median'] = float(np.median(ok_v)) if len(ok_v) > 0 else np.nan

                pair_rows.append(row)

    pair_025 = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame()

    # ────────────────────────────────────────────────────────────────────────
    # Engagement indices at bin resolution
    # ────────────────────────────────────────────────────────────────────────
    # Requires kin so that approach angles can be computed for bout attribution.
    eng_idx_rows = []

    if kin is not None and pair_beh:
        for pfx, (tA, tB, nA, nB) in pair_pfx_map.items():
            eng_key = f'{pfx}/Engaged'
            if eng_key not in pair_beh:
                continue

            eng_arr = pair_beh[eng_key]   # (n_frames,) int8

            # Build binary bin-level engaged sequence.
            # The 50% threshold is a deliberate conservative choice:
            # a bin is classified as "engaged" only if MORE THAN HALF of its frames
            # meet the engagement criteria.  For a 6-frame bin (at 24 fps / 0.25 s),
            # this means at least 4 out of 6 frames must be engaged — a single
            # engaged frame does not trigger an engagement bout at the bin level.
            # This prevents brief glances or noise spikes from inflating the
            # engagement count.  Adjust the > 0.5 threshold here if a more or less
            # conservative criterion is needed.
            engaged_bin = np.zeros(n_bins, dtype=np.int8)
            for bi, (_, s_idxs) in enumerate(bin_list):
                idx_a = np.array(s_idxs, dtype=int)
                prop  = float(eng_arr[idx_a].sum()) / n_expected
                engaged_bin[bi] = 1 if prop > 0.5 else 0

            # Detect bouts and attribute initiator/disengager at bin resolution
            bouts = _bouts_with_initiator_binned(
                engaged_bin, bin_list,
                tracks, kin, body_idx, tA, tB, fps
            )

            # Map bin sequence index → actual bin start time (seconds)
            # so we can filter bouts into per-minute and cumulative windows.
            bin_start_times = np.array([b * bin_size_s for b, _ in bin_list])

            # ── Full-video indices ──────────────────────────────────────────
            idx_full = _indices_from_bouts(bouts, fps=bin_fps, retreat_window_s=3.0)
            eng_idx_rows.append({
                'Pair': f'{nA}_vs_{nB}', 'Track_A': nA, 'Track_B': nB,
                'Window_Type': 'full_video', 'Window': 'full',
                **idx_full,
            })

            # ── Per-minute and cumulative windows ──────────────────────────
            if len(bin_start_times) > 0:
                max_min = int(bin_start_times.max() / 60.0) + 1
            else:
                max_min = 0

            for m in range(1, max_min + 1):
                t_lo = (m - 1) * 60.0
                t_hi = m * 60.0

                # Filter bouts by their bout-start bin's video time
                bouts_per = [b for b in bouts
                             if t_lo <= bin_start_times[b['start']] < t_hi]
                bouts_cum = [b for b in bouts
                             if bin_start_times[b['start']] < t_hi]

                if bouts_per:
                    idx_per = _indices_from_bouts(bouts_per, fps=bin_fps, retreat_window_s=3.0)
                    eng_idx_rows.append({
                        'Pair': f'{nA}_vs_{nB}', 'Track_A': nA, 'Track_B': nB,
                        'Window_Type': 'per_minute', 'Window': f'min_{m}',
                        **idx_per,
                    })

                if bouts_cum:
                    idx_cum = _indices_from_bouts(bouts_cum, fps=bin_fps, retreat_window_s=3.0)
                    eng_idx_rows.append({
                        'Pair': f'{nA}_vs_{nB}', 'Track_A': nA, 'Track_B': nB,
                        'Window_Type': 'cumulative', 'Window': f'0-{m}min',
                        **idx_cum,
                    })

    eng_idx_025 = pd.DataFrame(eng_idx_rows) if eng_idx_rows else pd.DataFrame()

    return animal_025, pair_025, eng_idx_025


# ─────────────────────────────────────────────────────────────────────────────
# 1-second bin derivation from 0.25 s bins
# ─────────────────────────────────────────────────────────────────────────────

def _1s_agg_rule(col_name: str) -> str:
    """
    Determine how to aggregate a 0.25 s column when building 1 s bins.

    Each 1 s bin re-aggregates four consecutive 0.25 s sub-bins.
    The rule is based purely on the column suffix so no schema lookup is needed.

    Returns one of:
        'cmean'  — circular mean of the four sub-bin circular means
                   (for *_cmean angular columns)
        'mean'   — arithmetic mean of the four sub-bin values
                   (for *_mean, *_median, *_p90, *_absmedian, *_absp90,
                    *_bin covariance/correlation columns)
        'skip'   — omit from 1 s output
                   (for *_prop binary proportions and engagement columns)
    """
    n = col_name.lower()

    # Binary proportions and raw engagement — excluded at 1 s resolution
    if n.endswith('_prop'):
        return 'skip'

    # Circular mean columns require the circular aggregation path
    if n.endswith('_cmean'):
        return 'cmean'

    # Everything else (mean, median, p90, abs*, bin covariances) → arithmetic mean
    return 'mean'


def build_1s_from_025(animal_025: pd.DataFrame,
                       pair_025: pd.DataFrame,
                       bin_size_s: float = 0.25):
    """
    Derive 1-second binned DataFrames by re-aggregating four 0.25 s sub-bins.

    Binary proportions (columns ending in '_prop') and raw engagement columns
    are excluded from the 1 s output: at 1 s resolution these require
    re-detection rather than averaging, which is beyond the scope of this
    derived table.

    For all remaining columns:
        *_cmean (angular circular means)  → circular mean of 4 sub-bins
        everything else (means, medians,  → arithmetic mean of 4 sub-bins
        percentiles, covariances, …)

    Parameters
    ----------
    animal_025  : pd.DataFrame from build_025s_bins
    pair_025    : pd.DataFrame from build_025s_bins
    bin_size_s  : float — must match the value used to build the 0.25 s bins

    Returns
    -------
    animal_1s : pd.DataFrame — BinTime_1s(s), Track, {feat}_* (no binary props)
    pair_1s   : pd.DataFrame — BinTime_1s(s), Track_A, Track_B, {feat}_*
    """
    # Note: _derive is designed specifically for bin_size_s = 0.25, giving
    # bins_per_s = 4 (exactly four 0.25 s sub-bins per 1 s bin).  Using other
    # bin sizes will misalign 1 s boundaries unless bin_size_s divides 1.0 evenly
    # (e.g. 0.5 s → 2 sub-bins, 0.1 s → 10 sub-bins are also safe).  Non-divisors
    # such as 0.3 s would cause fractional bins_per_s and incorrect bin grouping.
    assert abs(1.0 % bin_size_s) < 1e-9, (
        f"bin_size_s={bin_size_s} does not divide 1.0 evenly; "
        "1 s aggregation would misalign bin boundaries.")
    bins_per_s = round(1.0 / bin_size_s)   # 4 for 0.25 s bins

    def _derive(df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
        """Inner helper: aggregate one DataFrame from sub-bin to 1 s."""
        if df.empty:
            return df.copy()

        df = df.copy()
        # Assign each 0.25 s bin to the parent 1 s bin index
        df['_1s_bin'] = (df['BinIdx'] // bins_per_s).astype(int)
        # Compute 1 s bin start time for the output column
        df['BinTime_1s(s)'] = df['_1s_bin'].astype(float)

        # Columns to group by (identity fields + internal bin counter)
        group_keys = ['_1s_bin'] + [c for c in id_cols if c in df.columns
                                     and c not in ('BinIdx', 'BinTime(s)')]

        # Separate feature columns from identity columns
        skip_from_feat = set(id_cols) | {'BinIdx', 'BinTime(s)',
                                          '_1s_bin', 'BinTime_1s(s)'}
        feat_cols = [c for c in df.columns if c not in skip_from_feat]

        # Classify each feature column
        cmean_cols = [c for c in feat_cols if _1s_agg_rule(c) == 'cmean']
        mean_cols  = [c for c in feat_cols if _1s_agg_rule(c) == 'mean']
        # 'skip' columns are simply never added to the output

        rows = []
        for keys, chunk in df.groupby(group_keys, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict = dict(zip(group_keys, keys))
            row['BinTime_1s(s)'] = float(keys[0])   # first key is _1s_bin index

            # Arithmetic mean: covers position means, distance mean/median,
            # velocity median/p90, acceleration, jerk, shape, covariance, …
            for col in mean_cols:
                vals = chunk[col].to_numpy(dtype=np.float64)
                ok   = vals[np.isfinite(vals)]
                row[col] = float(np.mean(ok)) if len(ok) > 0 else np.nan

            # Circular mean: applied to sub-bin circular means (*_cmean columns).
            # These are already angle values in [−180°, 180°] so they can be
            # passed directly into _circular_mean_deg.
            for col in cmean_cols:
                vals = chunk[col].to_numpy(dtype=np.float64)
                row[col] = _circular_mean_deg(vals)

            rows.append(row)

        result = pd.DataFrame(rows).drop(columns=['_1s_bin'], errors='ignore')
        return result

    animal_1s = _derive(animal_025, id_cols=['BinIdx', 'BinTime(s)', 'Track'])
    pair_1s   = _derive(pair_025,   id_cols=['BinIdx', 'BinTime(s)', 'Track_A', 'Track_B'])
    return animal_1s, pair_1s


# ─────────────────────────────────────────────────────────────────────────────
# Top-level export entry point
# ─────────────────────────────────────────────────────────────────────────────

def write_binned_xlsx(track_arrays: dict,
                      pair_arrays: dict,
                      single_beh: dict,
                      pair_beh: dict,
                      tracks: np.ndarray,
                      node_names: list,
                      track_names: list,
                      fps: float,
                      frame_map: dict,
                      kin: dict,
                      output_path: str | Path) -> None:
    """
    Build all binned DataFrames and write them to a single Excel workbook.

    Sheets written:
        "Animal 0.25s"       — per-track features at 0.25 s bin resolution
        "Pair 0.25s"         — pairwise features at 0.25 s bin resolution
        "Eng Indices 0.25s"  — engagement / reciprocity / retreat indices
                               computed from bin-level bouts
        "Animal 1s"          — per-track features at 1 s resolution
                               (re-aggregated from 0.25 s, binary props omitted)
        "Pair 1s"            — pairwise features at 1 s resolution

    Threading / UI note:
        This function runs SYNCHRONOUSLY on the calling thread.  In run_popup.py it
        is called from _run_export() which runs inside a QThread worker, so the Qt
        main thread is not blocked.  However, if this function were ever called
        directly on the main thread (e.g. for a quick export shortcut), the
        build_025s_bins() computation and the openpyxl Excel write can each take
        several seconds for large sessions or many animals, which would freeze the UI.
        Consider wrapping in a QThread worker if direct main-thread calls are added.

    Parameters
    ----------
    track_arrays : dict[int, dict[str, np.ndarray]]  — from precompute_feature_arrays
    pair_arrays  : dict[str, np.ndarray]              — from precompute_feature_arrays
    single_beh   : dict  — from compute_single_animal
    pair_beh     : dict  — from compute_pairwise
    tracks       : (n_frames, 2, n_nodes, n_tracks)
    node_names   : list[str]
    track_names  : list[str]
    fps          : float
    frame_map    : {video_frame_idx -> sleap_data_idx}
    kin          : kinematics dict from compute_kinematics
    output_path  : destination .xlsx path (string or Path)
    """
    # Build 0.25 s bins
    animal_025, pair_025, eng_idx_025 = build_025s_bins(
        track_arrays, pair_arrays, single_beh, pair_beh,
        tracks, node_names, track_names, fps, frame_map,
        kin=kin,
    )

    # Derive 1 s bins from the 0.25 s bins
    animal_1s, pair_1s = build_1s_from_025(animal_025, pair_025)

    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        animal_025.to_excel(writer, sheet_name='Animal 0.25s', index=False)

        if not pair_025.empty:
            pair_025.to_excel(writer, sheet_name='Pair 0.25s', index=False)

        if not eng_idx_025.empty:
            eng_idx_025.to_excel(writer, sheet_name='Eng Indices 0.25s', index=False)

        animal_1s.to_excel(writer, sheet_name='Animal 1s', index=False)

        if not pair_1s.empty:
            pair_1s.to_excel(writer, sheet_name='Pair 1s', index=False)
