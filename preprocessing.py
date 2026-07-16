import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def _mask_2d_speed_outliers(coords, fps, n_sigma=5.0):
    """Two-pass outlier masking on (n_frames, n_nodes, 2) coords.

    Pass 1 — bilateral 2D speed check: iteratively flags isolated single-frame
    teleportations.  A frame is only flagged when the speed *into* it AND the
    speed *out of* it both exceed the robust threshold, preventing valid fast
    movements from being discarded.

    Pass 2 — rolling-median deviation check: flags multi-frame identity swaps.
    Uses a ~2 s median-filter baseline and a 15 px floor to avoid over-masking
    slow-moving animals.

    Both passes set flagged frames to NaN (x and y) so the gap-filler sees
    them as missing observations rather than trusted position anchors.

    Parameters
    ----------
    coords : (n_frames, n_nodes, 2) float array — raw SLEAP observations
    fps    : float
    n_sigma: float — robust-sigma multiplier (default 5.0)

    Returns
    -------
    (n_frames, n_nodes, 2) float32 array with outlier frames set to NaN
    """
    coords = coords.copy().astype(np.float64)
    n_frames, n_nodes, _ = coords.shape
    win = max(3, int(round(fps * 2.0)))   # ~2 s rolling window
    if win % 2 == 0:
        win += 1

    for node in range(n_nodes):
        xy = coords[:, node, :]   # (n_frames, 2)  — view into coords copy

        # ── Pass 1: bilateral 2D speed check ────────────────────────────────
        changed = True
        while changed:
            changed = False
            finite_mask = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            finite_idx = np.where(finite_mask)[0]
            if len(finite_idx) < 3:
                break

            dxy = np.diff(xy[finite_idx], axis=0)               # (m-1, 2)
            dt_frames = np.diff(finite_idx).astype(np.float64)  # (m-1,)
            speed = np.hypot(dxy[:, 0], dxy[:, 1]) / dt_frames  # px/frame

            med = np.median(speed)
            mad = np.median(np.abs(speed - med))
            threshold = med + n_sigma * mad * 1.4826   # 1.4826 = 1/Φ⁻¹(0.75)

            newly_flagged = set()
            for i in range(len(finite_idx) - 1):
                if speed[i] > threshold and i + 1 < len(speed):
                    if speed[i + 1] > threshold:
                        newly_flagged.add(finite_idx[i + 1])

            if newly_flagged:
                for fi in newly_flagged:
                    xy[fi, :] = np.nan
                changed = True

        # ── Pass 2: rolling-median deviation check ───────────────────────────
        finite_mask = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        if finite_mask.sum() < 3:
            coords[:, node, :] = xy
            continue

        # Cheap linear fill just to build a smooth baseline
        x_ax = np.arange(n_frames, dtype=np.float64)
        xy_filled = xy.copy()
        for ax in range(2):
            col = xy[:, ax]
            fin = np.isfinite(col)
            if fin.sum() >= 2:
                xy_filled[:, ax] = np.interp(x_ax, x_ax[fin], col[fin])
            else:
                xy_filled[:, ax] = np.nanmean(col) if fin.any() else 0.0

        baseline_x = median_filter(xy_filled[:, 0], size=win)
        baseline_y = median_filter(xy_filled[:, 1], size=win)

        # Deviation of original finite observations from the baseline
        fin_orig = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        dev = np.full(n_frames, np.nan)
        dev[fin_orig] = np.hypot(
            xy[fin_orig, 0] - baseline_x[fin_orig],
            xy[fin_orig, 1] - baseline_y[fin_orig],
        )

        dev_finite = dev[np.isfinite(dev)]
        if len(dev_finite) >= 3:
            med_dev = np.median(dev_finite)
            mad_dev = np.median(np.abs(dev_finite - med_dev))
            thr_dev = max(med_dev + n_sigma * mad_dev * 1.4826, 15.0)
            flag = fin_orig & (dev > thr_dev)
            xy[flag, :] = np.nan

        coords[:, node, :] = xy

    return coords.astype(np.float32)


def _kalman_fill_gap(trace, gap_start, gap_end, fps):
    """Run Kalman RTS smoother on a local window around a NaN gap.

    Hand-rolled numpy forward Kalman filter + RTS backward smoother.
    State model: position+velocity, F=[[1,1],[0,1]], H=[[1,0]],
    Q=eye(2)*1e-3, R=1e-2, P0=eye(2).  Numerically equivalent to pykalman.
    """
    context = max(int(fps * 1.5), (gap_end - gap_start + 1) * 2)
    win_s = max(0, gap_start - context)
    win_e = min(len(trace), gap_end + 1 + context)
    window = trace[win_s:win_e].astype(np.float64)

    finite_vals = window[np.isfinite(window)]
    if len(finite_vals) < 3:
        return

    n = len(window)

    # State-model constants (2-D state: position, velocity)
    F  = np.array([[1.0, 1.0], [0.0, 1.0]])
    FT = F.T
    H  = np.array([[1.0, 0.0]])
    HT = H.T
    Q  = np.eye(2) * 1e-3
    R  = 1e-2           # scalar observation noise
    I2 = np.eye(2)

    # Initial state (match pykalman default: P0 = eye(2))
    m = np.array([float(finite_vals[0]), 0.0])
    P = np.eye(2)

    # Storage for forward pass
    ms_pred = np.empty((n, 2))
    Ps_pred = np.empty((n, 2, 2))
    ms_filt = np.empty((n, 2))
    Ps_filt = np.empty((n, 2, 2))

    # ── Forward Kalman filter ──────────────────────────────────────────────
    for k in range(n):
        m_pred = F @ m
        P_pred = F @ P @ FT + Q

        ms_pred[k] = m_pred
        Ps_pred[k] = P_pred

        obs = window[k]
        if np.isfinite(obs):                       # observed frame
            innov = obs - (H @ m_pred)[0]
            S     = (H @ P_pred @ HT)[0, 0] + R   # scalar innovation cov
            K     = (P_pred @ HT) / S              # (2,1) Kalman gain
            m = m_pred + K[:, 0] * innov
            P = (I2 - K @ H) @ P_pred
        else:                                      # missing frame — predict only
            m = m_pred
            P = P_pred

        ms_filt[k] = m
        Ps_filt[k] = P

    # ── RTS backward smoother ─────────────────────────────────────────────
    ms_smooth = np.empty((n, 2))
    Ps_smooth = np.empty((n, 2, 2))
    ms_smooth[-1] = ms_filt[-1]
    Ps_smooth[-1] = Ps_filt[-1]

    for k in range(n - 2, -1, -1):
        M   = Ps_pred[k + 1]                        # 2×2, always PSD → invertible
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        Mi  = np.array([[ M[1, 1], -M[0, 1]],
                        [-M[1, 0],  M[0, 0]]]) / det  # closed-form 2×2 inverse
        G = Ps_filt[k] @ FT @ Mi
        ms_smooth[k] = ms_filt[k] + G @ (ms_smooth[k + 1] - ms_pred[k + 1])
        Ps_smooth[k] = Ps_filt[k] + G @ (Ps_smooth[k + 1] - Ps_pred[k + 1]) @ G.T

    smoothed_pos = ms_smooth[:, 0]

    for i in range(gap_start, gap_end + 1):
        if np.isnan(trace[i]):
            trace[i] = smoothed_pos[i - win_s]


def hybrid_convergent_fill(trace, fps=24, pchip_time_s=0.25):
    trace = np.asarray(trace, dtype=np.float32)

    if np.all(np.isnan(trace)):
        return np.zeros_like(trace)

    if np.sum(np.isfinite(trace)) < 3:
        val = np.nanmean(trace)
        return np.full_like(trace, val if np.isfinite(val) else 0.0)

    filled = trace.copy()
    n = len(trace)
    x = np.arange(n)
    isnan = np.isnan(trace)
    pchip_limit = max(2, int(round(fps * pchip_time_s)))

    nan_idx = np.where(isnan)[0]
    gaps = np.split(nan_idx, np.where(np.diff(nan_idx) > 1)[0] + 1) if len(nan_idx) else []

    finite_mask = np.isfinite(filled)
    pchip = (PchipInterpolator(x[finite_mask], filled[finite_mask], extrapolate=False)
             if finite_mask.sum() >= 2 else None)

    for gap in gaps:
        start, end = gap[0], gap[-1]
        gap_len = end - start + 1

        if gap_len > pchip_limit:
            continue

        if start == 0 or end >= n - 1:
            continue

        if pchip is not None:
            filled[start:end + 1] = pchip(x[start:end + 1])

    remaining_nan = np.isnan(filled)
    if np.any(remaining_nan):
        nan_idx2 = np.where(remaining_nan)[0]
        gaps2 = np.split(nan_idx2, np.where(np.diff(nan_idx2) > 1)[0] + 1)
        for gap in gaps2:
            g_start, g_end = int(gap[0]), int(gap[-1])
            if g_start == 0 or g_end >= n - 1:
                continue   # edge gaps: let constant edge-fill below handle them
            _kalman_fill_gap(filled, g_start, g_end, fps)

    if np.any(np.isnan(filled)):
        mask = np.isfinite(filled)

        if np.sum(mask) >= 2:
            filled[np.isnan(filled)] = np.interp(
                x[np.isnan(filled)],
                x[mask],
                filled[mask],
            )

        if np.isnan(filled[0]):
            first_valid = np.flatnonzero(np.isfinite(filled))[0]
            filled[:first_valid] = filled[first_valid]

        if np.isnan(filled[-1]):
            last_valid = np.flatnonzero(np.isfinite(filled))[-1]
            filled[last_valid + 1:] = filled[last_valid]

    return filled


def smooth_sleap_allnodes(coords, med_win=3, sg_win=5, poly=3):
    coords = np.asarray(coords, dtype=float)
    smoothed = np.copy(coords)

    n_frames, n_nodes, _ = coords.shape

    if sg_win % 2 == 0:
        sg_win += 1

    if sg_win > n_frames:
        sg_win = n_frames if n_frames % 2 == 1 else n_frames - 1

    for axis in range(2):
        for node_idx in range(n_nodes):
            d = smoothed[:, node_idx, axis]

            if med_win > 1:
                d = median_filter(d, size=med_win)

            if sg_win >= 3 and len(d) >= sg_win:
                d = savgol_filter(d, sg_win, poly)

            smoothed[:, node_idx, axis] = d

    return smoothed


def _smooth_body_axis_heading(front, rear, fps, sg_win, sg_poly):
    """Body-axis heading from rear→front anatomical landmarks.

    Parameters
    ----------
    front, rear : (n_frames, 2) arrays — x, y positions of front/rear landmarks
    fps         : float
    sg_win      : int   Savitzky-Golay window (odd)
    sg_poly     : int   polynomial order

    Returns
    -------
    (n_frames,) heading in degrees [-180, 180]
    """
    dx = front[:, 0] - rear[:, 0]
    dy = front[:, 1] - rear[:, 1]
    raw_deg = np.degrees(np.arctan2(dy, dx))

    # Mark degenerate frames (front≈rear, distance < 1 px) for interpolation
    dist = np.hypot(dx, dy)
    bad = dist < 1.0

    if np.all(bad):
        return np.zeros(len(front), dtype=np.float64)

    # Convert to sin/cos, interpolate bad frames, convert back
    sin_h = np.sin(np.radians(raw_deg))
    cos_h = np.cos(np.radians(raw_deg))
    if np.any(bad):
        good = ~bad
        x_ax = np.arange(len(raw_deg))
        sin_h[bad] = np.interp(x_ax[bad], x_ax[good], sin_h[good])
        cos_h[bad] = np.interp(x_ax[bad], x_ax[good], cos_h[good])
    filled_deg = np.degrees(np.arctan2(sin_h, cos_h))

    # Unwrap → smooth → rewrap
    unwrapped = np.unwrap(np.radians(filled_deg))
    n = len(unwrapped)
    w = sg_win
    if w % 2 == 0:
        w += 1
    w = max(w, sg_poly + 2)
    if w % 2 == 0:
        w += 1
    max_w = n if n % 2 == 1 else n - 1
    w = min(w, max_w)
    p = min(sg_poly, w - 1)
    if w >= 3 and n >= w:
        smoothed = savgol_filter(unwrapped, w, p)
    else:
        smoothed = unwrapped
    result = np.degrees(smoothed)
    result = (result + 180.0) % 360.0 - 180.0
    return result.astype(np.float64)


def _compute_body_heading(tracks, fps, sg_win, sg_poly, node_names):
    """Compute body-axis heading for all tracks.

    Fallback chain for front/rear landmark pairs:
      1. body → nose
      2. hip_mid → ear_mid
      3. hip_mid → nose
      4. velocity heading (last resort)

    Parameters
    ----------
    tracks     : (n_frames, 2, n_nodes, n_tracks)
    fps        : float
    sg_win     : int   — base SG window (widened internally for heading)
    sg_poly    : int
    node_names : list[str] or None

    Returns
    -------
    (n_frames, n_tracks) heading in degrees [-180, 180]
    """
    from behaviors import find_node_idx

    # Body-axis heading needs a wider smoothing window than velocity derivatives
    # because arctan2 of landmark pairs amplifies per-pixel tracking jitter.
    # Use ~375 ms at 24 fps (9 frames) as minimum, or 3× the base window.
    hdg_win = max(sg_win * 3, 9)
    if hdg_win % 2 == 0:
        hdg_win += 1

    n_frames, _, n_nodes, n_tracks = tracks.shape
    result = np.zeros((n_frames, n_tracks), dtype=np.float64)

    # Resolve node indices
    if node_names is not None:
        body_idx = find_node_idx(node_names, 'body')
        nose_idx = find_node_idx(node_names, 'nose')
        ear_l_idx = find_node_idx(node_names, 'ear_l')
        ear_r_idx = find_node_idx(node_names, 'ear_r')
        hip_l_idx = find_node_idx(node_names, 'hip_l')
        hip_r_idx = find_node_idx(node_names, 'hip_r')
    else:
        body_idx = nose_idx = ear_l_idx = ear_r_idx = hip_l_idx = hip_r_idx = None

    for t in range(n_tracks):
        front = rear = None

        # Chain 1: body → nose
        if body_idx is not None and nose_idx is not None:
            rear = np.stack([tracks[:, 0, body_idx, t],
                             tracks[:, 1, body_idx, t]], axis=1)
            front = np.stack([tracks[:, 0, nose_idx, t],
                              tracks[:, 1, nose_idx, t]], axis=1)

        # Chain 2: hip_mid → ear_mid
        if front is None and (hip_l_idx is not None and hip_r_idx is not None
                              and ear_l_idx is not None and ear_r_idx is not None):
            rear = np.stack([
                (tracks[:, 0, hip_l_idx, t] + tracks[:, 0, hip_r_idx, t]) / 2.0,
                (tracks[:, 1, hip_l_idx, t] + tracks[:, 1, hip_r_idx, t]) / 2.0,
            ], axis=1)
            front = np.stack([
                (tracks[:, 0, ear_l_idx, t] + tracks[:, 0, ear_r_idx, t]) / 2.0,
                (tracks[:, 1, ear_l_idx, t] + tracks[:, 1, ear_r_idx, t]) / 2.0,
            ], axis=1)

        # Chain 3: hip_mid → nose
        if front is None and (hip_l_idx is not None and hip_r_idx is not None
                              and nose_idx is not None):
            rear = np.stack([
                (tracks[:, 0, hip_l_idx, t] + tracks[:, 0, hip_r_idx, t]) / 2.0,
                (tracks[:, 1, hip_l_idx, t] + tracks[:, 1, hip_r_idx, t]) / 2.0,
            ], axis=1)
            front = np.stack([tracks[:, 0, nose_idx, t],
                              tracks[:, 1, nose_idx, t]], axis=1)

        if front is not None and rear is not None:
            result[:, t] = _smooth_body_axis_heading(front, rear, fps, hdg_win, sg_poly)
        else:
            # Fallback: velocity heading from body-center (or mean of all nodes)
            if body_idx is not None:
                vx = np.gradient(tracks[:, 0, body_idx, t])
                vy = np.gradient(tracks[:, 1, body_idx, t])
            else:
                vx = np.gradient(np.nanmean(tracks[:, 0, :, t], axis=1))
                vy = np.gradient(np.nanmean(tracks[:, 1, :, t], axis=1))
            result[:, t] = np.degrees(np.arctan2(vy, vx))

    return result


def compute_kinematics(tracks, fps, sg_win=3, sg_poly=3, node_names=None):
    """
    Compute per-frame kinematics for all tracks/nodes via Savitzky-Golay differentiation.

    Parameters
    ----------
    tracks     : np.ndarray  shape (n_frames, 2, n_nodes, n_tracks)  — already filled/smoothed
    fps        : float
    sg_win     : int   window length (odd; must satisfy window > deriv and >= poly+1)
    sg_poly    : int   polynomial order (>= 3 to support jerk; default 3)
    node_names : list[str] or None — needed for body-axis heading computation

    Returns
    -------
    dict of np.ndarrays:
        Per-node (n_frames, n_nodes, n_tracks):
            vx, vy, speed, heading_deg, ax, ay, accel, jx, jy, jerk
        Per-track (n_frames, n_tracks):
            body_heading_deg — body-axis heading from rear→front landmarks
    """
    tracks = np.asarray(tracks, dtype=np.float32)
    n_frames, _, n_nodes, n_tracks = tracks.shape
    dt = 1.0 / fps

    # Enforce constraints: odd, >= poly+1, <= n_frames (odd)
    if sg_win % 2 == 0:
        sg_win += 1
    sg_win = max(sg_win, sg_poly + 2)
    if sg_win % 2 == 0:
        sg_win += 1
    max_win = n_frames if n_frames % 2 == 1 else n_frames - 1
    sg_win = min(sg_win, max_win)
    sg_poly = min(sg_poly, sg_win - 1)

    shape = (n_frames, n_nodes, n_tracks)
    _x = tracks[:, 0, :, :]   # (n_frames, n_nodes, n_tracks)
    _y = tracks[:, 1, :, :]
    vx = savgol_filter(_x, sg_win, sg_poly, deriv=1, delta=dt, axis=0).astype(np.float32)
    vy = savgol_filter(_y, sg_win, sg_poly, deriv=1, delta=dt, axis=0).astype(np.float32)
    ax = savgol_filter(_x, sg_win, sg_poly, deriv=2, delta=dt, axis=0).astype(np.float32)
    ay = savgol_filter(_y, sg_win, sg_poly, deriv=2, delta=dt, axis=0).astype(np.float32)
    jx = np.zeros(shape, dtype=np.float32)
    jy = np.zeros(shape, dtype=np.float32)
    if sg_poly >= 3:
        jx = savgol_filter(_x, sg_win, sg_poly, deriv=3, delta=dt, axis=0).astype(np.float32)
        jy = savgol_filter(_y, sg_win, sg_poly, deriv=3, delta=dt, axis=0).astype(np.float32)

    speed   = np.hypot(vx, vy)
    heading = np.degrees(np.arctan2(vy, vx))
    accel   = np.hypot(ax, ay)
    jerk    = np.hypot(jx, jy)

    body_heading = _compute_body_heading(tracks, fps, sg_win, sg_poly, node_names)

    return {
        "vx": vx, "vy": vy, "speed": speed, "heading_deg": heading,
        "ax": ax, "ay": ay, "accel": accel,
        "jx": jx, "jy": jy, "jerk": jerk,
        "body_heading_deg": body_heading,
    }


def _process_one_track(track_data, fps, med_win, sg_win, poly):
    """Fill and smooth one track.  track_data: (n_frames, n_nodes, 2) float32."""
    coords = track_data.copy()
    coords = _mask_2d_speed_outliers(coords, fps=fps)
    n_nodes = coords.shape[1]
    for node_idx in range(n_nodes):
        for axis_idx in range(2):
            coords[:, node_idx, axis_idx] = hybrid_convergent_fill(
                coords[:, node_idx, axis_idx], fps=fps
            )
    return smooth_sleap_allnodes(coords, med_win=med_win, sg_win=sg_win, poly=poly)


def fill_and_smooth_tracks(tracks, fps, med_win=3, sg_win=5, poly=2, progress_callback=None):
    """
    Input:
        tracks shape: (n_frames, 2, n_nodes, n_tracks)

    Output:
        processed tracks with same shape.
    """
    tracks = np.asarray(tracks, dtype=np.float32)
    processed = np.copy(tracks)

    n_frames, n_axes, n_nodes, n_tracks = processed.shape

    if n_axes != 2:
        raise ValueError(f"Expected axis dimension of size 2, got {n_axes}")

    # Parallelize per-track when the problem is large enough to amortise
    # loky process-pool overhead (~5-50 ms per worker spawn).
    use_parallel = (n_tracks * n_nodes * 2 > 16) and (n_frames > 5000)

    if use_parallel:
        from joblib import Parallel, delayed

        track_inputs = [
            processed[:, :, :, t].transpose(0, 2, 1)   # (n_frames, n_nodes, 2)
            for t in range(n_tracks)
        ]

        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(_process_one_track)(td, fps, med_win, sg_win, poly)
            for td in track_inputs
        )

        for t, coords in enumerate(results):
            processed[:, :, :, t] = coords.transpose(0, 2, 1)
            if progress_callback:
                progress_callback(int((t + 1) * 100 / n_tracks))
    else:
        total_steps = n_tracks * n_nodes * 2
        step = 0

        for track_idx in range(n_tracks):
            # Convert one track to (n_frames, n_nodes, 2)
            coords = processed[:, :, :, track_idx].transpose(0, 2, 1)
            coords = _mask_2d_speed_outliers(coords, fps=fps)

            for node_idx in range(n_nodes):
                for axis_idx in range(2):
                    coords[:, node_idx, axis_idx] = hybrid_convergent_fill(
                        coords[:, node_idx, axis_idx],
                        fps=fps,
                    )
                    step += 1
                    if progress_callback:
                        progress_callback(int(step * 100 / total_steps))

            coords = smooth_sleap_allnodes(
                coords,
                med_win=med_win,
                sg_win=sg_win,
                poly=poly,
            )

            processed[:, :, :, track_idx] = coords.transpose(0, 2, 1)

    return processed