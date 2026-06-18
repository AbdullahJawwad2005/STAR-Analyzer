import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from pykalman import KalmanFilter


def _kalman_fill_gap(trace, gap_start, gap_end, fps):
    """Run Kalman smoother on a local window around a NaN gap, writing back only to NaN positions."""
    context = max(int(fps * 1.5), (gap_end - gap_start + 1) * 2)
    win_s = max(0, gap_start - context)
    win_e = min(len(trace), gap_end + 1 + context)
    window = trace[win_s:win_e].copy()

    finite_vals = window[np.isfinite(window)]
    if len(finite_vals) < 3:
        return

    kf = KalmanFilter(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        transition_covariance=np.eye(2) * 1e-3,
        observation_covariance=np.eye(1) * 1e-2,
        initial_state_mean=[float(finite_vals[0]), 0],
    )
    smoothed_means, _ = kf.smooth(window)

    for i in range(gap_start, gap_end + 1):
        if np.isnan(trace[i]):
            trace[i] = smoothed_means[i - win_s, 0]


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

    for gap in gaps:
        start, end = gap[0], gap[-1]
        gap_len = end - start + 1

        if gap_len > pchip_limit:
            continue

        left_idx = start - 1 if start > 0 else None
        right_idx = end + 1 if end < n - 1 else None

        if left_idx is None or right_idx is None:
            continue

        x_known = x[np.isfinite(filled)]
        y_known = filled[np.isfinite(filled)]

        interp = PchipInterpolator(x_known, y_known, extrapolate=False)
        filled[start:end + 1] = interp(x[start:end + 1])

    remaining_nan = np.isnan(filled)
    if np.any(remaining_nan):
        nan_idx2 = np.where(remaining_nan)[0]
        gaps2 = np.split(nan_idx2, np.where(np.diff(nan_idx2) > 1)[0] + 1)
        for gap in gaps2:
            _kalman_fill_gap(filled, int(gap[0]), int(gap[-1]), fps)

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
    vx = np.zeros(shape, dtype=np.float32)
    vy = np.zeros(shape, dtype=np.float32)
    ax = np.zeros(shape, dtype=np.float32)
    ay = np.zeros(shape, dtype=np.float32)
    jx = np.zeros(shape, dtype=np.float32)
    jy = np.zeros(shape, dtype=np.float32)

    for t in range(n_tracks):
        for n in range(n_nodes):
            x = tracks[:, 0, n, t]
            y = tracks[:, 1, n, t]
            vx[:, n, t] = savgol_filter(x, sg_win, sg_poly, deriv=1, delta=dt)
            vy[:, n, t] = savgol_filter(y, sg_win, sg_poly, deriv=1, delta=dt)
            ax[:, n, t] = savgol_filter(x, sg_win, sg_poly, deriv=2, delta=dt)
            ay[:, n, t] = savgol_filter(y, sg_win, sg_poly, deriv=2, delta=dt)
            if sg_poly >= 3:
                jx[:, n, t] = savgol_filter(x, sg_win, sg_poly, deriv=3, delta=dt)
                jy[:, n, t] = savgol_filter(y, sg_win, sg_poly, deriv=3, delta=dt)

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

    total_steps = n_tracks * n_nodes * 2
    step = 0

    for track_idx in range(n_tracks):
        # Convert one track to old format: (frames, nodes, 2)
        coords = processed[:, :, :, track_idx].transpose(0, 2, 1)

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

        # Convert back to current app format: (frames, 2, nodes)
        processed[:, :, :, track_idx] = coords.transpose(0, 2, 1)

    return processed