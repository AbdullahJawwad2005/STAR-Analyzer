import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from pykalman import KalmanFilter


def _kalman_fill_gap(trace, gap_start, gap_end, fps):
    """Run Kalman smoother on a local window around a NaN gap, writing back only to NaN positions.

    This is a constant-velocity 2-state Kalman model (state = [position, velocity]).
    It is used here because NaN gaps in SLEAP pose estimation are typically caused by
    brief occlusions of the animal — the animal does not stop moving during occlusion.
    A Kalman smoother with a constant-velocity prior respects the inertia of movement
    (i.e. the animal likely kept going in roughly the same direction at roughly the same
    speed) better than pure spline or linear interpolation, which have no physical prior
    about trajectory continuity.
    """
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
    # pykalman requires a masked array so it knows which observations are
    # missing (NaN).  Passing a plain ndarray with NaN values causes the
    # smoother to treat them as valid zero-ish observations and return
    # all-NaN smoothed means, causing long gaps to fall through to the
    # linear-interpolation fallback without ever using the Kalman smoother.
    smoothed_means, _ = kf.smooth(np.ma.masked_invalid(window))

    for i in range(gap_start, gap_end + 1):
        if np.isnan(trace[i]):
            trace[i] = smoothed_means[i - win_s, 0]


def hybrid_convergent_fill(trace, fps=24, pchip_time_s=0.25):
    """Fill NaN gaps in a single 1-D coordinate trace using a three-tier strategy.

    The tiers are applied in order; each tier handles gaps the previous tier left unfilled:

    Tier 1 — Short gaps (length <= fps * 0.25 s, default ~6 frames at 24 fps):
        PCHIP spline interpolation (Piecewise Cubic Hermite Interpolating Polynomial).
        PCHIP is shape-preserving and respects the gradient (slope) at the gap boundaries,
        so it produces smooth, physically plausible trajectories without the oscillation
        risk of a standard cubic spline.  It is the first choice for brief occlusions.

    Tier 2 — Longer gaps (those skipped by Tier 1):
        Kalman smoother on a local context window around the gap (_kalman_fill_gap).
        For gaps too long for reliable spline extrapolation, the constant-velocity Kalman
        prior gives a more defensible estimate than a polynomial that may overshoot.

    Tier 3 — Last-resort fallback (any NaNs still remaining after Tier 1 + 2):
        Linear interpolation (np.interp) between the nearest finite neighbours, followed
        by constant-pad from the first / last valid value to fill leading/trailing NaNs.
        This is purely a safety net for edge cases (e.g. NaNs at the very start or end
        of the trace where PCHIP and Kalman have no context).
    """
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


def smooth_sleap_allnodes(coords, med_win=3, sg_win=5, poly=2):
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
                # Median filter first to remove spike noise (single-frame outliers).
                # This step is critical because Savitzky-Golay (below) assumes
                # additive Gaussian noise — its polynomial fit is a least-squares
                # estimator that is highly sensitive to outliers.  A single large
                # spike corrupts the fit across the entire SG window.  The median
                # filter neutralises spikes before SG sees the data.
                d = median_filter(d, size=med_win)

            if sg_win >= 3 and len(d) >= sg_win:
                d = savgol_filter(d, sg_win, poly)

            smoothed[:, node_idx, axis] = d

    return smoothed


def compute_kinematics(tracks, fps, sg_win=11, sg_poly=3):
    """
    Compute per-frame kinematics for all tracks/nodes via Savitzky-Golay differentiation.

    Parameters
    ----------
    tracks  : np.ndarray  shape (n_frames, 2, n_nodes, n_tracks)  — already filled/smoothed
    fps     : float
    sg_win  : int   window length (odd; must satisfy window > deriv and >= poly+1)
    sg_poly : int   polynomial order (>= 3 to support jerk; default 3)

    Returns
    -------
    dict of np.ndarrays, each shape (n_frames, n_nodes, n_tracks):
        vx, vy          — velocity components  (px/s)
        speed           — speed magnitude       (px/s)
        heading_deg     — movement heading via arctan2(vy, vx)  (degrees, -180..180)
        ax, ay          — acceleration components  (px/s²)
        accel           — acceleration magnitude
        jx, jy          — jerk components      (px/s³)
        jerk            — jerk magnitude
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
            # Savitzky-Golay with the deriv= argument is used for differentiation
            # rather than finite differences (e.g. np.gradient).  SG fits a local
            # polynomial to the data window and then analytically differentiates
            # that polynomial — so the derivative is exact with respect to the fit
            # and does NOT amplify high-frequency noise.  By contrast, np.gradient
            # is essentially a finite-difference operator: dividing adjacent position
            # differences by dt magnifies any residual noise by 1/dt, which is large
            # at video frame rates (e.g. 1/0.042 ≈ 24 at 24 fps).  The SG approach
            # gives smooth, physically meaningful velocity and acceleration estimates
            # without a separate smoothing pass after differentiation.
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

    return {
        "vx": vx, "vy": vy, "speed": speed, "heading_deg": heading,
        "ax": ax, "ay": ay, "accel": accel,
        "jx": jx, "jy": jy, "jerk": jerk,
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
        # Axis transposition explanation:
        # The app-wide array layout is (n_frames, 2, n_nodes) — axis 1 is x/y, axis 2 is node index.
        # The fill and smooth helpers (hybrid_convergent_fill, smooth_sleap_allnodes) expect
        # (n_frames, n_nodes, 2) — axis 1 is node index, axis 2 is x/y.
        # transpose(0, 2, 1) swaps axes 1 and 2, converting between the two conventions.
        # The inverse transpose (also 0, 2, 1) is applied after processing to restore the
        # original layout before writing back into `processed`.
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