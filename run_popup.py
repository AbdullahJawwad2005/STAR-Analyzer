import gc
import threading
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer, QDateTime, QThread, QObject
import pandas as pd
from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QSlider,
    QLabel,
    QSpinBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QCheckBox,
    QScrollArea,
    QLineEdit,
    QListWidget,
    QTextEdit,
    QStyle,
)
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPixmap, QPainter, QPen, QBrush, QPainterPath
from pathlib import Path
from preprocessing import fill_and_smooth_tracks, compute_kinematics
from behaviors import (compute_single_animal, compute_pairwise, compute_behavior_summary,
                       compute_second_order, _SECOND_ORDER_KEYS, find_node_idx)
from features import (build_feature_dataframes, precompute_feature_arrays,
                      build_key_metrics_df, build_proximity_orientation_df,
                      _tailend_node_idxs, _detect_bouts, _general_min_dist,
                      PROX_THRESHOLD_CM, CONTACT_THRESHOLD_CM)
from binned_export import write_binned_xlsx
from roi_view import ROIView
try:
    from graph_export import write_graphs as _write_graphs
    _GRAPHS_AVAILABLE = True
except Exception:          # matplotlib not installed or import error
    _GRAPHS_AVAILABLE = False

try:
    from rf_analysis import run_full_rf_pipeline as _run_rf
    _RF_AVAILABLE = True
except Exception:
    _RF_AVAILABLE = False

# Limit simultaneous exports to prevent OOM when multiple windows are open
_EXPORT_SEMAPHORE = threading.Semaphore(2)


class _ScrollableMessageBox(QDialog):
    """Resizable, scrollable message dialog replacing bare QMessageBox for long content."""

    def __init__(self, parent, title, message, *, critical=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(450, 250)
        if critical:
            self.resize(550, 350)
        else:
            self.resize(480, 280)

        layout = QVBoxLayout(self)

        # Icon + text area in a horizontal layout
        top = QHBoxLayout()
        icon_label = QLabel()
        style = self.style()
        sp = QStyle.StandardPixmap.SP_MessageBoxCritical if critical else QStyle.StandardPixmap.SP_MessageBoxInformation
        icon_label.setPixmap(style.standardIcon(sp).pixmap(48, 48))
        icon_label.setAlignment(Qt.AlignTop)
        top.addWidget(icon_label)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(message)
        if critical:
            text_edit.setFont(QFont("Consolas", 9))
        top.addWidget(text_edit, 1)
        layout.addLayout(top)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)


def _make_icon(kind: str) -> QIcon:
    """Create a simple 32x32 programmatic icon for window title bars."""
    px = QPixmap(32, 32)
    px.fill(QColor(0, 0, 0, 0))
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)

    if kind == "roi":
        # Crosshair / frame selector — cyan square with corner marks
        pen = QPen(QColor(0, 200, 224), 2)
        p.setPen(pen)
        # Outer frame
        p.drawRect(4, 4, 24, 24)
        # Corner marks (thicker)
        pen.setWidth(3)
        p.setPen(pen)
        for cx, cy in ((4, 4), (28, 4), (4, 28), (28, 28)):
            dx = 6 if cx == 4 else -6
            dy = 6 if cy == 4 else -6
            p.drawLine(cx, cy, cx + dx, cy)
            p.drawLine(cx, cy, cx, cy + dy)

    elif kind == "inspector":
        # Magnifying glass — data inspection
        pen = QPen(QColor(0, 212, 240), 2)
        p.setPen(pen)
        p.drawEllipse(6, 4, 16, 16)
        pen.setWidth(3)
        p.setPen(pen)
        p.drawLine(20, 18, 27, 27)
        # Small lines inside (data rows)
        pen = QPen(QColor(0, 212, 240), 1)
        p.setPen(pen)
        p.drawLine(10, 9, 18, 9)
        p.drawLine(10, 13, 16, 13)

    elif kind == "export":
        # Download/export arrow into a tray
        pen = QPen(QColor(100, 200, 120), 2)
        p.setPen(pen)
        p.setBrush(QBrush(QColor(100, 200, 120)))
        # Arrow pointing down
        path = QPainterPath()
        path.moveTo(16, 4)
        path.lineTo(16, 18)
        p.drawPath(path)
        # Arrowhead
        path2 = QPainterPath()
        path2.moveTo(10, 15)
        path2.lineTo(16, 22)
        path2.lineTo(22, 15)
        p.drawPath(path2)
        # Tray
        p.setBrush(QBrush(QColor(0, 0, 0, 0)))
        p.drawLine(4, 24, 4, 28)
        p.drawLine(4, 28, 28, 28)
        p.drawLine(28, 28, 28, 24)

    p.end()
    return QIcon(px)


def _run_analysis(processed_data, fps, roi, px_per_cm, strip_cm, proc_opts) -> dict:
    """Pure analysis computation — safe to call off the main thread."""
    data = processed_data
    tracks      = data["tracks"]
    node_names  = data["node_names"]
    track_names = data["track_names"]
    frame_map   = data["frame_map"]

    (rx0, ry0), (rx1, ry1), side = roi

    po          = proc_opts or {}
    do_single   = po.get('proc_single_beh', True)
    do_pair     = po.get('proc_pair_beh', True)
    do_features = po.get('proc_features', True)
    do_zones    = po.get('proc_zones', True)
    do_prox     = po.get('proc_proximity', True)

    # Trim first 5 seconds (hand-placement buffer)
    n_frames = tracks.shape[0]
    analysis_start = min(int(5 * fps), n_frames)
    valid_vframes = [vf for vf, si in frame_map.items() if si >= analysis_start]
    analysis_start_vidframe = min(valid_vframes) if valid_vframes else 0
    tracks = tracks[analysis_start:]
    n_frames = tracks.shape[0]
    frame_map = {vf: si - analysis_start
                 for vf, si in frame_map.items()
                 if si >= analysis_start}

    n_tracks = tracks.shape[3]

    kin = compute_kinematics(tracks, fps, node_names=node_names)

    single_beh = (compute_single_animal(tracks, kin, node_names, fps, px_per_cm=px_per_cm)
                  if do_single else {})

    pair_beh = (compute_pairwise(tracks, node_names, fps, dsr=None, kin=kin)
                if (do_pair and n_tracks >= 2) else {})

    if do_features:
        track_feat, pair_feat = precompute_feature_arrays(
            tracks, kin, node_names, fps, roi=(rx0, ry0, rx1, ry1))
    else:
        track_feat, pair_feat = {}, {}

    body_idx = find_node_idx(node_names, 'body')

    cache = dict(
        tracks=tracks, kin=kin,
        single_beh=single_beh, pair_beh=pair_beh,
        track_feat=track_feat, pair_feat=pair_feat,
        node_names=node_names, track_names=track_names,
        frame_map=frame_map, body_idx=body_idx,
        inv_ppcm=1.0 / max(px_per_cm, 1e-9),
        px_per_cm=px_per_cm,
        proc_opts=po,
        analysis_start_vidframe=analysis_start_vidframe,
    )

    # Metrics panel precomputes
    sleap_idxs_arr = np.array(sorted(frame_map.values()))
    si_to_pos = {int(si): pos for pos, si in enumerate(sleap_idxs_arr)}

    ROLL_W = max(3, round(0.2 * fps))
    speed_rolling = {}
    for t in range(n_tracks):
        spd = (kin['speed'][:, body_idx, t] if body_idx is not None
               else np.nanmean(kin['speed'][:, :, t], axis=1))
        speed_rolling[t] = (pd.Series(spd)
                              .rolling(ROLL_W, center=True, min_periods=1)
                              .mean().to_numpy())

    if do_zones:
        strip_px = strip_cm * px_per_cm
        zone_label = {}
        for t in range(n_tracks):
            ni = body_idx if body_idx is not None else 0
            x = tracks[:, 0, ni, t]
            y = tracks[:, 1, ni, t]
            near_h = np.minimum(x - rx0, rx1 - x) < strip_px
            near_v = np.minimum(y - ry0, ry1 - y) < strip_px
            lbl = np.full(n_frames, 'Center', dtype=object)
            lbl[near_h ^ near_v] = 'Perimeter'
            lbl[near_h & near_v] = 'Corner'
            zone_label[t] = lbl
    else:
        zone_label = {}

    prox_px = PROX_THRESHOLD_CM * px_per_cm
    cont_px = CONTACT_THRESHOLD_CM * px_per_cm
    if do_prox and n_tracks >= 2 and len(sleap_idxs_arr) > 0:
        tailend_excl = _tailend_node_idxs(node_names)
        gen_dist_tracked, gen_closest_tracked = _general_min_dist(
            tracks, sleap_idxs_arr, 0, 1, node_names, exclude_idxs=tailend_excl)
        prox_bouts   = _detect_bouts(gen_dist_tracked <= prox_px, fps)
        cont_bouts   = _detect_bouts(gen_dist_tracked <= cont_px, fps)
        prox_cumtime = np.cumsum((gen_dist_tracked <= prox_px).astype(float)) / fps
        cont_cumtime = np.cumsum((gen_dist_tracked <= cont_px).astype(float)) / fps
        hdg_diff = np.abs(kin['body_heading_deg'][:, 0]
                          - kin['body_heading_deg'][:, 1]) % 360
        hdg_diff = np.minimum(hdg_diff, 360 - hdg_diff)
    else:
        gen_dist_tracked = gen_closest_tracked = None
        prox_bouts = cont_bouts = []
        prox_cumtime = cont_cumtime = hdg_diff = None

    cache.update(dict(
        n_tracks=n_tracks, si_to_pos=si_to_pos,
        speed_rolling=speed_rolling, zone_label=zone_label,
        gen_dist_tracked=gen_dist_tracked, gen_closest_tracked=gen_closest_tracked,
        prox_bouts=prox_bouts, cont_bouts=cont_bouts,
        prox_cumtime=prox_cumtime, cont_cumtime=cont_cumtime,
        hdg_diff=hdg_diff, prox_px=prox_px, cont_px=cont_px,
    ))
    return cache


class _AnalysisWorker(QObject):
    done  = Signal(object)   # completed cache dict
    error = Signal(str)

    def __init__(self, processed_data, fps, roi, px_per_cm, strip_cm, proc_opts):
        super().__init__()
        self._processed_data = processed_data
        self._fps            = fps
        self._roi            = roi
        self._px_per_cm      = px_per_cm
        self._strip_cm       = strip_cm
        self._proc_opts      = proc_opts or {}

    def run(self):
        try:
            self.done.emit(_run_analysis(
                self._processed_data, self._fps,
                self._roi, self._px_per_cm, self._strip_cm,
                self._proc_opts,
            ))
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


class _ProcessWorker(QObject):
    progress = Signal(int)    # 0-100
    finished = Signal(object) # the completed processed_data dict
    error    = Signal(str)

    def __init__(self, sleap_data, fps):
        super().__init__()
        self._sleap_data = sleap_data
        self._fps = fps

    def run(self):
        try:
            processed = dict(self._sleap_data)
            processed["tracks"] = fill_and_smooth_tracks(
                self._sleap_data["tracks"],
                fps=self._fps,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(processed)
        except Exception as exc:
            self.error.emit(str(exc))


class _ExportWorker(QObject):
    """Background worker that runs the full export pipeline off the main thread.

    All heavy computation (kinematics, behavior detection, feature extraction,
    binned aggregation) and file I/O happen here so the Qt event loop stays
    responsive.  The file-save dialog must be shown on the main thread BEFORE
    creating this worker; the chosen path is passed in at construction time.

    Signals
    -------
    finished(str) — emitted with the success message when all files are written
    error(str)    — emitted with the exception message if anything fails
    status(str)   — optional progress status label updates (not wired to progress bar)
    """
    finished = Signal(str)
    error    = Signal(str)
    status   = Signal(str)

    def __init__(self, analysis_cache, fps, roi, arena_cm, strip_cm, path,
                 arena_snapshot=None, export_opts=None):
        """
        Parameters
        ----------
        analysis_cache : dict   — from _precompute_analysis (already trimmed, kinematics computed)
        fps            : float  — video frame rate
        roi            : tuple  — ((rx0,ry0),(rx1,ry1), side) from view.roi_native()
        arena_cm       : int    — arena size in cm (from spin box)
        strip_cm       : int    — border strip width in cm
        path           : str    — destination .xlsx (or .csv) file path chosen by user
        arena_snapshot : ndarray or None — RGB image of arena cropped to ROI
        export_opts    : dict[str,bool] or None — from ExportOptionsDialog.options(); None = all
        """
        super().__init__()
        self._cache     = analysis_cache
        self._fps       = fps
        self._roi       = roi
        self._arena_cm  = arena_cm
        self._strip_cm  = strip_cm
        self._path      = path
        self._arena_snapshot = arena_snapshot
        self._opts      = export_opts or {}

    def _want(self, key: str) -> bool:
        """Return True if export option *key* is enabled (default True if not set)."""
        return self._opts.get(key, True)

    def run(self):
        """Execute the full export pipeline; emit finished or error when done."""
        try:
            (rx0, ry0), (rx1, ry1), side = self._roi
            arena_cm  = self._arena_cm
            strip_cm  = self._strip_cm
            px_per_cm = side / arena_cm
            strip     = strip_cm * px_per_cm
            inv_ppcm  = 1.0 / px_per_cm
            path      = self._path

            # ---- Pull from cache (already trimmed by _precompute_analysis) ----
            tracks       = self._cache['tracks']
            frame_map    = self._cache['frame_map']
            node_names   = self._cache['node_names']
            track_names  = self._cache['track_names']
            kin          = self._cache['kin']
            single_beh   = self._cache.get('single_beh', {})
            pair_beh     = dict(self._cache.get('pair_beh', {}))   # copy; will add second_order
            track_arrays = self._cache.get('track_feat', {})
            pair_arrays  = self._cache.get('pair_feat', {})
            zone_label   = self._cache.get('zone_label')
            po           = self._cache.get('proc_opts', {})

            # ---- 2nd-order compound social behaviors (export-only) ----
            if pair_beh and po.get('proc_pair_beh', True):
                self.status.emit("Detecting 2nd-order behaviors…")
                second_order = compute_second_order(
                    tracks, kin, node_names, self._fps, single_beh, pair_beh)
                pair_beh.update(second_order)

            # ---- Behavior summary ----
            beh_summary, engagement_idx_df = compute_behavior_summary(
                single_beh, pair_beh, track_names, self._fps,
                tracks=tracks, kin=kin, frame_map=frame_map, node_names=node_names)

            # ---- Feature DataFrames (reuse precomputed arrays from cache) ----
            if track_arrays and po.get('proc_features', True):
                self.status.emit("Building feature dataframes…")
                animal_feat_df, pair_feat_df = build_feature_dataframes(
                    tracks, kin, node_names, track_names, self._fps,
                    frame_map, roi=(rx0, ry0, rx1, ry1),
                    _precomputed=(track_arrays, pair_arrays),
                )
            else:
                animal_feat_df = pair_feat_df = pd.DataFrame()

            # ---- Build main data table ----
            self.status.emit("Building data table…")
            sorted_frames = sorted(frame_map.items())
            vid_frames_arr = np.array([vf for vf, _ in sorted_frames])
            sleap_idxs_arr = np.array([si for _, si in sorted_frames])
            vid_frame_0 = vid_frames_arr[0] if len(vid_frames_arr) else 0
            n_mapped = len(vid_frames_arr)
            n_t = tracks.shape[3]
            n_n = len(node_names)

            # Time array
            time_arr = np.round((vid_frames_arr - vid_frame_0) / self._fps, 4)

            # Build columns as flat arrays (n_mapped * n_t * n_n)
            total_rows = n_mapped * n_t * n_n
            col_frame = np.repeat(vid_frames_arr, n_t * n_n)
            col_time  = np.repeat(time_arr, n_t * n_n)

            # Track names repeated
            track_tile = np.tile(np.repeat(np.arange(n_t), n_n), n_mapped)
            col_track = np.array(track_names)[track_tile]

            # Node names tiled
            node_tile = np.tile(np.arange(n_n), n_mapped * n_t)
            col_node = np.array(node_names)[node_tile]

            # Sleap indices repeated for indexing
            sleap_rep = np.repeat(sleap_idxs_arr, n_t * n_n)

            # Extract x, y for all (sleap_idx, node, track) combinations
            x_all = tracks[sleap_rep, 0, node_tile, track_tile]
            y_all = tracks[sleap_rep, 1, node_tile, track_tile]

            # Zone classification (vectorized)
            nan_mask = np.isnan(x_all) | np.isnan(y_all)
            col_zone = np.full(total_rows, 'Undetected', dtype=object)
            valid = ~nan_mask
            if np.any(valid):
                col_zone[valid] = _classify_zone_vec(
                    x_all[valid], y_all[valid], rx0, ry0, rx1, ry1, strip)

            # Coordinate conversions
            x_cm = np.where(nan_mask, np.nan, np.round((x_all - rx0) * inv_ppcm, 3))
            y_cm = np.where(nan_mask, np.nan, np.round((y_all - ry0) * inv_ppcm, 3))
            x_px = np.where(nan_mask, np.nan, np.round(x_all, 2))
            y_px = np.where(nan_mask, np.nan, np.round(y_all, 2))

            # Kinematics columns
            col_vx    = np.round(kin["vx"][sleap_rep, node_tile, track_tile] * inv_ppcm, 3)
            col_vy    = np.round(kin["vy"][sleap_rep, node_tile, track_tile] * inv_ppcm, 3)
            col_speed = np.round(kin["speed"][sleap_rep, node_tile, track_tile] * inv_ppcm, 3)
            col_hdg   = np.round(kin["heading_deg"][sleap_rep, node_tile, track_tile], 2)
            col_accel = np.round(kin["accel"][sleap_rep, node_tile, track_tile] * inv_ppcm, 3)
            col_jerk  = np.round(kin["jerk"][sleap_rep, node_tile, track_tile] * inv_ppcm, 3)

            df = pd.DataFrame({
                "Frame":         col_frame,
                "Time (s)":      col_time,
                "Track":         col_track,
                "Body Part":     col_node,
                "X (px)":        x_px,
                "Y (px)":        y_px,
                "X (cm)":        x_cm,
                "Y (cm)":        y_cm,
                "Zone":          col_zone,
                "Vx (cm/s)":     col_vx,
                "Vy (cm/s)":     col_vy,
                "Speed (cm/s)":  col_speed,
                "Heading (deg)": col_hdg,
                "Accel (cm/s²)": col_accel,
                "Jerk (cm/s³)":  col_jerk,
            })

            # ---- Build behavior dataframes (one row per video frame) ----
            single_keys = ('stationary', 'walking', 'running', 'turning', 'dir_reversal')

            # Pre-build pair prefix → track-name column prefix mapping
            pair_col_map = {}
            for key in pair_beh:
                if '/' not in key:
                    continue
                pfx, beh = key.rsplit('/', 1)
                if pfx not in pair_col_map:
                    parts = pfx.split('_')
                    try:
                        tA = int(parts[0][1:])
                        tB = int(parts[1][1:])
                        nA = track_names[tA] if tA < len(track_names) else f't{tA}'
                        nB = track_names[tB] if tB < len(track_names) else f't{tB}'
                        pair_col_map[pfx] = f'{nA}_vs_{nB}'
                    except (ValueError, IndexError):
                        pair_col_map[pfx] = pfx

            # Separate 1st-order and 2nd-order pair behavior keys
            _so_key_set = set(_SECOND_ORDER_KEYS)
            _valid_pair = {k: v for k, v in pair_beh.items() if '/' in k}
            first_order_pair_keys = {k: v for k, v in _valid_pair.items()
                                     if k.rsplit('/', 1)[1] not in _so_key_set}
            second_order_pair_keys = {k: v for k, v in _valid_pair.items()
                                      if k.rsplit('/', 1)[1] in _so_key_set}

            # 1st Order Behaviors sheet
            beh_col = {
                'Frame':   vid_frames_arr,
                'Time(s)': np.round(vid_frames_arr / self._fps, 4),
            }
            for bname in single_keys:
                for t, tname in enumerate(track_names):
                    beh_col[f'{tname}/{bname}'] = single_beh[bname][sleap_idxs_arr, t].astype(int)
            for key, arr in first_order_pair_keys.items():
                pfx, beh = key.rsplit('/', 1)
                col_name = f'{pair_col_map[pfx]}/{beh}'
                vals = arr[sleap_idxs_arr]
                beh_col[col_name] = vals
            beh_df = pd.DataFrame(beh_col)

            # 2nd Order Behaviors sheet
            beh2_col = {
                'Frame':   vid_frames_arr,
                'Time(s)': np.round(vid_frames_arr / self._fps, 4),
            }
            for key, arr in second_order_pair_keys.items():
                pfx, beh = key.rsplit('/', 1)
                col_name = f'{pair_col_map[pfx]}/{beh}'
                beh2_col[col_name] = arr[sleap_idxs_arr]
            beh2_df = pd.DataFrame(beh2_col)

            binned_path = None    # set inside xlsx branch; used in success message
            graph_paths = []      # set inside xlsx branch; used in success message
            any_main = False      # set inside xlsx branch; used in success message

            if path.endswith(".csv"):
                df.to_csv(path, index=False)
                any_main = True   # CSV always writes
            else:
                # ---- Zone summary (unique frames per track/zone, NaN/Undetected excluded) ----
                # Filter to one node per animal per frame (body-centre preferred; first node
                # as fallback) so a single frame is never counted toward multiple zones due
                # to different body parts landing in different zones simultaneously.
                _zone_idx = find_node_idx(node_names, 'body')
                _zone_node = node_names[_zone_idx] if _zone_idx is not None else node_names[0]

                detected = df[(df["Zone"] != "Undetected") & (df["Body Part"] == _zone_node)]
                summary = (
                    detected.groupby(["Track", "Zone"], sort=False)
                    .size()
                    .reset_index(name="Frame Count")
                )
                total_frames = len(frame_map)
                summary["Time in Zone (s)"] = (summary["Frame Count"] / self._fps).round(2)
                summary["% of Session"]     = (100 * summary["Frame Count"] / total_frames).round(1)
                # Sort zones in a logical order
                zone_order = ["C1","C2","C3","C4","W1","W2","W3","W4","Open"]
                summary["_z"] = summary["Zone"].map({z: i for i, z in enumerate(zone_order)}).fillna(99)
                summary = summary.sort_values(["Track", "_z"]).drop(columns="_z").reset_index(drop=True)

                # ---- Session info ----
                info_rows = [
                    ("Export date",         QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")),
                    ("", ""),
                    ("--- ROI & Calibration ---", ""),
                    ("ROI top-left (px)",    f"({rx0}, {ry0})"),
                    ("ROI bottom-right (px)",f"({rx1}, {ry1})"),
                    ("ROI width (px)",       side),
                    ("Arena size (cm)",      arena_cm),
                    ("Scale (px/cm)",        round(px_per_cm, 4)),
                    ("Zone border strip (cm)", strip_cm),
                    ("", ""),
                    ("--- Session Stats ---", ""),
                    ("Total tracked frames", total_frames),
                    ("Video FPS",            round(self._fps, 4)),
                    ("Tracks",               ", ".join(track_names)),
                    ("Body parts",           ", ".join(node_names)),
                    ("Total data rows",      len(df)),
                ]
                info_df = pd.DataFrame(info_rows, columns=["Parameter", "Value"])

                # ---- Write main xlsx (only if at least one main sheet selected) ----
                any_main = any(self._want(k) for k, _ in ExportOptionsDialog._MAIN_SHEETS
                               if k != "main_key_metrics")
                if any_main:
                    self.status.emit("Writing Excel workbook…")
                    try:
                        with pd.ExcelWriter(path, engine="openpyxl") as writer:
                            if self._want("main_tracking_data"):
                                df.to_excel(writer,              sheet_name="Tracking Data",        index=False)
                            if self._want("main_zone_summary"):
                                summary.to_excel(writer,         sheet_name="Zone Summary",         index=False)
                            if self._want("main_session_info"):
                                info_df.to_excel(writer,         sheet_name="Session Info",         index=False)
                            if self._want("main_1st_order_behaviors"):
                                beh_df.to_excel(writer,          sheet_name="1st Order Behaviors",  index=False)
                            if self._want("main_2nd_order_behaviors") and not beh2_df.empty:
                                beh2_df.to_excel(writer,     sheet_name="2nd Order Behaviors", index=False)
                            if self._want("main_behavior_summary"):
                                beh_summary.to_excel(writer,     sheet_name="Behavior Summary",     index=False)
                            if self._want("main_engagement_indices") and not engagement_idx_df.empty:
                                engagement_idx_df.to_excel(writer, sheet_name="Engagement Indices", index=False)
                            if self._want("main_animal_features"):
                                animal_feat_df.to_excel(writer,  sheet_name="Animal Features",      index=False)
                            if self._want("main_pair_features") and not pair_feat_df.empty:
                                pair_feat_df.to_excel(writer, sheet_name="Pair Features",       index=False)
                    except PermissionError:
                        raise PermissionError(
                            f"Cannot write '{Path(path).name}' — close it in Excel and try again."
                        )
                # ---- Key Metrics separate file (*_key_metrics.xlsx) ----
                if self._want("main_key_metrics"):
                    key_metrics_df = build_key_metrics_df(
                        tracks, kin, single_beh, pair_beh,
                        track_arrays, pair_arrays, frame_map,
                        summary, node_names, track_names,
                        self._fps, px_per_cm,
                        zone_label=zone_label,
                    )
                    _p = Path(path)
                    km_path = str(_p.with_name(_p.stem + "_key_metrics" + _p.suffix))
                    self.status.emit("Writing key metrics file…")
                    try:
                        with pd.ExcelWriter(km_path, engine="openpyxl") as km_writer:
                            key_metrics_df.to_excel(km_writer, sheet_name="Key Metrics", index=False)
                            if tracks.shape[3] >= 2:
                                prox_df = build_proximity_orientation_df(
                                    tracks, kin, frame_map, node_names, track_names,
                                    self._fps, px_per_cm,
                                )
                                prox_df.to_excel(km_writer, sheet_name="Proximity & Orientation",
                                                 index=False)
                    except PermissionError:
                        raise PermissionError(
                            f"Cannot write '{Path(km_path).name}' — close it in Excel and try again."
                        )

                # ---- Binned export (*_binned.xlsx, same directory) ----
                any_binned = any(self._want(k) for k, _ in ExportOptionsDialog._BINNED_SHEETS)
                if any_binned:
                    self.status.emit("Writing binned export…")
                    _p = Path(path)
                    binned_path = str(_p.with_name(_p.stem + "_binned" + _p.suffix))
                    try:
                        write_binned_xlsx(
                            track_arrays, pair_arrays, single_beh, pair_beh,
                            tracks, node_names, track_names, self._fps, frame_map,
                            kin=kin, output_path=binned_path,
                            include_sheets=self._opts,
                        )
                    except PermissionError:
                        raise PermissionError(
                            f"Cannot write '{Path(binned_path).name}' — close it in Excel and try again."
                        )

                # ---- Graph export (PDF files, same directory) ----
                any_graph = any(self._want(k) for k, _ in ExportOptionsDialog._GRAPH_PDFS)
                graph_paths = []
                if _GRAPHS_AVAILABLE and any_graph:
                    try:
                        _p = Path(path)
                        base_for_graphs = str(_p.with_suffix(""))
                        graph_paths = _write_graphs(
                            zone_summary_df=summary,
                            track_arrays=track_arrays,
                            pair_arrays=pair_arrays,
                            frame_map=frame_map,
                            track_names=track_names,
                            node_names=node_names,
                            fps=self._fps,
                            base_path=base_for_graphs,
                            status_cb=self.status.emit,
                            arena_cm=self._arena_cm,
                            strip_cm=self._strip_cm,
                            px_per_cm=px_per_cm,
                            arena_snapshot=self._arena_snapshot,
                            graph_opts=self._opts,
                        )
                    except Exception as _ge:
                        import traceback as _tb
                        self.status.emit(f"Graph export error (skipped): {_ge}\n" + _tb.format_exc())

            # ---- RF analysis (CSV + optional PDF) ----
            rf_paths = []
            if _RF_AVAILABLE and self._want("rf_analysis"):
                try:
                    _p = Path(path)
                    base_for_rf = str(_p.with_suffix(""))
                    from rf_analysis import run_full_rf_pipeline
                    rf_paths = run_full_rf_pipeline(
                        track_arrays, pair_arrays, pair_beh, track_names,
                        self._fps, base_for_rf,
                        write_plots=self._want("rf_analysis_plots"),
                        status_cb=self.status.emit,
                    )
                except Exception as _re:
                    import traceback as _tb
                    self.status.emit(f"RF analysis error (skipped): {_re}\n" + _tb.format_exc())

            parts = []
            if any_main:
                parts.append(f"Main workbook:\n{path}")
            if binned_path:
                parts.append(f"Binned export:\n{binned_path}")
            if graph_paths:
                parts.append(f"Graphs ({len(graph_paths)} PDF files):\n"
                             + "\n".join(f"  • {Path(p).name}" for p in graph_paths))
            if rf_paths:
                parts.append(f"RF Analysis ({len(rf_paths)} files):\n"
                             + "\n".join(f"  • {Path(p).name}" for p in rf_paths))
            msg = "Export complete.\n\n" + "\n\n".join(parts)
            del tracks, kin, single_beh, pair_beh, track_arrays, pair_arrays
            self.finished.emit(msg)

        except Exception as exc:
            import traceback
            self.error.emit(traceback.format_exc())


class ProcessingOptionsDialog(QDialog):
    """Choose which analysis modules to run before processing."""

    def __init__(self, n_tracks=1, default_opts=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Options")
        self.setWindowIcon(_make_icon("export"))
        self.setMinimumWidth(340)

        po = default_opts or {}
        layout = QVBoxLayout(self)
        self._checks: dict[str, QCheckBox] = {}

        # Always-on group
        grp_core = QGroupBox("Core (always computed)")
        vb = QVBoxLayout(grp_core)
        cb_kin = QCheckBox("Kinematics  (speed / accel / jerk / heading)")
        cb_kin.setChecked(True); cb_kin.setEnabled(False)
        vb.addWidget(cb_kin)
        layout.addWidget(grp_core)

        # Single-animal group
        grp_single = QGroupBox("Single-Animal Analysis")
        vb2 = QVBoxLayout(grp_single)
        self._add_check(vb2, 'proc_single_beh',
                        "Single-animal behaviors  (stationary, locomotion, turning)",
                        po.get('proc_single_beh', True))
        self._add_check(vb2, 'proc_features',
                        "Feature arrays  (shape, path efficiency, entropy, curvature)",
                        po.get('proc_features', True))
        self._add_check(vb2, 'proc_zones',
                        "Zone analysis  (center / perimeter / corner)",
                        po.get('proc_zones', True))
        layout.addWidget(grp_single)

        # Pair group (disabled if single track)
        grp_pair = QGroupBox("Pair / Social Analysis")
        vb3 = QVBoxLayout(grp_pair)
        self._add_check(vb3, 'proc_pair_beh',
                        "Pair behaviors  (proximity subtypes, approach, following)",
                        po.get('proc_pair_beh', True))
        self._add_check(vb3, 'proc_proximity',
                        "Proximity tracking  (inter-animal distance, bouts)",
                        po.get('proc_proximity', True))
        if n_tracks < 2:
            grp_pair.setEnabled(False)
            grp_pair.setToolTip("Requires \u2265 2 tracked animals")
        layout.addWidget(grp_pair)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Analyze")
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _add_check(self, layout, key, label, checked):
        cb = QCheckBox(label); cb.setChecked(checked)
        self._checks[key] = cb; layout.addWidget(cb)

    def _accept(self):
        if not any(cb.isChecked() for cb in self._checks.values()):
            QMessageBox.warning(self, "Nothing selected",
                                "Select at least one analysis module.")
            return
        self.accept()

    def options(self) -> dict:
        return {'proc_kinematics': True,
                **{k: cb.isChecked() for k, cb in self._checks.items()}}


class ExportOptionsDialog(QDialog):
    """Modal dialog letting the user pick which export outputs to generate."""

    # (key, label) tuples for each group
    _MAIN_SHEETS = [
        ("main_tracking_data",       "Tracking Data"),
        ("main_zone_summary",        "Zone Summary"),
        ("main_session_info",        "Session Info"),
        ("main_1st_order_behaviors", "1st Order Behaviors"),
        ("main_2nd_order_behaviors", "2nd Order Behaviors"),
        ("main_behavior_summary",    "Behavior Summary"),
        ("main_engagement_indices",  "Engagement Indices"),
        ("main_animal_features",     "Animal Features"),
        ("main_pair_features",       "Pair Features"),
        ("main_key_metrics",         "Key Metrics"),
    ]
    _BINNED_SHEETS = [
        ("binned_animal_025",      "Animal 0.25s"),
        ("binned_pair_025",        "Pair 0.25s"),
        ("binned_eng_indices_025", "Engagement Indices 0.25s"),
        ("binned_animal_1s",       "Animal 1s"),
        ("binned_pair_1s",         "Pair 1s"),
    ]
    _GRAPH_PDFS = [
        ("graph_heatmaps",       "Heatmaps"),
        ("graph_cascade",        "Cascade (speed / accel / jerk)"),
        ("graph_distance",       "Distance"),
        ("graph_oncoplot",             "Feature Oncoplot"),
        ("graph_sync_oncoplot",        "Synchrony Oncoplot"),
        ("graph_oncoplot_clean",       "Feature Oncoplot (clean)"),
        ("graph_sync_oncoplot_clean",  "Synchrony Oncoplot (clean)"),
        ("graph_dist_features",        "Feature vs Distance (cumulative)"),
    ]
    _RF_ANALYSIS = [
        ("rf_analysis",       "Random Forest Bout Analysis"),
        ("rf_analysis_plots", "RF Analysis Plots (PDF)"),
    ]

    _GRAPH_SUFFIXES = {
        "graph_heatmaps":           "_graphs_heatmaps.pdf",
        "graph_cascade":            "_graphs_cascade.pdf",
        "graph_distance":           "_graphs_distance.pdf",
        "graph_oncoplot":           "_graphs_oncoplot.pdf",
        "graph_sync_oncoplot":      "_graphs_sync_oncoplot.pdf",
        "graph_oncoplot_clean":     "_graphs_oncoplot_clean.pdf",
        "graph_sync_oncoplot_clean":"_graphs_sync_oncoplot_clean.pdf",
        "graph_dist_features":      "_graphs_dist_features.pdf",
    }
    _RF_SUFFIXES = {
        "rf_analysis":       ["_rf_bouts.csv", "_rf_report.csv", "_rf_importance.csv"],
        "rf_analysis_plots": ["_rf_analysis.pdf"],
    }

    def __init__(self, graphs_available=True, default_dir="", default_name="export",
                 proc_opts=None, parent=None):
        super().__init__(parent)
        self._updating = False   # guard against recursive signal loops — must be set before any widget creation
        self.setWindowTitle("Export Options")
        self.setWindowIcon(_make_icon("export"))
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        # ---- Master checkbox (above scroll area) ----
        self._master = QCheckBox("Select All Outputs")
        self._master.setChecked(True)
        self._master.setTristate(True)
        self._master.stateChanged.connect(self._on_master_changed)
        layout.addWidget(self._master)

        # ---- Scrollable area for checkbox groups ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        self._checks: dict[str, QCheckBox] = {}
        self._group_all: list[QCheckBox] = []

        self._build_group(scroll_layout, "Main Excel Workbook", self._MAIN_SHEETS)
        self._build_group(scroll_layout, "Binned Excel Workbook", self._BINNED_SHEETS)
        grp = self._build_group(scroll_layout, "Graph PDFs", self._GRAPH_PDFS)
        if not graphs_available:
            grp.setEnabled(False)
            grp.setToolTip("matplotlib not available — graphs disabled")

        rf_grp = self._build_group(scroll_layout, "Analysis", self._RF_ANALYSIS)
        if not _RF_AVAILABLE:
            rf_grp.setEnabled(False)
            rf_grp.setToolTip("scikit-learn not available — RF analysis disabled")

        # ---- Gate checkboxes based on what was processed ----
        _PROC_GATES = {
            "main_1st_order_behaviors":  "proc_single_beh",
            "main_2nd_order_behaviors":  "proc_pair_beh",
            "main_behavior_summary":     "proc_single_beh",
            "main_engagement_indices":   "proc_pair_beh",
            "main_animal_features":      "proc_features",
            "main_pair_features":        "proc_features",
            "main_zone_summary":         "proc_zones",
            "binned_animal_025":         "proc_features",
            "binned_pair_025":           "proc_features",
            "binned_eng_indices_025":    "proc_pair_beh",
            "binned_animal_1s":          "proc_features",
            "binned_pair_1s":            "proc_features",
            "graph_heatmaps":            "proc_zones",
            "graph_distance":            "proc_proximity",
            "graph_oncoplot":            "proc_features",
            "graph_sync_oncoplot":       "proc_pair_beh",
            "graph_oncoplot_clean":      "proc_features",
            "graph_sync_oncoplot_clean": "proc_pair_beh",
            "graph_dist_features":       "proc_features",
        }
        if proc_opts:
            for key, proc_key in _PROC_GATES.items():
                if key in self._checks and not proc_opts.get(proc_key, True):
                    self._checks[key].setChecked(False)
                    self._checks[key].setEnabled(False)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # ---- Save Location ----
        loc_grp = QGroupBox("Save Location")
        loc_layout = QVBoxLayout(loc_grp)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Folder:"))
        self._folder_edit = QLineEdit(default_dir)
        self._folder_edit.setStyleSheet(
            "background:#2d2d2d; color:#eee; border:1px solid #444; padding:4px;")
        folder_row.addWidget(self._folder_edit, 1)
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse_btn)
        loc_layout.addLayout(folder_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Base name:"))
        self._name_edit = QLineEdit(default_name)
        self._name_edit.setStyleSheet(
            "background:#2d2d2d; color:#eee; border:1px solid #444; padding:4px;")
        name_row.addWidget(self._name_edit, 1)
        loc_layout.addLayout(name_row)

        layout.addWidget(loc_grp)

        # ---- File preview ----
        preview_grp = QGroupBox("Files to be created")
        preview_layout = QVBoxLayout(preview_grp)
        self._preview = QListWidget()
        self._preview.setStyleSheet(
            "background:#1a1a1a; color:#aaa; border:1px solid #333; font-size:11px;")
        self._preview.setSelectionMode(QListWidget.NoSelection)
        self._preview.setMaximumHeight(120)
        preview_layout.addWidget(self._preview)
        layout.addWidget(preview_grp)

        # ---- Buttons ----
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Export")
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Connect all checkboxes + name edit to preview updater
        for cb in self._checks.values():
            cb.stateChanged.connect(self._update_preview)
        self._name_edit.textChanged.connect(self._update_preview)
        self._update_preview()

    # ---- Group builder ----
    def _build_group(self, parent_layout, title, items):
        grp = QGroupBox(title)
        vbox = QVBoxLayout(grp)

        grp_all = QCheckBox("All")
        grp_all.setChecked(True)
        grp_all.setTristate(True)
        grp_all.stateChanged.connect(lambda _st, g=grp_all, it=items: self._on_group_changed(g, it))
        self._group_all.append(grp_all)
        vbox.addWidget(grp_all)

        for key, label in items:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.stateChanged.connect(lambda _st, g=grp_all, it=items: self._sync_group(g, it))
            self._checks[key] = cb
            vbox.addWidget(cb)

        parent_layout.addWidget(grp)
        return grp

    # ---- Sync helpers ----
    def _on_master_changed(self, state):
        if self._updating:
            return
        if state == Qt.PartiallyChecked:
            return
        self._updating = True
        checked = (state == Qt.Checked)
        for cb in self._checks.values():
            if cb.isEnabled():
                cb.setChecked(checked)
        for ga in self._group_all:
            if ga.isEnabled():
                ga.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self._updating = False

    def _on_group_changed(self, grp_all, items):
        if self._updating:
            return
        state = grp_all.checkState()
        if state == Qt.PartiallyChecked:
            return
        self._updating = True
        checked = (state == Qt.Checked)
        for key, _ in items:
            if self._checks[key].isEnabled():
                self._checks[key].setChecked(checked)
        self._updating = False
        self._sync_master()

    def _sync_group(self, grp_all, items):
        if self._updating:
            return
        self._updating = True
        states = [self._checks[k].isChecked() for k, _ in items]
        if all(states):
            grp_all.setCheckState(Qt.Checked)
        elif any(states):
            grp_all.setCheckState(Qt.PartiallyChecked)
        else:
            grp_all.setCheckState(Qt.Unchecked)
        self._updating = False
        self._sync_master()

    def _sync_master(self):
        all_checked = all(cb.isChecked() for cb in self._checks.values() if cb.isEnabled())
        any_checked = any(cb.isChecked() for cb in self._checks.values() if cb.isEnabled())
        self._updating = True
        if all_checked:
            self._master.setCheckState(Qt.Checked)
        elif any_checked:
            self._master.setCheckState(Qt.PartiallyChecked)
        else:
            self._master.setCheckState(Qt.Unchecked)
        self._updating = False

    # ---- Browse / preview helpers ----
    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder",
                                             self._folder_edit.text())
        if d:
            self._folder_edit.setText(d)

    def _update_preview(self, _=None):
        self._preview.clear()
        name = self._name_edit.text().strip()
        if not name:
            return
        opts = self.options()
        any_main = any(opts.get(k, False) for k, _ in self._MAIN_SHEETS)
        any_binned = any(opts.get(k, False) for k, _ in self._BINNED_SHEETS)
        if any_main:
            self._preview.addItem(f"{name}.xlsx")
        if any_binned:
            self._preview.addItem(f"{name}_binned.xlsx")
        for key, suffix in self._GRAPH_SUFFIXES.items():
            if opts.get(key, False):
                self._preview.addItem(f"{name}{suffix}")
        for key, suffixes in self._RF_SUFFIXES.items():
            if opts.get(key, False):
                for suffix in suffixes:
                    self._preview.addItem(f"{name}{suffix}")

    def export_path(self) -> str:
        """Return full base path: folder/basename (no extension)."""
        folder = self._folder_edit.text().strip()
        name = self._name_edit.text().strip()
        return str(Path(folder) / name)

    # ---- Accept / reject ----
    def _accept(self):
        if not any(cb.isChecked() for cb in self._checks.values() if cb.isEnabled()):
            QMessageBox.warning(self, "No output selected",
                                "Select at least one output to export.")
            return
        folder = self._folder_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Invalid folder",
                                "Choose a valid output folder.")
            return
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "No name",
                                "Enter a base name for the output files.")
            return
        self.accept()

    def options(self) -> dict:
        """Return {key: bool} for every export option."""
        return {k: cb.isChecked() for k, cb in self._checks.items()}


def _fmt(v):
    """Format a scalar value for the data inspector table."""
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        return f"{v:.4f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def _feat_unit(name, ipc):
    """Return (scale_factor, unit_label) for a feature shown in the inspector.

    *ipc* is 1/px_per_cm (converts px → cm).
    """
    nl = name.lower()
    # --- dimensionless / categorical ---
    if any(k in nl for k in ('ratio', 'cos_sim', 'efficiency', 'entropy',
                              'elongation', 'eccentricity', 'compactness',
                              'circularity', 'scope', 'correlation')):
        return 1.0, ''
    # angular features — no px conversion needed
    if any(k in nl for k in ('heading', 'angle', 'orient')):
        return 1.0, 'deg'
    if 'ang_mot' in nl:
        return 1.0, 'deg/frame'
    if 'curvature' in nl:
        return 1.0, 'deg/s'
    # jerk (must check before accel, since "accel" substring not in "jerk")
    if 'jerk' in nl:
        return ipc, 'cm/s\u00b3'
    # speed_accel (d(speed)/dt) — same unit as accel
    if 'speed_accel' in nl:
        return ipc, 'cm/s\u00b2'
    # acceleration
    if 'accel' in nl:
        return ipc, 'cm/s\u00b2'
    # speed / velocity
    if any(k in nl for k in ('speed', '_vx', '_vy')):
        return ipc, 'cm/s'
    # area (hourglass_area) — px² → cm²
    if 'area' in nl:
        return ipc * ipc, 'cm\u00b2'
    # covariance — px² → cm²
    if 'covariance' in nl:
        return ipc * ipc, 'cm\u00b2'
    # distance / displacement / position
    if any(k in nl for k in ('dist', 'disp', '_x', '_y')):
        return ipc, 'cm'
    return 1.0, ''


class _MetricsPanel(QWidget):
    """Live metrics panel shown below the post-processed video."""

    _STYLE = (
        "QWidget#MetricsPanel { background:#1a2428; }"
        "QLabel { color:#c0d8e0; font:11px 'Consolas','Courier New',monospace; }"
        "QLabel[role='hdr']     { color:#4a9ab0; font-weight:700; margin-top:3px; }"
        "QLabel[role='apart']   { color:#888888; }"
        "QLabel[role='prox']    { color:#f0c040; font-weight:700; }"
        "QLabel[role='contact'] { color:#ff6060; font-weight:700; }"
        "QLabel[role='bout']    { color:#50e090; font-weight:700; }"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MetricsPanel")
        self.setStyleSheet(self._STYLE)
        self.setFixedHeight(110)
        self._build()

    # ---- construction --------------------------------------------------

    def _lbl(self, text='—', role=None):
        l = QLabel(text)
        if role:
            l.setProperty('role', role)
        return l

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(14)

        # POSITION
        col0 = QVBoxLayout(); col0.setSpacing(1)
        col0.addWidget(self._lbl('POSITION', 'hdr'))
        self._lbl_time  = self._lbl(); col0.addWidget(self._lbl_time)
        self._lbl_frame = self._lbl(); col0.addWidget(self._lbl_frame)
        self._lbl_pct   = self._lbl(); col0.addWidget(self._lbl_pct)
        col0.addStretch()

        # PER MOUSE (two sub-columns)
        col1 = QVBoxLayout(); col1.setSpacing(1)
        col1.addWidget(self._lbl('PER MOUSE', 'hdr'))
        pm_row = QHBoxLayout(); pm_row.setSpacing(10)
        self._track_cols = []
        for _ in range(2):
            tc = QVBoxLayout(); tc.setSpacing(1)
            d = {'name': self._lbl(), 'spd': self._lbl(),
                 'state': self._lbl(), 'zone': self._lbl(), 'dist': self._lbl()}
            for l in d.values():
                tc.addWidget(l)
            self._track_cols.append(d)
            pm_row.addLayout(tc)
        col1.addLayout(pm_row)
        col1.addStretch()

        # INTERACTION
        col2 = QVBoxLayout(); col2.setSpacing(1)
        col2.addWidget(self._lbl('INTERACTION', 'hdr'))
        self._lbl_idist = self._lbl(); col2.addWidget(self._lbl_idist)
        self._lbl_chip  = self._lbl('Apart', 'apart'); col2.addWidget(self._lbl_chip)
        self._chip_role = 'apart'
        self._lbl_hdg   = self._lbl(); col2.addWidget(self._lbl_hdg)
        col2.addStretch()

        # BOUT STATUS
        col3 = QVBoxLayout(); col3.setSpacing(1)
        col3.addWidget(self._lbl('BOUT STATUS', 'hdr'))
        self._lbl_btype = self._lbl(); col3.addWidget(self._lbl_btype)
        self._lbl_belap = self._lbl(); col3.addWidget(self._lbl_belap)
        col3.addStretch()

        # RUNNING TALLY
        col4 = QVBoxLayout(); col4.setSpacing(1)
        col4.addWidget(self._lbl('RUNNING TALLY', 'hdr'))
        self._lbl_ptally = self._lbl(); col4.addWidget(self._lbl_ptally)
        self._lbl_ctally = self._lbl(); col4.addWidget(self._lbl_ctally)
        col4.addStretch()

        for col in (col0, col1, col2, col3, col4):
            root.addLayout(col)

    # ---- public API ----------------------------------------------------

    def clear(self):
        for l in (self._lbl_time, self._lbl_frame, self._lbl_pct,
                  self._lbl_idist, self._lbl_hdg,
                  self._lbl_btype, self._lbl_belap,
                  self._lbl_ptally, self._lbl_ctally):
            l.setText('—')
        self._set_chip('—', 'apart')
        for tc in self._track_cols:
            for l in tc.values():
                l.setText('—')

    def refresh(self, sleap_idx, video_frame_idx, n_frames, fps,
                analysis_start_vidframe, cache):
        if cache is None:
            self.clear()
            return

        inv_ppcm    = cache['inv_ppcm']
        kin         = cache['kin']
        single_beh  = cache['single_beh']
        track_names = cache['track_names']
        n_tracks    = cache.get('n_tracks', len(track_names))
        si_to_pos   = cache.get('si_to_pos', {})
        tracked_idx = si_to_pos.get(sleap_idx)   # position in frame-map-filtered arrays

        # ---- POSITION ----
        elapsed = max(0, video_frame_idx - analysis_start_vidframe) / fps
        total_s = (n_frames - 1) / fps
        self._lbl_time.setText(f"Time:  {elapsed:.2f}s / {total_s:.1f}s")
        self._lbl_frame.setText(f"Frame: {video_frame_idx} / {n_frames - 1}")
        self._lbl_pct.setText(f"Pos:   {100 * video_frame_idx / max(n_frames - 1, 1):.1f}%")

        # ---- PER MOUSE ----
        speed_rolling = cache.get('speed_rolling', {})
        zone_label    = cache.get('zone_label', {})
        for t, tc in enumerate(self._track_cols):
            if t >= n_tracks:
                for l in tc.values():
                    l.setText('—')
                continue
            tname = track_names[t] if t < len(track_names) else f't{t}'
            tc['name'].setText(f'[{tname}]')

            sr = speed_rolling.get(t)
            if sr is not None and sleap_idx < len(sr):
                tc['spd'].setText(f'Spd: {float(sr[sleap_idx]) * inv_ppcm:.1f} cm/s')
            else:
                tc['spd'].setText('Spd: —')

            state = 'Unknown'
            for sk in ('running', 'turning', 'walking', 'stationary'):
                arr = single_beh.get(sk)
                if arr is not None and sleap_idx < len(arr) and int(arr[sleap_idx, t]):
                    state = ('Immobile' if sk == 'stationary' else 'Moving')
                    break
            tc['state'].setText(f'State: {state}')

            zl = zone_label.get(t)
            zone = str(zl[sleap_idx]) if zl is not None and sleap_idx < len(zl) else '—'
            tc['zone'].setText(f'Zone: {zone}')

            fa = cache.get('track_feat', {}).get(t, {})
            dt = fa.get('cm_total_disp')
            if dt is not None and sleap_idx < len(dt):
                tc['dist'].setText(f'Dist: {float(dt[sleap_idx]) * inv_ppcm:.1f} cm')
            else:
                tc['dist'].setText('Dist: —')

        # ---- INTERACTION ----
        gen_dist = cache.get('gen_dist_tracked')
        gen_close = cache.get('gen_closest_tracked')
        if gen_dist is not None and tracked_idx is not None and tracked_idx < len(gen_dist):
            d_px = float(gen_dist[tracked_idx])
            d_cm = d_px * inv_ppcm
            pair_lbl = str(gen_close[tracked_idx]) if gen_close is not None else ''
            self._lbl_idist.setText(f'Dist: {d_cm:.2f} cm  [{pair_lbl}]')
            prox_px = cache.get('prox_px', 0)
            cont_px = cache.get('cont_px', 0)
            if d_px <= cont_px:
                self._set_chip('Contact  (<=1cm)', 'contact')
            elif d_px <= prox_px:
                self._set_chip('Proximity (<=3cm)', 'prox')
            else:
                self._set_chip('Apart', 'apart')
        else:
            self._lbl_idist.setText('Dist: —')
            self._set_chip('—', 'apart')

        hdg_diff = cache.get('hdg_diff')
        if hdg_diff is not None and sleap_idx < len(hdg_diff):
            self._lbl_hdg.setText(f'Heading \u0394: {float(hdg_diff[sleap_idx]):.0f}\u00b0')
        else:
            self._lbl_hdg.setText('Heading \u0394: —')

        # ---- BOUT STATUS ----
        prox_bouts = cache.get('prox_bouts', [])
        cont_bouts = cache.get('cont_bouts', [])
        bout_type = None; bout_start = None
        if tracked_idx is not None:
            for s, e in cont_bouts:
                if s <= tracked_idx <= e:
                    bout_type = 'Contact'; bout_start = s; break
            if bout_type is None:
                for s, e in prox_bouts:
                    if s <= tracked_idx <= e:
                        bout_type = 'Proximity'; bout_start = s; break
        if bout_type is not None:
            elapsed_b = (tracked_idx - bout_start) / fps
            self._lbl_btype.setText(f'{bout_type} (General)')
            self._lbl_belap.setText(f'Elapsed: {elapsed_b:.1f}s')
        else:
            self._lbl_btype.setText('none')
            self._lbl_belap.setText('')

        # ---- RUNNING TALLY ----
        prox_ct = cache.get('prox_cumtime')
        cont_ct = cache.get('cont_cumtime')
        pt = float(prox_ct[tracked_idx]) if (prox_ct is not None and tracked_idx is not None
                                             and tracked_idx < len(prox_ct)) else 0.0
        ct = float(cont_ct[tracked_idx]) if (cont_ct is not None and tracked_idx is not None
                                             and tracked_idx < len(cont_ct)) else 0.0
        self._lbl_ptally.setText(f'Prox so far:    {pt:.1f}s')
        self._lbl_ctally.setText(f'Contact so far: {ct:.1f}s')

    # ---- helpers -------------------------------------------------------

    def _set_chip(self, text, role):
        self._lbl_chip.setText(text)
        if role != self._chip_role:
            self._chip_role = role
            self._lbl_chip.setProperty('role', role)
            self._lbl_chip.style().unpolish(self._lbl_chip)
            self._lbl_chip.style().polish(self._lbl_chip)


class _DataPopup(QWidget):
    """Floating window showing all computed data for the current frame."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Frame Data Inspector")
        self.setWindowIcon(_make_icon("inspector"))
        self.resize(540, 740)
        self.setStyleSheet(
            "QWidget { background:#1a1a1a; color:#ccc;"
            "  font:11px 'Consolas','Courier New',monospace; }"
            "QTableWidget { background:#1e1e1e; gridline-color:#2d2d2d; border:none; }"
            "QTableWidget::item { padding:1px 5px; }"
            "QTableWidget::item:alternate { background:#212121; }"
            "QHeaderView::section { background:#252525; color:#888;"
            "  border:1px solid #333; padding:3px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._lbl_frame = QLabel("Frame: —")
        self._lbl_frame.setStyleSheet(
            "color:#00d4f0; font-size:13px; font-weight:bold; padding:2px 4px;")
        layout.addWidget(self._lbl_frame)

        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Feature", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        layout.addWidget(self._table)

        self._updaters = []   # list of (getter_fn(sleap_idx)->str, QTableWidgetItem)
        self._fps = 30.0
        self._ready = False

    # ---- helpers ----------------------------------------------------------

    def _add_section(self, title):
        r = self._table.rowCount()
        self._table.insertRow(r)
        item = QTableWidgetItem(f"  {title}")
        item.setBackground(QColor(35, 55, 75))
        item.setForeground(QColor(0, 210, 240))
        f = item.font(); f.setBold(True); item.setFont(f)
        self._table.setItem(r, 0, item)
        self._table.setSpan(r, 0, 1, 2)
        self._table.setRowHeight(r, 18)

    def _add_row(self, label, getter):
        r = self._table.rowCount()
        self._table.insertRow(r)
        li = QTableWidgetItem(f"  {label}")
        li.setForeground(QColor(160, 160, 160))
        vi = QTableWidgetItem("—")
        vi.setForeground(QColor(230, 230, 230))
        self._table.setItem(r, 0, li)
        self._table.setItem(r, 1, vi)
        self._table.setRowHeight(r, 16)
        self._updaters.append((getter, vi))

    # ---- public -----------------------------------------------------------

    def setup(self, tracks, kin, single_beh, pair_beh,
              track_feat, pair_feat, node_names, track_names, fps, px_per_cm,
              analysis_start_vidframe=0):
        """Build row structure once after processing. Call from main thread."""
        self._table.setRowCount(0)
        self._updaters.clear()
        self._fps = fps
        self._analysis_start_vidframe = analysis_start_vidframe
        self._ready = False

        _, _, n_nodes, n_tracks = tracks.shape
        ipc  = 1.0 / max(px_per_cm, 1e-9)

        single_keys = ('stationary', 'walking', 'running', 'turning', 'dir_reversal')

        for t, tname in enumerate(track_names):
            self._add_section(f"{tname}  —  Position & Kinematics")
            # Body-axis heading (per-track, used by all behaviors)
            self._add_row("body heading (deg)",
                lambda si, _t=t: _fmt(kin['body_heading_deg'][si, _t]))
            for n, nn in enumerate(node_names):
                self._add_row(f"{nn}  x (px)",
                    lambda si, _t=t, _n=n: _fmt(tracks[si, 0, _n, _t]))
                self._add_row(f"{nn}  y (px)",
                    lambda si, _t=t, _n=n: _fmt(tracks[si, 1, _n, _t]))
                self._add_row(f"{nn}  speed (cm/s)",
                    lambda si, _t=t, _n=n, _s=ipc: _fmt(kin['speed'][si, _n, _t] * _s))
                self._add_row(f"{nn}  vel heading (deg)",
                    lambda si, _t=t, _n=n: _fmt(kin['heading_deg'][si, _n, _t]))
                self._add_row(f"{nn}  accel (cm/s2)",
                    lambda si, _t=t, _n=n, _s=ipc: _fmt(kin['accel'][si, _n, _t] * _s))
                self._add_row(f"{nn}  jerk (cm/s3)",
                    lambda si, _t=t, _n=n, _s=ipc: _fmt(kin['jerk'][si, _n, _t] * _s))

            self._add_section(f"{tname}  —  Behaviors")
            for bk in single_keys:
                if bk not in single_beh:
                    continue
                arr = single_beh[bk]
                self._add_row(bk, lambda si, _a=arr, _t=t: str(int(_a[si, _t])))

            if t in track_feat:
                self._add_section(f"{tname}  —  Animal Features")
                # Skip features already shown in the Kinematics section above
                _kin_dupes = set()
                for nn in node_names:
                    nn_c = nn.replace(' ', '_')
                    _kin_dupes.update([
                        f'{nn_c}_x', f'{nn_c}_y',
                        f'{nn_c}_speed', f'{nn_c}_accel', f'{nn_c}_jerk',
                    ])
                _kin_dupes.add('body_heading_deg')
                for fname, farr in track_feat[t].items():
                    if fname in _kin_dupes:
                        continue
                    scale, unit = _feat_unit(fname, ipc)
                    label = f"{fname} ({unit})" if unit else fname
                    if scale == 1.0:
                        self._add_row(label, lambda si, _a=farr: _fmt(_a[si]))
                    else:
                        self._add_row(label,
                            lambda si, _a=farr, _s=scale: _fmt(_a[si] * _s))

        # Pair behaviors & features
        seen = set()
        for key in pair_beh:
            if '/' not in key:
                continue
            pfx = key.rsplit('/', 1)[0]
            if pfx in seen:
                continue
            seen.add(pfx)
            try:
                parts = pfx.split('_')
                tA = int(parts[0][1:]); tB = int(parts[1][1:])
                nA = track_names[tA] if tA < len(track_names) else f't{tA}'
                nB = track_names[tB] if tB < len(track_names) else f't{tB}'
            except (ValueError, IndexError):
                nA, nB = pfx, ''

            self._add_section(f"{nA} vs {nB}  —  Pair Behaviors")
            for pkey, arr in pair_beh.items():
                if '/' not in pkey:
                    continue
                kpfx, bname = pkey.rsplit('/', 1)
                if kpfx != pfx:
                    continue
                self._add_row(bname, lambda si, _a=arr: _fmt(_a[si]))

            self._add_section(f"{nA} vs {nB}  —  Pair Features")
            for pkey, arr in pair_feat.items():
                if '/' not in pkey:
                    continue
                kpfx, fname = pkey.rsplit('/', 1)
                if kpfx != pfx:
                    continue
                scale, unit = _feat_unit(fname, ipc)
                label = f"{fname} ({unit})" if unit else fname
                if scale == 1.0:
                    self._add_row(label, lambda si, _a=arr: _fmt(_a[si]))
                else:
                    self._add_row(label,
                        lambda si, _a=arr, _s=scale: _fmt(_a[si] * _s))

        self._ready = True

    def refresh(self, sleap_idx, vid_frame):
        if not self._ready:
            return
        analysis_t = max(0, vid_frame - self._analysis_start_vidframe) / self._fps
        self._lbl_frame.setText(
            f"Frame {vid_frame}   |   t = {analysis_t:.3f} s")
        for getter, item in self._updaters:
            try:
                item.setText(getter(sleap_idx))
            except Exception:
                item.setText("—")


def _classify_zone(x, y, rx0, ry0, rx1, ry1, strip):
    left   = x < rx0 + strip
    right  = x > rx1 - strip
    top    = y < ry0 + strip
    bottom = y > ry1 - strip
    if left  and top:     return "C1"
    if right and top:     return "C2"
    if right and bottom:  return "C3"
    if left  and bottom:  return "C4"
    if top:    return "W1"
    if right:  return "W2"
    if bottom: return "W3"
    if left:   return "W4"
    return "Open"


def _classify_zone_vec(x, y, rx0, ry0, rx1, ry1, strip):
    """Vectorized zone classification for arrays of coordinates."""
    n = len(x)
    zones = np.full(n, 'Open', dtype=object)
    left   = x < rx0 + strip
    right  = x > rx1 - strip
    top    = y < ry0 + strip
    bottom = y > ry1 - strip
    # Walls (overwritten by corners below)
    zones[top]    = 'W1'
    zones[right]  = 'W2'
    zones[bottom] = 'W3'
    zones[left]   = 'W4'
    # Corners
    zones[left  & top]    = 'C1'
    zones[right & top]    = 'C2'
    zones[right & bottom] = 'C3'
    zones[left  & bottom] = 'C4'
    return zones


class RunPopUp(QWidget):
    """Popup window for frame-by-frame navigation, ROI drawing, and SLEAP overlay.

    Parameters
    ----------
    video_path : str
        Path to the video file to display.
    sleap_data : dict or None
        Parsed SLEAP data returned by sleap_loader.load_sleap(). When provided,
        the skeleton is overlaid on every frame. Pass None to disable the overlay.
    """

    roi_selected = Signal(object)  # emits ((x0,y0),(x1,y1)) or None on Confirm

    # Track colors (BGR) cycled across instances.
    _TRACK_COLORS = [
        (255,  80,  80),  # blue-ish
        ( 80, 220,  80),  # green
        ( 80,  80, 255),  # red
        ( 80, 220, 220),  # yellow
        (220,  80, 220),  # magenta
        (220, 220,  80),  # cyan
    ]

    _ZONE_COLORS = {
        "C1": (0, 140, 255), "C2": (0, 140, 255),
        "C3": (0, 140, 255), "C4": (0, 140, 255),
        "W1": (40, 200, 40), "W2": (40, 200, 40),
        "W3": (40, 200, 40), "W4": (40, 200, 40),
        "Open": (200, 100, 200),
    }

    def __init__(self, video_path, sleap_data=None):
        super().__init__()
        self.setStyleSheet("""
            QWidget { background:#1a1a1a; color:#ccc; font:13px 'Segoe UI',sans-serif; }
            QPushButton { background:#2d2d2d; color:#eee; border:1px solid #444;
                          border-radius:4px; padding:6px 10px; min-width:54px; }
            QPushButton:hover  { background:#3d3d3d; }
            QPushButton:pressed{ background:#1f1f1f; }
            QSlider::groove:horizontal { height:4px; background:#444; border-radius:2px; }
            QSlider::handle:horizontal { width:14px; height:14px; margin:-5px 0;
                                          border-radius:7px; background:#00c8e0; }
            QSlider::sub-page:horizontal { background:#00c8e0; border-radius:2px; }
            QLabel { color:#888; font-size:12px; }
            QSpinBox { background:#2d2d2d; color:#eee; border:1px solid #444;
                       padding:4px 20px 4px 6px; }
            QSpinBox::up-button   { width:18px; border-left:1px solid #444; background:#2d2d2d; }
            QSpinBox::down-button { width:18px; border-left:1px solid #444; background:#2d2d2d; }
            QSpinBox::up-button:hover   { background:#3d3d3d; }
            QSpinBox::down-button:hover { background:#3d3d3d; }
            QSpinBox::up-arrow   { width:7px; height:7px; }
            QSpinBox::down-arrow { width:7px; height:7px; }
        """)

        self._video_path = video_path
        self._cap        = cv2.VideoCapture(video_path)
        self._n_frames   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self._index      = 0
        self._fps             = float(self._cap.get(cv2.CAP_PROP_FPS) or 30)
        self._sleap_data = sleap_data
        self._processed_sleap_data = None
        self._timer           = QTimer(self)
        self._timer.setInterval(min(1000, max(1, int(1000 / self._fps))))
        self._timer.timeout.connect(self._advance_frame)
        self._slider_dragging    = False
        self._slider_pending_val = None
        self._slider_debounce    = QTimer(self)
        self._slider_debounce.setSingleShot(True)
        self._slider_debounce.setInterval(30)
        self._slider_debounce.timeout.connect(self._on_slider_debounced)
        self._zones     = None   # dict zone_name → (x0,y0,x1,y1) native px, or None
        self._px_per_cm = None   # float, computed when ROI + cm spinbox are both valid

        # Analysis cache (populated after Process)
        self._analysis_cache = None   # dict with kin, single_beh, pair_beh, tracks, etc.
        self._analysis_start_vidframe = 0  # video frame where analysis begins (after late-placement trimming)
        self._data_popup = None

        self._build_ui()
        self._show_frame(0)

    # ---- UI construction -----------------------------------------------

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(6)

        # Two side-by-side video views
        video_row = QHBoxLayout()
        video_row.setSpacing(8)

        lbl_orig = QLabel("Original")
        lbl_orig.setAlignment(Qt.AlignCenter)
        self.view = ROIView()
        self.view.setMinimumSize(360, 200)
        left_col = QVBoxLayout()
        left_col.setSpacing(3)
        left_col.addWidget(lbl_orig)
        left_col.addWidget(self.view, stretch=1)

        lbl_post = QLabel("Post-processed")
        lbl_post.setAlignment(Qt.AlignCenter)
        self.view_b = ROIView()
        self.view_b.setMinimumSize(360, 200)
        self._metrics_panel = _MetricsPanel(self)
        right_col = QVBoxLayout()
        right_col.setSpacing(3)
        right_col.addWidget(lbl_post)
        right_col.addWidget(self.view_b, stretch=1)
        right_col.addWidget(self._metrics_panel, stretch=0)

        video_row.addLayout(left_col, stretch=1)
        video_row.addLayout(right_col, stretch=1)
        main.addLayout(video_row, stretch=1)

        # Progress row
        progress_row = QHBoxLayout()
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._n_frames - 1)
        self._frame_label = QLabel(f"0000 / {self._n_frames - 1:04d}")
        self._frame_label.setFixedWidth(90)
        self._frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._time_label = QLabel("0.00s")
        self._time_label.setFixedWidth(60)
        self._time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        progress_row.addWidget(self._slider)
        progress_row.addWidget(self._frame_label)
        progress_row.addWidget(self._time_label)
        main.addLayout(progress_row)

        # Controls row
        controls_row = QHBoxLayout()
        self._btn_prev = QPushButton("◀  Prev")
        self._btn_play = QPushButton("▶  Play")
        self._btn_next = QPushButton("▶  Next")
        for btn in (self._btn_prev, self._btn_play, self._btn_next):
            controls_row.addWidget(btn)
        controls_row.addStretch()
        controls_row.addWidget(QLabel("Width:"))
        self._lbl_width = QLabel("—")
        self._lbl_width.setMinimumWidth(60)
        self._lbl_width.setStyleSheet(
            "color:#00d4f0; font:12px 'Consolas','Courier New',monospace;"
            "background:#222; border:1px solid #383838; border-radius:3px; padding:3px 6px;"
        )
        controls_row.addWidget(self._lbl_width)
        controls_row.addSpacing(8)
        controls_row.addWidget(QLabel("Arena size (cm):"))
        self._spin_arena_cm = QSpinBox()
        self._spin_arena_cm.setRange(1, 9999)
        self._spin_arena_cm.setValue(40)
        self._spin_arena_cm.setMinimumWidth(85)
        self._spin_arena_cm.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._spin_arena_cm.setStyleSheet(
            "QSpinBox { background:#2d2d2d; color:#f0f0f0; border:1px solid #666;"
            "           border-radius:3px; padding:4px 22px 4px 8px; font-size:13px; }"
            "QSpinBox::up-button   { width:20px; border-left:1px solid #555; background:#383838; }"
            "QSpinBox::down-button { width:20px; border-left:1px solid #555; background:#383838; }"
            "QSpinBox::up-button:hover   { background:#4a4a4a; }"
            "QSpinBox::down-button:hover { background:#4a4a4a; }"
            "QSpinBox::up-arrow   { width:8px; height:8px; }"
            "QSpinBox::down-arrow { width:8px; height:8px; }"
        )
        controls_row.addWidget(self._spin_arena_cm)
        controls_row.addSpacing(8)
        controls_row.addWidget(QLabel("Border (cm):"))
        self._spin_strip_cm = QSpinBox()
        self._spin_strip_cm.setRange(1, 999)
        self._spin_strip_cm.setValue(8)
        self._spin_strip_cm.setMinimumWidth(85)
        self._spin_strip_cm.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._spin_strip_cm.setStyleSheet(
            "QSpinBox { background:#2d2d2d; color:#f0f0f0; border:1px solid #666;"
            "           border-radius:3px; padding:4px 22px 4px 8px; font-size:13px; }"
            "QSpinBox::up-button   { width:20px; border-left:1px solid #555; background:#383838; }"
            "QSpinBox::down-button { width:20px; border-left:1px solid #555; background:#383838; }"
            "QSpinBox::up-button:hover   { background:#4a4a4a; }"
            "QSpinBox::down-button:hover { background:#4a4a4a; }"
            "QSpinBox::up-arrow   { width:8px; height:8px; }"
            "QSpinBox::down-arrow { width:8px; height:8px; }"
        )
        controls_row.addWidget(self._spin_strip_cm)
        self._btn_clear   = QPushButton("Clear")
        self._btn_confirm = QPushButton("Confirm")
        for btn in (self._btn_clear, self._btn_confirm):
            controls_row.addWidget(btn)
        self._btn_process = QPushButton("Process")
        self._btn_process.setEnabled(False)
        self._btn_export  = QPushButton("Export")
        self._btn_inspect = QPushButton("Inspect")
        self._btn_inspect.setCheckable(True)
        self._btn_inspect.setEnabled(False)
        controls_row.addWidget(self._btn_process)
        controls_row.addWidget(self._btn_export)
        controls_row.addWidget(self._btn_inspect)
        main.addLayout(controls_row)

        # Progress bar (hidden until processing starts)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background:#222; border:none; border-radius:3px; }"
            "QProgressBar::chunk { background:#00c8e0; border-radius:3px; }"
        )
        main.addWidget(self._progress_bar)

        # (spacer removed — _MetricsPanel now occupies this area)

        # Signal connections
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._btn_prev.clicked.connect(lambda: self._show_frame(self._index - 1))
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_next.clicked.connect(lambda: self._show_frame(self._index + 1))
        self._btn_clear.clicked.connect(self._clear_roi_and_labels)
        self._btn_confirm.clicked.connect(self._confirm_roi)
        self._spin_arena_cm.valueChanged.connect(self._on_arena_cm_changed)
        self._spin_strip_cm.valueChanged.connect(self._on_arena_cm_changed)
        self._btn_process.clicked.connect(self._run_process)
        self._btn_export.clicked.connect(self._run_export)
        self._btn_inspect.toggled.connect(self._on_inspect_toggled)
        self.view.roi_changed.connect(self._on_roi_changed)
        self.view_b.roi_changed.connect(self._on_roi_changed)

        self._worker = None
        self._proc_thread = None
        self._analysis_worker = None
        self._analysis_thread = None
        self._export_worker = None
        self._export_thread = None

        self.setWindowTitle("ROI Selector")
        self.setWindowIcon(_make_icon("roi"))
        self.showMaximized()

    # ---- Frame display -------------------------------------------------

    def _show_frame(self, index):
        index = max(0, min(index, self._n_frames - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()

        if ok:
            self._index = index

            frame_orig = frame.copy()
            frame_post = frame.copy()

            if self._sleap_data is not None:
                frame_orig = self._draw_sleap(frame_orig, index, self._sleap_data)

            in_analysis = index >= self._analysis_start_vidframe
            if self._processed_sleap_data is not None and in_analysis:
                frame_post = self._draw_sleap(frame_post, index, self._processed_sleap_data)

            if self._zones:
                frame_orig = self._draw_zones(frame_orig)
                frame_post = self._draw_zones(frame_post)

            # Look up the TRIMMED sleap index from the analysis cache's
            # frame_map — NOT from _processed_sleap_data which is untrimmed.
            # The analysis arrays (kin, behaviors, features) all start at
            # index 0 corresponding to the first post-trim frame.
            sleap_idx = None
            if self._analysis_cache is not None and in_analysis:
                sleap_idx = self._analysis_cache['frame_map'].get(index)

            if sleap_idx is not None:
                frame_post = self._draw_analysis_overlay(frame_post, sleap_idx)
                if (self._data_popup is not None
                        and self._data_popup.isVisible()):
                    self._data_popup.refresh(sleap_idx, index)
                self._metrics_panel.refresh(
                    sleap_idx, index, self._n_frames, self._fps,
                    self._analysis_start_vidframe, self._analysis_cache)
            else:
                self._metrics_panel.clear()

            self.view.set_frame(frame_orig)
            self.view_b.set_frame(frame_post)

            if not self._slider_dragging:
                self._slider.setValue(self._index)

            self._frame_label.setText(f"{self._index:04d} / {self._n_frames - 1:04d}")

            # Time display: counts from 0 starting at analysis start (5s trim)
            elapsed = max(0, index - self._analysis_start_vidframe) / self._fps
            self._time_label.setText(f"{elapsed:.1f}s")

    # ---- SLEAP skeleton overlay ----------------------------------------

    def _draw_sleap(self, frame, video_frame_idx, sleap_data):
        frame_map = sleap_data["frame_map"]
        if video_frame_idx not in frame_map:
            return frame

        sleap_idx = frame_map[video_frame_idx]
        tracks = sleap_data["tracks"]
        edge_inds = sleap_data["edge_inds"]
        n_tracks = tracks.shape[3]

        for t in range(n_tracks):
            color = self._TRACK_COLORS[t % len(self._TRACK_COLORS)]
            pts = tracks[sleap_idx, :, :, t]

            for src, dst in edge_inds:
                x0, y0 = pts[0, src], pts[1, src]
                x1, y1 = pts[0, dst], pts[1, dst]
                if not any(np.isnan([x0, y0, x1, y1])):
                    cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 2, cv2.LINE_AA)

            for n in range(pts.shape[1]):
                x, y = pts[0, n], pts[1, n]
                if not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)

        return frame

    # ---- Playback handlers ---------------------------------------------

    def _toggle_play(self):
        if self._timer.isActive():
            self._timer.stop()
            self._btn_play.setText("▶  Play")
        else:
            self._timer.start()
            self._btn_play.setText("⏸  Pause")

    def _advance_frame(self):
        self._show_frame((self._index + 1) % self._n_frames)

    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_moved(self, v):
        self._slider_pending_val = v
        self._slider_debounce.start()   # restart timer each pixel; fires once after 30ms quiet

    def _on_slider_debounced(self):
        if self._slider_pending_val is not None:
            self._show_frame(self._slider_pending_val)

    def _on_slider_released(self):
        self._slider_dragging = False
        self._slider_debounce.stop()
        self._show_frame(self._slider.value())

    # ---- ROI handlers --------------------------------------------------

    def _on_roi_changed(self, rect):
        # Mirror rect to both views (setRect doesn't re-emit, so no loop)
        self.view._rect_item.setRect(rect)
        self.view_b._rect_item.setRect(rect)
        x0, y0 = int(round(rect.left())),  int(round(rect.top()))
        x1, y1 = int(round(rect.right())), int(round(rect.bottom()))
        self._lbl_width.setText(f"{x1 - x0} px")
        self._recompute_zones()
        self._btn_process.setEnabled(True)
        self._show_frame(self._index)

    def _clear_roi_and_labels(self):
        self.view.clear_roi()
        self.view_b.clear_roi()
        self._lbl_width.setText("—")
        self._zones = None
        self._btn_process.setEnabled(False)
        self._show_frame(self._index)

    def _confirm_roi(self):
        self.roi_selected.emit(self.view.roi_native())
        self.close()

    # ---- Zone computation & drawing ------------------------------------

    def _recompute_zones(self):
        roi = self.view.roi_native()
        if roi is None:
            self._zones = None; self._px_per_cm = None; return
        (rx0, ry0), (rx1, ry1), side = roi
        cm = self._spin_arena_cm.value()
        if cm <= 0 or side <= 0:
            self._zones = None; return
        self._px_per_cm = side / cm
        s = int(round(self._spin_strip_cm.value() * self._px_per_cm))
        self._zones = {
            "C1":     (rx0,     ry0,     rx0+s,  ry0+s),
            "C2":     (rx1-s,   ry0,     rx1,    ry0+s),
            "C3":     (rx1-s,   ry1-s,   rx1,    ry1  ),
            "C4":     (rx0,     ry1-s,   rx0+s,  ry1  ),
            "W1":     (rx0+s,   ry0,     rx1-s,  ry0+s),
            "W2":     (rx1-s,   ry0+s,   rx1,    ry1-s),
            "W3":     (rx0+s,   ry1-s,   rx1-s,  ry1  ),
            "W4":     (rx0,     ry0+s,   rx0+s,  ry1-s),
            "Open":   (rx0+s,   ry0+s,   rx1-s,  ry1-s),
        }

    def _draw_zones(self, frame):
        if not self._zones:
            return frame
        overlay = frame.copy()
        for name, (zx0, zy0, zx1, zy1) in self._zones.items():
            cv2.rectangle(overlay, (zx0, zy0), (zx1, zy1), self._ZONE_COLORS[name], -1)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        for name, (zx0, zy0, zx1, zy1) in self._zones.items():
            color = self._ZONE_COLORS[name]
            cv2.rectangle(frame, (zx0, zy0), (zx1, zy1), color, 1)
            cx, cy = (zx0 + zx1) // 2, (zy0 + zy1) // 2
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, name, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame

    def _on_arena_cm_changed(self, _):
        self._recompute_zones()
        if self._analysis_cache is not None:
            self._precompute_analysis(proc_opts=getattr(self, '_proc_opts', None))
        self._show_frame(self._index)

    # ---- Process -------------------------------------------------------

    def _run_process(self):
        if self._sleap_data is None:
            QMessageBox.warning(self, "No SLEAP data", "Load a SLEAP .h5 file first.")
            return
        try:
            if self._proc_thread is not None and self._proc_thread.isRunning():
                return
        except RuntimeError:
            pass

        # Show processing options dialog before starting
        n_tracks = self._sleap_data["tracks"].shape[3]
        proc_dlg = ProcessingOptionsDialog(
            n_tracks=n_tracks,
            default_opts=getattr(self, '_proc_opts', None),
            parent=self,
        )
        if proc_dlg.exec() != QDialog.Accepted:
            return
        self._proc_opts = proc_dlg.options()

        self._btn_process.setEnabled(False)
        self._btn_process.setText("Processing…")
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)

        self._worker = _ProcessWorker(self._sleap_data, self._fps)
        self._proc_thread = QThread()
        self._worker.moveToThread(self._proc_thread)

        self._proc_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress_bar.setValue)
        self._worker.finished.connect(self._on_process_done)
        self._worker.error.connect(self._on_process_error)
        self._worker.finished.connect(self._proc_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._proc_thread.quit)
        self._worker.error.connect(self._worker.deleteLater)
        self._proc_thread.finished.connect(self._proc_thread.deleteLater)

        self._proc_thread.start()

    def _on_process_done(self, processed_data):
        if self._analysis_cache is not None:
            self._analysis_cache.clear()
            self._analysis_cache = None
        self._processed_sleap_data = processed_data
        # Don't re-enable btn_process yet — analysis worker will do it
        self._progress_bar.setVisible(True)   # keep visible for analysis phase
        self._btn_process.setText("Analyzing…")
        self._precompute_analysis(proc_opts=getattr(self, '_proc_opts', None))
        # _show_frame called from _on_analysis_done

    def _on_process_error(self, msg):
        self._btn_process.setEnabled(True)
        self._btn_process.setText("Process")
        self._progress_bar.setVisible(False)
        _ScrollableMessageBox(self, "Process failed", msg, critical=True).exec()

    # ---- Analysis precompute & overlay ---------------------------------

    def _precompute_analysis(self, proc_opts=None):
        """Start background analysis worker. UI remains responsive."""
        data = self._processed_sleap_data
        if data is None:
            return
        roi = self.view.roi_native()
        if roi is None:
            return

        # Cancel any in-progress analysis
        try:
            if self._analysis_thread is not None and self._analysis_thread.isRunning():
                self._analysis_thread.quit()
                self._analysis_thread.wait(500)
        except RuntimeError:
            pass

        # Clear stale cache
        if self._analysis_cache is not None:
            self._analysis_cache.clear()
            self._analysis_cache = None

        (_, _), (_, _), side = roi
        px_per_cm = side / max(self._spin_arena_cm.value(), 1)
        strip_cm  = self._spin_strip_cm.value()

        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._btn_process.setEnabled(False)
        self._btn_process.setText("Analyzing…")

        self._analysis_worker = _AnalysisWorker(
            data, self._fps, roi, px_per_cm, strip_cm, proc_opts or {}
        )
        self._analysis_thread = QThread()
        self._analysis_worker.moveToThread(self._analysis_thread)

        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.done.connect(self._on_analysis_done)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.done.connect(self._analysis_thread.quit)
        self._analysis_worker.done.connect(self._analysis_worker.deleteLater)
        self._analysis_worker.error.connect(self._analysis_thread.quit)
        self._analysis_worker.error.connect(self._analysis_worker.deleteLater)
        self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)

        self._analysis_thread.start()

    def _on_analysis_done(self, cache_dict):
        self._analysis_cache = cache_dict
        self._analysis_start_vidframe = cache_dict['analysis_start_vidframe']
        self._progress_bar.setVisible(False)
        self._btn_process.setEnabled(True)
        self._btn_process.setText("Process")
        self._metrics_panel.clear()
        if self._data_popup is None:
            self._data_popup = _DataPopup(self)
        c = cache_dict
        self._data_popup.setup(
            c['tracks'], c['kin'], c['single_beh'], c['pair_beh'],
            c['track_feat'], c['pair_feat'],
            c['node_names'], c['track_names'], self._fps, c['px_per_cm'],
            analysis_start_vidframe=c['analysis_start_vidframe'],
        )
        self._btn_inspect.setEnabled(True)
        self._show_frame(self._index)

    def _on_analysis_error(self, msg):
        self._progress_bar.setVisible(False)
        self._btn_process.setEnabled(True)
        self._btn_process.setText("Process")
        _ScrollableMessageBox(self, "Analysis failed", msg, critical=True).exec()

    def _draw_analysis_overlay(self, frame, sleap_idx):
        """Info now shown in _MetricsPanel; this overlay is retired."""
        return frame

    def _on_inspect_toggled(self, checked):
        if self._data_popup is None:
            return
        if checked:
            self._data_popup.show()
            self._data_popup.raise_()
        else:
            self._data_popup.hide()

    # ---- Export --------------------------------------------------------

    def _run_export(self):
        """Show the save dialog on the main thread, then hand off all computation
        and file writing to _ExportWorker running on a background QThread."""
        if self._processed_sleap_data is None:
            QMessageBox.warning(self, "Not processed", "Run 'Process' first.")
            return
        roi = self.view.roi_native()
        if roi is None:
            QMessageBox.warning(self, "No ROI", "Draw an ROI first.")
            return
        try:
            if self._export_thread is not None and self._export_thread.isRunning():
                return
        except RuntimeError:
            pass

        if not _EXPORT_SEMAPHORE.acquire(blocking=False):
            QMessageBox.information(self, "Export Queued",
                "Another export is already running. Please wait for it to finish.")
            return

        # Unified export dialog (checkboxes + save location)
        vp = Path(self._video_path)
        dlg = ExportOptionsDialog(
            graphs_available=_GRAPHS_AVAILABLE,
            proc_opts=self._analysis_cache.get('proc_opts') if self._analysis_cache else None,
            default_dir=str(vp.parent),
            default_name=vp.stem,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            _EXPORT_SEMAPHORE.release()
            return
        export_opts = dlg.options()
        base_path = dlg.export_path()       # "C:/Users/results/my_experiment"
        path = base_path + ".xlsx"           # main workbook path

        # Disable button & show progress bar while export runs
        self._btn_export.setEnabled(False)
        self._btn_export.setText("Exporting…")
        self._progress_bar.setRange(0, 0)   # indeterminate (marquee) mode
        self._progress_bar.setVisible(True)

        # Grab a representative arena snapshot (middle frame, cropped to ROI)
        arena_snapshot = None
        try:
            (rx0_, ry0_), (rx1_, ry1_), _ = roi
            mid = self._n_frames // 2
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok_snap, snap_frame = self._cap.read()
            if ok_snap and snap_frame is not None:
                r0 = max(0, int(ry0_)); r1 = min(snap_frame.shape[0], int(ry1_))
                c0 = max(0, int(rx0_)); c1 = min(snap_frame.shape[1], int(rx1_))
                crop = snap_frame[r0:r1, c0:c1]
                arena_snapshot = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # Restore current playback position
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._index)
        except Exception:
            pass

        self._export_worker = _ExportWorker(
            analysis_cache=self._analysis_cache,
            fps=self._fps,
            roi=roi,
            arena_cm=self._spin_arena_cm.value(),
            strip_cm=self._spin_strip_cm.value(),
            path=path,
            arena_snapshot=arena_snapshot,
            export_opts=export_opts,
        )
        self._export_thread = QThread()
        self._export_worker.moveToThread(self._export_thread)

        self._export_thread.started.connect(self._export_worker.run)
        self._export_worker.finished.connect(self._on_export_done)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.finished.connect(self._export_thread.quit)
        self._export_worker.finished.connect(self._export_worker.deleteLater)
        self._export_worker.error.connect(self._export_thread.quit)
        self._export_worker.error.connect(self._export_worker.deleteLater)
        self._export_thread.finished.connect(self._export_thread.deleteLater)

        self._export_thread.start()

    def _on_export_done(self, msg):
        """Called on the main thread when _ExportWorker finishes successfully."""
        _EXPORT_SEMAPHORE.release()
        self._btn_export.setEnabled(True)
        self._btn_export.setText("Export")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(100)
        self._progress_bar.setVisible(False)
        _ScrollableMessageBox(self, "Export Complete", msg).exec()

    def _on_export_error(self, msg):
        """Called on the main thread when _ExportWorker raises an exception."""
        _EXPORT_SEMAPHORE.release()
        self._btn_export.setEnabled(True)
        self._btn_export.setText("Export")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        _ScrollableMessageBox(self, "Export failed", msg, critical=True).exec()

    def closeEvent(self, event):
        try:
            if self._export_thread is not None and self._export_thread.isRunning():
                QMessageBox.warning(
                    self, "Export in progress",
                    "An export is currently running.\n"
                    "Please wait for it to finish before closing.")
                event.ignore()
                return
        except RuntimeError:
            pass  # C++ QThread already deleted — export is done
        self._timer.stop()
        self._cap.release()
        if self._analysis_cache is not None:
            self._analysis_cache.clear()
            self._analysis_cache = None
        self._processed_sleap_data = None
        if self._data_popup is not None:
            self._data_popup.close()
            self._data_popup = None
        gc.collect()
        super().closeEvent(event)
