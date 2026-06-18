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
)
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QFont
from pathlib import Path
from preprocessing import fill_and_smooth_tracks, compute_kinematics
from behaviors import (compute_single_animal, compute_pairwise, compute_behavior_summary,
                       compute_second_order, _SECOND_ORDER_KEYS, find_node_idx)
from features import build_feature_dataframes, precompute_feature_arrays
from binned_export import write_binned_xlsx
from roi_view import ROIView
try:
    from graph_export import write_graphs as _write_graphs
    _GRAPHS_AVAILABLE = True
except Exception:          # matplotlib not installed or import error
    _GRAPHS_AVAILABLE = False


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

    def __init__(self, processed_data, fps, roi, arena_cm, strip_cm, path,
                 arena_snapshot=None, export_opts=None):
        """
        Parameters
        ----------
        processed_data : dict   — the smoothed SLEAP data dict (from _ProcessWorker)
        fps            : float  — video frame rate
        roi            : tuple  — ((rx0,ry0),(rx1,ry1), side) from view.roi_native()
        arena_cm       : int    — arena size in cm (from spin box)
        strip_cm       : int    — border strip width in cm
        path           : str    — destination .xlsx (or .csv) file path chosen by user
        arena_snapshot : ndarray or None — RGB image of arena cropped to ROI
        export_opts    : dict[str,bool] or None — from ExportOptionsDialog.options(); None = all
        """
        super().__init__()
        self._data      = processed_data
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

            tracks      = self._data["tracks"]
            frame_map   = self._data["frame_map"]
            node_names  = self._data["node_names"]
            track_names = self._data["track_names"]

            # ---- Trim first 15 seconds (hand placement buffer) ----
            n_frames = tracks.shape[0]
            analysis_start = min(int(15 * self._fps), n_frames)
            self.status.emit(
                f"Trimming first 15s ({analysis_start} frames); "
                f"analysis starts at frame {analysis_start}")
            tracks = tracks[analysis_start:]
            frame_map = {vf: si - analysis_start
                         for vf, si in frame_map.items()
                         if si >= analysis_start}

            # ---- Kinematics ----
            self.status.emit("Computing kinematics…")
            kin = compute_kinematics(tracks, self._fps, node_names=node_names)

            # ---- Behavior detection ----
            self.status.emit("Detecting behaviors…")
            single_beh  = compute_single_animal(tracks, kin, node_names, self._fps,
                                                   px_per_cm=px_per_cm)
            pair_beh    = compute_pairwise(tracks, node_names, self._fps, dsr=None, kin=kin)

            # Second-order compound social behaviors
            self.status.emit("Detecting 2nd-order behaviors…")
            second_order = compute_second_order(
                tracks, kin, node_names, self._fps, single_beh, pair_beh)
            pair_beh.update(second_order)

            beh_summary, engagement_idx_df = compute_behavior_summary(
                single_beh, pair_beh, track_names, self._fps,
                tracks=tracks, kin=kin, frame_map=frame_map, node_names=node_names)

            # ---- Primitive & derivative features ----
            # Compute arrays once; pass to both the full DataFrame builder and the
            # binned export so the heavy feature computation runs only once.
            self.status.emit("Computing features…")
            track_arrays, pair_arrays = precompute_feature_arrays(
                tracks, kin, node_names, self._fps, roi=(rx0, ry0, rx1, ry1))
            animal_feat_df, pair_feat_df = build_feature_dataframes(
                tracks, kin, node_names, track_names, self._fps,
                frame_map, roi=(rx0, ry0, rx1, ry1),
                _precomputed=(track_arrays, pair_arrays),
            )

            # ---- Build main data table ----
            self.status.emit("Building data table…")
            sorted_frames = sorted(frame_map.items())
            vid_frame_0 = sorted_frames[0][0] if sorted_frames else 0
            rows = []
            for vid_frame, sleap_idx in sorted_frames:
                time_s = round((vid_frame - vid_frame_0) / self._fps, 4)
                for t in range(tracks.shape[3]):
                    pts = tracks[sleap_idx, :, :, t]       # (2, n_nodes)
                    for n, node in enumerate(node_names):
                        x, y = float(pts[0, n]), float(pts[1, n])
                        if np.isnan(x) or np.isnan(y):
                            zone = "Undetected"
                            x_cm = y_cm = float("nan")
                        else:
                            zone  = _classify_zone(x, y, rx0, ry0, rx1, ry1, strip)
                            x_cm  = round((x - rx0) / px_per_cm, 3)
                            y_cm  = round((y - ry0) / px_per_cm, 3)
                        rows.append({
                            "Frame":          vid_frame,
                            "Time (s)":       time_s,
                            "Track":          track_names[t],
                            "Body Part":      node,
                            "X (px)":         round(x, 2) if not np.isnan(x) else float("nan"),
                            "Y (px)":         round(y, 2) if not np.isnan(y) else float("nan"),
                            "X (cm)":         x_cm,
                            "Y (cm)":         y_cm,
                            "Zone":           zone,
                            "Vx (cm/s)":      round(float(kin["vx"][sleap_idx, n, t])    * inv_ppcm,  3),
                            "Vy (cm/s)":      round(float(kin["vy"][sleap_idx, n, t])    * inv_ppcm,  3),
                            "Speed (cm/s)":   round(float(kin["speed"][sleap_idx, n, t]) * inv_ppcm,  3),
                            "Heading (deg)":  round(float(kin["heading_deg"][sleap_idx, n, t]),        2),
                            "Accel (cm/s²)":  round(float(kin["accel"][sleap_idx, n, t]) * inv_ppcm, 3),
                            "Jerk (cm/s³)":   round(float(kin["jerk"][sleap_idx, n, t])  * inv_ppcm, 3),
                        })

            df = pd.DataFrame(rows)

            # ---- Build behavior dataframes (one row per video frame) ----
            single_keys = ('stationary', 'walking', 'running', 'turning', 'dir_reversal')

            # Pre-build pair prefix → track-name column prefix mapping
            pair_col_map = {}
            for key in pair_beh:
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
            first_order_pair_keys = {k: v for k, v in pair_beh.items()
                                     if k.rsplit('/', 1)[1] not in _so_key_set}
            second_order_pair_keys = {k: v for k, v in pair_beh.items()
                                      if k.rsplit('/', 1)[1] in _so_key_set}

            # 1st Order Behaviors sheet
            beh_rows = []
            for vid_frame, sleap_idx in sorted(frame_map.items()):
                row = {
                    'Frame':   vid_frame,
                    'Time(s)': round(vid_frame / self._fps, 4),
                }
                for bname in single_keys:
                    for t, tname in enumerate(track_names):
                        row[f'{tname}/{bname}'] = int(single_beh[bname][sleap_idx, t])
                for key, arr in first_order_pair_keys.items():
                    pfx, beh = key.rsplit('/', 1)
                    col = f'{pair_col_map[pfx]}/{beh}'
                    val = arr[sleap_idx]
                    try:
                        row[col] = '' if pd.isna(val) else val
                    except (TypeError, ValueError):
                        row[col] = val
                beh_rows.append(row)

            beh_df = pd.DataFrame(beh_rows)

            # 2nd Order Behaviors sheet
            beh2_rows = []
            for vid_frame, sleap_idx in sorted(frame_map.items()):
                row = {
                    'Frame':   vid_frame,
                    'Time(s)': round(vid_frame / self._fps, 4),
                }
                for key, arr in second_order_pair_keys.items():
                    pfx, beh = key.rsplit('/', 1)
                    col = f'{pair_col_map[pfx]}/{beh}'
                    val = arr[sleap_idx]
                    try:
                        row[col] = '' if pd.isna(val) else val
                    except (TypeError, ValueError):
                        row[col] = val
                beh2_rows.append(row)

            beh2_df = pd.DataFrame(beh2_rows)

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
                any_main = any(self._want(k) for k, _ in ExportOptionsDialog._MAIN_SHEETS)
                if any_main:
                    self.status.emit("Writing Excel workbook…")
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

                # ---- Binned export (*_binned.xlsx, same directory) ----
                any_binned = any(self._want(k) for k, _ in ExportOptionsDialog._BINNED_SHEETS)
                if any_binned:
                    self.status.emit("Writing binned export…")
                    _p = Path(path)
                    binned_path = str(_p.with_name(_p.stem + "_binned" + _p.suffix))
                    write_binned_xlsx(
                        track_arrays, pair_arrays, single_beh, pair_beh,
                        tracks, node_names, track_names, self._fps, frame_map,
                        kin=kin, output_path=binned_path,
                        include_sheets=self._opts,
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
                        self.status.emit(f"Graph export error (skipped): {_ge}")

            parts = []
            if any_main:
                parts.append(f"Main workbook:\n{path}")
            if binned_path:
                parts.append(f"Binned export:\n{binned_path}")
            if graph_paths:
                parts.append(f"Graphs ({len(graph_paths)} PDF files):\n"
                             + "\n".join(f"  • {Path(p).name}" for p in graph_paths))
            msg = "Export complete.\n\n" + "\n\n".join(parts)
            self.finished.emit(msg)

        except Exception as exc:
            import traceback
            self.error.emit(traceback.format_exc())


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
    ]

    _GRAPH_SUFFIXES = {
        "graph_heatmaps":           "_graphs_heatmaps.pdf",
        "graph_cascade":            "_graphs_cascade.pdf",
        "graph_distance":           "_graphs_distance.pdf",
        "graph_oncoplot":           "_graphs_oncoplot.pdf",
        "graph_sync_oncoplot":      "_graphs_sync_oncoplot.pdf",
        "graph_oncoplot_clean":     "_graphs_oncoplot_clean.pdf",
        "graph_sync_oncoplot_clean":"_graphs_sync_oncoplot_clean.pdf",
    }

    def __init__(self, graphs_available=True, default_dir="", default_name="export", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Options")
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

        self._updating = False   # guard against recursive signal loops

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


class _DataPopup(QWidget):
    """Floating window showing all computed data for the current frame."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Frame Data Inspector")
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
                arr = single_beh[bk]
                self._add_row(bk, lambda si, _a=arr, _t=t: str(int(_a[si, _t])))

            if t in track_feat:
                self._add_section(f"{tname}  —  Animal Features")
                for fname, farr in track_feat[t].items():
                    self._add_row(fname, lambda si, _a=farr: _fmt(_a[si]))

        # Pair behaviors & features
        seen = set()
        for key in pair_beh:
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
                kpfx, bname = pkey.rsplit('/', 1)
                if kpfx != pfx:
                    continue
                self._add_row(bname, lambda si, _a=arr: _fmt(_a[si]))

            self._add_section(f"{nA} vs {nB}  —  Pair Features")
            for pkey, arr in pair_feat.items():
                kpfx, fname = pkey.rsplit('/', 1)
                if kpfx != pfx:
                    continue
                self._add_row(fname, lambda si, _a=arr: _fmt(_a[si]))

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
                          border-radius:4px; padding:6px 12px; min-width:70px; }
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
        self._slider_dragging = False
        self._zones     = None   # dict zone_name → (x0,y0,x1,y1) native px, or None
        self._px_per_cm = None   # float, computed when ROI + cm spinbox are both valid

        # Analysis cache (populated after Process)
        self._analysis_cache = None   # dict with kin, single_beh, pair_beh, tracks, etc.
        self._analysis_start_vidframe = 0  # video frame where analysis begins (after late-placement trimming)
        self._data_popup = None

        self._build_ui()
        self._show_frame(0)
        self.show()

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
        self.view.setMinimumSize(480, 270)
        left_col = QVBoxLayout()
        left_col.setSpacing(3)
        left_col.addWidget(lbl_orig)
        left_col.addWidget(self.view, stretch=1)

        lbl_post = QLabel("Post-processed")
        lbl_post.setAlignment(Qt.AlignCenter)
        self.view_b = ROIView()
        self.view_b.setMinimumSize(480, 270)
        right_col = QVBoxLayout()
        right_col.setSpacing(3)
        right_col.addWidget(lbl_post)
        right_col.addWidget(self.view_b, stretch=1)

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

        # Spacer where ROI pixel info used to be
        main.addSpacing(30)

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
        self._export_worker = None
        self._export_thread = None

        self.setWindowTitle("ROI Selector")
        self.resize(1440, 620)

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

            self.view.set_frame(frame_orig)
            self.view_b.set_frame(frame_post)

            if not self._slider_dragging:
                self._slider.setValue(self._index)

            self._frame_label.setText(f"{self._index:04d} / {self._n_frames - 1:04d}")

            # Time display: counts from 0 starting at analysis start (15s trim)
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
        self._show_frame(v)

    def _on_slider_released(self):
        self._slider_dragging = False
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
        self._show_frame(self._index)

    # ---- Process -------------------------------------------------------

    def _run_process(self):
        if self._sleap_data is None:
            QMessageBox.warning(self, "No SLEAP data", "Load a SLEAP .h5 file first.")
            return

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
        self._processed_sleap_data = processed_data
        self._btn_process.setEnabled(True)
        self._btn_process.setText("Process")
        self._progress_bar.setVisible(False)
        self._precompute_analysis()
        self._show_frame(self._index)

    def _on_process_error(self, msg):
        self._btn_process.setEnabled(True)
        self._btn_process.setText("Process")
        self._progress_bar.setVisible(False)
        QMessageBox.critical(self, "Process failed", msg)

    # ---- Analysis precompute & overlay ---------------------------------

    def _precompute_analysis(self):
        """Cache kinematics, behaviors and features for overlay/popup use."""
        data = self._processed_sleap_data
        if data is None:
            return

        tracks     = data["tracks"]
        node_names = data["node_names"]
        track_names= data["track_names"]
        frame_map  = data["frame_map"]

        roi = self.view.roi_native()
        if roi is None:
            # Overlay cm-unit values are meaningless without a calibrated ROI.
            # The Process button requires an ROI, so this path is only reachable
            # in unusual edge cases (e.g. ROI cleared after processing).
            return
        (rx0, ry0), (rx1, ry1), side = roi
        px_per_cm = side / max(self._spin_arena_cm.value(), 1)

        # ---- Trim first 15 seconds (hand placement buffer) ----
        n_frames = tracks.shape[0]
        analysis_start = min(int(15 * self._fps), n_frames)
        # Find the first video frame at or after the analysis start
        valid_vframes = [vf for vf, si in frame_map.items() if si >= analysis_start]
        self._analysis_start_vidframe = min(valid_vframes) if valid_vframes else 0
        tracks = tracks[analysis_start:]
        frame_map = {vf: si - analysis_start
                     for vf, si in frame_map.items()
                     if si >= analysis_start}

        kin         = compute_kinematics(tracks, self._fps, node_names=node_names)
        single_beh  = compute_single_animal(tracks, kin, node_names, self._fps,
                                               px_per_cm=px_per_cm)
        pair_beh    = compute_pairwise(tracks, node_names, self._fps, dsr=None, kin=kin)
        track_feat, pair_feat = precompute_feature_arrays(
            tracks, kin, node_names, self._fps,
            roi=(rx0, ry0, rx1, ry1))

        # Resolve body-center node index
        body_idx = find_node_idx(node_names, 'body')

        self._analysis_cache = dict(
            tracks=tracks, kin=kin,
            single_beh=single_beh, pair_beh=pair_beh,
            track_feat=track_feat, pair_feat=pair_feat,
            node_names=node_names, track_names=track_names,
            frame_map=frame_map, body_idx=body_idx,
            inv_ppcm=1.0 / max(px_per_cm, 1e-9),
            px_per_cm=px_per_cm,
        )

        # Build / rebuild the data popup
        if self._data_popup is None:
            self._data_popup = _DataPopup(self)
        self._data_popup.setup(
            tracks, kin, single_beh, pair_beh,
            track_feat, pair_feat,
            node_names, track_names, self._fps, px_per_cm,
            analysis_start_vidframe=self._analysis_start_vidframe)

        self._btn_inspect.setEnabled(True)

    def _draw_analysis_overlay(self, frame, sleap_idx):
        """Draw per-track state / speed / heading + pair flags on frame_post."""
        c = self._analysis_cache
        if c is None:
            return frame

        tracks     = c['tracks']
        kin        = c['kin']
        sb         = c['single_beh']
        pb         = c['pair_beh']
        track_names= c['track_names']
        node_names = c['node_names']
        body_idx   = c['body_idx']
        inv_ppcm   = c['inv_ppcm']
        n_tracks   = tracks.shape[3]

        FONT  = cv2.FONT_HERSHEY_SIMPLEX
        FS    = 0.46
        THICK = 1
        # State colors in BGR
        SC = {
            'running':    (0,   80, 255),
            'turning':    (0,  200, 255),
            'walking':    (60, 200,  60),
            'stationary': (160, 160, 160),
        }

        lines = []   # list of (text, BGR)
        for t in range(n_tracks):
            tname = track_names[t]
            state = 'UNKNOWN'; sc = (200, 200, 200)
            for sk in ('running', 'turning', 'walking', 'stationary'):
                arr = sb.get(sk)
                if arr is not None and int(arr[sleap_idx, t]):
                    state = sk.upper(); sc = SC[sk]; break
            n = body_idx if body_idx is not None else 0
            spd = float(kin['speed'][sleap_idx, n, t]) * inv_ppcm
            hdg = float(kin['body_heading_deg'][sleap_idx, t])
            lines.append((f"[{tname}] {state}", sc))
            lines.append((f"  Spd:{spd:.1f}cm/s  Hdg:{hdg:.0f}deg", (200, 200, 200)))

        # Pair summary
        seen = set()
        for key in pb:
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
                continue
            n = body_idx if body_idx is not None else 0
            xA = float(tracks[sleap_idx, 0, n, tA]); yA = float(tracks[sleap_idx, 1, n, tA])
            xB = float(tracks[sleap_idx, 0, n, tB]); yB = float(tracks[sleap_idx, 1, n, tB])
            if np.isnan(xA) or np.isnan(xB):
                dist_str = "?"
            else:
                dist_str = f"{np.hypot(xA - xB, yA - yB) * inv_ppcm:.1f}cm"
            lines.append((f"[{nA} vs {nB}] dist:{dist_str}", (80, 160, 255)))
            flags = []
            for flag in ('NoseNose', 'Contact', 'Engaged'):
                arr = pb.get(f'{pfx}/{flag}')
                if arr is not None:
                    try:
                        v = arr[sleap_idx]
                        if bool(v) and not (isinstance(v, float) and np.isnan(v)):
                            flags.append(flag)
                    except Exception:
                        pass
            if flags:
                lines.append((f"  {' | '.join(flags)}", (0, 220, 120)))

        # Draw semi-transparent box + text
        line_h = 20; pad = 6
        box_h  = len(lines) * line_h + pad * 2
        box_w  = min(frame.shape[1] - 16,
                     max(240, max(len(t) * 7 for t, _ in lines) + pad * 2))
        ox, oy = 8, 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + box_w, oy + box_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, (text, color) in enumerate(lines):
            y = oy + pad + (i + 1) * line_h - 4
            cv2.putText(frame, text, (ox + 5, y), FONT, FS, color, THICK, cv2.LINE_AA)
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

        # Unified export dialog (checkboxes + save location)
        vp = Path(self._video_path)
        dlg = ExportOptionsDialog(
            graphs_available=_GRAPHS_AVAILABLE,
            default_dir=str(vp.parent),
            default_name=vp.stem,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
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
            processed_data=self._processed_sleap_data,
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
        self._btn_export.setEnabled(True)
        self._btn_export.setText("Export")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(100)
        self._progress_bar.setVisible(False)
        QMessageBox.information(self, "Done", msg)

    def _on_export_error(self, msg):
        """Called on the main thread when _ExportWorker raises an exception."""
        self._btn_export.setEnabled(True)
        self._btn_export.setText("Export")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        QMessageBox.critical(self, "Export failed", msg)

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
        super().closeEvent(event)
