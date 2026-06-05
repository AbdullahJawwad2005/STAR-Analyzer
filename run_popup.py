import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer, QDateTime
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
)
from PySide6.QtCore import QRectF

from roi_view import ROIView


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
            QFrame#RoiInfo { background:#222; border:1px solid #383838; border-radius:4px; }
            QFrame#RoiInfo QLabel { color:#00d4f0; font:12px 'Consolas','Courier New',monospace; }
        """)

        self._cap        = cv2.VideoCapture(video_path)
        self._n_frames   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self._index      = 0
        self._sleap_data = sleap_data

        self._fps             = float(self._cap.get(cv2.CAP_PROP_FPS) or 30)
        self._timer           = QTimer(self)
        self._timer.setInterval(min(1000, max(1, int(1000 / self._fps))))
        self._timer.timeout.connect(self._advance_frame)
        self._slider_dragging = False
        self._zones     = None   # dict zone_name → (x0,y0,x1,y1) native px, or None
        self._px_per_cm = None   # float, computed when ROI + cm spinbox are both valid

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
        progress_row.addWidget(self._slider)
        progress_row.addWidget(self._frame_label)
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
        self._btn_clear   = QPushButton("Clear")
        self._btn_confirm = QPushButton("Confirm")
        for btn in (self._btn_clear, self._btn_confirm):
            controls_row.addWidget(btn)
        self._btn_export = QPushButton("Analyze && Export")
        controls_row.addWidget(self._btn_export)
        main.addLayout(controls_row)

        # ROI info frame
        self._roi_info_frame = QFrame()
        self._roi_info_frame.setObjectName("RoiInfo")
        roi_info_layout = QVBoxLayout(self._roi_info_frame)
        roi_info_layout.setContentsMargins(8, 6, 8, 6)
        roi_info_layout.setSpacing(2)
        self._lbl_tl = QLabel("TL:  —")
        self._lbl_br = QLabel("BR:  —")
        roi_info_layout.addWidget(self._lbl_tl)
        roi_info_layout.addWidget(self._lbl_br)
        main.addWidget(self._roi_info_frame)

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
        self._btn_export.clicked.connect(self._run_export)
        self.view.roi_changed.connect(self._on_roi_changed)
        self.view_b.roi_changed.connect(self._on_roi_changed)

        self.setWindowTitle("ROI Selector")
        self.resize(1440, 620)

    # ---- Frame display -------------------------------------------------

    def _show_frame(self, index):
        index = max(0, min(index, self._n_frames - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if ok:
            self._index = index
            if self._sleap_data is not None:
                frame = self._draw_sleap(frame.copy(), index)
            if self._zones:
                frame = self._draw_zones(frame)
            self.view.set_frame(frame)
            self.view_b.set_frame(frame)
            if not self._slider_dragging:
                self._slider.setValue(self._index)
            self._frame_label.setText(f"{self._index:04d} / {self._n_frames - 1:04d}")

    # ---- SLEAP skeleton overlay ----------------------------------------

    def _draw_sleap(self, frame, video_frame_idx):
        """Draw skeleton edges and keypoint nodes for the given video frame."""
        frame_map = self._sleap_data["frame_map"]
        if video_frame_idx not in frame_map:
            return frame

        sleap_idx = frame_map[video_frame_idx]
        tracks    = self._sleap_data["tracks"]    # (n_frames, 2, n_nodes, n_tracks)
        edge_inds = self._sleap_data["edge_inds"] # (n_edges, 2)
        n_tracks  = tracks.shape[3]

        for t in range(n_tracks):
            color = self._TRACK_COLORS[t % len(self._TRACK_COLORS)]
            pts   = tracks[sleap_idx, :, :, t]   # (2, n_nodes) — row0=x, row1=y

            # Skeleton edges
            for src, dst in edge_inds:
                x0, y0 = pts[0, src], pts[1, src]
                x1, y1 = pts[0, dst], pts[1, dst]
                if not any(np.isnan([x0, y0, x1, y1])):
                    cv2.line(frame,
                             (int(x0), int(y0)), (int(x1), int(y1)),
                             color, 2, cv2.LINE_AA)

            # Keypoint nodes
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
        self._lbl_tl.setText(f"TL:  ({x0}, {y0})")
        self._lbl_br.setText(f"BR:  ({x1}, {y1})")
        self._lbl_width.setText(f"{x1 - x0} px")
        self._recompute_zones()
        self._show_frame(self._index)

    def _clear_roi_and_labels(self):
        self.view.clear_roi()
        self.view_b.clear_roi()
        self._lbl_tl.setText("TL:  —")
        self._lbl_br.setText("BR:  —")
        self._lbl_width.setText("—")
        self._zones = None
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
        s = int(round(8 * self._px_per_cm))   # 8 cm strip in pixels
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

    # ---- Export --------------------------------------------------------

    def _run_export(self):
        roi = self.view.roi_native()
        if roi is None:
            QMessageBox.warning(self, "No ROI", "Draw an ROI first."); return
        if self._sleap_data is None:
            QMessageBox.warning(self, "No SLEAP data", "Load a SLEAP .h5 file first."); return

        (rx0, ry0), (rx1, ry1), side = roi
        arena_cm = self._spin_arena_cm.value()
        strip    = 8 * (side / arena_cm)
        px_per_cm = side / arena_cm

        tracks      = self._sleap_data["tracks"]       # (n_frames, 2, n_nodes, n_tracks)
        frame_map   = self._sleap_data["frame_map"]    # video_frame → sleap_idx
        node_names  = self._sleap_data["node_names"]
        track_names = self._sleap_data["track_names"]

        # ---- Build main data table ----
        rows = []
        for vid_frame, sleap_idx in sorted(frame_map.items()):
            time_s = round(vid_frame / self._fps, 4)
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
                        "Frame":       vid_frame,
                        "Time (s)":    time_s,
                        "Track":       track_names[t],
                        "Body Part":   node,
                        "X (px)":      round(x, 2) if not np.isnan(x) else float("nan"),
                        "Y (px)":      round(y, 2) if not np.isnan(y) else float("nan"),
                        "X (cm)":      x_cm,
                        "Y (cm)":      y_cm,
                        "Zone":        zone,
                    })

        df = pd.DataFrame(rows)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save zone analysis", "", "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return

        if path.endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            # ---- Zone summary (unique frames per track/zone, NaN/Undetected excluded) ----
            detected = df[df["Zone"] != "Undetected"]
            summary = (
                detected.drop_duplicates(subset=["Frame", "Track", "Zone"])
                .groupby(["Track", "Zone"], sort=False)
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
                ("Zone border strip",    "8 cm"),
                ("", ""),
                ("--- Session Stats ---", ""),
                ("Total tracked frames", total_frames),
                ("Video FPS",            round(self._fps, 4)),
                ("Tracks",               ", ".join(track_names)),
                ("Body parts",           ", ".join(node_names)),
                ("Total data rows",      len(df)),
            ]
            info_df = pd.DataFrame(info_rows, columns=["Parameter", "Value"])

            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer,      sheet_name="Tracking Data", index=False)
                summary.to_excel(writer, sheet_name="Zone Summary",  index=False)
                info_df.to_excel(writer, sheet_name="Session Info",  index=False)

        QMessageBox.information(self, "Done", f"Exported {len(rows):,} rows to:\n{path}")

    def closeEvent(self, event):
        self._timer.stop()
        self._cap.release()
        super().closeEvent(event)
