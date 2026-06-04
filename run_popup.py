import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer, QDateTime
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
)
from PySide6.QtCore import QRectF

from roi_view import ROIView


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
        self._spin_width = QSpinBox()
        self._spin_width.setRange(1, 9999)
        self._spin_width.setValue(200)
        self._spin_width.setFixedWidth(70)
        controls_row.addWidget(self._spin_width)
        self._btn_clear   = QPushButton("Clear")
        self._btn_confirm = QPushButton("Confirm")
        for btn in (self._btn_clear, self._btn_confirm):
            controls_row.addWidget(btn)
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
        self._spin_width.setValue(x1 - x0)

    def _clear_roi_and_labels(self):
        self.view.clear_roi()
        self.view_b.clear_roi()
        self._lbl_tl.setText("TL:  —")
        self._lbl_br.setText("BR:  —")

    def _confirm_roi(self):
        self.roi_selected.emit(self.view.roi_native())
        self.close()

    def closeEvent(self, event):
        self._timer.stop()
        self._cap.release()
        super().closeEvent(event)
