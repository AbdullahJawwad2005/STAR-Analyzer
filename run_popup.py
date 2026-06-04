import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
)

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

    roi_selected = Signal(object)  # emits ((x0,y0),(x1,y1)) or None on Apply

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
            QWidget { background: #1a1a1a; }
            QPushButton {
                background: #2d2d2d; color: #eee; border: 1px solid #444;
                border-radius: 4px; padding: 8px; min-width: 110px;
            }
            QPushButton:hover { background: #3d3d3d; }
            QFrame#video { background: #000; border: 1px solid #333; }
            QLabel { color: #888; }
        """)

        self._cap        = cv2.VideoCapture(video_path)
        self._n_frames   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self._index      = 0
        self._sleap_data = sleap_data

        self._build_ui()
        self._show_frame(0)
        self.show()

    # ---- UI construction -----------------------------------------------

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # Sidebar buttons
        sidebar = QVBoxLayout()
        sidebar.setSpacing(6)
        sidebar.setAlignment(Qt.AlignTop)
        for label, slot in (
            ("◀ Prev",    self.prev_frame),
            ("▶ Next",    self.next_frame),
            ("Clear ROI", self.clear_roi),
            ("Apply",     self.apply_roi),
        ):
            btn = QPushButton(label)
            btn.setFixedHeight(38)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.clicked.connect(slot)
            sidebar.addWidget(btn, alignment=Qt.AlignHCenter)
        sidebar.addStretch()

        # Video panel
        video_frame = QFrame()
        video_frame.setObjectName("video")
        self.view = ROIView()
        self.view.setMinimumSize(640, 360)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(self.view)

        outer.addLayout(sidebar, stretch=0)
        outer.addWidget(video_frame, stretch=1)

        self.setWindowTitle("ROI Selector")
        self.resize(900, 520)

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

    # ---- Button handlers -----------------------------------------------

    def prev_frame(self):
        self._show_frame(self._index - 1)

    def next_frame(self):
        self._show_frame(self._index + 1)

    def clear_roi(self):
        self.view.clear_roi()

    def apply_roi(self):
        self.roi_selected.emit(self.view.roi_native())
        self.close()

    def closeEvent(self, event):
        self._cap.release()
        super().closeEvent(event)
