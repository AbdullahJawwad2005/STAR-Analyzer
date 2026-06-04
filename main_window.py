import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from run_popup import RunPopUp
from sleap_loader import load_sleap


class MainWindow(QMainWindow):
    """Main application window for loading STAR video/tracking data."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STAR Analyzer")
        self.setMinimumSize(860, 440)

        self.video_path = None
        self.video_info = None
        self.sleap_data = None
        self.roi = None

        self._build_ui()

    def _center_on_screen(self):
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.resize(max(self.minimumWidth(), 860), max(self.minimumHeight(), 440))
        frame = self.frameGeometry()
        frame.moveCenter(geo.center())
        self.move(frame.topLeft())

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow { background: #eef3f5; }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d5e0e6;
                border-radius: 8px;
                margin-top: 12px;
                padding: 14px;
                font-weight: 700;
                color: #20323a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background: #1f6f8b;
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 10px 14px;
                font-weight: 700;
            }
            QPushButton:hover { background: #185a72; }
            QPushButton:pressed { background: #12475b; }
            QFrame#StatusCard {
                background: #ffffff;
                border: 1px solid #d5e0e6;
                border-radius: 8px;
            }
            QLabel#Title {
                color: #17272f;
                font-size: 24px;
                font-weight: 800;
            }
            QLabel#Subtitle {
                color: #5d7079;
                font-size: 13px;
            }
            QLabel#Status {
                color: #263840;
                font-size: 14px;
                line-height: 150%;
            }
        """)

        title = QLabel("STAR Analyzer")
        title.setObjectName("Title")
        subtitle = QLabel("Load video, validate SLEAP tracking, define ROIs, then run behavioral analysis.")
        subtitle.setObjectName("Subtitle")

        btn_video = QPushButton("Select Video")
        btn_h5 = QPushButton("Select .h5 File")
        btn_run = QPushButton("Open ROI Selector")
        for btn in (btn_video, btn_h5, btn_run):
            btn.setMinimumHeight(44)
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        btn_video.clicked.connect(self.select_video)
        btn_h5.clicked.connect(self.select_h5)
        btn_run.clicked.connect(self.open_run_popup)

        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)
        controls_layout.addWidget(btn_video)
        controls_layout.addWidget(btn_h5)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(btn_run)
        controls_layout.addStretch()

        controls = QGroupBox("Data Setup")
        controls.setLayout(controls_layout)

        status_frame = QFrame()
        status_frame.setObjectName("StatusCard")
        self.screen = QLabel("Ready.\nSelect a video to begin.")
        self.screen.setObjectName("Status")
        self.screen.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.screen.setWordWrap(True)

        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(18, 18, 18, 18)
        status_layout.addWidget(title)
        status_layout.addWidget(subtitle)
        status_layout.addSpacing(12)
        status_layout.addWidget(self.screen, stretch=1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(status_frame, stretch=1)

        outer_layout = QHBoxLayout()
        outer_layout.setContentsMargins(18, 18, 18, 18)
        outer_layout.setSpacing(14)
        outer_layout.addWidget(controls, stretch=0)
        outer_layout.addLayout(right_layout, stretch=1)

        container = QWidget()
        container.setLayout(outer_layout)
        self.setCentralWidget(container)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if not path:
            return

        try:
            self.video_info = self._validate_video(path)
            self.video_path = path
            info = self.video_info
            self.screen.setText(
                f"Video loaded: {path.split('/')[-1]}\n"
                f"Resolution: {info['width']} x {info['height']}\n"
                f"Frames: {info['frames']} | FPS: {info['fps']:.2f}\n"
                "Status: video validation passed."
            )
        except Exception as exc:
            self.video_path = None
            self.video_info = None
            self.screen.setText(f"Video validation failed:\n{exc}")

    def select_h5(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SLEAP analysis file",
            "",
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not path:
            return

        try:
            self.sleap_data = load_sleap(path)
            d = self.sleap_data
            self.screen.setText(
                f"H5 loaded: {path.split('/')[-1]}\n"
                f"Tracks: {len(d['track_names'])} | Nodes: {len(d['node_names'])}\n"
                f"Tracked frames: {len(d['frame_map'])}\n"
                f"Missing coordinate values: {d['nan_count']}\n"
                "Status: SLEAP validation passed."
            )
        except Exception as exc:
            self.sleap_data = None
            self.screen.setText(f"Failed to load H5:\n{exc}")

    def open_run_popup(self):
        if not self.video_path:
            self.screen.setText("Select and validate a video first.")
            return
        self._popup = RunPopUp(self.video_path, sleap_data=self.sleap_data)
        self._popup.roi_selected.connect(self._on_roi_selected)

    def _on_roi_selected(self, roi):
        self.roi = roi
        if roi:
            (x0, y0), (x1, y1) = roi
            self.screen.setText(f"ROI selected: ({x0}, {y0}) -> ({x1}, {y1})")
        else:
            self.screen.setText("No ROI drawn.")

    @staticmethod
    def _validate_video(path):
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise ValueError("OpenCV could not open this video file.")
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            if frames <= 0 or width <= 0 or height <= 0:
                raise ValueError("Video metadata is incomplete or invalid.")
            return {"frames": frames, "width": width, "height": height, "fps": fps}
        finally:
            cap.release()
