import gc
import logging
import sys
import traceback
import cv2

from PySide6.QtCore import Qt, QDateTime, QPointF, QRectF
from PySide6.QtGui import QGuiApplication, QFont, QIcon, QPixmap, QPainter, QPen, QBrush, QColor, QLinearGradient, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sleap_loader import load_sleap


def _render_mosiac_pixmap(sz: int) -> QPixmap:
    """Render a single MOSIAC icon pixmap — 2×2 mosaic tiles with tracking dots."""
    px = QPixmap(sz, sz)
    px.fill(QColor(0, 0, 0, 0))
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    s = float(sz)
    cr = s * 0.16

    bg = QLinearGradient(0, 0, s, s)
    bg.setColorAt(0.0, QColor(0x0d, 0x1b, 0x2a))
    bg.setColorAt(1.0, QColor(0x1a, 0x2e, 0x40))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(bg))
    p.drawRoundedRect(QRectF(0, 0, s, s), cr, cr)

    pad = s * 0.12; gap = s * 0.05
    tw = (s - 2 * pad - gap) / 2; th = tw
    tile_colors = [
        (QColor(0x00, 0x8b, 0xd4), QColor(0x00, 0x6a, 0xaa)),
        (QColor(0x00, 0xc8, 0x96), QColor(0x00, 0x9a, 0x70)),
        (QColor(0x7b, 0x5e, 0xff), QColor(0x59, 0x42, 0xcc)),
        (QColor(0xff, 0xb8, 0x1c), QColor(0xd4, 0x90, 0x00)),
    ]
    positions = [
        (pad, pad), (pad + tw + gap, pad),
        (pad, pad + th + gap), (pad + tw + gap, pad + th + gap),
    ]
    tr = s * 0.06
    for (tx, ty), (c1, c2) in zip(positions, tile_colors):
        tg = QLinearGradient(tx, ty, tx + tw, ty + th)
        tg.setColorAt(0.0, c1); tg.setColorAt(1.0, c2)
        p.setBrush(QBrush(tg)); p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(tx, ty, tw, th), tr, tr)

    dot_r = max(2.0, s * 0.052)
    cx_pt = QPointF(pad + tw, pad + th)
    tile_centers = [
        QPointF(pad + tw / 2,        pad + th / 2),
        QPointF(pad + tw + gap + tw / 2, pad + th / 2),
        QPointF(pad + tw / 2,        pad + th + gap + th / 2),
        QPointF(pad + tw + gap + tw / 2, pad + th + gap + th / 2),
    ]
    lp = QPen(QColor(255, 255, 255, 55))
    lp.setWidthF(max(1.0, s * 0.022)); lp.setCapStyle(Qt.RoundCap)
    p.setPen(lp)
    for tc in tile_centers:
        p.drawLine(cx_pt, tc)

    p.setPen(Qt.NoPen)
    for i, dp in enumerate([cx_pt] + tile_centers):
        rg = QRadialGradient(dp.x(), dp.y(), dot_r)
        if i == 0:
            rg.setColorAt(0.0, QColor(255, 255, 255, 240))
            rg.setColorAt(1.0, QColor(220, 220, 255, 100))
            r_use = dot_r
        else:
            rg.setColorAt(0.0, QColor(255, 255, 255, 210))
            rg.setColorAt(1.0, QColor(180, 220, 255, 80))
            r_use = dot_r * 0.75
        p.setBrush(QBrush(rg))
        p.drawEllipse(dp, r_use, r_use)

    p.end()
    return px


def _make_mosiac_icon() -> QIcon:
    """Return a multi-resolution QIcon for MOSIAC."""
    icon = QIcon()
    for sz in (16, 32, 48, 64, 128, 256):
        icon.addPixmap(_render_mosiac_pixmap(sz))
    return icon


# Keep old name as alias so any lingering references don't break at runtime.
_make_star_icon = _make_mosiac_icon


class _QtLogHandler(logging.Handler):
    """Forwards Python logging records to MainWindow.log()."""

    def __init__(self, window):
        super().__init__()
        self._window = window

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level_map = {
                logging.DEBUG:    'info',
                logging.INFO:     'info',
                logging.WARNING:  'warn',
                logging.ERROR:    'error',
                logging.CRITICAL: 'error',
            }
            level = level_map.get(record.levelno, 'info')
            self._window.log(msg, level)
        except Exception:
            pass


class _StderrRedirect:
    """Writes anything sent to stderr into MainWindow.log() as an error."""

    def __init__(self, window, original):
        self._window = window
        self._original = original
        self._buf = []

    def write(self, text: str) -> None:
        self._original.write(text)          # keep terminal copy
        stripped = text.rstrip('\n')
        if stripped:
            self._window.log(stripped, 'error')

    def flush(self) -> None:
        self._original.flush()

    def fileno(self):                       # needed by some libraries
        return self._original.fileno()


class MainWindow(QMainWindow):
    """Main application window for MOSIAC — multi-animal behavioral analysis."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOSIAC")
        self.setWindowIcon(_make_mosiac_icon())
        self.setMinimumSize(860, 440)

        self.video_path = None
        self.video_info = None
        self.sleap_data = None
        self.roi = None
        self._popups: list = []

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

    def setup_debug_routing(self) -> None:
        """Route Python logging + stderr + unhandled exceptions into the debug panel."""
        # Logging handler
        handler = _QtLogHandler(self)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)

        # stderr redirect
        sys.stderr = _StderrRedirect(self, sys.stderr)

        # Unhandled exception hook
        def _excepthook(exc_type, exc_value, exc_tb):
            lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            full = "".join(lines).strip()
            self.log(full, 'error')
            # also keep default behaviour (prints to terminal)
            sys.__excepthook__(exc_type, exc_value, exc_tb)

        sys.excepthook = _excepthook

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
            QFrame#StatusCard { background:#1c2529; border:1px solid #2a3a40; border-radius:8px; }
            QPlainTextEdit#DebugTerminal { background:#1c2529; color:#a8c8d0; border:none;
                                           padding:6px; selection-background-color:#2a4a54; }
            QLabel#Title   { color:#c8e8f0; font-size:24px; font-weight:800; }
            QLabel#Subtitle{ color:#6a8a94; font-size:13px; }
        """)

        title = QLabel("MOSIAC")
        title.setObjectName("Title")
        subtitle = QLabel("Multi-animal pose tracking and behavioral analysis. Load a video and SLEAP file to begin.")
        subtitle.setObjectName("Subtitle")

        btn_video = QPushButton("Select Video")
        btn_h5 = QPushButton("Select .h5 File")
        btn_run = QPushButton("Open Analysis Session")
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
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        _f = QFont("Consolas"); _f.setStyleHint(QFont.Monospace); _f.setPointSize(10)
        self._log.setFont(_f)
        self._log.setObjectName("DebugTerminal")

        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(18, 18, 18, 18)
        status_layout.addWidget(title)
        status_layout.addWidget(subtitle)
        status_layout.addSpacing(12)
        status_layout.addWidget(self._log, stretch=1)
        self.log("MOSIAC ready. Select a video to begin.")

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

    def log(self, message: str, level: str = 'info') -> None:
        COLORS = {'info': '#a8c8d0', 'ok': '#43d692', 'warn': '#fad165', 'error': '#e66550'}
        TAGS   = {'info': 'INFO', 'ok': 'OK  ', 'warn': 'WARN', 'error': 'ERR '}
        color  = COLORS.get(level, COLORS['info'])
        tag    = TAGS.get(level,   TAGS['info'])
        ts     = QDateTime.currentDateTime().toString("hh:mm:ss")
        esc    = (message.replace("&", "&amp;").replace("<", "&lt;")
                         .replace(">", "&gt;").replace("\n", "<br>"))
        self._log.appendHtml(
            f'<span style="color:#4a6a74">[{ts}]</span> '
            f'<span style="color:{color}">[{tag}] {esc}</span>'
        )
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

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
            self.log(path.split('/')[-1], 'ok')
            dur = info['frames'] / info['fps'] if info['fps'] else 0
            self.log(
                f"{info['width']} x {info['height']}  |  {info['frames']} frames  |  "
                f"{info['fps']:.2f} fps  |  {dur:.1f}s",
                'info'
            )
        except Exception as exc:
            self.video_path = None
            self.video_info = None
            self.log(f"Video validation failed: {exc}", 'error')

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
            self.log(path.split('/')[-1], 'ok')
            self.log(f"Tracks: {len(d['track_names'])}  |  Nodes: {len(d['node_names'])}  |  Tracked frames: {len(d['frame_map'])}", 'info')
            self.log(f"Missing coordinate values: {d['nan_count']}", 'info')
        except Exception as exc:
            self.sleap_data = None
            self.log(f"Failed to load SLEAP file: {exc}", 'error')

    @property
    def _popup(self):
        return self._popups[-1] if self._popups else None

    def open_run_popup(self):
        if not self.video_path:
            self.log("Select and validate a video first.", 'warn')
            return
        from run_popup import RunPopUp
        popup = RunPopUp(self.video_path, sleap_data=self.sleap_data)
        popup.roi_selected.connect(self._on_roi_selected)
        popup.destroyed.connect(
            lambda p=popup: self._popups.remove(p) if p in self._popups else None
        )
        self._popups.append(popup)

    def _on_roi_selected(self, roi):
        self.roi = roi
        if roi:
            (x0, y0), (x1, y1), side = roi
            self.log(f"ROI confirmed: TL ({x0},{y0})  BR ({x1},{y1})  W {side}px", 'ok')
        else:
            self.log("ROI cleared — no region defined.", 'warn')

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
