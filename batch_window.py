"""batch_window.py — Batch processing UI for MOSIAC.

Allows users to queue multiple video+h5 pairs, draw an ROI once, then run
the full analysis+export pipeline unattended with fine-grained progress.
"""

import time
import traceback
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QObject, QThread, QTimer, QDateTime, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget,
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QPlainTextEdit,
    QFileDialog,
    QMessageBox,
    QLineEdit,
    QSizePolicy,
)

from roi_view import ROIView


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


# ──────────────────────────────────────────────────────────────────────────────
# ROI dialog
# ──────────────────────────────────────────────────────────────────────────────

class _BatchROIDialog(QDialog):
    """Embed ROIView loaded from the first video's first frame."""

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Arena ROI")
        self.setMinimumSize(720, 540)

        layout = QVBoxLayout(self)
        lbl = QLabel(
            "Drag to draw the arena ROI (square). "
            "This ROI will be applied to all jobs in the batch."
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self._view = ROIView(self)
        layout.addWidget(self._view, stretch=1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Use ROI")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._roi = None
        self._load_frame(video_path)

    def _load_frame(self, path: str):
        cap = cv2.VideoCapture(path)
        try:
            ok, frame = cap.read()
            if ok and frame is not None:
                self._view.set_frame(frame)
        finally:
            cap.release()

    def _on_accept(self):
        roi = self._view.roi_native()
        if roi is None:
            QMessageBox.warning(self, "No ROI",
                                "Please draw a ROI before clicking OK.")
            return
        self._roi = roi
        self.accept()

    def roi(self):
        """Return ``((x0,y0),(x1,y1), side)`` or ``None``."""
        return self._roi


# ──────────────────────────────────────────────────────────────────────────────
# Batch worker
# ──────────────────────────────────────────────────────────────────────────────

class _BatchWorker(QObject):
    """Runs on a QThread; processes jobs sequentially with granular progress."""

    # emitted once per job when SLEAP is loaded and we know the shape
    job_started   = Signal(int, str, int, int, float)  # idx, name, n_frames, n_tracks, fps

    # fine-grained within-job progress: (idx, 0-100, step_label)
    step_progress = Signal(int, int, str)

    # terminal states
    job_done      = Signal(int, str, float)   # idx, out_path, elapsed_s
    job_error     = Signal(int, str, float)   # idx, traceback_text, elapsed_s

    # summary when every job is processed
    all_done      = Signal(int, int)          # n_ok, n_err

    def __init__(self, jobs, roi, arena_cm, strip_cm, output_opts, parent=None):
        super().__init__(parent)
        self._jobs        = jobs
        self._roi         = roi
        self._arena_cm    = arena_cm
        self._strip_cm    = strip_cm
        self._output_opts = output_opts or {}

    def run(self):
        from sleap_loader import load_sleap
        from preprocessing import fill_and_smooth_tracks
        from run_popup import _run_analysis, _ExportWorker

        roi      = self._roi
        arena_cm = self._arena_cm
        strip_cm = self._strip_cm
        opts     = self._output_opts
        (_, _), (_, _), side = roi
        px_per_cm = side / max(arena_cm, 1)

        n_ok = n_err = 0

        for i, (video_path, h5_path, out_path) in enumerate(self._jobs):
            t0   = time.monotonic()
            name = Path(video_path).name
            try:
                # ── 2% : video metadata ──────────────────────────────────
                self.step_progress.emit(i, 2, "Reading video metadata…")
                cap = cv2.VideoCapture(video_path)
                fps            = float(cap.get(cv2.CAP_PROP_FPS) or 30)
                n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # ── 5% : load SLEAP ──────────────────────────────────────
                self.step_progress.emit(i, 5, "Loading SLEAP file…")
                sleap_data = load_sleap(h5_path)
                n_tracks   = sleap_data["tracks"].shape[3]
                # Now we know shape — tell UI so it can log metadata
                self.job_started.emit(i, name, n_video_frames, n_tracks, fps)

                # ── 8–55% : fill & smooth (fine-grained via callback) ────
                # fill_and_smooth_tracks calls progress_callback(0..100)
                # Map that linearly to 8..55 in our overall 0-100 scale.
                def _smooth_cb(pct, _i=i):
                    mapped = 8 + int(pct * 0.47)   # 0→8, 100→55
                    self.step_progress.emit(
                        _i, mapped, f"Smoothing tracks… {pct}%")

                self.step_progress.emit(i, 8, "Smoothing & filling tracks…")
                processed = dict(sleap_data)
                processed["tracks"] = fill_and_smooth_tracks(
                    sleap_data["tracks"], fps=fps,
                    progress_callback=_smooth_cb)

                # ── 56–79% : analysis (monolithic, emit at start + end) ──
                self.step_progress.emit(i, 56, "Computing kinematics & behaviors…")
                cache = _run_analysis(processed, fps, roi, px_per_cm, strip_cm, opts)
                self.step_progress.emit(i, 79, "Analysis complete — preparing export…")

                # ── 80–99% : export (ExportWorker emits status strings) ──
                self.step_progress.emit(i, 80, "Writing outputs…")
                exp = _ExportWorker(
                    cache, fps, roi, arena_cm, strip_cm, out_path,
                    arena_snapshot=None, export_opts=opts)
                # Route ExportWorker's status strings into our step labels.
                # Both objects are on the worker thread here → direct connection.
                exp.status.connect(
                    lambda msg, _i=i: self.step_progress.emit(_i, 85, msg))
                exp.run()
                self.step_progress.emit(i, 100, "Done")

                elapsed = time.monotonic() - t0
                self.job_done.emit(i, out_path, elapsed)
                n_ok += 1

            except Exception:
                elapsed = time.monotonic() - t0
                self.job_error.emit(i, traceback.format_exc(), elapsed)
                n_err += 1

        self.all_done.emit(n_ok, n_err)


# ──────────────────────────────────────────────────────────────────────────────
# Table column indices
# ──────────────────────────────────────────────────────────────────────────────

_COL_VIDEO  = 0
_COL_H5     = 1
_COL_OUTPUT = 2
_COL_STATUS = 3


# ──────────────────────────────────────────────────────────────────────────────
# Main BatchWindow
# ──────────────────────────────────────────────────────────────────────────────

class BatchWindow(QWidget):
    """Top-level window for queuing and running batch MOSIAC jobs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MOSIAC — Batch Analysis")
        self.setMinimumSize(1020, 660)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._roi: tuple | None      = None
        self._thread: QThread | None = None
        self._worker: _BatchWorker | None = None

        self._batch_start: float = 0.0
        self._job_start:   float = 0.0

        # Ticks every second to update the elapsed label while running
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick_elapsed)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget { background: #f0f4f6; color: #1a2830; font-size: 13px; }
            QPushButton {
                background: #1f6f8b; color: white; border: 0;
                border-radius: 5px; padding: 7px 14px; font-weight: 700;
            }
            QPushButton:hover    { background: #185a72; }
            QPushButton:pressed  { background: #12475b; }
            QPushButton:disabled { background: #8aafbe; color: #c8dde5; }
            QTableWidget {
                background: white; gridline-color: #d0dde3;
                alternate-background-color: #f4f8fa;
            }
            QHeaderView::section {
                background: #d0e0e8; font-weight: 700; padding: 5px;
                border: none; border-right: 1px solid #b5cad4;
            }
            QPlainTextEdit {
                background: #141e22; color: #a8c8d0;
                font-family: Consolas, monospace; font-size: 10px;
                border: none;
            }
            QFrame#ProgCard {
                background: white; border: 1px solid #c8d8e0;
                border-radius: 7px;
            }
            QProgressBar {
                border: 1px solid #b0c8d4; border-radius: 4px;
                background: #e4eef2; text-align: center;
                min-height: 16px; max-height: 16px;
            }
            QProgressBar#BarStep::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1f6f8b, stop:1 #29a8d0);
                border-radius: 3px;
            }
            QProgressBar#BarOverall::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a8a50, stop:1 #30cc78);
                border-radius: 3px;
            }
            QSpinBox { padding: 3px 6px; border: 1px solid #c0d0d8;
                       border-radius: 4px; background: white; }
            QLineEdit { padding: 3px 6px; border: 1px solid #c0d0d8;
                        border-radius: 4px; background: white; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(9)

        # ── Job table ──────────────────────────────────────────────────────
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Video", "H5 File", "Output", "Status / Step"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.Fixed)
        hh.resizeSection(3, 210)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        root.addWidget(self._table, stretch=1)

        # ── File management buttons ────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_add_pairs  = QPushButton("Add Pairs…")
        self._btn_add_folder = QPushButton("Add Folder…")
        self._btn_remove     = QPushButton("Remove Selected")
        for b in (self._btn_add_pairs, self._btn_add_folder, self._btn_remove):
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._btn_add_pairs.clicked.connect(self._add_pairs)
        self._btn_add_folder.clicked.connect(self._add_folder_dialog)
        self._btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(self._btn_add_pairs)
        btn_row.addWidget(self._btn_add_folder)
        btn_row.addWidget(self._btn_remove)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # ── Settings / ROI / output dir (single compact row) ──────────────
        cfg = QHBoxLayout()
        cfg.setSpacing(12)

        cfg.addWidget(QLabel("Arena (cm):"))
        self._spin_arena = QSpinBox()
        self._spin_arena.setRange(1, 9999); self._spin_arena.setValue(40)
        cfg.addWidget(self._spin_arena)

        cfg.addWidget(QLabel("Strip (cm):"))
        self._spin_strip = QSpinBox()
        self._spin_strip.setRange(0, 999); self._spin_strip.setValue(8)
        cfg.addWidget(self._spin_strip)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #c0d0d8;")
        cfg.addWidget(sep)

        self._lbl_roi = QLabel("ROI: not set")
        self._lbl_roi.setStyleSheet("color: #b03030; font-weight: bold;")
        cfg.addWidget(self._lbl_roi)
        self._btn_roi = QPushButton("Set ROI…")
        self._btn_roi.setEnabled(False)
        self._btn_roi.clicked.connect(self._set_roi)
        cfg.addWidget(self._btn_roi)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #c0d0d8;")
        cfg.addWidget(sep2)

        cfg.addWidget(QLabel("Output dir:"))
        self._edit_outdir = QLineEdit()
        self._edit_outdir.setPlaceholderText("same folder as each video")
        cfg.addWidget(self._edit_outdir, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_browse.clicked.connect(self._browse_outdir)
        cfg.addWidget(btn_browse)

        root.addLayout(cfg)

        # ── Progress card ──────────────────────────────────────────────────
        prog_card = QFrame()
        prog_card.setObjectName("ProgCard")
        pc = QVBoxLayout(prog_card)
        pc.setContentsMargins(12, 10, 12, 10)
        pc.setSpacing(6)

        # Current job name + live elapsed
        hdr = QHBoxLayout()
        self._lbl_current = QLabel("No job running")
        self._lbl_current.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #123a4a;")
        self._lbl_elapsed = QLabel("")
        self._lbl_elapsed.setStyleSheet(
            "color: #5a7a88; font-size: 11px;")
        self._lbl_elapsed.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr.addWidget(self._lbl_current, stretch=1)
        hdr.addWidget(self._lbl_elapsed)
        pc.addLayout(hdr)

        # Current step description
        self._lbl_step = QLabel("Ready — add files and set the ROI to begin.")
        self._lbl_step.setStyleSheet("color: #3a6a7a; font-size: 11px;")
        pc.addWidget(self._lbl_step)

        # Per-job bar (0-100 with fine-grained updates)
        r1 = QHBoxLayout()
        lj = QLabel("Job:")
        lj.setStyleSheet("color: #5a7a88; font-size: 11px;")
        lj.setFixedWidth(52)
        self._bar_step = QProgressBar()
        self._bar_step.setObjectName("BarStep")
        self._bar_step.setRange(0, 100)
        self._bar_step.setValue(0)
        self._bar_step.setFormat("%p%")
        r1.addWidget(lj); r1.addWidget(self._bar_step)
        pc.addLayout(r1)

        # Overall jobs bar
        r2 = QHBoxLayout()
        lo = QLabel("Total:")
        lo.setStyleSheet("color: #5a7a88; font-size: 11px;")
        lo.setFixedWidth(52)
        self._bar_overall = QProgressBar()
        self._bar_overall.setObjectName("BarOverall")
        self._bar_overall.setRange(0, 1)
        self._bar_overall.setValue(0)
        self._bar_overall.setFormat("0 / 0 jobs")
        r2.addWidget(lo); r2.addWidget(self._bar_overall)
        pc.addLayout(r2)

        # Run button (right-aligned inside card)
        rb = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run Batch")
        self._btn_run.setMinimumHeight(36)
        self._btn_run.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._btn_run.clicked.connect(self._run_batch)
        rb.addStretch(); rb.addWidget(self._btn_run)
        pc.addLayout(rb)

        root.addWidget(prog_card)

        # ── Activity log ───────────────────────────────────────────────────
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(3000)
        self._log.setFixedHeight(190)
        _f = QFont("Consolas")
        _f.setStyleHint(QFont.Monospace)
        _f.setPointSize(9)
        self._log.setFont(_f)
        root.addWidget(self._log)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ts(self) -> str:
        return QDateTime.currentDateTime().toString("hh:mm:ss")

    def _log_msg(self, msg: str, color: str = "#a8c8d0"):
        ts  = self._ts()
        esc = (msg.replace("&", "&amp;")
                  .replace("<", "&lt;")
                  .replace(">", "&gt;")
                  .replace("\n", "<br>"))
        self._log.appendHtml(
            f'<span style="color:#3a5a64">[{ts}]</span> '
            f'<span style="color:{color}">{esc}</span>'
        )
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum())

    def _n_jobs(self) -> int:
        return self._table.rowCount()

    def _set_row_status(self, row: int, text: str,
                        fg: str | None = None,
                        bg: QColor | None = None):
        item = self._table.item(row, _COL_STATUS)
        if item is None:
            item = QTableWidgetItem()
            self._table.setItem(row, _COL_STATUS, item)
        item.setText(text)
        item.setTextAlignment(Qt.AlignCenter)
        if fg:
            item.setForeground(QColor(fg))
        if bg is not None:
            item.setBackground(bg)

    def _add_row(self, video_path: str, h5_path: str):
        """Insert a row — stores full paths in tooltips, shows filenames."""
        row = self._table.rowCount()
        self._table.insertRow(row)
        vi = QTableWidgetItem(Path(video_path).name)
        vi.setToolTip(video_path)
        hi = QTableWidgetItem(Path(h5_path).name)
        hi.setToolTip(h5_path)
        oi = QTableWidgetItem("")  # filled at run-time
        self._table.setItem(row, _COL_VIDEO,  vi)
        self._table.setItem(row, _COL_H5,     hi)
        self._table.setItem(row, _COL_OUTPUT, oi)
        self._set_row_status(row, "Pending", fg="#8aabba")
        self._update_roi_btn()

    def _update_roi_btn(self):
        self._btn_roi.setEnabled(self._n_jobs() > 0)

    def _resolve_out_path(self, video_path: str) -> str:
        stem     = Path(video_path).stem
        override = self._edit_outdir.text().strip()
        out_dir  = Path(override) if override else Path(video_path).parent
        candidate = out_dir / f"{stem}_analysis.xlsx"
        if not candidate.exists():
            return str(candidate)
        idx = 2
        while True:
            candidate = out_dir / f"{stem}_analysis_{idx}.xlsx"
            if not candidate.exists():
                return str(candidate)
            idx += 1

    def _tick_elapsed(self):
        job   = time.monotonic() - self._job_start
        total = time.monotonic() - self._batch_start
        self._lbl_elapsed.setText(
            f"job {_fmt_elapsed(job)}  ·  total {_fmt_elapsed(total)}")

    # ── File management ───────────────────────────────────────────────────────

    def _add_pairs(self):
        videos, _ = QFileDialog.getOpenFileNames(
            self, "Select video file(s)", "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)")
        if not videos:
            return
        for vp in videos:
            h5, _ = QFileDialog.getOpenFileName(
                self, f"Select H5 for {Path(vp).name}", "",
                "HDF5 files (*.h5 *.hdf5);;All files (*)")
            if h5:
                self._add_row(vp, h5)
            else:
                self._log_msg(
                    f"Skipped (no H5 chosen): {Path(vp).name}", "#fad165")

    def _add_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder with video+h5 pairs")
        if folder:
            self._add_folder(folder)

    def _add_folder(self, folder: str):
        folder_path = Path(folder)
        exts = {".mp4", ".avi", ".mov", ".mkv"}

        # Collect all videos and h5 files recursively in one pass each
        videos = sorted(p for p in folder_path.rglob("*")
                        if p.is_file() and p.suffix.lower() in exts)
        all_h5 = [p for p in folder_path.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}]

        added = 0
        for vp in videos:
            h5 = self._match_h5(vp, all_h5)
            rel = vp.relative_to(folder_path)
            if h5:
                self._add_row(str(vp), str(h5))
                added += 1
            else:
                self._log_msg(f"No .h5 found for {rel} — skipped", "#fad165")

        if added:
            self._log_msg(
                f"Added {added} pair(s) from {folder_path.name} (searched subfolders)",
                "#43d692")

    def _match_h5(self, video_path: Path, all_h5: list) -> "Path | None":
        """Return the best-matching h5 for *video_path* from *all_h5*.

        Priority (high → low):
        1. Exact stem match in the same directory
        2. H5 stem starts with video stem, same directory (shortest stem wins)
        3. Exact stem match anywhere in the tree
        4. H5 stem starts with video stem, anywhere (shortest stem wins)

        The prefix rule handles SLEAP's common convention of appending
        suffixes to the video name, e.g.
            video.mp4  →  video.mp4.analysis.h5  (stem = "video.mp4.analysis")
            video.mp4  →  video_analysis.h5       (stem = "video_analysis")
        """
        stem_lower = video_path.stem.lower()
        vdir = video_path.parent

        for pool in ([h for h in all_h5 if h.parent == vdir], all_h5):
            exact  = [h for h in pool if h.stem.lower() == stem_lower]
            if exact:
                return exact[0]
            prefix = [h for h in pool if h.stem.lower().startswith(stem_lower)]
            if prefix:
                return min(prefix, key=lambda h: len(h.stem))

        return None

    def _remove_selected(self):
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()},
            reverse=True)
        for r in rows:
            self._table.removeRow(r)
        self._update_roi_btn()

    # ── ROI ───────────────────────────────────────────────────────────────────

    def _set_roi(self):
        if self._n_jobs() == 0:
            return
        video_path = self._table.item(0, _COL_VIDEO).toolTip()
        dlg = _BatchROIDialog(video_path, self)
        if dlg.exec() == QDialog.Accepted:
            self._roi = dlg.roi()
            if self._roi:
                (x0, y0), (x1, y1), side = self._roi
                self._lbl_roi.setText(f"ROI ✓  {side}×{side} px")
                self._lbl_roi.setStyleSheet(
                    "color: #1a7a40; font-weight: bold;")
                self._log_msg(
                    f"ROI set — TL ({x0},{y0})  BR ({x1},{y1})  side={side}px",
                    "#43d692")

    def _browse_outdir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select output directory")
        if folder:
            self._edit_outdir.setText(folder)

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run_batch(self):
        if self._n_jobs() == 0:
            QMessageBox.warning(self, "No jobs",
                                "Add at least one video+H5 pair first.")
            return
        if self._roi is None:
            QMessageBox.warning(self, "No ROI",
                                "Draw the arena ROI before running.")
            return
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "A batch is already running.")
            return

        from run_popup import ProcessingOutputDialog
        dlg = ProcessingOutputDialog(n_tracks=2, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        output_opts = dlg.options()

        # Resolve output paths and populate table Output column
        jobs = []
        for row in range(self._n_jobs()):
            vp  = self._table.item(row, _COL_VIDEO).toolTip()
            h5  = self._table.item(row, _COL_H5).toolTip()
            out = self._resolve_out_path(vp)
            oi  = self._table.item(row, _COL_OUTPUT)
            oi.setText(Path(out).name); oi.setToolTip(out)
            self._set_row_status(row, "Queued", fg="#8aabba")
            jobs.append((vp, h5, out))

        n = len(jobs)
        self._bar_overall.setMaximum(n)
        self._bar_overall.setValue(0)
        self._bar_overall.setFormat(f"0 / {n} jobs")
        self._bar_step.setValue(0)
        self._btn_run.setEnabled(False)

        self._batch_start = time.monotonic()
        self._job_start   = self._batch_start
        self._timer.start()

        self._lbl_current.setText(f"Starting {n} job(s)…")
        self._lbl_step.setText("Initialising…")
        self._log_msg(f"── Batch started: {n} job(s) ──", "#4a9ab0")

        self._thread = QThread(self)
        self._worker = _BatchWorker(
            jobs, self._roi,
            self._spin_arena.value(),
            self._spin_strip.value(),
            output_opts,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.job_started.connect(self._on_job_started)
        self._worker.step_progress.connect(self._on_step_progress)
        self._worker.job_done.connect(self._on_job_done)
        self._worker.job_error.connect(self._on_job_error)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.all_done.connect(self._thread.quit)
        self._thread.start()

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_job_started(self, i: int, name: str,
                        n_frames: int, n_tracks: int, fps: float):
        """Called once per job after SLEAP loads — we know shape now."""
        self._job_start = time.monotonic()
        n    = self._bar_overall.maximum()
        dur  = n_frames / fps if fps else 0
        self._lbl_current.setText(f"Job {i+1} / {n}  —  {name}")
        self._log_msg(
            f"[{i+1}/{n}]  {name}  "
            f"│  {n_frames:,} frames  │  {n_tracks} track(s)  "
            f"│  {fps:.1f} fps  │  {dur:.0f}s video",
            "#5ab8d8")

    def _on_step_progress(self, i: int, pct: int, label: str):
        """Fine-grained within-job progress."""
        self._bar_step.setValue(pct)
        self._lbl_step.setText(label)
        # Keep table status column up-to-date (truncate long labels)
        short = label if len(label) <= 26 else label[:23] + "…"
        self._set_row_status(i, short, fg="#1a5a70")

    def _on_job_done(self, i: int, path: str, elapsed: float):
        n_done  = self._bar_overall.value() + 1
        n_total = self._bar_overall.maximum()
        self._bar_overall.setValue(n_done)
        self._bar_overall.setFormat(f"{n_done} / {n_total} jobs")
        self._bar_step.setValue(100)

        dur = _fmt_elapsed(elapsed)
        self._set_row_status(i, f"✓  {dur}", fg="#1a7040", bg=QColor("#cff0da"))
        self._log_msg(
            f"[{i+1}] ✓ Done in {dur}  →  {Path(path).name}", "#43d692")

    def _on_job_error(self, i: int, tb: str, elapsed: float):
        n_done  = self._bar_overall.value() + 1
        n_total = self._bar_overall.maximum()
        self._bar_overall.setValue(n_done)
        self._bar_overall.setFormat(f"{n_done} / {n_total} jobs")

        dur  = _fmt_elapsed(elapsed)
        name = self._table.item(i, _COL_VIDEO).text()
        self._set_row_status(i, f"✗  {dur}", fg="#a02020", bg=QColor("#f5d0d0"))

        # Log one-line summary first so it's easy to spot, then full traceback
        last = tb.strip().splitlines()[-1]
        self._log_msg(f"[{i+1}] ✗ {name} — failed after {dur}: {last}", "#e66550")
        self._log_msg(tb, "#c05040")

    def _on_all_done(self, n_ok: int, n_err: int):
        self._timer.stop()
        self._btn_run.setEnabled(True)
        self._bar_step.setValue(0)

        total = _fmt_elapsed(time.monotonic() - self._batch_start)
        self._lbl_current.setText("Batch complete")
        self._lbl_step.setText(
            f"{n_ok} succeeded  ·  {n_err} failed  ·  total {total}")
        self._lbl_elapsed.setText("")

        color = "#43d692" if n_err == 0 else "#fad165"
        self._log_msg(
            f"── Batch complete: {n_ok}/{n_ok + n_err} succeeded"
            f" · {total} total ──", color)
