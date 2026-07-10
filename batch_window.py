# -*- coding: utf-8 -*-
"""batch_window.py - Batch processing UI for MOSIAC.

Allows users to queue multiple video+h5 pairs, draw per-video ROIs in a
step-through wizard, then run the full analysis+export pipeline unattended.
"""

import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
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


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


def _load_best_frame(video_path: str):
    """Return the first non-black BGR frame from the video (tries 0-30),
    or a grey placeholder if the video can't be opened."""
    cap = cv2.VideoCapture(video_path)
    frame_out = None
    try:
        for _ in range(31):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.mean() > 4.0:   # not essentially black
                frame_out = frame
                break
    finally:
        cap.release()
    if frame_out is None:
        # Grey placeholder so ROIView has something to show
        frame_out = np.full((480, 640, 3), 80, dtype=np.uint8)
    return frame_out


# ------------------------------------------------------------------------------
# Per-video ROI wizard
# ------------------------------------------------------------------------------

class _BatchROIWizard(QDialog):
    """Step through each video so the user can draw a per-video arena ROI."""

    def __init__(self, video_paths: list[str], existing_rois: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Arena ROIs")
        self.setMinimumSize(780, 580)

        self._paths = video_paths          # ordered list of video paths
        self._rois: dict[str, tuple] = dict(existing_rois)  # path -> roi tuple
        self._idx = 0

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # -- Info bar ----------------------------------------------------------
        info_row = QHBoxLayout()
        self._lbl_title = QLabel()
        self._lbl_title.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #123a4a;")
        self._lbl_count = QLabel()
        self._lbl_count.setStyleSheet("color: #5a7a88; font-size: 11px;")
        self._lbl_count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_row.addWidget(self._lbl_title, stretch=1)
        info_row.addWidget(self._lbl_count)
        layout.addLayout(info_row)

        self._lbl_hint = QLabel(
            "Drag to draw a square ROI for this video. "
            "Use Prev / Next to step through all videos.")
        self._lbl_hint.setWordWrap(True)
        self._lbl_hint.setStyleSheet("color: #3a6a7a; font-size: 11px;")
        layout.addWidget(self._lbl_hint)

        # -- ROI view ----------------------------------------------------------
        self._view = ROIView(self)
        layout.addWidget(self._view, stretch=1)

        # -- Navigation buttons ------------------------------------------------
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("<  Prev")
        self._btn_next = QPushButton("Next  >")
        self._btn_copy = QPushButton("Copy from prev video")
        self._btn_clear = QPushButton("Clear ROI")
        for b in (self._btn_prev, self._btn_next,
                  self._btn_copy, self._btn_clear):
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._btn_prev.clicked.connect(lambda: self._go_to(self._idx - 1))
        self._btn_next.clicked.connect(lambda: self._go_to(self._idx + 1))
        self._btn_copy.clicked.connect(self._copy_from_prev)
        self._btn_clear.clicked.connect(self._clear_current)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._btn_next)
        nav.addStretch()
        nav.addWidget(self._btn_copy)
        nav.addWidget(self._btn_clear)
        layout.addLayout(nav)

        # -- Dialog buttons ----------------------------------------------------
        self._btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._btn_done = self._btns.button(QDialogButtonBox.Ok)
        self._btn_done.setText("Done")
        self._btns.accepted.connect(self.accept)
        self._btns.rejected.connect(self.reject)
        layout.addWidget(self._btns)

        # Save ROI whenever it changes in the view
        self._view.roi_changed.connect(self._on_roi_changed)

        # Show the first video
        self._go_to(0)

    # -- Navigation ------------------------------------------------------------

    def _go_to(self, idx: int):
        if not self._paths:
            return
        idx = max(0, min(idx, len(self._paths) - 1))
        # Save current slot before moving
        self._save_current()
        self._idx = idx
        path = self._paths[idx]

        # Load frame
        frame = _load_best_frame(path)
        self._view.set_frame(frame)

        # Restore saved ROI for this slot, if any
        roi = self._rois.get(path)
        if roi is not None:
            (x0, y0), (x1, y1), side = roi
            from PySide6.QtCore import QRectF
            self._view._rect_item.setRect(
                QRectF(x0, y0, side, side))
        else:
            self._view.clear_roi()

        # Update labels / button states
        n = len(self._paths)
        self._lbl_title.setText(Path(path).name)
        set_count = sum(1 for p in self._paths if p in self._rois)
        self._lbl_count.setText(
            f"Video {idx + 1} / {n}  .  {set_count}/{n} ROIs set")
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < n - 1)
        self._btn_copy.setEnabled(idx > 0)
        self._btn_done.setEnabled(set_count == n)

    def _save_current(self):
        """Snapshot the current ROIView state into _rois."""
        if not self._paths or self._idx >= len(self._paths):
            return
        path = self._paths[self._idx]
        roi = self._view.roi_native()
        if roi is not None:
            self._rois[path] = roi
        else:
            self._rois.pop(path, None)

    def _on_roi_changed(self, _rect):
        """Called every time the user finishes drawing - auto-save and refresh."""
        self._save_current()
        # Refresh count label / Done button
        n = len(self._paths)
        set_count = sum(1 for p in self._paths if p in self._rois)
        self._lbl_count.setText(
            f"Video {self._idx + 1} / {n}  .  {set_count}/{n} ROIs set")
        self._btn_done.setEnabled(set_count == n)

    def _copy_from_prev(self):
        """Copy the nearest previous slot's ROI to the current slot."""
        for j in range(self._idx - 1, -1, -1):
            prev_path = self._paths[j]
            if prev_path in self._rois:
                (x0, y0), (x1, y1), side = self._rois[prev_path]
                from PySide6.QtCore import QRectF
                self._view._rect_item.setRect(QRectF(x0, y0, side, side))
                # Trigger save/count refresh
                self._on_roi_changed(None)
                return
        QMessageBox.information(self, "No previous ROI",
                                "No earlier video has an ROI to copy from.")

    def _clear_current(self):
        self._view.clear_roi()
        path = self._paths[self._idx] if self._paths else None
        if path:
            self._rois.pop(path, None)
        n = len(self._paths)
        set_count = sum(1 for p in self._paths if p in self._rois)
        self._lbl_count.setText(
            f"Video {self._idx + 1} / {n}  .  {set_count}/{n} ROIs set")
        self._btn_done.setEnabled(set_count == n)

    def accept(self):
        self._save_current()
        super().accept()

    def result_rois(self) -> dict:
        """Return ``{video_path: ((x0,y0),(x1,y1),side)}`` for all set videos."""
        return dict(self._rois)


# ------------------------------------------------------------------------------
# Batch worker
# ------------------------------------------------------------------------------

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

    def __init__(self, jobs, rois: dict, arena_cm, strip_cm, output_opts,
                 parent=None):
        super().__init__(parent)
        self._jobs        = jobs
        self._rois        = rois          # {video_path: roi_tuple}
        self._arena_cm    = arena_cm
        self._strip_cm    = strip_cm
        self._output_opts = output_opts or {}

    def run(self):
        from sleap_loader import load_sleap
        from preprocessing import fill_and_smooth_tracks
        from run_popup import _run_analysis, _ExportWorker

        arena_cm = self._arena_cm
        strip_cm = self._strip_cm
        opts     = self._output_opts

        n_ok = n_err = 0

        for i, (video_path, h5_path, out_path) in enumerate(self._jobs):
            t0   = time.monotonic()
            name = Path(video_path).name
            try:
                # -- Validate inputs -------------------------------------------
                if not Path(video_path).is_file():
                    raise FileNotFoundError(f"Video not found: {video_path}")
                if not Path(h5_path).is_file():
                    raise FileNotFoundError(f"H5 not found: {h5_path}")

                roi = self._rois.get(video_path)
                if roi is None:
                    raise ValueError(f"No ROI set for: {name}")

                (_, _), (_, _), side = roi
                px_per_cm = side / max(arena_cm, 1)

                # Ensure output directory exists
                os.makedirs(Path(out_path).parent, exist_ok=True)

                # -- 2% : video metadata ---------------------------------------
                self.step_progress.emit(i, 2, "Reading video metadata...")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    cap.release()
                    raise IOError(f"cv2 could not open: {video_path}")
                fps            = float(cap.get(cv2.CAP_PROP_FPS) or 30)
                n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # -- 5% : load SLEAP -------------------------------------------
                self.step_progress.emit(i, 5, "Loading SLEAP file...")
                sleap_data = load_sleap(h5_path)
                n_tracks   = sleap_data["tracks"].shape[3]
                self.job_started.emit(i, name, n_video_frames, n_tracks, fps)

                # -- 8-55% : fill & smooth -------------------------------------
                def _smooth_cb(pct, _i=i):
                    mapped = 8 + int(pct * 0.47)   # 0->8, 100->55
                    self.step_progress.emit(
                        _i, mapped, f"Smoothing tracks... {pct}%")

                self.step_progress.emit(i, 8, "Smoothing & filling tracks...")
                processed = dict(sleap_data)
                processed["tracks"] = fill_and_smooth_tracks(
                    sleap_data["tracks"], fps=fps,
                    progress_callback=_smooth_cb)

                # -- 56-79% : analysis -----------------------------------------
                self.step_progress.emit(i, 56, "Computing kinematics & behaviors...")
                cache = _run_analysis(processed, fps, roi, px_per_cm, strip_cm, opts)
                self.step_progress.emit(i, 79, "Analysis complete - preparing export...")

                # -- 80-99% : export -------------------------------------------
                self.step_progress.emit(i, 80, "Writing outputs...")
                exp = _ExportWorker(
                    cache, fps, roi, arena_cm, strip_cm, out_path,
                    arena_snapshot=None, export_opts=opts)
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


# ------------------------------------------------------------------------------
# Table column indices
# ------------------------------------------------------------------------------

_COL_VIDEO  = 0
_COL_H5     = 1
_COL_OUTPUT = 2
_COL_ROI    = 3
_COL_STATUS = 4


# ------------------------------------------------------------------------------
# Main BatchWindow
# ------------------------------------------------------------------------------

class BatchWindow(QWidget):
    """Top-level window for queuing and running batch MOSIAC jobs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MOSIAC - Batch Analysis")
        self.setMinimumSize(1060, 680)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._rois: dict[str, tuple] = {}    # video_path -> roi tuple
        self._thread: QThread | None = None
        self._worker: _BatchWorker | None = None

        self._batch_start: float = 0.0
        self._job_start:   float = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick_elapsed)

        self._build_ui()

    # -- UI construction -------------------------------------------------------

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

        # -- Job table ------------------------------------------------------
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Video", "H5 File", "Output", "ROI", "Status / Step"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.Fixed)
        hh.setSectionResizeMode(4, QHeaderView.Fixed)
        hh.resizeSection(3, 48)
        hh.resizeSection(4, 210)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        root.addWidget(self._table, stretch=1)

        # -- File management buttons ----------------------------------------
        btn_row = QHBoxLayout()
        self._btn_add_pairs  = QPushButton("Add Pairs...")
        self._btn_add_folder = QPushButton("Add Folder...")
        self._btn_remove     = QPushButton("Remove Selected")
        btn_clear_log        = QPushButton("Clear Log")
        for b in (self._btn_add_pairs, self._btn_add_folder,
                  self._btn_remove, btn_clear_log):
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._btn_add_pairs.clicked.connect(self._add_pairs)
        self._btn_add_folder.clicked.connect(self._add_folder_dialog)
        self._btn_remove.clicked.connect(self._remove_selected)
        btn_clear_log.clicked.connect(lambda: self._log.clear())
        btn_row.addWidget(self._btn_add_pairs)
        btn_row.addWidget(self._btn_add_folder)
        btn_row.addWidget(self._btn_remove)
        btn_row.addStretch()
        btn_row.addWidget(btn_clear_log)
        root.addLayout(btn_row)

        # -- Settings / ROI / output dir -----------------------------------
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

        self._btn_roi = QPushButton("Set ROIs...")
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
        btn_browse = QPushButton("Browse...")
        btn_browse.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_browse.clicked.connect(self._browse_outdir)
        cfg.addWidget(btn_browse)

        root.addLayout(cfg)

        # -- Progress card --------------------------------------------------
        prog_card = QFrame()
        prog_card.setObjectName("ProgCard")
        pc = QVBoxLayout(prog_card)
        pc.setContentsMargins(12, 10, 12, 10)
        pc.setSpacing(6)

        hdr = QHBoxLayout()
        self._lbl_current = QLabel("No job running")
        self._lbl_current.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #123a4a;")
        self._lbl_elapsed = QLabel("")
        self._lbl_elapsed.setStyleSheet("color: #5a7a88; font-size: 11px;")
        self._lbl_elapsed.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr.addWidget(self._lbl_current, stretch=1)
        hdr.addWidget(self._lbl_elapsed)
        pc.addLayout(hdr)

        self._lbl_step = QLabel("Ready - add files and set ROIs to begin.")
        self._lbl_step.setStyleSheet("color: #3a6a7a; font-size: 11px;")
        pc.addWidget(self._lbl_step)

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

        rb = QHBoxLayout()
        self._btn_run = QPushButton(">  Run Batch")
        self._btn_run.setMinimumHeight(36)
        self._btn_run.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._btn_run.clicked.connect(self._run_batch)
        rb.addStretch(); rb.addWidget(self._btn_run)
        pc.addLayout(rb)

        root.addWidget(prog_card)

        # -- Activity log ---------------------------------------------------
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(3000)
        self._log.setFixedHeight(190)
        _f = QFont("Consolas")
        _f.setStyleHint(QFont.Monospace)
        _f.setPointSize(9)
        self._log.setFont(_f)
        root.addWidget(self._log)

    # -- Internal helpers ------------------------------------------------------

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

    def _video_paths(self) -> list[str]:
        return [self._table.item(r, _COL_VIDEO).toolTip()
                for r in range(self._n_jobs())]

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

    def _set_row_roi(self, row: int, has_roi: bool):
        item = self._table.item(row, _COL_ROI)
        if item is None:
            item = QTableWidgetItem()
            self._table.setItem(row, _COL_ROI, item)
        item.setText("v" if has_roi else "x")
        item.setTextAlignment(Qt.AlignCenter)
        item.setForeground(QColor("#1a7040" if has_roi else "#b03030"))

    def _refresh_roi_cols(self):
        """Refresh the ROI column for every row and update the ROI button label."""
        for r in range(self._n_jobs()):
            vp = self._table.item(r, _COL_VIDEO).toolTip()
            self._set_row_roi(r, vp in self._rois)
        self._update_roi_btn()

    def _add_row(self, video_path: str, h5_path: str):
        row = self._table.rowCount()
        self._table.insertRow(row)
        vi = QTableWidgetItem(Path(video_path).name)
        vi.setToolTip(video_path)
        hi = QTableWidgetItem(Path(h5_path).name)
        hi.setToolTip(h5_path)
        oi = QTableWidgetItem("")
        self._table.setItem(row, _COL_VIDEO,  vi)
        self._table.setItem(row, _COL_H5,     hi)
        self._table.setItem(row, _COL_OUTPUT, oi)
        self._set_row_roi(row, video_path in self._rois)
        self._set_row_status(row, "Pending", fg="#8aabba")
        self._update_roi_btn()

    def _update_roi_btn(self):
        n     = self._n_jobs()
        n_set = sum(1 for vp in self._video_paths() if vp in self._rois)
        self._btn_roi.setEnabled(n > 0)
        if n == 0:
            self._btn_roi.setText("Set ROIs...")
        elif n_set == n:
            self._btn_roi.setText(f"Set ROIs v ({n_set}/{n})")
        else:
            self._btn_roi.setText(f"Set ROIs... ({n_set}/{n} set)")

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
            f"job {_fmt_elapsed(job)}  .  total {_fmt_elapsed(total)}")

    # -- File management -------------------------------------------------------

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
                self._log_msg(f"No .h5 found for {rel} - skipped", "#fad165")

        if added:
            self._log_msg(
                f"Added {added} pair(s) from {folder_path.name} (searched subfolders)",
                "#43d692")

    def _match_h5(self, video_path: Path, all_h5: list) -> "Path | None":
        """Return the best-matching h5 for *video_path* from *all_h5*.

        Priority (high -> low):
        1. Exact stem match in the same directory
        2. H5 stem starts with video stem, same directory (shortest stem wins)
        3. Exact stem match anywhere in the tree
        4. H5 stem starts with video stem, anywhere (shortest stem wins)
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
            # Remove orphaned ROI entries for removed videos
            vp_item = self._table.item(r, _COL_VIDEO)
            if vp_item:
                self._rois.pop(vp_item.toolTip(), None)
            self._table.removeRow(r)
        self._update_roi_btn()

    # -- ROI -------------------------------------------------------------------

    def _set_roi(self):
        paths = self._video_paths()
        if not paths:
            return
        wiz = _BatchROIWizard(paths, self._rois, self)
        if wiz.exec() == QDialog.Accepted:
            self._rois = wiz.result_rois()
            self._refresh_roi_cols()
            n_set = sum(1 for p in paths if p in self._rois)
            self._log_msg(
                f"ROIs saved - {n_set}/{len(paths)} videos have an ROI set",
                "#43d692")

    def _browse_outdir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select output directory")
        if folder:
            self._edit_outdir.setText(folder)

    # -- Controls enable/disable -----------------------------------------------

    def _set_controls_enabled(self, enabled: bool):
        self._btn_add_pairs.setEnabled(enabled)
        self._btn_add_folder.setEnabled(enabled)
        self._btn_remove.setEnabled(enabled)
        self._btn_roi.setEnabled(enabled and self._n_jobs() > 0)
        self._btn_run.setEnabled(enabled)

    # -- Run -------------------------------------------------------------------

    def _run_batch(self):
        if self._n_jobs() == 0:
            QMessageBox.warning(self, "No jobs",
                                "Add at least one video+H5 pair first.")
            return
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "A batch is already running.")
            return

        # Check all ROIs are set
        paths = self._video_paths()
        missing = [Path(p).name for p in paths if p not in self._rois]
        if missing:
            names = "\n  - ".join(missing)
            QMessageBox.warning(
                self, "ROIs missing",
                f"The following videos have no ROI set:\n  - {names}\n\n"
                'Open "Set ROIs..." to set an ROI for every video.')
            return

        from run_popup import ProcessingOutputDialog
        dlg = ProcessingOutputDialog(n_tracks=2, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        output_opts = dlg.options()

        # Resolve output paths
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

        self._set_controls_enabled(False)

        self._batch_start = time.monotonic()
        self._job_start   = self._batch_start
        self._timer.start()

        self._lbl_current.setText(f"Starting {n} job(s)...")
        self._lbl_step.setText("Initialising...")
        self._log_msg(f"-- Batch started: {n} job(s) --", "#4a9ab0")

        self._thread = QThread(self)
        self._worker = _BatchWorker(
            jobs, self._rois,
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

    # -- Worker callbacks ------------------------------------------------------

    def _on_job_started(self, i: int, name: str,
                        n_frames: int, n_tracks: int, fps: float):
        self._job_start = time.monotonic()
        n   = self._bar_overall.maximum()
        dur = n_frames / fps if fps else 0
        self._lbl_current.setText(f"Job {i+1} / {n}  -  {name}")
        self._log_msg(
            f"[{i+1}/{n}]  {name}  "
            f"|  {n_frames:,} frames  |  {n_tracks} track(s)  "
            f"|  {fps:.1f} fps  |  {dur:.0f}s video",
            "#5ab8d8")

    def _on_step_progress(self, i: int, pct: int, label: str):
        self._bar_step.setValue(pct)
        self._lbl_step.setText(label)
        short = label if len(label) <= 26 else label[:23] + "..."
        self._set_row_status(i, short, fg="#1a5a70")

    def _on_job_done(self, i: int, path: str, elapsed: float):
        n_done  = self._bar_overall.value() + 1
        n_total = self._bar_overall.maximum()
        self._bar_overall.setValue(n_done)
        self._bar_overall.setFormat(f"{n_done} / {n_total} jobs")
        self._bar_step.setValue(100)

        dur = _fmt_elapsed(elapsed)
        self._set_row_status(i, f"v  {dur}", fg="#1a7040", bg=QColor("#cff0da"))
        self._log_msg(
            f"[{i+1}] v Done in {dur}  ->  {Path(path).name}", "#43d692")

    def _on_job_error(self, i: int, tb: str, elapsed: float):
        n_done  = self._bar_overall.value() + 1
        n_total = self._bar_overall.maximum()
        self._bar_overall.setValue(n_done)
        self._bar_overall.setFormat(f"{n_done} / {n_total} jobs")

        dur  = _fmt_elapsed(elapsed)
        name = self._table.item(i, _COL_VIDEO).text()
        self._set_row_status(i, f"x  {dur}", fg="#a02020", bg=QColor("#f5d0d0"))

        last = tb.strip().splitlines()[-1]
        self._log_msg(f"[{i+1}] x {name} - failed after {dur}: {last}", "#e66550")
        self._log_msg(tb, "#c05040")

    def _on_all_done(self, n_ok: int, n_err: int):
        self._timer.stop()
        self._bar_step.setValue(0)
        self._set_controls_enabled(True)

        total = _fmt_elapsed(time.monotonic() - self._batch_start)
        self._lbl_current.setText("Batch complete")
        self._lbl_step.setText(
            f"{n_ok} succeeded  .  {n_err} failed  .  total {total}")
        self._lbl_elapsed.setText("")

        color = "#43d692" if n_err == 0 else "#fad165"
        self._log_msg(
            f"-- Batch complete: {n_ok}/{n_ok + n_err} succeeded"
            f" . {total} total --", color)

    # -- Close guard -----------------------------------------------------------

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            ans = QMessageBox.question(
                self, "Batch running",
                "A batch is still running. Cancel it and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
            if ans != QMessageBox.Yes:
                event.ignore()
                return
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)
