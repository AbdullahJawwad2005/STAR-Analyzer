import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QImage, QPixmap, QPen, QColor
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QFrame,
    QSizePolicy,
)


class ROIView(QGraphicsView):
    """A QGraphicsView whose scene coordinates equal native video pixels.

    The frame is placed at scene (0, 0) at full resolution and the scene rect
    is set to the frame size, so the scene IS native-pixel space. fitInView
    scales the whole frame to fit the widget (with letterboxing if aspect
    ratios differ); mapToScene() inverts that scaling, turning any widget click
    straight into native-frame coordinates. No manual scale/offset math.

    Drag with the left mouse button to draw a square ROI; side = max(|dx|, |dy|).
    """

    roi_changed = Signal(QRectF)  # emitted on release, in native pixel coords

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        # The visible ROI rectangle.
        self._rect_item = QGraphicsRectItem()
        pen = QPen(QColor("#00e0ff"))
        pen.setWidth(2)
        pen.setCosmetic(True)  # constant on-screen width no matter the zoom
        self._rect_item.setPen(pen)
        self._scene.addItem(self._rect_item)

        self._native_size = None     # (width, height) of the loaded frame
        self._roi_size = 200  # square side length for Create button
        self._origin = None          # drag start in scene/native coords

        self.setDragMode(QGraphicsView.NoDrag)  # we draw our own rect
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QColor("#000000"))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ---- frame loading -------------------------------------------------

    def set_frame(self, frame_bgr):
        """Display an HxWx3 BGR numpy array (OpenCV's native format)."""
        previous_roi = self._rect_item.rect()
        h, w = frame_bgr.shape[:2]
        self._native_size = (w, h)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)  # guarantee a 3*w byte stride
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        # fromImage copies the pixels now, while `rgb` is still alive, so the
        # numpy buffer can be freed safely afterwards.
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg))

        self._scene.setSceneRect(QRectF(0, 0, w, h))
        if previous_roi.isNull():
            self.clear_roi()
        else:
            self._rect_item.setRect(previous_roi.intersected(QRectF(0, 0, w, h)))
        self._fit()

    def _fit(self):
        # Exact "fit, keep aspect ratio" transform. We avoid fitInView() because
        # it reserves a small margin, which throws mapToScene off by a few pixels
        # at the frame edges. Here the scale is exactly min(fit_x, fit_y) and the
        # frame is centered, so mapToScene returns true native pixels everywhere.
        if not self._native_size:
            return
        w, h = self._native_size
        vp = self.viewport().rect()
        if vp.width() == 0 or vp.height() == 0:
            return
        s = min(vp.width() / w, vp.height() / h)
        self.resetTransform()
        self.scale(s, s)
        self.centerOn(w / 2.0, h / 2.0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()  # keep the whole frame visible when the widget resizes

    # ---- ROI drawing ---------------------------------------------------

    def _clamp(self, pt):
        """Keep a scene point inside the frame so the ROI can't run off-image."""
        w, h = self._native_size
        return QPointF(min(max(pt.x(), 0.0), w), min(max(pt.y(), 0.0), h))

    def set_roi_size(self, px: int) -> None:
        self._roi_size = max(1, int(px))

    def _square_from_drag(self, origin, cur):
        """Build a square QRectF from two points; side = max(|dx|, |dy|),
        clamped so the square never exceeds the frame boundary."""
        dx = cur.x() - origin.x()
        dy = cur.y() - origin.y()
        side = max(abs(dx), abs(dy))
        if side == 0:
            return QRectF(origin, origin)
        # Clamp the side so neither corner goes outside the frame.
        # The origin is already clamped; we compute the maximum side that
        # fits in each direction from origin, then take the smallest.
        w, h = self._native_size
        max_x = (w - origin.x()) if dx >= 0 else origin.x()
        max_y = (h - origin.y()) if dy >= 0 else origin.y()
        side  = min(side, max_x, max_y)
        x1 = origin.x() + (side if dx >= 0 else -side)
        y1 = origin.y() + (side if dy >= 0 else -side)
        return QRectF(origin, QPointF(x1, y1)).normalized()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._native_size:
            self._origin = self._clamp(self.mapToScene(event.position().toPoint()))
            self._rect_item.setRect(QRectF(self._origin, self._origin))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._origin is not None:
            cur = self._clamp(self.mapToScene(event.position().toPoint()))
            self._rect_item.setRect(self._square_from_drag(self._origin, cur))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._origin is not None:
            cur = self._clamp(self.mapToScene(event.position().toPoint()))
            rect = self._square_from_drag(self._origin, cur)
            self._origin = None
            self._rect_item.setRect(rect)
            if not rect.isNull():
                self.roi_changed.emit(rect)
        else:
            super().mouseReleaseEvent(event)

    def clear_roi(self):
        self._rect_item.setRect(QRectF())

    def roi_native(self):
        """ROI as ((x0, y0), (x1, y1), side) integer native pixels, or None.

        Slicing convention: crop = frame[y0:y1, x0:x1].
        """
        r = self._rect_item.rect()
        if r.isNull() or self._native_size is None:
            return None
        x0, y0 = int(round(r.left())), int(round(r.top()))
        x1, y1 = int(round(r.right())), int(round(r.bottom()))
        return (x0, y0), (x1, y1), x1 - x0
