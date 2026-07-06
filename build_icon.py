"""
build_icon.py — Generate star_analyzer.ico with the Augusta-colored SLEAP skeleton icon.

Usage:
    python build_icon.py
"""

import math
import sys

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QBrush, QPixmap, QIcon
from PySide6.QtWidgets import QApplication


def _render_icon(size: int) -> QImage:
    """Render the SLEAP-skeleton icon into a QImage of the given size."""
    px = QPixmap(size, size)
    px.fill(QColor(0, 0, 0, 0))
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)

    # Dark navy-teal background
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(QColor(0x15, 0x25, 0x35)))
    r = size * 0.18
    p.drawRoundedRect(QRectF(0, 0, size, size), r, r)

    # Node positions: head (top), left body, right body
    cx = size / 2.0
    head  = QPointF(cx,              size * 0.22)
    left  = QPointF(cx - size * 0.28, size * 0.78)
    right = QPointF(cx + size * 0.28, size * 0.78)

    # Augusta green skeleton lines
    pen = QPen(QColor(0x00, 0x79, 0x32))
    pen.setWidthF(max(1.5, size * 0.065))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.drawLine(head, left)
    p.drawLine(head, right)
    p.drawLine(left, right)

    # Augusta gold tracking nodes
    node_r = max(1.8, size * 0.115)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(QColor(0xFF, 0xB8, 0x1C)))
    for node in (head, left, right):
        p.drawEllipse(node, node_r, node_r)

    p.end()
    return px.toImage().convertToFormat(QImage.Format_ARGB32)


def build_ico(output_path: str = "star_analyzer.ico"):
    app = QApplication.instance() or QApplication(sys.argv)

    sizes = [16, 32, 48, 128, 256, 512]
    images = [_render_icon(s) for s in sizes]

    try:
        from PIL import Image
    except ImportError:
        images[1].save(output_path.replace('.ico', '.png'))
        print(f"PIL not available; saved {output_path.replace('.ico', '.png')} instead.")
        print("Install Pillow (`pip install Pillow`) for proper .ico generation.")
        return

    pil_images = []
    for qimg in images:
        w, h = qimg.width(), qimg.height()
        pil_images.append(Image.frombytes("RGBA", (w, h), bytes(qimg.bits()), "raw", "BGRA"))

    pil_images[0].save(
        output_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=pil_images[1:],
    )
    print(f"Created {output_path} with sizes {sizes}")


if __name__ == "__main__":
    build_ico()
