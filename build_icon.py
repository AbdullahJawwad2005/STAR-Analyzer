"""
build_icon.py — Generate mosiac.ico for the MOSIAC application.

Usage:
    python build_icon.py
"""

import math
import sys

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QBrush, QPixmap, QLinearGradient, QRadialGradient
from PySide6.QtWidgets import QApplication


def _render_icon(size: int) -> QImage:
    """Render the MOSIAC icon — a 2×2 mosaic grid with a tracking-dot motif."""
    px = QPixmap(size, size)
    px.fill(QColor(0, 0, 0, 0))
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)

    s = float(size)
    r = s * 0.16   # corner radius

    # --- Background: deep navy gradient ---
    bg_grad = QLinearGradient(0, 0, s, s)
    bg_grad.setColorAt(0.0, QColor(0x0d, 0x1b, 0x2a))
    bg_grad.setColorAt(1.0, QColor(0x1a, 0x2e, 0x40))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(bg_grad))
    p.drawRoundedRect(QRectF(0, 0, s, s), r, r)

    # --- 2×2 mosaic tile grid ---
    pad   = s * 0.12
    gap   = s * 0.05
    tile_w = (s - 2 * pad - gap) / 2
    tile_h = tile_w

    tile_colors = [
        (QColor(0x00, 0x8b, 0xd4), QColor(0x00, 0x6a, 0xaa)),  # top-left:  cyan-blue
        (QColor(0x00, 0xc8, 0x96), QColor(0x00, 0x9a, 0x70)),  # top-right: teal-green
        (QColor(0x7b, 0x5e, 0xff), QColor(0x59, 0x42, 0xcc)),  # bot-left:  violet
        (QColor(0xff, 0xb8, 0x1c), QColor(0xd4, 0x90, 0x00)),  # bot-right: gold
    ]

    positions = [
        (pad,             pad),
        (pad + tile_w + gap, pad),
        (pad,             pad + tile_h + gap),
        (pad + tile_w + gap, pad + tile_h + gap),
    ]

    tile_r = s * 0.06
    for (tx, ty), (c1, c2) in zip(positions, tile_colors):
        tg = QLinearGradient(tx, ty, tx + tile_w, ty + tile_h)
        tg.setColorAt(0.0, c1)
        tg.setColorAt(1.0, c2)
        p.setBrush(QBrush(tg))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(tx, ty, tile_w, tile_h), tile_r, tile_r)

    # --- Tracking dots at tile corners / center ---
    dot_r = max(2.0, s * 0.052)
    dot_positions = [
        QPointF(pad + tile_w,            pad + tile_h),           # center cross
        QPointF(pad + tile_w / 2,        pad + tile_h / 2),       # top-left tile center
        QPointF(pad + tile_w + gap + tile_w / 2, pad + tile_h / 2),  # top-right tile center
        QPointF(pad + tile_w / 2,        pad + tile_h + gap + tile_h / 2),  # bot-left
        QPointF(pad + tile_w + gap + tile_w / 2, pad + tile_h + gap + tile_h / 2),  # bot-right
    ]

    # thin connector lines between center dot and tile centers
    line_pen = QPen(QColor(255, 255, 255, 55))
    line_pen.setWidthF(max(1.0, s * 0.022))
    line_pen.setCapStyle(Qt.RoundCap)
    p.setPen(line_pen)
    center_pt = dot_positions[0]
    for dp in dot_positions[1:]:
        p.drawLine(center_pt, dp)

    # draw dots
    p.setPen(Qt.NoPen)
    for i, dp in enumerate(dot_positions):
        if i == 0:
            # center dot: white
            rg = QRadialGradient(dp.x(), dp.y(), dot_r)
            rg.setColorAt(0.0, QColor(255, 255, 255, 240))
            rg.setColorAt(1.0, QColor(220, 220, 255, 100))
        else:
            rg = QRadialGradient(dp.x(), dp.y(), dot_r * 0.85)
            rg.setColorAt(0.0, QColor(255, 255, 255, 210))
            rg.setColorAt(1.0, QColor(180, 220, 255, 80))
        p.setBrush(QBrush(rg))
        r_use = dot_r if i == 0 else dot_r * 0.75
        p.drawEllipse(dp, r_use, r_use)

    p.end()
    return px.toImage().convertToFormat(QImage.Format_ARGB32)


def _write_ico_manual(output_path: str, pil_images):
    """Write an ICO file with PNG-compressed frames for all sizes."""
    import io, struct

    png_chunks = []
    for img in pil_images:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        png_chunks.append(buf.getvalue())

    n = len(png_chunks)
    header_size = 6 + n * 16
    offsets = []
    pos = header_size
    for chunk in png_chunks:
        offsets.append(pos)
        pos += len(chunk)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<HHH", 0, 1, n))
        for img, chunk, offset in zip(pil_images, png_chunks, offsets):
            w, h = img.size
            bw = w if w < 256 else 0
            bh = h if h < 256 else 0
            f.write(struct.pack("<BBBBHHII", bw, bh, 0, 0, 1, 32, len(chunk), offset))
        for chunk in png_chunks:
            f.write(chunk)


def build_ico(output_path: str = "mosiac.ico"):
    app = QApplication.instance() or QApplication(sys.argv)

    sizes = [16, 32, 48, 128, 256, 512]
    images = [_render_icon(s) for s in sizes]

    try:
        from PIL import Image
    except ImportError:
        images[1].save(output_path.replace('.ico', '.png'))
        print(f"PIL not available; saved PNG instead. Install Pillow for .ico generation.")
        return

    pil_images = []
    for qimg in images:
        w, h = qimg.width(), qimg.height()
        pil_images.append(Image.frombytes("RGBA", (w, h), bytes(qimg.bits()), "raw", "BGRA"))

    _write_ico_manual(output_path, pil_images)
    print(f"Created {output_path} with sizes {sizes}")


if __name__ == "__main__":
    build_ico()
