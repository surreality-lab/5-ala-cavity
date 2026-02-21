"""Main frame display widget with zoom, mouse interaction, and multi-layer overlays."""

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush

from ..engine.mask_store import parse_key


class FrameViewer(QLabel):
    """Full-frame display with mouse-driven annotation and zoom."""

    mouse_moved = pyqtSignal(int, int)
    mouse_clicked = pyqtSignal(int, int, bool)
    mouse_dragged = pyqtSignal(int, int)
    mouse_released = pyqtSignal()
    wheel_zoomed = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a;")

        self._image: np.ndarray | None = None
        self._display_pixmap = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._crop_offset = (0, 0)

        self._brush_size = 20
        self._brush_mode = False
        self._show_brush = False
        self._cursor_pos = QPoint(0, 0)
        self._prev_cursor_pos = QPoint(0, 0)
        self._is_dragging = False
        self._rgb_buffer: np.ndarray | None = None

        self._layer_name = ""
        self._layer_rgb = (200, 80, 200)

        self.zoom_level = 1.0
        self.zoom_center: tuple[int, int] | None = None

    # ── public API ──────────────────────────────────────────────────

    def set_image(self, img_bgr: np.ndarray | None, zoom_level=1.0, zoom_center=None):
        if img_bgr is None:
            self._image = None
            self._display_pixmap = None
            self.clear()
            return
        self._image = img_bgr
        self.zoom_level = zoom_level
        self.zoom_center = zoom_center
        self._refresh()

    def set_brush_params(self, size: int, mode_on: bool):
        self._brush_size = size
        self._brush_mode = mode_on
        self.update()

    def set_layer_label(self, name: str, rgb: tuple[int, int, int]):
        self._layer_name = name
        self._layer_rgb = rgb
        self.update()

    def set_cursor_mode(self, mode: str):
        if mode == "brush":
            self.setCursor(Qt.BlankCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    # ── rendering ───────────────────────────────────────────────────

    def _refresh(self):
        if self._image is None:
            return
        h, w = self._image.shape[:2]

        if self.zoom_level > 1.0:
            vw, vh = int(w / self.zoom_level), int(h / self.zoom_level)
            if self.zoom_center:
                cx, cy = self.zoom_center
            else:
                cx, cy = w // 2, h // 2
            x1 = max(0, min(w - vw, int(cx - vw / 2)))
            y1 = max(0, min(h - vh, int(cy - vh / 2)))
            display_img = self._image[y1:y1 + vh, x1:x1 + vw]
            self._crop_offset = (x1, y1)
        else:
            display_img = self._image
            self._crop_offset = (0, 0)

        dh, dw = display_img.shape[:2]
        self._rgb_buffer = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(self._rgb_buffer.data, dw, dh, dw * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._display_pixmap = scaled
        self._scale = scaled.width() / dw
        self._offset = QPoint(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    # ── coordinate mapping ──────────────────────────────────────────

    def _to_image(self, pos) -> tuple[int | None, int | None]:
        if self._image is None:
            return None, None
        x = (pos.x() - self._offset.x()) / self._scale + self._crop_offset[0]
        y = (pos.y() - self._offset.y()) / self._scale + self._crop_offset[1]
        h, w = self._image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return int(x), int(y)
        return None, None

    # ── mouse events ────────────────────────────────────────────────

    def _brush_dirty_rect(self, center: QPoint) -> QRect:
        """Return the widget-space rect that needs repainting for the brush cursor."""
        r = int(self._brush_size * self._scale) + 4
        return QRect(center.x() - r, center.y() - r, 2 * r, 2 * r)

    def mouseMoveEvent(self, event):
        self._prev_cursor_pos = self._cursor_pos
        self._cursor_pos = event.pos()
        self._show_brush = True
        x, y = self._to_image(event.pos())
        if x is not None:
            self.mouse_moved.emit(x, y)
            if self._is_dragging:
                self.mouse_dragged.emit(x, y)
        if self._brush_mode:
            self.update(self._brush_dirty_rect(self._prev_cursor_pos)
                        .united(self._brush_dirty_rect(self._cursor_pos)))
        else:
            self.update()

    def mousePressEvent(self, event):
        x, y = self._to_image(event.pos())
        if x is not None:
            self.mouse_clicked.emit(x, y, event.button() == Qt.LeftButton)
            if event.button() == Qt.LeftButton and self._brush_mode:
                self._is_dragging = True

    def mouseReleaseEvent(self, event):
        if self._is_dragging:
            self._is_dragging = False
            self.mouse_released.emit()

    def leaveEvent(self, event):
        self._show_brush = False
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        x, y = self._to_image(event.pos())
        ix = x if x is not None else -1
        iy = y if y is not None else -1
        self.wheel_zoomed.emit(1 if delta > 0 else -1, ix, iy)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._layer_name and self._display_pixmap:
            r, g, b = self._layer_rgb
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(r, g, b, 160)))
            painter.drawRoundedRect(self._offset.x() + 8, self._offset.y() + 8, 12, 12, 3, 3)
            painter.setPen(QPen(QColor(255, 255, 255, 220)))
            from PyQt5.QtGui import QFont
            f = QFont("Arial", 11, QFont.Bold)
            painter.setFont(f)
            painter.drawText(self._offset.x() + 26, self._offset.y() + 19, self._layer_name)

        if self._show_brush and self._brush_mode and self._display_pixmap:
            r = int(self._brush_size * self._scale)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self._cursor_pos, r, r)
            painter.setBrush(QBrush(QColor(255, 255, 255, 150)))
            painter.drawEllipse(self._cursor_pos, 3, 3)

        painter.end()


# ── overlay compositing (pure function, no Qt dependency) ───────────

def _draw_dotted_contour(img, contours, color, gap=8):
    """Draw dashed contour lines (OpenCV has no native dashed-line support)."""
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        for i in range(0, len(pts), gap):
            j = min(i + gap // 2, len(pts) - 1)
            cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, 1, cv2.LINE_AA)


def compose_overlays(
    base_img: np.ndarray,
    layers: dict,
    active_name: str,
    visibility: dict[str, bool],
    overlay_mode: str = "normal",
) -> np.ndarray:
    """Draw all visible mask layers onto *base_img* (returns a copy).

    *layers* is keyed by instance key (``"cavity"``, ``"tool_1"``, etc.).
    *visibility* is keyed by category name.
    *overlay_mode*: ``"normal"`` | ``"ghost"`` | ``"hidden"``.
    """
    display = base_img.copy()
    if overlay_mode == "hidden":
        return display

    bh, bw = base_img.shape[:2]

    for key, layer in layers.items():
        category, _ = parse_key(key)
        if not visibility.get(category, True):
            continue
        if layer.mask is None:
            continue

        mask = layer.mask
        if mask.shape[:2] != (bh, bw):
            mask = cv2.resize(mask, (bw, bh), interpolation=cv2.INTER_NEAREST)

        bgr = layer.color_bgr
        is_active = (key == active_name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if overlay_mode == "ghost":
            _draw_dotted_contour(display, contours, bgr)
        elif is_active:
            overlay = display.copy()
            overlay[mask > 127] = bgr
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
            cv2.drawContours(display, contours, -1, bgr, 2)
        else:
            cv2.drawContours(display, contours, -1, bgr, 1)

    if overlay_mode == "ghost":
        return display

    active = layers.get(active_name)
    if active:
        if active.inverted:
            for x, y in active.points_pos:
                cv2.circle(display, (x, y), 6, (0, 165, 255), -1)  # orange = "non-cavity click"
                cv2.circle(display, (x, y), 6, (0, 0, 0), 1)
        else:
            for x, y in active.points_pos:
                cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
                cv2.circle(display, (x, y), 6, (0, 0, 0), 1)
        for x, y in active.points_neg:
            cv2.circle(display, (x, y), 6, (0, 0, 255), 2)
            cv2.line(display, (x - 4, y - 4), (x + 4, y + 4), (0, 0, 255), 2)
            cv2.line(display, (x - 4, y + 4), (x + 4, y - 4), (0, 0, 255), 2)

    return display
