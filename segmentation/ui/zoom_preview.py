"""Magnified preview widget that follows the cursor."""

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap


class ZoomPreview(QLabel):
    """Small widget showing a magnified crop around the cursor."""

    def __init__(self, size=150, zoom=4, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setStyleSheet("border: 2px solid #444; background-color: #222;")
        self._zoom = zoom
        self._size = size
        self._source: np.ndarray | None = None
        self._brush_size = 20
        self._brush_mode = False
        self._rgb_buffer: np.ndarray | None = None

    def set_source(self, img_bgr: np.ndarray | None):
        self._source = img_bgr

    def set_brush_params(self, size: int, mode_on: bool):
        self._brush_size = size
        self._brush_mode = mode_on

    def update_position(self, x: int, y: int):
        if self._source is None:
            return
        h, w = self._source.shape[:2]
        cs = self._size // self._zoom
        x1, y1 = max(0, x - cs // 2), max(0, y - cs // 2)
        x2, y2 = min(w, x1 + cs), min(h, y1 + cs)
        if x2 - x1 < cs:
            x1 = max(0, x2 - cs)
        if y2 - y1 < cs:
            y1 = max(0, y2 - cs)

        crop = self._source[y1:y2, x1:x2].copy()
        zoomed = cv2.resize(crop, (self._size, self._size), interpolation=cv2.INTER_NEAREST)

        if self._brush_mode:
            c = self._size // 2
            cv2.circle(zoomed, (c, c), int(self._brush_size * self._zoom), (255, 255, 255), 1)
        # crosshair
        c = self._size // 2
        cv2.line(zoomed, (c - 10, c), (c + 10, c), (0, 255, 255), 1)
        cv2.line(zoomed, (c, c - 10), (c, c + 10), (0, 255, 255), 1)

        self._rgb_buffer = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
        qimg = QImage(self._rgb_buffer.data, self._size, self._size, self._size * 3, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))
