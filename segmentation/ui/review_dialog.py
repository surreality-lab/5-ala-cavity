"""Matrix review dialog for batch-propagated masks."""

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QGridLayout, QApplication,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from ..engine.mask_store import LAYER_CATEGORIES, parse_key, instance_color


class ReviewDialog(QDialog):
    """Grid of propagated frames for accept / edit-selected / cancel."""

    PER_PAGE = 8
    COLS = 4
    ROWS = 2

    def __init__(self, frames_data: list, parent=None):
        """*frames_data*: list of (frame_idx, bgr_img, {layer_name: mask})."""
        super().__init__(parent)
        self.setWindowTitle("Review Propagated Frames")
        self.setModal(True)
        self.setStyleSheet("""
            QDialog { background-color: #2a2a2a; }
            QLabel { color: #fff; }
            QPushButton { color: #fff; background: #444; border: 1px solid #555;
                          border-radius: 4px; padding: 8px 16px; }
            QPushButton:hover { background: #555; }
        """)
        self.data = frames_data
        self.selected: int | None = None
        self.page = 0
        self.pages = max(1, (len(frames_data) + self.PER_PAGE - 1) // self.PER_PAGE)
        self._build()
        self._refresh()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)

        top = QHBoxLayout()
        self.instr = QLabel("Click the FIRST bad frame to edit from there.")
        self.instr.setStyleSheet("color: #ffcc00; font-weight: bold;")
        top.addWidget(self.instr)
        top.addStretch()
        self.page_lbl = QLabel()
        top.addWidget(self.page_lbl)
        lay.addLayout(top)

        self.sel_lbl = QLabel("No frame selected -- Accept All to save everything")
        self.sel_lbl.setStyleSheet("color: #228B22; font-size: 13px;")
        lay.addWidget(self.sel_lbl)

        self.grid_w = QWidget()
        self.grid = QGridLayout(self.grid_w)
        self.grid.setSpacing(4)
        self.grid.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.grid_w)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("Prev")
        self.prev_btn.clicked.connect(lambda: self._page(-1))
        nav.addWidget(self.prev_btn)
        nav.addStretch()
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(lambda: self._page(1))
        nav.addWidget(self.next_btn)
        lay.addLayout(nav)

        btns = QHBoxLayout()
        ab = QPushButton("Accept All")
        ab.setStyleSheet("background: #228B22; padding: 10px;")
        ab.clicked.connect(self.accept)
        btns.addWidget(ab)
        self.edit_btn = QPushButton("Edit Selected")
        self.edit_btn.setEnabled(False)
        self.edit_btn.setStyleSheet("background: #555; padding: 10px;")
        self.edit_btn.clicked.connect(lambda: self.done(2))
        btns.addWidget(self.edit_btn)
        cb = QPushButton("Cancel")
        cb.setStyleSheet("padding: 10px;")
        cb.clicked.connect(self.reject)
        btns.addWidget(cb)
        lay.addLayout(btns)

        scr = QApplication.primaryScreen().availableGeometry()
        self.resize(int(scr.width() * 0.92), int(scr.height() * 0.88))

    def _refresh(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        s = self.page * self.PER_PAGE
        e = min(s + self.PER_PAGE, len(self.data))
        self.page_lbl.setText(f"Page {self.page + 1}/{self.pages}  ({len(self.data)} frames)")
        self.prev_btn.setEnabled(self.page > 0)
        self.next_btn.setEnabled(self.page < self.pages - 1)

        aw = self.width() - 40
        ah = self.height() - 160
        ts = max(200, min((aw - self.COLS * 10) // self.COLS, (ah - self.ROWS * 25) // self.ROWS))

        for gi, di in enumerate(range(s, e)):
            fidx, img, masks_dict = self.data[di]
            r, c = divmod(gi, self.COLS)
            thumb = self._thumb(img, masks_dict, ts)
            lbl = QLabel()
            lbl.setPixmap(thumb)
            lbl.setFixedSize(ts + 4, ts + 4)
            border = "#228B22"
            if self.selected is not None:
                if di < self.selected:
                    border = "#228B22"
                elif di == self.selected:
                    border = "#ffcc00"
                else:
                    border = "#cc3333"
            lbl.setStyleSheet(f"border: 2px solid {border};")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.mousePressEvent = lambda _, idx=di: self._select(idx)

            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(1)
            bl.addWidget(lbl)
            fl = QLabel(f"Frame {fidx}")
            fl.setAlignment(Qt.AlignCenter)
            fl.setStyleSheet("font-weight: bold; font-size: 11px;")
            bl.addWidget(fl)
            self.grid.addWidget(box, r, c)

    def _thumb(self, img, masks_dict, size):
        h, w = img.shape[:2]
        sc = size / max(h, w)
        nw, nh = int(w * sc), int(h * sc)
        t = cv2.resize(img, (nw, nh))
        for obj_id, mask in masks_dict.items():
            if mask is None:
                continue
            mr = cv2.resize(mask, (nw, nh))
            color = (200, 200, 200)
            for cat_name, cfg in LAYER_CATEGORIES.items():
                base = cfg["base_obj_id"]
                mx = cfg["max_instances"]
                if base <= obj_id < base + mx:
                    inst = obj_id - base + 1
                    color = instance_color(cat_name, inst)
                    break
            ov = t.copy()
            ov[mr > 127] = color
            t = cv2.addWeighted(t, 0.6, ov, 0.4, 0)
        rgb = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, nw, nh, nw * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qi)

    def _page(self, delta):
        self.page = max(0, min(self.pages - 1, self.page + delta))
        self._refresh()

    def _select(self, idx):
        if self.selected == idx:
            self.selected = None
            self.sel_lbl.setText("No frame selected -- Accept All to save everything")
            self.sel_lbl.setStyleSheet("color: #228B22; font-size: 13px;")
            self.edit_btn.setEnabled(False)
        else:
            self.selected = idx
            fi = self.data[idx][0]
            self.sel_lbl.setText(f"Frame {fi} selected. {idx} saved, {len(self.data) - idx - 1} discarded.")
            self.sel_lbl.setStyleSheet("color: #ffcc00; font-size: 13px;")
            self.edit_btn.setEnabled(True)
        self._refresh()

    def get_selected_index(self) -> int | None:
        return self.selected
