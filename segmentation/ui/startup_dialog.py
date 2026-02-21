"""Startup dialog for video and frame-range selection."""

import cv2
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QScrollArea, QWidget, QComboBox,
    QFileDialog, QButtonGroup,
)
from PyQt5.QtCore import Qt

from ..engine.mask_store import MaskStore


class StartupDialog(QDialog):
    """Select a video file + 500-frame batch before opening the annotation tool."""

    BATCH = 500

    def __init__(self, base_dir: str | Path, parent=None):
        super().__init__(parent)
        self.base_dir = Path(base_dir)
        self.setWindowTitle("Video Segmentation Tool")
        self.setMinimumSize(640, 440)
        self.selected_video = None
        self.selected_range = None
        self.video_info: dict = {}
        self._build_ui()
        self._scan()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(12)

        title = QLabel("Video Segmentation Tool")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #4a9eff;")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        # folder
        fg = QGroupBox("Base Directory")
        fl = QHBoxLayout(fg)
        self.folder_lbl = QLabel(str(self.base_dir))
        self.folder_lbl.setStyleSheet("font-size: 12px; padding: 4px; background: #2b2b2b; border-radius: 3px;")
        fl.addWidget(self.folder_lbl, stretch=1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        fl.addWidget(browse)
        lay.addWidget(fg)

        # video combo
        vg = QGroupBox("Select Video")
        vl = QVBoxLayout(vg)
        self.combo = QComboBox()
        self.combo.currentTextChanged.connect(self._on_video)
        vl.addWidget(self.combo)
        lay.addWidget(vg)

        # info
        self.info = QLabel("Select a video to see details")
        self.info.setStyleSheet("font-size: 13px;")
        lay.addWidget(self.info)

        # range grid
        rg = QGroupBox("Frame Range (500 per batch)")
        rl = QVBoxLayout(rg)
        self.range_scroll = QScrollArea()
        self.range_scroll.setWidgetResizable(True)
        self.range_widget = QWidget()
        self.range_grid = QGridLayout(self.range_widget)
        self.range_scroll.setWidget(self.range_widget)
        rl.addWidget(self.range_scroll)
        lay.addWidget(rg)

        # buttons
        bl = QHBoxLayout()
        self.open_btn = QPushButton("Open Tool")
        self.open_btn.setStyleSheet("background-color: #228B22; padding: 10px; font-weight: bold;")
        self.open_btn.clicked.connect(self.accept)
        self.open_btn.setEnabled(False)
        bl.addWidget(self.open_btn)
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet("padding: 10px;")
        cancel.clicked.connect(self.reject)
        bl.addWidget(cancel)
        lay.addLayout(bl)

    # ── scanning ────────────────────────────────────────────────────

    def _scan(self):
        if not self.base_dir.exists():
            self.info.setText("Directory does not exist!")
            return
        mp4s = sorted(set(self.base_dir.rglob("*.mp4")) | set(self.base_dir.rglob("*.MP4")))
        if not mp4s:
            self.info.setText("No MP4 videos found.")
            return
        for vp in mp4s:
            try:
                rel = str(vp.relative_to(self.base_dir))
            except ValueError:
                rel = vp.name
            cap = cv2.VideoCapture(str(vp))
            ok = cap.isOpened()
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
            fps = cap.get(cv2.CAP_PROP_FPS) if ok else 0.0
            cap.release()
            # detect existing masks
            folder = vp.parent
            masks_dir = None
            masks_n = 0
            for candidate in [folder / "masks" / "cavity",
                              folder / "pipeline" / "02_cavity" / "cavity_only" / "frames"]:
                if candidate.exists():
                    masks_dir = candidate
                    masks_n = sum(1 for d in candidate.iterdir() if d.is_dir())
                    break
            self.video_info[rel] = dict(
                path=vp, folder=folder, total_frames=total, fps=fps,
                masks_dir=masks_dir, masks_saved=masks_n, ok=ok,
            )
            self.combo.addItem(rel)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select video directory",
                                             str(self.base_dir), QFileDialog.ShowDirsOnly)
        if d:
            self.base_dir = Path(d)
            self.folder_lbl.setText(str(d))
            self.video_info.clear()
            self.combo.clear()
            self.selected_video = self.selected_range = None
            self.open_btn.setEnabled(False)
            self._clear_grid()
            self._scan()

    # ── range buttons ───────────────────────────────────────────────

    def _on_video(self, name):
        if name not in self.video_info:
            return
        self.selected_video = name
        info = self.video_info[name]
        self.info.setText(f"{info['total_frames']} frames @ {info['fps']:.1f} fps  |  {info['masks_saved']} masks saved")
        self._build_range(info)

    @staticmethod
    def _range_color(pct: float) -> str:
        """Return a background color based on mask completion percentage."""
        if pct >= 80:
            return "#228B22"
        if pct >= 30:
            return "#1a6ba0"
        if pct > 0:
            return "#b87333"
        return "#333"

    def _build_range(self, info):
        self._clear_grid()
        self._range_group = QButtonGroup(self)
        self._range_group.setExclusive(True)
        self._range_group.buttonClicked.connect(self._on_range)
        total = info["total_frames"]

        masks_dir = info.get("masks_dir")
        store = None
        if masks_dir:
            store = MaskStore(
                base_output_dir=masks_dir,
                video_folder=info.get("folder"),
            )

        col, row, cols = 0, 0, 5
        for s in range(0, total, self.BATCH):
            e = min(s + self.BATCH - 1, total - 1)
            batch_len = e - s + 1
            pct = 0.0
            if store:
                n_masks = store.count_masks_in_range("cavity", s, e)
                pct = 100.0 * n_masks / batch_len if batch_len else 0.0
            bg = self._range_color(pct)
            label = f"{s}-{e}"
            if pct > 0:
                label += f"  ({pct:.0f}%)"
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty("range", (s, e))
            btn.setMinimumHeight(44)
            btn.setStyleSheet(f"background-color: {bg}; color: #fff; font-weight: bold;")
            self._range_group.addButton(btn)
            self.range_grid.addWidget(btn, row, col)
            col += 1
            if col >= cols:
                col, row = 0, row + 1

    def _on_range(self, btn):
        self.selected_range = btn.property("range")
        self.open_btn.setEnabled(True)
        s, e = self.selected_range
        self.open_btn.setText(f"Open Tool  (frames {s}-{e})")

    def _clear_grid(self):
        while self.range_grid.count():
            item = self.range_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def get_selection(self):
        if self.selected_video and self.selected_range:
            return self.video_info[self.selected_video], self.selected_range[0], self.selected_range[1]
        return None
