"""Right-side control panel with dynamic multi-instance layer controls."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QButtonGroup, QGroupBox, QComboBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from .zoom_preview import ZoomPreview
from ..engine.mask_store import (
    LAYER_CATEGORIES, CATEGORY_NAMES, instance_key, instance_color,
    LAYER_PRIORITY,
)

STARTING_LAYER_OPTIONS = [
    ("auto", "Auto (stay)"),
    ("hand", "Hand first"),
    ("tool", "Tool first"),
    ("cavity", "Cavity first"),
]

# Sizing: 1080p full screen → 1080 - 24(menu) - 28(status) - 40(banner) = 988px panel height.
# Content: margins 12 + spacing 5×14 + sections ~680 = 762px, fits with ~226px spare.
_PANEL_CSS = """
QWidget { background-color: #2a2a2a; color: #eee; font-size: 11px; }
QGroupBox {
    border: 1px solid #444; border-radius: 4px;
    margin-top: 6px; padding-top: 6px; font-weight: bold; font-size: 11px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton {
    background-color: #444; border: 1px solid #555;
    border-radius: 4px; padding: 4px 8px; min-height: 22px; font-size: 11px;
}
QPushButton:hover { background-color: #555; }
QPushButton:pressed { background-color: #333; }
QPushButton:checked { background-color: #0066cc; border-color: #0088ff; }
QSlider::groove:horizontal { height: 8px; background: #444; border-radius: 4px; }
QSlider::handle:horizontal { background: #cc3333; width: 14px; margin: -3px 0; border-radius: 7px; }
QSlider::sub-page:horizontal { background: #cc3333; border-radius: 4px; }
QComboBox { padding: 4px 6px; min-height: 22px; font-size: 11px; }
"""


class ControlPanel(QWidget):
    """Right-side panel: dynamic instance layers, frame nav, tool mode, zoom preview."""

    frame_changed = pyqtSignal(int)
    display_mode_changed = pyqtSignal(str)
    correction_mode_changed = pyqtSignal(str)
    tool_mode_changed = pyqtSignal(str)
    brush_size_changed = pyqtSignal(int)
    undo_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    save_requested = pyqtSignal()
    propagate_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    layer_changed = pyqtSignal(str)       # emits instance key like "tool_1"
    visibility_changed = pyqtSignal(str, bool)
    invert_toggled = pyqtSignal()
    instance_added = pyqtSignal(str)      # category name
    instance_removed = pyqtSignal(str)    # instance key
    help_requested = pyqtSignal()
    starting_layer_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setStyleSheet(_PANEL_CSS)
        self._start = 0
        self._end = 0
        self._current = 0
        self._brush_size = 20
        self._playing = False
        self._visibility = {cat: True for cat in CATEGORY_NAMES}
        self._active_key = "cavity"
        self._instance_btns: dict[str, QPushButton] = {}
        self._slot_data: dict[str, bool] = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(5)

        # -- starting layer selector --
        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel("Start on:"))
        self._start_layer_combo = QComboBox()
        self._start_layer_combo.setFixedWidth(120)
        for val, label in STARTING_LAYER_OPTIONS:
            self._start_layer_combo.addItem(label, val)
        self._start_layer_combo.setToolTip(
            "Which layer to auto-select when navigating to a new frame"
        )
        self._start_layer_combo.currentIndexChanged.connect(
            lambda _: self.starting_layer_changed.emit(self.starting_layer)
        )
        sl_row.addWidget(self._start_layer_combo)
        sl_row.addStretch()
        root.addLayout(sl_row)

        # -- layer selector with pre-rendered instance slots --
        lg = QGroupBox("Active Layer")
        ll = QVBoxLayout(lg)

        sorted_cats = sorted(CATEGORY_NAMES, key=lambda c: LAYER_PRIORITY.get(c, 0), reverse=True)

        for cat in sorted_cats:
            cfg = LAYER_CATEGORIES[cat]
            b, g, r_c = cfg["color_bgr"]
            rgb_str = f"rgb({r_c},{g},{b})"
            max_inst = cfg["max_instances"]

            cat_row = QHBoxLayout()
            cat_row.setSpacing(4)

            cat_lbl = QLabel(f"{cfg['shortcut']} {cat.capitalize()}")
            cat_lbl.setFixedWidth(62)
            cat_lbl.setStyleSheet(f"color: {rgb_str}; font-weight: bold; font-size: 11px;")
            cat_row.addWidget(cat_lbl)

            for i in range(1, max_inst + 1):
                key = instance_key(cat, i)
                btn = QPushButton(str(i))
                btn.setFixedSize(28, 24)
                btn.setCheckable(True)
                btn.setProperty("inst_key", key)
                btn.setProperty("cat", cat)
                btn.setProperty("inst_id", i)
                btn.setProperty("rgb_str", rgb_str)
                self._apply_slot_style(btn, rgb_str, state="dimmed")
                btn.clicked.connect(lambda _, k=key: self._select_instance(k))
                cat_row.addWidget(btn)
                self._instance_btns[key] = btn
                self._slot_data[key] = False

            cat_row.addStretch()
            ll.addLayout(cat_row)

        root.addWidget(lg)

        # -- category visibility --
        vs = QHBoxLayout()
        self._vis_btns: dict[str, QPushButton] = {}
        self._status_labels: dict[str, QLabel] = {}
        for cat in CATEGORY_NAMES:
            cfg = LAYER_CATEGORIES[cat]
            b, g, r_c = cfg["color_bgr"]
            col = QVBoxLayout()
            vb = QPushButton("Show")
            vb.setFixedSize(46, 22)
            vb.setCheckable(True)
            vb.setChecked(True)
            vb.setToolTip(f"Toggle {cat} visibility")
            vb.setStyleSheet(
                f"QPushButton {{ font-size: 11px; }}"
                f"QPushButton:checked {{ border: 2px solid rgb({r_c},{g},{b}); }}"
            )
            vb.clicked.connect(lambda checked, n=cat: self._on_vis_toggle(n, checked))
            self._vis_btns[cat] = vb
            col.addWidget(vb, alignment=Qt.AlignCenter)
            sl = QLabel("-")
            sl.setAlignment(Qt.AlignCenter)
            sl.setStyleSheet("font-size: 10px; color: #888;")
            self._status_labels[cat] = sl
            col.addWidget(sl)
            vs.addLayout(col)
        root.addLayout(vs)

        # -- invert mode (cavity only) --
        self.invert_btn = QPushButton("Invert (I)")
        self.invert_btn.setCheckable(True)
        self.invert_btn.setStyleSheet(
            "QPushButton:checked { background-color: #8B0000; color: #fff; font-weight: bold; }"
        )
        self.invert_btn.clicked.connect(lambda: self.invert_toggled.emit())
        root.addWidget(self.invert_btn)

        # -- frame navigation --
        ng = QGroupBox("Frame Navigation")
        nl = QVBoxLayout(ng)
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setFont(QFont("Arial", 12, QFont.Bold))
        nl.addWidget(self.frame_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._on_slider)
        nl.addWidget(self.slider)
        play_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setStyleSheet("QPushButton { background-color: #228B22; font-weight: bold; padding: 2px 6px; }")
        self.play_btn.clicked.connect(self._toggle_play)
        play_row.addWidget(self.play_btn)
        self._speed_combo = QComboBox()
        self._speed_combo.setFixedWidth(64)
        self._speed_options = [1, 2, 5, 10, 30]
        for s in self._speed_options:
            self._speed_combo.addItem(f"{s} fps")
        self._speed_combo.setCurrentIndex(2)
        self._speed_combo.setToolTip("Playback speed")
        play_row.addWidget(self._speed_combo)
        nl.addLayout(play_row)
        root.addWidget(ng)

        # -- display mode --
        dg = QGroupBox("Display (D)")
        dl = QHBoxLayout(dg)
        self._disp_group = QButtonGroup(self)
        self._disp_group.setExclusive(True)
        for i, (label, mode) in enumerate([("Orig", "original"), ("Gamma", "gamma"), ("HSV-V", "hsv_v")]):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setProperty("mode", mode)
            if i == 0:
                b.setChecked(True)
            self._disp_group.addButton(b, i)
            dl.addWidget(b)
        self._disp_group.buttonClicked.connect(lambda btn: self.display_mode_changed.emit(btn.property("mode")))
        root.addWidget(dg)

        # -- edit controls --
        tg = QGroupBox("Edit Controls")
        tl = QVBoxLayout(tg)
        ml = QHBoxLayout()
        ml.addWidget(QLabel("Mode:"))
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self.add_btn = QPushButton("ADD (A)")
        self.add_btn.setCheckable(True)
        self.add_btn.setChecked(True)
        self.add_btn.setStyleSheet("QPushButton:checked { background-color: #228B22; }")
        self._mode_group.addButton(self.add_btn, 0)
        ml.addWidget(self.add_btn)
        self.excl_btn = QPushButton("EXCL (X)")
        self.excl_btn.setCheckable(True)
        self.excl_btn.setStyleSheet("QPushButton:checked { background-color: #cc3333; }")
        self._mode_group.addButton(self.excl_btn, 1)
        ml.addWidget(self.excl_btn)
        self._mode_group.buttonClicked.connect(
            lambda btn: self.correction_mode_changed.emit("add" if btn == self.add_btn else "exclude")
        )
        tl.addLayout(ml)

        trl = QHBoxLayout()
        trl.addWidget(QLabel("Tool:"))
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self.click_btn = QPushButton("Click")
        self.click_btn.setCheckable(True)
        self.click_btn.setChecked(True)
        self._tool_group.addButton(self.click_btn, 0)
        trl.addWidget(self.click_btn)
        self.brush_btn = QPushButton("Brush")
        self.brush_btn.setCheckable(True)
        self._tool_group.addButton(self.brush_btn, 1)
        trl.addWidget(self.brush_btn)
        self._tool_group.buttonClicked.connect(
            lambda btn: self.tool_mode_changed.emit("click" if btn == self.click_btn else "brush")
        )
        tl.addLayout(trl)

        sl = QHBoxLayout()
        sl.addWidget(QLabel("Size:"))
        self.size_label = QLabel(str(self._brush_size))
        self.size_label.setMinimumWidth(24)
        self.size_label.setAlignment(Qt.AlignCenter)
        sl.addWidget(self.size_label)
        for delta, sym in [(-5, "-"), (5, "+")]:
            b = QPushButton(sym)
            b.setFixedWidth(30)
            b.clicked.connect(lambda _, d=delta: self._change_brush(d))
            sl.addWidget(b)
        sl.addStretch()
        tl.addLayout(sl)
        root.addWidget(tg)

        # -- zoom preview --
        zg = QGroupBox("Zoom Preview")
        zl = QVBoxLayout(zg)
        self.zoom_preview = ZoomPreview(size=110, zoom=4)
        zl.addWidget(self.zoom_preview, alignment=Qt.AlignCenter)
        self.pos_label = QLabel("X: --- Y: ---")
        self.pos_label.setAlignment(Qt.AlignCenter)
        zl.addWidget(self.pos_label)
        self._zoom_label = QLabel("Zoom: 100%")
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_label.setStyleSheet("font-size: 11px; color: #8cf; font-weight: bold;")
        zl.addWidget(self._zoom_label)
        root.addWidget(zg)

        # -- overlay mode indicator --
        self._overlay_label = QLabel("Overlay: Normal")
        self._overlay_label.setAlignment(Qt.AlignCenter)
        self._overlay_label.setStyleSheet(
            "font-size: 10px; color: #aaa; padding: 4px; "
            "border: 1px solid #555; border-radius: 3px;"
        )
        root.addWidget(self._overlay_label)

        # -- actions --
        al = QHBoxLayout()
        self.undo_btn = QPushButton("Undo (Z)")
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        self.undo_btn.setEnabled(False)
        al.addWidget(self.undo_btn)
        rb = QPushButton("Reset (R)")
        rb.clicked.connect(self.reset_requested.emit)
        al.addWidget(rb)
        root.addLayout(al)

        al2 = QHBoxLayout()
        sb = QPushButton("Save (S)")
        sb.setStyleSheet("QPushButton { background-color: #1a6ba0; font-weight: bold; }")
        sb.clicked.connect(self.save_requested.emit)
        al2.addWidget(sb)
        pb = QPushButton("Propagate (P)")
        pb.setStyleSheet("QPushButton { background-color: #6a3d9a; font-weight: bold; }")
        pb.clicked.connect(self.propagate_requested.emit)
        al2.addWidget(pb)
        root.addLayout(al2)

        help_btn = QPushButton("? Hotkeys")
        help_btn.setStyleSheet(
            "QPushButton { background: #353535; color: #999; font-size: 10px; "
            "border: 1px solid #555; border-radius: 3px; padding: 4px 6px; }"
            "QPushButton:hover { background: #444; color: #ccc; }"
        )
        help_btn.setFixedHeight(24)
        help_btn.clicked.connect(self.help_requested.emit)
        root.addWidget(help_btn)
        root.addStretch()

    # ── instance slot styling ──────────────────────────────────────────

    @staticmethod
    def _apply_slot_style(btn: QPushButton, rgb_str: str, state: str):
        if state == "dimmed":
            btn.setStyleSheet(
                "QPushButton { background: #333; color: #555; "
                "border: 1px dashed #555; border-radius: 4px; font-weight: bold; }"
                "QPushButton:hover { border-color: #777; color: #888; }"
            )
        elif state == "active":
            btn.setStyleSheet(
                f"QPushButton {{ background: #333; color: #fff; "
                f"border: 2px solid {rgb_str}; border-radius: 4px; font-weight: bold; }}"
            )
        elif state == "has_data":
            btn.setStyleSheet(
                f"QPushButton {{ background: {rgb_str}; color: #fff; "
                f"border: 1px solid {rgb_str}; border-radius: 4px; font-weight: bold; }}"
                f"QPushButton:hover {{ border: 2px solid #fff; }}"
            )

    def refresh_instance_buttons(self, store):
        """Update all slot button styles based on current store state."""
        for key, btn in self._instance_btns.items():
            rgb_str = btn.property("rgb_str")
            layer = store.layers.get(key)
            has_data = layer is not None and (layer.has_mask or layer.points_pos or layer.points_neg)
            self._slot_data[key] = has_data

            if key == self._active_key:
                self._apply_slot_style(btn, rgb_str, "active")
            elif has_data:
                self._apply_slot_style(btn, rgb_str, "has_data")
            else:
                self._apply_slot_style(btn, rgb_str, "dimmed")
            btn.setChecked(key == self._active_key)

    # ── public setters ──────────────────────────────────────────────

    def set_frame_range(self, start: int, end: int):
        self._start, self._end = start, end
        self.slider.setMinimum(start)
        self.slider.setMaximum(end)
        self._refresh_label()

    def set_current_frame(self, frame: int):
        self._current = frame
        self.slider.blockSignals(True)
        self.slider.setValue(frame)
        self.slider.blockSignals(False)
        self._refresh_label()

    def enable_undo(self, on: bool):
        self.undo_btn.setEnabled(on)

    def update_cursor_pos(self, x: int, y: int):
        self.pos_label.setText(f"X: {x}  Y: {y}")

    def update_layer_status(self, statuses: dict[str, bool]):
        """Update category-level mask indicators. *statuses* keyed by category."""
        for cat, has in statuses.items():
            lbl = self._status_labels.get(cat)
            if lbl:
                lbl.setText("HAS MASK" if has else "-")
                lbl.setStyleSheet(f"font-size: 10px; color: {'#4f4' if has else '#888'};")

    def set_active_layer(self, key: str):
        self._active_key = key
        for k, btn in self._instance_btns.items():
            rgb_str = btn.property("rgb_str")
            if k == key:
                self._apply_slot_style(btn, rgb_str, "active")
                btn.setChecked(True)
            elif self._slot_data.get(k, False):
                self._apply_slot_style(btn, rgb_str, "has_data")
                btn.setChecked(False)
            else:
                self._apply_slot_style(btn, rgb_str, "dimmed")
                btn.setChecked(False)

    @property
    def starting_layer(self) -> str:
        return self._start_layer_combo.currentData()

    def set_display_mode(self, mode: str):
        for b in self._disp_group.buttons():
            if b.property("mode") == mode:
                b.setChecked(True)
                break

    def cycle_display_mode(self):
        modes = ["original", "gamma", "hsv_v"]
        cur = self._disp_group.checkedButton().property("mode")
        nxt = modes[(modes.index(cur) + 1) % len(modes)]
        self.set_display_mode(nxt)
        self.display_mode_changed.emit(nxt)

    def set_overlay_mode(self, mode: str):
        labels = {"normal": "Overlay: Normal", "ghost": "Overlay: Ghost", "hidden": "Overlay: Hidden"}
        colors = {"normal": "#aaa", "ghost": "#ff0", "hidden": "#f44"}
        self._overlay_label.setText(labels.get(mode, mode))
        self._overlay_label.setStyleSheet(
            f"font-size: 10px; color: {colors.get(mode, '#aaa')}; padding: 4px; "
            f"border: 1px solid #555; border-radius: 3px;"
        )

    def set_zoom_level(self, level: float):
        pct = int(round(level * 100))
        self._zoom_label.setText(f"Zoom: {pct}%")

    def set_correction_mode(self, mode: str):
        (self.add_btn if mode == "add" else self.excl_btn).setChecked(True)

    def set_tool_mode(self, mode: str):
        (self.click_btn if mode == "click" else self.brush_btn).setChecked(True)

    def stop_playback(self):
        if self._playing:
            self._playing = False
            self.play_btn.setText("Play")
            self.play_btn.setStyleSheet("QPushButton { background-color: #228B22; font-weight: bold; padding: 2px 6px; }")

    def is_playing(self) -> bool:
        return self._playing

    @property
    def brush_size(self) -> int:
        return self._brush_size

    @property
    def playback_fps(self) -> int:
        return self._speed_options[self._speed_combo.currentIndex()]

    # ── internal slots ──────────────────────────────────────────────

    def _on_slider(self, val):
        self._current = val
        self._refresh_label()
        self.frame_changed.emit(val)

    def _refresh_label(self):
        batch_pos = self._current - self._start + 1
        batch_total = self._end - self._start + 1
        self.frame_label.setText(f"Frame {self._current}  ({batch_pos}/{batch_total})")

    def _select_instance(self, key: str):
        self._active_key = key
        for k, btn in self._instance_btns.items():
            btn.setChecked(k == key)
        self.layer_changed.emit(key)

    def _on_vis_toggle(self, name: str, checked: bool):
        self._visibility[name] = checked
        btn = self._vis_btns.get(name)
        if btn:
            btn.setText("Show" if checked else "Hide")
            btn.setChecked(checked)
        self.visibility_changed.emit(name, checked)

    def _change_brush(self, delta):
        self._brush_size = max(5, min(100, self._brush_size + delta))
        self.size_label.setText(str(self._brush_size))
        self.brush_size_changed.emit(self._brush_size)

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self.play_btn.setText("Pause")
            self.play_btn.setStyleSheet("QPushButton { background-color: #cc3333; font-weight: bold; padding: 2px 6px; }")
        else:
            self.play_btn.setText("Play")
            self.play_btn.setStyleSheet("QPushButton { background-color: #228B22; font-weight: bold; padding: 2px 6px; }")
        self.play_toggled.emit(self._playing)

    def get_visibility(self) -> dict[str, bool]:
        return dict(self._visibility)
