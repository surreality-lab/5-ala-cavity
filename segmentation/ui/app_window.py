"""Main annotation window -- orchestrates viewer, control panel, engine."""

import cv2
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QDialog, QLabel,
    QPushButton, QMessageBox, QInputDialog, QApplication,
)
from PyQt5.QtCore import Qt, QTimer

from ..engine.video_reader import VideoReader
from ..engine.mask_store import (
    MaskStore, LAYER_CATEGORIES, CATEGORY_NAMES,
    instance_key, parse_key, instance_color,
)
from ..engine.sam2_engine import SAM2Engine, SAM2_AVAILABLE
from .frame_viewer import FrameViewer, compose_overlays
from .control_panel import ControlPanel
from .review_dialog import ReviewDialog


class AnnotationTool(QMainWindow):
    """Video segmentation tool powered by SAM2."""

    def __init__(self, video_path, output_dir, start_frame=0, end_frame=None, checkpoint=None):
        super().__init__()
        self.video = VideoReader(video_path)
        self.start_frame = start_frame
        self.end_frame = end_frame if end_frame is not None else self.video.total_frames - 1
        self.frames = list(range(self.start_frame, self.end_frame + 1))
        self._frame_to_idx = {f: i for i, f in enumerate(self.frames)}
        self.idx = 0

        self.store = MaskStore(
            base_output_dir=Path(output_dir) / "cavity",
            video_folder=Path(video_path).parent,
        )
        self.sam2 = SAM2Engine(checkpoint)
        self.modified: set[int] = set()

        self.display_mode = "original"
        self.correction_mode = "add"
        self.tool_mode = "click"
        self.brush_size = 20
        self.history: list[dict] = []

        self.zoom_level = 1.0
        self.zoom_center: tuple[int, int] | None = None
        self.last_cursor: tuple[int, int] | None = None

        self.wobble_active = False
        self.wobble_origin = 0
        self.wobble_snapshot: dict | None = None

        self.current_image: np.ndarray | None = None
        self.original_image: np.ndarray | None = None
        self._loading = False
        self.overlay_mode = "normal"

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)

        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._auto_save)
        self._autosave_timer.start(60_000)

        self.store.acquire_lock()
        self._build_ui()
        self._update_layer_label()
        self.viewer.set_cursor_mode("click")
        self._load_frame(self.frames[0], save_current=False)

    # ── UI wiring ───────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Video Segmentation Tool")
        self.setStyleSheet("background-color: #1a1a1a;")

        cw = QWidget()
        self.setCentralWidget(cw)
        outer = QVBoxLayout(cw)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if not self.sam2.available:
            warn = QLabel("  SAM2 not available — click-to-segment disabled. "
                          "Install PyTorch + SAM2 to enable.")
            warn.setStyleSheet(
                "background: #d4a017; color: #1a1a1a; font-weight: bold; "
                "padding: 6px; font-size: 13px;"
            )
            outer.addWidget(warn)
        else:
            info = QLabel(f"  SAM2 ready on {self.sam2.device}  —  Weights: {self.sam2.weights_label}")
            info.setStyleSheet(
                "background: #1a5c2a; color: #cfc; font-weight: bold; "
                "padding: 6px; font-size: 13px;"
            )
            outer.addWidget(info)

        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        outer.addLayout(lay, stretch=1)

        self.viewer = FrameViewer()
        self.viewer.mouse_moved.connect(self._on_mouse_move)
        self.viewer.mouse_clicked.connect(self._on_click)
        self.viewer.mouse_dragged.connect(self._on_drag)
        self.viewer.mouse_released.connect(self._on_brush_release)
        self.viewer.wheel_zoomed.connect(self._on_wheel_zoom)
        lay.addWidget(self.viewer, stretch=1)

        self.cp = ControlPanel()
        self.cp.set_frame_range(self.start_frame, self.end_frame)
        self.cp.frame_changed.connect(self._on_slider)
        self.cp.display_mode_changed.connect(self._set_display)
        self.cp.correction_mode_changed.connect(lambda m: setattr(self, 'correction_mode', m))
        self.cp.tool_mode_changed.connect(self._set_tool)
        self.cp.brush_size_changed.connect(lambda s: setattr(self, 'brush_size', s))
        self.cp.undo_requested.connect(self._undo)
        self.cp.reset_requested.connect(self._reset)
        self.cp.play_toggled.connect(self._on_play)
        self.cp.layer_changed.connect(self._set_layer)
        self.cp.visibility_changed.connect(lambda *_: self._refresh_viewer())
        self.cp.invert_toggled.connect(self._toggle_invert)
        self.cp.save_requested.connect(self._save_current)
        self.cp.propagate_requested.connect(self._start_batch_propagate)
        self.cp.help_requested.connect(self._show_help)
        lay.addWidget(self.cp)

        sb = self.statusBar()
        sb.setStyleSheet("QStatusBar { background: #222; color: #aaa; font-size: 12px; }")
        self._update_status_bar()

        avail = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(
            min(960, avail.width()),
            min(600, avail.height()),
        )
        self.resize(avail.size())
        self._first_show_done = False
        self.showMaximized()

    # ── geometry safeguard ───────────────────────────────────────────

    def showEvent(self, event):
        super().showEvent(event)
        if not self._first_show_done:
            self._first_show_done = True
            QTimer.singleShot(0, self._fit_to_screen)

    def _fit_to_screen(self):
        """Clamp window to available screen rect after OS chrome is applied."""
        avail = QApplication.primaryScreen().availableGeometry()
        frame = self.frameGeometry()
        if not avail.contains(frame):
            self.setGeometry(avail)

    # ── layer / instance helpers ─────────────────────────────────────

    def _update_layer_label(self):
        """Push active layer name + color to the viewer overlay."""
        layer = self.store.active
        bgr = layer.color_bgr
        label = layer.key.replace("_", " ").title()
        self.viewer.set_layer_label(label, (bgr[2], bgr[1], bgr[0]))

    def _all_keys(self) -> list[str]:
        return list(self.store.layers.keys())

    def _any_data(self) -> bool:
        return any(
            l.has_mask or l.points_pos or l.points_neg
            for l in self.store.layers.values()
        )

    def _category_has_mask(self) -> dict[str, bool]:
        """Return per-category bool indicating if any instance has a mask."""
        result = {}
        for cat in CATEGORY_NAMES:
            result[cat] = any(l.has_mask for l in self.store.instances_for(cat))
        return result

    # ── instance management ──────────────────────────────────────────

    def _ensure_instance(self, key: str):
        """Ensure instance exists in the store, creating it if needed."""
        if key in self.store.layers:
            return True
        cat, _ = parse_key(key)
        result = self.store.add_instance(cat)
        if result is None:
            cfg = LAYER_CATEGORIES[cat]
            self.statusBar().showMessage(
                f"Max {cfg['max_instances']} {cat} instances", 2000
            )
            return False
        return True

    # ── frame loading ───────────────────────────────────────────────

    def _load_frame(self, fidx, save_current=True, propagate=True):
        if self._loading or fidx not in self._frame_to_idx:
            return
        self._loading = True
        try:
            self._load_frame_inner(fidx, save_current, propagate)
        finally:
            self._loading = False

    def _load_frame_inner(self, fidx, save_current, propagate):
        if save_current and hasattr(self, 'idx'):
            cur = self.frames[self.idx]
            if self._any_data():
                self.store.save_frame(cur)
                self.modified.add(cur)

            if propagate and SAM2_AVAILABLE:
                new_i = self._frame_to_idx.get(fidx)
                if new_i is not None and new_i == self.idx + 1:
                    active_masks = self.store.get_active_masks_for_propagation()
                    if active_masks and not self._frame_has_masks(fidx):
                        src = self.video.read_original(cur)
                        dst = self.video.read_original(fidx)
                        if src is not None and dst is not None:
                            results = self.sam2.propagate_single(
                                src, dst,
                                [(oid, m) for _, oid, m in active_masks],
                            )
                            if results:
                                self.store.load_frame(fidx)
                                self.store.apply_propagated(results)
                                self.cp.refresh_instance_buttons(self.store)
                                self._post_load(fidx)
                                return

        self.idx = self._frame_to_idx[fidx]
        self.store.load_frame(fidx)
        self.cp.refresh_instance_buttons(self.store)
        self._post_load(fidx)

    def _post_load(self, fidx):
        self.idx = self._frame_to_idx[fidx]
        self._update_image()
        self.history.clear()
        self.cp.enable_undo(False)
        self.cp.set_current_frame(fidx)

        start_pref = self.cp.starting_layer
        if start_pref != "auto":
            target = instance_key(start_pref, 1)
            if target in self.store.layers:
                self._set_layer(target)
                self.cp.set_active_layer(target)

        self._refresh_viewer()

    def _frame_has_masks(self, fidx) -> bool:
        for cat in CATEGORY_NAMES:
            cfg = LAYER_CATEGORIES[cat]
            for i in range(1, cfg["max_instances"] + 1):
                mp, _ = self.store._find_mask(cat, i, fidx)
                if mp and mp.exists():
                    return True
        return False

    def _update_image(self):
        fidx = self.frames[self.idx]
        self.original_image = self.video.read_original(fidx)
        self.current_image = self.video.read_display(fidx, self.display_mode)
        if self.original_image is not None and self.sam2.available:
            self.sam2.set_image(self.original_image)
        self.cp.zoom_preview.set_source(self.current_image)

    def _cycle_overlay_mode(self):
        cycle = {"normal": "ghost", "ghost": "hidden", "hidden": "normal"}
        self.overlay_mode = cycle[self.overlay_mode]
        self.cp.set_overlay_mode(self.overlay_mode)
        self._refresh_viewer()

    def _refresh_viewer(self):
        if self.current_image is None:
            return
        display = compose_overlays(
            self.current_image,
            self.store.layers,
            self.store.active_layer_name,
            self.cp.get_visibility(),
            overlay_mode=self.overlay_mode,
        )
        self.viewer.set_image(display, self.zoom_level, self.zoom_center)
        self.viewer.set_brush_params(self.brush_size, self.tool_mode == "brush")
        self.cp.zoom_preview.set_brush_params(self.brush_size, self.tool_mode == "brush")
        self.cp.update_layer_status(self._category_has_mask())
        self.cp.refresh_instance_buttons(self.store)
        self._update_status_bar()

    def _update_status_bar(self):
        active = self.store.active
        label = active.key.replace("_", " ").title()
        mask_state = "Has mask" if active.has_mask else "No mask"
        device = self.sam2.device if self.sam2.available else "N/A"
        zoom = f"{self.zoom_level:.1f}x"
        disp = self.display_mode.replace("_", "-").upper()
        batch_pos = self.idx + 1
        batch_total = len(self.frames)
        fidx = self.frames[self.idx]
        unsaved = "*" if fidx in self.modified else ""
        inv = " [INV]" if active.inverted else ""
        self.statusBar().showMessage(
            f"  {unsaved}Layer: {label}{inv}  |  {mask_state}  |  "
            f"Frame {fidx} ({batch_pos}/{batch_total})  |  "
            f"Display: {disp}  |  SAM2: {device}  |  Zoom: {zoom}"
        )

    # ── editing ─────────────────────────────────────────────────────

    def _push_history(self):
        layer = self.store.active
        entry = layer.copy().__dict__
        entry["_layer_key"] = self.store.active_layer_name
        self.history.append(entry)
        if len(self.history) > 20:
            self.history.pop(0)
        self.cp.enable_undo(True)

    def _undo(self):
        if not self.history:
            return
        snap = self.history.pop()
        target_key = snap.pop("_layer_key", self.store.active_layer_name)
        layer = self.store.layers.get(target_key)
        if layer:
            layer.mask = snap["mask"]
            layer.points_pos = snap["points_pos"]
            layer.points_neg = snap["points_neg"]
            layer.inverted = snap.get("inverted", False)
        self.cp.enable_undo(bool(self.history))
        self._refresh_viewer()

    def _reset(self):
        self._push_history()
        self.store.reset_all()
        self._refresh_viewer()

    def _paint(self, x, y):
        layer = self.store.active
        if self.current_image is None:
            return
        h, w = self.current_image.shape[:2]
        if layer.mask is None:
            layer.mask = np.zeros((h, w), dtype=np.uint8)
        val = 255 if self.correction_mode == "add" else 0
        cv2.circle(layer.mask, (x, y), self.brush_size, val, -1)
        if val == 255:
            self.store.subtract_from_others(self.store.active_layer_name)

    def _click_sam2(self, x, y, force_exclude=False):
        if not self.sam2.available:
            self.statusBar().showMessage("SAM2 not loaded — clicks have no effect", 3000)
            return
        layer = self.store.active
        if force_exclude or self.correction_mode != "add":
            layer.points_neg.append((x, y))
        else:
            layer.points_pos.append((x, y))

        prior_negs = self.store.sample_prior_negatives(layer.category)
        all_neg = layer.points_neg + prior_negs

        result = self.sam2.predict_mask(layer.points_pos, all_neg)
        if result is not None:
            if layer.inverted:
                result = cv2.bitwise_not(result)
            layer.mask = result
            self.store.hard_subtract_priors(layer.category)
            self.store.subtract_from_others(self.store.active_layer_name)

    # ── event handlers ──────────────────────────────────────────────

    def _on_mouse_move(self, x, y):
        self.cp.update_cursor_pos(x, y)
        self.cp.zoom_preview.update_position(x, y)
        self.last_cursor = (x, y)

    def _on_click(self, x, y, is_left):
        if self.wobble_active or self._loading:
            return
        if self.cp.is_playing():
            self.cp.stop_playback()
            self.play_timer.stop()
        self._push_history()
        self.modified.add(self.frames[self.idx])
        if self.tool_mode == "brush":
            self._paint(x, y)
        elif not is_left:
            self._click_sam2(x, y, force_exclude=True)
        else:
            self._click_sam2(x, y)
        self._refresh_viewer()

    def _on_drag(self, x, y):
        if self.wobble_active or self._loading:
            return
        if self.tool_mode == "brush":
            self._paint(x, y)
            self.viewer.update()

    def _on_brush_release(self):
        if self.tool_mode == "brush":
            self._refresh_viewer()

    def _on_slider(self, fidx):
        self._load_frame(fidx)

    def _set_display(self, mode):
        self.display_mode = mode
        self._update_image()
        self._refresh_viewer()

    def _set_tool(self, mode):
        self.tool_mode = mode
        self.viewer.set_cursor_mode(mode)
        self._refresh_viewer()

    def _set_layer(self, key: str):
        self._ensure_instance(key)
        self.store.set_active(key)
        self.history.clear()
        self.cp.enable_undo(False)
        self.cp.invert_btn.setChecked(self.store.active.inverted)
        cat, _ = parse_key(key)
        self.cp.invert_btn.setEnabled(cat == "cavity")
        self._update_layer_label()
        self._refresh_viewer()

    def _toggle_invert(self):
        layer = self.store.active
        if layer.category != "cavity":
            self.cp.invert_btn.setChecked(False)
            return
        self._push_history()
        layer.inverted = not layer.inverted
        self.cp.invert_btn.setChecked(layer.inverted)
        if layer.points_pos or layer.points_neg:
            prior_negs = self.store.sample_prior_negatives(layer.category)
            all_neg = layer.points_neg + prior_negs
            result = self.sam2.predict_mask(layer.points_pos, all_neg)
            if result is not None:
                if layer.inverted:
                    result = cv2.bitwise_not(result)
                layer.mask = result
                self.store.hard_subtract_priors(layer.category)
                self.store.subtract_from_others(self.store.active_layer_name)
        self.modified.add(self.frames[self.idx])
        self._refresh_viewer()

    def _save_current(self):
        fidx = self.frames[self.idx]
        self.store.save_frame(fidx)
        self.modified.add(fidx)
        self.statusBar().showMessage(f"  Saved frame {fidx}", 2000)

    # ── batch propagation ────────────────────────────────────────────

    def _start_batch_propagate(self):
        if not self.sam2.available:
            self.statusBar().showMessage("SAM2 not available — cannot propagate", 3000)
            return
        active_masks = self.store.get_active_masks_for_propagation()
        if not active_masks:
            self.statusBar().showMessage("No masks to propagate — annotate at least one layer first", 3000)
            return
        remaining = len(self.frames) - 1 - self.idx
        if remaining < 1:
            self.statusBar().showMessage("No frames ahead to propagate to", 3000)
            return
        n, ok = QInputDialog.getInt(
            self, "Batch Propagate",
            f"Propagate masks forward how many frames?\n(max {remaining})",
            value=min(20, remaining), min=1, max=remaining,
        )
        if not ok:
            return
        self.statusBar().showMessage(f"  Propagating {n} frames...")
        QApplication.processEvents()

        target_frames = []
        for i in range(self.idx, self.idx + n + 1):
            img = self.video.read_original(self.frames[i])
            if img is not None:
                target_frames.append(img)
        if len(target_frames) < 2:
            return

        results = self.sam2.propagate_batch(
            target_frames,
            [(oid, m) for _, oid, m in active_masks],
            progress_cb=lambda cur, total: self.statusBar().showMessage(
                f"  Propagating... {cur}/{total}"
            ),
        )
        if not results:
            self.statusBar().showMessage("Propagation produced no results", 3000)
            return

        review_data = []
        for sam_idx in sorted(results.keys()):
            if sam_idx == 0:
                continue
            real_frame = self.frames[self.idx + sam_idx]
            img = self.video.read_original(real_frame)
            if img is not None:
                review_data.append((real_frame, img, results[sam_idx]))

        if not review_data:
            return

        dlg = ReviewDialog(review_data, parent=self)
        result_code = dlg.exec_()
        if result_code == QDialog.Rejected:
            self.statusBar().showMessage("Propagation cancelled", 2000)
            return

        cutoff = dlg.get_selected_index()
        accepted = review_data if cutoff is None else review_data[:cutoff]
        for fidx_r, _, masks_dict in accepted:
            self.store.load_frame(fidx_r)
            self.store.apply_propagated(masks_dict)
            self.store.save_frame(fidx_r)
            self.modified.add(fidx_r)

        self.statusBar().showMessage(f"  Saved {len(accepted)} propagated frames", 3000)
        if accepted:
            last = accepted[-1][0]
            self._load_frame(last, save_current=False, propagate=False)

    # ── playback ────────────────────────────────────────────────────

    def _on_play(self, playing):
        if playing:
            fps = self.cp.playback_fps
            self.play_timer.start(int(1000 / fps))
        else:
            self.play_timer.stop()

    def _play_tick(self):
        if self.idx < len(self.frames) - 1:
            self._load_frame(self.frames[self.idx + 1])
        else:
            self.cp.stop_playback()
            self.play_timer.stop()

    def _auto_save(self):
        if self._loading or self.wobble_active:
            return
        fidx = self.frames[self.idx]
        if self._any_data():
            self.store.save_frame(fidx)
            self.modified.add(fidx)

    # ── wobble preview ──────────────────────────────────────────────

    def _wobble(self, direction):
        peek = self.idx + direction
        if peek < 0 or peek >= len(self.frames):
            return
        if not self.wobble_active:
            self.wobble_active = True
            self.wobble_origin = self.idx
            self.wobble_snapshot = {k: l.copy() for k, l in self.store.layers.items()}
        if peek == self.wobble_origin:
            self._wobble_return()
            return
        self.idx = peek
        fidx = self.frames[peek]
        self.current_image = self.video.read_display(fidx, self.display_mode)
        for k in self._all_keys():
            self.store.reset_layer(k)
        self.cp.set_current_frame(fidx)
        self._refresh_viewer()

    def _wobble_return(self):
        if not self.wobble_active:
            return
        self.wobble_active = False
        self.idx = self.wobble_origin
        fidx = self.frames[self.idx]
        self.current_image = self.video.read_display(fidx, self.display_mode)
        if self.wobble_snapshot:
            for k, snap in self.wobble_snapshot.items():
                self.store.layers[k] = snap
        self.cp.set_current_frame(fidx)
        self._refresh_viewer()

    # ── zoom ────────────────────────────────────────────────────────

    MAX_ZOOM = 4.0

    def _zoom_in(self, step=0.1):
        if self.zoom_level >= self.MAX_ZOOM:
            return
        if self.last_cursor:
            self.zoom_center = self.last_cursor
        elif self.zoom_center is None and self.current_image is not None:
            h, w = self.current_image.shape[:2]
            self.zoom_center = (w // 2, h // 2)
        self.zoom_level = min(self.MAX_ZOOM, round(self.zoom_level + step, 2))
        self.cp.set_zoom_level(self.zoom_level)
        self._refresh_viewer()

    def _zoom_out(self, step=0.1):
        if self.zoom_level <= 1.0:
            return
        self.zoom_level = max(1.0, round(self.zoom_level - step, 2))
        if self.zoom_level == 1.0:
            self.zoom_center = None
        self.cp.set_zoom_level(self.zoom_level)
        self._refresh_viewer()

    def _on_wheel_zoom(self, delta, img_x, img_y):
        if img_x is not None:
            self.zoom_center = (img_x, img_y)
        if delta > 0:
            self._zoom_in(step=0.15)
        else:
            self._zoom_out(step=0.15)

    def _toggle_max_zoom(self):
        if self.zoom_level < self.MAX_ZOOM:
            if self.last_cursor:
                self.zoom_center = self.last_cursor
            elif self.zoom_center is None and self.current_image is not None:
                h, w = self.current_image.shape[:2]
                self.zoom_center = (w // 2, h // 2)
            self.zoom_level = self.MAX_ZOOM
        else:
            self.zoom_level = 1.0
            self.zoom_center = None
        self.cp.set_zoom_level(self.zoom_level)
        self._refresh_viewer()

    # ── keyboard ────────────────────────────────────────────────────

    def _cycle_instance(self, direction: int):
        """Cycle through instances within the current category that have data."""
        cat = self.store.active.category
        instances = self.store.instances_for(cat)
        with_data = [l for l in instances if l.has_mask or l.points_pos or l.points_neg]
        if len(with_data) <= 1:
            return
        keys = [l.key for l in with_data]
        cur = self.store.active_layer_name
        try:
            idx = keys.index(cur)
        except ValueError:
            idx = 0
        nxt = keys[(idx + direction) % len(keys)]
        self._set_layer(nxt)
        self.cp.set_active_layer(nxt)

    def keyPressEvent(self, event):
        key = event.key()
        mod = event.modifiers()

        if self.wobble_active and key not in (Qt.Key_Comma, Qt.Key_Period):
            self._wobble_return()

        if key == Qt.Key_Comma:
            self._wobble(-1)
        elif key == Qt.Key_Period:
            self._wobble(1)
        elif key == Qt.Key_BracketLeft:
            if self.idx > 0:
                self._load_frame(self.frames[self.idx - 1])
        elif key == Qt.Key_BracketRight:
            if self.idx < len(self.frames) - 1:
                self._load_frame(self.frames[self.idx + 1])
        elif key == Qt.Key_D:
            self.cp.cycle_display_mode()
        elif key == Qt.Key_A:
            self.correction_mode = "add"
            self.cp.set_correction_mode("add")
        elif key == Qt.Key_X:
            self.correction_mode = "exclude"
            self.cp.set_correction_mode("exclude")
        elif key == Qt.Key_B:
            self.tool_mode = "click" if self.tool_mode == "brush" else "brush"
            self.cp.set_tool_mode(self.tool_mode)
            self._refresh_viewer()
        elif key == Qt.Key_C:
            self._set_layer("cavity")
            self.cp.set_active_layer("cavity")
        elif key == Qt.Key_T:
            first = self.store.instances_for("tool")
            if first:
                k = first[0].key
                self._set_layer(k)
                self.cp.set_active_layer(k)
        elif key == Qt.Key_J:
            first = self.store.instances_for("object")
            if first:
                k = first[0].key
                self._set_layer(k)
                self.cp.set_active_layer(k)
        elif key == Qt.Key_H:
            first = self.store.instances_for("hand")
            if first:
                k = first[0].key
                self._set_layer(k)
                self.cp.set_active_layer(k)
        elif key == Qt.Key_I:
            self._toggle_invert()
        elif key == Qt.Key_Tab:
            self._cycle_instance(1)
        elif key == Qt.Key_Backtab:
            self._cycle_instance(-1)
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            if mod & Qt.ShiftModifier:
                self.brush_size = min(100, self.brush_size + 5)
                self.cp._brush_size = self.brush_size
                self.cp.size_label.setText(str(self.brush_size))
                self._refresh_viewer()
            else:
                self._zoom_in()
        elif key == Qt.Key_Minus:
            if mod & Qt.ShiftModifier:
                self.brush_size = max(5, self.brush_size - 5)
                self.cp._brush_size = self.brush_size
                self.cp.size_label.setText(str(self.brush_size))
                self._refresh_viewer()
            else:
                self._zoom_out()
        elif key == Qt.Key_F:
            self._toggle_max_zoom()
        elif key == Qt.Key_1:
            self._toggle_cat_visibility("cavity")
        elif key == Qt.Key_2:
            self._toggle_cat_visibility("tool")
        elif key == Qt.Key_3:
            self._toggle_cat_visibility("object")
        elif key == Qt.Key_4:
            self._toggle_cat_visibility("hand")
        elif key == Qt.Key_Z:
            self._undo()
        elif key == Qt.Key_R:
            self._reset()
        elif key == Qt.Key_S:
            self._save_current()
        elif key == Qt.Key_P:
            self._start_batch_propagate()
        elif key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        elif key == Qt.Key_Space:
            self.cp._toggle_play()
        elif key == Qt.Key_O:
            self._cycle_overlay_mode()
        elif key == Qt.Key_Question or (key == Qt.Key_Slash and mod & Qt.ShiftModifier):
            self._show_help()

    def _toggle_cat_visibility(self, cat: str):
        cur = self.cp._visibility.get(cat, True)
        self.cp._on_vis_toggle(cat, not cur)
        btn = self.cp._vis_btns.get(cat)
        if btn:
            btn.setChecked(not cur)

    # ── help dialog ─────────────────────────────────────────────────

    _SHORTCUTS = [
        ("[ / ]", "Previous / Next frame"),
        (", / .", "Wobble preview left / right"),
        ("Space", "Play / Pause"),
        ("D", "Cycle display mode"),
        ("O", "Cycle overlay mode"),
        ("A / X", "Add / Exclude correction mode"),
        ("Right-click", "Exclude point (shortcut for X + click)"),
        ("B", "Toggle click / brush tool"),
        ("Shift + / -", "Brush size +5 / -5"),
        ("+ / - / Scroll", "Zoom in / out"),
        ("F", "Toggle max zoom at cursor"),
        ("C / T / J / H", "Switch to Cavity / Tool / Object / Hand"),
        ("Tab / Shift-Tab", "Cycle instances within category"),
        ("1 / 2 / 3 / 4", "Toggle Cavity / Tool / Object / Hand visibility"),
        ("I", "Invert mode (cavity only)"),
        ("S", "Save current frame"),
        ("P", "Propagate masks forward (batch)"),
        ("Z", "Undo"),
        ("R", "Reset current layer"),
        ("?", "Show this help"),
        ("Q / Esc", "Quit"),
    ]

    def _show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        dlg.setMinimumWidth(420)
        dlg.setStyleSheet(
            "QDialog { background: #2a2a2a; color: #eee; }"
            "QLabel { color: #ddd; font-size: 13px; }"
        )
        layout = QVBoxLayout(dlg)
        title = QLabel("Keyboard Shortcuts")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #fff; margin-bottom: 8px;")
        layout.addWidget(title)
        for key, desc in self._SHORTCUTS:
            row = QLabel(f"<span style='color:#66bbff; font-weight:bold;'>{key}</span>"
                         f"&nbsp;&nbsp;&mdash;&nbsp;&nbsp;{desc}")
            layout.addWidget(row)
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            "QPushButton { background: #444; border: 1px solid #555; border-radius: 4px; "
            "padding: 6px 18px; color: #eee; } QPushButton:hover { background: #555; }"
        )
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()

    # ── close ───────────────────────────────────────────────────────

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Quit",
            f"Save and quit?\n({len(self.modified)} frames modified this session)",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Cancel:
            event.ignore()
            return

        if reply == QMessageBox.Save:
            fidx = self.frames[self.idx]
            if self._any_data():
                self.store.save_frame(fidx)
                self.modified.add(fidx)

        summary_path = self.store.write_session_summary(
            self.video.video_path, self.start_frame, self.end_frame, self.modified,
        )
        print(f"Saved {len(self.modified)} frames -> {summary_path}")
        self.store.release_lock()
        self.video.release()
        event.accept()
