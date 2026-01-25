#!/usr/bin/env python3
"""
Polished Cavity Definition Tool with PyQt5 UI.

Features:
- Split layout: main frame view (left) + control panel (right)
- Frame navigation with draggable slider and step buttons
- Display mode toggle (Original/Gamma/HSV-V)
- Tool mode: Add/Exclude with Click/Brush
- Zoom preview with brush halo visualization
- Propagation with matrix review
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil
import sys
import os


def _opencv_videoio_build_summary():
    """Return a small, relevant subset of cv2.getBuildInformation()."""
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return "(cv2.getBuildInformation() unavailable)"

    keywords = (
        "Video I/O",
        "FFMPEG",
        "GStreamer",
        "v4l",
        "V4L",
        "avcodec",
        "avformat",
        "avutil",
        "swscale",
        "Media I/O",
        "CUDA",
    )
    lines = []
    for line in info.splitlines():
        if any(k.lower() in line.lower() for k in keywords):
            lines.append(line.rstrip())

    if not lines:
        return "(no Video I/O lines found in build info)"
    return "\n".join(lines[:80])


def _opencv_runtime_diagnostics(video_path: Path | None = None):
    parts = [
        f"cv2 version: {getattr(cv2, '__version__', 'unknown')}",
        f"cv2 file: {getattr(cv2, '__file__', 'unknown')}",
    ]
    if video_path is not None:
        try:
            resolved = video_path.resolve()
        except Exception:
            resolved = video_path
        parts.append(f"video path: {video_path}")
        parts.append(f"video exists: {video_path.exists()}")
        parts.append(f"video resolved: {resolved}")
        if video_path.exists():
            try:
                parts.append(f"video size: {video_path.stat().st_size} bytes")
            except OSError:
                pass

    return "\n".join(parts)


def _try_open_video(video_path: Path):
    """Try multiple backends to open a video; returns (cap, backend_name) or (None, None)."""
    candidates = [("ANY", None)]
    if hasattr(cv2, "CAP_FFMPEG"):
        candidates.append(("FFMPEG", cv2.CAP_FFMPEG))
    if hasattr(cv2, "CAP_GSTREAMER"):
        candidates.append(("GSTREAMER", cv2.CAP_GSTREAMER))

    for name, api in candidates:
        try:
            if api is None:
                cap = cv2.VideoCapture(str(video_path))
            else:
                cap = cv2.VideoCapture(str(video_path), api)
        except Exception:
            cap = None

        if cap is not None and cap.isOpened():
            return cap, name

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

    return None, None

# Add utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
from common import find_video_file

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSlider, QButtonGroup, QFrame, QSizePolicy,
    QGroupBox, QGridLayout, QDialog, QScrollArea, QMessageBox, QProgressDialog,
    QComboBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont

# SAM2 imports
try:
    import torch
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM2 not available. Click mode will be disabled.")


class StartupDialog(QDialog):
    """Startup dialog to select video and frame range."""
    
    BATCH_SIZE = 500
    
    def __init__(self, base_dir, parent=None):
        super().__init__(parent)
        self.base_dir = Path(base_dir)
        self.setWindowTitle("Cavity Tool - Select Video & Range")
        self.setMinimumSize(600, 400)
        
        self.selected_video = None
        self.selected_range = None
        self.video_info = {}
        
        self._init_ui()
        self._scan_videos()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Cavity Definition Tool")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4a9eff;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Folder selection
        folder_group = QGroupBox("Base Directory")
        folder_layout = QHBoxLayout(folder_group)
        self.folder_label = QLabel(str(self.base_dir))
        self.folder_label.setStyleSheet("font-size: 12px; padding: 5px; background-color: #2b2b2b; border-radius: 3px;")
        folder_layout.addWidget(self.folder_label, stretch=1)
        self.folder_btn = QPushButton("Browse...")
        self.folder_btn.clicked.connect(self._select_folder)
        folder_layout.addWidget(self.folder_btn)
        layout.addWidget(folder_group)
        
        # Video selection
        video_group = QGroupBox("Select Video")
        video_layout = QVBoxLayout(video_group)
        self.video_combo = QComboBox()
        self.video_combo.currentTextChanged.connect(self._on_video_selected)
        video_layout.addWidget(self.video_combo)
        layout.addWidget(video_group)
        
        # Info panel
        info_group = QGroupBox("Video Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("Select a video to see details")
        self.info_label.setStyleSheet("font-size: 14px;")
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_group)
        
        # Range selection
        range_group = QGroupBox("Select Frame Range (500 frames per batch)")
        self.range_layout = QVBoxLayout(range_group)
        self.range_scroll = QScrollArea()
        self.range_scroll.setWidgetResizable(True)
        self.range_widget = QWidget()
        self.range_grid = QGridLayout(self.range_widget)
        self.range_scroll.setWidget(self.range_widget)
        self.range_layout.addWidget(self.range_scroll)
        layout.addWidget(range_group)
        
        # Pre-render instructions
        self.prerender_label = QLabel("")
        self.prerender_label.setStyleSheet("color: #ffcc00; font-size: 11px;")
        self.prerender_label.setWordWrap(True)
        layout.addWidget(self.prerender_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("Open Tool")
        self.open_btn.setStyleSheet("background-color: #228B22; padding: 10px; font-weight: bold;")
        self.open_btn.clicked.connect(self.accept)
        self.open_btn.setEnabled(False)
        btn_layout.addWidget(self.open_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("padding: 10px;")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def _scan_videos(self):
        """Scan for available MP4 videos."""
        if not self.base_dir.exists():
            self.info_label.setText("Selected directory does not exist!")
            return
        
        # Find all MP4 files recursively
        mp4_files = list(self.base_dir.rglob("*.mp4"))
        mp4_files.extend(self.base_dir.rglob("*.MP4"))
        
        if not mp4_files:
            self.info_label.setText("No MP4 videos found in selected directory!")
            return
        
        # Sort by path
        mp4_files = sorted(mp4_files)
        
        for video_path in mp4_files:
            video_folder = video_path.parent
            
            # Use relative path from base_dir as display name
            try:
                rel_path = video_path.relative_to(self.base_dir)
                name = str(rel_path)
            except ValueError:
                name = video_path.name
            
            # Get video info
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                total_frames = 0
                fps = 0.0
                open_ok = False
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                open_ok = True
            cap.release()
            
            # Check saved masks (new location first, then legacy locations)
            masks_dir_new = video_folder / "masks" / "cavity"
            masks_dir_legacy_1 = video_folder / "pipeline" / "02_cavity" / "cavity_only" / "frames"
            masks_dir_legacy_2 = video_folder / "pipeline" / "02_cavity" / "masks"
            masks_saved = 0
            masks_dir_used = None
            if masks_dir_new.exists():
                masks_saved = len(list(masks_dir_new.glob("frame_*")))
                masks_dir_used = masks_dir_new
            elif masks_dir_legacy_1.exists():
                masks_saved = len(list(masks_dir_legacy_1.glob("frame_*")))
                masks_dir_used = masks_dir_legacy_1
            elif masks_dir_legacy_2.exists():
                masks_saved = len(list(masks_dir_legacy_2.glob("frame_*")))
                masks_dir_used = masks_dir_legacy_2
            
            self.video_info[name] = {
                'path': video_path,
                'folder': video_folder,
                'total_frames': total_frames,
                'fps': fps,
                'masks_saved': masks_saved,
                'masks_dir': masks_dir_used,
                'opencv_open_ok': open_ok,
            }
            
            self.video_combo.addItem(name)
    
    def _select_folder(self):
        """Allow user to select a different base directory."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Directory Containing MP4 Videos",
            str('/opt/5-ALA-Videos' if Path('/opt/5-ALA-Videos').exists() else str(Path.home())),
            QFileDialog.ShowDirsOnly
        )
        
        if folder:
            self.base_dir = Path(folder)
            self.folder_label.setText(str(self.base_dir))
            
            # Clear existing video info and rescan
            self.video_info.clear()
            self.video_combo.clear()
            self.selected_video = None
            self.selected_range = None
            self.info_label.setText("Select a video to see details")
            self.open_btn.setEnabled(False)
            
            # Clear range buttons
            while self.range_grid.count():
                item = self.range_grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Rescan videos in new directory
            self._scan_videos()
    
    def _on_video_selected(self, name):
        if name not in self.video_info:
            return
        
        self.selected_video = name
        info = self.video_info[name]
        
        # Update range buttons
        self._update_range_buttons(info)
    
    def _update_range_buttons(self, info):
        # Clear existing buttons
        while self.range_grid.count():
            item = self.range_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.range_buttons = QButtonGroup(self)
        self.range_buttons.setExclusive(True)
        self.range_buttons.buttonClicked.connect(self._on_range_selected)
        
        total = info['total_frames']
        
        col = 0
        row = 0
        max_cols = 5
        
        for start in range(0, total, self.BATCH_SIZE):
            end = min(start + self.BATCH_SIZE - 1, total - 1)
            range_size = end - start + 1
            
            # Count masks in this range
            masks_count = self._count_masks_in_range(info, start, end)
            mask_pct = (masks_count / range_size * 100) if range_size > 0 else 0
            
            # Button text shows range and mask progress
            if masks_count > 0:
                btn_text = f"{start}-{end}\n({masks_count} masks)"
            else:
                btn_text = f"{start}-{end}"
            
            btn = QPushButton(btn_text)
            btn.setCheckable(True)
            btn.setProperty("range", (start, end))
            btn.setMinimumHeight(50)
            
            # Color by mask completion (all ranges available now)
            if mask_pct >= 90:
                btn.setStyleSheet("background-color: #228B22;")  # Green - mostly done
            elif mask_pct >= 50:
                btn.setStyleSheet("background-color: #4a9eff;")  # Blue - in progress
            elif mask_pct > 0:
                btn.setStyleSheet("background-color: #cc8800;")  # Orange - started
            else:
                btn.setStyleSheet("background-color: #444;")  # Gray - not started
                btn.setStyleSheet("background-color: #333;")
            
            self.range_buttons.addButton(btn)
            self.range_grid.addWidget(btn, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Legend - simple text
        legend = QLabel("Colors: Green=Complete | Blue=In Progress | Orange=Started | Gray=Not Started")
        legend.setStyleSheet("color: #666; font-size: 10px;")
        self.range_grid.addWidget(legend, row + 1, 0, 1, max_cols)
    
    def _on_range_selected(self, btn):
        self.selected_range = btn.property("range")
        start, end = self.selected_range
        
        info = self.video_info[self.selected_video]
        
        # Count masks in this range
        masks_in_range = self._count_masks_in_range(info, start, end)
        range_size = end - start + 1
        
        # All ranges are now available (no pre-rendering needed)
        self.open_btn.setEnabled(True)
        pct = (masks_in_range / range_size * 100) if range_size > 0 else 0
        self.open_btn.setText(f"Open Tool ({masks_in_range}/{range_size} masks = {pct:.0f}%)")
        self.prerender_label.setText("")
    
    def _count_masks_in_range(self, info, start, end):
        """Count how many masks exist in the given frame range."""
        masks_dir = info.get('masks_dir')
        if masks_dir is None:
            # Try new location first, then legacy locations
            folder = info.get('folder')
            if folder:
                masks_dir_new = folder / "masks" / "cavity"
                masks_dir_legacy = folder / "pipeline" / "02_cavity" / "cavity_only" / "frames"
                if masks_dir_new.exists():
                    masks_dir = masks_dir_new
                elif masks_dir_legacy.exists():
                    masks_dir = masks_dir_legacy
        
        if masks_dir is None or not masks_dir.exists():
            return 0
        
        count = 0
        for frame_idx in range(start, end + 1):
            mask_path = masks_dir / f"frame_{frame_idx:06d}" / "cavity_mask.png"
            if mask_path.exists():
                count += 1
        return count
    
    def get_selection(self):
        """Return (video_info, start_frame, end_frame) or None."""
        if self.selected_video and self.selected_range:
            return (self.video_info[self.selected_video], 
                    self.selected_range[0], 
                    self.selected_range[1])
        return None


class FrameViewer(QLabel):
    """Main frame display widget with mouse interaction."""
    
    mouse_moved = pyqtSignal(int, int)  # x, y in image coords
    mouse_clicked = pyqtSignal(int, int, bool)  # x, y, is_left_button
    mouse_dragged = pyqtSignal(int, int)  # x, y during drag
    mouse_released = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a;")
        
        self._image = None
        self._display_pixmap = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        
        # Brush visualization
        self._brush_size = 20
        self._brush_mode = False
        self._show_brush = False
        self._cursor_pos = QPoint(0, 0)
        
        # Drag state
        self._is_dragging = False
    
    def set_image(self, img_bgr):
        """Set image from BGR numpy array."""
        if img_bgr is None:
            self._image = None
            self._display_pixmap = None
            self.clear()
            return
        
        self._image = img_bgr.copy()
        self._update_display()
    
    def set_brush_params(self, size, mode_on):
        """Update brush visualization parameters."""
        self._brush_size = size
        self._brush_mode = mode_on
        self.update()
    
    def _update_display(self):
        """Convert image to pixmap and scale to fit."""
        if self._image is None:
            return
        
        h, w = self._image.shape[:2]
        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit widget while maintaining aspect ratio
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._display_pixmap = scaled
        
        # Calculate scale and offset for coordinate mapping
        self._scale = scaled.width() / w
        self._offset = QPoint(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2
        )
        
        self.setPixmap(scaled)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()
    
    def _widget_to_image_coords(self, pos):
        """Convert widget coordinates to image coordinates."""
        if self._image is None:
            return None, None
        
        # Adjust for centering offset
        x = (pos.x() - self._offset.x()) / self._scale
        y = (pos.y() - self._offset.y()) / self._scale
        
        h, w = self._image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return int(x), int(y)
        return None, None
    
    def mouseMoveEvent(self, event):
        self._cursor_pos = event.pos()
        self._show_brush = True
        
        x, y = self._widget_to_image_coords(event.pos())
        if x is not None:
            self.mouse_moved.emit(x, y)
            if self._is_dragging:
                self.mouse_dragged.emit(x, y)
        
        self.update()  # Trigger repaint for brush cursor
    
    def mousePressEvent(self, event):
        x, y = self._widget_to_image_coords(event.pos())
        if x is not None:
            is_left = event.button() == Qt.LeftButton
            self.mouse_clicked.emit(x, y, is_left)
            if is_left and self._brush_mode:
                self._is_dragging = True
    
    def mouseReleaseEvent(self, event):
        if self._is_dragging:
            self._is_dragging = False
            self.mouse_released.emit()
    
    def leaveEvent(self, event):
        self._show_brush = False
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Draw brush cursor overlay
        if self._show_brush and self._brush_mode and self._display_pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Brush circle
            radius = int(self._brush_size * self._scale)
            pen = QPen(QColor(255, 255, 255, 200), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self._cursor_pos, radius, radius)
            
            # Inner dot
            painter.setBrush(QBrush(QColor(255, 255, 255, 150)))
            painter.drawEllipse(self._cursor_pos, 3, 3)
            
            painter.end()


class ZoomPreview(QLabel):
    """Magnified preview of cursor area."""
    
    def __init__(self, size=150, zoom=4, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setStyleSheet("border: 2px solid #444; background-color: #222;")
        self._zoom = zoom
        self._size = size
        self._source_image = None
        self._brush_size = 20
        self._brush_mode = False
    
    def set_source(self, img_bgr):
        """Set source image for zoom."""
        self._source_image = img_bgr
    
    def set_brush_params(self, size, mode_on):
        self._brush_size = size
        self._brush_mode = mode_on
    
    def update_position(self, x, y):
        """Update zoom to show area around (x, y)."""
        if self._source_image is None:
            return
        
        h, w = self._source_image.shape[:2]
        
        # Calculate crop region
        crop_size = self._size // self._zoom
        x1 = max(0, x - crop_size // 2)
        y1 = max(0, y - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        
        # Adjust if at edge
        if x2 - x1 < crop_size:
            x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size:
            y1 = max(0, y2 - crop_size)
        
        crop = self._source_image[y1:y2, x1:x2].copy()
        
        # Scale up
        zoomed = cv2.resize(crop, (self._size, self._size), interpolation=cv2.INTER_NEAREST)
        
        # Draw brush circle if in brush mode
        if self._brush_mode:
            center = self._size // 2
            radius = int(self._brush_size * self._zoom)
            cv2.circle(zoomed, (center, center), radius, (255, 255, 255), 1)
        
        # Draw crosshair
        cv2.line(zoomed, (self._size//2 - 10, self._size//2), 
                 (self._size//2 + 10, self._size//2), (0, 255, 255), 1)
        cv2.line(zoomed, (self._size//2, self._size//2 - 10),
                 (self._size//2, self._size//2 + 10), (0, 255, 255), 1)
        
        # Convert to pixmap
        rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, self._size, self._size, self._size * 3, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))


class ControlPanel(QWidget):
    """Right-side control panel."""
    
    # Signals
    frame_changed = pyqtSignal(int)
    display_mode_changed = pyqtSignal(str)
    correction_mode_changed = pyqtSignal(str)  # 'add' or 'exclude'
    tool_mode_changed = pyqtSignal(str)  # 'click' or 'brush'
    brush_size_changed = pyqtSignal(int)
    save_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)  # True=play, False=pause
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setStyleSheet("""
            QWidget { background-color: #2a2a2a; color: #eee; }
            QGroupBox { 
                border: 1px solid #444; 
                border-radius: 4px; 
                margin-top: 8px; 
                padding-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 8px; 
                padding: 0 4px; 
            }
            QPushButton {
                background-color: #444;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 24px;
            }
            QPushButton:hover { background-color: #555; }
            QPushButton:pressed { background-color: #333; }
            QPushButton:checked { 
                background-color: #0066cc; 
                border-color: #0088ff;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #444;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #cc3333;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #cc3333;
                border-radius: 4px;
            }
        """)
        
        self._start_frame = 0
        self._end_frame = 0
        self._current_frame = 0
        self._brush_size = 20
        self._is_playing = False
        
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # --- Frame Navigation ---
        nav_group = QGroupBox("Frame Navigation")
        nav_layout = QVBoxLayout(nav_group)
        
        # Frame counter
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setFont(QFont("Arial", 14, QFont.Bold))
        nav_layout.addWidget(self.frame_label)
        
        # Slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        nav_layout.addWidget(self.frame_slider)
        
        # Play/Pause button
        play_layout = QHBoxLayout()
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #228B22;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover { background-color: #2aa62a; }
        """)
        self.play_pause_btn.clicked.connect(self._toggle_play)
        play_layout.addWidget(self.play_pause_btn)
        nav_layout.addLayout(play_layout)
        
        layout.addWidget(nav_group)
        
        # --- Display Mode ---
        display_group = QGroupBox("Display Mode (D to cycle)")
        display_layout = QHBoxLayout(display_group)
        
        self.display_buttons = QButtonGroup(self)
        self.display_buttons.setExclusive(True)
        
        for i, (name, mode) in enumerate([("Orig", "original"), ("Gamma", "gamma"), ("HSV-V", "hsv_v")]):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setProperty("mode", mode)
            if i == 0:
                btn.setChecked(True)
            self.display_buttons.addButton(btn, i)
            display_layout.addWidget(btn)
        
        self.display_buttons.buttonClicked.connect(self._on_display_changed)
        layout.addWidget(display_group)
        
        # --- Tool Mode ---
        tool_group = QGroupBox("Tool Mode")
        tool_layout = QVBoxLayout(tool_group)
        
        # Add/Exclude row
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        
        self.mode_buttons = QButtonGroup(self)
        self.mode_buttons.setExclusive(True)
        
        self.add_btn = QPushButton("ADD (A)")
        self.add_btn.setCheckable(True)
        self.add_btn.setChecked(True)
        self.add_btn.setStyleSheet("QPushButton:checked { background-color: #228B22; }")
        self.mode_buttons.addButton(self.add_btn, 0)
        mode_layout.addWidget(self.add_btn)
        
        self.exclude_btn = QPushButton("EXCL (X)")
        self.exclude_btn.setCheckable(True)
        self.exclude_btn.setStyleSheet("QPushButton:checked { background-color: #cc3333; }")
        self.mode_buttons.addButton(self.exclude_btn, 1)
        mode_layout.addWidget(self.exclude_btn)
        
        self.mode_buttons.buttonClicked.connect(self._on_mode_changed)
        tool_layout.addLayout(mode_layout)
        
        # Click/Brush row
        tool_row = QHBoxLayout()
        tool_row.addWidget(QLabel("Tool:"))
        
        self.tool_buttons = QButtonGroup(self)
        self.tool_buttons.setExclusive(True)
        
        self.click_btn = QPushButton("Click")
        self.click_btn.setCheckable(True)
        self.click_btn.setChecked(True)
        self.tool_buttons.addButton(self.click_btn, 0)
        tool_row.addWidget(self.click_btn)
        
        self.brush_btn = QPushButton("Brush")
        self.brush_btn.setCheckable(True)
        self.tool_buttons.addButton(self.brush_btn, 1)
        tool_row.addWidget(self.brush_btn)
        
        self.tool_buttons.buttonClicked.connect(self._on_tool_changed)
        tool_layout.addLayout(tool_row)
        
        # Brush size row
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        
        self.size_label = QLabel(str(self._brush_size))
        self.size_label.setMinimumWidth(30)
        self.size_label.setAlignment(Qt.AlignCenter)
        size_layout.addWidget(self.size_label)
        
        self.size_down_btn = QPushButton("-")
        self.size_down_btn.setFixedWidth(30)
        self.size_down_btn.clicked.connect(lambda: self._change_brush_size(-5))
        size_layout.addWidget(self.size_down_btn)
        
        self.size_up_btn = QPushButton("+")
        self.size_up_btn.setFixedWidth(30)
        self.size_up_btn.clicked.connect(lambda: self._change_brush_size(5))
        size_layout.addWidget(self.size_up_btn)
        
        size_layout.addStretch()
        tool_layout.addLayout(size_layout)
        
        layout.addWidget(tool_group)
        
        # --- Zoom Preview ---
        zoom_group = QGroupBox("Zoom Preview")
        zoom_layout = QVBoxLayout(zoom_group)
        
        self.zoom_preview = ZoomPreview(size=150, zoom=4)
        zoom_layout.addWidget(self.zoom_preview, alignment=Qt.AlignCenter)
        
        # Position bar
        self.pos_label = QLabel("X: --- Y: ---")
        self.pos_label.setAlignment(Qt.AlignCenter)
        zoom_layout.addWidget(self.pos_label)
        
        layout.addWidget(zoom_group)
        
        # --- Actions ---
        action_layout = QHBoxLayout()
        
        self.undo_btn = QPushButton("Undo (Z)")
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        self.undo_btn.setEnabled(False)  # Disabled by default
        action_layout.addWidget(self.undo_btn)
        
        self.reset_btn = QPushButton("Reset (R)")
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        action_layout.addWidget(self.reset_btn)
        
        layout.addLayout(action_layout)
        
        layout.addStretch()
    
    def enable_undo(self, enabled):
        """Enable or disable undo button."""
        self.undo_btn.setEnabled(enabled)
    
    def set_frame_range(self, start_frame, end_frame):
        """Set the frame range for display and slider."""
        self._start_frame = start_frame
        self._end_frame = end_frame
        self.frame_slider.setMinimum(start_frame)
        self.frame_slider.setMaximum(end_frame)
        self._update_frame_label()
    
    def set_current_frame(self, frame):
        self._current_frame = frame
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame)
        self.frame_slider.blockSignals(False)
        self._update_frame_label()
    
    def _update_frame_label(self):
        self.frame_label.setText(f"Frame: {self._current_frame} / {self._end_frame}")
    
    def _on_slider_changed(self, value):
        self._current_frame = value
        self._update_frame_label()
        self.frame_changed.emit(value)
    
    def _on_display_changed(self, btn):
        mode = btn.property("mode")
        self.display_mode_changed.emit(mode)
    
    def set_display_mode(self, mode):
        """Programmatically set display mode."""
        for btn in self.display_buttons.buttons():
            if btn.property("mode") == mode:
                btn.setChecked(True)
                break
    
    def cycle_display_mode(self):
        """Cycle to next display mode."""
        modes = ["original", "gamma", "hsv_v"]
        current = self.display_buttons.checkedButton().property("mode")
        idx = modes.index(current)
        next_mode = modes[(idx + 1) % len(modes)]
        self.set_display_mode(next_mode)
        self.display_mode_changed.emit(next_mode)
    
    def _on_mode_changed(self, btn):
        mode = "add" if btn == self.add_btn else "exclude"
        self.correction_mode_changed.emit(mode)
    
    def set_correction_mode(self, mode):
        if mode == "add":
            self.add_btn.setChecked(True)
        else:
            self.exclude_btn.setChecked(True)
    
    def _on_tool_changed(self, btn):
        tool = "click" if btn == self.click_btn else "brush"
        self.tool_mode_changed.emit(tool)
    
    def set_tool_mode(self, mode):
        if mode == "click":
            self.click_btn.setChecked(True)
        else:
            self.brush_btn.setChecked(True)
    
    def _change_brush_size(self, delta):
        self._brush_size = max(5, min(100, self._brush_size + delta))
        self.size_label.setText(str(self._brush_size))
        self.brush_size_changed.emit(self._brush_size)
    
    def get_brush_size(self):
        return self._brush_size
    
    def update_cursor_pos(self, x, y):
        self.pos_label.setText(f"X: {x} Y: {y}")
    
    def _toggle_play(self):
        """Toggle play/pause state."""
        self._is_playing = not self._is_playing
        if self._is_playing:
            self.play_pause_btn.setText("⏸ Pause")
            self.play_pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: #cc3333;
                    font-weight: bold;
                    padding: 8px;
                }
                QPushButton:hover { background-color: #dd4444; }
            """)
        else:
            self.play_pause_btn.setText("▶ Play")
            self.play_pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: #228B22;
                    font-weight: bold;
                    padding: 8px;
                }
                QPushButton:hover { background-color: #2aa62a; }
            """)
        self.play_toggled.emit(self._is_playing)
    
    def is_playing(self):
        """Return current play state."""
        return self._is_playing
    
    def stop_playback(self):
        """Stop playback (set to paused state)."""
        if self._is_playing:
            self._is_playing = False
            self.play_pause_btn.setText("▶ Play")
            self.play_pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: #228B22;
                    font-weight: bold;
                    padding: 8px;
                }
                QPushButton:hover { background-color: #2aa62a; }
            """)


class CavityTool(QMainWindow):
    """Main cavity annotation tool window."""
    
    def __init__(self, video_path, output_dir, start_frame=0, end_frame=None, checkpoint_path=None):
        super().__init__()
        
        self.video_path = Path(video_path) if video_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        
        # Track modified frames for auto-save on close
        self.modified_frames = set()
        
        # Validate video exists
        if not self.video_path or not self.video_path.exists():
            raise ValueError(f"Video file not found: {self.video_path}")
        
        # Open video and get info
        self.video_cap, backend = _try_open_video(self.video_path)
        if self.video_cap is None:
            diag = _opencv_runtime_diagnostics(self.video_path)
            raise ValueError(
                "Could not open video with OpenCV.\n\n"
                f"{diag}\n\n"
                "Suggested debugging:\n"
                "- Run: OPENCV_VIDEOIO_DEBUG=1 python define_cavity_polished.py ...\n"
                "- Check build info for 'FFMPEG: YES' or 'GStreamer: YES'.\n"
                "- Validate the MP4 with ffprobe/ffmpeg (codec support).\n"
            )
        self.video_backend = backend or "ANY"
        
        total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        
        # Set frame range
        self.start_frame = start_frame
        self.end_frame = end_frame if end_frame is not None else total_frames - 1
        self.frame_indices = list(range(self.start_frame, self.end_frame + 1))
        
        if not self.frame_indices:
            raise ValueError(f"No frames in range {start_frame}-{end_frame}")
        
        # Frame cache to avoid re-reading from video
        self.frame_cache = {}  # {frame_idx: original_bgr}
        self.cache_size = 10  # Keep last 10 frames in memory
        
        if not self.frame_indices:
            raise ValueError(
                f"No frames found in: {self.original_dir}\n\n"
                f"Frames must be named: frame_XXXXXX.png (6-digit zero-padded)\n"
                f"Example: frame_000000.png, frame_000001.png, etc.\n\n"
                f"Run the pre-render script to create properly named frames."
            )
        
        # Set starting frame
        self.current_idx = 0
        if start_frame is not None and start_frame in self.frame_indices:
            self.current_idx = self.frame_indices.index(start_frame)
        
        # Current state
        self.display_mode = "original"
        self.correction_mode = "add"
        self.tool_mode = "click"
        self.brush_size = 20
        
        # Current frame data
        self.current_image = None  # Display image (BGR)
        self.cavity_mask = None
        self.cavity_points_pos = []
        self.cavity_points_neg = []
        self.history = []
        
        # Comparison overlay (for navigation)
        self.comparison_mask = None
        self.reference_frame_idx = None  # Track which frame the reference mask came from
        self.working_frame_idx = None
        self.working_mask = None
        self.working_points_pos = []
        self.working_points_neg = []
        
        # Wobble preview state (,/. keys)
        self.wobble_active = False
        self.wobble_origin_idx = 0
        self.wobble_saved_mask = None
        self.wobble_saved_pos = []
        self.wobble_saved_neg = []
        
        # SAM2 predictors
        self.predictor = None  # For click-based segmentation
        self.video_predictor = None  # Lazy-loaded only when propagation is needed (saves memory)
        if SAM2_AVAILABLE and checkpoint_path:
            self._init_sam2()
        
        # Play timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_timer_tick)
        
        self._init_ui()
        self._load_frame(self.frame_indices[0] if self.frame_indices else 0, auto_save_current=False)
    
    @staticmethod
    def _apply_gamma(img, gamma=2.2):
        """Apply gamma correction."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img, table)
    
    @staticmethod
    def _apply_clahe(img, clip_limit=3.0, tile_size=8):
        """Apply CLAHE for contrast enhancement."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def _apply_bilateral(img, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter."""
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    def _process_gamma(self, frame_bgr, gamma=2.2):
        """Apply gamma + CLAHE + bilateral."""
        gamma_corrected = self._apply_gamma(frame_bgr, gamma)
        clahe_enhanced = self._apply_clahe(gamma_corrected)
        bilateral = self._apply_bilateral(clahe_enhanced)
        return bilateral
    
    def _process_hsv_v(self, frame_bgr):
        """Extract HSV Value channel as BGR."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        return cv2.cvtColor(v_channel, cv2.COLOR_GRAY2BGR)
    
    def _load_frame_from_video(self, frame_idx):
        """Load and crop frame from video."""
        # Check cache first
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx].copy()
        
        # Read from video
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        if not ret:
            return None
        
        # Import crop_ui
        sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
        from common import crop_ui
        
        # Crop UI elements
        frame_cropped = crop_ui(frame)
        
        # Cache it
        self.frame_cache[frame_idx] = frame_cropped.copy()
        
        # Keep cache size limited
        if len(self.frame_cache) > self.cache_size:
            oldest = min(self.frame_cache.keys())
            del self.frame_cache[oldest]
        
        return frame_cropped
    
    def _init_sam2(self):
        """Initialize SAM2 predictor."""
        try:
            # Use GPU if available
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            print(f"Initializing SAM2 on {device}...")
            
            config_name = "sam2_hiera_l.yaml"

            # Build model WITHOUT relying on a base checkpoint.
            # This works as long as the fine-tuned checkpoint contains a full model_state_dict.
            sam2_model = build_sam2(config_name, None, device=device)
            sam2_model = sam2_model.to(device)
            self.predictor = SAM2ImagePredictor(sam2_model)

            if not self.checkpoint_path or not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"SAM2 fine-tuned checkpoint not found: {self.checkpoint_path}"
                )

            print(f"Loading SAM2 weights from (fine-tuned only): {self.checkpoint_path}")
            checkpoint = torch.load(str(self.checkpoint_path), map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.predictor.model.load_state_dict(state_dict)
            if isinstance(checkpoint, dict):
                print(f"✓ Loaded fine-tuned checkpoint from step: {checkpoint.get('step', 'unknown')}")
                if 'mean_iou' in checkpoint:
                    print(f"  Checkpoint mean IoU: {checkpoint['mean_iou']:.4f}")
            
            self.predictor.model.eval()
            print("SAM2 ready")
        except Exception as e:
            print(f"Failed to initialize SAM2: {e}")
            import traceback
            traceback.print_exc()
            self.predictor = None
    
    def _init_ui(self):
        self.setWindowTitle("Cavity Definition Tool")
        self.setStyleSheet("background-color: #1a1a1a;")
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Frame viewer (left)
        self.frame_viewer = FrameViewer()
        self.frame_viewer.mouse_moved.connect(self._on_mouse_move)
        self.frame_viewer.mouse_clicked.connect(self._on_mouse_click)
        self.frame_viewer.mouse_dragged.connect(self._on_mouse_drag)
        self.frame_viewer.mouse_released.connect(self._on_mouse_release)
        layout.addWidget(self.frame_viewer, stretch=1)
        
        # Control panel (right)
        self.control_panel = ControlPanel()
        self.control_panel.set_frame_range(self.start_frame, self.end_frame)
        self.control_panel.frame_changed.connect(self._on_frame_changed)
        self.control_panel.display_mode_changed.connect(self._on_display_mode_changed)
        self.control_panel.correction_mode_changed.connect(self._on_correction_mode_changed)
        self.control_panel.tool_mode_changed.connect(self._on_tool_mode_changed)
        self.control_panel.brush_size_changed.connect(self._on_brush_size_changed)
        self.control_panel.undo_requested.connect(self._undo)
        self.control_panel.reset_requested.connect(self._reset)
        self.control_panel.play_toggled.connect(self._on_play_toggled)
        layout.addWidget(self.control_panel)
        
        # Load initial frame
        initial_frame = self.frame_indices[self.current_idx]
        self._load_frame(initial_frame, auto_save_current=False)
        
        # Show fullscreen
        self.showMaximized()
    
    def _find_existing_mask(self, frame_idx):
        """Search for existing mask for this frame and video.
        
        Searches in order:
        1. Current output directory (masks/cavity)
        2. Legacy pipeline directory structure
        
        Returns: (mask_path, data_path) tuple or (None, None)
        """
        # First check current output directory
        mask_path = self.output_dir / f"frame_{frame_idx:06d}" / "cavity_mask.png"
        data_path = self.output_dir / f"frame_{frame_idx:06d}" / "instance_data.json"
        
        if mask_path.exists():
            return mask_path, data_path
        
        # Search in legacy locations for backwards compatibility
        if self.video_path:
            video_folder = self.video_path.parent
            
            # Legacy locations where masks might be stored
            search_paths = [
                video_folder / "pipeline" / "02_cavity" / "cavity_only" / "frames" / f"frame_{frame_idx:06d}",
                video_folder / "pipeline" / "02_cavity" / "masks" / f"frame_{frame_idx:06d}",
            ]
            
            for search_dir in search_paths:
                mask_p = search_dir / "cavity_mask.png"
                data_p = search_dir / "instance_data.json"
                if mask_p.exists():
                    return mask_p, data_p
        
        return None, None
    
    def _load_frame(self, frame_idx, auto_save_current=True, auto_propagate=True):
        """Load frame and its masks.
        
        Args:
            frame_idx: Frame index to load
            auto_save_current: If True, save current frame before loading new one
            auto_propagate: If True, auto-propagate mask from current to next frame if moving forward
        """
        if frame_idx not in self.frame_indices:
            return
        
        # Auto-save current frame before navigating away (if it has modifications)
        if auto_save_current and hasattr(self, 'current_idx'):
            current_frame_idx = self.frame_indices[self.current_idx]
            if self.cavity_mask is not None or len(self.cavity_points_pos) > 0 or len(self.cavity_points_neg) > 0:
                # Save without advancing
                self._save_current_silent(current_frame_idx)
                
                # Auto-propagate if moving forward to next frame
                # Only propagate if destination frame doesn't have annotation yet
                if auto_propagate and SAM2_AVAILABLE and self.cavity_mask is not None:
                    new_idx = self.frame_indices.index(frame_idx)
                    # Check if moving forward by 1 frame
                    if new_idx == self.current_idx + 1:
                        # Check if destination frame already has a mask
                        dst_mask_path, _ = self._find_existing_mask(frame_idx)
                        if dst_mask_path is None or not dst_mask_path.exists():
                            # No existing mask - propagate from current frame
                            print(f"[Auto-propagate] Frame {frame_idx} has no annotation, propagating from {current_frame_idx}")
                            propagated = self._propagate_mask_to_next(current_frame_idx, frame_idx)
                            if propagated is not None:
                                # Temporarily store propagated mask to load it
                                self._temp_propagated_mask = propagated
                        else:
                            # Existing mask found - skip propagation, will load existing annotation
                            print(f"[Auto-propagate] Frame {frame_idx} already has annotation, skipping propagation")
        
        self.current_idx = self.frame_indices.index(frame_idx)
        
        # Load image based on display mode
        self._update_display_image()
        
        # Check if we have a temp propagated mask to use
        if hasattr(self, '_temp_propagated_mask') and self._temp_propagated_mask is not None:
            self.cavity_mask = self._temp_propagated_mask
            self._temp_propagated_mask = None
            self.comparison_mask = None
            self.reference_frame_idx = None
            self.cavity_points_pos = []
            self.cavity_points_neg = []
        else:
            # Search for existing mask
            mask_path, data_path = self._find_existing_mask(frame_idx)
            
            if mask_path and mask_path.exists():
                self.cavity_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                self.comparison_mask = None  # Clear reference when we have our own mask
                self.reference_frame_idx = None
                
                # Load points if available
                if data_path and data_path.exists():
                    with open(data_path) as f:
                        data = json.load(f)
                        self.cavity_points_pos = data.get('cavity_points_pos', [])
                        self.cavity_points_neg = data.get('cavity_points_neg', [])
                else:
                    self.cavity_points_pos = []
                    self.cavity_points_neg = []
            else:
                self.cavity_mask = None
                self.cavity_points_pos = []
                self.cavity_points_neg = []
                # Look backwards for the most recent saved mask to use as reference
                self._load_reference_mask(frame_idx)
        
        self.history = []
        self.control_panel.enable_undo(False)  # Disable undo when loading new frame
        self.control_panel.set_current_frame(frame_idx)
        self._update_viewer()
    
    def _update_display_image(self):
        """Load and process current frame based on display mode."""
        frame_idx = self.frame_indices[self.current_idx]
        
        # Load original frame from video
        original = self._load_frame_from_video(frame_idx)
        if original is None:
            self.current_image = None
            return
        
        # Apply processing based on display mode
        if self.display_mode == "original":
            self.current_image = original
        elif self.display_mode == "gamma":
            self.current_image = self._process_gamma(original)
        else:  # hsv_v
            self.current_image = self._process_hsv_v(original)
        
        # Update SAM2 predictor with current image
        if self.predictor is not None:
            rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(rgb)
        
        self.control_panel.zoom_preview.set_source(self.current_image)
    
    def _update_viewer(self):
        """Update frame viewer with current image and overlays."""
        if self.current_image is None:
            return
        
        display = self.current_image.copy()
        
        # Draw comparison mask (reference from previous saved frame)
        if self.comparison_mask is not None and self.cavity_mask is None:
            contours, _ = cv2.findContours(self.comparison_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Yellow outline for reference (not editable)
            cv2.drawContours(display, contours, -1, (0, 200, 255), 2)
            # Add semi-transparent fill
            overlay = display.copy()
            overlay[self.comparison_mask > 127] = [0, 200, 255]
            display = cv2.addWeighted(display, 0.85, overlay, 0.15, 0)
            # Draw text showing reference source
            if self.reference_frame_idx is not None:
                ref_text = f"Ref: Frame {self.reference_frame_idx}"
                cv2.putText(display, ref_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 200, 255), 2, cv2.LINE_AA)
        
        # Draw cavity mask (editable)
        if self.cavity_mask is not None:
            contours, _ = cv2.findContours(self.cavity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (128, 0, 128), 2)
            
            overlay = display.copy()
            overlay[self.cavity_mask > 127] = [200, 100, 200]
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
            cv2.drawContours(display, contours, -1, (128, 0, 128), 2)
        
        # Draw points
        for x, y in self.cavity_points_pos:
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.circle(display, (x, y), 6, (0, 0, 0), 1)
        
        for x, y in self.cavity_points_neg:
            cv2.circle(display, (x, y), 6, (0, 0, 255), 2)
            cv2.line(display, (x-4, y-4), (x+4, y+4), (0, 0, 255), 2)
            cv2.line(display, (x-4, y+4), (x+4, y-4), (0, 0, 255), 2)
        
        self.frame_viewer.set_image(display)
        self.frame_viewer.set_brush_params(self.brush_size, self.tool_mode == "brush")
        self.control_panel.zoom_preview.set_brush_params(self.brush_size, self.tool_mode == "brush")
    
    def _save_state(self):
        """Save state for undo."""
        state = {
            'cavity_mask': self.cavity_mask.copy() if self.cavity_mask is not None else None,
            'cavity_points_pos': list(self.cavity_points_pos),
            'cavity_points_neg': list(self.cavity_points_neg),
        }
        self.history.append(state)
        if len(self.history) > 20:
            self.history.pop(0)
        
        # Enable undo button since we have history
        self.control_panel.enable_undo(True)
    
    def _undo(self):
        if self.history:
            state = self.history.pop()
            self.cavity_mask = state['cavity_mask']
            self.cavity_points_pos = state['cavity_points_pos']
            self.cavity_points_neg = state['cavity_points_neg']
            self._update_viewer()
            
            # Disable undo if no more history
            if len(self.history) == 0:
                self.control_panel.enable_undo(False)
    
    def _reset(self):
        self._save_state()
        self.cavity_mask = None
        self.cavity_points_pos = []
        self.cavity_points_neg = []
        self._update_viewer()
    
    def _save_current_silent(self, frame_idx):
        """Save frame's mask and data silently (no print, no advance)."""
        frame_dir = self.output_dir / f"frame_{frame_idx:06d}"
        frame_dir.mkdir(exist_ok=True)
        
        # Save mask
        if self.cavity_mask is not None:
            cv2.imwrite(str(frame_dir / "cavity_mask.png"), self.cavity_mask)
        
        # Save points
        data = {
            'cavity_points_pos': self.cavity_points_pos,
            'cavity_points_neg': self.cavity_points_neg,
        }
        with open(frame_dir / "instance_data.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        # Track this frame as modified
        self.modified_frames.add(frame_idx)
    
    def _save_current(self, advance=True):
        """Save current frame's mask and data, optionally advance to next frame."""
        frame_idx = self.frame_indices[self.current_idx]
        frame_dir = self.output_dir / f"frame_{frame_idx:06d}"
        frame_dir.mkdir(exist_ok=True)
        
        # Save original
        if self.current_image is not None:
            orig_path = self.original_dir / f"frame_{frame_idx:06d}.png"
            if orig_path.exists():
                orig = cv2.imread(str(orig_path))
                cv2.imwrite(str(frame_dir / "original.png"), orig)
        
        # Save mask
        if self.cavity_mask is not None:
            cv2.imwrite(str(frame_dir / "cavity_mask.png"), self.cavity_mask)
        
        # Save points
        data = {
            'cavity_points_pos': self.cavity_points_pos,
            'cavity_points_neg': self.cavity_points_neg,
        }
        with open(frame_dir / "instance_data.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save visualization
        vis = cv2.imread(str(self.original_dir / f"frame_{frame_idx:06d}.png"))
        if vis is not None and self.cavity_mask is not None:
            contours, _ = cv2.findContours(self.cavity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 3)
            overlay = vis.copy()
            overlay[self.cavity_mask > 127] = [0, 200, 0]
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 3)
            cv2.imwrite(str(frame_dir / "visualization.png"), vis)
        
        # Track this frame as modified
        self.modified_frames.add(frame_idx)
        
        # Clear working state after save
        self.working_frame_idx = None
        self.working_mask = None
        self.working_points_pos = []
        self.working_points_neg = []
        self.comparison_mask = None
        
        # Advance to next frame
        if advance and self.current_idx < len(self.frame_indices) - 1:
            next_frame = self.frame_indices[self.current_idx + 1]
            self._load_frame(next_frame)
    
    def _load_reference_mask(self, current_frame_idx):
        """Load the most recent saved mask as a reference overlay."""
        # Search backwards from current frame for the most recent saved mask
        for check_idx in range(self.current_idx - 1, -1, -1):
            check_frame = self.frame_indices[check_idx]
            check_mask_path = self.output_dir / f"frame_{check_frame:06d}" / "cavity_mask.png"
            if check_mask_path.exists():
                self.comparison_mask = cv2.imread(str(check_mask_path), cv2.IMREAD_GRAYSCALE)
                self.reference_frame_idx = check_frame
                return
        
        # No previous mask found
        self.comparison_mask = None
        self.reference_frame_idx = None
    
    def _run_sam2(self):
        """Run SAM2 segmentation with current points."""
        if self.predictor is None or len(self.cavity_points_pos) == 0:
            return
        
        all_points = self.cavity_points_pos + self.cavity_points_neg
        all_labels = [1] * len(self.cavity_points_pos) + [0] * len(self.cavity_points_neg)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array(all_points),
            point_labels=np.array(all_labels),
            multimask_output=False
        )
        
        full_mask = (masks[0] * 255).astype(np.uint8)
        
        # Keep only components containing positive points
        num_labels, labels = cv2.connectedComponents(full_mask)
        keep = set()
        for px, py in self.cavity_points_pos:
            label = labels[py, px]
            if label > 0:
                keep.add(label)
        
        filtered = np.zeros_like(full_mask)
        for label_id in keep:
            filtered[labels == label_id] = 255
        
        self.cavity_mask = filtered
    
    def _paint_brush(self, x, y):
        """Paint with brush at position."""
        if self.current_image is None:
            return
        
        h, w = self.current_image.shape[:2]
        if self.cavity_mask is None:
            self.cavity_mask = np.zeros((h, w), dtype=np.uint8)
        
        if self.correction_mode == "add":
            cv2.circle(self.cavity_mask, (x, y), self.brush_size, 255, -1)
        else:
            cv2.circle(self.cavity_mask, (x, y), self.brush_size, 0, -1)
    
    def _propagate_mask_to_next(self, src_frame_idx, dst_frame_idx):
        """Use SAM2 to propagate mask from src to dst frame (for auto-propagation).
        Returns the propagated mask or None if failed."""
        if not SAM2_AVAILABLE:
            print("[Propagation] SAM2 not available")
            return None
        
        if self.cavity_mask is None:
            print("[Propagation] No mask to propagate")
            return None
        
        try:
            from sam2.build_sam import build_sam2_video_predictor
            print(f"[Propagation] Propagating mask from frame {src_frame_idx} to {dst_frame_idx}...")
            
            # Create temp directory with 2 frames
            temp_dir = tempfile.mkdtemp()
            
            # Get current frames from video
            src_img = self._load_frame_from_video(src_frame_idx)
            dst_img = self._load_frame_from_video(dst_frame_idx)
            
            if src_img is None or dst_img is None:
                print(f"[Propagation] Failed to load frames: src={src_img is not None}, dst={dst_img is not None}")
                shutil.rmtree(temp_dir)
                return None
            
            # Save to temp directory
            cv2.imwrite(f"{temp_dir}/00000.jpg", src_img)
            cv2.imwrite(f"{temp_dir}/00001.jpg", dst_img)
            
            # Build/reuse video predictor
            if self.video_predictor is None:
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                print(f"[Propagation] Building video predictor on {device}...")
                config_name = "sam2_hiera_l.yaml"

                # Build WITHOUT relying on a base checkpoint; load fine-tuned weights below.
                self.video_predictor = build_sam2_video_predictor(
                    config_name,
                    None,
                    device=device
                )
                
                # Load fine-tuned weights if available
                if self.checkpoint_path and self.checkpoint_path.exists():
                    print(f"[Propagation] Loading fine-tuned weights from {self.checkpoint_path}")
                    checkpoint = torch.load(str(self.checkpoint_path), map_location=device, weights_only=False)
                    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                    self.video_predictor.load_state_dict(state_dict)
                    self.video_predictor.eval()
                print("[Propagation] Video predictor ready")
            
            # Initialize state and add mask
            inference_state = self.video_predictor.init_state(video_path=temp_dir)
            self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                mask=(self.cavity_mask > 127).astype(np.float32)
            )
            
            # Propagate
            propagated_mask = None
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                if out_frame_idx == 1:  # Destination frame
                    for i, obj_id in enumerate(out_obj_ids):
                        if obj_id == 1:
                            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                            propagated_mask = (mask * 255).astype(np.uint8)
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            if propagated_mask is not None:
                print(f"[Propagation] ✓ Successfully propagated mask")
            else:
                print(f"[Propagation] ✗ No mask generated")
            
            return propagated_mask
            
        except Exception as e:
            print(f"[Propagation] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _propagate_single_frame(self, src_frame, dst_frame, src_mask):
        """Use SAM2 to propagate mask from src_frame to dst_frame.
        Returns the propagated mask or None if failed."""
        try:
            from sam2.build_sam import build_sam2_video_predictor
            
            # Get frame source directory based on display mode
            if self.display_mode == "original":
                source_dir = self.original_dir
            elif self.display_mode == "gamma":
                source_dir = self.gamma_dir
            else:
                source_dir = self.hsv_v_dir
            
            # Create temp directory with 2 frames
            temp_dir = tempfile.mkdtemp()
            
            src_img_path = source_dir / f"frame_{src_frame:06d}.png"
            dst_img_path = source_dir / f"frame_{dst_frame:06d}.png"
            
            if src_img_path.exists():
                img = cv2.imread(str(src_img_path))
                cv2.imwrite(f"{temp_dir}/00000.jpg", img)
            if dst_img_path.exists():
                img = cv2.imread(str(dst_img_path))
                cv2.imwrite(f"{temp_dir}/00001.jpg", img)
            
            # Build/reuse video predictor
            if self.video_predictor is None:
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                print(f"Building video predictor on {device}...")
                config_name = "sam2_hiera_l.yaml"

                # Build WITHOUT relying on a base checkpoint; load fine-tuned weights below.
                self.video_predictor = build_sam2_video_predictor(
                    config_name,
                    None,
                    device=device
                )
                
                # Load fine-tuned weights if available
                if self.checkpoint_path and self.checkpoint_path.exists():
                    checkpoint = torch.load(str(self.checkpoint_path), map_location=device, weights_only=False)
                    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                    self.video_predictor.load_state_dict(state_dict)
                    self.video_predictor.eval()
            
            # Initialize state and add mask
            inference_state = self.video_predictor.init_state(video_path=temp_dir)
            self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                mask=(src_mask > 127).astype(np.float32)
            )
            
            # Propagate
            propagated_mask = None
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                if out_frame_idx == 1:  # Destination frame
                    for i, obj_id in enumerate(out_obj_ids):
                        if obj_id == 1:
                            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                            propagated_mask = (mask * 255).astype(np.uint8)
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            return propagated_mask
            
        except Exception as e:
            return None
    
    def _wobble_preview(self, direction):
        """Temporarily show adjacent frame RAW (no mask) to see tool/tissue.
        This is just a visual peek - doesn't change working state."""
        peek_idx = self.current_idx + direction
        if peek_idx < 0 or peek_idx >= len(self.frame_indices):
            return
        
        # If first wobble, save current working state
        if not self.wobble_active:
            self.wobble_active = True
            self.wobble_origin_idx = self.current_idx
            self.wobble_saved_mask = self.cavity_mask.copy() if self.cavity_mask is not None else None
            self.wobble_saved_pos = list(self.cavity_points_pos)
            self.wobble_saved_neg = list(self.cavity_points_neg)
        
        # If wobbling back to origin frame, restore mask instead of raw view
        if peek_idx == self.wobble_origin_idx:
            self._wobble_return()
            return
        
        # Load the peek frame's image only (no mask)
        peek_frame = self.frame_indices[peek_idx]
        self.current_idx = peek_idx
        
        # Get image based on display mode
        if self.display_mode == "original":
            img_path = self.original_dir / f"frame_{peek_frame:06d}.png"
        elif self.display_mode == "gamma":
            img_path = self.gamma_dir / f"frame_{peek_frame:06d}.png"
        else:
            img_path = self.hsv_v_dir / f"frame_{peek_frame:06d}.png"
        
        if img_path.exists():
            self.current_image = cv2.imread(str(img_path))
        
        # Clear mask display for raw view
        self.cavity_mask = None
        self.comparison_mask = None
        self.cavity_points_pos = []
        self.cavity_points_neg = []
        
        # Update UI (just the frame label, not triggering load)
        self.control_panel.set_current_frame(peek_frame)
        self._update_viewer()
    
    def _wobble_return(self):
        """Return to working frame and restore mask state."""
        if not self.wobble_active:
            return
        
        self.wobble_active = False
        self.current_idx = self.wobble_origin_idx
        
        # Reload working frame
        frame_idx = self.frame_indices[self.current_idx]
        
        # Get image
        if self.display_mode == "original":
            img_path = self.original_dir / f"frame_{frame_idx:06d}.png"
        elif self.display_mode == "gamma":
            img_path = self.gamma_dir / f"frame_{frame_idx:06d}.png"
        else:
            img_path = self.hsv_v_dir / f"frame_{frame_idx:06d}.png"
        
        if img_path.exists():
            self.current_image = cv2.imread(str(img_path))
        
        # Restore saved state
        self.cavity_mask = self.wobble_saved_mask
        self.cavity_points_pos = self.wobble_saved_pos
        self.cavity_points_neg = self.wobble_saved_neg
        self.comparison_mask = None
        
        self.control_panel.set_current_frame(frame_idx)
        self._update_viewer()
    
    # --- Event handlers ---
    
    def _on_frame_changed(self, frame_idx):
        self._load_frame(frame_idx)
    
    def _on_display_mode_changed(self, mode):
        self.display_mode = mode
        self._update_display_image()
        self._update_viewer()
    
    def _on_correction_mode_changed(self, mode):
        self.correction_mode = mode
    
    def _on_tool_mode_changed(self, mode):
        self.tool_mode = mode
        self._update_viewer()
    
    def _on_brush_size_changed(self, size):
        self.brush_size = size
        self._update_viewer()
    
    def _on_mouse_move(self, x, y):
        self.control_panel.update_cursor_pos(x, y)
        self.control_panel.zoom_preview.update_position(x, y)
    
    def _on_mouse_click(self, x, y, is_left):
        if not is_left:
            return
        
        # Pause video when clicking to add/exclude
        if self.control_panel.is_playing():
            self.control_panel.stop_playback()
            self.play_timer.stop()
        
        # Editing commits us to this frame - clear comparison state
        self.working_frame_idx = self.frame_indices[self.current_idx]
        self.comparison_mask = None
        self.reference_frame_idx = None
        
        # Mark frame as modified
        frame_idx = self.frame_indices[self.current_idx]
        self.modified_frames.add(frame_idx)
        
        self._save_state()
        
        if self.tool_mode == "brush":
            self._paint_brush(x, y)
        else:
            # Click mode - add point
            if self.correction_mode == "add":
                self.cavity_points_pos.append((x, y))
            else:
                self.cavity_points_neg.append((x, y))
            self._run_sam2()
        
        self._update_viewer()
    
    def _on_mouse_drag(self, x, y):
        if self.tool_mode == "brush":
            self._paint_brush(x, y)
            self._update_viewer()
    
    def _on_mouse_release(self):
        pass
    
    def _on_play_toggled(self, is_playing):
        """Handle play/pause toggle."""
        if is_playing:
            # Calculate interval based on FPS (default 30 fps = ~33ms)
            interval_ms = int(1000 / self.fps) if self.fps > 0 else 33
            self.play_timer.start(interval_ms)
        else:
            self.play_timer.stop()
    
    def _play_timer_tick(self):
        """Advance to next frame during playback."""
        if self.current_idx < len(self.frame_indices) - 1:
            # Advance one frame
            next_idx = self.current_idx + 1
            self._load_frame(self.frame_indices[next_idx])
        else:
            # Reached the end, stop playback
            self.control_panel.stop_playback()
            self.play_timer.stop()
    
    def keyPressEvent(self, event):
        key = event.key()
        
        # If in wobble mode and pressing non-wobble key, return first
        if self.wobble_active and key not in (Qt.Key_Comma, Qt.Key_Period):
            self._wobble_return()
        
        if key == Qt.Key_Comma:
            # Wobble preview - peek at previous frame (raw, no mask)
            self._wobble_preview(-1)
        
        elif key == Qt.Key_Period:
            # Wobble preview - peek at next frame (raw, no mask)
            self._wobble_preview(1)
        
        elif key == Qt.Key_BracketLeft:
            # Navigate backward - move to previous frame
            new_idx = max(0, self.current_idx - 1)
            self._load_frame(self.frame_indices[new_idx])
        
        elif key == Qt.Key_BracketRight:
            # Navigate forward - move to next frame
            new_idx = min(len(self.frame_indices) - 1, self.current_idx + 1)
            self._load_frame(self.frame_indices[new_idx])
        
        elif key == Qt.Key_D:
            self.control_panel.cycle_display_mode()
        
        elif key == Qt.Key_A:
            self.control_panel.set_correction_mode("add")
            self.correction_mode = "add"
        
        elif key == Qt.Key_X:
            self.control_panel.set_correction_mode("exclude")
            self.correction_mode = "exclude"
        
        elif key == Qt.Key_B:
            new_mode = "click" if self.tool_mode == "brush" else "brush"
            self.control_panel.set_tool_mode(new_mode)
            self.tool_mode = new_mode
            self._update_viewer()
        
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.brush_size = min(100, self.brush_size + 5)
            self.control_panel._brush_size = self.brush_size
            self.control_panel.size_label.setText(str(self.brush_size))
            self._update_viewer()
        
        elif key == Qt.Key_Minus:
            self.brush_size = max(5, self.brush_size - 5)
            self.control_panel._brush_size = self.brush_size
            self.control_panel.size_label.setText(str(self.brush_size))
            self._update_viewer()
        
        elif key == Qt.Key_Z:
            self._undo()
        
        elif key == Qt.Key_R:
            self._reset()
        
        elif key == Qt.Key_Q or key == Qt.Key_Escape:
            self.close()
        
        elif key == Qt.Key_1:
            self.control_panel.set_display_mode("original")
            self._on_display_mode_changed("original")
        
        elif key == Qt.Key_2:
            self.control_panel.set_display_mode("gamma")
            self._on_display_mode_changed("gamma")
        
        elif key == Qt.Key_3:
            self.control_panel.set_display_mode("hsv_v")
            self._on_display_mode_changed("hsv_v")
        
        elif key == Qt.Key_Space:
            # Toggle play/pause with spacebar
            self.control_panel._toggle_play()
    
    def closeEvent(self, event):
        """Handle window close - save all modified frames and create summary."""
        # Save current frame if it has modifications
        current_frame_idx = self.frame_indices[self.current_idx]
        if current_frame_idx in self.modified_frames:
            if self.cavity_mask is not None or len(self.cavity_points_pos) > 0:
                self._save_current(advance=False)
        
        # Create summary data
        summary_data = {
            'video_name': self.video_path.stem if self.video_path else 'unknown',
            'video_path': str(self.video_path) if self.video_path else '',
            'frame_range': {
                'start': self.start_frame,
                'end': self.end_frame
            },
            'total_frames_in_range': len(self.frame_indices),
            'modified_frames_count': len(self.modified_frames),
            'modified_frames': sorted(list(self.modified_frames)),
            'frames': []
        }
        
        # Collect data for all modified frames
        for frame_idx in sorted(self.modified_frames):
            frame_dir = self.output_dir / f"frame_{frame_idx:06d}"
            
            # Load saved data for this frame
            frame_data = {
                'frame_number': int(frame_idx),
                'has_mask': False,
                'cavity_points_pos': [],
                'cavity_points_neg': []
            }
            
            # Check if mask exists
            mask_path = frame_dir / "cavity_mask.png"
            if mask_path.exists():
                frame_data['has_mask'] = True
            
            # Load point data if exists
            data_path = frame_dir / "instance_data.json"
            if data_path.exists():
                with open(data_path, 'r') as f:
                    instance_data = json.load(f)
                    frame_data['cavity_points_pos'] = instance_data.get('cavity_points_pos', [])
                    frame_data['cavity_points_neg'] = instance_data.get('cavity_points_neg', [])
            
            summary_data['frames'].append(frame_data)
        
        # Save summary JSON
        summary_path = self.output_dir / "session_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"✓ Saved {len(self.modified_frames)} frames to {summary_path}")
        
        # Clean up video capture
        if hasattr(self, 'video_cap') and self.video_cap:
            self.video_cap.release()
        
        event.accept()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Polished Cavity Definition Tool")
    parser.add_argument("--base-dir", type=str, 
                       default=str(Path(__file__).parent.parent.parent),
                       help="Base directory of the project")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to video file (skip startup dialog)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for masks and data")
    parser.add_argument("--start", type=int, default=0,
                       help="Start frame index")
    parser.add_argument("--end", type=int, default=None,
                       help="End frame index")
    parser.add_argument("--checkpoint", type=str, 
                       default='/opt/5-ALA-Videos/weights/sam2_cavity_finetuned.pt',
                       help="Path to SAM2 checkpoint")
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # If video/output not provided, show startup dialog
    if args.video is None or args.output is None:
        base_dir = Path(args.base_dir)
        
        startup = StartupDialog(base_dir)
        if startup.exec_() != QDialog.Accepted:
            sys.exit(0)
        
        selection = startup.get_selection()
        if selection is None:
            sys.exit(0)
        
        video_info, start_frame, end_frame = selection
        
        video_path = video_info['path']
        video_folder = video_info['folder']
        
        # Always use masks/cavity structure in video folder
        output_dir = video_folder / "masks" / "cavity"
        
        tool = CavityTool(
            video_path=str(video_path),
            output_dir=str(output_dir),
            start_frame=start_frame,
            end_frame=end_frame,
            checkpoint_path=args.checkpoint
        )
    else:
        tool = CavityTool(
            video_path=args.video,
            output_dir=args.output,
            start_frame=args.start,
            end_frame=args.end,
            checkpoint_path=args.checkpoint
        )
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

