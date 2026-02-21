#!/usr/bin/env python3
"""
Shared Utilities for 5-ALA Analysis Pipeline
==============================================
Centralizes functions duplicated across multiple pipeline scripts.
Import from here instead of redefining.

Functions
---------
crop_ui
    Remove surgical-UI overlays from video frames.
discover_mask_frames
    Find all frame indices that have cavity masks in a directory.
load_mask
    Load a grayscale mask PNG, resize if needed, threshold to bool.
load_analysis_mask
    Load cavity + exclusion masks and compose into analysis mask.
load_frame_data
    Read a video frame, crop UI, and load/compose analysis masks.
json_convert
    Serialize numpy types for JSON output (NaN/Inf → None).
parse_include_csv
    Parse a manual_frame_classifications.csv and return include-set.
setup_logging
    Configure logging to console + optional file for pipeline scripts.
SequentialVideoReader
    Efficiently read video frames with sequential-access optimization.
"""

from __future__ import annotations

import csv
import cv2
import logging
import numpy as np
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, NamedTuple


# ── Frame Cropping ────────────────────────────────────────────────────────

def crop_ui(img: np.ndarray, left_percent: int = 20,
            right_percent: int = 5) -> np.ndarray:
    """Remove surgical-system UI overlays by cropping left/right margins.

    Parameters
    ----------
    img : ndarray (H, W, ...) or (H, W)
        Input image (BGR, grayscale, etc.).
    left_percent : int
        Percentage of width to crop from the left (default 20%).
    right_percent : int
        Percentage of width to crop from the right (default 5%).

    Returns
    -------
    cropped : ndarray
        Cropped image with the same number of dimensions.
    """
    height, width = img.shape[:2]
    left_crop = int(width * left_percent / 100)
    right_crop = int(width * (100 - right_percent) / 100)
    return img[:, left_crop:right_crop]


# ── Mask Discovery ────────────────────────────────────────────────────────

def discover_mask_frames(mask_dir: Path | str) -> list[int]:
    """Find all frame indices that have cavity masks.

    Scans ``mask_dir`` for directories named ``frame_NNNNNN`` containing
    ``cavity_mask.png`` and returns the sorted list of integer frame indices.

    Parameters
    ----------
    mask_dir : Path or str
        Directory containing per-frame mask subdirectories.

    Returns
    -------
    frame_indices : list[int]
        Sorted frame indices.
    """
    mask_dir = Path(mask_dir)
    if not mask_dir.is_dir():
        return []
    frame_indices: list[int] = []
    for d in sorted(mask_dir.iterdir()):
        if d.is_dir() and d.name.startswith("frame_"):
            if (d / "cavity_mask.png").exists():
                try:
                    idx = int(d.name.replace("frame_", ""))
                    frame_indices.append(idx)
                except ValueError:
                    continue
    return sorted(frame_indices)


# ── Mask Loading ──────────────────────────────────────────────────────────

def load_mask(path: Path | str,
              shape: tuple[int, int]) -> np.ndarray | None:
    """Load a grayscale mask, resize if needed, threshold to bool.

    Parameters
    ----------
    path : Path
        Path to a grayscale PNG mask (0/255 or 0-255).
    shape : tuple (H, W)
        Expected spatial dimensions. If the loaded mask differs,
        it is resized with nearest-neighbor interpolation.

    Returns
    -------
    mask : ndarray[bool] or None
        Boolean mask (True where mask > 127), or None if file
        does not exist or cannot be read.
    """
    path = Path(path)
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    return m > 127


# ── Analysis Mask Loading ─────────────────────────────────────────────────

def load_analysis_mask(
    frame_idx: int,
    cavity_dir: Path | str,
    refl_dir: Path | str,
    blood_dir: Path | str,
    shape: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Load cavity mask and compose with exclusion masks for a single frame.

    Encapsulates the repeated pattern::

        cavity_mask  = load(cavity_mask.png)
        excl         = load(reflection_mask.png) | load(blood_mask.png)
        analysis_mask = cavity_mask & ~excl

    Parameters
    ----------
    frame_idx : int
        Zero-based video frame index.
    cavity_dir, refl_dir, blood_dir : Path or str
        Root directories containing ``frame_NNNNNN/`` subdirectories.
    shape : tuple (H, W)
        Expected spatial dimensions (used for resize if needed).

    Returns
    -------
    cavity_mask : ndarray[bool] or None
        Boolean cavity mask, or None if not found.
    analysis_mask : ndarray[bool] or None
        ``cavity & ~(reflection | blood)``, or None if cavity is missing
        or the analyzable area is < 100 px.
    info : dict
        ``refl_mask``, ``blood_mask``, ``refl_px``, ``blood_px``,
        ``cavity_px``, ``analysis_px``.
    """
    cavity_dir = Path(cavity_dir)
    refl_dir = Path(refl_dir)
    blood_dir = Path(blood_dir)
    tag = f"frame_{frame_idx:06d}"

    cavity_mask = load_mask(cavity_dir / tag / "cavity_mask.png", shape)
    if cavity_mask is None:
        return None, None, {}

    excl = np.zeros(shape, dtype=bool)
    refl_m = load_mask(refl_dir / tag / "reflection_mask.png", shape)
    blood_m = load_mask(blood_dir / tag / "blood_mask.png", shape)
    if refl_m is not None:
        excl |= refl_m
    if blood_m is not None:
        excl |= blood_m

    analysis_mask = cavity_mask & ~excl
    analysis_px = int(np.sum(analysis_mask))

    info = {
        'refl_mask': refl_m,
        'blood_mask': blood_m,
        'refl_px': int(np.sum(refl_m)) if refl_m is not None else 0,
        'blood_px': int(np.sum(blood_m)) if blood_m is not None else 0,
        'cavity_px': int(np.sum(cavity_mask)),
        'analysis_px': analysis_px,
    }

    if analysis_px < 100:
        return cavity_mask, None, info

    return cavity_mask, analysis_mask, info


def load_frame_data(
    cap: cv2.VideoCapture,
    frame_idx: int,
    cavity_dir: Path | str,
    refl_dir: Path | str,
    blood_dir: Path | str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    """Read a video frame, crop UI overlays, and load/compose analysis masks.

    Combines :func:`crop_ui` and :func:`load_analysis_mask` into one call
    to eliminate the ~20-line boilerplate repeated across every analysis
    script.

    .. note::

        Light-stability checking is **not** included here to avoid a
        circular import with ``lib.blood_detection``.  Call
        ``check_light_stability`` on the returned frame yourself.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Open video capture positioned at any frame.  The function seeks
        to *frame_idx* internally.
    frame_idx : int
        Zero-based video frame index.
    cavity_dir, refl_dir, blood_dir : Path or str
        Root directories containing ``frame_NNNNNN/`` subdirectories.

    Returns
    -------
    frame_bgr : ndarray or None
        Cropped BGR frame, or None on read failure.
    cavity_mask : ndarray[bool] or None
        Boolean cavity mask (True inside cavity).
    analysis_mask : ndarray[bool] or None
        ``cavity & ~(reflection | blood)``, or None if cavity missing /
        analyzable area < 100 px.
    info : dict
        Same keys as :func:`load_analysis_mask` info dict.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        return None, None, None, {}
    frame_bgr = crop_ui(frame_bgr)

    shape = frame_bgr.shape[:2]
    cavity_mask, analysis_mask, info = load_analysis_mask(
        frame_idx, cavity_dir, refl_dir, blood_dir, shape)

    return frame_bgr, cavity_mask, analysis_mask, info


# ── JSON Serialization ────────────────────────────────────────────────────

def json_convert(obj: object) -> object:
    """Convert numpy types to JSON-safe Python types.

    - ``np.integer`` → ``int``
    - ``np.floating`` → ``float`` (NaN/Inf → ``None``)
    - ``np.ndarray`` → ``list``
    - ``np.bool_`` → ``bool``
    - Python ``float`` NaN/Inf → ``None``
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ── Include-List CSV Parsing ──────────────────────────────────────────────

def parse_include_csv(csv_path: Path | str) -> set[int]:
    """Parse a manual_frame_classifications.csv and return the set of
    frame indices classified as 'include'.

    Parameters
    ----------
    csv_path : Path or str

    Returns
    -------
    include_set : set[int]
    """
    csv_path = Path(csv_path)
    include_set: set[int] = set()
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['classification'].strip().lower() == 'include':
                start = int(row['frame_start'])
                end = int(row['frame_end'])
                for idx in range(start, end + 1):
                    include_set.add(idx)
    return include_set


# ── Logging ───────────────────────────────────────────────────────────────

def setup_logging(name: str,
                  output_dir: Path | str | None = None,
                  level: int = logging.INFO) -> logging.Logger:
    """Configure logging for a pipeline script.

    Creates a logger with:
    - Console handler (level as specified, clean format matching print())
    - File handler (DEBUG level, timestamped) if output_dir is provided

    Can be called twice: first without output_dir (during import / early
    startup), then again with output_dir once paths are resolved. The
    second call adds the file handler without duplicating the console
    handler.

    Parameters
    ----------
    name : str
        Logger name (typically the script name without extension).
    output_dir : Path or str, optional
        Directory for the log file. File is named ``{name}.log``.
    level : int
        Console logging level (default: logging.INFO).

    Returns
    -------
    logger : logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Add console handler only once
    has_console = any(isinstance(h, logging.StreamHandler)
                      and not isinstance(h, logging.FileHandler)
                      for h in logger.handlers)
    if not has_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    # Add file handler if output_dir given and not already present
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f'{name}.log'
        has_file = any(isinstance(h, logging.FileHandler)
                       and h.baseFilename == str(log_path.resolve())
                       for h in logger.handlers)
        if not has_file:
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(fh)

    return logger


# ── Sequential Video Reading ──────────────────────────────────────────────

class SequentialVideoReader:
    """Efficiently read video frames with sequential-access optimization.

    For H.264/H.265 video, ``cap.set(CAP_PROP_POS_FRAMES)`` is
    O(keyframe-distance) because the decoder must seek to the nearest
    I-frame and decode forward. Sequential ``cap.read()`` is much faster.

    This reader tracks the current position and uses sequential reads
    when possible, falling back to seeking only for large gaps.

    Usage::

        reader = SequentialVideoReader(video_path)
        for frame_idx in sorted_frame_indices:
            ret, frame = reader.read(frame_idx)
            if not ret:
                continue
            # ... process frame ...
        reader.release()

    Parameters
    ----------
    video_path : str or Path
        Path to the video file.
    sequential_threshold : int
        Maximum gap between consecutive reads before seeking.
        If the next frame is within this many frames of the current
        position, intermediate frames are grabbed (fast) instead of
        seeking (slow). Default: 30 (typical H.264 keyframe interval).
    """

    def __init__(self, video_path: str | Path,
                 sequential_threshold: int = 30):
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        self._pos: int = -1
        self._threshold = sequential_threshold
        self.total_frames: int = int(
            self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS)

    def read(self, frame_idx: int) -> tuple[bool, np.ndarray | None]:
        """Read a specific frame, using sequential access when possible.

        Returns
        -------
        ret : bool
            True if the frame was read successfully.
        frame : ndarray or None
            BGR frame, or None on failure.
        """
        gap = frame_idx - self._pos

        if gap == 1:
            # Next sequential frame -- just read (fastest path)
            ret, frame = self._cap.read()
        elif 1 < gap <= self._threshold:
            # Small gap -- grab and discard intermediates
            for _ in range(gap - 1):
                self._cap.grab()
            ret, frame = self._cap.read()
        else:
            # Large gap or backwards -- must seek
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()

        if ret:
            self._pos = frame_idx
        return ret, frame

    def release(self) -> None:
        """Release the underlying VideoCapture."""
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
