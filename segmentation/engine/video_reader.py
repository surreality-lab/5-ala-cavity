"""Video frame reading with LRU cache and on-the-fly display transforms."""

import cv2
import numpy as np
from collections import OrderedDict
from pathlib import Path

from lib.utils import crop_ui


def try_open_video(video_path: Path):
    """Try multiple backends to open a video; returns (cap, backend) or (None, None)."""
    candidates = [("ANY", None)]
    if hasattr(cv2, "CAP_FFMPEG"):
        candidates.append(("FFMPEG", cv2.CAP_FFMPEG))
    if hasattr(cv2, "CAP_GSTREAMER"):
        candidates.append(("GSTREAMER", cv2.CAP_GSTREAMER))

    for name, api in candidates:
        try:
            cap = cv2.VideoCapture(str(video_path)) if api is None else cv2.VideoCapture(str(video_path), api)
        except Exception:
            cap = None
        if cap is not None and cap.isOpened():
            return cap, name
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return None, None


class VideoReader:
    """Reads frames from an MP4 with an LRU cache and applies display transforms."""

    CACHE_SIZE = 30

    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        self.cap, self.backend = try_open_video(self.video_path)
        if self.cap is None:
            raise ValueError(f"Could not open video: {self.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._last_read_pos: int = -1

    def read_original(self, frame_idx: int) -> np.ndarray | None:
        """Return cropped BGR frame (LRU cached).

        Skips the expensive seek when the requested frame immediately follows
        the last read position, which is the common case during forward
        navigation and playback.
        """
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx].copy()

        if frame_idx != self._last_read_pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = self.cap.read()
        if not ret:
            return None

        self._last_read_pos = frame_idx
        frame = crop_ui(frame)
        self._cache[frame_idx] = frame.copy()
        while len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)
        return frame

    def read_display(self, frame_idx: int, mode: str = "original") -> np.ndarray | None:
        """Return a frame processed according to display *mode*."""
        original = self.read_original(frame_idx)
        if original is None:
            return None
        if mode == "gamma":
            return self._process_gamma(original)
        if mode == "hsv_v":
            return self._process_hsv_v(original)
        return original.copy()

    # ── display transforms ──────────────────────────────────────────

    @staticmethod
    def _process_gamma(img, gamma=2.2):
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
        corrected = cv2.LUT(img, table)
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.bilateralFilter(enhanced, 9, 75, 75)

    @staticmethod
    def _process_hsv_v(img):
        v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
        return cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    def release(self):
        if self.cap:
            self.cap.release()
