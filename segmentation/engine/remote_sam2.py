"""HTTP client for the shared GPU SAM2 inference manager."""

from __future__ import annotations

import base64
import getpass
import os
import socket
import threading
from dataclasses import dataclass
from typing import Callable

import cv2
import httpx
import numpy as np


def _encode_jpeg(image: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode frame for GPU manager")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_mask(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("Failed to encode mask for GPU manager")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_mask(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("GPU manager returned an invalid mask")
    return mask.astype(np.uint8)


@dataclass
class _SessionInfo:
    session_id: str
    device_label: str
    weights_label: str
    heartbeat_interval: float


class RemoteSAM2Engine:
    """Drop-in replacement for ``SAM2Engine`` that proxies through HTTP."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._current_image: np.ndarray | None = None
        self._session = self._register()
        self._stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="sam2-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    # ── public API expected by AnnotationTool ──────────────────────

    @property
    def available(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return self._session.device_label

    @property
    def weights_label(self) -> str:
        return self._session.weights_label

    def set_image(self, bgr_image: np.ndarray):
        self._current_image = bgr_image.copy()

    def predict_mask(self, points_pos, points_neg):
        if self._current_image is None:
            raise RuntimeError("Remote SAM2 image not primed; call set_image() first")
        payload = {
            "session_id": self._session.session_id,
            "points_pos": points_pos,
            "points_neg": points_neg,
            "frame_jpeg": _encode_jpeg(self._current_image),
        }
        resp = self._client.post("/predict_click", json=payload)
        resp.raise_for_status()
        data = resp.json()
        mask_b64 = data.get("mask")
        return _decode_mask(mask_b64) if mask_b64 else None

    def propagate_single(self, src_img, dst_img, masks):
        payload = {
            "session_id": self._session.session_id,
            "src_frame": _encode_jpeg(src_img),
            "dst_frame": _encode_jpeg(dst_img),
            "masks": [{"obj_id": int(obj_id), "mask": _encode_mask(mask)} for obj_id, mask in masks],
        }
        resp = self._client.post("/propagate_single", json=payload)
        resp.raise_for_status()
        blobs = resp.json().get("masks", [])
        result = {}
        for entry in blobs:
            result[int(entry["obj_id"])] = _decode_mask(entry["mask"])
        return result

    def propagate_batch(self, frames, masks, progress_cb: Callable | None = None):
        payload = {
            "session_id": self._session.session_id,
            "frames": [_encode_jpeg(f) for f in frames],
            "masks": [{"obj_id": int(obj_id), "mask": _encode_mask(mask)} for obj_id, mask in masks],
        }
        resp = self._client.post("/propagate_batch", json=payload)
        resp.raise_for_status()
        body = resp.json()
        results = {}
        for entry in body.get("results", []):
            sam_idx = int(entry["sam_index"])
            masks_dict = {}
            for blob in entry.get("masks", []):
                masks_dict[int(blob["obj_id"])] = _decode_mask(blob["mask"])
            results[sam_idx] = masks_dict
        if progress_cb:
            progress_cb(len(results), len(frames))
        return results

    def shutdown(self):
        self._stop_event.set()
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)
        try:
            self._client.post("/sessions/release", json={"session_id": self._session.session_id})
        except httpx.HTTPError:
            pass
        self._client.close()

    # ── internal helpers ───────────────────────────────────────────

    def _register(self) -> _SessionInfo:
        meta = {
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        resp = self._client.post("/sessions/register", json=meta)
        resp.raise_for_status()
        body = resp.json()
        interval = float(body.get("heartbeat_interval", 30.0))
        return _SessionInfo(
            session_id=body["session_id"],
            device_label=body.get("device_label", "gpu-manager"),
            weights_label=body.get("weights_label", "SAM2"),
            heartbeat_interval=interval,
        )

    def _heartbeat_loop(self):
        while not self._stop_event.wait(self._session.heartbeat_interval):
            try:
                self._client.post("/sessions/heartbeat", json={"session_id": self._session.session_id})
            except httpx.HTTPError:
                # Transient network hiccups are expected over X11 remoting; keep trying.
                continue

