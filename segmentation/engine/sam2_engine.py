"""SAM2 wrapper for single-image prediction and multi-object video propagation.

NOTE: SAM2's ``init_state(video_path=...)`` API requires a directory of image
files on disk -- there is no in-memory alternative as of SAM2 v2.1.  We
mitigate the I/O cost by:
  1. Writing JPEG (faster encode/decode than PNG, ~3x).
  2. Using /dev/shm (Linux tmpfs) when available for near-zero disk latency.
"""

import os
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path

try:
    import torch
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


def _pick_device() -> str:
    if not SAM2_AVAILABLE:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _fast_tmpdir() -> str:
    """Return /dev/shm on Linux (RAM-backed tmpfs) if available, else default."""
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return tempfile.mkdtemp(dir=str(shm))
    return tempfile.mkdtemp()


_BASE_CHECKPOINT_STEMS = {"sam2_hiera_large", "sam2_hiera_base_plus",
                          "sam2_hiera_small", "sam2_hiera_tiny",
                          "sam2.1_hiera_large", "sam2.1_hiera_base_plus",
                          "sam2.1_hiera_small", "sam2.1_hiera_tiny"}

_FRIENDLY_NAMES = {
    "large": "SAM2 Hiera-Large",
    "base_plus": "SAM2 Hiera-Base+",
    "small": "SAM2 Hiera-Small",
    "tiny": "SAM2 Hiera-Tiny",
}


def _friendly_name(ckpt_path: Path | None) -> str:
    """Derive a human-readable label from the checkpoint filename."""
    if ckpt_path is None:
        return "None"
    stem = ckpt_path.stem.lower()
    if any(stem == b for b in _BASE_CHECKPOINT_STEMS):
        for key, name in _FRIENDLY_NAMES.items():
            if key in stem:
                return name
        return "SAM2 (base)"
    return "Custom fine-tuned"


class SAM2Engine:
    """Unified SAM2 interface for click prediction + video propagation."""

    CONFIG = "sam2_hiera_l.yaml"

    def __init__(self, checkpoint_path: str | Path | None = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = _pick_device()
        self.weights_label: str = "None"
        self.image_predictor: SAM2ImagePredictor | None = None
        self.video_predictor = None
        self._init_image_predictor()

    @property
    def available(self) -> bool:
        return self.image_predictor is not None

    # ── initialisation ──────────────────────────────────────────────

    def _build_model(self, builder=build_sam2 if SAM2_AVAILABLE else None):
        """Build a SAM2 model, trying the checkpoint in standard format first,
        then falling back to manual weight loading for fine-tuned checkpoints."""
        ckpt = self.checkpoint_path
        has_ckpt = ckpt and ckpt.exists()

        if not has_ckpt:
            print(f"WARNING: No SAM2 checkpoint found at {ckpt}  — model disabled.")
            return None

        try:
            model = builder(self.CONFIG, str(ckpt), device=self.device)
            model = model.to(self.device)
            self.weights_label = _friendly_name(ckpt)
            return model
        except Exception:
            pass

        try:
            model = builder(self.CONFIG, None, device=self.device)
            model = model.to(self.device)
            self._load_finetuned_weights(model)
            self.weights_label = "Custom fine-tuned"
            return model
        except Exception as exc:
            print(f"SAM2 model build failed: {exc}")
            return None

    def _init_image_predictor(self):
        if not SAM2_AVAILABLE:
            return
        model = self._build_model(builder=build_sam2)
        if model is None:
            return
        try:
            self.image_predictor = SAM2ImagePredictor(model)
            self.image_predictor.model.eval()
            print(f"SAM2 image predictor ready on {self.device}  [{self.weights_label}]")
        except Exception as exc:
            print(f"SAM2 image predictor init failed: {exc}")
            self.image_predictor = None

    def _ensure_video_predictor(self):
        if self.video_predictor is not None:
            return
        if not SAM2_AVAILABLE:
            return
        model = self._build_model(builder=build_sam2_video_predictor)
        if model is None:
            return
        model.eval()
        self.video_predictor = model
        print(f"SAM2 video predictor ready on {self.device}  [{self.weights_label}]")

    def _load_finetuned_weights(self, model):
        """Load a fine-tuned checkpoint with 'model_state_dict' key format."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return
        # NOTE: weights_only=False is required for fine-tuned checkpoints that may
        # contain non-tensor objects. Safe here because checkpoints are locally produced.
        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(sd)

    # ── single-image prediction ─────────────────────────────────────

    def set_image(self, bgr_image: np.ndarray):
        if self.image_predictor is None:
            return
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self.image_predictor.set_image(rgb)

    def predict_mask(self, points_pos, points_neg) -> np.ndarray | None:
        """Run SAM2 with positive/negative points. Returns uint8 mask (0/255)."""
        if self.image_predictor is None or not points_pos:
            return None
        all_pts = points_pos + points_neg
        labels = [1] * len(points_pos) + [0] * len(points_neg)
        masks, _, _ = self.image_predictor.predict(
            point_coords=np.array(all_pts),
            point_labels=np.array(labels),
            multimask_output=False,
        )
        raw = (masks[0] * 255).astype(np.uint8)
        # Keep only connected components containing a positive point
        n_labels, label_map = cv2.connectedComponents(raw)
        keep = set()
        for px, py in points_pos:
            lbl = label_map[py, px]
            if lbl > 0:
                keep.add(lbl)
        filtered = np.zeros_like(raw)
        for lid in keep:
            filtered[label_map == lid] = 255
        return filtered

    # ── multi-object single-step propagation ────────────────────────

    def propagate_single(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
        masks: list[tuple[int, np.ndarray]],
    ) -> dict[int, np.ndarray]:
        """Propagate multiple masks from src to dst frame (one step).

        *masks*: list of (obj_id, uint8_mask) tuples.
        Returns {obj_id: uint8_mask} for each propagated object.
        """
        if not SAM2_AVAILABLE or not masks:
            return {}
        self._ensure_video_predictor()
        if self.video_predictor is None:
            return {}

        tmp = _fast_tmpdir()
        try:
            cv2.imwrite(f"{tmp}/00000.jpg", src_img)
            cv2.imwrite(f"{tmp}/00001.jpg", dst_img)

            state = self.video_predictor.init_state(video_path=tmp)
            for obj_id, mask in masks:
                self.video_predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=(mask > 127).astype(np.float32),
                )

            results: dict[int, np.ndarray] = {}
            for out_idx, out_ids, out_logits in self.video_predictor.propagate_in_video(state):
                if out_idx == 1:
                    for i, oid in enumerate(out_ids):
                        m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                        results[oid] = (m * 255).astype(np.uint8)
            return results
        except Exception as exc:
            print(f"[SAM2] propagation error: {exc}")
            return {}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ── batch propagation (N frames) ────────────────────────────────

    def propagate_batch(
        self,
        frames: list[np.ndarray],
        masks: list[tuple[int, np.ndarray]],
        progress_cb=None,
    ) -> dict[int, dict[int, np.ndarray]]:
        """Propagate masks across *frames[1:]* starting from *frames[0]*.

        Returns {sam2_frame_idx: {obj_id: uint8_mask}}.
        """
        if not SAM2_AVAILABLE or not masks or len(frames) < 2:
            return {}
        self._ensure_video_predictor()
        if self.video_predictor is None:
            return {}

        tmp = _fast_tmpdir()
        try:
            for i, f in enumerate(frames):
                cv2.imwrite(f"{tmp}/{i:05d}.jpg", f)

            state = self.video_predictor.init_state(video_path=tmp)
            for obj_id, mask in masks:
                self.video_predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=(mask > 127).astype(np.float32),
                )

            all_results: dict[int, dict[int, np.ndarray]] = {}
            for out_idx, out_ids, out_logits in self.video_predictor.propagate_in_video(state):
                frame_masks: dict[int, np.ndarray] = {}
                for i, oid in enumerate(out_ids):
                    m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                    frame_masks[oid] = (m * 255).astype(np.uint8)
                all_results[out_idx] = frame_masks
                if progress_cb:
                    progress_cb(out_idx, len(frames))
            return all_results
        except Exception as exc:
            print(f"[SAM2] batch propagation error: {exc}")
            return {}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
