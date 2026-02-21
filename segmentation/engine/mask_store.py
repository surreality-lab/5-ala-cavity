"""Multi-layer mask data model with hybrid multi-instance support and disk I/O.

Each category (cavity / tool / object / hand) can have multiple instances,
each tracked independently by SAM2. Instances within a category are saved as
separate mask files but grouped in the same category directory.

Backward-compatible: existing ``masks/cavity/`` and ``masks/tool/`` directories
with single-instance masks are loaded transparently.
"""

import json
import hashlib
import os
import socket
import getpass
import tempfile
from datetime import datetime, timezone

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

# ── category configuration ───────────────────────────────────────────

LAYER_CATEGORIES = {
    "cavity": {"max_instances": 1, "base_obj_id": 1,  "color_bgr": (200, 80, 200), "shortcut": "C"},
    "tool":   {"max_instances": 3, "base_obj_id": 10, "color_bgr": (0, 140, 255),  "shortcut": "T"},
    "object": {"max_instances": 3, "base_obj_id": 20, "color_bgr": (211, 206, 0),  "shortcut": "J"},
    "hand":   {"max_instances": 2, "base_obj_id": 30, "color_bgr": (0, 200, 100),  "shortcut": "H"},
}

CATEGORY_NAMES = list(LAYER_CATEGORIES.keys())
MIN_MASK_PIXELS = 500

LAYER_PRIORITY: dict[str, int] = {
    "hand": 3,
    "tool": 2,
    "object": 1,
    "cavity": 0,
}


def prior_categories(category: str) -> list[str]:
    """Return categories with strictly higher priority than *category*."""
    my_pri = LAYER_PRIORITY.get(category, -1)
    return [c for c, p in LAYER_PRIORITY.items() if p > my_pri]


def instance_key(category: str, instance_id: int) -> str:
    if LAYER_CATEGORIES[category]["max_instances"] == 1:
        return category
    return f"{category}_{instance_id}"


def parse_key(key: str) -> tuple[str, int]:
    """Parse 'tool_2' -> ('tool', 2), 'cavity' -> ('cavity', 1)."""
    for cat in CATEGORY_NAMES:
        if key == cat:
            return cat, 1
        if key.startswith(cat + "_"):
            try:
                return cat, int(key[len(cat) + 1:])
            except ValueError:
                pass
    return key, 1


def obj_id_for(category: str, instance_id: int) -> int:
    return LAYER_CATEGORIES[category]["base_obj_id"] + instance_id - 1


def instance_color(category: str, instance_id: int) -> tuple[int, int, int]:
    """Return a shade-shifted BGR color for visual distinction between instances."""
    base = LAYER_CATEGORIES[category]["color_bgr"]
    if instance_id <= 1:
        return base
    pixel = np.array([[base]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    hsv[0, 0, 0] = (hsv[0, 0, 0] + 25 * (instance_id - 1)) % 180
    hsv[0, 0, 1] = max(80, int(hsv[0, 0, 1]) - 20 * (instance_id - 1))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


# ── backward compat aliases (used by imports that expect the old API) ─

LAYERS = LAYER_CATEGORIES
LAYER_NAMES = CATEGORY_NAMES


# ── data model ───────────────────────────────────────────────────────

@dataclass
class MaskLayer:
    """State for a single annotation instance on one frame."""
    category: str
    instance_id: int = 1
    mask: np.ndarray | None = None
    points_pos: list = field(default_factory=list)
    points_neg: list = field(default_factory=list)
    active: bool = True
    inverted: bool = False

    @property
    def key(self) -> str:
        return instance_key(self.category, self.instance_id)

    @property
    def obj_id(self) -> int:
        return obj_id_for(self.category, self.instance_id)

    @property
    def color_bgr(self) -> tuple[int, int, int]:
        return instance_color(self.category, self.instance_id)

    def copy(self):
        return MaskLayer(
            category=self.category,
            instance_id=self.instance_id,
            mask=self.mask.copy() if self.mask is not None else None,
            points_pos=list(self.points_pos),
            points_neg=list(self.points_neg),
            active=self.active,
            inverted=self.inverted,
        )

    @property
    def has_mask(self):
        return self.mask is not None and np.any(self.mask > 0)

    @property
    def pixel_count(self):
        return int(np.sum(self.mask > 0)) if self.mask is not None else 0

    @property
    def name(self) -> str:
        """Backward compat: some code checks layer.name == 'cavity'."""
        return self.category


class MaskStore:
    """Manages all layer instances for the current frame and handles disk persistence."""

    def __init__(self, base_output_dir: Path, video_folder: Path | None = None):
        self.base = Path(base_output_dir)
        self.video_folder = Path(video_folder) if video_folder else None
        self.layers: dict[str, MaskLayer] = {}
        self._init_default_layers()
        self.active_layer_name: str = "cavity"
        self._lock_path: Path | None = None

    def _init_default_layers(self):
        """Start with one instance per category."""
        for cat in CATEGORY_NAMES:
            key = instance_key(cat, 1)
            self.layers[key] = MaskLayer(category=cat, instance_id=1)

    # ── instance management ──────────────────────────────────────────

    def add_instance(self, category: str) -> str | None:
        """Create a new instance for *category*. Returns the key or None if max reached."""
        cfg = LAYER_CATEGORIES[category]
        existing = [l for l in self.layers.values() if l.category == category]
        if len(existing) >= cfg["max_instances"]:
            return None
        next_id = max(l.instance_id for l in existing) + 1 if existing else 1
        key = instance_key(category, next_id)
        self.layers[key] = MaskLayer(category=category, instance_id=next_id)
        return key

    def remove_instance(self, key: str) -> bool:
        """Remove an instance if it has no data. Returns True if removed."""
        layer = self.layers.get(key)
        if layer is None:
            return False
        cat_layers = [l for l in self.layers.values() if l.category == layer.category]
        if len(cat_layers) <= 1:
            return False
        if layer.has_mask:
            return False
        del self.layers[key]
        if self.active_layer_name == key:
            self.active_layer_name = cat_layers[0].key if cat_layers[0].key != key else cat_layers[1].key
        return True

    def instances_for(self, category: str) -> list[MaskLayer]:
        """Return all instances for a category, sorted by instance_id."""
        return sorted(
            [l for l in self.layers.values() if l.category == category],
            key=lambda l: l.instance_id,
        )

    def get_category_aggregate(self, category: str) -> np.ndarray | None:
        """OR-merge all instance masks for a category into a single mask."""
        merged = None
        for layer in self.instances_for(category):
            if layer.mask is not None:
                if merged is None:
                    merged = layer.mask.copy()
                else:
                    merged = cv2.bitwise_or(merged, layer.mask)
        return merged

    # ── priority / mutual exclusion ─────────────────────────────────

    def get_prior_mask(self, category: str) -> np.ndarray | None:
        """OR-merge masks from all categories with higher priority than *category*."""
        merged = None
        for prior_cat in prior_categories(category):
            agg = self.get_category_aggregate(prior_cat)
            if agg is not None:
                if merged is None:
                    merged = agg.copy()
                else:
                    merged = cv2.bitwise_or(merged, agg)
        return merged

    def subtract_from_others(self, active_key: str):
        """Remove the active layer's mask pixels from every other layer."""
        active = self.layers.get(active_key)
        if active is None or active.mask is None:
            return
        active_mask = active.mask
        for key, layer in self.layers.items():
            if key == active_key or layer.mask is None:
                continue
            layer.mask = cv2.bitwise_and(layer.mask, cv2.bitwise_not(active_mask))

    def hard_subtract_priors(self, category: str):
        """Remove higher-priority mask pixels from all instances of *category*."""
        prior = self.get_prior_mask(category)
        if prior is None:
            return
        inv = cv2.bitwise_not(prior)
        for layer in self.instances_for(category):
            if layer.mask is not None:
                layer.mask = cv2.bitwise_and(layer.mask, inv)

    def sample_prior_negatives(self, category: str, n_per_prior: int = 12) -> list[tuple[int, int]]:
        """Sample (x, y) points from higher-priority masks for SAM2 negative context."""
        prior = self.get_prior_mask(category)
        if prior is None:
            return []
        ys, xs = np.where(prior > 127)
        if len(ys) == 0:
            return []
        n = min(n_per_prior, len(ys))
        indices = np.linspace(0, len(ys) - 1, n, dtype=int)
        return [(int(xs[i]), int(ys[i])) for i in indices]

    # ── lock file for concurrent access ─────────────────────────────

    def acquire_lock(self):
        lock = self.base.parent / ".annotation.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        if lock.exists():
            try:
                info = json.loads(lock.read_text())
                print(f"[WARN] Lock held by {info.get('user')}@{info.get('host')} "
                      f"(pid {info.get('pid')})")
            except Exception:
                pass
        lock.write_text(json.dumps({
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "time": datetime.now(timezone.utc).isoformat(),
        }))
        self._lock_path = lock

    def release_lock(self):
        if self._lock_path and self._lock_path.exists():
            try:
                self._lock_path.unlink()
            except OSError:
                pass

    # ── atomic JSON helper ──────────────────────────────────────────

    @staticmethod
    def _atomic_write_json(path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, str(path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @property
    def active(self) -> MaskLayer:
        return self.layers[self.active_layer_name]

    def set_active(self, key: str):
        if key in self.layers:
            self.active_layer_name = key

    # ── per-layer disk paths ────────────────────────────────────────

    def _category_dir(self, category: str) -> Path:
        return self.base.parent / category

    def _frame_dir(self, category: str, frame_idx: int) -> Path:
        return self._category_dir(category) / f"frame_{frame_idx:06d}"

    # ── load ────────────────────────────────────────────────────────

    def load_frame(self, frame_idx: int):
        """Load all layer masks/points for *frame_idx* from disk."""
        for key in list(self.layers.keys()):
            cat, inst = parse_key(key)
            self._load_instance(key, cat, inst, frame_idx)
        self._discover_extra_instances(frame_idx)

    def _load_instance(self, key: str, category: str, inst_id: int, frame_idx: int):
        layer = self.layers[key]
        mask_path, data = self._find_mask(category, inst_id, frame_idx)
        if mask_path and mask_path.exists():
            layer.mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if data:
                layer.points_pos = data.get("points_pos", [])
                layer.points_neg = data.get("points_neg", [])
                layer.inverted = data.get("inverted", False)
                layer.active = data.get("active", True)
            else:
                layer.points_pos, layer.points_neg = [], []
                layer.inverted = False
                layer.active = True
        else:
            layer.mask = None
            layer.points_pos, layer.points_neg = [], []
            layer.inverted = False
            layer.active = True

    def _discover_extra_instances(self, frame_idx: int):
        """Check disk for instances we don't have in memory yet (e.g., tool_2)."""
        for cat in CATEGORY_NAMES:
            cfg = LAYER_CATEGORIES[cat]
            if cfg["max_instances"] <= 1:
                continue
            fdir = self._frame_dir(cat, frame_idx)
            if not fdir.exists():
                continue
            for i in range(1, cfg["max_instances"] + 1):
                key = instance_key(cat, i)
                if key in self.layers:
                    continue
                mask_file = fdir / f"{cat}_{i}_mask.png"
                if mask_file.exists():
                    self.layers[key] = MaskLayer(category=cat, instance_id=i)
                    self._load_instance(key, cat, i, frame_idx)

    def _find_mask(self, category: str, inst_id: int, frame_idx: int):
        """Return (mask_path, instance_data_dict_or_None)."""
        fdir = self._frame_dir(category, frame_idx)

        if LAYER_CATEGORIES[category]["max_instances"] == 1:
            mask_p = fdir / f"{category}_mask.png"
            data_p = fdir / "instance_data.json"
            if mask_p.exists():
                data = None
                if data_p.exists():
                    try:
                        raw = json.loads(data_p.read_text())
                        data = raw.get(category, raw) if isinstance(raw, dict) and "points_pos" not in raw else raw
                    except Exception:
                        pass
                return mask_p, data

            if category == "cavity":
                for legacy in self._cavity_legacy_dirs():
                    alt_mask = legacy / f"frame_{frame_idx:06d}" / "cavity_mask.png"
                    alt_data = legacy / f"frame_{frame_idx:06d}" / "instance_data.json"
                    if alt_mask.exists():
                        data = None
                        if alt_data.exists():
                            try:
                                data = json.loads(alt_data.read_text())
                            except Exception:
                                pass
                        return alt_mask, data
            return None, None

        inst_key = f"{category}_{inst_id}"
        mask_p = fdir / f"{inst_key}_mask.png"
        data_p = fdir / "instance_data.json"

        if mask_p.exists():
            data = None
            if data_p.exists():
                try:
                    raw = json.loads(data_p.read_text())
                    data = raw.get(inst_key, None)
                except Exception:
                    pass
            return mask_p, data

        if inst_id == 1:
            legacy_mask = fdir / f"{category}_mask.png"
            if legacy_mask.exists():
                data = None
                if data_p.exists():
                    try:
                        raw = json.loads(data_p.read_text())
                        data = raw if "points_pos" in raw else raw.get(category, None)
                    except Exception:
                        pass
                return legacy_mask, data

        return None, None

    def _cavity_legacy_dirs(self):
        if self.video_folder:
            yield self.video_folder / "masks" / "cavity"
            yield self.video_folder / "pipeline" / "02_cavity" / "cavity_only" / "frames"
            yield self.video_folder / "pipeline" / "02_cavity" / "masks"

    # ── save ────────────────────────────────────────────────────────

    def save_frame(self, frame_idx: int, keys_to_save: list[str] | None = None):
        """Persist mask + points for each requested instance (default: all with data)."""
        keys = keys_to_save or list(self.layers.keys())

        by_category: dict[str, list[str]] = {}
        for k in keys:
            cat, _ = parse_key(k)
            by_category.setdefault(cat, []).append(k)

        for cat, cat_keys in by_category.items():
            fdir = self._frame_dir(cat, frame_idx)
            any_data = False

            for k in cat_keys:
                layer = self.layers[k]
                if not layer.has_mask and not layer.points_pos and not layer.points_neg:
                    continue
                any_data = True
                fdir.mkdir(parents=True, exist_ok=True)

                if layer.mask is not None:
                    if LAYER_CATEGORIES[cat]["max_instances"] == 1:
                        mask_filename = f"{cat}_mask.png"
                    else:
                        mask_filename = f"{cat}_{layer.instance_id}_mask.png"
                    cv2.imwrite(str(fdir / mask_filename), layer.mask)

            if not any_data:
                continue

            all_data = {}
            for k in cat_keys:
                layer = self.layers[k]
                if not layer.has_mask and not layer.points_pos and not layer.points_neg:
                    continue
                mask_sha = ""
                if layer.mask is not None:
                    if LAYER_CATEGORIES[cat]["max_instances"] == 1:
                        mp = fdir / f"{cat}_mask.png"
                    else:
                        mp = fdir / f"{cat}_{layer.instance_id}_mask.png"
                    if mp.exists():
                        mask_sha = hashlib.sha256(mp.read_bytes()).hexdigest()

                inst_data = {
                    "points_pos": layer.points_pos,
                    "points_neg": layer.points_neg,
                    "inverted": layer.inverted,
                    "active": layer.active,
                    "mask_sha256": mask_sha,
                }

                if LAYER_CATEGORIES[cat]["max_instances"] == 1:
                    all_data = inst_data
                else:
                    all_data[k] = inst_data

            self._atomic_write_json(fdir / "instance_data.json", all_data)

    # ── session summary ─────────────────────────────────────────────

    def write_session_summary(self, video_path, start_frame, end_frame, modified_frames):
        summary = {
            "video_name": Path(video_path).stem,
            "video_path": str(video_path),
            "frame_range": {"start": start_frame, "end": end_frame},
            "modified_frames_count": len(modified_frames),
            "modified_frames": sorted(modified_frames),
            "layers": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
        }
        for cat in CATEGORY_NAMES:
            cat_dir = self._category_dir(cat)
            count = 0
            if cat_dir.exists():
                for d in cat_dir.iterdir():
                    if d.is_dir():
                        masks = list(d.glob(f"{cat}*_mask.png"))
                        if masks:
                            count += 1
            summary["layers"][cat] = {"total_masks": count}

        out = self.base.parent / "session_summary.json"
        self._atomic_write_json(out, summary)
        return out

    # ── helpers ──────────────────────────────────────────────────────

    def reset_layer(self, key: str):
        layer = self.layers.get(key)
        if layer:
            layer.mask = None
            layer.points_pos, layer.points_neg = [], []
            layer.inverted = False

    def reset_all(self):
        for key in list(self.layers.keys()):
            self.reset_layer(key)

    def get_active_masks_for_propagation(self):
        """Return list of (key, obj_id, mask) for all instances that have masks."""
        result = []
        for key, layer in self.layers.items():
            if layer.has_mask and layer.active:
                result.append((key, layer.obj_id, layer.mask))
        return result

    def apply_propagated(self, frame_results: dict[int, np.ndarray]):
        """Apply propagated masks (keyed by obj_id) to the current layers."""
        for key, layer in self.layers.items():
            if layer.obj_id in frame_results:
                mask = frame_results[layer.obj_id]
                pixel_count = int(np.sum(mask > 0))
                if pixel_count < MIN_MASK_PIXELS:
                    layer.mask = None
                    layer.active = False
                else:
                    layer.mask = mask
                    layer.active = True
                layer.points_pos, layer.points_neg = [], []

    def count_masks_in_range(self, category: str, start: int, end: int) -> int:
        """Count saved masks for *category* across a frame range."""
        primary_dir = self._category_dir(category)
        using_legacy = False
        layer_dir = primary_dir
        if not layer_dir.exists():
            if category == "cavity":
                for legacy in self._cavity_legacy_dirs():
                    if legacy.exists():
                        layer_dir = legacy
                        using_legacy = True
                        break
                else:
                    return 0
            else:
                return 0
        count = 0
        for idx in range(start, end + 1):
            fdir = layer_dir / f"frame_{idx:06d}"
            if not fdir.exists():
                continue
            if using_legacy:
                if (fdir / "cavity_mask.png").exists():
                    count += 1
            else:
                if list(fdir.glob(f"{category}*_mask.png")):
                    count += 1
        return count
