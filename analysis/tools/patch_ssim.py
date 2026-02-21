#!/usr/bin/env python3
"""
Patch SSIM Matrices into Existing functional_data.npz
======================================================
Targeted fix: computes only the 17x17 SSIM matrices that failed on the
server (scikit-image was missing), then patches them into the existing
functional_data.npz. Skips all other computations (hypsometric, polar,
radial, contour, histogram) since those are already correct.

Usage:
    python scripts/03_analysis/patch_ssim.py
    python scripts/03_analysis/patch_ssim.py --workers 8
"""

from __future__ import annotations

import cv2
import logging
import numpy as np
from pathlib import Path
import argparse
import time
import sys
import os
import warnings

_log = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='Maximum number of iterations')

from lib.fluorescence_transforms import compute_all_transforms, TRANSFORM_REGISTRY
from lib.topographic_analysis import normalize_surface
from lib.utils import crop_ui, load_mask, load_analysis_mask
from skimage.metrics import structural_similarity as ssim

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ── Defaults (from centralized config) ─────────────────────────────────────
from config import DEFAULT_VIDEO, DEFAULT_MASKS, DEFAULT_NPZ
_DEFAULT_VIDEO = DEFAULT_VIDEO
_DEFAULT_MASKS = DEFAULT_MASKS
_DEFAULT_NPZ = DEFAULT_NPZ

TRANSFORM_NAMES = list(TRANSFORM_REGISTRY.keys())
N_TRANSFORMS = len(TRANSFORM_NAMES)


def compute_ssim_for_frame(
    cap: cv2.VideoCapture,
    frame_idx: int,
    cavity_dir: str | Path,
    refl_dir: str | Path,
    blood_dir: str | Path,
    downsample: int = 4,
) -> np.ndarray:
    """Compute only the 17x17 SSIM matrix for one frame.

    Downsamples surfaces before SSIM to reduce computation (~16x faster at 4x).
    SSIM uses local windows so downsampling preserves structural information.
    """
    nan_matrix = np.full((N_TRANSFORMS, N_TRANSFORMS), np.nan)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        return nan_matrix
    frame_bgr = crop_ui(frame_bgr)

    cavity_mask, analysis_mask, _ = load_analysis_mask(
        frame_idx, cavity_dir, refl_dir, blood_dir, frame_bgr.shape[:2])
    if cavity_mask is None or analysis_mask is None:
        return nan_matrix

    # Compute transforms and normalize
    surfaces = compute_all_transforms(frame_bgr, analysis_mask)
    filled = {}
    for key in TRANSFORM_NAMES:
        if key in surfaces:
            norm_surf, _ = normalize_surface(surfaces[key], analysis_mask, 'p99')
            s = norm_surf.copy()
            s[~analysis_mask] = 0.0
            s = np.nan_to_num(s, nan=0.0)
            # Downsample for faster SSIM
            if downsample > 1:
                s = cv2.resize(s, (s.shape[1] // downsample, s.shape[0] // downsample),
                               interpolation=cv2.INTER_AREA)
            filled[key] = s
        else:
            h, w = analysis_mask.shape
            if downsample > 1:
                h, w = h // downsample, w // downsample
            filled[key] = np.zeros((h, w), dtype=np.float32)

    # Compute 17x17 SSIM
    matrix = np.eye(N_TRANSFORMS, dtype=np.float64)
    for i in range(N_TRANSFORMS):
        for j in range(i + 1, N_TRANSFORMS):
            try:
                val = ssim(filled[TRANSFORM_NAMES[i]], filled[TRANSFORM_NAMES[j]],
                           data_range=1.0)
                matrix[i, j] = val
                matrix[j, i] = val
            except Exception:
                matrix[i, j] = np.nan
                matrix[j, i] = np.nan

    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Patch SSIM matrices into existing functional_data.npz")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--masks-dir", type=str, default=None)
    parser.add_argument("--npz", type=str, default=None)
    args = parser.parse_args()

    VIDEO_PATH = Path(args.video) if args.video else _DEFAULT_VIDEO
    masks_base = Path(args.masks_dir) if args.masks_dir else _DEFAULT_MASKS
    npz_path = Path(args.npz) if args.npz else _DEFAULT_NPZ
    for name, val in [("--video", VIDEO_PATH), ("--masks-dir", masks_base),
                      ("--npz", npz_path)]:
        if val is None:
            print(f"ERROR: {name} is required (or set ALA_WORKSPACE)", file=sys.stderr)
            sys.exit(1)
    CAVITY_DIR = masks_base / "cavity"
    REFL_DIR = masks_base / "reflection"
    BLOOD_DIR = masks_base / "blood"

    # Validate
    for p, name in [(VIDEO_PATH, "Video"), (CAVITY_DIR, "Cavity masks"),
                     (npz_path, "functional_data.npz")]:
        if not p.exists():
            _log.error("%s not found: %s", name, p)
            sys.exit(1)

    # Load existing npz to get frame indices
    existing = dict(np.load(npz_path))
    frame_indices = existing['frame_indices']
    n_frames = len(frame_indices)
    _log.info("Loaded %s", npz_path)
    _log.info("  %d frames to process", n_frames)
    _log.info("  Current SSIM all NaN: %s", np.all(np.isnan(existing['ssim_matrices'])))

    # Verify scikit-image works
    test_a = np.random.rand(10, 10).astype(np.float32)
    test_b = np.random.rand(10, 10).astype(np.float32)
    test_val = ssim(test_a, test_b, data_range=1.0)
    _log.info("  scikit-image SSIM test: %.4f (working)", test_val)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    ssim_matrices = np.full((n_frames, N_TRANSFORMS, N_TRANSFORMS), np.nan,
                            dtype=np.float64)

    t_start = time.time()
    success = 0

    for fi, frame_idx in enumerate(tqdm(frame_indices, desc="Computing SSIM")):
        matrix = compute_ssim_for_frame(
            cap, int(frame_idx), CAVITY_DIR, REFL_DIR, BLOOD_DIR)
        ssim_matrices[fi] = matrix
        if not np.all(np.isnan(matrix)):
            success += 1

    cap.release()
    elapsed = time.time() - t_start

    # Patch into existing data
    existing['ssim_matrices'] = ssim_matrices

    # Save (backup original first)
    backup_path = npz_path.with_suffix('.npz.bak')
    if not backup_path.exists():
        import shutil
        shutil.copy2(npz_path, backup_path)
        _log.info("Backed up original to: %s", backup_path)

    np.savez_compressed(npz_path, **existing)

    _log.info("SSIM PATCH COMPLETE in %.1fs (%.1f min)", elapsed, elapsed / 60)
    _log.info("  Frames with valid SSIM: %d/%d", success, n_frames)
    _log.info("  Mean SSIM (non-diagonal): %.4f",
              np.nanmean(ssim_matrices[ssim_matrices < 1.0]))
    _log.info("  Patched: %s", npz_path)


if __name__ == "__main__":
    main()
