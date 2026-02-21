#!/usr/bin/env python3
"""
Functional Data Extraction for Statistical Evaluation
=======================================================
Lightweight pass over video frames to extract array-level data that the
batch topographic pipeline discards (keeping only scalar summaries).

Extracts per frame x per transform:
  - Hypsometric curve:     100-point A(t) array
  - Radial profile:        200-point median radial decay
  - Boundary contour:      360-point r(theta)
  - Pixel histogram:       100-bin normalized intensity histogram
  - Pairwise SSIM matrix:  17x17 structural similarity
  - Pairwise corr matrix:  17x17 Pearson spatial correlation

Output: single functional_data.npz file for use by
comprehensive_statistical_evaluation.py

Usage:
    python scripts/03_analysis/extract_functional_data.py
    python scripts/03_analysis/extract_functional_data.py --workers 8
    python scripts/03_analysis/extract_functional_data.py --max-frames 20
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
import multiprocessing as mp

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Maximum number of iterations')

_log = logging.getLogger(__name__)

from lib.fluorescence_transforms import (
    compute_all_transforms, TRANSFORM_REGISTRY, can_be_negative
)
from lib.topographic_analysis import (
    normalize_surface, compute_hypsometric_curve, find_peak
)
from lib.polar_surface_analysis import (
    build_polar_surface_fast, compute_radial_profiles,
    extract_boundary_contour
)
from lib.blood_detection import check_light_stability
from lib.utils import (
    crop_ui,
    discover_mask_frames,
    load_mask,
    load_analysis_mask,
    parse_include_csv,
    setup_logging,
    SequentialVideoReader,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ── Default paths & analysis constants (from centralized config) ───────────
from config import (
    DEFAULT_VIDEO, DEFAULT_MASKS, DEFAULT_OUTPUT,
    N_HYPSO, N_RADII, N_ANGLES, N_HIST_BINS, POLAR_MAX_RADIUS,
)
_DEFAULT_VIDEO = DEFAULT_VIDEO
_DEFAULT_MASKS = DEFAULT_MASKS
_DEFAULT_OUTPUT = DEFAULT_OUTPUT

TRANSFORM_NAMES = list(TRANSFORM_REGISTRY.keys())
N_TRANSFORMS = len(TRANSFORM_NAMES)
CONTOUR_THRESHOLD = 0.5


def compute_pairwise_ssim(
    surfaces: dict[str, np.ndarray],
    analysis_mask: np.ndarray,
    transform_keys: list[str],
) -> np.ndarray:
    """Compute 17x17 SSIM matrix between normalized surfaces."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return np.full((len(transform_keys), len(transform_keys)), np.nan)

    n = len(transform_keys)
    matrix = np.eye(n, dtype=np.float64)

    filled = {}
    for key in transform_keys:
        if key in surfaces:
            s = surfaces[key].copy()
            s[~analysis_mask] = 0.0
            s = np.nan_to_num(s, nan=0.0)
            filled[key] = s
        else:
            filled[key] = np.zeros(analysis_mask.shape, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                val = ssim(filled[transform_keys[i]], filled[transform_keys[j]],
                           data_range=1.0)
                matrix[i, j] = val
                matrix[j, i] = val
            except Exception as e:
                _log.debug("SSIM failed for %s vs %s: %s",
                           transform_keys[i], transform_keys[j], e)
                matrix[i, j] = np.nan
                matrix[j, i] = np.nan

    return matrix


def compute_pairwise_correlation(
    surfaces: dict[str, np.ndarray],
    analysis_mask: np.ndarray,
    transform_keys: list[str],
) -> np.ndarray:
    """Compute 17x17 Pearson correlation matrix between surfaces within mask."""
    n = len(transform_keys)
    matrix = np.eye(n, dtype=np.float64)

    vectors = {}
    for key in transform_keys:
        if key in surfaces:
            vals = surfaces[key][analysis_mask]
            vals = np.nan_to_num(vals, nan=0.0)
            vectors[key] = vals
        else:
            vectors[key] = np.zeros(int(np.sum(analysis_mask)), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                v1 = vectors[transform_keys[i]]
                v2 = vectors[transform_keys[j]]
                if np.std(v1) > 0 and np.std(v2) > 0:
                    corr = float(np.corrcoef(v1, v2)[0, 1])
                else:
                    corr = np.nan
                matrix[i, j] = corr
                matrix[j, i] = corr
            except Exception as e:
                _log.debug("Correlation failed for %s vs %s: %s",
                           transform_keys[i], transform_keys[j], e)
                matrix[i, j] = np.nan
                matrix[j, i] = np.nan

    return matrix


# ── Per-worker state ──────────────────────────────────────────────────────
_worker_cap = None


def _worker_init(video_path_str: str) -> None:
    """Initialize per-worker VideoCapture."""
    import atexit
    global _worker_cap
    _worker_cap = cv2.VideoCapture(video_path_str)
    atexit.register(lambda: _worker_cap.release() if _worker_cap else None)


def process_frame_worker(args: tuple[int, dict]) -> tuple[int, dict | None]:
    """Process a single frame in a worker process.

    Parameters: tuple of (frame_idx, config_dict)
    Returns: (frame_idx, result_dict) or (frame_idx, None)
    """
    frame_idx, cfg = args
    global _worker_cap

    cavity_dir = Path(cfg['cavity_dir'])
    refl_dir = Path(cfg['refl_dir'])
    blood_dir = Path(cfg['blood_dir'])
    min_median_v = cfg.get('min_median_v', 0.08)
    skip_transforms = set(cfg.get('skip_transforms', []))
    skip_ssim = cfg.get('skip_ssim', False)

    # Use per-worker VideoCapture
    cap = _worker_cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(cfg['video_path'])

    # Read frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        print(f"Frame {frame_idx}: video read failed", file=sys.stderr)
        return frame_idx, None
    frame_bgr = crop_ui(frame_bgr)

    # Load cavity + exclusion masks
    cavity_mask, analysis_mask, mask_info = load_analysis_mask(
        frame_idx, cavity_dir, refl_dir, blood_dir, frame_bgr.shape[:2])
    if cavity_mask is None:
        print(f"Frame {frame_idx}: cavity mask not found", file=sys.stderr)
        return frame_idx, None
    if analysis_mask is None:
        print(f"Frame {frame_idx}: insufficient analysis pixels ({mask_info.get('analysis_px', 0)})",
              file=sys.stderr)
        return frame_idx, None

    analysis_px = mask_info['analysis_px']

    # Light stability check
    stable, median_v = check_light_stability(frame_bgr, cavity_mask, min_median_v)

    # V channel for peak finding
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2].astype(np.float32) / 255.0

    # Compute all transforms
    surfaces = compute_all_transforms(frame_bgr, analysis_mask, skip=skip_transforms)

    # Storage arrays for this frame
    hypso_curves = np.full((N_TRANSFORMS, N_HYPSO), np.nan, dtype=np.float64)
    radial_profiles = np.full((N_TRANSFORMS, N_RADII), np.nan, dtype=np.float64)
    boundary_contours = np.full((N_TRANSFORMS, N_ANGLES), np.nan, dtype=np.float64)
    pixel_histograms = np.full((N_TRANSFORMS, N_HIST_BINS), np.nan, dtype=np.float64)

    normalized_surfaces = {}
    hist_bin_edges = np.linspace(0, 1, N_HIST_BINS + 1)

    for ti, tkey in enumerate(TRANSFORM_NAMES):
        if tkey not in surfaces:
            continue

        surf = surfaces[tkey]
        try:
            norm_surf, _ = normalize_surface(surf, analysis_mask, 'p99')
            normalized_surfaces[tkey] = norm_surf

            # Hypsometric curve
            thresholds, area_fracs = compute_hypsometric_curve(
                norm_surf, analysis_mask, n_levels=N_HYPSO)
            hypso_curves[ti, :] = area_fracs

            # Peak finding
            use_v = v_ch if can_be_negative(tkey) or tkey in (
                'R', 'R_G', 'R_RpG', 'R_B', 'R_GpB', 'R_minus_kG',
                'NMF_fluor', 'PCA_PC1') else None
            peak_yx, _ = find_peak(surf, analysis_mask, use_v)

            # Polar transform + radial profile
            polar, radii, angles = build_polar_surface_fast(
                norm_surf, analysis_mask, peak_yx,
                max_radius=POLAR_MAX_RADIUS,
                n_radii=N_RADII, n_angles=N_ANGLES)
            rad_stats = compute_radial_profiles(polar, radii)
            radial_profiles[ti, :] = rad_stats['median']

            # Boundary contour
            contour_r = extract_boundary_contour(
                polar, radii, threshold=CONTOUR_THRESHOLD)
            boundary_contours[ti, :] = contour_r

            # Pixel histogram
            valid_px = norm_surf[analysis_mask]
            valid_px = valid_px[~np.isnan(valid_px)]
            if len(valid_px) > 0:
                hist, _ = np.histogram(valid_px, bins=hist_bin_edges, density=True)
                hist_sum = hist.sum()
                if hist_sum > 0:
                    hist = hist / hist_sum
                pixel_histograms[ti, :] = hist

        except Exception as e:
            print(f"Frame {frame_idx}: transform {tkey} failed: {e}", file=sys.stderr)
            continue

    # Pairwise SSIM and correlation matrices
    if skip_ssim:
        ssim_matrix = np.full((N_TRANSFORMS, N_TRANSFORMS), np.nan)
    else:
        ssim_matrix = compute_pairwise_ssim(
            normalized_surfaces, analysis_mask, TRANSFORM_NAMES)
    corr_matrix = compute_pairwise_correlation(
        normalized_surfaces, analysis_mask, TRANSFORM_NAMES)

    return frame_idx, {
        'frame_idx': frame_idx,
        'light_stable': stable,
        'median_v': median_v,
        'analysis_px': analysis_px,
        'hypso_curves': hypso_curves,
        'radial_profiles': radial_profiles,
        'boundary_contours': boundary_contours,
        'pixel_histograms': pixel_histograms,
        'ssim_matrix': ssim_matrix,
        'corr_matrix': corr_matrix,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract functional data (curves, profiles, contours) for "
                    "comprehensive statistical evaluation")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--masks-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip-nmf", action="store_true",
                        help="Skip NMF transform (slow)")
    parser.add_argument("--skip-ssim", action="store_true",
                        help="Skip pairwise SSIM computation (major bottleneck: "
                             "136 full-frame SSIM calls per frame)")
    parser.add_argument("--min-median-v", type=float, default=0.08)
    parser.add_argument("--stable-only", action="store_true", default=False,
                        help="Only process light-stable frames")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = auto: ncpu-1, 1 = single-threaded)")
    parser.add_argument("--include-csv", type=str, default=None,
                        help="Path to manual_frame_classifications.csv. "
                             "Only frames classified as 'include' will be processed.")
    args = parser.parse_args()

    # Resolve paths
    VIDEO_PATH = Path(args.video) if args.video else _DEFAULT_VIDEO
    masks_base = Path(args.masks_dir) if args.masks_dir else _DEFAULT_MASKS
    OUTPUT_DIR = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT
    for name, val in [("--video", VIDEO_PATH), ("--masks-dir", masks_base),
                      ("--output-dir", OUTPUT_DIR)]:
        if val is None:
            print(f"ERROR: {name} is required (or set ALA_WORKSPACE)", file=sys.stderr)
            sys.exit(1)
    CAVITY_DIR = masks_base / "cavity"
    REFL_DIR = masks_base / "reflection"
    BLOOD_DIR = masks_base / "blood"

    logger = setup_logging('extract_functional_data', output_dir=OUTPUT_DIR)

    # Auto-detect workers
    if args.workers == 0:
        args.workers = max(1, (os.cpu_count() or 1) - 1)
    logger.info(f"Workers: {args.workers}")

    # Validate
    if not VIDEO_PATH.exists():
        logger.error(f"Video not found: {VIDEO_PATH}")
        sys.exit(1)
    if not CAVITY_DIR.exists():
        logger.error(f"Cavity masks not found: {CAVITY_DIR}")
        sys.exit(1)

    frame_indices = discover_mask_frames(CAVITY_DIR)
    logger.info(f"Found {len(frame_indices)} frames with cavity masks")

    # ── Include-list filtering ──────────────────────────────────────────
    if args.include_csv:
        include_csv_path = Path(args.include_csv)
        if not include_csv_path.exists():
            logger.error(f"Include CSV not found: {include_csv_path}")
            sys.exit(1)
        include_set = parse_include_csv(include_csv_path)
        before = len(frame_indices)
        frame_indices = [fi for fi in frame_indices if fi in include_set]
        logger.info(f"Include CSV:   {include_csv_path}")
        logger.info(f"  Include-list frames: {len(include_set)}")
        logger.info(f"  Filtered {before} -> {len(frame_indices)} frames")

    if args.max_frames is not None:
        frame_indices = frame_indices[:args.max_frames]
        logger.info(f"Limited to first {args.max_frames}")

    skip_transforms = list()
    if args.skip_nmf:
        skip_transforms.append('NMF_fluor')

    # Verify video opens
    test_reader = SequentialVideoReader(VIDEO_PATH)
    total_frames = test_reader.total_frames
    test_reader.release()
    logger.info(f"Video has {total_frames} frames")

    # Build config for workers
    cfg = {
        'video_path': str(VIDEO_PATH),
        'cavity_dir': str(CAVITY_DIR),
        'refl_dir': str(REFL_DIR),
        'blood_dir': str(BLOOD_DIR),
        'skip_transforms': skip_transforms,
        'skip_ssim': args.skip_ssim,
        'min_median_v': args.min_median_v,
    }
    work_items = [(fidx, cfg) for fidx in frame_indices]

    # Collect results
    all_hypso = []
    all_radial = []
    all_contour = []
    all_hist = []
    all_ssim = []
    all_corr = []
    all_frame_idx = []
    all_stable = []
    all_median_v = []

    t_start = time.time()
    success = 0
    failed = 0

    # Set multiprocessing start method
    if args.workers > 1:
        import platform
        method = 'forkserver' if platform.system() == 'Linux' else 'spawn'
        try:
            mp.set_start_method(method, force=True)
            logger.info(f"Multiprocessing method: {method}")
        except RuntimeError:
            actual = mp.get_start_method()
            logger.info(f"Multiprocessing method: {actual} (requested {method}, "
                       f"already set by another module)")

    def collect_result(frame_idx, result):
        nonlocal success, failed
        if result is None:
            failed += 1
            return
        if args.stable_only and not result['light_stable']:
            return
        all_frame_idx.append(result['frame_idx'])
        all_stable.append(result['light_stable'])
        all_median_v.append(result['median_v'])
        all_hypso.append(result['hypso_curves'])
        all_radial.append(result['radial_profiles'])
        all_contour.append(result['boundary_contours'])
        all_hist.append(result['pixel_histograms'])
        all_ssim.append(result['ssim_matrix'])
        all_corr.append(result['corr_matrix'])
        success += 1

    if args.workers <= 1:
        # Single-threaded
        _worker_init(str(VIDEO_PATH))
        for item in tqdm(work_items, desc="Extracting functional data"):
            fidx, result = process_frame_worker(item)
            collect_result(fidx, result)
    else:
        # Parallel
        logger.info(f"Starting {args.workers} worker processes...")
        with mp.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(str(VIDEO_PATH),)
        ) as pool:
            results_iter = pool.imap_unordered(process_frame_worker, work_items)
            for fidx, result in tqdm(results_iter, total=len(work_items),
                                      desc="Extracting functional data"):
                collect_result(fidx, result)

    elapsed = time.time() - t_start

    if success == 0:
        logger.error("No frames processed successfully")
        sys.exit(1)

    # Sort by frame index for consistent ordering
    sort_idx = np.argsort(all_frame_idx)
    all_frame_idx = [all_frame_idx[i] for i in sort_idx]
    all_stable = [all_stable[i] for i in sort_idx]
    all_median_v = [all_median_v[i] for i in sort_idx]
    all_hypso = [all_hypso[i] for i in sort_idx]
    all_radial = [all_radial[i] for i in sort_idx]
    all_contour = [all_contour[i] for i in sort_idx]
    all_hist = [all_hist[i] for i in sort_idx]
    all_ssim = [all_ssim[i] for i in sort_idx]
    all_corr = [all_corr[i] for i in sort_idx]

    # Stack into arrays
    hypso_curves = np.stack(all_hypso, axis=0)
    radial_profiles = np.stack(all_radial, axis=0)
    boundary_contours = np.stack(all_contour, axis=0)
    pixel_histograms = np.stack(all_hist, axis=0)
    ssim_matrices = np.stack(all_ssim, axis=0)
    corr_matrices = np.stack(all_corr, axis=0)
    frame_indices_arr = np.array(all_frame_idx)
    light_stable = np.array(all_stable)
    median_v = np.array(all_median_v)

    # Shared coordinate arrays
    hypso_thresholds = np.linspace(0, 1, N_HYPSO)
    radii = np.linspace(0, POLAR_MAX_RADIUS, N_RADII)
    angles_deg = np.linspace(0, 360, N_ANGLES, endpoint=False)
    hist_bin_centers = (np.linspace(0, 1, N_HIST_BINS + 1)[:-1]
                        + np.linspace(0, 1, N_HIST_BINS + 1)[1:]) / 2

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "functional_data.npz"
    np.savez_compressed(
        out_path,
        hypso_curves=hypso_curves,
        radial_profiles=radial_profiles,
        boundary_contours=boundary_contours,
        pixel_histograms=pixel_histograms,
        ssim_matrices=ssim_matrices,
        corr_matrices=corr_matrices,
        hypso_thresholds=hypso_thresholds,
        radii=radii,
        angles_deg=angles_deg,
        hist_bin_centers=hist_bin_centers,
        frame_indices=frame_indices_arr,
        light_stable=light_stable,
        median_v=median_v,
        transform_names=np.array(TRANSFORM_NAMES),
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Workers:          {args.workers}")
    logger.info(f"  Frames processed: {success}")
    logger.info(f"  Frames failed:    {failed}")
    logger.info(f"  Output:           {out_path}")
    logger.info(f"  File size:        {out_path.stat().st_size / 1e6:.1f} MB")
    logger.info(f"\nArray shapes:")
    logger.info(f"  hypso_curves:      {hypso_curves.shape}")
    logger.info(f"  radial_profiles:   {radial_profiles.shape}")
    logger.info(f"  boundary_contours: {boundary_contours.shape}")
    logger.info(f"  pixel_histograms:  {pixel_histograms.shape}")
    logger.info(f"  ssim_matrices:     {ssim_matrices.shape}")
    logger.info(f"  corr_matrices:     {corr_matrices.shape}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
