#!/usr/bin/env python3
"""
Batch Topographic Fluorescence Analysis for 5fps Data
======================================================
Runs the full topographic analysis pipeline on all frames:
  1. Load frame + cavity mask + exclusion masks
  2. Check light stability (skip unstable frames)
  3. Compute all color transformations
  4. For each transform: normalize ONCE, then global + polar + TDA
  5. Save per-frame JSON, periodic + final summary CSV

Usage:
    python scripts/03_analysis/run_5fps_topographic_analysis.py --no-skip
    python scripts/03_analysis/run_5fps_topographic_analysis.py --max-frames 10
    python scripts/03_analysis/run_5fps_topographic_analysis.py --skip-tda --skip-nmf
    python scripts/03_analysis/run_5fps_topographic_analysis.py --include-csv path/to/manual_frame_classifications.csv --no-skip
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import json
import csv
import argparse
import time
import sys
import warnings
import multiprocessing as mp

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Maximum number of iterations')

from lib.fluorescence_transforms import (
    compute_all_transforms, can_be_negative
)
from lib.topographic_analysis import (
    normalize_surface, compute_global_metrics, find_peak,
    compute_chan_vese_boundary,
)
from lib.polar_surface_analysis import compute_polar_metrics
from lib.persistent_homology import (
    compute_persistence_cubical, persistence_summary_metrics
)
from lib.blood_detection import check_light_stability
from lib.utils import (
    crop_ui, discover_mask_frames, load_mask, load_analysis_mask, json_convert,
    parse_include_csv, setup_logging, SequentialVideoReader
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ── Default Paths (from centralized config, overridable via CLI) ───────────
from config import DEFAULT_VIDEO, DEFAULT_MASKS, DEFAULT_OUTPUT
_DEFAULT_VIDEO = DEFAULT_VIDEO
_DEFAULT_MASKS = DEFAULT_MASKS
_DEFAULT_OUTPUT = DEFAULT_OUTPUT


def write_csv(all_rows: list[dict], csv_path: Path) -> None:
    """Write summary CSV from accumulated rows."""
    if not all_rows:
        return
    all_keys = set()
    for row in all_rows:
        all_keys.update(row.keys())

    # Group columns: frame-level first, then by transform prefix
    frame_keys = ['frame_idx', 'cavity_px', 'reflection_px', 'blood_px',
                  'analysis_px', 'light_stable', 'median_v']
    frame_keys = [k for k in frame_keys if k in all_keys]
    transform_keys = sorted(k for k in all_keys if '__' in k)
    other_keys = sorted(k for k in all_keys if k not in frame_keys and '__' not in k)
    fieldnames = frame_keys + other_keys + transform_keys

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in sorted(all_rows, key=lambda r: r.get('frame_idx', 0)):
            writer.writerow({k: json_convert(v) for k, v in row.items()})


# ── Per-frame worker function ──────────────────────────────────────────────

# Global per-process VideoCapture (initialized by pool worker init)
_worker_cap = None

def _worker_init(video_path_str: str) -> None:
    """Initialize per-worker VideoCapture."""
    import atexit
    global _worker_cap
    _worker_cap = cv2.VideoCapture(video_path_str)
    atexit.register(lambda: _worker_cap.release() if _worker_cap else None)


def process_single_frame(args: tuple[int, dict]) -> tuple[int, dict | None]:
    """Process one frame. Designed to run in a worker process.

    Parameters: tuple of (frame_idx, config_dict)
    Returns: (frame_idx, metrics_dict) or (frame_idx, None) on failure
    """
    frame_idx, cfg = args
    global _worker_cap

    video_path = cfg['video_path']
    cavity_dir = Path(cfg['cavity_dir'])
    refl_dir = Path(cfg['refl_dir'])
    blood_dir = Path(cfg['blood_dir'])
    json_dir = Path(cfg['json_dir'])
    skip_transforms = set(cfg.get('skip_transforms', []))
    skip_tda = cfg.get('skip_tda', False)
    skip_chanvese = cfg.get('skip_chanvese', False)
    polar_max_radius = cfg.get('polar_max_radius', 200)
    tda_subsample = cfg.get('tda_subsample', 2)
    min_median_v = cfg.get('min_median_v', 0.08)
    total_video = cfg.get('total_video', 999999)

    json_out = json_dir / f"frame_{frame_idx:06d}.json"

    if frame_idx >= total_video:
        return frame_idx, None

    # Read frame (use per-worker capture, or open fresh if not available)
    cap = _worker_cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        return frame_idx, None
    frame_bgr = crop_ui(frame_bgr)

    # Load cavity + exclusion masks
    cavity_mask, analysis_mask, mask_info = load_analysis_mask(
        frame_idx, cavity_dir, refl_dir, blood_dir, frame_bgr.shape[:2])
    if cavity_mask is None or analysis_mask is None:
        return frame_idx, None

    refl_px = mask_info['refl_px']
    blood_px = mask_info['blood_px']
    analysis_px = mask_info['analysis_px']

    # Light stability
    stable, median_v = check_light_stability(
        frame_bgr, cavity_mask, min_median_v=min_median_v
    )

    # V and S channels
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2].astype(np.float32) / 255.0
    s_ch = hsv[:, :, 1].astype(np.float32) / 255.0

    # ── Saturation safety net ──────────────────────────────────────────
    # Exposed tissue retains high saturation (S ~0.85-1.0) under blue light.
    # Undetected blood has low saturation (S ~0.50-0.68).
    # This computes a flag (not a hard skip) so the metric is available
    # for downstream filtering.
    sat_floor = cfg.get('sat_floor', 0.40)
    analysis_s_vals = s_ch[analysis_mask]
    median_s_analysis = float(np.median(analysis_s_vals)) if len(analysis_s_vals) > 0 else 0.0
    low_sat_frac = float(np.mean(analysis_s_vals < 0.65)) if len(analysis_s_vals) > 0 else 0.0
    sat_flag = (sat_floor > 0) and (median_s_analysis < sat_floor)

    # Transforms
    surfaces = compute_all_transforms(frame_bgr, analysis_mask, skip=skip_transforms)

    frame_metrics = {
        'frame_idx': frame_idx,
        'cavity_px': int(np.sum(cavity_mask)),
        'reflection_px': refl_px,
        'blood_px': blood_px,
        'analysis_px': analysis_px,
        'light_stable': stable,
        'median_v': round(median_v, 4),
        'median_s_analysis': round(median_s_analysis, 4),
        'low_sat_frac': round(low_sat_frac, 4),
        'sat_flag': sat_flag,
    }

    for tkey, surf in surfaces.items():
        prefix = tkey
        try:
            use_v = v_ch if can_be_negative(tkey) or tkey in (
                'R', 'R_G', 'R_RpG', 'R_B', 'R_GpB', 'R_minus_kG',
                'NMF_fluor', 'PCA_PC1') else None

            peak_yx, peak_val = find_peak(surf, analysis_mask, use_v)

            gmetrics, garrays, norm_surf = compute_global_metrics(
                surf, analysis_mask, peak_yx, 'p99')
            for k, v in gmetrics.items():
                frame_metrics[f'{prefix}__{k}'] = json_convert(v)

            pmetrics, _ = compute_polar_metrics(
                norm_surf, analysis_mask, peak_yx,
                max_radius=polar_max_radius, contour_threshold=0.5)
            for k, v in pmetrics.items():
                frame_metrics[f'{prefix}__polar_{k}'] = json_convert(v)

            if not skip_tda:
                dgm = compute_persistence_cubical(
                    norm_surf, analysis_mask, subsample_factor=tda_subsample)
                tda_m = persistence_summary_metrics(dgm)
                for k, v in tda_m.items():
                    frame_metrics[f'{prefix}__tda_{k}'] = json_convert(v)

        except Exception as e:
            frame_metrics[f'{prefix}__error'] = str(e)[:200]
            import traceback
            print(f"WARNING: Transform {prefix} failed on frame {frame_idx}: {e}\n"
                  f"  {traceback.format_exc().splitlines()[-1]}", file=sys.stderr)

    if not skip_chanvese:
        for cv_key in ['R_G', 'R_RpG']:
            if cv_key in surfaces:
                try:
                    norm_s, _ = normalize_surface(surfaces[cv_key], analysis_mask, 'p99')
                    seg, area = compute_chan_vese_boundary(norm_s, analysis_mask, n_iter=100)
                    frame_metrics[f'{cv_key}__chanvese_area'] = json_convert(area)
                except Exception as e:
                    print(f"Chan-Vese failed on {cv_key} frame {frame_idx}: {e}",
                          file=sys.stderr)

    # Save JSON
    save_data = {'frame_idx': frame_idx, 'metrics': frame_metrics}
    with open(json_out, 'w') as f:
        json.dump(save_data, f, indent=1, default=json_convert)

    return frame_idx, frame_metrics


def main():
    parser = argparse.ArgumentParser(description="Batch topographic fluorescence analysis")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (default: auto-detect)")
    parser.add_argument("--masks-dir", type=str, default=None,
                        help="Path to masks/ directory containing cavity/, blood/, reflection/")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip-tda", action="store_true", help="Skip persistent homology")
    parser.add_argument("--skip-nmf", action="store_true", help="Skip NMF transform")
    parser.add_argument("--skip-chanvese", action="store_true", help="Skip Chan-Vese")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip", dest="skip_existing", action="store_false")
    parser.add_argument("--polar-max-radius", type=int, default=200)
    parser.add_argument("--tda-subsample", type=int, default=2,
                        help="TDA downscale factor (2=half res, 4=quarter)")
    parser.add_argument("--min-median-v", type=float, default=0.08,
                        help="Min median V for light stability")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1 = single-threaded)")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                        help="Write CSV checkpoint every N frames")
    parser.add_argument("--include-csv", type=str, default=None,
                        help="Path to manual_frame_classifications.csv. "
                             "Only frames classified as 'include' will be processed.")
    parser.add_argument("--sat-floor", type=float, default=0.40,
                        help="Saturation safety net: if median S of analysis pixels "
                             "is below this value, flag the frame as likely undetected "
                             "blood (default: 0.40). Set to 0 to disable.")
    args = parser.parse_args()

    # Resolve paths (CLI overrides defaults)
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
    logger = setup_logging('run_5fps_topographic_analysis', output_dir=OUTPUT_DIR)

    # Startup validation
    errors = []
    if not VIDEO_PATH.exists():
        errors.append(f"Video not found: {VIDEO_PATH}")
    if not CAVITY_DIR.exists():
        errors.append(f"Cavity masks not found: {CAVITY_DIR}")
    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)
    if not REFL_DIR.exists():
        logger.warning("Reflection masks not found: %s (will skip)", REFL_DIR)
    if not BLOOD_DIR.exists():
        logger.warning("Blood masks not found: %s (will skip)", BLOOD_DIR)

    logger.info(f"Video:         {VIDEO_PATH}")
    logger.info(f"Cavity masks:  {CAVITY_DIR}")
    logger.info(f"Skip TDA:      {args.skip_tda}")
    logger.info(f"Skip NMF:      {args.skip_nmf}")
    logger.info(f"Skip ChanVese: {args.skip_chanvese}")
    logger.info(f"Light min V:   {args.min_median_v}")

    frame_indices = discover_mask_frames(CAVITY_DIR)
    logger.info(f"\nFound {len(frame_indices)} frames with cavity masks")
    if not frame_indices:
        logger.error("No cavity masks found in %s — nothing to process.", CAVITY_DIR)
        sys.exit(1)
    logger.info(f"Range: {frame_indices[0]} - {frame_indices[-1]}")

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
        logger.info(f"  Filtered {before} -> {len(frame_indices)} frames (removed {before - len(frame_indices)})")

    if args.max_frames is not None:
        frame_indices = frame_indices[:args.max_frames]
        logger.info(f"Limited to first {args.max_frames}")

    json_dir = OUTPUT_DIR / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = OUTPUT_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "frame_metrics_summary.csv"

    skip_transforms = set()
    if args.skip_nmf:
        skip_transforms.add('NMF_fluor')

    # Get video frame count
    reader = SequentialVideoReader(VIDEO_PATH)
    total_video = reader.total_frames
    reader.release()
    logger.info(f"Video has {total_video} frames")
    logger.info(f"Workers:       {args.workers}\n")

    # Filter frames: skip existing if requested
    all_rows = []
    to_process = []
    skipped = 0
    for frame_idx in frame_indices:
        json_out = json_dir / f"frame_{frame_idx:06d}.json"
        if args.skip_existing and json_out.exists():
            try:
                with open(json_out) as f:
                    cached = json.load(f)
                if cached.get('metrics'):
                    all_rows.append(cached['metrics'])
                    skipped += 1
                    continue
            except Exception as e:
                logger.warning("Corrupted cache %s, will reprocess: %s",
                               json_out.name, e)
                json_out.unlink(missing_ok=True)
            to_process.append(frame_idx)
        else:
            to_process.append(frame_idx)

    logger.info(f"To process: {len(to_process)}, Cached: {skipped}")

    # Build config dict for workers
    cfg = {
        'video_path': str(VIDEO_PATH),
        'cavity_dir': str(CAVITY_DIR),
        'refl_dir': str(REFL_DIR),
        'blood_dir': str(BLOOD_DIR),
        'json_dir': str(json_dir),
        'skip_transforms': list(skip_transforms),
        'skip_tda': args.skip_tda,
        'skip_chanvese': args.skip_chanvese,
        'polar_max_radius': args.polar_max_radius,
        'tda_subsample': args.tda_subsample,
        'min_median_v': args.min_median_v,
        'total_video': total_video,
        'sat_floor': args.sat_floor,
    }
    work_items = [(fidx, cfg) for fidx in to_process]

    t_start = time.time()
    success = 0
    failed = 0
    light_unstable = 0

    # Set multiprocessing start method based on OS
    # NOTE: Each worker uses cv2.VideoCapture with cap.set(CAP_PROP_POS_FRAMES)
    # for random access. For H.264/H.265, this is O(keyframe-distance) per seek.
    # Consider using --workers 1 for short videos where seek overhead dominates.
    if args.workers > 1:
        import platform
        method = 'forkserver' if platform.system() == 'Linux' else 'spawn'
        try:
            mp.set_start_method(method, force=True)
            logger.info(f'Multiprocessing method: {method}')
        except RuntimeError:
            actual = mp.get_start_method()
            logger.info(f'Multiprocessing method: {actual} (requested {method}, '
                        f'already set by another module)')

    if args.workers <= 1:
        # Single-threaded mode (with progress bar)
        _worker_init(str(VIDEO_PATH))
        for item in tqdm(work_items, desc="Analyzing"):
            fidx, metrics = process_single_frame(item)
            if metrics is not None:
                all_rows.append(metrics)
                success += 1
                if not metrics.get('light_stable', True):
                    light_unstable += 1
                if args.checkpoint_every > 0 and success % args.checkpoint_every == 0:
                    write_csv(all_rows, csv_path)
            else:
                failed += 1
    else:
        # Parallel mode
        logger.info(f"Starting {args.workers} worker processes...")
        with mp.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(str(VIDEO_PATH),)
        ) as pool:
            results = pool.imap_unordered(process_single_frame, work_items)
            for fidx, metrics in tqdm(results, total=len(work_items), desc="Analyzing"):
                if metrics is not None:
                    all_rows.append(metrics)
                    success += 1
                    if not metrics.get('light_stable', True):
                        light_unstable += 1
                    if args.checkpoint_every > 0 and success % args.checkpoint_every == 0:
                        write_csv(all_rows, csv_path)
                else:
                    failed += 1

    elapsed = time.time() - t_start

    # Final CSV
    write_csv(all_rows, csv_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Processed:      {success}")
    logger.info(f"  Skipped:        {skipped}")
    logger.info(f"  Failed:         {failed}")
    logger.info(f"  Light unstable: {light_unstable} (still processed, tagged in output)")
    logger.info(f"  JSON dir:       {json_dir}")
    logger.info(f"  Summary CSV:    {csv_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
