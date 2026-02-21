#!/usr/bin/env python3
"""
Batch Exclusion Mask Generation for 5fps Data
===============================================
Runs reflection + blood detection on ALL cavity-masked frames from the 5fps video.
Saves per-frame masks and spot-check visualizations.

Usage:
    python scripts/03_analysis/run_5fps_exclusion_masks.py --no-skip
    python scripts/03_analysis/run_5fps_exclusion_masks.py --video data/2025-03-25_5fps/video.mp4 --masks-dir data/2025-03-25_5fps/masks --no-skip
    python scripts/03_analysis/run_5fps_exclusion_masks.py --include-csv data/2025-03-25_5fps/manual_frame_classifications.csv --no-skip
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
import csv
from tqdm import tqdm

import sys

from lib.reflection_detection import detect_reflections_two_stage
from lib.blood_detection import detect_blood, check_light_stability
from lib.utils import crop_ui, discover_mask_frames, parse_include_csv, setup_logging, SequentialVideoReader

# ── Default Paths (from centralized config, overridable via CLI) ───────────
from config import DEFAULT_VIDEO, DEFAULT_MASKS
_DEFAULT_VIDEO = DEFAULT_VIDEO
_DEFAULT_MASKS = DEFAULT_MASKS


def save_spot_check_viz(frame_bgr, cavity_mask, reflection_mask, blood_mask,
                        frame_idx, output_path, refl_px, blood_px, cavity_px):
    """Save a visualization showing what was excluded and what remains."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original frame with cavity outline
    ax = axes[0]
    display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
    ax.imshow(display)
    ax.contour(cavity_mask, colors='cyan', linewidths=1)
    ax.set_title(f'Frame {frame_idx} - Original')
    ax.axis('off')

    # 2. Exclusion overlay
    ax = axes[1]
    overlay = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
    # Reflections in yellow
    if reflection_mask is not None and np.any(reflection_mask):
        overlay[reflection_mask] = [255, 255, 0]
    # Blood in red
    if blood_mask is not None and np.any(blood_mask):
        overlay[blood_mask] = [255, 50, 50]
    ax.imshow(overlay)
    ax.contour(cavity_mask, colors='cyan', linewidths=1)
    refl_pct = 100 * refl_px / cavity_px if cavity_px > 0 else 0
    blood_pct = 100 * blood_px / cavity_px if cavity_px > 0 else 0
    ax.set_title(f'Exclusions\nReflection (yellow): {refl_px:,}px ({refl_pct:.1f}%)\n'
                 f'Blood (red): {blood_px:,}px ({blood_pct:.1f}%)')
    ax.axis('off')

    # 3. Clean tissue only (what radial analysis will use)
    ax = axes[2]
    clean = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
    exclude = np.zeros_like(cavity_mask, dtype=bool)
    if reflection_mask is not None:
        exclude |= reflection_mask
    if blood_mask is not None:
        exclude |= blood_mask
    # Dim excluded regions
    clean[exclude] = (clean[exclude] * 0.3).astype(np.uint8)
    # Dim outside cavity
    clean[~cavity_mask] = (clean[~cavity_mask] * 0.2).astype(np.uint8)
    ax.imshow(clean)
    remaining = int(np.sum(cavity_mask & ~exclude))
    remaining_pct = 100 * remaining / cavity_px if cavity_px > 0 else 0
    ax.set_title(f'Analyzable tissue\n{remaining:,}px ({remaining_pct:.1f}% of cavity)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Batch exclusion mask generation for 5fps data")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (default: auto-detect)")
    parser.add_argument("--masks-dir", type=str, default=None,
                        help="Path to masks/ directory containing cavity/ (blood/ and reflection/ created here)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process at most N frames (for testing)")
    parser.add_argument("--viz-every", type=int, default=100,
                        help="Save spot-check visualization every N frames (default: 100)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip frames that already have masks")
    parser.add_argument("--no-skip", dest="skip_existing", action="store_false",
                        help="Reprocess all frames")
    # Reflection params
    parser.add_argument("--seed-v", type=float, default=0.65)
    parser.add_argument("--seed-r", type=float, default=0.40)
    parser.add_argument("--expand-v", type=float, default=0.60)
    parser.add_argument("--expand-radius", type=int, default=15)
    # Blood params (Otsu-based)
    parser.add_argument("--otsu-scale", type=float, default=0.6,
                        help="Fraction of Otsu threshold to use as seed cutoff")
    parser.add_argument("--expand-blood-radius", type=int, default=15,
                        help="Dilation radius for blood expansion stage")
    parser.add_argument("--expand-blood-factor", type=float, default=1.6,
                        help="Relaxation multiplier on seed threshold for expansion")
    # Blood detection V-floor
    parser.add_argument("--v-abs-floor", type=float, default=0.12,
                        help="Absolute minimum V threshold. Prevents Otsu collapse "
                             "on uniformly dark frames. Set to 0 to disable.")
    # Light stability / early frame skip
    parser.add_argument("--min-median-v", type=float, default=0.08,
                        help="Min median V in cavity for frame to be considered stable")
    # Include list
    parser.add_argument("--include-csv", type=str, default=None,
                        help="Path to manual_frame_classifications.csv. "
                             "Only 'include' frames will get masks generated.")
    args = parser.parse_args()

    # Resolve paths (CLI overrides defaults)
    VIDEO_PATH = Path(args.video) if args.video else _DEFAULT_VIDEO
    masks_base = Path(args.masks_dir) if args.masks_dir else _DEFAULT_MASKS
    for name, val in [("--video", VIDEO_PATH), ("--masks-dir", masks_base)]:
        if val is None:
            print(f"ERROR: {name} is required (or set ALA_WORKSPACE)", file=sys.stderr)
            sys.exit(1)
    CAVITY_DIR = masks_base / "cavity"
    REFL_DIR   = masks_base / "reflection"
    BLOOD_DIR  = masks_base / "blood"
    VIZ_DIR    = masks_base / "exclusion_viz"
    SUMMARY_CSV = masks_base / "exclusion_summary.csv"

    logger = setup_logging('run_5fps_exclusion_masks', output_dir=masks_base)

    # Validate
    errors = []
    if not VIDEO_PATH.exists():
        errors.append(f"Video not found: {VIDEO_PATH}")
    if not CAVITY_DIR.exists():
        errors.append(f"Cavity masks not found: {CAVITY_DIR}")
    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)

    logger.info("Video:     %s", VIDEO_PATH)
    logger.info("Masks:     %s", masks_base)
    logger.info("Reflection params: seed_v=%s, seed_r=%s, expand_v=%s, expand_radius=%s",
                args.seed_v, args.seed_r, args.expand_v, args.expand_radius)
    logger.info("Blood params (Otsu): otsu_scale=%s, expand_radius=%s, expand_factor=%s",
                args.otsu_scale, args.expand_blood_radius, args.expand_blood_factor)
    logger.info("V-floor:       %s", args.v_abs_floor)
    logger.info("Light min V:   %s", args.min_median_v)

    # Discover frames
    logger.info("Discovering mask frames...")
    frame_indices = discover_mask_frames(CAVITY_DIR)
    logger.info("  Found %d frames with cavity masks", len(frame_indices))
    if not frame_indices:
        logger.error("No cavity masks found in %s — nothing to process.", CAVITY_DIR)
        sys.exit(1)
    logger.info("  Range: %d – %d", frame_indices[0], frame_indices[-1])

    # ── Include-list filtering ──────────────────────────────────────────
    if args.include_csv:
        include_csv_path = Path(args.include_csv)
        if not include_csv_path.exists():
            logger.error("Include CSV not found: %s", include_csv_path)
            sys.exit(1)
        include_set = parse_include_csv(include_csv_path)
        before = len(frame_indices)
        frame_indices = [fi for fi in frame_indices if fi in include_set]
        logger.info("  Include CSV:   %s", include_csv_path)
        logger.info("  Include-list:  %d frames", len(include_set))
        logger.info("  Filtered %d -> %d frames", before, len(frame_indices))

    if args.max_frames is not None:
        frame_indices = frame_indices[:args.max_frames]
        logger.info("  Limited to first %d frames", args.max_frames)

    # Setup output dirs
    REFL_DIR.mkdir(parents=True, exist_ok=True)
    BLOOD_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Open video
    reader = SequentialVideoReader(VIDEO_PATH)
    total_video_frames = reader.total_frames
    logger.info("  Video has %d frames", total_video_frames)

    # Process
    all_stats = []
    skipped = 0
    failed = 0
    unstable_frames = []
    processed = 0
    t0 = time.time()

    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Generating exclusion masks")):
        # Output paths
        refl_frame_dir = REFL_DIR / f"frame_{frame_idx:06d}"
        blood_frame_dir = BLOOD_DIR / f"frame_{frame_idx:06d}"
        refl_path = refl_frame_dir / "reflection_mask.png"
        blood_path = blood_frame_dir / "blood_mask.png"

        # Skip if already processed
        if args.skip_existing and refl_path.exists() and blood_path.exists():
            skipped += 1
            continue

        # Validate
        if frame_idx >= total_video_frames:
            logger.debug("Frame %d: skipped (index beyond video length %d)", frame_idx, total_video_frames)
            failed += 1
            continue

        # Load cavity mask
        cavity_mask_path = CAVITY_DIR / f"frame_{frame_idx:06d}" / "cavity_mask.png"
        cavity_mask_raw = cv2.imread(str(cavity_mask_path), cv2.IMREAD_GRAYSCALE)
        if cavity_mask_raw is None:
            logger.debug("Frame %d: skipped (cavity mask unreadable: %s)", frame_idx, cavity_mask_path)
            failed += 1
            continue

        # Read video frame
        ret, frame_bgr = reader.read(frame_idx)
        if not ret:
            logger.debug("Frame %d: skipped (video read failed)", frame_idx)
            failed += 1
            continue

        # Crop UI overlays
        frame_cropped = crop_ui(frame_bgr)

        # Handle dimension mismatch
        cavity_bool = cavity_mask_raw > 127
        if cavity_bool.shape != frame_cropped.shape[:2]:
            if cavity_bool.shape == frame_bgr.shape[:2]:
                frame_cropped = frame_bgr
            else:
                cavity_mask_raw = cv2.resize(
                    cavity_mask_raw,
                    (frame_cropped.shape[1], frame_cropped.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                cavity_bool = cavity_mask_raw > 127

        # ── Light stability check ──
        light_stable, median_v = check_light_stability(
            frame_cropped, cavity_mask_raw,
            min_median_v=args.min_median_v
        )
        if not light_stable:
            unstable_frames.append((frame_idx, median_v))

        # Detect reflections
        reflection_mask, refl_debug = detect_reflections_two_stage(
            frame_cropped, cavity_mask_raw,
            seed_v=args.seed_v, seed_r=args.seed_r,
            expand_v=args.expand_v, expand_radius=args.expand_radius
        )

        # Detect blood (Otsu-based, with V-floor)
        blood_mask, blood_debug = detect_blood(
            frame_cropped, cavity_mask_raw,
            otsu_scale=args.otsu_scale,
            expand_radius=args.expand_blood_radius,
            expand_factor=args.expand_blood_factor,
            v_abs_floor=args.v_abs_floor if args.v_abs_floor > 0 else None,
        )

        # Stats
        cavity_px = int(np.sum(cavity_bool))
        refl_px = int(np.sum(reflection_mask))
        blood_px = int(np.sum(blood_mask))

        all_stats.append({
            'frame_idx': frame_idx,
            'cavity_px': cavity_px,
            'reflection_px': refl_px,
            'blood_px': blood_px,
            'reflection_pct': round(100 * refl_px / cavity_px, 2) if cavity_px > 0 else 0,
            'blood_pct': round(100 * blood_px / cavity_px, 2) if cavity_px > 0 else 0,
            'excluded_total_px': refl_px + blood_px,
            'excluded_total_pct': round(100 * (refl_px + blood_px) / cavity_px, 2) if cavity_px > 0 else 0,
            'median_v': blood_debug['median_v'],
            'otsu_raw': blood_debug['otsu_raw'],
            'otsu_scaled': blood_debug['otsu_scaled'],
            'otsu_override': blood_debug.get('otsu_override', False),
            'v_abs_floor': blood_debug.get('v_abs_floor', None),
            'seed_px': blood_debug['seed_px'],
            'expanded_px': blood_debug['expanded_px'],
            'light_stable': light_stable,
        })

        # Save masks
        refl_frame_dir.mkdir(parents=True, exist_ok=True)
        blood_frame_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(refl_path), (reflection_mask * 255).astype(np.uint8))
        cv2.imwrite(str(blood_path), (blood_mask * 255).astype(np.uint8))

        # Spot-check visualization
        if i % args.viz_every == 0:
            viz_path = VIZ_DIR / f"exclusion_{frame_idx:06d}.png"
            save_spot_check_viz(
                frame_cropped, cavity_bool, reflection_mask, blood_mask,
                frame_idx, viz_path, refl_px, blood_px, cavity_px
            )

        processed += 1

    reader.release()
    elapsed = time.time() - t0

    # Write summary CSV
    if all_stats:
        fieldnames = list(all_stats[0].keys())
        with open(SUMMARY_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(all_stats, key=lambda r: r['frame_idx']):
                writer.writerow(row)

    # Report
    logger.info("=" * 60)
    logger.info("DONE in %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("  Processed: %d", processed)
    logger.info("  Skipped (cached): %d", skipped)
    logger.info("  Failed: %d", failed)
    if unstable_frames:
        logger.info("  Unstable light (%d frames, median_v < %s):",
                     len(unstable_frames), args.min_median_v)
        for fidx, mv in unstable_frames[:10]:
            logger.info("      frame %d: median_v = %.4f", fidx, mv)
        if len(unstable_frames) > 10:
            logger.info("      ... and %d more", len(unstable_frames) - 10)
    if all_stats:
        stable_stats = [s for s in all_stats if s['light_stable']]
        avg_refl = np.mean([s['reflection_pct'] for s in stable_stats]) if stable_stats else 0
        avg_blood = np.mean([s['blood_pct'] for s in stable_stats]) if stable_stats else 0
        avg_total = np.mean([s['excluded_total_pct'] for s in stable_stats]) if stable_stats else 0
        logger.info("  Stable frames: %d / %d", len(stable_stats), len(all_stats))
        logger.info("  Avg reflection (stable): %.1f%% of cavity", avg_refl)
        logger.info("  Avg blood (stable):      %.1f%% of cavity", avg_blood)
        logger.info("  Avg total excl (stable): %.1f%% of cavity", avg_total)
    logger.info("Outputs:")
    logger.info("  Reflection masks: %s", REFL_DIR)
    logger.info("  Blood masks:      %s", BLOOD_DIR)
    logger.info("  Visualizations:   %s", VIZ_DIR)
    logger.info("  Summary CSV:      %s", SUMMARY_CSV)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
