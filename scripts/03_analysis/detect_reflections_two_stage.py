#!/usr/bin/env python3
"""
Two-stage reflection detection:
1. High-confidence seeds: V > 0.65 & R < 0.40
2. Local expansion: V > 0.60 within 15px of seeds

This approach maximizes precision while maintaining good recall.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# Add utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
from common import crop_ui, find_video_file


def detect_reflections_two_stage(image_bgr, cavity_mask=None,
                                  seed_v=0.65, seed_r=0.40,
                                  expand_v=0.60, expand_radius=15):
    """
    Two-stage reflection detection.
    
    Stage 1: High-confidence seeds (V > seed_v & R < seed_r)
    Stage 2: Expand locally (V > expand_v within expand_radius of seeds)
    
    Returns:
        reflection_mask: Binary mask of detected reflections
        debug_info: Dict with intermediate masks
    """
    # Extract channels
    b, g, r = cv2.split(image_bgr.astype(np.float32) / 255.0)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2].astype(np.float32) / 255.0
    
    # Cavity constraint
    if cavity_mask is not None:
        cavity_bool = cavity_mask > 127
    else:
        cavity_bool = np.ones(v_ch.shape, dtype=bool)
    
    # Stage 1: High confidence seeds
    seeds = (v_ch > seed_v) & (r < seed_r) & cavity_bool
    seeds_uint8 = (seeds * 255).astype(np.uint8)
    
    # Stage 2: Relaxed threshold for expansion
    relaxed = (v_ch > expand_v) & (r < seed_r) & cavity_bool
    
    # Dilate seeds to create expansion zone
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (expand_radius * 2 + 1, expand_radius * 2 + 1)
    )
    expansion_zone = cv2.dilate(seeds_uint8, kernel) > 0
    
    # Final mask: seeds + (relaxed AND in expansion zone)
    final_mask = seeds | (relaxed & expansion_zone)
    final_mask = final_mask & cavity_bool
    
    debug_info = {
        'seeds': seeds,
        'relaxed': relaxed,
        'expansion_zone': expansion_zone,
        'v_channel': v_ch,
        'r_channel': r,
        'seed_count': np.sum(seeds),
        'final_count': np.sum(final_mask),
    }
    
    return final_mask, debug_info


def process_frames(video_path, cavity_dir, output_dir, frame_range=None,
                   seed_v=0.65, seed_r=0.40, expand_v=0.60, expand_radius=15):
    """Process frames with two-stage reflection detection."""
    video_path = Path(video_path)
    cavity_dir = Path(cavity_dir)
    output_dir = Path(output_dir)
    
    mask_dir = output_dir / "reflection_masks"
    clean_dir = output_dir / "cleaned_cavity_masks"
    viz_dir = output_dir / "visualizations"
    
    for d in [mask_dir, clean_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    print(f"Video: {video_path.name}")
    print(f"Two-Stage Detection:")
    print(f"  Seeds: V > {seed_v} & R < {seed_r}")
    print(f"  Expand: V > {expand_v} within {expand_radius}px radius")
    print()
    
    frame_dirs = sorted([d for d in cavity_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('frame_')])
    
    if frame_range:
        start, end = frame_range
        frame_dirs = [d for d in frame_dirs 
                     if start <= int(d.name.split('_')[1]) <= end]
    
    print(f"Processing {len(frame_dirs)} frames...\n")
    
    results = []
    
    for i, frame_dir in enumerate(frame_dirs):
        frame_name = frame_dir.name
        frame_idx = int(frame_name.split('_')[1])
        
        # Read from video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        original_bgr = crop_ui(frame)
        
        # Load cavity mask
        cavity_path = frame_dir / "cavity_mask.png"
        if not cavity_path.exists():
            continue
        
        cavity_mask = cv2.imread(str(cavity_path), cv2.IMREAD_GRAYSCALE)
        if cavity_mask is None or np.sum(cavity_mask) == 0:
            continue
        
        # Detect reflections
        reflection_mask, debug_info = detect_reflections_two_stage(
            original_bgr, cavity_mask,
            seed_v=seed_v, seed_r=seed_r,
            expand_v=expand_v, expand_radius=expand_radius
        )
        
        # Stats
        refl_px = np.sum(reflection_mask)
        cavity_px = np.sum(cavity_mask > 127)
        pct = 100 * refl_px / cavity_px if cavity_px > 0 else 0
        
        results.append({
            'frame_idx': frame_idx,
            'seed_px': debug_info['seed_count'],
            'refl_px': refl_px,
            'pct': pct,
        })
        
        # Save reflection mask
        cv2.imwrite(str(mask_dir / f"{frame_name}_reflections.png"),
                   (reflection_mask * 255).astype(np.uint8))
        
        # Save cleaned cavity mask
        cleaned = cavity_mask.copy()
        cleaned[reflection_mask] = 0
        cv2.imwrite(str(clean_dir / f"{frame_name}_cavity_clean.png"), cleaned)
        
        # Create visualization
        viz = original_bgr.copy()
        viz[debug_info['seeds']] = [0, 255, 0]  # Green = seeds
        viz[reflection_mask & ~debug_info['seeds']] = [0, 255, 255]  # Yellow = expanded
        contours, _ = cv2.findContours(cavity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz, contours, -1, (0, 255, 0), 1)
        cv2.putText(viz, f"{frame_name}: {debug_info['seed_count']}+{refl_px-debug_info['seed_count']}={refl_px}px ({pct:.1f}%)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imwrite(str(viz_dir / f"{frame_name}_reflection_viz.png"), viz)
        
        print(f"  [{i+1}/{len(frame_dirs)}] {frame_name}: "
              f"seeds={debug_info['seed_count']}, final={refl_px} ({pct:.1f}%)")
    
    cap.release()
    
    if results:
        avg_pct = np.mean([r['pct'] for r in results])
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Average reflections: {avg_pct:.1f}% of cavity")
        print(f"  Output: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Two-stage reflection detection")
    parser.add_argument("--version", type=str, default="two_stage_v1")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--seed-v", type=float, default=0.65)
    parser.add_argument("--seed-r", type=float, default=0.40)
    parser.add_argument("--expand-v", type=float, default=0.60)
    parser.add_argument("--expand-radius", type=int, default=15)
    parser.add_argument("--video-folder", type=str, required=True,
                       help="Video folder under workspace (e.g., data/my-surgery-video)")
    
    args = parser.parse_args()
    
    # Derive paths from video folder
    from pathlib import Path
    video_folder = Path(args.video_folder)
    
    # Find video file
    video_path = find_video_file(video_folder)
    if video_path is None:
        raise ValueError(f"No video file found in {video_folder}")
    video_path = str(video_path)
    cavity_dir = str(video_folder / "pipeline" / "02_cavity" / "cavity_only" / "frames")
    output_dir = str(video_folder / "pipeline" / "02_cavity" / f"reflection_{args.version}")
    
    frame_range = (args.start, args.end) if args.start and args.end else None
    
    process_frames(
        video_path=video_path,
        cavity_dir=cavity_dir,
        output_dir=output_dir,
        frame_range=frame_range,
        seed_v=args.seed_v,
        seed_r=args.seed_r,
        expand_v=args.expand_v,
        expand_radius=args.expand_radius
    )


if __name__ == "__main__":
    main()

