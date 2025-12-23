#!/usr/bin/env python3
"""
Simple blood/dark tissue detection: V < 0.20 & B < 0.20
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# Add utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
from common import crop_ui, find_video_file


def detect_blood(image_bgr, cavity_mask=None, v_thresh=0.20, b_thresh=0.20):
    """Detect blood/dark tissue: V < v_thresh & B < b_thresh"""
    b, g, r = cv2.split(image_bgr.astype(np.float32) / 255.0)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2].astype(np.float32) / 255.0
    
    if cavity_mask is not None:
        cavity_bool = cavity_mask > 127
    else:
        cavity_bool = np.ones(v_ch.shape, dtype=bool)
    
    blood_mask = (v_ch < v_thresh) & (b < b_thresh) & cavity_bool
    
    return blood_mask


def process_frames(video_path, cavity_dir, output_dir, frame_range=None,
                   v_thresh=0.20, b_thresh=0.20):
    """Process frames with blood detection."""
    video_path = Path(video_path)
    cavity_dir = Path(cavity_dir)
    output_dir = Path(output_dir)
    
    mask_dir = output_dir / "blood_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    print(f"Video: {video_path.name}")
    print(f"Blood detection: V < {v_thresh} & B < {b_thresh}")
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
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        original_bgr = crop_ui(frame)
        
        cavity_path = frame_dir / "cavity_mask.png"
        if not cavity_path.exists():
            continue
        
        cavity_mask = cv2.imread(str(cavity_path), cv2.IMREAD_GRAYSCALE)
        if cavity_mask is None or np.sum(cavity_mask) == 0:
            continue
        
        blood_mask = detect_blood(original_bgr, cavity_mask, v_thresh, b_thresh)
        
        blood_px = np.sum(blood_mask)
        cavity_px = np.sum(cavity_mask > 127)
        pct = 100 * blood_px / cavity_px if cavity_px > 0 else 0
        
        results.append({
            'frame_idx': frame_idx,
            'blood_px': blood_px,
            'pct': pct,
        })
        
        cv2.imwrite(str(mask_dir / f"{frame_name}_blood.png"),
                   (blood_mask * 255).astype(np.uint8))
        
        if (i + 1) % 50 == 0 or i == len(frame_dirs) - 1:
            print(f"  [{i+1}/{len(frame_dirs)}] {frame_name}: {blood_px:,}px ({pct:.1f}%)")
    
    cap.release()
    
    if results:
        avg_pct = np.mean([r['pct'] for r in results])
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Average blood/dark: {avg_pct:.1f}% of cavity")
        print(f"  Output: {mask_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Blood/dark tissue detection")
    parser.add_argument("--video-folder", type=str, required=True,
                       help="Video folder under workspace (e.g., data/my-surgery-video)")
    parser.add_argument("--v-thresh", type=float, default=0.20)
    parser.add_argument("--b-thresh", type=float, default=0.20)
    
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
    output_dir = str(video_folder / "pipeline" / "02_cavity" / f"blood_v{int(args.v_thresh*100)}_b{int(args.b_thresh*100)}")
    
    process_frames(
        video_path=video_path,
        cavity_dir=cavity_dir,
        output_dir=output_dir,
        v_thresh=args.v_thresh,
        b_thresh=args.b_thresh
    )


if __name__ == "__main__":
    main()

