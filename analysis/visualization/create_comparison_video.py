#!/usr/bin/env python3
"""
Create side-by-side comparison video of Original, Normalized R, and Normalized R/G heatmaps.

Output: 3-panel video showing fluorescence metrics in sync.
"""

from __future__ import annotations

import cv2
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

_log = logging.getLogger(__name__)

from lib.utils import crop_ui


def create_heatmap(frame_bgr: np.ndarray, cavity_mask: np.ndarray, metric: str = 'r') -> np.ndarray:
    """
    Create heatmap visualization for a metric.
    
    Args:
        frame_bgr: BGR image
        cavity_mask: Boolean mask of cavity region
        metric: 'r' for normalized red, 'rg' for R/G ratio
    
    Returns:
        BGR visualization with heatmap overlay
    """
    B = frame_bgr[:, :, 0].astype(np.float32) / 255.0
    G = frame_bgr[:, :, 1].astype(np.float32) / 255.0
    R = frame_bgr[:, :, 2].astype(np.float32) / 255.0
    
    if metric == 'r':
        values = R
    else:  # rg
        values = R / (G + 0.01)
    
    # Normalize to 99th percentile within cavity
    if np.any(cavity_mask):
        max_val = np.percentile(values[cavity_mask], 99)
    else:
        max_val = 1.0
    
    normalized = np.clip(values / max_val, 0, 1)
    
    # Apply colormap
    color_map = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend with grayscale background
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    result = gray_bgr.copy()
    alpha = 0.7
    for c in range(3):
        result[:, :, c][cavity_mask] = (
            (1 - alpha) * gray_bgr[:, :, c][cavity_mask] + 
            alpha * color_map[:, :, c][cavity_mask]
        ).astype(np.uint8)
    
    return result


def create_comparison_video(
    video_path: str | Path,
    cavity_dir: str | Path,
    output_path: str | Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> None:
    """
    Create 3-panel comparison video.
    
    Args:
        video_path: Path to input video
        cavity_dir: Directory containing cavity masks (frame_NNNNNN/cavity_mask.png)
        output_path: Output video path
        start_frame: First frame to process (None = auto-detect from masks)
        end_frame: Last frame to process (None = auto-detect from masks)
    """
    video_path = Path(video_path)
    cavity_dir = Path(cavity_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find available frames with masks
    mask_dirs = sorted([d for d in cavity_dir.glob("frame_*") if d.is_dir()])
    if not mask_dirs:
        _log.warning("No cavity masks found!")
        return
    
    frame_indices = [int(d.name.split('_')[1]) for d in mask_dirs]
    
    if start_frame is None:
        start_frame = min(frame_indices)
    if end_frame is None:
        end_frame = max(frame_indices)
    
    # Filter to requested range
    frame_indices = [f for f in frame_indices if start_frame <= f <= end_frame]
    
    _log.info("Creating comparison video for frames %d-%d", start_frame, end_frame)
    _log.info("Found %d frames with masks", len(frame_indices))
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get frame size from first mask
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
    ret, frame = cap.read()
    frame = crop_ui(frame)
    h, w = frame.shape[:2]
    
    # Output: 3 panels side by side
    out_w = w * 3
    out_h = h
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    
    _log.info("Output: %dx%d @ %.1f fps", out_w, out_h, fps)
    
    for frame_idx in tqdm(frame_indices, desc="Creating video"):
        # Load frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = crop_ui(frame)
        
        # Load cavity mask
        mask_path = cavity_dir / f"frame_{frame_idx:06d}" / "cavity_mask.png"
        if not mask_path.exists():
            continue
        cavity_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if cavity_raw is None:
            continue
        cavity_mask = cavity_raw > 127
        
        # Create panels
        original = frame.copy()
        r_heatmap = create_heatmap(frame, cavity_mask, 'r')
        rg_heatmap = create_heatmap(frame, cavity_mask, 'rg')
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original, f'Frame {frame_idx} - Original', (10, 40), font, 1.2, (255, 255, 255), 2)
        cv2.putText(r_heatmap, 'Normalized R', (10, 40), font, 1.2, (255, 255, 255), 2)
        cv2.putText(rg_heatmap, 'Normalized R/G', (10, 40), font, 1.2, (255, 255, 255), 2)
        
        # Stack horizontally
        combined = np.hstack([original, r_heatmap, rg_heatmap])
        out.write(combined)
    
    cap.release()
    out.release()
    
    _log.info("Saved: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Create comparison video of fluorescence metrics")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video")
    parser.add_argument("--cavity-dir", type=str, required=True,
                       help="Directory containing cavity masks")
    parser.add_argument("--output", type=str, required=True,
                       help="Output video path")
    parser.add_argument("--start", type=int, default=None,
                       help="Start frame (default: auto-detect)")
    parser.add_argument("--end", type=int, default=None,
                       help="End frame (default: auto-detect)")
    
    args = parser.parse_args()
    
    create_comparison_video(
        video_path=args.video,
        cavity_dir=args.cavity_dir,
        output_path=args.output,
        start_frame=args.start,
        end_frame=args.end
    )


if __name__ == "__main__":
    main()



