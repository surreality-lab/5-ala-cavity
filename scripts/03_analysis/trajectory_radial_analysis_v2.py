#!/usr/bin/env python3
"""
Trajectory-based Radial Analysis v2
- Polar plot rotated to match frame orientation
- Both R and R/G intensity decay heatmaps
- Both R and R/G decay profiles
- Excludes reflections AND blood
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import sys

# Add utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
from common import crop_ui, find_video_file


def find_single_max(values, valid_mask):
    masked_values = values.copy()
    masked_values[~valid_mask] = -np.inf
    max_idx = np.argmax(masked_values)
    max_y, max_x = np.unravel_index(max_idx, values.shape)
    return [(max_y, max_x)]


def cast_ray(start_y, start_x, angle_rad, cavity_mask, exclude_mask=None, max_length=500):
    h, w = cavity_mask.shape
    dy, dx = np.sin(angle_rad), np.cos(angle_rad)
    trajectory = []
    for step in range(1, max_length):
        y, x = int(round(start_y + step * dy)), int(round(start_x + step * dx))
        if y < 0 or y >= h or x < 0 or x >= w or not cavity_mask[y, x]:
            break
        if exclude_mask is not None and exclude_mask[y, x]:
            continue
        if len(trajectory) == 0 or (y, x) != trajectory[-1]:
            trajectory.append((y, x))
    return trajectory


def compute_trajectory_profiles(frame_bgr, cavity_mask, peaks, exclude_mask=None, n_directions=360):
    B = frame_bgr[:, :, 0].astype(np.float32) / 255.0
    G = frame_bgr[:, :, 1].astype(np.float32) / 255.0
    R = frame_bgr[:, :, 2].astype(np.float32) / 255.0
    
    rg_ratio = R / (G + 0.01)
    
    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    
    all_trajectories = []
    
    for peak_idx, (peak_y, peak_x) in enumerate(peaks):
        peak_r = R[peak_y, peak_x]
        peak_rg = rg_ratio[peak_y, peak_x]
        
        peak_data = {
            'peak_idx': peak_idx,
            'peak_y': int(peak_y),
            'peak_x': int(peak_x),
            'peak_r': float(peak_r),
            'peak_rg': float(peak_rg),
            'trajectories': []
        }
        
        for dir_idx, angle in enumerate(angles):
            trajectory_coords = cast_ray(peak_y, peak_x, angle, cavity_mask, exclude_mask)
            
            if len(trajectory_coords) < 3:
                continue
            
            r_values = [R[y, x] for y, x in trajectory_coords]
            rg_values = [rg_ratio[y, x] for y, x in trajectory_coords]
            
            distances = [np.sqrt((y - peak_y)**2 + (x - peak_x)**2) 
                        for y, x in trajectory_coords]
            
            traj_data = {
                'direction_idx': dir_idx,
                'angle_deg': float(np.degrees(angle)),
                'length': len(trajectory_coords),
                'distances': distances,
                'r_values': r_values,
                'rg_values': rg_values,
                'r_normalized': [v / peak_r if peak_r > 0 else v for v in r_values],
                'rg_normalized': [v / peak_rg if peak_rg > 0 else v for v in rg_values],
                'coords': trajectory_coords,
            }
            
            peak_data['trajectories'].append(traj_data)
        
        all_trajectories.append(peak_data)
    
    return all_trajectories


def visualize_trajectories_v2(frame_bgr, cavity_mask, exclude_mask, peaks, trajectories, 
                               frame_idx, output_path, reflection_px=0, blood_px=0):
    """Create 3x2 visualization with both R and R/G metrics."""
    
    fig = plt.figure(figsize=(16, 20))
    
    # 1. Original with rays (top-left)
    ax1 = fig.add_subplot(3, 2, 1)
    display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
    
    if exclude_mask is not None and np.any(exclude_mask):
        display[exclude_mask] = [255, 100, 100]
    
    ax1.imshow(display)
    ax1.contour(cavity_mask, colors='cyan', linewidths=1)
    
    for peak_data in trajectories:
        py, px = peak_data['peak_y'], peak_data['peak_x']
        for traj in peak_data['trajectories'][::10]:
            coords = traj['coords']
            if len(coords) > 0:
                xs = [px] + [c[1] for c in coords]
                ys = [py] + [c[0] for c in coords]
                color = plt.cm.hsv(traj['angle_deg'] / 360)
                ax1.plot(xs, ys, '-', color=color, alpha=0.5, linewidth=0.8)
    
    ax1.set_title(f'Frame {frame_idx} - Excluded: {reflection_px+blood_px:,}px\n(Reflections: {reflection_px:,}, Blood: {blood_px:,})')
    ax1.axis('off')
    
    # 2. Polar plot - rotated to match frame (top-right)
    ax2 = fig.add_subplot(3, 2, 2, projection='polar')
    
    for peak_data in trajectories:
        angles = []
        lengths = []
        for traj in peak_data['trajectories']:
            # Image coords: 0° = right, 90° = down, 180° = left, 270° = up
            # With theta_direction=-1 (clockwise) and theta_zero='E':
            # Polar: theta=0 → right, theta=90 (clockwise) → down
            # So just use the angle directly (no negation needed)
            angles.append(np.radians(traj['angle_deg']))
            lengths.append(traj['length'])
        
        ax2.plot(angles, lengths, 'b-', linewidth=0.5, alpha=0.7)
        ax2.fill(angles, lengths, alpha=0.3)
    
    ax2.set_theta_zero_location('E')  # 0° at right (East)
    ax2.set_theta_direction(-1)  # Clockwise to match image coords (down = 90°)
    ax2.set_title('Trajectory Length by Direction\n(rotated to match frame)')
    
    # 3. R intensity heatmap (middle-left)
    ax3 = fig.add_subplot(3, 2, 3)
    max_dist, n_dist_bins = 150, 75
    
    for peak_data in trajectories:
        n_angles = len(peak_data['trajectories'])
        heatmap = np.full((n_angles, n_dist_bins), np.nan)
        
        for i, traj in enumerate(peak_data['trajectories']):
            for d, v in zip(traj['distances'], traj['r_normalized']):
                bin_idx = int(d / max_dist * n_dist_bins)
                if 0 <= bin_idx < n_dist_bins:
                    heatmap[i, bin_idx] = v
        
        im = ax3.imshow(heatmap, aspect='auto', cmap='hot', 
                       extent=[0, max_dist, 360, 0], vmin=0, vmax=1.2)
        plt.colorbar(im, ax=ax3, label='Normalized R')
    
    ax3.set_xlabel('Distance from peak (px)')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title('RED Channel Decay: Angle vs Distance')
    
    # 4. R/G intensity heatmap (middle-right)
    ax4 = fig.add_subplot(3, 2, 4)
    
    for peak_data in trajectories:
        n_angles = len(peak_data['trajectories'])
        heatmap = np.full((n_angles, n_dist_bins), np.nan)
        
        for i, traj in enumerate(peak_data['trajectories']):
            for d, v in zip(traj['distances'], traj['rg_normalized']):
                bin_idx = int(d / max_dist * n_dist_bins)
                if 0 <= bin_idx < n_dist_bins:
                    heatmap[i, bin_idx] = v
        
        im = ax4.imshow(heatmap, aspect='auto', cmap='hot', 
                       extent=[0, max_dist, 360, 0], vmin=0, vmax=1.2)
        plt.colorbar(im, ax=ax4, label='Normalized R/G')
    
    ax4.set_xlabel('Distance from peak (px)')
    ax4.set_ylabel('Angle (degrees)')
    ax4.set_title('R/G Ratio Decay: Angle vs Distance')
    
    # 5. R decay profile (bottom-left)
    ax5 = fig.add_subplot(3, 2, 5)
    common_distances = np.arange(0, max_dist, 2)
    all_r_norm = []
    
    for peak_data in trajectories:
        for traj in peak_data['trajectories']:
            if len(traj['distances']) > 5:
                interp = np.interp(common_distances, traj['distances'], 
                                  traj['r_normalized'], right=np.nan)
                all_r_norm.append(interp)
    
    if len(all_r_norm) > 0:
        all_r_norm = np.array(all_r_norm)
        p10 = np.nanpercentile(all_r_norm, 10, axis=0)
        p25 = np.nanpercentile(all_r_norm, 25, axis=0)
        p50 = np.nanpercentile(all_r_norm, 50, axis=0)
        p75 = np.nanpercentile(all_r_norm, 75, axis=0)
        p90 = np.nanpercentile(all_r_norm, 90, axis=0)
        
        ax5.fill_between(common_distances, p10, p90, alpha=0.2, color='red', label='10-90%')
        ax5.fill_between(common_distances, p25, p75, alpha=0.3, color='red', label='25-75%')
        ax5.plot(common_distances, p50, 'r-', linewidth=2, label='Median')
    
    ax5.set_xlabel('Distance from peak (px)')
    ax5.set_ylabel('Normalized R')
    ax5.set_title('RED Channel Decay Profile')
    ax5.set_xlim(0, max_dist)
    ax5.set_ylim(0, 1.5)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax5.legend(loc='upper right')
    
    # 6. R/G decay profile (bottom-right)
    ax6 = fig.add_subplot(3, 2, 6)
    all_rg_norm = []
    
    for peak_data in trajectories:
        for traj in peak_data['trajectories']:
            if len(traj['distances']) > 5:
                interp = np.interp(common_distances, traj['distances'], 
                                  traj['rg_normalized'], right=np.nan)
                all_rg_norm.append(interp)
    
    if len(all_rg_norm) > 0:
        all_rg_norm = np.array(all_rg_norm)
        p10 = np.nanpercentile(all_rg_norm, 10, axis=0)
        p25 = np.nanpercentile(all_rg_norm, 25, axis=0)
        p50 = np.nanpercentile(all_rg_norm, 50, axis=0)
        p75 = np.nanpercentile(all_rg_norm, 75, axis=0)
        p90 = np.nanpercentile(all_rg_norm, 90, axis=0)
        
        ax6.fill_between(common_distances, p10, p90, alpha=0.2, color='blue', label='10-90%')
        ax6.fill_between(common_distances, p25, p75, alpha=0.3, color='blue', label='25-75%')
        ax6.plot(common_distances, p50, 'b-', linewidth=2, label='Median')
    
    ax6.set_xlabel('Distance from peak (px)')
    ax6.set_ylabel('Normalized R/G')
    ax6.set_title('R/G Ratio Decay Profile')
    ax6.set_xlim(0, max_dist)
    ax6.set_ylim(0, 1.5)
    ax6.grid(True, alpha=0.3)
    ax6.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax6.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_frame(frame_bgr, cavity_mask, exclude_mask, frame_idx):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    
    if exclude_mask is not None:
        analysis_mask = cavity_mask & ~exclude_mask
    else:
        analysis_mask = cavity_mask
    
    if np.sum(analysis_mask) > 0:
        cavity_V = V[analysis_mask]
        thresh_75 = np.percentile(cavity_V, 75)
        valid_mask = analysis_mask & (V >= thresh_75)
    else:
        valid_mask = analysis_mask
    
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)
    fluorescence = R / (G + 1.0)
    
    peaks = find_single_max(fluorescence, valid_mask)
    
    if len(peaks) == 0:
        return None, peaks
    
    trajectories = compute_trajectory_profiles(frame_bgr, cavity_mask, peaks, exclude_mask, n_directions=360)
    
    return trajectories, peaks


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Trajectory radial analysis v2")
    parser.add_argument("--video-folder", type=str, required=True,
                       help="Video folder under workspace (e.g., data/my-surgery-video)")
    parser.add_argument("--reflection-version", type=str, default="two_stage_v65",
                       help="Reflection detection version (e.g., two_stage_v65)")
    parser.add_argument("--blood-version", type=str, default="v20_b20",
                       help="Blood detection version (e.g., v20_b20)")
    parser.add_argument("--frames", type=str, default="50,54,100,150,200",
                       help="Comma-separated frame indices")
    
    args = parser.parse_args()
    
    # Derive paths from video folder
    video_folder = Path(args.video_folder)
    
    # Find video file
    video_path = find_video_file(video_folder)
    if video_path is None:
        raise ValueError(f"No video file found in {video_folder}")
    video_path = str(video_path)
    cavity_dir = video_folder / "pipeline" / "02_cavity" / "cavity_only" / "frames"
    reflection_dir = video_folder / "pipeline" / "02_cavity" / f"reflection_{args.reflection_version}" / "reflection_masks"
    blood_dir = video_folder / "pipeline" / "02_cavity" / f"blood_{args.blood_version}" / "blood_masks"
    output_dir = video_folder / "pipeline" / "03_analysis" / "trajectory_analysis_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_indices = [int(x.strip()) for x in args.frames.split(',') if x.strip()]
    
    cap = cv2.VideoCapture(video_path)
    
    for frame_idx in tqdm(frame_indices, desc="Analyzing"):
        cavity_path = cavity_dir / f"frame_{frame_idx:06d}" / "cavity_mask.png"
        if not cavity_path.exists():
            continue
        
        cavity_mask = cv2.imread(str(cavity_path), cv2.IMREAD_GRAYSCALE) > 127
        
        # Load reflection mask
        reflection_path = reflection_dir / f"frame_{frame_idx:06d}_reflections.png"
        reflection_mask = cv2.imread(str(reflection_path), cv2.IMREAD_GRAYSCALE) > 127 if reflection_path.exists() else None
        
        # Load blood mask
        blood_path = blood_dir / f"frame_{frame_idx:06d}_blood.png"
        blood_mask = cv2.imread(str(blood_path), cv2.IMREAD_GRAYSCALE) > 127 if blood_path.exists() else None
        
        # Combine exclusion masks
        exclude_mask = np.zeros_like(cavity_mask)
        if reflection_mask is not None:
            exclude_mask = exclude_mask | reflection_mask
        if blood_mask is not None:
            exclude_mask = exclude_mask | blood_mask
        
        reflection_px = np.sum(reflection_mask) if reflection_mask is not None else 0
        blood_px = np.sum(blood_mask) if blood_mask is not None else 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = crop_ui(frame)
        
        trajectories, peaks = analyze_frame(frame, cavity_mask, exclude_mask, frame_idx)
        
        if trajectories is not None:
            visualize_trajectories_v2(
                frame, cavity_mask, exclude_mask, peaks, trajectories, frame_idx,
                output_dir / f"trajectories_{frame_idx:06d}.png",
                reflection_px=reflection_px, blood_px=blood_px
            )
    
    cap.release()
    print(f"\n✓ Done! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

