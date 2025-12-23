#!/usr/bin/env python3
"""
Utility script to read and display session summary data from cavity annotation sessions.
"""

import json
from pathlib import Path
import argparse


def display_session_summary(summary_path):
    """Display session summary in a human-readable format."""
    summary_path = Path(summary_path)
    
    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}")
        return
    
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("CAVITY ANNOTATION SESSION SUMMARY")
    print("=" * 80)
    print(f"\nVideo: {data['video_name']}")
    print(f"Path:  {data['video_path']}")
    print(f"\nFrame Range: {data['frame_range']['start']} - {data['frame_range']['end']}")
    print(f"Total frames in range: {data['total_frames_in_range']}")
    print(f"Modified frames: {data['modified_frames_count']}")
    
    if data['modified_frames_count'] > 0:
        print(f"\nModified frame numbers: {', '.join(map(str, data['modified_frames']))}")
        
        print(f"\n{'='*80}")
        print("FRAME DETAILS")
        print(f"{'='*80}")
        
        for frame_info in data['frames']:
            print(f"\nFrame {frame_info['frame_number']}:")
            print(f"  Has mask: {frame_info['has_mask']}")
            print(f"  Positive points: {len(frame_info['cavity_points_pos'])} points")
            if frame_info['cavity_points_pos']:
                for i, (x, y) in enumerate(frame_info['cavity_points_pos'], 1):
                    print(f"    {i}. ({x}, {y})")
            print(f"  Negative points: {len(frame_info['cavity_points_neg'])} points")
            if frame_info['cavity_points_neg']:
                for i, (x, y) in enumerate(frame_info['cavity_points_neg'], 1):
                    print(f"    {i}. ({x}, {y})")


def get_frames_with_masks(summary_path):
    """Return list of frame numbers that have masks."""
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    return [f['frame_number'] for f in data['frames'] if f['has_mask']]


def get_frames_with_points(summary_path):
    """Return list of frame numbers that have annotation points."""
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    frames_with_points = []
    for f in data['frames']:
        if f['cavity_points_pos'] or f['cavity_points_neg']:
            frames_with_points.append(f['frame_number'])
    
    return frames_with_points


def export_frame_list(summary_path, output_path):
    """Export list of modified frames to a text file."""
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    with open(output_path, 'w') as f:
        for frame_num in data['modified_frames']:
            f.write(f"{frame_num}\n")
    
    print(f"✓ Exported {len(data['modified_frames'])} frame numbers to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Read and display cavity annotation session summaries"
    )
    parser.add_argument(
        "summary_file",
        type=str,
        help="Path to session_summary.json file"
    )
    parser.add_argument(
        "--export-frames",
        type=str,
        help="Export modified frame numbers to a text file"
    )
    parser.add_argument(
        "--list-masks-only",
        action="store_true",
        help="Only list frames that have masks"
    )
    parser.add_argument(
        "--list-points-only",
        action="store_true",
        help="Only list frames that have annotation points"
    )
    
    args = parser.parse_args()
    
    if args.export_frames:
        export_frame_list(args.summary_file, args.export_frames)
    elif args.list_masks_only:
        frames = get_frames_with_masks(args.summary_file)
        print(f"Frames with masks ({len(frames)}): {', '.join(map(str, frames))}")
    elif args.list_points_only:
        frames = get_frames_with_points(args.summary_file)
        print(f"Frames with points ({len(frames)}): {', '.join(map(str, frames))}")
    else:
        display_session_summary(args.summary_file)


if __name__ == "__main__":
    main()
