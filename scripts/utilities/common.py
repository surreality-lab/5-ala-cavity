#!/usr/bin/env python3
"""
Common utility functions shared across the 5-ALA annotation pipeline.
"""

import cv2
from pathlib import Path


def crop_ui(img, left_percent=20, right_percent=5):
    """
    Crop UI elements from surgical video frames.
    
    Args:
        img: Input image (BGR or grayscale)
        left_percent: Percentage to crop from left side (default: 20%)
        right_percent: Percentage to crop from right side (default: 5%)
    
    Returns:
        Cropped image
    """
    height, width = img.shape[:2]
    left_crop = int(width * left_percent / 100)
    right_crop = int(width * (100 - right_percent) / 100)
    return img[:, left_crop:right_crop]


def find_video_file(folder):
    """
    Find the first video file in a folder.
    
    Args:
        folder: Path to folder (string or Path object)
    
    Returns:
        Path to video file, or None if not found
    """
    folder = Path(folder)
    for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
        videos = list(folder.glob(ext))
        if videos:
            return videos[0]
    return None
