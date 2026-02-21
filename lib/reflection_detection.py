#!/usr/bin/env python3
"""
Two-stage reflection detection:
1. High-confidence seeds: V > 0.65 & R < 0.40
2. Local expansion: V > 0.60 within 15px of seeds

This approach maximizes precision while maintaining good recall.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_reflections_two_stage(image_bgr: np.ndarray, cavity_mask: np.ndarray | None = None, seed_v: float = 0.65, seed_r: float = 0.40, expand_v: float = 0.60, expand_radius: int = 15) -> tuple[np.ndarray, dict]:
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

