#!/usr/bin/env python3
"""
Blood / dark-tissue detection for blue-light fluorescence imaging.

Multi-channel approach (V + S + R/G):
    Under blue-light excitation (~405nm), V-only thresholding over-detects
    because the entire cavity is dark. Normal (non-fluorescent) tissue is
    dark but NOT blood. Saturation (S) is the strongest discriminator:
    blood is desaturated (S ~0.47-0.68), tissue retains saturation from
    blue-light interaction (S ~0.84-0.96).

    Seeds require ALL criteria:
        V < v_threshold  AND  S < s_threshold  AND  (optionally) R/G < rg_threshold
    Expansion relaxes all criteria within a dilation radius.

Also provides a helper to flag frames whose illumination is too dim
to trust (light-ramp detection).
"""

from __future__ import annotations

import cv2
import numpy as np


# ── Light-ramp / early-frame detection ────────────────────────────────────
def check_light_stability(image_bgr: np.ndarray, cavity_mask: np.ndarray | None = None, min_median_v: float = 0.08) -> tuple[bool, float]:
    """Return True if the frame appears adequately illuminated.

    Parameters
    ----------
    min_median_v : float
        Minimum median V (0-1 scale) inside the cavity for the frame to be
        considered stable.  Frames below this are likely still in the
        blue-light ramp-up phase and should be flagged / skipped.

    Returns
    -------
    stable : bool
    median_v : float   (for logging)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2].astype(np.float32) / 255.0

    if cavity_mask is not None:
        cavity_bool = cavity_mask > 127 if cavity_mask.dtype == np.uint8 else cavity_mask.astype(bool)
    else:
        cavity_bool = np.ones(v_ch.shape, dtype=bool)

    if np.sum(cavity_bool) == 0:
        return False, 0.0

    median_v = float(np.median(v_ch[cavity_bool]))
    return median_v >= min_median_v, median_v


# ── Blood detection (multi-channel, two-stage) ───────────────────────────
def detect_blood(image_bgr: np.ndarray, cavity_mask: np.ndarray | None = None,
                 otsu_scale: float = 0.6,
                 s_thresh: float | None = 0.75,
                 rg_thresh: float | None = None,
                 expand_radius: int = 15, expand_factor: float = 1.6,
                 v_abs_floor: float | None = 0.12) -> tuple[np.ndarray, dict]:
    """Detect blood using multi-channel HSV + R/G criteria.

    Under blue-light excitation, V-only thresholding over-detects because
    the entire cavity is dark.  Adding Saturation (S < s_thresh) as a
    conjunction prevents flagging normal dark tissue that retains high
    saturation from blue-light interaction.

    Seeds require ALL specified criteria simultaneously:
        V < (otsu_raw * otsu_scale)
        S < s_thresh
        R/G < rg_thresh  (if rg_thresh is not None)

    Expansion relaxes V by expand_factor within a dilation of seed regions.

    When the frame is uniformly dark (all blood), Otsu produces a very low
    threshold and under-detects. The v_abs_floor parameter sets a minimum
    effective V threshold to handle this case.

    Parameters
    ----------
    otsu_scale : float
        Fraction of Otsu threshold on V to use as seed V cutoff.
    s_thresh : float
        Maximum saturation for blood seeds (0-1). Blood is desaturated
        under blue light (~0.47-0.68), tissue has high S (~0.84-0.96).
        Set to None to disable S criterion (legacy V-only mode).
    rg_thresh : float or None
        Maximum R/G ratio for blood seeds. Blood has R/G < 1 under blue
        light because hemoglobin absorbs red preferentially. Set to None
        to disable (default).
    expand_radius : int
        Dilation radius (px) for expansion stage.
    expand_factor : float
        Relaxation multiplier on the V threshold for expansion.
    v_abs_floor : float or None
        Absolute minimum for the effective V threshold. When Otsu collapses
        on uniformly dark frames, this floor prevents under-detection.
        Set to None to disable. Default 0.12 based on empirical analysis:
        good tissue frames have otsu_scaled >= 0.12, blood-dominated frames
        collapse to 0.04-0.07.

    Returns
    -------
    blood_mask : ndarray[bool]
    debug : dict   Contains threshold values and pixel counts.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1].astype(np.float32) / 255.0
    v_ch = hsv[:, :, 2].astype(np.float32) / 255.0

    # R/G ratio (only compute if needed)
    if rg_thresh is not None:
        R = image_bgr[:, :, 2].astype(np.float32)
        G = image_bgr[:, :, 1].astype(np.float32)
        rg_ratio = R / (G + 1.0)
    else:
        rg_ratio = None

    if cavity_mask is not None:
        cavity_bool = cavity_mask > 127 if cavity_mask.dtype == np.uint8 else cavity_mask.astype(bool)
    else:
        cavity_bool = np.ones(v_ch.shape, dtype=bool)

    cavity_px = int(np.sum(cavity_bool))
    if cavity_px == 0:
        return np.zeros(v_ch.shape, dtype=bool), {
            'median_v': 0, 'median_s': 0, 'otsu_raw': 0, 'otsu_scaled': 0,
            'seed_px': 0, 'expanded_px': 0,
        }

    # Otsu on cavity V values
    cavity_v_vals = v_ch[cavity_bool]
    cavity_v_uint8 = (cavity_v_vals * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(cavity_v_uint8, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_raw = otsu_val / 255.0
    effective_v = otsu_raw * otsu_scale

    # V-floor override: when the frame is uniformly dark, Otsu collapses
    # and produces a very low threshold. The floor prevents under-detection.
    otsu_override = False
    if v_abs_floor is not None and effective_v < v_abs_floor:
        effective_v = v_abs_floor
        otsu_override = True

    median_v = float(np.median(cavity_v_vals))
    median_s = float(np.median(s_ch[cavity_bool]))

    # Stage 1: seed — multi-channel conjunction
    seed_mask = cavity_bool & (v_ch < effective_v)
    if s_thresh is not None:
        seed_mask = seed_mask & (s_ch < s_thresh)
    if rg_thresh is not None:
        seed_mask = seed_mask & (rg_ratio < rg_thresh)

    # Stage 2: expand into softer blood at the edges
    blood_expanded = np.zeros_like(cavity_bool, dtype=bool)
    if np.any(seed_mask):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * expand_radius + 1, 2 * expand_radius + 1)
        )
        dilated = cv2.dilate(seed_mask.astype(np.uint8), kernel, iterations=1) > 0

        # Expansion: relax V threshold, keep S relaxed too
        relaxed_v = effective_v * expand_factor
        expand_crit = cavity_bool & (v_ch < relaxed_v)
        if s_thresh is not None:
            # Relax S threshold by same factor (but cap at 1.0)
            relaxed_s = min(s_thresh * expand_factor, 1.0)
            expand_crit = expand_crit & (s_ch < relaxed_s)
        blood_expanded = dilated & expand_crit

    blood_mask = seed_mask | blood_expanded

    seed_px = int(np.sum(seed_mask))
    total_px = int(np.sum(blood_mask))

    debug = {
        'median_v': round(median_v, 4),
        'median_s': round(median_s, 4),
        'otsu_raw': round(otsu_raw, 4),
        'otsu_scaled': round(effective_v, 4),
        'otsu_override': otsu_override,
        'v_abs_floor': v_abs_floor,
        's_thresh': s_thresh,
        'rg_thresh': rg_thresh,
        'seed_px': seed_px,
        'expanded_px': total_px - seed_px,
    }
    return blood_mask, debug

