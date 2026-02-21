#!/usr/bin/env python3
"""
Fluorescence Signal Transformations
====================================
Computes multiple intensity transformations from a BGR frame within an
analysis mask. Each produces a 2D float32 array representing one candidate
fluorescence signal.

Categories:
    A - RGB channel ratios (9 transforms)
    B - Perceptual color spaces (6 transforms)
    C - Data-driven decompositions (2 transforms)

Usage:
    from fluorescence_transforms import compute_all_transforms
    surfaces = compute_all_transforms(frame_bgr, analysis_mask)
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import OrderedDict

# Epsilon for division: ~1 quantization level at 8-bit
_EPS = 1.0 / 255.0


# ---------------------------------------------------------------------------
# Category A: RGB Channel Ratios
# ---------------------------------------------------------------------------

def _t_R(R, G, B, **kw):
    return R

def _t_R_G(R, G, B, **kw):
    return R / (G + _EPS)

def _t_R_RpG(R, G, B, **kw):
    return R / (R + G + _EPS)

def _t_R_B(R, G, B, **kw):
    return R / (B + _EPS)

def _t_R_GpB(R, G, B, **kw):
    return R / (G + B + _EPS)

def _t_NDI(R, G, B, **kw):
    return (R - G) / (R + G + _EPS)

def _t_ExcessR(R, G, B, **kw):
    return (2.0 * R - G - B) / (R + G + B + _EPS)

def _t_logRG(R, G, B, **kw):
    ratio = R / (G + _EPS)
    return np.log(np.maximum(ratio, 1e-6))

def _t_R_minus_kG(R, G, B, analysis_mask=None, **kw):
    rg = R / (G + _EPS)
    if analysis_mask is not None and np.sum(analysis_mask) > 0:
        rg_vals = rg[analysis_mask]
        p10 = np.percentile(rg_vals, 10)
        low_mask = analysis_mask & (rg <= p10)
        k = float(np.median(rg[low_mask])) if np.sum(low_mask) > 0 else float(p10)
    else:
        k = float(np.median(rg))
    return R - k * G


# ---------------------------------------------------------------------------
# Category B: Perceptual Color Spaces
# ---------------------------------------------------------------------------

def _t_LAB_a(frame_bgr, **kw):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 1].astype(np.float32) - 128.0

def _t_LAB_chroma(frame_bgr, **kw):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    return np.sqrt(a**2 + b**2)

def _t_LAB_L(frame_bgr, **kw):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].astype(np.float32) / 255.0

def _t_HSV_H(frame_bgr, **kw):
    # NOTE: Hue is circular (0 and 180 are adjacent in OpenCV).
    # Under blue excitation, tissue hue is in narrow range (~117-125),
    # so linear statistics are approximately valid for this dataset.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 0].astype(np.float32)

def _t_HSV_S(frame_bgr, **kw):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].astype(np.float32) / 255.0

def _t_YCrCb_Cr(frame_bgr, **kw):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 1].astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Category C: Data-Driven Decompositions
# ---------------------------------------------------------------------------

def _t_PCA_PC1(R, G, B, analysis_mask=None, **kw):
    if analysis_mask is None or np.sum(analysis_mask) < 10:
        return R
    pixels = np.column_stack([R[analysis_mask], G[analysis_mask], B[analysis_mask]])
    mean = pixels.mean(axis=0)
    cov = np.cov(pixels - mean, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    pc1 = eigenvectors[:, -1]
    # Sign convention: positive correlation with R channel
    r_vals = R[analysis_mask]
    scores_masked = (pixels - mean) @ pc1
    if np.corrcoef(r_vals, scores_masked)[0, 1] < 0:
        pc1 = -pc1
    all_px = np.column_stack([R.ravel(), G.ravel(), B.ravel()])
    scores = (all_px - mean) @ pc1
    return scores.reshape(R.shape)

def _t_NMF_fluor(R, G, B, analysis_mask=None, **kw):
    """NMF fluorescence component.

    Performance note: NMF fit + full-frame pseudo-inverse projection is the
    most expensive transform (~100ms per frame on 1080p).  Use --skip-nmf
    on runner scripts to omit this when speed matters.
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        return R
    if analysis_mask is None or np.sum(analysis_mask) < 10:
        return R
    try:
        pixels = np.column_stack([
            np.maximum(R[analysis_mask], 0) + 1e-6,
            np.maximum(G[analysis_mask], 0) + 1e-6,
            np.maximum(B[analysis_mask], 0) + 1e-6,
        ])
        pixels = np.nan_to_num(pixels, nan=1e-6, posinf=1e-6, neginf=1e-6)
        pixels = np.maximum(pixels, 1e-6)
        n_comp = min(3, pixels.shape[0], pixels.shape[1])
        model = NMF(n_components=n_comp, init="nndsvda", max_iter=300, random_state=42)
        model.fit(pixels)
        H = model.components_
        r_frac = H[:, 0] / (H.sum(axis=1) + 1e-8)
        fluor_idx = np.argmax(r_frac)
        all_px = np.column_stack([
            np.maximum(R.ravel(), 0) + 1e-6,
            np.maximum(G.ravel(), 0) + 1e-6,
            np.maximum(B.ravel(), 0) + 1e-6,
        ])
        all_px = np.nan_to_num(all_px, nan=1e-6, posinf=1e-6, neginf=1e-6)
        H_pinv = np.linalg.pinv(H)
        W_all = np.maximum(all_px @ H_pinv, 0)
        return W_all[:, fluor_idx].reshape(R.shape)
    except Exception as e:
        import warnings
        warnings.warn(f"NMF deconvolution failed ({e}), returning NaN surface")
        return np.full_like(R, np.nan, dtype=np.float32)


# ---------------------------------------------------------------------------
# Registry and master compute function
# ---------------------------------------------------------------------------

# Metadata includes whether transform can produce negative values
TRANSFORM_REGISTRY = OrderedDict([
    ('R',          {'cat': 'A', 'label': 'R (raw red)',         'fn_type': 'rgb', 'can_be_negative': False}),
    ('R_G',        {'cat': 'A', 'label': 'R/G',                'fn_type': 'rgb', 'can_be_negative': False}),
    ('R_RpG',      {'cat': 'A', 'label': 'R/(R+G)',            'fn_type': 'rgb', 'can_be_negative': False}),
    ('R_B',        {'cat': 'A', 'label': 'R/B',                'fn_type': 'rgb', 'can_be_negative': False}),
    ('R_GpB',      {'cat': 'A', 'label': 'R/(G+B)',            'fn_type': 'rgb', 'can_be_negative': False}),
    ('NDI',        {'cat': 'A', 'label': '(R-G)/(R+G)',        'fn_type': 'rgb', 'can_be_negative': True}),
    ('ExcessR',    {'cat': 'A', 'label': '(2R-G-B)/(R+G+B)',   'fn_type': 'rgb', 'can_be_negative': True}),
    ('logRG',      {'cat': 'A', 'label': 'log(R/G)',           'fn_type': 'rgb', 'can_be_negative': True}),
    ('R_minus_kG', {'cat': 'A', 'label': 'R - kG',             'fn_type': 'rgb', 'can_be_negative': True}),
    ('LAB_a',      {'cat': 'B', 'label': 'LAB a*',             'fn_type': 'bgr', 'can_be_negative': True}),
    ('LAB_chroma', {'cat': 'B', 'label': 'LAB Chroma',         'fn_type': 'bgr', 'can_be_negative': False}),
    ('LAB_L',      {'cat': 'B', 'label': 'LAB L*',             'fn_type': 'bgr', 'can_be_negative': False}),
    ('HSV_H',      {'cat': 'B', 'label': 'HSV Hue',            'fn_type': 'bgr', 'can_be_negative': False}),
    ('HSV_S',      {'cat': 'B', 'label': 'HSV Saturation',     'fn_type': 'bgr', 'can_be_negative': False}),
    ('YCrCb_Cr',   {'cat': 'B', 'label': 'YCrCb Cr',           'fn_type': 'bgr', 'can_be_negative': False}),
    ('PCA_PC1',    {'cat': 'C', 'label': 'PCA PC1',            'fn_type': 'rgb', 'can_be_negative': True}),
    ('NMF_fluor',  {'cat': 'C', 'label': 'NMF fluorescence',   'fn_type': 'rgb', 'can_be_negative': False}),
])

_RGB_FNS = {
    'R': _t_R, 'R_G': _t_R_G, 'R_RpG': _t_R_RpG, 'R_B': _t_R_B,
    'R_GpB': _t_R_GpB, 'NDI': _t_NDI, 'ExcessR': _t_ExcessR,
    'logRG': _t_logRG, 'R_minus_kG': _t_R_minus_kG,
    'PCA_PC1': _t_PCA_PC1, 'NMF_fluor': _t_NMF_fluor,
}
_BGR_FNS = {
    'LAB_a': _t_LAB_a, 'LAB_chroma': _t_LAB_chroma, 'LAB_L': _t_LAB_L,
    'HSV_H': _t_HSV_H, 'HSV_S': _t_HSV_S, 'YCrCb_Cr': _t_YCrCb_Cr,
}


def compute_all_transforms(frame_bgr: np.ndarray, analysis_mask: np.ndarray, skip: set[str] | None = None) -> OrderedDict[str, np.ndarray]:
    """Compute all registered transformations for a single frame.

    Parameters
    ----------
    frame_bgr : ndarray (H, W, 3) uint8
    analysis_mask : ndarray (H, W) bool
    skip : set of str, optional

    Returns
    -------
    surfaces : OrderedDict[str, ndarray float32]
        Pixels outside analysis_mask are NaN.
    """
    skip = skip or set()
    B_ch = frame_bgr[:, :, 0].astype(np.float32) / 255.0
    G_ch = frame_bgr[:, :, 1].astype(np.float32) / 255.0
    R_ch = frame_bgr[:, :, 2].astype(np.float32) / 255.0

    surfaces = OrderedDict()
    for key, meta in TRANSFORM_REGISTRY.items():
        if key in skip:
            continue
        try:
            if meta['fn_type'] == 'rgb':
                fn = _RGB_FNS[key]
                raw = fn(R_ch, G_ch, B_ch, analysis_mask=analysis_mask)
            else:
                fn = _BGR_FNS[key]
                raw = fn(frame_bgr, analysis_mask=analysis_mask)

            surface = np.full(raw.shape, np.nan, dtype=np.float32)
            surface[analysis_mask] = raw[analysis_mask].astype(np.float32)
            surfaces[key] = surface
        except Exception as e:
            import warnings
            warnings.warn(f"Transform '{key}' failed ({e}), storing NaN surface")
            surface = np.full(R_ch.shape, np.nan, dtype=np.float32)
            surfaces[key] = surface

    return surfaces


def get_transform_names(categories: list[str] | None = None) -> list[str]:
    if categories is None:
        return list(TRANSFORM_REGISTRY.keys())
    return [k for k, v in TRANSFORM_REGISTRY.items() if v['cat'] in categories]


def get_transform_label(key: str) -> str:
    return TRANSFORM_REGISTRY.get(key, {}).get('label', key)


def can_be_negative(key: str) -> bool:
    return TRANSFORM_REGISTRY.get(key, {}).get('can_be_negative', False)


# ── Derived exports (single source of truth) ─────────────────────────────

TRANSFORM_LABELS: dict[str, str] = {
    k: v['label'] for k, v in TRANSFORM_REGISTRY.items()
}

TRANSFORM_CATEGORIES: dict[str, str] = {
    k: v['cat'] for k, v in TRANSFORM_REGISTRY.items()
}

CATEGORY_NAMES: dict[str, str] = {
    'A': 'RGB Channel Ratios',
    'B': 'Perceptual Color Spaces',
    'C': 'Data-Driven Decompositions',
}
