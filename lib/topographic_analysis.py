#!/usr/bin/env python3
"""
Topographic Surface Analysis
==============================
Treats a 2D fluorescence intensity field as a 3D topographic surface
(x, y = spatial, z = intensity) and computes global metrics.

Handles transforms with negative values (NDI, ExcessR, logRG, LAB a*,
PCA PC1, R-kG) via min-max normalization before computing metrics that
assume [0,1] range.

Gradient computation erodes the analysis mask inward to avoid artificial
edge gradients from the mask boundary.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Surface normalization
# ---------------------------------------------------------------------------

def normalize_surface(surface: np.ndarray, analysis_mask: np.ndarray, method: str = 'minmax') -> tuple[np.ndarray, dict]:
    """Normalize surface values to [0, 1] within the analysis mask.

    Parameters
    ----------
    surface : ndarray float32 (H, W), NaN outside analysis_mask
    analysis_mask : ndarray bool (H, W)
    method : str
        'minmax' - shift min to 0, divide by range (handles negatives)
        'p99'    - shift p1 to 0, divide by (p99 - p1) (robust)
        'max'    - divide by max (only for non-negative surfaces)

    Returns
    -------
    normalized : ndarray float32 (H, W), NaN outside mask, ~[0,1] inside
    norm_info : dict with 'shift' and 'scale' used
    """
    valid = surface[analysis_mask]
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        return surface.copy(), {'shift': 0.0, 'scale': 1.0}

    if method == 'minmax':
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        scale = vmax - vmin if vmax > vmin else 1.0
        shift = vmin
    elif method == 'p99':
        p1 = float(np.percentile(valid, 1))
        p99 = float(np.percentile(valid, 99))
        scale = p99 - p1 if p99 > p1 else 1.0
        shift = p1
    elif method == 'max':
        shift = max(0.0, float(np.min(valid)))  # shift negatives to 0
        scale = float(np.max(valid)) - shift
        if scale <= 0:
            scale = 1.0
    else:
        raise ValueError(f"Unknown normalize method: {method!r}. "
                         f"Expected 'minmax', 'p99', or 'max'.")

    normalized = surface.copy()
    # Operate on all mask pixels (preserves NaN naturally: NaN arithmetic → NaN)
    mask_vals = surface[analysis_mask]
    normalized[analysis_mask] = np.clip((mask_vals - shift) / scale, 0.0, 1.0)

    return normalized, {'shift': shift, 'scale': scale}


# ---------------------------------------------------------------------------
# Area-threshold curve (hypsometric curve)
# ---------------------------------------------------------------------------

def compute_hypsometric_curve(surface: np.ndarray, analysis_mask: np.ndarray, n_levels: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Compute area-threshold curve A(t).

    Surface MUST be normalized to [0, 1] before calling this.
    """
    valid = surface[analysis_mask]
    valid = valid[~np.isnan(valid)]
    n_valid = len(valid)
    if n_valid == 0:
        return np.linspace(0, 1, n_levels), np.zeros(n_levels)

    thresholds = np.linspace(0, 1, n_levels)
    # Vectorized: sort once, use searchsorted
    sorted_vals = np.sort(valid)
    counts_above = n_valid - np.searchsorted(sorted_vals, thresholds)
    area_fractions = counts_above / n_valid
    return thresholds, area_fractions.astype(np.float64)


def compute_hypsometric_metrics(thresholds: np.ndarray, area_fractions: np.ndarray) -> dict:
    """Extract summary metrics from the hypsometric curve."""
    dt = thresholds[1] - thresholds[0] if len(thresholds) > 1 else 1.0
    dA = np.gradient(area_fractions, dt)
    steepest_idx = np.argmax(np.abs(dA))

    def area_at(t):
        idx = np.searchsorted(thresholds, t)
        idx = min(idx, len(area_fractions) - 1)
        return float(area_fractions[idx])

    return {
        'hypso_steepness': float(np.abs(dA[steepest_idx])),
        'hypso_transition_t': float(thresholds[steepest_idx]),
        'hypso_area_above_025': area_at(0.25),
        'hypso_area_above_050': area_at(0.50),
        'hypso_area_above_075': area_at(0.75),
        'hypso_auc': float(np.trapezoid(area_fractions, thresholds)
                          if hasattr(np, 'trapezoid')
                          else np.trapz(area_fractions, thresholds)),
    }


# ---------------------------------------------------------------------------
# Volume above threshold
# ---------------------------------------------------------------------------

def compute_volume_above_threshold(surface: np.ndarray, analysis_mask: np.ndarray,
                                    thresholds: tuple[float, ...] = (0.25, 0.50, 0.75)) -> dict:
    """Integral of (z - t) over pixels where z > t. Surface must be normalized."""
    valid = surface[analysis_mask]
    valid = valid[~np.isnan(valid)]
    result = {}
    for t in thresholds:
        above = valid[valid > t]
        vol = float(np.sum(above - t)) if len(above) > 0 else 0.0
        result['vol_above_%03d' % int(t * 100)] = vol
    return result


# ---------------------------------------------------------------------------
# Surface roughness
# ---------------------------------------------------------------------------

def compute_roughness(surface: np.ndarray, analysis_mask: np.ndarray) -> dict:
    """Standard deviation of normalized intensity."""
    valid = surface[analysis_mask]
    valid = valid[~np.isnan(valid)]
    if len(valid) < 2:
        return {'roughness': 0.0, 'mean_z': 0.0, 'median_z': 0.0}
    return {
        'roughness': float(np.std(valid)),
        'mean_z': float(np.mean(valid)),
        'median_z': float(np.median(valid)),
    }


# ---------------------------------------------------------------------------
# Gradient field
# ---------------------------------------------------------------------------

def compute_gradient_field(surface: np.ndarray, analysis_mask: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spatial gradient, eroding mask to avoid boundary artifacts.

    The analysis mask is eroded inward by (ksize//2 + 1) pixels before
    gradient statistics are computed. This prevents artificial gradients
    at the NaN-to-value boundary from contaminating the metrics.
    """
    # Fill NaN with local median (better than 0 for boundary behavior)
    filled = np.where(np.isnan(surface), 0.0, surface).astype(np.float32)

    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=ksize)

    mag = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)

    # Erode mask to avoid boundary artifacts
    erode_px = ksize // 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*erode_px+1, 2*erode_px+1))
    eroded_mask = cv2.erode(analysis_mask.astype(np.uint8), kernel) > 0

    grad_mag = np.full_like(mag, np.nan)
    grad_dir = np.full_like(direction, np.nan)
    grad_mag[eroded_mask] = mag[eroded_mask]
    grad_dir[eroded_mask] = direction[eroded_mask]

    return grad_mag, grad_dir, eroded_mask


def compute_gradient_metrics(grad_mag: np.ndarray, eroded_mask: np.ndarray) -> dict:
    """Summary statistics of gradient magnitude within eroded mask."""
    valid = grad_mag[eroded_mask]
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        return {
            'grad_mean': 0.0, 'grad_median': 0.0, 'grad_p90': 0.0,
            'grad_max': 0.0, 'grad_p90_threshold': 0.0,
        }

    p90 = float(np.percentile(valid, 90))
    return {
        'grad_mean': float(np.mean(valid)),
        'grad_median': float(np.median(valid)),
        'grad_p90': p90,
        'grad_max': float(np.max(valid)),
        # The threshold VALUE at p90, not the fraction (which is always ~10%)
        'grad_p90_threshold': p90,
    }


# ---------------------------------------------------------------------------
# Transition zone analysis
# ---------------------------------------------------------------------------

def compute_transition_zone(grad_mag: np.ndarray, eroded_mask: np.ndarray, peak_yx: tuple[int, int] | None = None,
                            percentile_threshold: int = 90) -> tuple[np.ndarray, dict]:
    """Identify transition zone and compute spatial metrics."""
    valid = grad_mag[eroded_mask]
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        return np.zeros_like(eroded_mask), {}

    thresh = float(np.percentile(valid, percentile_threshold))
    zone_mask = eroded_mask & (grad_mag >= thresh) & (~np.isnan(grad_mag))

    zone_px = int(np.sum(zone_mask))
    total_px = int(np.sum(eroded_mask))

    metrics = {
        'zone_area_px': zone_px,
        'zone_grad_threshold': thresh,
    }

    if peak_yx is not None and zone_px > 0:
        py, px = peak_yx
        ys, xs = np.where(zone_mask)
        dists = np.sqrt((ys - py)**2 + (xs - px)**2)
        metrics['zone_mean_dist'] = float(np.mean(dists))
        metrics['zone_median_dist'] = float(np.median(dists))
        metrics['zone_radial_width'] = float(
            np.percentile(dists, 90) - np.percentile(dists, 10))

    return zone_mask, metrics


# ---------------------------------------------------------------------------
# Chan-Vese segmentation
# ---------------------------------------------------------------------------

def compute_chan_vese_boundary(surface: np.ndarray, analysis_mask: np.ndarray, n_iter: int = 100, smoothing: int = 3) -> tuple[np.ndarray, float]:
    """Chan-Vese segmentation. Surface must be normalized [0,1]."""
    try:
        from skimage.segmentation import morphological_chan_vese
    except ImportError:
        valid = surface[analysis_mask]
        valid = valid[~np.isnan(valid)]
        med = float(np.median(valid)) if len(valid) > 0 else 0.5
        seg = analysis_mask & (surface > med)
        return seg, float(np.sum(seg) / max(np.sum(analysis_mask), 1))

    img = np.where(np.isnan(surface), 0.0, surface).astype(np.float64)

    # Initialize within mask only (not global checkerboard)
    init = np.zeros(img.shape, dtype=np.int8)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    checker = ((yy // 10) + (xx // 10)) % 2
    init[analysis_mask & (checker == 1)] = 1

    ls = morphological_chan_vese(img, num_iter=n_iter, smoothing=smoothing,
                                 init_level_set=init)

    mask0 = analysis_mask & (ls == 0)
    mask1 = analysis_mask & (ls == 1)
    mean0 = float(np.nanmean(surface[mask0])) if np.any(mask0) else 0.0
    mean1 = float(np.nanmean(surface[mask1])) if np.any(mask1) else 0.0

    high_mask = mask1 if mean1 >= mean0 else mask0
    area_frac = float(np.sum(high_mask) / max(np.sum(analysis_mask), 1))
    return high_mask, area_frac


# ---------------------------------------------------------------------------
# Peak finding
# ---------------------------------------------------------------------------

def find_peak(surface: np.ndarray, analysis_mask: np.ndarray, v_channel: np.ndarray | None = None, v_percentile: int = 75) -> tuple[tuple[int, int], float]:
    """Find peak. For non-fluorescence transforms (HSV_S, LAB_L),
    v_channel restriction may exclude the true peak -- caller can
    pass v_channel=None to disable."""
    valid_mask = analysis_mask.copy()

    if v_channel is not None:
        v_vals = v_channel[analysis_mask]
        if len(v_vals) > 0:
            v_thresh = np.percentile(v_vals, v_percentile)
            candidate = analysis_mask & (v_channel >= v_thresh)
            if np.sum(candidate) > 0:
                valid_mask = candidate

    masked = np.where(valid_mask, surface, -np.inf)
    masked = np.where(np.isnan(masked), -np.inf, masked)
    flat_idx = np.argmax(masked)
    peak_y, peak_x = np.unravel_index(flat_idx, surface.shape)
    peak_val = float(surface[peak_y, peak_x])

    return (peak_y, peak_x), peak_val


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def compute_global_metrics(surface: np.ndarray, analysis_mask: np.ndarray, peak_yx: tuple[int, int] | None = None,
                           normalize_method: str = 'p99') -> tuple[dict, dict, np.ndarray]:
    """Compute all global topographic metrics for a single surface.

    Returns
    -------
    metrics : dict of scalar metrics
    arrays : dict of array data
    norm_surface : ndarray, the normalized surface (for reuse by caller)
    """
    # Normalize
    norm_surf, norm_info = normalize_surface(surface, analysis_mask, normalize_method)

    # Hypsometric curve
    thresholds, area_fracs = compute_hypsometric_curve(norm_surf, analysis_mask)
    hypso_metrics = compute_hypsometric_metrics(thresholds, area_fracs)

    # Volume
    vol_metrics = compute_volume_above_threshold(norm_surf, analysis_mask)

    # Roughness
    rough_metrics = compute_roughness(norm_surf, analysis_mask)

    # Gradient field (with eroded mask)
    grad_mag, grad_dir, eroded_mask = compute_gradient_field(norm_surf, analysis_mask)
    grad_metrics = compute_gradient_metrics(grad_mag, eroded_mask)

    # Transition zone
    _, zone_metrics = compute_transition_zone(grad_mag, eroded_mask, peak_yx)

    # Combine
    metrics = {
        'norm_shift': norm_info['shift'],
        'norm_scale': norm_info['scale'],
        'analysis_px': int(np.sum(analysis_mask)),
    }
    metrics.update(hypso_metrics)
    metrics.update(vol_metrics)
    metrics.update(rough_metrics)
    metrics.update(grad_metrics)
    metrics.update(zone_metrics)

    arrays = {
        'hypsometric_thresholds': thresholds,
        'hypsometric_area_fractions': area_fracs,
        'gradient_magnitude': grad_mag,
        'gradient_direction': grad_dir,
    }

    return metrics, arrays, norm_surf
