#!/usr/bin/env python3
"""
Polar Surface Analysis
=======================
Transforms Cartesian fluorescence surface to polar coordinates centered
on the peak. Computes radial/circumferential profiles, boundary contour,
and Fourier decomposition.

Fourier metrics are only computed when contour_valid_fraction > 0.5
to avoid artifacts from over-interpolation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def build_polar_surface_fast(surface: np.ndarray, analysis_mask: np.ndarray, peak_yx: tuple[int, int],
                              max_radius: int = 200, n_radii: int = 200, n_angles: int = 360) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized polar transform using scipy map_coordinates."""
    py, px = peak_yx
    H, W = surface.shape

    radii = np.linspace(0, max_radius, n_radii)
    angles_rad = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    R, A = np.meshgrid(radii, angles_rad)
    Y = py + R * np.sin(A)
    X = px + R * np.cos(A)

    filled = np.where(np.isnan(surface), 0.0, surface)
    polar_vals = ndimage.map_coordinates(
        filled, [Y.ravel(), X.ravel()],
        order=1, mode='constant', cval=0.0
    ).reshape(n_angles, n_radii).astype(np.float32)

    mask_float = analysis_mask.astype(np.float32)
    polar_mask = ndimage.map_coordinates(
        mask_float, [Y.ravel(), X.ravel()],
        order=0, mode='constant', cval=0.0
    ).reshape(n_angles, n_radii)

    in_bounds = (Y >= 0) & (Y < H) & (X >= 0) & (X < W)
    valid = (polar_mask > 0.5) & in_bounds

    polar = np.full((n_angles, n_radii), np.nan, dtype=np.float32)
    polar[valid] = polar_vals[valid]

    angles_deg = np.rad2deg(angles_rad)
    return polar, radii, angles_deg


def compute_radial_profiles(polar: np.ndarray, radii: np.ndarray) -> dict:
    """Summary radial profiles across all angles (vectorized)."""
    import warnings
    n_angles, n_radii = polar.shape

    valid_counts = np.sum(~np.isnan(polar), axis=0)
    enough = valid_counts >= 3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pcts = np.nanpercentile(polar, [10, 25, 50, 75, 90], axis=0)

    stats = {
        'median':               np.where(enough, pcts[2], np.nan),
        'p25':                  np.where(enough, pcts[1], np.nan),
        'p75':                  np.where(enough, pcts[3], np.nan),
        'p10':                  np.where(enough, pcts[0], np.nan),
        'p90':                  np.where(enough, pcts[4], np.nan),
        'circumferential_std':  np.where(enough, np.nanstd(polar, axis=0), np.nan),
        'radii':                radii,
    }
    return stats


def compute_circumferential_profiles(polar: np.ndarray, radii: np.ndarray, angles_deg: np.ndarray) -> dict:
    """Circumferential variance at each radius (vectorized)."""
    import warnings
    n_angles, n_radii = polar.shape

    valid_counts = np.sum(~np.isnan(polar), axis=0)
    enough = valid_counts >= 10

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        circ_var_raw = np.nanvar(polar, axis=0)
        circ_mean_raw = np.nanmean(polar, axis=0)

    circ_var = np.where(enough, circ_var_raw, np.nan)
    circ_cv = np.where(
        enough & (circ_mean_raw > 0),
        np.sqrt(circ_var_raw) / circ_mean_raw,
        np.nan,
    )

    valid_idx = np.where(~np.isnan(circ_var))[0]
    max_var_radius = (
        float(radii[valid_idx[np.argmax(circ_var[valid_idx])]])
        if len(valid_idx) > 0 else np.nan
    )

    return {
        'radii': radii,
        'circ_variance': circ_var,
        'circ_cv': circ_cv,
        'max_variance_radius': max_var_radius,
    }


def extract_boundary_contour(polar: np.ndarray, radii: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Extract boundary contour r(theta) at given threshold (vectorized).

    For each angle, finds the first radius where the value drops below
    *threshold* after having been at or above it.
    """
    n_angles, n_radii = polar.shape

    valid = ~np.isnan(polar)
    above = valid & (polar >= threshold)
    below = valid & (polar < threshold)

    # Once the signal has been above threshold, it stays flagged
    was_above = np.maximum.accumulate(above.astype(np.int8), axis=1).astype(bool)

    # First position where was_above AND currently below threshold
    crossing = was_above & below
    has_crossing = np.any(crossing, axis=1)
    first_crossing_idx = np.argmax(crossing, axis=1)

    contour_r = np.full(n_angles, np.nan)
    contour_r[has_crossing] = radii[first_crossing_idx[has_crossing]]
    return contour_r


def compute_contour_metrics(contour_r: np.ndarray) -> dict:
    """Statistics of the boundary contour."""
    valid = contour_r[~np.isnan(contour_r)]
    n_total = len(contour_r)
    n_valid = len(valid)
    valid_fraction = n_valid / n_total if n_total > 0 else 0

    if n_valid < 10:
        return {
            'mean_radius': np.nan, 'std_radius': np.nan, 'cv_radius': np.nan,
            'circularity': np.nan, 'max_min_ratio': np.nan,
            'valid_fraction': valid_fraction,
        }

    mean_r = float(np.mean(valid))
    std_r = float(np.std(valid))
    cv_r = std_r / mean_r if mean_r > 0 else np.nan

    return {
        'mean_radius': mean_r,
        'std_radius': std_r,
        'cv_radius': cv_r,
        'circularity': mean_r**2 / (mean_r**2 + std_r**2) if mean_r > 0 else np.nan,
        'max_min_ratio': float(np.max(valid) / np.min(valid)) if np.min(valid) > 0 else np.nan,
        'valid_fraction': valid_fraction,
    }


def fourier_decompose_contour(contour_r: np.ndarray, min_valid_fraction: float = 0.5) -> dict:
    """Fourier decomposition of boundary contour r(theta).

    Only computed if valid_fraction >= min_valid_fraction to avoid
    artifacts from over-interpolation of NaN gaps.
    """
    nans = np.isnan(contour_r)
    n_total = len(contour_r)
    n_valid = int(np.sum(~nans))
    valid_frac = n_valid / n_total if n_total > 0 else 0

    empty = {
        'dc': np.nan, 'harmonics': np.array([]),
        'low_freq_energy': np.nan, 'high_freq_energy': np.nan,
        'roughness_ratio': np.nan,
    }

    if valid_frac < min_valid_fraction:
        return empty

    valid = contour_r.copy()
    if np.all(nans):
        return empty

    # Interpolate NaN gaps for FFT
    if np.any(nans):
        not_nan = ~nans
        indices = np.arange(len(valid))
        valid[nans] = np.interp(indices[nans], indices[not_nan], valid[not_nan])

    fft_vals = np.fft.rfft(valid)
    amplitudes = np.abs(fft_vals) / len(valid)

    dc = amplitudes[0]
    harmonics = amplitudes[1:]

    n_h = len(harmonics)
    low_cut = min(5, n_h)
    low_energy = float(np.sum(harmonics[:low_cut]**2))
    high_energy = float(np.sum(harmonics[low_cut:]**2)) if n_h > low_cut else 0.0
    total = low_energy + high_energy

    return {
        'dc': float(dc),
        'harmonics': harmonics,
        'low_freq_energy': low_energy,
        'high_freq_energy': high_energy,
        'roughness_ratio': high_energy / total if total > 0 else np.nan,
    }


def compute_polar_metrics(surface: np.ndarray, analysis_mask: np.ndarray, peak_yx: tuple[int, int],
                          max_radius: int = 200, n_radii: int = 200, n_angles: int = 360,
                          contour_threshold: float = 0.5) -> tuple[dict, dict]:
    """All polar metrics for a single normalized surface."""
    polar, radii, angles = build_polar_surface_fast(
        surface, analysis_mask, peak_yx, max_radius, n_radii, n_angles
    )

    radial = compute_radial_profiles(polar, radii)
    circ = compute_circumferential_profiles(polar, radii, angles)

    contour_r = extract_boundary_contour(polar, radii, contour_threshold)
    contour_m = compute_contour_metrics(contour_r)
    fourier = fourier_decompose_contour(contour_r)

    metrics = {
        'peak_y': int(peak_yx[0]),
        'peak_x': int(peak_yx[1]),
        'max_variance_radius': circ['max_variance_radius'],
    }
    for k, v in contour_m.items():
        metrics[f'contour_{k}'] = v
    metrics['fourier_dc'] = fourier['dc']
    metrics['fourier_low_energy'] = fourier['low_freq_energy']
    metrics['fourier_high_energy'] = fourier['high_freq_energy']
    metrics['fourier_roughness'] = fourier['roughness_ratio']

    # Half-decay from radial median profile (use max of profile, not first point)
    med_prof = radial['median']
    valid_med = ~np.isnan(med_prof)
    half_decay_dist = np.nan
    if np.any(valid_med):
        peak_val = float(np.nanmax(med_prof[valid_med]))
        if peak_val > 0:
            half_target = peak_val * 0.5
            # Find first crossing below half_target AFTER the max
            max_ri = int(np.nanargmax(med_prof))
            for ri in range(max_ri, len(radii)):
                if valid_med[ri] and med_prof[ri] < half_target:
                    half_decay_dist = float(radii[ri])
                    break
    metrics['half_decay_dist'] = half_decay_dist

    arrays = {
        'polar_surface': polar,
        'radii': radii,
        'angles': angles,
        'radial_profiles': radial,
        'circumferential': circ,
        'boundary_contour': contour_r,
        'fourier_harmonics': fourier['harmonics'],
    }

    return metrics, arrays
