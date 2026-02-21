#!/usr/bin/env python3
"""
Persistent Homology for Fluorescence Surface Analysis
======================================================
Uses sublevel-set persistent homology via cubical complexes (gudhi)
to capture the SPATIAL topological structure of fluorescence landscapes.

Cubical complexes operate on the 2D pixel grid directly, preserving
spatial relationships. This is the correct approach for image data
(unlike Rips complexes on point clouds, which lose spatial structure).

Dependencies: gudhi (pip install gudhi), persim (pip install persim)
"""
from __future__ import annotations

import numpy as np


def compute_persistence_cubical(surface: np.ndarray, analysis_mask: np.ndarray, subsample_factor: int = 1) -> np.ndarray:
    """Compute sublevel-set persistence using gudhi CubicalComplex.

    Operates on the 2D pixel grid, preserving spatial topology.
    Uses negated intensity so that fluorescence peaks correspond to
    persistent features (born early, die late).

    Parameters
    ----------
    surface : ndarray float32 (H, W), normalized [0,1], NaN outside mask
    analysis_mask : ndarray bool (H, W)
    subsample_factor : int
        Downsample factor for speed (2 = half resolution, 4 = quarter).
        Set to 1 for full resolution.

    Returns
    -------
    diagram : ndarray (N, 2) -- birth/death pairs for 0-dim features
    """
    try:
        import gudhi
    except ImportError:
        # Fallback to 1D persistence if gudhi unavailable
        vals = surface[analysis_mask]
        vals = vals[~np.isnan(vals)]
        return compute_persistence_1d_fallback(vals, subsample=3000)

    # Prepare: fill non-mask with high value (will be born last, topologically irrelevant)
    filled = np.where(analysis_mask, surface, np.nan)
    filled = np.where(np.isnan(filled), 1.0, filled)

    # Negate so peaks become minima (sublevel set tracks peaks)
    neg = -filled

    # Subsample for speed
    if subsample_factor > 1:
        from scipy.ndimage import zoom
        neg = zoom(neg, 1.0 / subsample_factor, order=1)

    # Build cubical complex
    cc = gudhi.CubicalComplex(top_dimensional_cells=neg)
    cc.compute_persistence()

    # Extract 0-dim persistence pairs (connected components)
    pairs = cc.persistence_intervals_in_dimension(0)

    # Filter out infinite persistence (the global component)
    finite = pairs[np.isfinite(pairs[:, 1])] if len(pairs) > 0 else np.zeros((0, 2))

    return finite


def compute_persistence_1d_fallback(values: np.ndarray, subsample: int | None = 3000) -> np.ndarray:
    """Fallback: 1D persistence from intensity values only (no spatial info).

    Use this when gudhi is unavailable or for quick comparison.
    NOTE: This loses spatial structure -- two different patterns with the
    same intensity histogram produce identical diagrams.
    """
    try:
        from ripser import ripser
    except ImportError:
        return np.zeros((0, 2))

    if len(values) < 10:
        return np.zeros((0, 2))

    if subsample is not None and len(values) > subsample:
        idx = np.random.RandomState(42).choice(len(values), subsample, replace=False)
        values = values[idx]

    pts = (-values).reshape(-1, 1)
    result = ripser(pts, maxdim=0, thresh=2.0)
    dgm = result['dgms'][0]
    finite = dgm[np.isfinite(dgm[:, 1])]
    return finite


def persistence_summary_metrics(diagram: np.ndarray) -> dict:
    """Summary statistics from a persistence diagram."""
    if len(diagram) == 0:
        return {
            'n_features': 0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'median_persistence': 0.0,
            'persistence_entropy': 0.0,
            'total_persistence': 0.0,
            'persistence_ratio': 0.0,
        }

    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return {
            'n_features': 0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'median_persistence': 0.0,
            'persistence_entropy': 0.0,
            'total_persistence': 0.0,
            'persistence_ratio': 0.0,
        }

    total = float(np.sum(lifetimes))
    probs = lifetimes / total
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    # Persistence ratio: max / 2nd max (how dominant is the primary peak)
    sorted_lt = np.sort(lifetimes)[::-1]
    pers_ratio = float(sorted_lt[0] / sorted_lt[1]) if len(sorted_lt) > 1 else float('inf')

    return {
        'n_features': len(lifetimes),
        'max_persistence': float(np.max(lifetimes)),
        'mean_persistence': float(np.mean(lifetimes)),
        'median_persistence': float(np.median(lifetimes)),
        'persistence_entropy': entropy,
        'total_persistence': total,
        'persistence_ratio': min(pers_ratio, 100.0),  # cap for JSON
    }


def wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Wasserstein distance between two persistence diagrams."""
    try:
        from persim import wasserstein as persim_wasserstein
    except ImportError:
        raise ImportError(
            "persim is required for Wasserstein distance computation. "
            "Install with: pip install persim"
        )

    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    if len(dgm1) == 0:
        dgm1 = np.zeros((1, 2))
    if len(dgm2) == 0:
        dgm2 = np.zeros((1, 2))

    return float(persim_wasserstein(dgm1, dgm2))


def pairwise_wasserstein_distances(diagrams: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Pairwise Wasserstein distances between a list of diagrams."""
    n = len(diagrams)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = wasserstein_distance(diagrams[i], diagrams[j])
            dist[i, j] = d
            dist[j, i] = d

    upper = dist[np.triu_indices(n, k=1)]
    mean_d = float(np.mean(upper)) if len(upper) > 0 else 0.0

    return dist, mean_d
