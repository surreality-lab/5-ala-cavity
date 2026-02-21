#!/usr/bin/env python3
"""
Comprehensive Statistical Evaluation of Fluorescence Transforms
=================================================================
Loads functional_data.npz (from extract_functional_data.py) and the
scalar frame_metrics_summary.csv (from run_5fps_topographic_analysis.py)
to perform rigorous inter- and intra-frame statistical comparisons of
hypsometric curves, 3D surface representations, and pixel distributions
across all 17 fluorescence transforms.

Table of Contents (search "SECTION X:" to navigate)
-----------------------------------------------------
    Section A   Data Loading & Assembly
    Section B   Functional Curve Analysis (FPCA, L2, band depth)
    Section C   Surface-to-Surface Comparison (SSIM, correlation)
    Section D   Distributional Comparison (KS, Wasserstein, shape)
    Section E   Enhanced Hypothesis Testing (Friedman, Nemenyi, FDR)
    Section F   Temporal Analysis (ACF, effective n, subsampling)
    Section G   Transform Characterization (profiling, clustering)
    Section H   TDA Integration (persistence in stats framework)
    Summary     Consolidated report generation
    Main        CLI entry point

Usage:
    python scripts/03_analysis/comprehensive_statistical_evaluation.py
    python scripts/03_analysis/comprehensive_statistical_evaluation.py \\
        --npz path/to/functional_data.npz --csv path/to/frame_metrics_summary.csv
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import warnings
import json
import sys

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Maximum number of iterations')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Default paths (from centralized config) ────────────────────────────────
from config import DEFAULT_NPZ, DEFAULT_CSV, DEFAULT_STATS
from lib.utils import setup_logging
from lib.fluorescence_transforms import (
    TRANSFORM_LABELS, TRANSFORM_CATEGORIES
)
_DEFAULT_NPZ = DEFAULT_NPZ
_DEFAULT_CSV = DEFAULT_CSV
_DEFAULT_OUTPUT = DEFAULT_STATS

logger = logging.getLogger('comprehensive_statistical_evaluation')


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A: Data Loading & Assembly
# ═══════════════════════════════════════════════════════════════════════════

def load_functional_data(npz_path):
    """Load the functional_data.npz file."""
    data = np.load(npz_path)
    result = {key: data[key] for key in data.files}
    # Convert transform_names from ndarray to list of strings
    result['transform_names'] = [str(s) for s in result['transform_names']]
    logger.info("Loaded functional data: %s", npz_path)
    logger.info("  Frames: %d", result['hypso_curves'].shape[0])
    logger.info("  Transforms: %d", result['hypso_curves'].shape[1])
    return result


def load_scalar_metrics(csv_path, stable_only=True):
    """Load the scalar metrics CSV."""
    df = pd.read_csv(csv_path)
    logger.info("Loaded scalar metrics: %d frames, %d columns", len(df), len(df.columns))
    if stable_only:
        if 'light_stable' in df.columns:
            df = df[df['light_stable'] == True].copy()
            logger.info("  After light-stable filter: %d frames", len(df))
    return df


def get_labels(transform_names):
    """Get display labels for transform names."""
    return [TRANSFORM_LABELS.get(t, t) for t in transform_names]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B: Functional Curve Analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_l2_distance_matrix(curves):
    """Compute L2 distance between curves for each pair of transforms.

    Vectorized over frames: inner loop is only n_transforms*(n_transforms-1)/2
    iterations (136 for 17 transforms), each processing all frames at once.

    Parameters
    ----------
    curves : ndarray (n_frames, n_transforms, n_points)

    Returns
    -------
    mean_dist : ndarray (n_transforms, n_transforms) -- mean L2 across frames
    std_dist : ndarray (n_transforms, n_transforms) -- std of L2 across frames
    """
    n_frames, n_transforms, n_points = curves.shape
    all_dists = np.full((n_frames, n_transforms, n_transforms), np.nan)

    for i in range(n_transforms):
        for j in range(i + 1, n_transforms):
            c1 = curves[:, i, :]  # (n_frames, n_points)
            c2 = curves[:, j, :]
            valid = ~np.isnan(c1) & ~np.isnan(c2)
            valid_count = np.sum(valid, axis=1)  # (n_frames,)
            diff = np.where(valid, c1 - c2, 0.0)
            sq_sum = np.sum(diff ** 2, axis=1)
            l2 = np.where(valid_count > 5,
                          np.sqrt(sq_sum / valid_count), np.nan)
            all_dists[:, i, j] = l2
            all_dists[:, j, i] = l2

    mean_dist = np.nanmean(all_dists, axis=0)
    std_dist = np.nanstd(all_dists, axis=0)
    return mean_dist, std_dist


def run_functional_pca(curves, transform_names):
    """Run Functional PCA on curves using scikit-fda.

    Parameters
    ----------
    curves : ndarray (n_frames, n_transforms, n_points)
        Organized as n_frames * n_transforms observations of n_points length.

    Returns
    -------
    results : dict with FPCA components, scores, variance explained
    """
    try:
        from skfda import FDataGrid
        from skfda.preprocessing.dim_reduction import FPCA
    except ImportError:
        logger.warning("scikit-fda not installed, skipping FPCA")
        return None

    n_frames, n_transforms, n_points = curves.shape
    grid_points = np.linspace(0, 1, n_points)

    # Organize: for each transform, collect curves across frames
    results = {'per_transform': {}, 'global': None}

    # -- Global FPCA: pool all curves from all transforms --
    all_curves = []
    all_labels = []
    for ti in range(n_transforms):
        for fi in range(n_frames):
            c = curves[fi, ti, :]
            if not np.any(np.isnan(c)):
                all_curves.append(c)
                all_labels.append(transform_names[ti])

    if len(all_curves) < 10:
        logger.warning("Too few valid curves for FPCA")
        return results

    fd = FDataGrid(np.array(all_curves), grid_points)
    n_components = min(5, len(all_curves) - 1)
    fpca = FPCA(n_components=n_components)
    scores = fpca.fit_transform(fd)

    results['global'] = {
        'scores': np.array(scores),
        'labels': all_labels,
        'variance_explained': fpca.explained_variance_ratio_,
        'n_components': n_components,
    }

    # -- Per-transform FPCA: variation within each transform across frames --
    for ti, tname in enumerate(transform_names):
        t_curves = []
        for fi in range(n_frames):
            c = curves[fi, ti, :]
            if not np.any(np.isnan(c)):
                t_curves.append(c)
        if len(t_curves) < 5:
            continue
        fd_t = FDataGrid(np.array(t_curves), grid_points)
        n_comp_t = min(3, len(t_curves) - 1)
        fpca_t = FPCA(n_components=n_comp_t)
        scores_t = fpca_t.fit_transform(fd_t)
        results['per_transform'][tname] = {
            'scores': np.array(scores_t),
            'variance_explained': fpca_t.explained_variance_ratio_,
        }

    return results


def compute_band_depth(curves, transform_names):
    """Compute modified band depth for ranking curve centrality.

    Parameters
    ----------
    curves : ndarray (n_frames, n_transforms, n_points)

    Returns
    -------
    depths : dict mapping transform_name -> mean band depth across frames
    """
    try:
        from skfda import FDataGrid
        from skfda.exploratory.depth import ModifiedBandDepth
    except ImportError:
        logger.warning("scikit-fda not installed, skipping band depth")
        return None

    n_frames, n_transforms, n_points = curves.shape
    grid_points = np.linspace(0, 1, n_points)
    depth_fn = ModifiedBandDepth()

    # For each frame, compute band depth of each transform's curve
    # relative to all transforms in that frame
    frame_depths = np.full((n_frames, n_transforms), np.nan)

    for fi in range(n_frames):
        frame_curves = []
        valid_idx = []
        for ti in range(n_transforms):
            c = curves[fi, ti, :]
            if not np.any(np.isnan(c)):
                frame_curves.append(c)
                valid_idx.append(ti)

        if len(frame_curves) < 3:
            continue

        fd = FDataGrid(np.array(frame_curves), grid_points)
        try:
            depths = depth_fn(fd)
            for k, ti in enumerate(valid_idx):
                frame_depths[fi, ti] = depths[k]
        except Exception as e:
            logger.debug("Band depth failed for frame %d: %s", fi, e)
            continue

    # Average depth across frames
    mean_depths = {}
    for ti, tname in enumerate(transform_names):
        vals = frame_depths[:, ti]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            mean_depths[tname] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'median': float(np.median(vals)),
            }

    return mean_depths


def functional_curve_analysis(fdata, output_dir):
    """Run all functional curve analyses (Section B).

    Analyzes hypsometric curves, radial profiles, and boundary contours
    as functional data objects.
    """
    logger.info("\n" + "=" * 80)
    logger.info("SECTION B: FUNCTIONAL CURVE ANALYSIS")
    logger.info("=" * 80)

    transform_names = fdata['transform_names']
    labels = get_labels(transform_names)
    sec_dir = output_dir / "B_functional"
    sec_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for curve_type, curve_key, x_key, x_label, y_label in [
        ('hypsometric', 'hypso_curves', 'hypso_thresholds',
         'Threshold', 'Area Fraction A(t)'),
        ('radial_profile', 'radial_profiles', 'radii',
         'Distance from Peak (px)', 'Normalized Intensity'),
        ('boundary_contour', 'boundary_contours', 'angles_deg',
         'Angle (degrees)', 'Boundary Radius (px)'),
    ]:
        logger.info("\n--- %s ---", curve_type)
        curves = fdata[curve_key]  # (n_frames, n_transforms, n_points)
        x_vals = fdata[x_key]
        n_frames, n_transforms, n_points = curves.shape

        # L2 distance matrix
        logger.info("  Computing L2 distance matrix...")
        l2_mean, l2_std = compute_l2_distance_matrix(curves)
        results[f'{curve_type}_l2_mean'] = l2_mean
        results[f'{curve_type}_l2_std'] = l2_std

        # Save L2 distance matrix as CSV
        l2_df = pd.DataFrame(l2_mean, index=labels, columns=labels)
        l2_df.to_csv(sec_dir / f'{curve_type}_l2_distance_matrix.csv')

        # Band depth
        logger.info("  Computing band depth ranking...")
        bd = compute_band_depth(curves, transform_names)
        if bd:
            results[f'{curve_type}_band_depth'] = bd
            bd_df = pd.DataFrame(bd).T
            bd_df.index = [TRANSFORM_LABELS.get(t, t) for t in bd_df.index]
            bd_df.to_csv(sec_dir / f'{curve_type}_band_depth.csv')
            logger.info("  Band depth (most central curves):")
            for t in sorted(bd, key=lambda x: bd[x]['mean'], reverse=True)[:5]:
                logger.info("    %20s  depth=%.4f +/- %.4f",
                            TRANSFORM_LABELS.get(t, t), bd[t]['mean'], bd[t]['std'])

        # FPCA (only for hypsometric -- most interpretable)
        if curve_type == 'hypsometric':
            logger.info("  Running Functional PCA...")
            fpca_results = run_functional_pca(curves, transform_names)
            if fpca_results and fpca_results['global']:
                results['fpca_hypsometric'] = fpca_results
                ve = fpca_results['global']['variance_explained']
                logger.info("  FPCA variance explained: %s",
                            ', '.join('%.1f%%' % (v * 100) for v in ve[:3]))

        # -- Visualization: curve overlays with confidence bands --
        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.cm.get_cmap('tab20', n_transforms)
        for ti in range(n_transforms):
            frame_curves = curves[:, ti, :]
            # Remove frames with all NaN
            valid_mask = ~np.all(np.isnan(frame_curves), axis=1)
            valid_curves = frame_curves[valid_mask]
            if len(valid_curves) < 2:
                continue
            median_c = np.nanmedian(valid_curves, axis=0)
            p25 = np.nanpercentile(valid_curves, 25, axis=0)
            p75 = np.nanpercentile(valid_curves, 75, axis=0)
            color = cmap(ti)
            ax.plot(x_vals, median_c, color=color, linewidth=1.5,
                    label=labels[ti], alpha=0.9)
            ax.fill_between(x_vals, p25, p75, color=color, alpha=0.08)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{curve_type.replace("_", " ").title()}: '
                     f'Median with IQR across {n_frames} frames',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, ncol=3, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(sec_dir / f'{curve_type}_overlay.png', dpi=150,
                    bbox_inches='tight')
        plt.close()

        # -- L2 distance heatmap --
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(l2_mean, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(n_transforms))
        ax.set_yticks(range(n_transforms))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        plt.colorbar(im, ax=ax, label='Mean L2 Distance')
        ax.set_title(f'{curve_type.replace("_", " ").title()}: '
                     f'Pairwise L2 Distance',
                     fontsize=13, fontweight='bold')
        for i in range(n_transforms):
            for j in range(n_transforms):
                if not np.isnan(l2_mean[i, j]):
                    color = 'white' if l2_mean[i, j] > np.nanmax(l2_mean) * 0.6 \
                        else 'black'
                    ax.text(j, i, f'{l2_mean[i, j]:.3f}', ha='center',
                            va='center', fontsize=6, color=color)
        plt.tight_layout()
        plt.savefig(sec_dir / f'{curve_type}_l2_heatmap.png', dpi=150,
                    bbox_inches='tight')
        plt.close()

    # -- FPCA biplot (hypsometric) --
    if 'fpca_hypsometric' in results and results['fpca_hypsometric']['global']:
        fpca_g = results['fpca_hypsometric']['global']
        scores = fpca_g['scores']
        flabels = fpca_g['labels']
        ve = fpca_g['variance_explained']

        fig, ax = plt.subplots(figsize=(10, 8))
        unique_transforms = list(dict.fromkeys(flabels))
        cmap = plt.cm.get_cmap('tab20', len(unique_transforms))
        color_map = {t: cmap(i) for i, t in enumerate(unique_transforms)}

        for t in unique_transforms:
            mask = [l == t for l in flabels]
            s = scores[mask]
            display = TRANSFORM_LABELS.get(t, t)
            ax.scatter(s[:, 0], s[:, 1], color=color_map[t], label=display,
                       alpha=0.5, s=20, edgecolors='none')
            ax.scatter(np.mean(s[:, 0]), np.mean(s[:, 1]),
                       color=color_map[t], s=100, marker='x', linewidths=2)
        ax.set_xlabel(f'FPC 1 ({ve[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'FPC 2 ({ve[1]:.1%} variance)', fontsize=12)
        ax.set_title('Functional PCA: Hypsometric Curves\n'
                     '(points = frames, x = transform centroid)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, ncol=3, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(sec_dir / 'fpca_hypsometric_biplot.png', dpi=150,
                    bbox_inches='tight')
        plt.close()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C: Surface-to-Surface Comparison
# ═══════════════════════════════════════════════════════════════════════════

def surface_comparison_analysis(fdata, output_dir):
    """Analyze pairwise SSIM and correlation matrices (Section C)."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION C: SURFACE-TO-SURFACE COMPARISON")
    logger.info("=" * 80)

    transform_names = fdata['transform_names']
    labels = get_labels(transform_names)
    sec_dir = output_dir / "C_surface"
    sec_dir.mkdir(parents=True, exist_ok=True)

    ssim_matrices = fdata['ssim_matrices']    # (n_frames, 17, 17)
    corr_matrices = fdata['corr_matrices']    # (n_frames, 17, 17)
    n_frames = ssim_matrices.shape[0]
    n_transforms = len(transform_names)

    results = {}

    # -- Average SSIM and correlation matrices --
    mean_ssim = np.nanmean(ssim_matrices, axis=0)
    std_ssim = np.nanstd(ssim_matrices, axis=0)
    mean_corr = np.nanmean(corr_matrices, axis=0)
    std_corr = np.nanstd(corr_matrices, axis=0)

    results['mean_ssim'] = mean_ssim
    results['mean_corr'] = mean_corr

    # Save matrices
    pd.DataFrame(mean_ssim, index=labels, columns=labels).to_csv(
        sec_dir / 'mean_ssim_matrix.csv')
    pd.DataFrame(mean_corr, index=labels, columns=labels).to_csv(
        sec_dir / 'mean_correlation_matrix.csv')

    logger.info("\n  Average SSIM range: [%.3f, %.3f]",
                np.nanmin(mean_ssim), np.nanmax(mean_ssim[mean_ssim < 1.0]))
    logger.info("  Average Corr range: [%.3f, %.3f]",
                np.nanmin(mean_corr), np.nanmax(mean_corr[mean_corr < 1.0]))

    # -- Hierarchical clustering on SSIM --
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform

    # Convert SSIM similarity to distance
    ssim_dist = 1.0 - mean_ssim
    np.fill_diagonal(ssim_dist, 0)
    ssim_dist = np.clip(ssim_dist, 0, None)
    ssim_dist = (ssim_dist + ssim_dist.T) / 2  # ensure symmetry

    try:
        condensed = squareform(ssim_dist)
        Z = linkage(condensed, method='ward')
        results['ssim_linkage'] = Z

        # -- SSIM dendrogram --
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9)
        ax.set_title('Transform Clustering by Surface Similarity (SSIM)',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Distance (1 - SSIM)', fontsize=11)
        plt.tight_layout()
        plt.savefig(sec_dir / 'ssim_dendrogram.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Extract clusters (k=4 for A/B/C + outliers)
        n_clusters = min(4, n_transforms - 1)
        cluster_ids = fcluster(Z, n_clusters, criterion='maxclust')
        results['ssim_clusters'] = dict(zip(transform_names, cluster_ids.tolist()))
        logger.info("\n  SSIM-based clusters (k=%d):", n_clusters)
        for c in range(1, n_clusters + 1):
            members = [labels[i] for i in range(n_transforms)
                       if cluster_ids[i] == c]
            logger.info("    Cluster %d: %s", c, ', '.join(members))
    except Exception as e:
        logger.warning("Clustering failed: %s", e)

    # -- SSIM heatmap --
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mean_ssim, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(n_transforms))
    ax.set_yticks(range(n_transforms))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Mean SSIM')
    ax.set_title('Average Surface Structural Similarity (SSIM)',
                 fontsize=13, fontweight='bold')
    for i in range(n_transforms):
        for j in range(n_transforms):
            val = mean_ssim[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)
    plt.tight_layout()
    plt.savefig(sec_dir / 'ssim_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # -- Correlation heatmap --
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mean_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(n_transforms))
    ax.set_yticks(range(n_transforms))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Mean Pearson Correlation')
    ax.set_title('Average Spatial Correlation Between Surfaces',
                 fontsize=13, fontweight='bold')
    for i in range(n_transforms):
        for j in range(n_transforms):
            val = mean_corr[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)
    plt.tight_layout()
    plt.savefig(sec_dir / 'correlation_heatmap.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    # -- Temporal SSIM stability --
    # For each pair, compute CV of SSIM across frames
    ssim_cv = np.full((n_transforms, n_transforms), np.nan)
    for i in range(n_transforms):
        for j in range(i + 1, n_transforms):
            pairwise = ssim_matrices[:, i, j]
            pairwise = pairwise[~np.isnan(pairwise)]
            if len(pairwise) > 5 and np.mean(pairwise) != 0:
                cv = np.std(pairwise) / abs(np.mean(pairwise))
                ssim_cv[i, j] = cv
                ssim_cv[j, i] = cv
    results['ssim_temporal_cv'] = ssim_cv

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(ssim_cv, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_transforms))
    ax.set_yticks(range(n_transforms))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='CV of SSIM Across Frames')
    ax.set_title('Temporal Stability of Pairwise Surface Similarity',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(sec_dir / 'ssim_temporal_stability.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION D: Distributional Comparison
# ═══════════════════════════════════════════════════════════════════════════

def distributional_comparison(fdata, output_dir):
    """KS tests, Wasserstein distances, and distribution shape (Section D)."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION D: DISTRIBUTIONAL COMPARISON")
    logger.info("=" * 80)

    # scipy stats: KS/Wasserstein/skew/kurtosis computed vectorized below

    transform_names = fdata['transform_names']
    labels = get_labels(transform_names)
    sec_dir = output_dir / "D_distributional"
    sec_dir.mkdir(parents=True, exist_ok=True)

    histograms = fdata['pixel_histograms']  # (n_frames, n_transforms, n_bins)
    bin_centers = fdata['hist_bin_centers']
    n_frames, n_transforms, n_bins = histograms.shape

    results = {}

    # -- Pairwise KS statistic and Wasserstein distance (from histograms) --
    # Vectorized over frames: inner loop is only 136 iterations (17 choose 2),
    # each processing all frames via numpy CDF operations.
    logger.info("  Computing pairwise KS and Wasserstein distances...")

    ks_matrices = np.full((n_frames, n_transforms, n_transforms), np.nan)
    wass_matrices = np.full((n_frames, n_transforms, n_transforms), np.nan)

    bin_width = (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0

    for i in range(n_transforms):
        for j in range(i + 1, n_transforms):
            h1 = histograms[:, i, :]  # (n_frames, n_bins)
            h2 = histograms[:, j, :]

            # Valid frames: no NaN, non-zero sum
            valid = (~np.any(np.isnan(h1), axis=1)
                     & ~np.any(np.isnan(h2), axis=1))
            h1_sum = np.sum(h1, axis=1)
            h2_sum = np.sum(h2, axis=1)
            valid = valid & (h1_sum > 0) & (h2_sum > 0)

            if not np.any(valid):
                continue

            # Normalize to PMFs (only valid frames)
            h1_norm = np.where(valid[:, None], h1 / h1_sum[:, None], 0.0)
            h2_norm = np.where(valid[:, None], h2 / h2_sum[:, None], 0.0)

            # CDFs
            cdf1 = np.cumsum(h1_norm, axis=1)
            cdf2 = np.cumsum(h2_norm, axis=1)
            cdf_diff = np.abs(cdf1 - cdf2)

            # KS statistic: max |CDF1 - CDF2|
            ks_stat = np.max(cdf_diff, axis=1)
            ks_stat[~valid] = np.nan
            ks_matrices[:, i, j] = ks_stat
            ks_matrices[:, j, i] = ks_stat

            # Wasserstein-1: integral of |CDF1 - CDF2|
            w = np.sum(cdf_diff, axis=1) * bin_width
            w[~valid] = np.nan
            wass_matrices[:, i, j] = w
            wass_matrices[:, j, i] = w

    mean_ks = np.nanmean(ks_matrices, axis=0)
    mean_wass = np.nanmean(wass_matrices, axis=0)
    results['mean_ks'] = mean_ks
    results['mean_wasserstein'] = mean_wass

    pd.DataFrame(mean_ks, index=labels, columns=labels).to_csv(
        sec_dir / 'mean_ks_distance_matrix.csv')
    pd.DataFrame(mean_wass, index=labels, columns=labels).to_csv(
        sec_dir / 'mean_wasserstein_distance_matrix.csv')

    # -- Distribution shape characterization --
    logger.info("  Computing distribution shape metrics...")
    shape_metrics = {}
    for ti, tname in enumerate(transform_names):
        frame_skew = []
        frame_kurt = []
        frame_bimod = []
        for fi in range(n_frames):
            h = histograms[fi, ti, :]
            if np.any(np.isnan(h)) or np.sum(h) == 0:
                continue
            # Reconstruct approximate distribution from histogram
            pmf = h / np.sum(h)
            # Compute moments from PMF
            mu = np.sum(bin_centers * pmf)
            var = np.sum((bin_centers - mu)**2 * pmf)
            std = np.sqrt(var) if var > 0 else 1e-8
            sk = np.sum(((bin_centers - mu) / std)**3 * pmf) if std > 0 else 0
            ku = np.sum(((bin_centers - mu) / std)**4 * pmf) - 3 if std > 0 else 0
            # Bimodality coefficient: (skewness^2 + 1) / kurtosis_excess + 3)
            bc = (sk**2 + 1) / (ku + 3) if (ku + 3) > 0 else np.nan
            frame_skew.append(sk)
            frame_kurt.append(ku)
            frame_bimod.append(bc)

        if len(frame_skew) > 0:
            shape_metrics[tname] = {
                'skewness_mean': float(np.mean(frame_skew)),
                'skewness_std': float(np.std(frame_skew)),
                'kurtosis_mean': float(np.mean(frame_kurt)),
                'kurtosis_std': float(np.std(frame_kurt)),
                'bimodality_mean': float(np.nanmean(frame_bimod)),
                'bimodality_std': float(np.nanstd(frame_bimod)),
            }

    results['shape_metrics'] = shape_metrics
    shape_df = pd.DataFrame(shape_metrics).T
    shape_df.index = [TRANSFORM_LABELS.get(t, t) for t in shape_df.index]
    shape_df.to_csv(sec_dir / 'distribution_shape_metrics.csv')

    logger.info("\n  Distribution shape summary:")
    for t in sorted(shape_metrics, key=lambda x: abs(shape_metrics[x]['skewness_mean']),
                    reverse=True)[:5]:
        m = shape_metrics[t]
        logger.info("    %20s  skew=%+.3f  kurt=%+.3f  bimod=%.3f",
                    TRANSFORM_LABELS.get(t, t), m['skewness_mean'],
                    m['kurtosis_mean'], m['bimodality_mean'])

    # -- Visualizations --

    # KS distance heatmap
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mean_ks, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_transforms))
    ax.set_yticks(range(n_transforms))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Mean KS Statistic')
    ax.set_title('Pairwise KS Distance Between Pixel Distributions',
                 fontsize=13, fontweight='bold')
    for i in range(n_transforms):
        for j in range(n_transforms):
            if not np.isnan(mean_ks[i, j]) and i != j:
                color = 'white' if mean_ks[i, j] > np.nanmax(mean_ks) * 0.6 \
                    else 'black'
                ax.text(j, i, f'{mean_ks[i, j]:.2f}', ha='center',
                        va='center', fontsize=6, color=color)
    plt.tight_layout()
    plt.savefig(sec_dir / 'ks_distance_heatmap.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    # Wasserstein distance heatmap
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mean_wass, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_transforms))
    ax.set_yticks(range(n_transforms))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Mean Wasserstein Distance')
    ax.set_title('Pairwise Wasserstein Distance Between Pixel Distributions',
                 fontsize=13, fontweight='bold')
    for i in range(n_transforms):
        for j in range(n_transforms):
            if not np.isnan(mean_wass[i, j]) and i != j:
                color = 'white' if mean_wass[i, j] > np.nanmax(mean_wass) * 0.6 \
                    else 'black'
                ax.text(j, i, f'{mean_wass[i, j]:.3f}', ha='center',
                        va='center', fontsize=6, color=color)
    plt.tight_layout()
    plt.savefig(sec_dir / 'wasserstein_distance_heatmap.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    # Distribution overlay (average histograms)
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.get_cmap('tab20', n_transforms)
    for ti in range(n_transforms):
        mean_hist = np.nanmean(histograms[:, ti, :], axis=0)
        if np.any(np.isnan(mean_hist)):
            continue
        ax.plot(bin_centers, mean_hist, color=cmap(ti), linewidth=1.2,
                label=labels[ti], alpha=0.8)
    ax.set_xlabel('Normalized Intensity', fontsize=12)
    ax.set_ylabel('Probability Density (avg across frames)', fontsize=12)
    ax.set_title('Average Pixel Intensity Distribution by Transform',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(sec_dir / 'distribution_overlay.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION E: Enhanced Hypothesis Testing
# ═══════════════════════════════════════════════════════════════════════════

def enhanced_hypothesis_testing(df, transform_names, output_dir):
    """Friedman + Kendall's W, Nemenyi post-hoc, Cliff's delta, FDR (Section E)."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION E: ENHANCED HYPOTHESIS TESTING")
    logger.info("=" * 80)

    labels = get_labels(transform_names)
    sec_dir = output_dir / "E_hypothesis"
    sec_dir.mkdir(parents=True, exist_ok=True)

    # Metrics to test (from H1, H2, H3)
    test_metrics = {
        'H1_contour_cv':       ('polar_contour_cv_radius', 'lower'),
        'H1_fourier_rough':    ('polar_fourier_roughness', 'lower'),
        'H1_circularity':      ('polar_contour_circularity', 'higher'),
        'H3_grad_median':      ('grad_median', 'higher'),
        'H3_zone_width':       ('zone_radial_width', 'lower'),
        'H3_steepness':        ('hypso_steepness', 'higher'),
        'hypso_auc':           ('hypso_auc', None),
        'hypso_transition_t':  ('hypso_transition_t', None),
        'roughness':           ('roughness', None),
        'mean_z':              ('mean_z', None),
        'tda_entropy':         ('tda_persistence_entropy', None),
        'tda_max_pers':        ('tda_max_persistence', None),
    }

    results = {
        'friedman': {},
        'posthoc': {},
        'effect_sizes': {},
    }

    all_p_values = []
    all_p_labels = []

    for test_name, (metric, direction) in test_metrics.items():
        # Build data matrix: rows=frames, columns=transforms
        data = {}
        for t in transform_names:
            col = f"{t}__{metric}"
            if col in df.columns:
                data[t] = pd.to_numeric(df[col], errors='coerce')

        if len(data) < 3:
            continue

        data_df = pd.DataFrame(data).dropna()
        if len(data_df) < 10:
            continue

        n_subjects = len(data_df)
        n_groups = len(data_df.columns)

        # -- Friedman test with Kendall's W --
        try:
            import pingouin as pg

            # Reshape to long format for pingouin
            long_data = []
            for idx, row in data_df.iterrows():
                for t_col in data_df.columns:
                    long_data.append({
                        'frame': idx,
                        'transform': t_col,
                        'value': row[t_col]
                    })
            long_df = pd.DataFrame(long_data)

            friedman = pg.friedman(data=long_df, dv='value',
                                  within='transform', subject='frame')
            chi2 = float(friedman['Q'].iloc[0])
            p_val = float(friedman['p-unc'].iloc[0])

            # Kendall's W = chi2 / (n_subjects * (n_groups - 1))
            kendall_w = chi2 / (n_subjects * (n_groups - 1)) \
                if n_subjects > 0 and n_groups > 1 else np.nan

            results['friedman'][test_name] = {
                'chi2': chi2,
                'p_value': p_val,
                'kendall_w': kendall_w,
                'n_frames': n_subjects,
                'n_transforms': n_groups,
            }
            all_p_values.append(p_val)
            all_p_labels.append(f'friedman_{test_name}')

            sig = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                   else "*" if p_val < 0.05 else "ns")
            w_interp = ("large" if kendall_w > 0.5 else "medium"
                        if kendall_w > 0.3 else "small")
            logger.info("  %25s  chi2=%9.2f  p=%.2e %s  W=%.3f (%s)",
                        test_name, chi2, p_val, sig, kendall_w, w_interp)

        except ImportError:
            logger.warning("pingouin not installed, using scipy Friedman")
            from scipy.stats import friedmanchisquare
            arrays = [data_df[t].values for t in data_df.columns]
            stat, p_val = friedmanchisquare(*arrays)
            results['friedman'][test_name] = {
                'chi2': float(stat), 'p_value': float(p_val),
                'n_frames': n_subjects, 'n_transforms': n_groups,
            }
            all_p_values.append(p_val)
            all_p_labels.append(f'friedman_{test_name}')

        # -- Nemenyi post-hoc test --
        try:
            import scikit_posthocs as sp
            nemenyi = sp.posthoc_nemenyi_friedman(data_df.values)
            nemenyi.index = [TRANSFORM_LABELS.get(c, c) for c in data_df.columns]
            nemenyi.columns = nemenyi.index
            results['posthoc'][test_name] = nemenyi

            nemenyi.to_csv(sec_dir / f'nemenyi_{test_name}.csv')

            # Collect pairwise p-values for FDR
            for i in range(len(nemenyi)):
                for j in range(i + 1, len(nemenyi)):
                    pv = nemenyi.iloc[i, j]
                    if not np.isnan(pv):
                        all_p_values.append(pv)
                        all_p_labels.append(
                            f'nemenyi_{test_name}_{nemenyi.index[i]}_vs_{nemenyi.columns[j]}')
        except ImportError:
            logger.warning("scikit-posthocs not installed, skipping Nemenyi")

        # -- Cliff's delta (pairwise effect sizes) --
        try:
            import pingouin as pg
            n_t = len(data_df.columns)
            cliff_matrix = np.full((n_t, n_t), np.nan)
            t_cols = list(data_df.columns)
            for i in range(n_t):
                for j in range(i + 1, n_t):
                    try:
                        ef = pg.compute_effsize(data_df[t_cols[i]],
                                                data_df[t_cols[j]],
                                                eftype='CLES')
                        # Convert CLES to Cliff's delta: delta = 2*CLES - 1
                        cliff_d = 2 * ef - 1
                        cliff_matrix[i, j] = cliff_d
                        cliff_matrix[j, i] = -cliff_d
                    except Exception as e:
                        logger.debug("Cliff delta failed for %s vs %s: %s", t_cols[i], t_cols[j], e)
                        pass

            cliff_labels = [TRANSFORM_LABELS.get(c, c) for c in t_cols]
            cliff_df = pd.DataFrame(cliff_matrix, index=cliff_labels,
                                    columns=cliff_labels)
            results['effect_sizes'][test_name] = cliff_df
            cliff_df.to_csv(sec_dir / f'cliff_delta_{test_name}.csv')
        except ImportError:
            pass

    # -- FDR correction (Benjamini-Hochberg) --
    logger.info("\n  FDR Correction (Benjamini-Hochberg)")
    if len(all_p_values) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            reject, p_adj, _, _ = multipletests(
                all_p_values, alpha=0.05, method='fdr_bh')
            fdr_results = pd.DataFrame({
                'test': all_p_labels,
                'p_raw': all_p_values,
                'p_adjusted': p_adj,
                'reject_H0': reject,
            })
            fdr_results.to_csv(sec_dir / 'fdr_correction.csv', index=False)
            results['fdr'] = fdr_results

            n_reject = int(np.sum(reject))
            logger.info("  %d/%d tests significant after FDR correction",
                        n_reject, len(all_p_values))

            # Friedman-level FDR summary
            friedman_mask = [l.startswith('friedman_') for l in all_p_labels]
            friedman_fdr = fdr_results[friedman_mask]
            logger.info("\n  Friedman tests after FDR:")
            for _, row in friedman_fdr.iterrows():
                sig = "SIGNIFICANT" if row['reject_H0'] else "ns"
                logger.info("    %40s  p_adj=%.2e  %s",
                            row['test'], row['p_adjusted'], sig)

        except ImportError:
            logger.warning("statsmodels not installed, skipping FDR")

    # -- Critical difference diagram --
    try:
        import scikit_posthocs as sp
        # Use H3 steepness as the primary metric for CD diagram
        cd_metric = 'hypso_steepness'
        cd_data = {}
        for t in transform_names:
            col = f"{t}__{cd_metric}"
            if col in df.columns:
                cd_data[TRANSFORM_LABELS.get(t, t)] = pd.to_numeric(
                    df[col], errors='coerce')
        cd_df = pd.DataFrame(cd_data).dropna()

        if len(cd_df) >= 10 and len(cd_df.columns) >= 3:
            # Rank data (higher steepness = better = rank 1)
            ranks = cd_df.rank(axis=1, ascending=False)
            avg_ranks = ranks.mean()

            fig, ax = plt.subplots(figsize=(14, 6))
            try:
                sp.critical_difference_diagram(
                    ranks=avg_ranks, sig_matrix=results['posthoc'].get(
                        'H3_steepness'),
                    ax=ax)
            except Exception:
                # Manual CD diagram if the library method fails
                logger.info("Using manual CD diagram (library method unavailable)")
                sorted_ranks = avg_ranks.sort_values()
                y_positions = np.arange(len(sorted_ranks))
                ax.barh(y_positions, sorted_ranks, color='steelblue', alpha=0.7)
                ax.set_yticks(y_positions)
                ax.set_yticklabels(sorted_ranks.index, fontsize=9)
                ax.set_xlabel('Average Rank (lower = better)', fontsize=11)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')

            ax.set_title('Critical Difference Diagram: Transition Steepness',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(sec_dir / 'critical_difference_diagram.png', dpi=150,
                        bbox_inches='tight')
            plt.close()
    except Exception as e:
        logger.warning("CD diagram failed: %s", e)

    # -- Pairwise effect size heatmap (Cliff's delta for steepness) --
    if 'H3_steepness' in results['effect_sizes']:
        cliff_df = results['effect_sizes']['H3_steepness']
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(cliff_df.values, cmap='RdBu_r', aspect='auto',
                       vmin=-1, vmax=1)
        ax.set_xticks(range(len(cliff_df)))
        ax.set_yticks(range(len(cliff_df)))
        ax.set_xticklabels(cliff_df.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(cliff_df.index, fontsize=8)
        plt.colorbar(im, ax=ax, label="Cliff's Delta")
        ax.set_title("Pairwise Effect Sizes (Cliff's Delta): Transition Steepness",
                     fontsize=13, fontweight='bold')
        for i in range(len(cliff_df)):
            for j in range(len(cliff_df)):
                val = cliff_df.iloc[i, j]
                if not np.isnan(val):
                    mag = ("L" if abs(val) > 0.474 else "M"
                           if abs(val) > 0.33 else "S" if abs(val) > 0.147
                           else "")
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}\n{mag}', ha='center',
                            va='center', fontsize=5, color=color)
        plt.tight_layout()
        plt.savefig(sec_dir / 'cliff_delta_heatmap.png', dpi=150,
                    bbox_inches='tight')
        plt.close()

    # Save Friedman summary
    if results['friedman']:
        friedman_df = pd.DataFrame(results['friedman']).T
        friedman_df.to_csv(sec_dir / 'friedman_summary.csv')

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION F: Temporal Analysis
# ═══════════════════════════════════════════════════════════════════════════

def temporal_analysis(df, fdata, transform_names, output_dir):
    """ACF, effective sample size, subsampled validation (Section F)."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION F: TEMPORAL ANALYSIS")
    logger.info("=" * 80)

    labels = get_labels(transform_names)
    sec_dir = output_dir / "F_temporal"
    sec_dir.mkdir(parents=True, exist_ok=True)

    results = {'acf': {}, 'effective_n': {}, 'subsampled': {}}

    # Key metrics to analyze temporally
    temporal_metrics = [
        'hypso_auc', 'hypso_steepness', 'hypso_transition_t',
        'grad_median', 'roughness', 'mean_z', 'median_z',
        'polar_contour_cv_radius', 'polar_half_decay_dist',
    ]

    # -- ACF computation --
    logger.info("\n  Computing autocorrelation functions...")
    max_lag = min(50, len(df) // 3)

    acf_results = {}
    effective_n_results = {}

    for metric in temporal_metrics:
        acf_results[metric] = {}
        effective_n_results[metric] = {}

        for t in transform_names:
            col = f"{t}__{metric}"
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) < 20:
                continue

            try:
                from statsmodels.tsa.stattools import acf as compute_acf
                acf_vals, confint = compute_acf(
                    series.values, nlags=max_lag, alpha=0.05, fft=True)
                acf_results[metric][t] = acf_vals

                # Effective sample size: n_eff = n / (1 + 2*sum(acf))
                # Sum positive ACF values until first negative
                n = len(series)
                acf_sum = 0
                for lag in range(1, len(acf_vals)):
                    if acf_vals[lag] < 0:
                        break
                    acf_sum += acf_vals[lag]
                n_eff = n / (1 + 2 * acf_sum)
                n_eff = max(1, min(n, n_eff))
                effective_n_results[metric][t] = {
                    'n_total': n,
                    'n_effective': float(n_eff),
                    'reduction_ratio': float(n_eff / n),
                    'acf_lag1': float(acf_vals[1]) if len(acf_vals) > 1 else np.nan,
                }
            except ImportError:
                # Fallback: manual ACF
                vals = series.values
                n = len(vals)
                mu = np.mean(vals)
                var = np.var(vals)
                if var == 0:
                    continue
                acf_vals = np.zeros(max_lag + 1)
                acf_vals[0] = 1.0
                for lag in range(1, max_lag + 1):
                    if lag >= n:
                        break
                    acf_vals[lag] = np.mean((vals[:n-lag] - mu) * (vals[lag:] - mu)) / var
                acf_results[metric][t] = acf_vals

                acf_sum = 0
                for lag in range(1, len(acf_vals)):
                    if acf_vals[lag] < 0:
                        break
                    acf_sum += acf_vals[lag]
                n_eff = n / (1 + 2 * acf_sum)
                n_eff = max(1, min(n, n_eff))
                effective_n_results[metric][t] = {
                    'n_total': n,
                    'n_effective': float(n_eff),
                    'reduction_ratio': float(n_eff / n),
                    'acf_lag1': float(acf_vals[1]) if len(acf_vals) > 1 else np.nan,
                }

    results['acf'] = acf_results
    results['effective_n'] = effective_n_results

    # -- Print effective sample size summary --
    logger.info("\n  Effective Sample Size Summary (metric: median across transforms):")
    for metric in temporal_metrics:
        if metric in effective_n_results and effective_n_results[metric]:
            ratios = [v['reduction_ratio']
                      for v in effective_n_results[metric].values()
                      if not np.isnan(v['reduction_ratio'])]
            n_effs = [v['n_effective']
                      for v in effective_n_results[metric].values()
                      if not np.isnan(v['n_effective'])]
            lag1s = [v['acf_lag1']
                     for v in effective_n_results[metric].values()
                     if not np.isnan(v['acf_lag1'])]
            if ratios:
                n_total = list(effective_n_results[metric].values())[0]['n_total']
                logger.info("    %30s  n=%d  n_eff=%.1f  ratio=%.2f%%  ACF(1)=%.3f",
                            metric, n_total, np.median(n_effs),
                            np.median(ratios) * 100, np.median(lag1s))

    # Save effective N table
    eff_n_rows = []
    for metric in temporal_metrics:
        if metric not in effective_n_results:
            continue
        for t, vals in effective_n_results[metric].items():
            row = {'metric': metric, 'transform': TRANSFORM_LABELS.get(t, t)}
            row.update(vals)
            eff_n_rows.append(row)
    if eff_n_rows:
        eff_n_df = pd.DataFrame(eff_n_rows)
        eff_n_df.to_csv(sec_dir / 'effective_sample_sizes.csv', index=False)

    # -- Subsampled validation --
    logger.info("\n  Subsampled Friedman validation...")
    subsample_factors = [2, 5, 10]
    for factor in subsample_factors:
        sub_df = df.iloc[::factor].copy()
        if len(sub_df) < 15:
            logger.info("    Factor %d: too few frames (%d), skipping", factor, len(sub_df))
            continue

        # Run Friedman on hypso_steepness
        data = {}
        for t in transform_names:
            col = f"{t}__hypso_steepness"
            if col in sub_df.columns:
                data[t] = pd.to_numeric(sub_df[col], errors='coerce')
        data_sub = pd.DataFrame(data).dropna()
        if len(data_sub) < 10:
            continue

        from scipy.stats import friedmanchisquare
        arrays = [data_sub[t].values for t in data_sub.columns]
        try:
            stat, p = friedmanchisquare(*arrays)
            sig = ("***" if p < 0.001 else "**" if p < 0.01
                   else "*" if p < 0.05 else "ns")
            results['subsampled'][factor] = {
                'chi2': float(stat), 'p_value': float(p),
                'n_frames': len(data_sub)
            }
            logger.info("    Every-%d-th frame (n=%3d):  chi2=%9.2f  p=%.2e %s",
                        factor, len(data_sub), stat, p, sig)
        except Exception as e:
            logger.warning("Friedman subsampled (factor=%d) failed: %s", factor, e)
            pass

    # -- ACF plot --
    metric_to_plot = 'hypso_steepness'
    if metric_to_plot in acf_results and acf_results[metric_to_plot]:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        plot_idx = 0
        for t in transform_names:
            if t not in acf_results[metric_to_plot]:
                continue
            if plot_idx >= 9:
                break
            acf_vals = acf_results[metric_to_plot][t]
            ax = axes[plot_idx]
            lags = np.arange(len(acf_vals))
            ax.bar(lags, acf_vals, color='steelblue', alpha=0.7, width=0.8)
            ax.axhline(0, color='black', linewidth=0.5)
            # Approximate 95% confidence band
            eff_entry = effective_n_results.get(metric_to_plot, {}).get(t, {})
            n = eff_entry.get('n_total', 100) if isinstance(eff_entry, dict) else 100
            ci = 1.96 / np.sqrt(max(n, 1))
            ax.axhline(ci, color='red', linestyle='--', alpha=0.5)
            ax.axhline(-ci, color='red', linestyle='--', alpha=0.5)
            ax.set_title(TRANSFORM_LABELS.get(t, t), fontsize=10)
            ax.set_xlim(-0.5, max_lag + 0.5)
            ax.set_ylim(-0.3, 1.05)
            plot_idx += 1

        # Hide unused axes
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'Autocorrelation: {metric_to_plot.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        fig.supxlabel('Lag (frames at 5fps)', fontsize=11)
        fig.supylabel('ACF', fontsize=11)
        plt.tight_layout()
        plt.savefig(sec_dir / 'acf_steepness.png', dpi=150, bbox_inches='tight')
        plt.close()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION G: Transform Characterization
# ═══════════════════════════════════════════════════════════════════════════

def transform_characterization(df, fdata, transform_names, output_dir):
    """Multi-dimensional profiling, clustering, discriminant features (Section G)."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION G: TRANSFORM CHARACTERIZATION")
    logger.info("=" * 80)

    labels = get_labels(transform_names)
    sec_dir = output_dir / "G_characterization"
    sec_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Metrics that define each transform's "fingerprint"
    profile_metrics = [
        ('hypso_auc', 'Hypsometric AUC'),
        ('hypso_steepness', 'Transition Steepness'),
        ('hypso_transition_t', 'Transition Threshold'),
        ('roughness', 'Surface Roughness'),
        ('grad_median', 'Gradient Median'),
        ('polar_contour_cv_radius', 'Contour CV'),
        ('polar_contour_circularity', 'Circularity'),
        ('polar_half_decay_dist', 'Half-Decay Distance'),
        ('polar_fourier_roughness', 'Fourier Roughness'),
        ('mean_z', 'Mean Intensity'),
    ]

    # -- Build profile matrix --
    profile_data = {}
    for t in transform_names:
        row = {}
        for metric, label in profile_metrics:
            col = f"{t}__{metric}"
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                row[label] = float(vals.median()) if len(vals) > 0 else np.nan
            else:
                row[label] = np.nan
        profile_data[t] = row

    profile_df = pd.DataFrame(profile_data).T
    profile_df.index = labels

    # Z-score normalization for comparability
    profile_z = (profile_df - profile_df.mean()) / profile_df.std()
    profile_z = profile_z.fillna(0)

    results['profile_df'] = profile_df
    results['profile_z'] = profile_z
    profile_df.to_csv(sec_dir / 'transform_profiles.csv')
    profile_z.to_csv(sec_dir / 'transform_profiles_zscore.csv')

    # -- Radar/spider plot --
    n_metrics = len(profile_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(2, 3, figsize=(18, 12),
                              subplot_kw=dict(polar=True))
    axes = axes.ravel()

    # Group transforms by category
    cat_groups = {'A': [], 'B': [], 'C': []}
    for t in transform_names:
        cat = TRANSFORM_CATEGORIES.get(t, 'A')
        cat_groups[cat].append(t)

    plot_groups = [
        ('Cat A: RGB Ratios (top 5)', cat_groups['A'][:5]),
        ('Cat A: RGB Ratios (rest)', cat_groups['A'][5:]),
        ('Cat B: Perceptual', cat_groups['B']),
        ('Cat C: Data-Driven', cat_groups['C']),
        ('Top 5 Overall', None),  # Will be filled
        ('All Transforms', transform_names),
    ]

    cmap = plt.cm.get_cmap('tab10')
    for idx, (title, group) in enumerate(plot_groups):
        if idx >= len(axes):
            break
        ax = axes[idx]

        if group is None:
            # Top 5 by median steepness
            steep_col = 'Transition Steepness'
            if steep_col in profile_df.columns:
                top5 = profile_df[steep_col].nlargest(5).index.tolist()
                group_labels = top5
            else:
                continue
        else:
            group_labels = [TRANSFORM_LABELS.get(t, t) for t in group]

        for gi, glabel in enumerate(group_labels):
            if glabel not in profile_z.index:
                continue
            values = profile_z.loc[glabel].values.tolist()
            values += values[:1]
            color = cmap(gi % 10)
            ax.plot(angles, values, linewidth=1.5, label=glabel, color=color)
            ax.fill(angles, values, alpha=0.05, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[1] for m in profile_metrics], fontsize=6)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=15)
        ax.legend(fontsize=6, loc='upper right',
                  bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(sec_dir / 'radar_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()

    # -- Hierarchical clustering by metric profile --
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import pdist

    try:
        distances = pdist(profile_z.values, metric='euclidean')
        Z = linkage(distances, method='ward')
        results['profile_linkage'] = Z

        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9)
        ax.set_title('Transform Clustering by Metric Profile (Ward Linkage)',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Euclidean Distance (z-scored metrics)', fontsize=11)
        plt.tight_layout()
        plt.savefig(sec_dir / 'profile_dendrogram.png', dpi=150,
                    bbox_inches='tight')
        plt.close()

        # Cluster assignment
        n_clusters = min(4, len(transform_names) - 1)
        cluster_ids = fcluster(Z, n_clusters, criterion='maxclust')
        cluster_map = dict(zip(labels, cluster_ids.tolist()))
        results['profile_clusters'] = cluster_map

        logger.info("\n  Metric-based clusters (k=%d):", n_clusters)
        for c in range(1, n_clusters + 1):
            members = [l for l, cid in cluster_map.items() if cid == c]
            # Describe cluster by z-scored means
            cluster_mask = profile_z.index.isin(members)
            cluster_means = profile_z[cluster_mask].mean()
            top_features = cluster_means.abs().nlargest(3)
            desc = ', '.join(
                '%s=%+.2f' % (f, cluster_means[f])
                for f in top_features.index
            )
            logger.info("    Cluster %d: %s", c, ', '.join(members))
            logger.info("      Distinguishing: %s", desc)

    except Exception as e:
        logger.warning("Profile clustering failed: %s", e)

    # -- Discriminant features between clusters --
    if 'profile_clusters' in results:
        logger.info("\n  Discriminant features between clusters:")
        cluster_ids = results['profile_clusters']
        unique_clusters = sorted(set(cluster_ids.values()))

        discrim_results = {}
        for ci in unique_clusters:
            for cj in unique_clusters:
                if ci >= cj:
                    continue
                members_i = [l for l, c in cluster_ids.items() if c == ci]
                members_j = [l for l, c in cluster_ids.items() if c == cj]
                if len(members_i) < 2 or len(members_j) < 2:
                    continue

                diffs = {}
                for col in profile_z.columns:
                    vals_i = profile_z.loc[members_i, col].values
                    vals_j = profile_z.loc[members_j, col].values
                    diff = abs(np.mean(vals_i) - np.mean(vals_j))
                    diffs[col] = diff

                top_disc = sorted(diffs, key=diffs.get, reverse=True)[:3]
                pair_key = f'C{ci}_vs_C{cj}'
                discrim_results[pair_key] = {
                    f: diffs[f] for f in top_disc
                }
                logger.info("    %s: top discriminants = %s",
                            pair_key, ', '.join('%s (%.2f)' % (f, diffs[f]) for f in top_disc))

        results['discriminant_features'] = discrim_results

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION H: TDA Integration
# ═══════════════════════════════════════════════════════════════════════════

def tda_integration(df, transform_names, output_dir):
    """Integrate TDA metrics into the statistical framework (Section H)."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION H: TDA INTEGRATION")
    logger.info("=" * 80)

    labels = get_labels(transform_names)
    sec_dir = output_dir / "H_tda"
    sec_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    tda_metrics = [
        ('tda_persistence_entropy', 'Persistence Entropy'),
        ('tda_max_persistence', 'Max Persistence'),
        ('tda_n_features', 'N Features'),
        ('tda_total_persistence', 'Total Persistence'),
        ('tda_persistence_ratio', 'Persistence Ratio'),
    ]

    # Check which TDA metrics are available
    available_tda = []
    for metric, label in tda_metrics:
        sample_col = f"{transform_names[0]}__{metric}"
        if sample_col in df.columns:
            available_tda.append((metric, label))

    if not available_tda:
        logger.info("No TDA metrics found in CSV. Skipping TDA integration.")
        return results

    logger.info("Found %d TDA metrics", len(available_tda))

    # -- TDA profiles per transform --
    tda_profile = {}
    for t in transform_names:
        row = {}
        for metric, label in available_tda:
            col = f"{t}__{metric}"
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(vals) > 0:
                    row[f'{label} (median)'] = float(vals.median())
                    row[f'{label} (CV)'] = (float(vals.std() / abs(vals.mean()))
                                            if vals.mean() != 0 else np.nan)
        tda_profile[TRANSFORM_LABELS.get(t, t)] = row

    tda_df = pd.DataFrame(tda_profile).T
    tda_df.to_csv(sec_dir / 'tda_profiles.csv')
    results['tda_profiles'] = tda_df

    logger.info("\n  TDA Profile (top 5 by persistence entropy):")
    entropy_col = 'Persistence Entropy (median)'
    if entropy_col in tda_df.columns:
        top5 = tda_df[entropy_col].nlargest(5)
        for t, v in top5.items():
            logger.info("    %20s  entropy=%.4f", t, v)

    # -- Friedman tests on TDA metrics --
    logger.info("\n  Friedman tests on TDA metrics:")
    from scipy.stats import friedmanchisquare

    for metric, label in available_tda:
        data = {}
        for t in transform_names:
            col = f"{t}__{metric}"
            if col in df.columns:
                data[t] = pd.to_numeric(df[col], errors='coerce')
        data_df = pd.DataFrame(data).dropna()
        if len(data_df) < 10 or len(data_df.columns) < 3:
            continue

        arrays = [data_df[t].values for t in data_df.columns]
        try:
            stat, p = friedmanchisquare(*arrays)
            n = len(data_df)
            k = len(arrays)
            w = stat / (n * (k - 1)) if n > 0 and k > 1 else np.nan
            sig = ("***" if p < 0.001 else "**" if p < 0.01
                   else "*" if p < 0.05 else "ns")
            logger.info("    %30s  chi2=%9.2f  p=%.2e %s  W=%.3f",
                        label, stat, p, sig, w)
            results[f'friedman_{metric}'] = {
                'chi2': float(stat), 'p_value': float(p),
                'kendall_w': float(w), 'n': n
            }
        except Exception as e:
            logger.warning("Friedman TDA (%s) failed: %s", metric, e)
            pass

    # -- TDA box plots --
    fig, axes = plt.subplots(1, min(len(available_tda), 3),
                              figsize=(6 * min(len(available_tda), 3), 6))
    if len(available_tda) == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, available_tda[:3]):
        data = []
        for t in transform_names:
            col = f"{t}__{metric}"
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna().values
                data.append(vals)
            else:
                data.append([])
        ax.boxplot(data, labels=labels, vert=True)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.suptitle('TDA Metrics by Transform', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(sec_dir / 'tda_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary_report(all_results, transform_names, output_dir):
    """Generate a consolidated summary of all analyses."""
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("=" * 80)

    labels = get_labels(transform_names)
    report_lines = []
    report_lines.append("# Comprehensive Statistical Evaluation Report\n")

    # -- Section B: Functional analysis summary --
    report_lines.append("## B. Functional Curve Analysis\n")
    for curve_type in ['hypsometric', 'radial_profile', 'boundary_contour']:
        bd_key = f'{curve_type}_band_depth'
        if bd_key in all_results.get('functional', {}):
            bd = all_results['functional'][bd_key]
            report_lines.append(f"### {curve_type.replace('_', ' ').title()}\n")
            report_lines.append("Band depth ranking (higher = more central):\n")
            for i, t in enumerate(sorted(
                    bd, key=lambda x: bd[x]['mean'], reverse=True)):
                report_lines.append(
                    f"{i+1}. {TRANSFORM_LABELS.get(t, t)}: "
                    f"{bd[t]['mean']:.4f} +/- {bd[t]['std']:.4f}")
            report_lines.append("")

    # -- Section C: Surface comparison summary --
    if 'surface' in all_results:
        report_lines.append("## C. Surface Comparison\n")
        if 'ssim_clusters' in all_results['surface']:
            report_lines.append("SSIM-based clusters:\n")
            clusters = all_results['surface']['ssim_clusters']
            for c in sorted(set(clusters.values())):
                members = [TRANSFORM_LABELS.get(t, t)
                           for t, cid in clusters.items() if cid == c]
                report_lines.append(f"  Cluster {c}: {', '.join(members)}")
            report_lines.append("")

    # -- Section E: Hypothesis testing summary --
    if 'hypothesis' in all_results and 'friedman' in all_results['hypothesis']:
        report_lines.append("## E. Hypothesis Testing\n")
        report_lines.append("### Friedman Tests\n")
        report_lines.append("| Metric | chi2 | p | Kendall W | Interpretation |")
        report_lines.append("|--------|------|---|-----------|----------------|")
        for name, r in all_results['hypothesis']['friedman'].items():
            sig = ("***" if r['p_value'] < 0.001 else "**"
                   if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05
                   else "ns")
            w = r.get('kendall_w', np.nan)
            w_int = ("large" if w > 0.5 else "medium" if w > 0.3
                     else "small" if w > 0.1 else "negligible")
            report_lines.append(
                f"| {name} | {r['chi2']:.1f} | {r['p_value']:.2e} {sig} "
                f"| {w:.3f} | {w_int} |")
        report_lines.append("")

    # -- Section F: Temporal analysis summary --
    if 'temporal' in all_results and 'effective_n' in all_results['temporal']:
        report_lines.append("## F. Temporal Analysis\n")
        report_lines.append(
            "At 5fps, temporal autocorrelation significantly reduces "
            "effective sample size:\n")
        eff_n = all_results['temporal']['effective_n']
        for metric in eff_n:
            if eff_n[metric]:
                ratios = [v['reduction_ratio'] for v in eff_n[metric].values()]
                report_lines.append(
                    f"  {metric}: median effective ratio = "
                    f"{np.median(ratios):.1%}")
        report_lines.append("")

    # -- Section G: Characterization summary --
    if 'characterization' in all_results:
        char = all_results['characterization']
        if 'profile_clusters' in char:
            report_lines.append("## G. Transform Families\n")
            clusters = char['profile_clusters']
            for c in sorted(set(clusters.values())):
                members = [l for l, cid in clusters.items() if cid == c]
                report_lines.append(f"  Family {c}: {', '.join(members)}")
            report_lines.append("")

    # Write report
    report_path = output_dir / "evaluation_report.md"
    report_text = "\n".join(report_lines)
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info("  Report saved to: %s", report_path)

    # Also save all numeric results as JSON
    json_results = {}
    for section, data in all_results.items():
        if isinstance(data, dict):
            json_section = {}
            for key, val in data.items():
                if isinstance(val, np.ndarray):
                    json_section[key] = val.tolist()
                elif isinstance(val, pd.DataFrame):
                    json_section[key] = val.to_dict()
                elif isinstance(val, (int, float, str, bool, list)):
                    json_section[key] = val
                elif isinstance(val, dict):
                    # Nested dict -- try to serialize
                    sub = {}
                    for k2, v2 in val.items():
                        if isinstance(v2, (int, float, str, bool)):
                            sub[k2] = v2
                        elif isinstance(v2, dict):
                            sub[k2] = {
                                k3: (v3 if isinstance(v3, (int, float, str, bool))
                                     else str(v3))
                                for k3, v3 in v2.items()
                            }
                    json_section[key] = sub
            json_results[section] = json_section

    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    logger.info("  Results JSON saved to: %s", json_path)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive statistical evaluation of fluorescence transforms")
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to functional_data.npz")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to frame_metrics_summary.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results and figures")
    parser.add_argument("--stable-only", action="store_true", default=True)
    parser.add_argument("--all-frames", dest="stable_only", action="store_false")
    parser.add_argument("--skip-functional", action="store_true",
                        help="Skip Section B (functional analysis)")
    parser.add_argument("--skip-surface", action="store_true",
                        help="Skip Section C (surface comparison)")
    parser.add_argument("--skip-distributional", action="store_true",
                        help="Skip Section D (distributional comparison)")
    parser.add_argument("--skip-temporal", action="store_true",
                        help="Skip Section F (temporal analysis)")
    parser.add_argument("--skip-tda", action="store_true",
                        help="Skip Section H (TDA integration)")
    args = parser.parse_args()

    # Resolve paths
    npz_path = Path(args.npz) if args.npz else _DEFAULT_NPZ
    csv_path = Path(args.csv) if args.csv else _DEFAULT_CSV
    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT
    for name, val in [("--npz/--csv", npz_path if npz_path is not None else csv_path),
                      ("--output-dir", output_dir)]:
        if val is None:
            print(f"ERROR: {name} is required (or set ALA_WORKSPACE)", file=sys.stderr)
            sys.exit(1)

    setup_logging('comprehensive_statistical_evaluation', output_dir=output_dir)

    # Validate inputs
    has_npz = npz_path is not None and npz_path.exists()
    has_csv = csv_path is not None and csv_path.exists()

    if not has_npz and not has_csv:
        logger.error("Neither NPZ nor CSV found.")
        logger.error("  NPZ: %s", npz_path)
        logger.error("  CSV: %s", csv_path)
        logger.error("\nRun extract_functional_data.py first to generate the NPZ,")
        logger.error("and/or run_5fps_topographic_analysis.py to generate the CSV.")
        sys.exit(1)

    if not has_npz:
        logger.warning("NPZ not found at %s", npz_path)
        logger.warning("  Sections B, C, D (functional/surface/distributional) will be skipped.")
        logger.warning("  Run extract_functional_data.py first to enable these.\n")
        args.skip_functional = True
        args.skip_surface = True
        args.skip_distributional = True

    if not has_csv:
        logger.warning("CSV not found at %s", csv_path)
        logger.warning("  Sections E, F, G, H (hypothesis/temporal/characterization/TDA) will be skipped.\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    fdata = load_functional_data(npz_path) if has_npz else None
    df = load_scalar_metrics(csv_path, stable_only=args.stable_only) if has_csv else None

    # Determine transform names
    if fdata is not None:
        transform_names = fdata['transform_names']
    elif df is not None:
        # Infer from registry (single source of truth)
        from lib.fluorescence_transforms import TRANSFORM_REGISTRY
        transform_names = list(TRANSFORM_REGISTRY.keys())
    else:
        logger.error("No data loaded")
        sys.exit(1)

    all_results = {}

    # ── Section B: Functional Curve Analysis ──
    if not args.skip_functional and fdata is not None:
        all_results['functional'] = functional_curve_analysis(fdata, output_dir)

    # ── Section C: Surface Comparison ──
    if not args.skip_surface and fdata is not None:
        all_results['surface'] = surface_comparison_analysis(fdata, output_dir)

    # ── Section D: Distributional Comparison ──
    if not args.skip_distributional and fdata is not None:
        all_results['distributional'] = distributional_comparison(fdata, output_dir)

    # ── Section E: Enhanced Hypothesis Testing ──
    if df is not None:
        all_results['hypothesis'] = enhanced_hypothesis_testing(
            df, transform_names, output_dir)

    # ── Section F: Temporal Analysis ──
    if not args.skip_temporal and df is not None:
        all_results['temporal'] = temporal_analysis(
            df, fdata, transform_names, output_dir)

    # ── Section G: Transform Characterization ──
    if df is not None:
        all_results['characterization'] = transform_characterization(
            df, fdata, transform_names, output_dir)

    # ── Section H: TDA Integration ──
    if not args.skip_tda and df is not None:
        all_results['tda'] = tda_integration(df, transform_names, output_dir)

    # ── Summary Report ──
    generate_summary_report(all_results, transform_names, output_dir)

    logger.info("\n%s", "=" * 80)
    logger.info("COMPREHENSIVE STATISTICAL EVALUATION COMPLETE")
    logger.info("  Output directory: %s", output_dir)
    logger.info("%s", "=" * 80)


if __name__ == "__main__":
    main()
