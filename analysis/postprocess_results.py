#!/usr/bin/env python3
"""
Post-Processing and Statistical Comparison of Topographic Analysis Results
===========================================================================
Reads the frame_metrics_summary.csv and performs:
  1. Transformation ranking on key metrics (H1, H2, H3)
  2. Friedman tests with post-hoc pairwise comparisons
  3. Transformation correlation matrix
  4. Visualization: box plots, heatmaps, ranking charts

Usage:
    python scripts/03_analysis/postprocess_results.py
    python scripts/03_analysis/postprocess_results.py --csv path/to/frame_metrics_summary.csv
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
import warnings

from lib.utils import setup_logging
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='Maximum number of iterations')
warnings.filterwarnings('ignore', message='.*divide by zero.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')

# ── Key metrics mapped to hypotheses ──────────────────────────────────────

# H1: Within-frame boundary consistency (lower CV/roughness = better)
H1_METRICS = {
    'polar_contour_cv_radius':    {'direction': 'lower', 'label': 'Contour CV'},
    'polar_fourier_roughness':    {'direction': 'lower', 'label': 'Fourier Roughness'},
    'polar_contour_circularity':  {'direction': 'higher', 'label': 'Circularity'},
}

# H2: Across-frame stability (lower CV across frames = better)
H2_METRICS = {
    'hypso_transition_t':         {'label': 'Transition Threshold'},
    'polar_contour_mean_radius':  {'label': 'Contour Mean Radius'},
    'hypso_auc':                  {'label': 'Hypsometric AUC'},
    'polar_half_decay_dist':      {'label': 'Half-Decay Distance'},
    'hypso_steepness':            {'label': 'Transition Steepness'},
}

# H3: Transition sharpness (higher gradient, lower width = better)
H3_METRICS = {
    'grad_median':                {'direction': 'higher', 'label': 'Gradient Median'},
    'zone_radial_width':          {'direction': 'lower',  'label': 'Zone Radial Width'},
    'hypso_steepness':            {'direction': 'higher', 'label': 'Transition Steepness'},
}

# Derive from the single source of truth in fluorescence_transforms
from lib.fluorescence_transforms import (
    TRANSFORM_REGISTRY, TRANSFORM_LABELS
)
TRANSFORMS = list(TRANSFORM_REGISTRY.keys())


def load_data(csv_path: Path, stable_only: bool = True, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load CSV and optionally filter to light-stable frames."""
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d frames, %d columns", len(df), len(df.columns))
    if stable_only:
        df = df[df['light_stable'] == True].copy()
        logger.info("  After light-stable filter: %d frames", len(df))
    return df


def get_col(df: pd.DataFrame, transform: str, metric: str) -> pd.Series | None:
    """Get a column for a specific transform and metric."""
    col = f"{transform}__{metric}"
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    return None


def compute_h1_table(df: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    """H1: Within-frame boundary consistency. Report median of each metric per transform."""
    logger.info("\n" + "="*80)
    logger.info("HYPOTHESIS 1: Within-Frame Boundary Consistency")
    logger.info("="*80)

    results = {}
    for t in TRANSFORMS:
        row = {'transform': TRANSFORM_LABELS.get(t, t)}
        for metric, info in H1_METRICS.items():
            vals = get_col(df, t, metric)
            if vals is not None:
                row[info['label']] = vals.median()
                row[info['label'] + ' (mean)'] = vals.mean()
            else:
                row[info['label']] = np.nan
        results[t] = row

    result_df = pd.DataFrame(results).T
    result_df.index.name = 'Transform'

    # Rank transforms
    for metric, info in H1_METRICS.items():
        col = info['label']
        if col in result_df.columns:
            ascending = info['direction'] == 'lower'
            result_df[col + ' rank'] = result_df[col].rank(ascending=ascending)

    # Average rank
    rank_cols = [c for c in result_df.columns if c.endswith(' rank')]
    if rank_cols:
        result_df['H1 avg rank'] = result_df[rank_cols].mean(axis=1)
        result_df = result_df.sort_values('H1 avg rank')

    logger.info("%s", result_df.to_string())
    return result_df


def compute_h2_table(df: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    """H2: Across-frame stability. Report CV of each metric across frames, per transform."""
    logger.info("\n" + "="*80)
    logger.info("HYPOTHESIS 2: Across-Frame Stability (CV across frames)")
    logger.info("="*80)

    results = {}
    for t in TRANSFORMS:
        row = {'transform': TRANSFORM_LABELS.get(t, t)}
        for metric, info in H2_METRICS.items():
            vals = get_col(df, t, metric)
            if vals is not None:
                vals_clean = vals.dropna()
                if len(vals_clean) > 10 and vals_clean.mean() != 0:
                    cv = vals_clean.std() / abs(vals_clean.mean())
                    row[info['label'] + ' CV'] = cv
                else:
                    row[info['label'] + ' CV'] = np.nan
            else:
                row[info['label'] + ' CV'] = np.nan
        results[t] = row

    result_df = pd.DataFrame(results).T
    result_df.index.name = 'Transform'

    # Rank: lower CV = more stable = better
    cv_cols = [c for c in result_df.columns if c.endswith(' CV')]
    for col in cv_cols:
        result_df[col.replace(' CV', ' rank')] = result_df[col].rank(ascending=True)

    rank_cols = [c for c in result_df.columns if c.endswith(' rank')]
    if rank_cols:
        result_df['H2 avg rank'] = result_df[rank_cols].mean(axis=1)
        result_df = result_df.sort_values('H2 avg rank')

    logger.info("%s", result_df.to_string())
    return result_df


def compute_h3_table(df: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    """H3: Transition sharpness. Report median of each metric per transform."""
    logger.info("\n" + "="*80)
    logger.info("HYPOTHESIS 3: Transition Sharpness")
    logger.info("="*80)

    results = {}
    for t in TRANSFORMS:
        row = {'transform': TRANSFORM_LABELS.get(t, t)}
        for metric, info in H3_METRICS.items():
            vals = get_col(df, t, metric)
            if vals is not None:
                row[info['label']] = vals.median()
            else:
                row[info['label']] = np.nan
        results[t] = row

    result_df = pd.DataFrame(results).T
    result_df.index.name = 'Transform'

    for metric, info in H3_METRICS.items():
        col = info['label']
        if col in result_df.columns:
            ascending = info['direction'] == 'lower'
            result_df[col + ' rank'] = result_df[col].rank(ascending=ascending)

    rank_cols = [c for c in result_df.columns if c.endswith(' rank')]
    if rank_cols:
        result_df['H3 avg rank'] = result_df[rank_cols].mean(axis=1)
        result_df = result_df.sort_values('H3 avg rank')

    logger.info("%s", result_df.to_string())
    return result_df


def compute_overall_ranking(
    h1_df: pd.DataFrame, h2_df: pd.DataFrame, h3_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Combine rankings across all three hypotheses."""
    logger.info("\n" + "="*80)
    logger.info("OVERALL TRANSFORMATION RANKING")
    logger.info("="*80)

    ranking = pd.DataFrame(index=TRANSFORMS)
    ranking['Label'] = [TRANSFORM_LABELS.get(t, t) for t in TRANSFORMS]

    if 'H1 avg rank' in h1_df.columns:
        ranking['H1 Rank'] = h1_df['H1 avg rank']
    if 'H2 avg rank' in h2_df.columns:
        ranking['H2 Rank'] = h2_df['H2 avg rank']
    if 'H3 avg rank' in h3_df.columns:
        ranking['H3 Rank'] = h3_df['H3 avg rank']

    rank_cols = [c for c in ranking.columns if 'Rank' in c]
    ranking['Overall'] = ranking[rank_cols].mean(axis=1)
    ranking = ranking.sort_values('Overall')

    logger.info("%s", ranking.to_string())
    return ranking


def friedman_test(df: pd.DataFrame, metric: str, logger: logging.Logger | None = None) -> dict | None:
    """Run Friedman test across transforms for a given metric."""
    from scipy.stats import friedmanchisquare

    # Build matrix: rows=frames, columns=transforms
    data = {}
    for t in TRANSFORMS:
        vals = get_col(df, t, metric)
        if vals is not None:
            data[t] = vals

    if len(data) < 3:
        return None

    data_df = pd.DataFrame(data).dropna()
    if len(data_df) < 10:
        return None

    arrays = [data_df[t].values for t in data_df.columns]
    try:
        stat, p = friedmanchisquare(*arrays)
        return {'statistic': stat, 'p_value': p, 'n_frames': len(data_df),
                'n_transforms': len(arrays)}
    except Exception as e:
        if logger is not None:
            logger.warning("Friedman test failed: %s", e)
        return None


def run_friedman_tests(df: pd.DataFrame, logger: logging.Logger | None = None) -> None:
    """Friedman tests for key metrics."""
    logger.info("\n" + "="*80)
    logger.info("FRIEDMAN TESTS (all transforms compared)")
    logger.info("="*80)

    test_metrics = list(H1_METRICS.keys()) + list(H3_METRICS.keys())
    # Remove duplicates
    test_metrics = list(dict.fromkeys(test_metrics))

    for metric in test_metrics:
        label = H1_METRICS.get(metric, H3_METRICS.get(metric, {})).get('label', metric)
        result = friedman_test(df, metric, logger=logger)
        if result:
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
            logger.info("  %-30s  chi2=%10.2f  p=%.2e  %s  (n=%d)", label, result['statistic'], result['p_value'], sig, result['n_frames'])
        else:
            logger.info("  %-30s  FAILED (insufficient data)", label)


def compute_correlation_matrix(df: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Correlation between transforms using median intensity values."""
    logger.info("\n" + "="*80)
    logger.info("TRANSFORMATION CORRELATION MATRIX (median_z)")
    logger.info("="*80)

    data = {}
    for t in TRANSFORMS:
        vals = get_col(df, t, 'median_z')
        if vals is not None:
            data[TRANSFORM_LABELS.get(t, t)] = vals

    corr_df = pd.DataFrame(data).corr(method='spearman')
    logger.info("%s", corr_df.round(2).to_string())
    return corr_df


def generate_visualizations(
    df: pd.DataFrame,
    h1_df: pd.DataFrame,
    h2_df: pd.DataFrame,
    h3_df: pd.DataFrame,
    ranking: pd.DataFrame,
    corr_df: pd.DataFrame,
    output_dir: str | Path,
    logger: logging.Logger | None = None,
) -> None:
    """Generate all visualization figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [TRANSFORM_LABELS.get(t, t) for t in TRANSFORMS]

    # 1. H1 Box plots: contour CV across transforms
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (metric, info) in zip(axes, H1_METRICS.items()):
        data = []
        for t in TRANSFORMS:
            vals = get_col(df, t, metric)
            if vals is not None:
                data.append(vals.dropna().values)
            else:
                data.append([])
        ax.boxplot(data, labels=labels, vert=True)
        ax.set_title(info['label'])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    plt.suptitle('H1: Within-Frame Boundary Consistency', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'h1_boundary_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. H2 Bar chart: CV across frames
    fig, ax = plt.subplots(figsize=(14, 6))
    h2_cv_cols = [c for c in h2_df.columns if c.endswith(' CV')]
    if h2_cv_cols:
        h2_plot = h2_df[['transform'] + h2_cv_cols].set_index('transform')
        h2_plot.plot(kind='bar', ax=ax)
        ax.set_title('H2: Across-Frame Stability (CV, lower = more stable)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coefficient of Variation')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'h2_stability.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. H3 Box plots: gradient and sharpness
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (metric, info) in zip(axes, H3_METRICS.items()):
        data = []
        for t in TRANSFORMS:
            vals = get_col(df, t, metric)
            if vals is not None:
                data.append(vals.dropna().values)
            else:
                data.append([])
        ax.boxplot(data, labels=labels, vert=True)
        ax.set_title(info['label'])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    plt.suptitle('H3: Transition Sharpness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'h3_sharpness.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Overall ranking chart
    if ranking is not None and 'Overall' in ranking.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        rank_sorted = ranking.sort_values('Overall')
        colors = ['#2ecc71' if i < 3 else '#3498db' if i < 8 else '#95a5a6'
                  for i in range(len(rank_sorted))]
        ax.barh(rank_sorted['Label'], rank_sorted['Overall'], color=colors)
        ax.set_xlabel('Average Rank (lower = better)')
        ax.set_title('Overall Transformation Ranking (H1 + H2 + H3)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_ranking.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Correlation heatmap
    if corr_df is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_df.index)
        plt.colorbar(im, ax=ax, label='Spearman Correlation')
        ax.set_title('Transformation Correlation Matrix (median intensity)', fontsize=14, fontweight='bold')
        # Add text values
        for i in range(len(corr_df)):
            for j in range(len(corr_df)):
                val = corr_df.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 6. TDA summary: persistence entropy across transforms
    fig, ax = plt.subplots(figsize=(14, 6))
    data = []
    for t in TRANSFORMS:
        vals = get_col(df, t, 'tda_persistence_entropy')
        if vals is not None:
            data.append(vals.dropna().values)
        else:
            data.append([])
    ax.boxplot(data, labels=labels, vert=True)
    ax.set_title('TDA: Persistence Entropy by Transform', fontsize=14, fontweight='bold')
    ax.set_ylabel('Persistence Entropy')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'tda_entropy.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("\nVisualization saved to: %s", output_dir)
    logger.info("  h1_boundary_consistency.png")
    logger.info("  h2_stability.png")
    logger.info("  h3_sharpness.png")
    logger.info("  overall_ranking.png")
    logger.info("  correlation_matrix.png")
    logger.info("  tda_entropy.png")


def main():
    parser = argparse.ArgumentParser(description="Post-process topographic analysis results")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to frame_metrics_summary.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output figures and tables")
    parser.add_argument("--stable-only", action="store_true", default=True,
                        help="Filter to light-stable frames only (default)")
    parser.add_argument("--all-frames", dest="stable_only", action="store_false")
    args = parser.parse_args()

    from config import DEFAULT_CSV

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = DEFAULT_CSV

    if csv_path is None:
        print("ERROR: --csv is required (or set ALA_WORKSPACE)", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent.parent / "figures"
    logger = setup_logging('postprocess_results', output_dir=output_dir)

    if not csv_path.exists():
        logger.error("CSV not found at %s", csv_path)
        logger.error("Specify with --csv /path/to/frame_metrics_summary.csv")
        sys.exit(1)

    # Load
    df = load_data(str(csv_path), stable_only=args.stable_only, logger=logger)

    # Hypothesis tables
    h1_df = compute_h1_table(df, logger=logger)
    h2_df = compute_h2_table(df, logger=logger)
    h3_df = compute_h3_table(df, logger=logger)

    # Overall ranking
    ranking = compute_overall_ranking(h1_df, h2_df, h3_df, logger=logger)

    # Friedman tests
    run_friedman_tests(df, logger=logger)

    # Correlation matrix
    corr_df = compute_correlation_matrix(df, logger=logger)

    # Visualizations
    generate_visualizations(df, h1_df, h2_df, h3_df, ranking, corr_df, output_dir, logger=logger)

    # Save ranking tables as CSV
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    h1_df.to_csv(tables_dir / "h1_boundary_consistency.csv")
    h2_df.to_csv(tables_dir / "h2_stability.csv")
    h3_df.to_csv(tables_dir / "h3_sharpness.csv")
    ranking.to_csv(tables_dir / "overall_ranking.csv")
    corr_df.to_csv(tables_dir / "correlation_matrix.csv")
    logger.info("\nTables saved to: %s", tables_dir)

    logger.info("\n%s", "="*80)
    logger.info("POST-PROCESSING COMPLETE")
    logger.info("%s", "="*80)


if __name__ == "__main__":
    main()
