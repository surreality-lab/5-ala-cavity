#!/usr/bin/env python3
"""
Transform Family Visualization
================================
Generates manuscript-quality figures connecting statistical findings
to spatial/3D representations of fluorescence transforms.

Figures:
  1. Boundary Overlay Comparison -- contours from 3 transform families
     overlaid on the original frame (shows where each places the margin)
  2. 3D Surface Comparison -- side-by-side mesh plots showing the
     topographic surface shape for representative transforms
  3. Radial Cross-Section -- radial rays on frame + inset profiles

Usage:
    python scripts/04_visualization/visualize_transform_families.py
    python scripts/04_visualization/visualize_transform_families.py --frame 1875
    python scripts/04_visualization/visualize_transform_families.py --frames 1345 1875 2183
"""

from __future__ import annotations

import cv2
import logging
import numpy as np
from pathlib import Path
import argparse
import sys
import warnings

_log = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='Maximum number of iterations')

from lib.fluorescence_transforms import (
    compute_all_transforms, TRANSFORM_REGISTRY, can_be_negative, get_transform_label
)
from lib.topographic_analysis import normalize_surface, find_peak
from lib.polar_surface_analysis import (
    build_polar_surface_fast, compute_radial_profiles, extract_boundary_contour
)
from lib.blood_detection import check_light_stability
from lib.utils import crop_ui, load_mask, load_analysis_mask

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# ── Defaults (from centralized config) ─────────────────────────────────────
from config import WORKSPACE, DEFAULT_VIDEO, DEFAULT_MASKS, ALA_DIR, N_RADII, N_ANGLES, POLAR_MAX_RADIUS
_DEFAULT_VIDEO = DEFAULT_VIDEO
_DEFAULT_MASKS = DEFAULT_MASKS
_DEFAULT_OUTPUT = ALA_DIR / "analysis" / "topographic_v1" / "family_figures" if ALA_DIR else None

# Representative transforms from each family (from statistical analysis)
FAMILY_TRANSFORMS = {
    'Spike Detectors': ['R_G', 'YCrCb_Cr'],
    'Gradient Mappers': ['R_RpG', 'NDI', 'LAB_a'],
    'Field Detectors': ['HSV_S'],
}

# Key transforms for detailed comparison
KEY_TRANSFORMS = ['R_G', 'R_RpG', 'NDI', 'HSV_S', 'R', 'LAB_a']

# Colors for boundary overlays
FAMILY_COLORS = {
    'Spike Detectors': (1.0, 0.2, 0.2),
    'Gradient Mappers': (0.2, 0.8, 0.2),
    'Field Detectors': (0.3, 0.5, 1.0),
}

TRANSFORM_NAMES = list(TRANSFORM_REGISTRY.keys())


def load_frame_data(
    cap: cv2.VideoCapture,
    frame_idx: int,
    cavity_dir: str | Path,
    refl_dir: str | Path,
    blood_dir: str | Path,
) -> dict | None:
    """Load a frame with masks and compute all transforms."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        return None
    frame_bgr = crop_ui(frame_bgr)

    cavity_mask, analysis_mask, _ = load_analysis_mask(
        frame_idx, cavity_dir, refl_dir, blood_dir, frame_bgr.shape[:2])
    if cavity_mask is None or analysis_mask is None:
        return None

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2].astype(np.float32) / 255.0

    surfaces = compute_all_transforms(frame_bgr, analysis_mask)

    # Normalize all surfaces
    norm_surfaces = {}
    peaks = {}
    for tkey, surf in surfaces.items():
        norm_surf, _ = normalize_surface(surf, analysis_mask, 'p99')
        norm_surfaces[tkey] = norm_surf
        use_v = v_ch if can_be_negative(tkey) or tkey in (
            'R', 'R_G', 'R_RpG', 'R_B', 'R_GpB', 'R_minus_kG',
            'NMF_fluor', 'PCA_PC1') else None
        peak_yx, _ = find_peak(surf, analysis_mask, use_v)
        peaks[tkey] = peak_yx

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return {
        'frame_bgr': frame_bgr,
        'frame_rgb': frame_rgb,
        'cavity_mask': cavity_mask,
        'analysis_mask': analysis_mask,
        'surfaces': surfaces,
        'norm_surfaces': norm_surfaces,
        'peaks': peaks,
        'v_ch': v_ch,
    }


def extract_boundary_pixels(norm_surf: np.ndarray, analysis_mask: np.ndarray, threshold: float = 0.5) -> list:
    """Extract boundary pixels at a given threshold using contour finding."""
    binary = (norm_surf > threshold).astype(np.uint8)
    binary[~analysis_mask] = 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Boundary Overlay Comparison
# ═══════════════════════════════════════════════════════════════════════════

def figure_boundary_overlay(
    data: dict, frame_idx: int, output_dir: Path, thresholds: list[float] | None = None,
) -> None:
    """Overlay boundary contours from three transform families on the original frame.

    Produces a two-panel figure per threshold: left shows all family
    boundaries, right shows one representative per family with thicker lines.

    Parameters
    ----------
    data : dict
        Frame data bundle returned by :func:`load_frame_data`.
    frame_idx : int
        Frame number (used in titles and filenames).
    output_dir : Path
        Directory to write the PNG output.
    thresholds : list[float] | None
        Normalized intensity thresholds for contour extraction (default [0.3, 0.5]).
    """
    if thresholds is None:
        thresholds = [0.3, 0.5]
    _log.info("Generating boundary overlay (frame %d)...", frame_idx)

    for threshold in thresholds:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Left: original frame with all family boundaries
        ax = axes[0]
        img_overlay = data['frame_rgb'].copy().astype(np.float32) / 255.0
        # Brighten slightly for visibility
        img_overlay = np.clip(img_overlay * 1.3, 0, 1)

        ax.imshow(img_overlay)

        for family_name, transforms in FAMILY_TRANSFORMS.items():
            color = FAMILY_COLORS[family_name]
            for tkey in transforms:
                if tkey not in data['norm_surfaces']:
                    continue
                contours = extract_boundary_pixels(
                    data['norm_surfaces'][tkey], data['analysis_mask'], threshold)
                for cnt in contours:
                    if len(cnt) > 10:
                        pts = cnt.squeeze()
                        ax.plot(pts[:, 0], pts[:, 1], color=color,
                                linewidth=1.5, alpha=0.8)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = []
        for family_name in FAMILY_TRANSFORMS:
            color = FAMILY_COLORS[family_name]
            transforms = FAMILY_TRANSFORMS[family_name]
            label = f"{family_name} ({', '.join(get_transform_label(t) for t in transforms)})"
            legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=label))
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8,
                  framealpha=0.9)
        ax.set_title(f'Frame {frame_idx}: Boundary Contours at t={threshold}',
                     fontsize=13, fontweight='bold')
        ax.axis('off')

        # Right: zoomed comparison of key 3 transforms
        ax2 = axes[1]
        ax2.imshow(img_overlay)

        # Show only one representative per family with thicker lines
        rep_transforms = {
            'Spike Detectors': ('R_G', (1.0, 0.2, 0.2)),
            'Gradient Mappers': ('R_RpG', (0.2, 0.8, 0.2)),
            'Field Detectors': ('HSV_S', (0.3, 0.5, 1.0)),
        }
        legend2 = []
        for family_name, (tkey, color) in rep_transforms.items():
            if tkey not in data['norm_surfaces']:
                continue
            contours = extract_boundary_pixels(
                data['norm_surfaces'][tkey], data['analysis_mask'], threshold)
            for cnt in contours:
                if len(cnt) > 10:
                    pts = cnt.squeeze()
                    ax2.plot(pts[:, 0], pts[:, 1], color=color,
                             linewidth=2.5, alpha=0.9)
            legend2.append(Line2D([0], [0], color=color, linewidth=3,
                                  label=f"{family_name}: {get_transform_label(tkey)}"))

        ax2.legend(handles=legend2, loc='lower left', fontsize=9, framealpha=0.9)
        ax2.set_title(f'Representative Boundaries (t={threshold})',
                      fontsize=13, fontweight='bold')
        ax2.axis('off')

        plt.suptitle(f'Where Each Transform Family Places the Tumor Margin',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        t_str = str(threshold).replace('.', '')
        plt.savefig(output_dir / f'boundary_overlay_frame{frame_idx}_t{t_str}.png',
                    dpi=200, bbox_inches='tight')
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: 3D Surface Comparison
# ═══════════════════════════════════════════════════════════════════════════

def figure_3d_surfaces(data: dict, frame_idx: int, output_dir: Path) -> None:
    """Side-by-side 3D mesh plots for representative transforms.

    Renders one subplot per transform (R/G, R/(R+G), NDI, HSV_S) showing
    the normalized fluorescence surface with masked regions as NaN.

    Parameters
    ----------
    data : dict
        Frame data bundle returned by :func:`load_frame_data`.
    frame_idx : int
        Frame number (used in titles and filenames).
    output_dir : Path
        Directory to write the PNG output.
    """
    _log.info("Generating 3D surface comparison (frame %d)...", frame_idx)

    plot_transforms = ['R_G', 'R_RpG', 'NDI', 'HSV_S']
    plot_labels = [get_transform_label(t) for t in plot_transforms]
    family_labels = ['Spike Detector', 'Gradient Mapper', 'Gradient Mapper', 'Field Detector']

    fig = plt.figure(figsize=(20, 10))

    subsample = 4  # Downsample for rendering speed

    for idx, (tkey, label, fam) in enumerate(
            zip(plot_transforms, plot_labels, family_labels)):
        if tkey not in data['norm_surfaces']:
            continue

        ax = fig.add_subplot(1, len(plot_transforms), idx + 1, projection='3d')

        surf = data['norm_surfaces'][tkey].copy()
        mask = data['analysis_mask'].copy()

        # Subsample
        surf_sub = surf[::subsample, ::subsample]
        mask_sub = mask[::subsample, ::subsample]

        H, W = surf_sub.shape
        X, Y = np.meshgrid(np.arange(W), np.arange(H))

        # Set non-mask to NaN for clean rendering
        Z = np.where(mask_sub, surf_sub, np.nan)
        Z = np.nan_to_num(Z, nan=0.0)

        # Only plot masked region
        Z[~mask_sub] = np.nan

        ax.plot_surface(X, Y, Z, cmap='inferno', alpha=0.9,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=True, shade=True)

        ax.set_zlim(0, 1)
        ax.set_title(f'{label}\n({fam})', fontsize=11, fontweight='bold')
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Intensity', fontsize=8)
        ax.view_init(elev=35, azim=225)
        ax.tick_params(labelsize=7)

    plt.suptitle(f'Frame {frame_idx}: 3D Fluorescence Surface by Transform Family',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'surface_3d_frame{frame_idx}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Radial Cross-Section
# ═══════════════════════════════════════════════════════════════════════════

def figure_radial_crosssection(data: dict, frame_idx: int, output_dir: Path) -> None:
    """Radial decay profile comparison across transform families.

    Left panel: original frame with radial rays from the peak and boundary
    contours for key transforms. Right panel: median radial intensity
    profiles with inter-quartile shading for six representative transforms.

    Parameters
    ----------
    data : dict
        Frame data bundle returned by :func:`load_frame_data`.
    frame_idx : int
        Frame number (used in titles and filenames).
    output_dir : Path
        Directory to write the PNG output.
    """
    _log.info("Generating radial cross-section (frame %d)...", frame_idx)

    plot_transforms = ['R_G', 'R_RpG', 'NDI', 'LAB_a', 'HSV_S', 'R']
    colors = ['#e74c3c', '#2ecc71', '#27ae60', '#f39c12', '#3498db', '#9b59b6']

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.05)

    # Left: frame with radial rays
    ax_img = fig.add_subplot(gs[0])
    img = data['frame_rgb'].copy().astype(np.float32) / 255.0
    img = np.clip(img * 1.3, 0, 1)
    ax_img.imshow(img)

    # Draw rays from peak of R_RpG (good representative peak)
    ref_key = 'R_RpG'
    if ref_key in data['peaks']:
        peak_y, peak_x = data['peaks'][ref_key]

        n_rays = 36
        ray_len = 200
        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

        for angle in angles:
            end_x = peak_x + ray_len * np.cos(angle)
            end_y = peak_y + ray_len * np.sin(angle)
            ax_img.plot([peak_x, end_x], [peak_y, end_y],
                        color='white', alpha=0.15, linewidth=0.5)

        ax_img.plot(peak_x, peak_y, 'w+', markersize=15, markeredgewidth=2)

        # Draw boundary contours for key transforms
        for tkey, color in zip(['R_G', 'R_RpG', 'HSV_S'],
                                ['#e74c3c', '#2ecc71', '#3498db']):
            if tkey in data['norm_surfaces']:
                contours = extract_boundary_pixels(
                    data['norm_surfaces'][tkey], data['analysis_mask'], 0.5)
                for cnt in contours:
                    if len(cnt) > 10:
                        pts = cnt.squeeze()
                        ax_img.plot(pts[:, 0], pts[:, 1], color=color,
                                    linewidth=2, alpha=0.8)

    ax_img.set_title(f'Frame {frame_idx}: Radial Rays from Peak\n'
                     f'(contours: R/G=red, R/(R+G)=green, HSV Sat=blue)',
                     fontsize=11, fontweight='bold')
    ax_img.axis('off')

    # Right: radial profiles
    ax_prof = fig.add_subplot(gs[1])

    for tkey, color in zip(plot_transforms, colors):
        if tkey not in data['norm_surfaces'] or tkey not in data['peaks']:
            continue

        peak_yx = data['peaks'][tkey]
        norm_surf = data['norm_surfaces'][tkey]
        analysis_mask = data['analysis_mask']

        polar, radii, angles = build_polar_surface_fast(
            norm_surf, analysis_mask, peak_yx,
            max_radius=POLAR_MAX_RADIUS, n_radii=N_RADII, n_angles=N_ANGLES)
        rad_stats = compute_radial_profiles(polar, radii)
        median_prof = rad_stats['median']
        p25 = rad_stats['p25']
        p75 = rad_stats['p75']

        label = get_transform_label(tkey)
        ax_prof.plot(radii, median_prof, color=color, linewidth=2,
                     label=label, alpha=0.9)
        ax_prof.fill_between(radii, p25, p75, color=color, alpha=0.1)

    ax_prof.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_prof.annotate('boundary\nthreshold', xy=(185, 0.52), fontsize=8,
                     color='gray', ha='center')
    ax_prof.set_xlabel('Distance from Peak (px)', fontsize=12)
    ax_prof.set_ylabel('Normalized Intensity', fontsize=12)
    ax_prof.set_title('Radial Decay Profile by Transform',
                      fontsize=11, fontweight='bold')
    ax_prof.legend(fontsize=9, loc='upper right')
    ax_prof.set_xlim(0, 200)
    ax_prof.set_ylim(0, 1.05)
    ax_prof.grid(True, alpha=0.3)

    plt.suptitle('Spatial Fluorescence Decay: How Each Transform Sees the Boundary',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'radial_crosssection_frame{frame_idx}.png',
                dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Combined Family Summary Panel
# ═══════════════════════════════════════════════════════════════════════════

def figure_family_summary(data: dict, frame_idx: int, output_dir: Path) -> None:
    """2x2 summary panel: original frame + three family-representative heatmaps.

    Each heatmap is rendered with the 'inferno' colormap, boundary contour
    at t=0.5, and a white cross at the detected peak location.

    Parameters
    ----------
    data : dict
        Frame data bundle returned by :func:`load_frame_data`.
    frame_idx : int
        Frame number (used in titles and filenames).
    output_dir : Path
        Directory to write the PNG output.
    """
    _log.info("Generating family summary panel (frame %d)...", frame_idx)

    rep_transforms = [
        ('R_G', 'Spike Detector: R/G', '#e74c3c'),
        ('R_RpG', 'Gradient Mapper: R/(R+G)', '#2ecc71'),
        ('HSV_S', 'Field Detector: HSV Sat', '#3498db'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: original
    ax = axes[0, 0]
    img = data['frame_rgb'].copy().astype(np.float32) / 255.0
    ax.imshow(np.clip(img * 1.3, 0, 1))
    ax.set_title(f'Frame {frame_idx}: Original', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Others: heatmaps with contour
    for idx, (tkey, title, color) in enumerate(rep_transforms):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]

        if tkey not in data['norm_surfaces']:
            ax.set_visible(False)
            continue

        surf = data['norm_surfaces'][tkey].copy()
        surf[~data['analysis_mask']] = np.nan
        display = np.where(data['analysis_mask'], surf, 0)

        ax.imshow(display, cmap='inferno', vmin=0, vmax=1)

        # Overlay contour at 0.5
        contours = extract_boundary_pixels(
            data['norm_surfaces'][tkey], data['analysis_mask'], 0.5)
        for cnt in contours:
            if len(cnt) > 10:
                pts = cnt.squeeze()
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, alpha=0.9)

        # Mark peak
        if tkey in data['peaks']:
            py, px = data['peaks'][tkey]
            ax.plot(px, py, 'w+', markersize=12, markeredgewidth=2)

        ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        ax.axis('off')

    plt.suptitle('Three Ways to See the Fluorescence Boundary',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'family_summary_frame{frame_idx}.png',
                dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate transform family visualization figures")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--masks-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--frames", type=int, nargs='+', default=[1875],
                        help="Frame indices to visualize (default: 1875)")
    args = parser.parse_args()

    VIDEO_PATH = Path(args.video) if args.video else _DEFAULT_VIDEO
    masks_base = Path(args.masks_dir) if args.masks_dir else _DEFAULT_MASKS
    OUTPUT_DIR = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT
    for name, val in [("--video", VIDEO_PATH), ("--masks-dir", masks_base),
                      ("--output-dir", OUTPUT_DIR)]:
        if val is None:
            print(f"ERROR: {name} is required (or set ALA_WORKSPACE)", file=sys.stderr)
            sys.exit(1)
    CAVITY_DIR = masks_base / "cavity"
    REFL_DIR = masks_base / "reflection"
    BLOOD_DIR = masks_base / "blood"

    if not VIDEO_PATH.exists():
        _log.error("Video not found: %s", VIDEO_PATH)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    for frame_idx in args.frames:
        _log.info("Processing frame %d...", frame_idx)
        data = load_frame_data(cap, frame_idx, CAVITY_DIR, REFL_DIR, BLOOD_DIR)
        if data is None:
            _log.warning("Could not load frame %d, skipping", frame_idx)
            continue

        figure_family_summary(data, frame_idx, OUTPUT_DIR)
        figure_boundary_overlay(data, frame_idx, OUTPUT_DIR)
        figure_3d_surfaces(data, frame_idx, OUTPUT_DIR)
        figure_radial_crosssection(data, frame_idx, OUTPUT_DIR)

    cap.release()

    _log.info("VISUALIZATION COMPLETE -- Output: %s, Frames: %s", OUTPUT_DIR, args.frames)


if __name__ == "__main__":
    main()
