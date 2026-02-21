"""
Centralized workspace and path configuration.
================================================
All scripts import from here instead of re-implementing workspace detection.

Resolution order:
  1. ALA_WORKSPACE environment variable (explicit override)
  2. Walk up from this file until we find a directory containing BOTH
     "ALA 5fps/" AND "scripts/"  (requires both to avoid false positives)
  3. Paths default to None (validated at CLI runtime, not at import time)

Usage in any script::

    from config import DEFAULT_VIDEO, DEFAULT_MASKS
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)


def _find_workspace() -> Path | None:
    """Locate the project root by walking up from this file.

    Returns None instead of exiting so that ``import config`` never kills
    the interpreter (important for tests, notebooks, and IDE indexing).
    """
    env = os.environ.get("ALA_WORKSPACE")
    if env:
        ws = Path(env)
        if ws.is_dir():
            return ws
        _log.warning("ALA_WORKSPACE=%r does not exist, auto-detecting.", env)

    candidate = Path(__file__).resolve().parent.parent
    for _ in range(5):
        if (candidate / "ALA 5fps").is_dir() and (candidate / "scripts").is_dir():
            return candidate
        candidate = candidate.parent

    _log.debug(
        "Cannot auto-detect workspace. Set the ALA_WORKSPACE environment "
        "variable to the directory containing 'ALA 5fps/' and 'scripts/'."
    )
    return None


WORKSPACE = _find_workspace()

# ── Canonical data paths (None-safe when workspace is unresolved) ─────────
ALA_DIR: Path | None = WORKSPACE / "ALA 5fps" if WORKSPACE else None
DEFAULT_VIDEO: Path | None = ALA_DIR / "2025-03-25-Blue-frames_5fps.mp4" if ALA_DIR else None
DEFAULT_MASKS: Path | None = ALA_DIR / "masks" if ALA_DIR else None
DEFAULT_OUTPUT: Path | None = ALA_DIR / "analysis" / "topographic_v1" if ALA_DIR else None
DEFAULT_SUMMARY: Path | None = DEFAULT_OUTPUT / "summary" if DEFAULT_OUTPUT else None
DEFAULT_STATS: Path | None = DEFAULT_OUTPUT / "statistical_evaluation" if DEFAULT_OUTPUT else None
DEFAULT_NPZ: Path | None = DEFAULT_OUTPUT / "functional_data.npz" if DEFAULT_OUTPUT else None
DEFAULT_CSV: Path | None = DEFAULT_SUMMARY / "frame_metrics_summary.csv" if DEFAULT_SUMMARY else None

# ── Analysis defaults (shared grid dimensions & thresholds) ───────────────
N_HYPSO: int = 100
N_RADII: int = 200
N_ANGLES: int = 360
N_HIST_BINS: int = 100
POLAR_MAX_RADIUS: int = 200

# Statistical significance thresholds
P_THRESHOLDS: tuple[float, ...] = (0.001, 0.01, 0.05)
