#!/usr/bin/env python3
"""Video segmentation tool entry point.

Usage:
    python -m segmentation                                # startup dialog
    python -m segmentation --video v.mp4                  # direct open
"""

import os
import sys
import argparse
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QDialog

from .ui.startup_dialog import StartupDialog
from .ui.app_window import AnnotationTool


def main():
    parser = argparse.ArgumentParser(description="Video Segmentation Tool")
    parser.add_argument("--base-dir", type=str,
                        default=str(Path(__file__).parent.parent.parent),
                        help="Project root for startup dialog")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to MP4 video (skip startup dialog)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for masks")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame")
    _env_ckpt = os.environ.get("SAM2_CHECKPOINT")
    if _env_ckpt and Path(_env_ckpt).exists():
        _default_ckpt = Path(_env_ckpt)
    else:
        _default_ckpt = Path("/opt/5-ALA-Videos/weights/sam2_cavity_finetuned.pt")
        if not _default_ckpt.exists():
            _local = Path.home() / "Desktop/VScode/ICG FA/checkpoints/sam2_hiera_large.pt"
            if _local.exists():
                _default_ckpt = _local
    parser.add_argument("--checkpoint", type=str,
                        default=str(_default_ckpt),
                        help="SAM2 checkpoint path (or set SAM2_CHECKPOINT)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if args.video and args.output:
        tool = AnnotationTool(
            video_path=args.video,
            output_dir=args.output,
            start_frame=args.start,
            end_frame=args.end,
            checkpoint=args.checkpoint,
        )
    else:
        dlg = StartupDialog(args.base_dir)
        if dlg.exec_() != QDialog.Accepted:
            sys.exit(0)
        sel = dlg.get_selection()
        if sel is None:
            sys.exit(0)
        info, sf, ef = sel
        out = info["folder"] / "masks"
        tool = AnnotationTool(
            video_path=str(info["path"]),
            output_dir=str(out),
            start_frame=sf,
            end_frame=ef,
            checkpoint=args.checkpoint,
        )

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
