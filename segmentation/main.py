#!/usr/bin/env python3
"""Video segmentation tool entry point.

Usage:
    python -m segmentation                                # startup dialog
    python -m segmentation --video v.mp4                  # direct open
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from PyQt5.QtWidgets import QApplication, QDialog

from .ui.startup_dialog import StartupDialog
from .ui.app_window import AnnotationTool
from .engine.sam2_engine import SAM2Engine
from .engine.remote_sam2 import RemoteSAM2Engine

import httpx


def _ensure_gpu_manager(url: str, checkpoint: str | None, autostart: bool = True, timeout: float = 60.0):
    try:
        httpx.get(url.rstrip("/") + "/health", timeout=3.0)
        return
    except httpx.HTTPError:
        if not autostart:
            raise RuntimeError("GPU manager is not running and autostart is disabled")

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 9777
    cmd = [
        sys.executable,
        "-m",
        "segmentation.server.gpu_manager",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]

    log_path = Path.home() / ".cache/5ala/gpu_manager.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(url.rstrip("/") + "/health", timeout=3.0)
            return
        except httpx.HTTPError:
            time.sleep(1.0)
    raise RuntimeError("GPU manager failed to start within timeout")


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
    parser.add_argument("--gpu-manager-url", type=str,
                        default=os.environ.get("SAM2_MANAGER_URL", "http://127.0.0.1:9777"),
                        help="Base URL for the shared GPU manager")
    parser.add_argument("--local-sam2", action="store_true",
                        help="Bypass GPU manager and load SAM2 inside the GUI process")
    parser.add_argument("--no-manager-autostart", action="store_true",
                        help="Do not auto-start the GPU manager if it is not running")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if args.local_sam2:
        sam2 = SAM2Engine(args.checkpoint)
    else:
        _ensure_gpu_manager(args.gpu_manager_url, args.checkpoint, autostart=not args.no_manager_autostart)
        sam2 = RemoteSAM2Engine(args.gpu_manager_url)

    if args.video and args.output:
        tool = AnnotationTool(
            video_path=args.video,
            output_dir=args.output,
            start_frame=args.start,
            end_frame=args.end,
            checkpoint=args.checkpoint,
            sam2_engine=sam2,
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
            sam2_engine=sam2,
        )

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
