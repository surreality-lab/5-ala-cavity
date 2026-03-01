# 5-ALA Fluorescence Video Annotation Pipeline

A streamlined pipeline for annotating 5-ALA fluorescence surgical videos with AI-assisted cavity segmentation, automatic mask propagation, and intelligent workflow management.

---

## Quick Start

### 1. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Install SAM2 (Segment Anything Model 2)
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 2. Download SAM2 Model
The SAM2 checkpoint (~900MB) is required but not included in the repository:

```bash
# Download to models/sam2_checkpoints/
cd models/sam2_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
cd ../..
```

The config file (`sam2_hiera_l.yaml`) is already included.

### 3. Launch the Annotation GUI
```bash
python -m segmentation
```

On first launch the app automatically starts the shared GPU manager (FastAPI + SAM2) on the workstation, then opens the PyQt GUI. Subsequent users simply run the same command; their sessions register with the GPU manager, queue fairly for the dual A5000s, and reuse the same mask output directories. Use `python -m segmentation --video path/to/video.mp4 --output path/to/masks` to skip the startup dialog, or `--local-sam2` to bypass the manager in offline environments.

---

## Environment Setup (Recommended)

Run everything under a dedicated conda env so PyQt, CUDA, and the GPU manager share the same dependency set:

```bash
# 0. (Optional) remove any stale copy
conda env remove -n alaenv -y

# 1. Create + activate Python 3.10 env
conda create -n alaenv python=3.10 -y
conda activate alaenv

# 2. Install project deps (uses opencv-python-headless to avoid Qt conflicts)
pip install -r requirements.txt

# 3. Install SAM2 from source
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# 4. Grab the checkpoint once per machine (see Quick Start step 2)
mkdir -p models/sam2_checkpoints
wget -O models/sam2_checkpoints/sam2_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# 5. Launch the GUI inside the env
python -m segmentation
```

Remote GUI sessions (XQuartz, X11 forwarding) need a working display before launching PyQt. Verify with `xclock`, then run `export LIBGL_ALWAYS_INDIRECT=1` if you see `glx: failed to create drisw screen` while connected over SSH.

---

## Shared GPU Manager

- The first GUI session spawns a FastAPI GPU manager on the annotation workstation. It loads SAM2 once per GPU (both RTX A5000s bridged with NVLink), keeps the weights in GPU memory, and exposes HTTP endpoints that all GUIs call for clicks or propagation.
- Each GUI registers a session token; the manager enforces round-robin fairness so no user can monopolize the accelerators. Jobs queue transparently, so expect short delays if more sessions are active than available GPUs.
- When the last GUI disconnects or everyone logs off, the manager waits ~45 s, then shuts itself down so the box stays clean for the next workday.
- Need to run it manually (e.g., headless server maintenance)?
  ```bash
  python -m segmentation.server.gpu_manager \
    --host 0.0.0.0 --port 9777 \
    --checkpoint /opt/5-ALA-Videos/weights/sam2_cavity_finetuned.pt
  ```
- GUI flags:
  - `--gpu-manager-url http://host:9777` → connect to a custom deployment
  - `--no-manager-autostart` → require the service to be running already
  - `--local-sam2` → bypass the manager and fall back to single-process SAM2 (use only for offline debugging; no fairness or shared caching)

---

## Cavity Definition Tool

### Features

**🎯 Flexible Video Selection**
- Browse to any directory containing MP4 videos
- Recursive search finds all videos in subdirectories
- Select any frame range (500-frame batches)
- Works with any folder structure

**🤖 AI-Assisted Segmentation**
- SAM2 (Segment Anything Model 2) integration
- Click or brush to define cavity boundaries
- Automatic mask propagation to next frame
- Real-time preview and refinement

**💾 Intelligent Auto-Save**
- Auto-saves when moving between frames
- Auto-saves when closing the tool
- Creates session summary with complete metadata
- Never lose your work

**🔄 Cascading Updates**
- Modify any frame, changes propagate forward automatically
- Edit frame 100 → frames 101, 102, 103... update as you navigate
- Existing masks are intelligently replaced with updated versions
- Consistent annotations throughout the sequence

**📊 Frame Display**
- Shows actual frame numbers (e.g., 500-999, not 0-499)
- Slider operates on real frame indices
- Progress tracking shows completed masks per range

### Workflow

1. **Launch Tool**
  ```bash
  python -m segmentation [--video data/my-case/video.mp4 --output data/my-case/masks]
  ```

2. **Select Video**
   - Click "Browse..." to select a folder
   - Choose video from dropdown (shows all MP4s found)
   - View video info: total frames, FPS, existing masks

3. **Select Frame Range**
   - Choose 500-frame batch to work on
   - See mask completion status (e.g., "500-999 (123 masks)")
   - Click "Open Tool" to start

4. **Annotate Cavities**
   - **First frame**: Click/brush to define cavity boundary with SAM2
   - **Next frame** (`]` key): Mask automatically propagates from previous frame
   - **Refine**: Add/remove points to adjust the propagated mask
   - **Continue**: Keep moving forward, masks auto-save and auto-propagate
   - **Edit existing**: Go back to any frame, modify it, then move forward to update subsequent frames

5. **Close & Review**
   - Close tool (Q/Escape/X) - everything saves automatically
   - Session summary created in `masks/cavity/session_summary.json`
   - All modified frames and metadata saved

### Controls

| Key/Action | Function |
|------------|----------|
| **Left Click** | Add point to include region (Add mode) |
| **Left Click** | Add point to exclude region (Exclude mode) |
| **Brush Drag** | Paint to include/exclude areas (Brush mode) |
| `,` (comma) | Wobble left - peek at previous frame (raw, no mask) |
| `.` (period) | Wobble right - peek at next frame (raw, no mask) |
| `[` | Navigate to previous frame (auto-saves current) |
| `]` | Navigate to next frame (auto-saves current, auto-propagates) |
| **Slider** | Jump to any frame by actual frame number |
| `D` | Cycle display mode: Original → Gamma → HSV-V |
| `A` | Add mode (include regions in mask) |
| `X` | Exclude mode (remove regions from mask) |
| `B` | Toggle Click/Brush tool |
| `+` / `=` | **Zoom in** by 10% (at cursor position, max 200%) |
| `-` | **Zoom out** by 10% (min 100%, no zoom-out below original) |
| `Shift` + `+` | Increase brush size |
| `Shift` + `-` | Decrease brush size |
| `Z` | Undo last edit (enabled only after modifications) |
| `R` | Reset - clear all mask and points for current frame |
| `Q` / Escape | Quit tool (auto-saves everything) |

### Display Modes

- **Original**: Raw video frame
- **Gamma**: Gamma correction + CLAHE + bilateral filter (enhances fluorescence)
- **HSV-V**: HSV Value channel (brightness only, useful for detecting artifacts)

All processing happens on-the-fly from the video file - no pre-rendering needed.

### Zoom Functionality

The tool includes intelligent zoom controls for detailed annotation work:

**How to Zoom:**
- Press `+` or `=` to **zoom in** by 10% increments
- Press `-` to **zoom out** by 10% increments
- Use `Shift + +` or `Shift + -` to adjust **brush size** instead

**Zoom Behavior:**
- **Default**: 100% (full frame visible)
- **Maximum zoom**: 200% (2x magnification)
- **Minimum zoom**: 100% (cannot zoom out beyond original size)
- **Zoom center**: Automatically centers on your cursor position
  - If cursor is outside the frame, centers on the middle of the screen
- **Persistence**: Zoom level is maintained across frames as you navigate
- **Smart cropping**: Shows only the zoomed region without altering pixels

**Use Cases:**
- Fine-tune cavity boundaries in complex areas
- Precisely place exclusion points on small artifacts
- Refine brush strokes in tight spaces
- Verify mask edges at high magnification

### Auto-Save System

**Frame Navigation Auto-Save**
- Moving to a different frame automatically saves the current frame
- Applies when using `[`, `]`, slider, or any navigation method
- Silent operation - no interruption to workflow

**Forward Propagation**
- Moving forward (`]`) to next frame triggers SAM2 propagation
- Existing masks are **always updated** based on previous frame
- Creates consistent, cascading annotations
- No manual propagation button needed

**Close-on-Exit Auto-Save**
- Closing tool saves all modified frames
- Creates `session_summary.json` with complete metadata
- Shows single summary: "✓ Saved 25 frames to session_summary.json"

**Session Summary Contents**
```json
{
  "video_name": "recording.mp4",
  "video_path": "/path/to/data/my-video/recording.mp4",
  "timestamp": "2025-12-22T15:30:45",
  "frame_range": {
    "start": 500,
    "end": 999
  },
  "total_frames_in_range": 500,
  "modified_frames_count": 123,
  "modified_frames": [500, 501, 502, ...],
  "frames": [
    {
      "frame_number": 500,
      "has_mask": true,
      "cavity_points_pos": [[234, 567], [345, 678]],
      "cavity_points_neg": [[120, 300]]
    }
  ]
}
```

### Cascading Updates

**How It Works:**
1. Annotate frame 100 normally
2. Press `]` to move to frame 101 → SAM2 propagates mask from frame 100
3. **If frame 101 already had a mask**, it's **replaced** with the new propagation
4. Modify frame 101 (add/remove points to refine)
5. Press `]` to move to frame 102 → SAM2 propagates updated mask
6. Continue forward → all subsequent frames update based on your changes

**Benefits:**
- ✓ Make corrections anywhere in the sequence
- ✓ Changes automatically flow forward
- ✓ No need to re-annotate subsequent frames manually
- ✓ Maintains consistency across the entire video

**Example Workflow:**
```
Frame 100: Define initial cavity
Frame 101-150: Auto-propagated from frame 100
[Notice issue at frame 125 - tool boundary changed]
Frame 125: Fix boundary by adding points
Frame 126-150: Auto-update with corrected boundary as you navigate forward
```

---

## Directory Structure

```
Project Root/
├── segmentation/                  # PyQt GUI, engines, GPU manager
│   ├── main.py                    # python -m segmentation entry point
│   ├── engine/
│   │   ├── sam2_engine.py         # Local SAM2 wrapper
│   │   ├── remote_sam2.py         # HTTP client for GPU manager
│   │   └── mask_store.py          # Persistence + locking
│   ├── ui/                        # Frame viewer, panels, dialogs
│   └── server/
│       └── gpu_manager.py         # FastAPI service (auto-launched)
├── analysis/                      # Post-processing + statistics
├── exclusion/                     # Exclusion mask pipeline
├── lib/                           # Fluorescence + topology utilities
├── models/
│   └── sam2_checkpoints/          # Downloaded SAM2 weights (.pt, .yaml)
├── data/
│   └── {video-folder}/masks/      # Shared output written by all sessions
├── requirements.txt
└── README.md
```

**Mask Output Location:**
- **New (current)**: `{video-folder}/masks/cavity/`
- **Legacy (backwards compatible)**: `{video-folder}/pipeline/02_cavity/cavity_only/frames/`

The tool automatically searches both locations and saves to the new structure.

---

## Utilities

### Read Session Summary
```bash
# View human-readable session data
python scripts/utilities/read_session_summary.py \
  data/my-video/masks/cavity/session_summary.json

# Export list of frames with masks
python scripts/utilities/read_session_summary.py \
  data/my-video/masks/cavity/session_summary.json \
  --export-frames annotated_frames.txt
```

Output example:
```
Session Summary
==================================================
Video: recording.mp4
Frame Range: 500-999 (500 frames)
Modified Frames: 123 (24.6%)

Frames with Masks: 123
Frames with Points: 45

Frame Details:
- Frame 500: Mask ✓, Points: 12 add, 3 exclude
- Frame 501: Mask ✓, Points: 0 add, 0 exclude (propagated)
...
```

---

## Analysis Pipeline (Coming Soon)

### 2. Detect Artifacts
```bash
# Reflections
python scripts/03_analysis/detect_reflections_two_stage.py \
  --video-folder data/my-video

# Blood
python scripts/03_analysis/detect_blood_simple.py \
  --video-folder data/my-video
```

### 3. Fluorescence Analysis
```bash
python scripts/03_analysis/trajectory_radial_analysis_v2.py \
  --video-folder data/my-video \
  --frames "50,100,150,200"
```

### 4. Create Comparison Video
```bash
python scripts/04_visualization/create_comparison_video.py \
  --video data/my-video/video.mp4 \
  --cavity-dir data/my-video/masks/cavity \
  --output data/my-video/comparison_video.mp4
```

---

## Troubleshooting

### No videos found in directory
- Make sure you're selecting a folder that contains `.mp4` files
- Tool searches recursively, so videos can be in subfolders
- Check file extension is `.mp4` or `.MP4` (case-sensitive on Linux)

### SAM2 not loading
```bash
# Verify model file exists and is ~900MB
ls -lh models/sam2_checkpoints/sam2_hiera_large.pt

# Re-download if missing or corrupted
cd models/sam2_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### Module not found errors
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# Reinstall SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Frame numbers seem wrong
- The tool now displays **actual frame numbers** from the video
- If you select range 500-999, the slider shows 500, 501, 502... (not 0, 1, 2...)
- Saved files use actual frame numbers: `frame_000500`, `frame_000501`, etc.

### Existing masks not loading
- Tool checks new location first: `{video-folder}/masks/cavity/`
- Then checks legacy locations for backwards compatibility
- If masks are elsewhere, copy them to `{video-folder}/masks/cavity/frame_NNNNNN/`

### Auto-propagation not working
- Requires SAM2 to be installed: `pip install git+https://github.com/facebookresearch/segment-anything-2.git`
- Model checkpoint must exist: `models/sam2_checkpoints/sam2_hiera_large.pt`
- Only propagates when moving forward by one frame (pressing `]`)
- Requires an existing mask to propagate from

---

## System Requirements

- **Python**: 3.10+
- **GPU**: Recommended for SAM2 (CUDA or MPS)
  - CPU mode available but significantly slower
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 1GB for SAM2 model + space for video and masks

## Dependencies

- PyQt5 - GUI framework
- OpenCV - Video and image processing
- NumPy - Numerical operations
- Matplotlib - Visualization
- PyTorch - SAM2 backend
- SAM2 - Segmentation model

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{5ala_annotation_2025,
  title={5-ALA Fluorescence Video Annotation Pipeline},
  author={Surreality Lab},
  year={2025},
  url={https://github.com/surreality-lab/5-ala-cavity}
}
```

---

## License

MIT License - see LICENSE file for details.


