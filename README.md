# 5-ALA Fluorescence Video Annotation Pipeline

A complete pipeline for annotating 5-ALA fluorescence surgical videos, including cavity segmentation, artifact detection, and fluorescence analysis.

---

## Setup (Do This First!)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install SAM2 (Segment Anything Model 2)
**This is a separate step - SAM2 is not on PyPI!**
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Step 3: SAM2 Model (Already Included!)
The model checkpoint (`sam2_hiera_large.pt`) and config (`sam2_hiera_l.yaml`) are **already included** in the `models/sam2_checkpoints/` folder. No download needed!

### Step 4: Set Up Your Video Data

**CRITICAL**: The folder structure must be like this:

```
5-ALA-Video-Annotation/          <-- Project folder (run scripts from HERE)
├── scripts/                     <-- Scripts folder
├── data/                        <-- MUST be named "data" exactly
│   └── my-surgery-video/        <-- Can be any name you want
│       └── recording.mp4        <-- Any video name works! (.mp4, .avi, .mov)
└── models/
```

**Common Mistakes:**
- Video directly in `data/video.mp4` - WON'T WORK (needs subfolder)
- Running scripts from inside `scripts/` folder - WON'T WORK

**Step-by-step setup:**
```bash
# 1. Navigate to the project folder
cd 5-ALA-Video-Annotation

# 2. Create the data folder structure
mkdir -p data/my-surgery-video

# 3. Copy your video (any name works!)
cp /path/to/your/surgery_recording.mp4 data/my-surgery-video/

# 4. Verify structure is correct
ls data/my-surgery-video/
# Should show your video file
```

**Now you're ready to use the tools!** The cavity definition tool will process frames on-the-fly as you work.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  1. CAVITY DEFINITION                                           │
│     define_cavity_polished.py                                   │
│     Interactive tool with SAM2 + propagation                    │
│     Processes frames on-the-fly (original/gamma/HSV-V modes)    │
│     Output: cavity_mask.png per frame                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. ARTIFACT DETECTION                                          │
│     detect_reflections_two_stage.py → reflection masks          │
│     detect_blood_simple.py → blood masks                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. ANALYSIS                                                    │
│     trajectory_radial_analysis_v2.py                            │
│     Fluorescence decay analysis (R, R/G ratio)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. VISUALIZATION                                               │
│     create_comparison_video.py                                  │
│     Side-by-side video: Original | R heatmap | R/G heatmap      │
└─────────────────────────────────────────────────────────────────┘
```

## Using the Tools

### 1. Define Cavities (Interactive GUI)
```bash
python scripts/02_cavity_definition/define_cavity_polished.py
```

This opens a GUI where you:
1. Select your video from the dropdown
2. Select a frame range (must be pre-rendered first!)
3. Click "Open Tool"
4. Use SAM2 click or brush to define the cavity boundary
5. Press `]` to move to next frame - **SAM2 automatically propagates your mask to the next frame!**
6. Refine the propagated mask if needed, then continue
7. Close the tool - **session summary with all annotations is created automatically!**

**Note:** The tool automatically handles masks:
- **Moving forward** (`]`) to an unannotated frame: SAM2 auto-propagates the current mask
- **Moving to any frame**: Auto-saves current frame's mask
- **Going back**: Loads previously annotated masks for editing
- **SAM2 propagation**: Works on-the-fly using current video frames

### 2. Detect Artifacts (After Cavity Masks Exist)

Replace `my-surgery-video` with your actual folder name:

```bash
# Reflections (bright non-fluorescing spots)
python scripts/03_analysis/detect_reflections_two_stage.py \
  --video-folder data/my-surgery-video

# Blood (dark regions)
python scripts/03_analysis/detect_blood_simple.py \
  --video-folder data/my-surgery-video
```

### 3. Run Analysis
```bash
python scripts/03_analysis/trajectory_radial_analysis_v2.py \
  --video-folder data/my-surgery-video \
  --frames "50,100,150,200"
```

### 4. Create Comparison Video
```bash
python scripts/04_visualization/create_comparison_video.py \
  --video data/my-surgery-video/video.mp4 \
  --cavity-dir data/my-surgery-video/masks/cavity \
  --output data/my-surgery-video/comparison_video.mp4
```

## Directory Structure

```
5-ALA Video Annotation/
├── data/
│   └── {video-name}/              # One folder per video
│       ├── video.mp4              # Source video
│       ├── masks/                 # All annotation masks
│       │   ├── cavity/            # Cavity annotation output
│       │   │   ├── session_summary.json  # Auto-saved session metadata
│       │   │   └── frame_NNNNNN/
│       │   │       ├── cavity_mask.png
│       │   │       └── instance_data.json
│       │   └── blood/             # (Future) Blood masks
│       │       └── reflection/    # (Future) Reflection masks
│       └── pipeline/              # Analysis outputs (legacy)
│           ├── 02_cavity/         # Legacy cavity location
│           └── 03_analysis/       # Analysis results
├── models/
│   └── sam2_checkpoints/          # SAM2 model weights
├── scripts/
│   ├── 02_cavity_definition/
│   │   └── define_cavity_polished.py
│   ├── 03_analysis/
│   │   ├── detect_reflections_two_stage.py
│   │   ├── detect_blood_simple.py
│   │   └── trajectory_radial_analysis_v2.py
│   ├── 04_visualization/
│   │   └── create_comparison_video.py
│   └── utilities/
│       └── common.py              # Shared utilities
└── old/                           # Legacy/archived files
```

## Cavity Tool Controls

| Key | Action |
|-----|--------|
| `,` / `.` | **Wobble preview** - peek at adjacent frame (raw, no mask) to see tool/tissue |
| `[` / `]` | Navigate to different frame - auto-saves current, auto-propagates forward |
| `D` | Cycle display mode (Original → Gamma → HSV-V) |
| `A` / `X` | Add / Exclude mode |
| `B` | Toggle brush mode |
| `+` / `-` | Adjust brush size |
| `Z` | Undo (enabled only after modifications) |
| `R` | Reset - erase mask for current frame |
| `Q` | Quit |

**Workflow tips:**
- **Auto-propagation** - press `]` to move forward, SAM2 automatically generates mask for next frame
- **Auto-save on navigation** - masks are saved automatically when you move between frames
- **Undo intelligently** - undo button is only enabled after you make modifications
- **Reset to clear** - use `R` to completely erase the mask and start over on current frame
- **No manual save needed** - everything saves automatically as you work
- **Frames are processed on-the-fly** - no pre-rendering needed

## Auto-Save on Close 🆕

**The tool now automatically saves all your work when you close it!**

### What Gets Saved
Masks are automatically saved in two ways:
1. **On frame navigation** - when you move to a different frame (using `[`, `]`, slider, or frame navigation)
2. **On tool close** - when you quit (Q, Escape, or X button)

A **session summary** (`session_summary.json`) is created on close with complete metadata about all annotated frames.

### Session Summary Contents
```json
{
  "video_name": "2025-03-25-Blue-frames",
  "video_path": "/path/to/video.mp4",
  "frame_range": {"start": 0, "end": 499},
  "total_frames_in_range": 500,
  "modified_frames_count": 25,
  "modified_frames": [10, 15, 20, 25, ...],
  "frames": [
    {
      "frame_number": 10,
      "has_mask": true,
      "cavity_points_pos": [[234, 567], [345, 678]],
      "cavity_points_neg": []
    },
    ...
  ]
}
```

### Reading Session Data
```bash
# View session summary
python scripts/utilities/read_session_summary.py \
  data/my-video/masks/cavity/session_summary.json

# Export frame list
python scripts/utilities/read_session_summary.py \
  data/my-video/masks/cavity/session_summary.json \
  --export-frames modified_frames.txt
```

### Benefits
- ✓ **Never lose work** - no need to remember to save every frame
- ✓ **Complete tracking** - know exactly which frames you annotated
- ✓ **Batch processing** - use session data for automated analysis
- ✓ **Resume work** - see what's done, what needs attention

## Detection Parameters

### Reflections (Two-Stage)
- **Stage 1 (Seeds)**: V ≥ 0.65 AND R < 0.40
- **Stage 2 (Expand)**: V ≥ 0.60 within 15px of seeds

### Blood
- V < 0.20 AND B < 0.20

## Troubleshooting

### "No videos found" or "No data directory found"
```bash
# Check you're in the right directory
pwd
# Should end with: 5-ALA-Video-Annotation (or your project folder name)

# Check data folder exists
ls data/
# Should show your video folder(s)

# Check video exists
ls data/YOUR_FOLDER_NAME/
# Should show your video file (.mp4, .avi, or .mov)
```

### Video not showing in dropdown
- Video must be inside a subfolder of `data/`, not directly in `data/`
- Supported formats: .mp4, .avi, .mov

### SAM2 not loading
```bash
# Check model file exists
ls -la models/sam2_checkpoints/
# Should show: sam2_hiera_large.pt (~900MB)
```

### "No module named X" errors
```bash
pip install -r requirements.txt
```

### Scripts can't find paths
- **Always run scripts from the project root directory**
- `cd` to the folder containing `scripts/`, `data/`, `README.md`
- Use relative paths: `data/my-video` not `/Users/name/...`


