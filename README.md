# NudeBlock

NudeBlock is a Python desktop application that uses ONNX-based AI object detection to mask NSFW content in images, videos, and live screen capture.

It is designed as a practical privacy tool and as a strong portfolio project that demonstrates real-time computer vision, desktop AI integration, and production-style application architecture.

## Highlights

- ONNX Runtime inference pipeline with automatic CPU/CUDA provider selection.
- Real-time object detection and masking for live screen protection.
- Multi-modal AI workflows: image, video, and live screen processing.
- PyQt6 desktop app architecture with responsive multi-threaded workers.
- Advanced video logic: class-aware object tracking, strict confirmation, and size-aware confidence filtering.
- Configurable detection controls: class selection, confidence threshold, IoU threshold, scan rate, and box scaling.

## Core Features

### 1) Secure Image

- Load an image and detect selected NudeNet classes.
- Mask detected regions with black boxes.
- Optional class label + confidence rendering.
- Preview output and save processed image.

### 2) Secure Video

- Process video frame-by-frame in a background worker.
- Optional trim window (start/end time).
- Class-aware tracking and confirmation flow to reduce unstable detections.
- Size-aware confidence mode:
  - smaller objects can pass with lower confidence
  - larger objects require higher confidence
- Optional labels with score and object area ratio.

### 3) Secure Screen (Live)

- Capture monitor frames using mss.
- Run real-time AI detection.
- Draw black masks in a transparent always-on-top overlay.
- Adjustable scans per second for performance tuning.
- System tray support for background protection flow.

## Tech Stack

The project dependencies are managed in requirements.txt:

- numpy
- opencv-python
- onnxruntime
- PyQt6
- mss
- keyboard

Notes:

- For NVIDIA acceleration, use onnxruntime-gpu instead of onnxruntime.
- Current desktop implementation is Windows-first.

## Project Structure

```
app/
  core/       # constants and typed settings
  services/   # shared detection + image workflow services
  ui/         # PyQt6 main window, overlay, theme
  workers/    # threaded video/screen processing
archive/      # original prototype scripts kept for reference
docs/         # architecture decisions and implementation notes
run.py        # application entry point
```

## Detection Flow (High Level)

1. Load ONNX model session in DetectionService.
2. Capture media frame (image/video/screen).
3. Run tiled detection + post-processing.
4. Filter by selected classes and confidence strategy.
5. Apply masking.
6. Optionally render labels and confidence metadata.

## Setup and Run

### 1) Create and activate virtual environment

Windows PowerShell:

```
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 2) Install packages

```
pip install -r requirements.txt
```

### 3) Launch app

```
python run.py
```

## Build Windows Executable (PyInstaller)

The repo includes run.spec.

Example:

```
pyinstaller run.spec
```

## Evolution Snapshot (From Git History)

- 2026-04-21: Project planning and starter docs/assets committed.
- 2026-04-21: Migrated prototype scripts into a modular PyQt6 desktop architecture.
- 2026-04-22: Improved video stability beyond a simple 5-frame approach.
- 2026-04-22: Introduced class-aware tracking with strict confirmation logic.
- 2026-04-23: Added optional label + score rendering.
- 2026-04-23: Added video size-aware confidence controls and tuning docs.
- 2026-04-23: Added object screen-area ratio to detection labels.
- 2026-04-23: Added packaging support artifacts (run.spec / build ignore updates).

## Keywords

ONNX, ONNX Runtime, Real-Time Object Detection, AI Integration, Computer Vision, PyQt6, OpenCV, Desktop AI App, Video Analytics, Screen Privacy, Python Multithreading, CUDA Inference.

## Responsible Use

This project is intended for privacy and safety use cases. Always comply with local laws, platform rules, and ethical content handling standards.
