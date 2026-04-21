# Desktop App Plan

This plan describes how to turn the current prototype scripts into a maintainable desktop application.

## Product Goal

Build a Windows desktop application with three features:

1. Secure Image: open an image, detect selected classes, mask them, preview, and save the result.
2. Secure Video: open a video, process it frame by frame, mask selected classes, and save the output video.
3. Secure Screen: capture the screen live, detect selected classes, and display a masking overlay in real time.

## Required User Controls

The application must expose these options in the GUI for all relevant features:

1. A class-selection control that lets the user choose exactly which NudeNet classes to block.
2. A confidence threshold control that lets the user adjust model sensitivity.
3. A screen processing rate control for secure screen, expressed as scans per second or interval in milliseconds.

## Proposed Architecture

### 1. App Shell

Use PyQt6 to build the desktop UI.

Recommended structure:

1. app/main.py: application entry point.
2. app/ui/: windows, dialogs, reusable widgets.
3. app/core/: model metadata, configuration models, shared constants.
4. app/services/: model session manager, image service, video service, screen service.
5. app/workers/: background threads for long-running inference.
6. app/utils/: path helpers, logging, validation.

### 2. Shared Detection Engine

Create one reusable detection layer with these responsibilities:

1. Load the ONNX model once and share the session safely where appropriate.
2. Store the 18 NudeNet class labels in one place.
3. Normalize input frames into model tensors.
4. Run direct inference or tiled inference depending on workflow.
5. Filter detections by selected classes and confidence threshold.
6. Return a consistent detection model used by all three features.

### 3. Feature Services

Implement one service per feature:

1. Image service:
   - load image
   - run detection
   - apply masking
   - return preview and save output
2. Video service:
   - open source video
   - process frames in a worker thread
   - emit progress and cancellation state
   - save output video
3. Screen service:
   - capture monitor frames
   - control scan interval
   - emit boxes to overlay window
   - start and stop safely from the GUI

### 4. Settings and State

Use typed configuration objects such as dataclasses for:

1. model path
2. provider selection
3. selected classes
4. confidence threshold
5. IOU threshold
6. screen scan interval
7. output paths and overwrite behavior

Persist user settings in a lightweight configuration file later if needed.

## Packages To Use

1. PyQt6: desktop GUI, signals, windows, dialogs, threading integration.
2. opencv-python: image and video I/O, drawing, frame conversion, NMS helpers.
3. numpy: array processing for image tensors and captured frames.
4. onnxruntime: ONNX model inference on CPU.
5. mss: fast screen capture for the secure screen feature.
6. keyboard: only if a global hotkey remains necessary. Prefer native GUI controls and Qt shortcuts when possible.

Optional runtime choice:

1. onnxruntime-gpu can replace onnxruntime on supported NVIDIA systems.

Future packaging options:

1. pyinstaller for distributing the desktop application on Windows.

## Development Sequence

1. Extract shared constants and class metadata from the prototype scripts.
2. Create a detection service that wraps model loading and inference.
3. Refactor tiling.py into the shared detection layer.
4. Implement masking helpers for image, video frames, and screen boxes.
5. Build the main window with three feature tabs or a left navigation layout.
6. Add settings controls for selected classes, confidence, and screen scan rate.
7. Move video and screen processing into worker threads.
8. Add progress reporting, validation messages, and cancellation.
9. Remove prototype-only behavior such as console input and machine shutdown.
10. Add tests and package the application.

## Design Patterns To Apply

1. Service layer pattern for detection and media workflows.
2. Strategy pattern if multiple masking styles are later supported.
3. Factory pattern for inference provider selection or session creation.
4. Model-view separation so the UI does not own inference logic.

## Risks To Address Early

1. Real-time screen inference may be CPU-heavy without careful throttling and worker design.
2. ONNX runtime provider behavior differs between CPU and CUDA installations.
3. Overlay and capture behavior is Windows-specific and must be tested carefully on high-DPI systems.
4. Video processing needs cancellation and progress updates to avoid a frozen user experience.