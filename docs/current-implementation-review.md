# Current Implementation Review

This document explains how the existing scripts work and identifies the main issues that should be addressed before building the desktop GUI application.

## Shared Detection Flow

The current project uses the NudeNet YOLOv8 ONNX model in two different ways:

1. secure_image.py performs a direct full-image inference pass.
2. secure_video.py and secure_screen.py call tiling.py to split large frames into overlapping tiles, run inference per tile, and merge the results with global non-maximum suppression.

The model output is interpreted as:

1. 4 box values: center x, center y, width, height.
2. 18 class scores matching the NudeNet class list.

The scripts then convert boxes back to source-frame coordinates and draw or fill black rectangles over detected regions.

## File Behavior

### secure_image.py

Purpose:
Run one inference pass on a single image and draw red detection boxes with labels.

How it works:

1. Loads the ONNX model with onnxruntime on CPU.
2. Reads an image path from console input.
3. Resizes the image into a 640x640 YOLO input blob.
4. Runs inference.
5. Filters detections by confidence threshold.
6. Applies OpenCV NMS.
7. Draws rectangles and labels on the original image.
8. Shows the result in an OpenCV window.

Notes:

1. This script currently visualizes detections instead of masking them.
2. It does not let the user choose which classes to block.
3. It does not validate whether cv2.imread succeeded before using the image.

### secure_video.py

Purpose:
Open a video, process each frame, detect blocked classes, fill the detected regions with black, and save a new output video.

How it works:

1. Loads the ONNX model with GPU if CUDA is available, otherwise CPU.
2. Opens a video file with OpenCV.
3. Reads width, height, fps, and frame count from the source video.
4. Writes processed frames to a new mp4 file.
5. Runs tiled detection for each frame through tiling.detect_tiled.
6. Draws filled black rectangles over detected regions.
7. Prints progress every 10 frames.

Notes:

1. This script masks all 18 classes by default.
2. The input file is currently hardcoded.
3. The output path is currently hardcoded.
4. The confidence threshold is currently hardcoded.
5. After processing, it starts a Windows shutdown command, which is not acceptable application behavior for the desktop product.

### secure_screen.py

Purpose:
Capture the primary screen continuously, detect blocked classes, and paint black rectangles in a transparent always-on-top overlay.

How it works:

1. Uses mss to capture the primary monitor.
2. Starts a PyQt6 overlay window that stays on top and ignores mouse input.
3. Starts a worker thread that repeatedly grabs the screen and runs tiled detection.
4. Emits detection boxes to the overlay through a Qt signal.
5. Paints opaque black rectangles for current detections.
6. Uses a hold-frame mechanism to reduce flicker.
7. Uses a global hotkey through keyboard to quit.

Notes:

1. Class selection is hardcoded to all classes.
2. Confidence threshold is hardcoded.
3. Processing frequency is indirectly tied to the worker loop and overlay refresh constant, not a user-configurable screen scan setting.
4. The script is Windows-oriented because of DPI handling and display affinity capture exclusion.

### tiling.py

Purpose:
Provide the reusable tile-based detection workflow for high-resolution frames.

How it works:

1. Computes overlapping tiles that cover the whole frame.
2. Runs inference on each tile.
3. Filters detections by selected classes and confidence.
4. Remaps tile-local boxes into full-frame coordinates.
5. Applies a final global NMS pass.
6. Clamps output boxes to frame bounds.

This is the best starting point for the shared detection engine because it already contains the reusable inference and post-processing logic for video and screen workflows.

## Review Findings

### High Priority

1. secure_video.py contains a forced Windows shutdown command after successful processing. See [secure_video.py](../secure_video.py#L110). This is a dangerous side effect and must be removed before GUI integration.
2. The required product options are not implemented. Class selection, model sensitivity, and screen scan frequency are all hardcoded across the scripts. See [secure_image.py](../secure_image.py#L8), [secure_video.py](../secure_video.py#L12), and [secure_screen.py](../secure_screen.py#L35).

### Medium Priority

1. The image workflow asks for a video filename in its console prompt, which is misleading. See [secure_image.py](../secure_image.py#L7).
2. The image workflow does not guard against a failed image load before accessing shape information. See [secure_image.py](../secure_image.py#L41).
3. The video workflow hardcodes the input file instead of accepting a user-selected path. See [secure_video.py](../secure_video.py#L17).
4. The screen workflow masks all classes by default with no user control. See [secure_screen.py](../secure_screen.py#L63).

### Low Priority

1. There is duplicated class metadata across multiple files.
2. Configuration is spread across module-level constants instead of configuration objects.
3. The current scripts are useful prototypes but not yet structured for a maintainable GUI application.

## Recommended Refactor Direction

1. Extract model loading, class metadata, preprocessing, and post-processing into shared modules.
2. Replace module-level constants with configuration dataclasses or typed settings models.
3. Build feature services for image, video, and screen workflows on top of the shared detection engine.
4. Put all user-adjustable settings in the GUI and persist them in a simple application configuration file.
5. Remove console-driven and hardcoded behavior from the product code path.