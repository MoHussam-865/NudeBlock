# Video Object Consistency Summary

## Problem

Secure Video processing was frame-by-frame only. When the model missed an object for a short period (for example 1 to 30 frames), the black mask disappeared temporarily even though the object was still present.

This caused visible flicker and privacy leakage in the output video.

## Root Cause

The previous video pipeline used only current-frame detections:

1. Read frame.
2. Detect boxes.
3. Mask boxes.
4. Write frame.

No temporal memory or future-frame context was used.

## Solution Overview

A new video-only temporal consistency pipeline has been implemented.

Important scope decision:

1. Applied only to Secure Video.
2. Not applied to Secure Screen, because real-time mode cannot depend on future frames.

The video pipeline now supports a future-aware consistency pass when enabled.

## What Was Implemented

### 1. Video-only settings

Added video consistency defaults and limits in [app/core/constants.py](../app/core/constants.py):

1. `DEFAULT_VIDEO_OBJECT_CONSISTENCY_FRAMES = 30`
2. `MIN_VIDEO_OBJECT_CONSISTENCY_FRAMES = 0`
3. `MAX_VIDEO_OBJECT_CONSISTENCY_FRAMES = 120`
4. Internal matching defaults for temporal linking.

Added `VideoConsistencySettings` in [app/core/settings.py](../app/core/settings.py) with validation:

1. `object_consistency_frames`
2. `min_track_frames`
3. `match_iou_threshold`
4. `match_distance_ratio`

### 2. Temporal consistency service

Created [app/services/video_temporal_consistency.py](../app/services/video_temporal_consistency.py).

This module:

1. Builds object tracks across frames.
2. Matches detections by overlap and center distance.
3. Bridges short detection gaps up to the configured frame limit.
4. Interpolates boxes inside bridged gaps (middle-position masks).
5. Suppresses short-lived transient tracks.

### 3. Video worker integration

Refactored [app/workers/video_worker.py](../app/workers/video_worker.py):

1. If consistency is disabled (`0`), keep single-pass behavior.
2. If consistency is enabled (`>0`), run a 3-stage flow:
   - Stage 1: detect and collect raw boxes for each frame.
   - Stage 2: stabilize boxes with temporal post-processing.
   - Stage 3: re-render video using stabilized boxes.

Progress and status messages are updated per stage.

### 4. UI wiring (video tab only)

Updated [app/ui/main_window.py](../app/ui/main_window.py):

1. Added `Object consistency (video only)` controls in the Secure Video tab.
2. Added `Gap tolerance (frames)` spinbox.
3. Added explanatory hint: at 30 FPS, 30 frames is about 1 second.
4. Passed video consistency settings only to `VideoWorker`.

No screen worker changes were made for this feature.

## Behavior After Change

When the same object disappears briefly from detections and reappears within the configured frame window:

1. The gap is treated as the same object.
2. The mask is interpolated through missing frames.
3. Short one-off false detections are reduced via minimum track length filtering.

## Notes

1. Higher consistency values improve continuity but may keep masks longer after true object disappearance.
2. Setting the value to `0` disables temporal consistency and returns to near-original frame-by-frame behavior.
3. This implementation is intentionally offline for video so it can use future frames for better stability.
