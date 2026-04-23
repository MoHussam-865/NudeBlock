# Video Size-Aware Confidence Changes

Date: 2026-04-23

## Goal

Improve video masking accuracy by using a size-aware confidence rule:

- Small detections can pass with lower confidence.
- Large detections require higher confidence.

This targets the observed pattern where some small true positives have low scores, while some medium/large low-score detections are false positives.

## Scope

Implemented for video processing only.

Unchanged behavior:

- Secure Image keeps the existing fixed confidence flow.
- Secure Screen keeps the existing fixed confidence flow.
- Existing tracking and pessimistic mask-hold logic remains active.

## What Changed

## 1) New constants

File: app/core/constants.py

Added defaults for video size-aware confidence tuning:

- DEFAULT_VIDEO_SIZE_AWARE_CONFIDENCE_ENABLED = True
- DEFAULT_VIDEO_SMALL_OBJECT_CONFIDENCE = 0.10
- DEFAULT_VIDEO_LARGE_OBJECT_CONFIDENCE = 0.40
- DEFAULT_VIDEO_SIZE_CURVE_MAX_AREA_RATIO = 0.150

These defaults keep the new behavior disabled by default to preserve existing behavior unless explicitly enabled.

## 2) Detection settings extended

File: app/core/settings.py

DetectionSettings now includes video-only tuning fields:

- enable_video_size_confidence
- small_object_confidence
- large_object_confidence
- size_curve_max_area_ratio

Validation was added for ranges and size-aware relationships when enabled:

- Confidence values in [0.0, 1.0]
- Area ratios in [0.0, 1.0]
- small_object_confidence <= large_object_confidence
- middle boundary area ratio > 0

## 3) Detection service size-aware filtering helpers

File: app/services/detection_service.py

Added helper methods:

- video_candidate_conf_threshold(settings)
- _box_area_ratio(box, frame_w, frame_h)
- _size_aware_threshold_for_ratio(area_ratio, ...)
- filter_video_detections_by_size_confidence(detections, frame_w, frame_h, settings)

Behavior:

1. Candidate detections are collected with a permissive threshold in video mode when size-aware is enabled.
2. Final accept/reject is done by comparing each detection score against a linear threshold based on box area ratio.

## 4) Video worker integration

File: app/workers/video_worker.py

Video flow now does the following when size-aware mode is enabled:

1. Uses a candidate confidence threshold that is low enough to keep potential small true positives.
2. Applies size-aware score filtering before tracking/masking.
3. Keeps existing immediate masking for accepted detections.
4. Keeps existing track confirmation and short mask-hold behavior.

A status message now includes active size-aware settings when enabled.

## 5) Video UI controls

File: app/ui/main_window.py

Added a new group in the Secure Video tab:

- Object tracking toggle
- Small-object min confidence
- Max-object min confidence
- Middle boundary (% frame area)

Wiring:

- New values are applied only when starting video processing.
- When object tracking is off, masking is per-frame only (no temporal confirmation/hold).
- Image and screen continue using existing shared controls and behavior.

## Size-Aware Rule

For each video detection, let:

- area_ratio = (box_width * box_height) / (frame_width * frame_height)
- small_conf = small-object min confidence
- large_conf = max-object min confidence
- middle_ratio = user-configured middle boundary ratio

Threshold used for acceptance:

- If area_ratio < middle_ratio: use small_conf
- If area_ratio >= middle_ratio: use large_conf

With defaults (small=0.10, max=0.40, middle=15%),
objects smaller than 15% use 0.10 and 15%+ use 0.40.

Detection is accepted when:

- score >= required_threshold

## Manual Tuning Guide

Start with defaults and test on representative clips.

Suggested loop:

1. Enable size-aware confidence.
2. If small true positives are still missed, reduce small-object min confidence.
3. If medium/large false positives remain, increase large-object min confidence.
4. Adjust the middle boundary to move where the threshold steps from small to max.
5. Re-test until recall and false-positive tradeoff is acceptable.

## Regression Safety Notes

- Feature is opt-in (disabled by default).
- Changes are isolated to video path usage.
- Image and screen pipelines were not switched to size-aware filtering.

## Files Updated

- app/core/constants.py
- app/core/settings.py
- app/services/detection_service.py
- app/workers/video_worker.py
- app/ui/main_window.py
- docs/video-size-aware-confidence-changes.md
