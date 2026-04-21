# Development Rules

Read this file before making any code change in this project.

## Core Rules

1. Keep the codebase modular. Shared detection logic must live in reusable services, not be duplicated across image, video, and screen features.
2. Use clear design patterns where they improve maintainability. Prefer a layered structure with GUI, application services, domain/config models, and infrastructure adapters.
3. Do not hardcode user-facing settings. Blocking classes, confidence threshold, input and output paths, screen scan rate, and provider selection must be configurable.
4. Keep comments concise and useful. Add comments only where behavior, threading, inference flow, or coordinate mapping would otherwise be hard to follow.
5. Preserve a single source of truth for model metadata. Class labels, model path defaults, and post-processing constants should be defined centrally.
6. Avoid side effects unrelated to the selected feature. Processing a video must not shut down the machine or trigger unrelated system actions.
7. Validate inputs before processing. Check that files exist, images and videos open correctly, the model loads, and screen capture dependencies are available.
8. Keep the UI responsive. Long-running inference must run in worker threads or background tasks, never on the main GUI thread.
9. Favor explicit configuration objects over loose globals. Pass settings into services instead of reading mutable module-level constants.
10. Add every new Python dependency to requirements.txt as part of the same change that introduces it.

## Architecture Direction

1. Build the desktop app around PyQt6.
2. Extract a shared detection engine used by all three features.
3. Keep feature-specific behavior in separate workflows:
   - secure image: single image load, detect, mask, preview, save
   - secure video: decode frame-by-frame, detect, mask, encode, progress reporting
   - secure screen: live capture loop, detect, mask, overlay refresh, adjustable scan interval
4. Represent user settings with typed configuration models.
5. Prefer dependency injection for model session creation and feature services.

## Quality Expectations

1. New code should be testable even if the project starts without tests.
2. Public methods should have predictable inputs and outputs.
3. Error messages should help the user recover.
4. Avoid silent failures.
5. Update docs in the docs folder whenever architecture or workflow changes.

## Required Product Options

The GUI application must support all of the following:

1. The user can choose which NudeNet classes to block in all three features.
2. The user can choose the confidence threshold used to trigger masking.
3. The user can choose how often screen processing runs in the secure screen feature.
4. The same class-selection and threshold concepts should behave consistently across image, video, and screen modes.