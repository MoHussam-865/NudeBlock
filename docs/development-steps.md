# Development Steps

This file summarizes the current delivery steps for the desktop application effort.

## Step Status

1. Review current Python scripts and shared helper logic. Status: completed.
2. Document current implementation behavior and main risks. Status: completed.
3. Define development rules and architecture constraints. Status: completed.
4. Establish dependency baseline in requirements.txt. Status: completed.
5. Design the desktop GUI architecture and delivery plan. Status: completed.
6. Extract shared detection core from prototype scripts. Status: pending.
7. Build GUI shell for secure image, secure video, and secure screen. Status: pending.
8. Add user controls for classes, confidence, and screen processing rate. Status: pending.
9. Replace prototype entry scripts with reusable app workflows. Status: pending.
10. Add tests for configuration, detection filtering, and feature workflows. Status: pending.
11. Package the desktop application for Windows. Status: pending.

## Immediate Next Build Targets

1. Create an application package structure instead of standalone prototype scripts.
2. Move the shared class list and model settings into one module.
3. Wrap ONNX session creation in a reusable service.
4. Build a settings model that includes selected classes, confidence threshold, and screen scan interval.
5. Implement a GUI wireframe before adding full processing controls.

## Maintenance Rule

Whenever a new dependency is introduced, update requirements.txt in the same change.