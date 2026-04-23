"""Shared constants used by the desktop app."""

APP_NAME = "NudeBlock"
MODEL_PATH = "nudenet_v8.onnx"

INFER_INPUT_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_BOX_SCALE = 1.25
DEFAULT_SCREEN_SCANS_PER_SECOND = 8
# Enabled by default for video so size-based thresholds are used out of the box.
DEFAULT_VIDEO_SIZE_AWARE_CONFIDENCE_ENABLED = True
# Enabled by default so video keeps temporal confirmation and short mask-hold behavior.
DEFAULT_VIDEO_OBJECT_TRACKING_ENABLED = True
# Minimum confidence used for objects smaller than the middle boundary.
DEFAULT_VIDEO_SMALL_OBJECT_CONFIDENCE = 0.10
# Maximum confidence used for larger boxes (>= middle boundary).
DEFAULT_VIDEO_LARGE_OBJECT_CONFIDENCE = 0.40
# User-tunable middle boundary where max confidence fully applies (0.150 = 15.0%).
DEFAULT_VIDEO_SIZE_CURVE_MAX_AREA_RATIO = 0.150

NUDENET_CLASSES = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]
