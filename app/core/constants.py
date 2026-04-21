"""Shared constants used by the desktop app."""

APP_NAME = "NudeBlock"
MODEL_PATH = "nudenet_v8.onnx"

INFER_INPUT_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_BOX_SCALE = 1.0
DEFAULT_SCREEN_SCANS_PER_SECOND = 8

# Video-only temporal consistency defaults.
# Maximum consecutive missed frames allowed before a track is retired.
# In the video temporal post-process this controls gap bridging; at 30 FPS,
# 30 means we can bridge about 1.0 second of detector dropouts.
DEFAULT_VIDEO_OBJECT_CONSISTENCY_FRAMES = 30

# Lower validation/UI bound for gap-bridging tolerance.
# A value of 0 disables temporal consistency and keeps legacy per-frame
# behavior (detect and mask each frame independently with no lookahead).
MIN_VIDEO_OBJECT_CONSISTENCY_FRAMES = 0

# Upper validation/UI bound for gap-bridging tolerance.
# This cap limits over-persistence/ghost masking and bounds worst-case
# temporal linking cost; at 30 FPS, 120 corresponds to about 4 seconds.
MAX_VIDEO_OBJECT_CONSISTENCY_FRAMES = 120

# Minimum number of observed detections required for a track to emit output.
# Tracks shorter than this are treated as transient noise and suppressed.
DEFAULT_VIDEO_MIN_TRACK_FRAMES = 2

# IoU gate used in track association against predicted boxes.
# A candidate match with IoU below this value must satisfy distance gating
# to remain linkable, which helps preserve tracks during moderate motion.
DEFAULT_VIDEO_MATCH_IOU_THRESHOLD = 0.10

# Center-distance gate for association, normalized by average box diagonal:
# distance_ratio = center_distance / avg_diagonal. Lower values are stricter;
# 0.35 allows moderate displacement when IoU is temporarily weak.
DEFAULT_VIDEO_MATCH_DISTANCE_RATIO = 0.35

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
