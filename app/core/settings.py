"""Typed settings models used by services and workers."""

from dataclasses import dataclass, field

from app.core.constants import (
    DEFAULT_BOX_SCALE,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_VIDEO_MATCH_DISTANCE_RATIO,
    DEFAULT_VIDEO_MATCH_IOU_THRESHOLD,
    DEFAULT_VIDEO_MIN_TRACK_FRAMES,
    DEFAULT_VIDEO_OBJECT_CONSISTENCY_FRAMES,
    DEFAULT_SCREEN_SCANS_PER_SECOND,
    INFER_INPUT_SIZE,
    MAX_VIDEO_OBJECT_CONSISTENCY_FRAMES,
    MIN_VIDEO_OBJECT_CONSISTENCY_FRAMES,
    NUDENET_CLASSES,
)


@dataclass(frozen=True)
class DetectionSettings:
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    box_scale: float = DEFAULT_BOX_SCALE
    input_size: int = INFER_INPUT_SIZE
    selected_class_ids: frozenset[int] = field(
        default_factory=lambda: frozenset(range(len(NUDENET_CLASSES)))
    )

    def validate(self) -> None:
        if not 0.0 <= self.conf_threshold <= 1.0:
            raise ValueError("Confidence threshold must be in [0.0, 1.0].")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("IoU threshold must be in [0.0, 1.0].")
        if self.box_scale < 1.0:
            raise ValueError("Box scale must be >= 1.0.")
        if self.input_size <= 0:
            raise ValueError("Input size must be > 0.")
        if not self.selected_class_ids:
            raise ValueError("At least one class must be selected.")


@dataclass(frozen=True)
class ScreenSettings:
    scans_per_second: int = DEFAULT_SCREEN_SCANS_PER_SECOND

    @property
    def interval_ms(self) -> int:
        value = max(1, self.scans_per_second)
        return max(1, int(1000 / value))


@dataclass(frozen=True)
class VideoConsistencySettings:
    object_consistency_frames: int = DEFAULT_VIDEO_OBJECT_CONSISTENCY_FRAMES
    min_track_frames: int = DEFAULT_VIDEO_MIN_TRACK_FRAMES
    match_iou_threshold: float = DEFAULT_VIDEO_MATCH_IOU_THRESHOLD
    match_distance_ratio: float = DEFAULT_VIDEO_MATCH_DISTANCE_RATIO

    @property
    def enabled(self) -> bool:
        return self.object_consistency_frames > 0

    def validate(self) -> None:
        if not (
            MIN_VIDEO_OBJECT_CONSISTENCY_FRAMES
            <= self.object_consistency_frames
            <= MAX_VIDEO_OBJECT_CONSISTENCY_FRAMES
        ):
            raise ValueError(
                "Object consistency frames must be in "
                f"[{MIN_VIDEO_OBJECT_CONSISTENCY_FRAMES}, {MAX_VIDEO_OBJECT_CONSISTENCY_FRAMES}]."
            )
        if self.min_track_frames < 1:
            raise ValueError("Minimum track frames must be >= 1.")
        if not 0.0 <= self.match_iou_threshold <= 1.0:
            raise ValueError("Video match IoU threshold must be in [0.0, 1.0].")
        if not 0.0 <= self.match_distance_ratio <= 1.0:
            raise ValueError("Video match distance ratio must be in [0.0, 1.0].")
