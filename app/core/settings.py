"""Typed settings models used by services and workers."""

from dataclasses import dataclass, field

from app.core.constants import (
    DEFAULT_BOX_SCALE,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCREEN_SCANS_PER_SECOND,
    DEFAULT_VIDEO_LARGE_OBJECT_CONFIDENCE,
    DEFAULT_VIDEO_OBJECT_TRACKING_ENABLED,
    DEFAULT_VIDEO_SIZE_AWARE_CONFIDENCE_ENABLED,
    DEFAULT_VIDEO_SIZE_CURVE_MAX_AREA_RATIO,
    DEFAULT_VIDEO_SMALL_OBJECT_CONFIDENCE,
    INFER_INPUT_SIZE,
    NUDENET_CLASSES,
)


@dataclass(frozen=True)
class DetectionSettings:
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    box_scale: float = DEFAULT_BOX_SCALE
    show_labels_and_scores: bool = True
    input_size: int = INFER_INPUT_SIZE
    enable_video_size_confidence: bool = DEFAULT_VIDEO_SIZE_AWARE_CONFIDENCE_ENABLED
    video_object_tracking: bool = DEFAULT_VIDEO_OBJECT_TRACKING_ENABLED
    small_object_confidence: float = DEFAULT_VIDEO_SMALL_OBJECT_CONFIDENCE
    large_object_confidence: float = DEFAULT_VIDEO_LARGE_OBJECT_CONFIDENCE
    size_curve_max_area_ratio: float = DEFAULT_VIDEO_SIZE_CURVE_MAX_AREA_RATIO
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
        if not 0.0 <= self.small_object_confidence <= 1.0:
            raise ValueError("Small-object confidence must be in [0.0, 1.0].")
        if not 0.0 <= self.large_object_confidence <= 1.0:
            raise ValueError("Large-object confidence must be in [0.0, 1.0].")
        if not 0.0 <= self.size_curve_max_area_ratio <= 1.0:
            raise ValueError("Middle boundary area ratio must be in [0.0, 1.0].")
        if self.enable_video_size_confidence:
            if self.small_object_confidence > self.large_object_confidence:
                raise ValueError(
                    "Small-object confidence must be <= large-object confidence "
                    "when size-aware confidence is enabled."
                )
            if self.size_curve_max_area_ratio <= 0.0:
                raise ValueError(
                    "Middle boundary area ratio must be > 0.0 when "
                    "size-aware confidence is enabled."
                )
        if not self.selected_class_ids:
            raise ValueError("At least one class must be selected.")


@dataclass(frozen=True)
class ScreenSettings:
    scans_per_second: int = DEFAULT_SCREEN_SCANS_PER_SECOND

    @property
    def interval_ms(self) -> int:
        value = max(1, self.scans_per_second)
        return max(1, int(1000 / value))
