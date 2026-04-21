"""Typed settings models used by services and workers."""

from dataclasses import dataclass, field

from app.core.constants import (
    DEFAULT_BOX_SCALE,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCREEN_SCANS_PER_SECOND,
    INFER_INPUT_SIZE,
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
