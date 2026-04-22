"""Image workflow service."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.settings import DetectionSettings
from app.services.detection_service import DetectionService


class ImageService:
    def __init__(self, detector: DetectionService):
        self._detector = detector

    def process_image(
        self,
        image_path: str,
        settings: DetectionSettings,
        output_path: str | None = None,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        detections = self._detector.detect_boxes_with_details(image, settings)
        boxes = [(bx, by, bw, bh) for (bx, by, bw, bh, _class_id, _score) in detections]
        if settings.show_labels_and_scores:
            masked = self._detector.apply_mask_and_labels(image, detections)
        else:
            masked = self._detector.apply_mask(image, boxes)

        if output_path:
            ok = cv2.imwrite(output_path, masked)
            if not ok:
                raise ValueError(f"Could not save output image: {output_path}")

        return masked, boxes
