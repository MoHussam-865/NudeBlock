"""Shared ONNX detection and masking logic for all app features."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

from app.core.constants import MODEL_PATH, NUDENET_CLASSES
from app.core.settings import DetectionSettings
from archive.tiling import detect_tiled_with_details


class DetectionService:
    SCORE_COMPARE_EPSILON = 1e-4

    def __init__(self, model_path: str = MODEL_PATH, prefer_cuda: bool = True):
        providers = ort.get_available_providers()
        if prefer_cuda and "CUDAExecutionProvider" in providers:
            chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.provider_name = "CUDAExecutionProvider"
        else:
            chosen = ["CPUExecutionProvider"]
            self.provider_name = "CPUExecutionProvider"

        self._session = ort.InferenceSession(model_path, providers=chosen)
        self._input_name = self._session.get_inputs()[0].name

    def detect_boxes(
        self,
        frame: np.ndarray,
        settings: DetectionSettings,
    ) -> list[tuple[int, int, int, int]]:
        detections = self.detect_boxes_with_classes(frame, settings)
        return [(bx, by, bw, bh) for (bx, by, bw, bh, _class_id) in detections]

    def detect_boxes_with_classes(
        self,
        frame: np.ndarray,
        settings: DetectionSettings,
    ) -> list[tuple[int, int, int, int, int]]:
        detections = self.detect_boxes_with_details(frame, settings)
        return [(bx, by, bw, bh, class_id) for (bx, by, bw, bh, class_id, _score) in detections]

    def detect_boxes_with_details(
        self,
        frame: np.ndarray,
        settings: DetectionSettings,
    ) -> list[tuple[int, int, int, int, int, float]]:
        settings.validate()
        frame_h, frame_w = frame.shape[:2]
        detections = detect_tiled_with_details(
            frame,
            self._session,
            self._input_name,
            settings.input_size,
            set(settings.selected_class_ids),
            settings.conf_threshold,
            settings.iou_threshold,
            frame_w,
            frame_h,
        )
        return self._scale_detections_from_center(
            detections,
            settings.box_scale,
            frame_w,
            frame_h,
        )

    @staticmethod
    def video_candidate_conf_threshold(settings: DetectionSettings) -> float:
        if not settings.enable_video_size_confidence:
            return settings.conf_threshold

        return min(settings.small_object_confidence, settings.large_object_confidence)

    @staticmethod
    def _box_area_ratio(
        box: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ) -> float:
        frame_area = max(1, frame_w * frame_h)
        box_area = max(0, box[2]) * max(0, box[3])
        ratio = float(box_area / frame_area)
        return min(1.0, max(0.0, ratio))

    @staticmethod
    def _size_aware_threshold_for_ratio(
        area_ratio: float,
        small_confidence: float,
        large_confidence: float,
        middle_area_ratio: float,
    ) -> float:
        # Step rule requested by user:
        # - area < middle boundary uses small threshold
        # - area >= middle boundary uses max threshold
        if area_ratio >= middle_area_ratio:
            return large_confidence
        return small_confidence

    def filter_video_detections_by_size_confidence(
        self,
        detections: Iterable[tuple[int, int, int, int, int, float]],
        frame_w: int,
        frame_h: int,
        settings: DetectionSettings,
    ) -> list[tuple[int, int, int, int, int, float]]:
        detections_list = list(detections)
        if not settings.enable_video_size_confidence:
            return detections_list

        accepted: list[tuple[int, int, int, int, int, float]] = []
        for bx, by, bw, bh, class_id, score in detections_list:
            area_ratio = self._box_area_ratio((bx, by, bw, bh), frame_w, frame_h)
            required_score = self._size_aware_threshold_for_ratio(
                area_ratio,
                settings.small_object_confidence,
                settings.large_object_confidence,
                settings.size_curve_max_area_ratio,
            )
            if score + self.SCORE_COMPARE_EPSILON >= required_score:
                accepted.append((bx, by, bw, bh, class_id, score))

        return accepted

    @staticmethod
    def _scale_detections_from_center(
        detections: Iterable[tuple[int, int, int, int, int, float]],
        scale: float,
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[int, int, int, int, int, float]]:
        if scale <= 1.0:
            return list(detections)

        scaled: list[tuple[int, int, int, int, int, float]] = []
        for bx, by, bw, bh, class_id, score in detections:
            cx = bx + (bw / 2.0)
            cy = by + (bh / 2.0)

            half_w = (bw * scale) / 2.0
            half_h = (bh * scale) / 2.0

            x1 = max(0, int(round(cx - half_w)))
            y1 = max(0, int(round(cy - half_h)))
            x2 = min(frame_w, int(round(cx + half_w)))
            y2 = min(frame_h, int(round(cy + half_h)))

            if x2 > x1 and y2 > y1:
                scaled.append((x1, y1, x2 - x1, y2 - y1, class_id, score))

        return scaled

    @staticmethod
    def draw_labels(
        frame: np.ndarray,
        detections: Iterable[tuple[int, int, int, int, int, float]],
    ) -> np.ndarray:
        labeled = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        baseline_pad = 4

        for bx, by, bw, _bh, class_id, score in detections:
            if class_id < 0 or class_id >= len(NUDENET_CLASSES):
                label_name = f"class_{class_id}"
            else:
                label_name = NUDENET_CLASSES[class_id]

            text = f"{label_name} {score:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            text_x = max(0, min(bx, labeled.shape[1] - text_w - 1))
            top_y = by - text_h - baseline_pad - baseline
            if top_y < 0:
                top_y = min(labeled.shape[0] - text_h - baseline - baseline_pad, by + baseline_pad)
            top_y = max(0, top_y)
            bottom_y = min(labeled.shape[0] - 1, top_y + text_h + baseline + baseline_pad)
            right_x = min(labeled.shape[1] - 1, text_x + text_w + 4)

            cv2.rectangle(labeled, (text_x, top_y), (right_x, bottom_y), (0, 0, 0), -1)
            cv2.putText(
                labeled,
                text,
                (text_x + 2, min(labeled.shape[0] - 1, bottom_y - baseline - 2)),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return labeled

    @staticmethod
    def apply_mask_and_labels(
        frame: np.ndarray,
        detections: Iterable[tuple[int, int, int, int, int, float]],
    ) -> np.ndarray:
        detections_list = list(detections)
        masked = DetectionService.apply_mask(
            frame,
            ((bx, by, bw, bh) for (bx, by, bw, bh, _class_id, _score) in detections_list),
        )
        return DetectionService.draw_labels(masked, detections_list)

    @staticmethod
    def apply_mask(
        frame: np.ndarray,
        boxes: Iterable[tuple[int, int, int, int]],
    ) -> np.ndarray:
        masked = frame.copy()
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(masked, (bx, by), (bx + bw, by + bh), (0, 0, 0), -1)
        return masked
