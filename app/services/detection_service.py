"""Shared ONNX detection and masking logic for all app features."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

from app.core.constants import MODEL_PATH
from app.core.settings import DetectionSettings
from archive.tiling import detect_tiled


class DetectionService:
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
        settings.validate()
        frame_h, frame_w = frame.shape[:2]
        boxes = detect_tiled(
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
        return self._scale_boxes_from_center(boxes, settings.box_scale, frame_w, frame_h)

    @staticmethod
    def _scale_boxes_from_center(
        boxes: Iterable[tuple[int, int, int, int]],
        scale: float,
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[int, int, int, int]]:
        if scale <= 1.0:
            return list(boxes)

        scaled: list[tuple[int, int, int, int]] = []
        for bx, by, bw, bh in boxes:
            cx = bx + (bw / 2.0)
            cy = by + (bh / 2.0)

            half_w = (bw * scale) / 2.0
            half_h = (bh * scale) / 2.0

            x1 = max(0, int(round(cx - half_w)))
            y1 = max(0, int(round(cy - half_h)))
            x2 = min(frame_w, int(round(cx + half_w)))
            y2 = min(frame_h, int(round(cy + half_h)))

            if x2 > x1 and y2 > y1:
                scaled.append((x1, y1, x2 - x1, y2 - y1))

        return scaled

    @staticmethod
    def apply_mask(
        frame: np.ndarray,
        boxes: Iterable[tuple[int, int, int, int]],
    ) -> np.ndarray:
        masked = frame.copy()
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(masked, (bx, by), (bx + bw, by + bh), (0, 0, 0), -1)
        return masked
