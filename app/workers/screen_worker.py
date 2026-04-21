"""QThread worker for live secure screen detection."""

from __future__ import annotations

import mss
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from app.core.settings import DetectionSettings, ScreenSettings
from app.services.detection_service import DetectionService


class ScreenWorker(QThread):
    boxes_ready = pyqtSignal(list)
    status_changed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        detector: DetectionService,
        monitor: dict,
        detection_settings: DetectionSettings,
        screen_settings: ScreenSettings,
    ):
        super().__init__()
        self._detector = detector
        self._monitor = monitor
        self._detection_settings = detection_settings
        self._screen_settings = screen_settings
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            with mss.mss() as sct:
                self.status_changed.emit("Screen protection started.")
                while self._running:
                    grabbed = sct.grab(self._monitor)
                    frame = np.array(grabbed, dtype=np.uint8)[:, :, :3].copy()

                    boxes = self._detector.detect_boxes(frame, self._detection_settings)
                    self.boxes_ready.emit(boxes)
                    self.msleep(self._screen_settings.interval_ms)

            self.boxes_ready.emit([])
            self.status_changed.emit("Screen protection stopped.")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
