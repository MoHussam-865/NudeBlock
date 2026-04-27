"""QThread worker for live secure screen detection."""

from __future__ import annotations

import time

import mss
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from app.core.constants import SCREEN_METRICS_REPORT_INTERVAL_MS
from app.core.settings import DetectionSettings, ScreenSettings
from app.services.detection_service import DetectionService


class ScreenWorker(QThread):
    boxes_ready = pyqtSignal(list)
    status_changed = pyqtSignal(str)
    metrics_changed = pyqtSignal(str)
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
                provider_name = getattr(self._detector, "provider_name", "unknown")
                report_started = time.perf_counter()
                loops = 0
                empty_loops = 0
                capture_total_ms = 0.0
                detect_total_ms = 0.0
                total_loop_ms = 0.0

                self.status_changed.emit(
                    f"Screen protection started on {provider_name}."
                )
                while self._running:
                    loop_started = time.perf_counter()
                    grabbed = sct.grab(self._monitor)
                    grabbed_at = time.perf_counter()
                    frame = np.array(grabbed, dtype=np.uint8)[:, :, :3].copy()

                    boxes = self._detector.detect_boxes(frame, self._detection_settings)
                    detected_at = time.perf_counter()
                    self.boxes_ready.emit(boxes)
                    self.msleep(self._screen_settings.interval_ms)
                    loop_finished = time.perf_counter()

                    loops += 1
                    if not boxes:
                        empty_loops += 1

                    capture_total_ms += (grabbed_at - loop_started) * 1000.0
                    detect_total_ms += (detected_at - grabbed_at) * 1000.0
                    total_loop_ms += (loop_finished - loop_started) * 1000.0

                    if (loop_finished - report_started) * 1000.0 < SCREEN_METRICS_REPORT_INTERVAL_MS:
                        continue

                    elapsed_sec = max(loop_finished - report_started, 1e-6)
                    self.metrics_changed.emit(
                        self._build_metrics_message(
                            provider_name,
                            loops,
                            empty_loops,
                            capture_total_ms,
                            detect_total_ms,
                            total_loop_ms,
                            elapsed_sec,
                        )
                    )

                    report_started = loop_finished
                    loops = 0
                    empty_loops = 0
                    capture_total_ms = 0.0
                    detect_total_ms = 0.0
                    total_loop_ms = 0.0

            self.boxes_ready.emit([])
            self.metrics_changed.emit("")
            self.status_changed.emit("Screen protection stopped.")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _build_metrics_message(
        self,
        provider_name: str,
        loops: int,
        empty_loops: int,
        capture_total_ms: float,
        detect_total_ms: float,
        total_loop_ms: float,
        elapsed_sec: float,
    ) -> str:
        if loops <= 0:
            return ""

        avg_capture_ms = capture_total_ms / loops
        avg_detect_ms = detect_total_ms / loops
        avg_loop_ms = total_loop_ms / loops
        effective_scans = loops / elapsed_sec
        empty_ratio = (empty_loops / loops) * 100.0

        return (
            f"Effective {effective_scans:.1f}/s | Loop {avg_loop_ms:.0f} ms | "
            f"Capture {avg_capture_ms:.0f} ms | "
            f"Detect {avg_detect_ms:.0f} ms | Empty {empty_ratio:.0f}%"
        )
