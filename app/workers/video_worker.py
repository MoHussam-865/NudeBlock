"""QThread worker that processes videos frame by frame."""

from __future__ import annotations

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from app.core.settings import DetectionSettings
from app.services.detection_service import DetectionService


class VideoWorker(QThread):
    progress_changed = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        detector: DetectionService,
        input_path: str,
        output_path: str,
        settings: DetectionSettings,
        start_time_sec: float | None = None,
        end_time_sec: float | None = None,
    ):
        super().__init__()
        self._detector = detector
        self._input_path = input_path
        self._output_path = output_path
        self._settings = settings
        self._start_time_sec = start_time_sec
        self._end_time_sec = end_time_sec
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        cap = None
        out = None

        try:
            cap = cv2.VideoCapture(self._input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open input video: {self._input_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            start_time_sec = max(0.0, self._start_time_sec or 0.0)
            end_time_sec = self._end_time_sec

            start_frame = int(round(start_time_sec * fps))
            end_frame: int | None = None
            if end_time_sec is not None:
                end_frame = int(round(max(0.0, end_time_sec) * fps))

            if total_frames > 0:
                if start_frame >= total_frames:
                    raise RuntimeError("Trim start time is beyond the video duration.")
                if end_frame is None:
                    end_frame = total_frames
                else:
                    end_frame = min(end_frame, total_frames)

            if end_frame is not None and end_frame <= start_frame:
                raise RuntimeError("Trim end time must be greater than trim start time.")

            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

            frames_to_process = 0
            if end_frame is not None:
                frames_to_process = max(0, end_frame - start_frame)
            elif total_frames > 0:
                frames_to_process = max(0, total_frames - start_frame)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self._output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise RuntimeError(f"Could not open output video: {self._output_path}")

            self.status_changed.emit("Processing video...")
            processed = 0
            current_frame = start_frame

            while self._running:
                if end_frame is not None and current_frame >= end_frame:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                boxes = self._detector.detect_boxes(frame, self._settings)
                masked = self._detector.apply_mask(frame, boxes)
                out.write(masked)

                current_frame += 1
                processed += 1
                if frames_to_process > 0:
                    progress = int((processed / frames_to_process) * 100)
                    self.progress_changed.emit(min(100, progress))

            if not self._running:
                self.status_changed.emit("Video processing stopped.")
                return

            self.progress_changed.emit(100)
            self.status_changed.emit("Video processing completed.")
            self.completed.emit(self._output_path)

        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
