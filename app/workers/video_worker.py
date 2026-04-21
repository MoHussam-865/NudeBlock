"""QThread worker that processes videos frame by frame."""

from __future__ import annotations

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from app.core.settings import DetectionSettings, VideoConsistencySettings
from app.services.detection_service import DetectionService
from app.services.video_temporal_consistency import VideoTemporalConsistency


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
        video_consistency: VideoConsistencySettings,
        start_time_sec: float | None = None,
        end_time_sec: float | None = None,
    ):
        super().__init__()
        self._detector = detector
        self._input_path = input_path
        self._output_path = output_path
        self._settings = settings
        self._video_consistency = video_consistency
        self._start_time_sec = start_time_sec
        self._end_time_sec = end_time_sec
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        cap = None

        try:
            self._settings.validate()
            self._video_consistency.validate()

            cap = cv2.VideoCapture(self._input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open input video: {self._input_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            start_frame, end_frame, frames_to_process = self._resolve_trim_bounds(
                fps,
                total_frames,
            )

            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

            if self._video_consistency.enabled:
                completed = self._run_with_temporal_consistency(
                    cap,
                    width,
                    height,
                    fps,
                    start_frame,
                    end_frame,
                    frames_to_process,
                )
            else:
                completed = self._run_single_pass(
                    cap,
                    width,
                    height,
                    fps,
                    start_frame,
                    end_frame,
                    frames_to_process,
                )

            if not completed:
                return

        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()

    def _resolve_trim_bounds(
        self,
        fps: float,
        total_frames: int,
    ) -> tuple[int, int | None, int]:
        start_time_sec = max(0.0, self._start_time_sec or 0.0)
        end_time_sec = self._end_time_sec

        start_frame = int(round(start_time_sec * fps))
        end_frame: int | None = None
        if end_time_sec is not None:
            end_frame = int(round(max(0.0, end_time_sec) * fps))

        if total_frames > 0:
            if end_frame is not None and start_frame >= total_frames:
                raise RuntimeError("Trim start time is beyond the video duration.")
            if end_frame is not None:
                end_frame = min(end_frame, total_frames)

        if end_frame is not None and end_frame <= start_frame:
            raise RuntimeError("Trim end time must be greater than trim start time.")

        frames_to_process = 0
        if end_frame is not None:
            frames_to_process = max(0, end_frame - start_frame)
        elif total_frames > 0:
            frames_to_process = max(0, total_frames - start_frame)

        return start_frame, end_frame, frames_to_process

    def _run_single_pass(
        self,
        cap,
        width: int,
        height: int,
        fps: float,
        start_frame: int,
        end_frame: int | None,
        frames_to_process: int,
    ) -> bool:
        out = self._create_writer(width, height, fps)

        try:
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

            if processed == 0:
                raise RuntimeError("No frames were read from input video in the selected range.")

            if not self._running:
                self.status_changed.emit("Video processing stopped.")
                return False

            self.progress_changed.emit(100)
            self.status_changed.emit("Video processing completed.")
            self.completed.emit(self._output_path)
            return True
        finally:
            out.release()

    def _run_with_temporal_consistency(
        self,
        cap,
        width: int,
        height: int,
        fps: float,
        start_frame: int,
        end_frame: int | None,
        frames_to_process: int,
    ) -> bool:
        self.status_changed.emit("Stage 1/3: Detecting frames...")

        raw_boxes_per_frame: list[list[tuple[int, int, int, int]]] = []
        current_frame = start_frame
        work_done = 0
        total_work = (frames_to_process * 2) + 1 if frames_to_process > 0 else 0

        while self._running:
            if end_frame is not None and current_frame >= end_frame:
                break

            ok, frame = cap.read()
            if not ok:
                break

            boxes = self._detector.detect_boxes(frame, self._settings)
            raw_boxes_per_frame.append(boxes)
            current_frame += 1
            work_done += 1

            self._emit_temporal_progress(work_done, total_work)

        if not self._running:
            self.status_changed.emit("Video processing stopped.")
            return False

        if not raw_boxes_per_frame:
            raise RuntimeError("No frames were read from input video in the selected range.")

        if total_work <= 0:
            total_work = (len(raw_boxes_per_frame) * 2) + 1
            work_done = len(raw_boxes_per_frame)
            self._emit_temporal_progress(work_done, total_work)

        self.status_changed.emit("Stage 2/3: Stabilizing detections...")
        temporal = VideoTemporalConsistency(self._video_consistency)
        stabilized_boxes = temporal.stabilize(raw_boxes_per_frame)
        work_done += 1
        self._emit_temporal_progress(work_done, total_work)

        if not self._running:
            self.status_changed.emit("Video processing stopped.")
            return False

        self.status_changed.emit("Stage 3/3: Rendering output...")
        render_cap = cv2.VideoCapture(self._input_path)
        if not render_cap.isOpened():
            raise RuntimeError(f"Could not reopen input video: {self._input_path}")

        out = self._create_writer(width, height, fps)

        try:
            if start_frame > 0:
                render_cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

            for frame_idx, boxes in enumerate(stabilized_boxes):
                if not self._running:
                    self.status_changed.emit("Video processing stopped.")
                    return False

                if end_frame is not None and (start_frame + frame_idx) >= end_frame:
                    break

                ok, frame = render_cap.read()
                if not ok:
                    break

                masked = self._detector.apply_mask(frame, boxes)
                out.write(masked)

                work_done += 1
                self._emit_temporal_progress(work_done, total_work)

            if not self._running:
                self.status_changed.emit("Video processing stopped.")
                return False

            self.progress_changed.emit(100)
            self.status_changed.emit("Video processing completed.")
            self.completed.emit(self._output_path)
            return True
        finally:
            render_cap.release()
            out.release()

    def _create_writer(self, width: int, height: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self._output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Could not open output video: {self._output_path}")
        return out

    def _emit_temporal_progress(self, work_done: int, total_work: int) -> None:
        if total_work <= 0:
            return
        progress = int((work_done / total_work) * 100)
        self.progress_changed.emit(min(99, max(0, progress)))
