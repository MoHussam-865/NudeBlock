"""QThread worker that processes videos frame by frame."""

from __future__ import annotations

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from app.core.settings import DetectionSettings
from app.services.detection_service import DetectionService


class VideoWorker(QThread):
    TEMPORAL_WINDOW_SIZE = 5
    MIN_STABLE_HITS = 2
    BOX_MATCH_MIN_IOU = 0.35
    BOX_MATCH_CENTER_RATIO = 0.35

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

    @staticmethod
    def _box_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, aw, ah = box_a
        bx1, by1, bw, bh = box_b

        ax2 = ax1 + aw
        ay2 = ay1 + ah
        bx2 = bx1 + bw
        by2 = by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, aw) * max(0, ah)
        area_b = max(0, bw) * max(0, bh)
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    @classmethod
    def _boxes_close(
        cls,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> bool:
        if cls._box_iou(box_a, box_b) >= cls.BOX_MATCH_MIN_IOU:
            return True

        acx = box_a[0] + (box_a[2] / 2.0)
        acy = box_a[1] + (box_a[3] / 2.0)
        bcx = box_b[0] + (box_b[2] / 2.0)
        bcy = box_b[1] + (box_b[3] / 2.0)

        dx = acx - bcx
        dy = acy - bcy

        diag_a = (box_a[2] ** 2 + box_a[3] ** 2) ** 0.5
        diag_b = (box_b[2] ** 2 + box_b[3] ** 2) ** 0.5
        max_center_distance = max(12.0, min(diag_a, diag_b) * cls.BOX_MATCH_CENTER_RATIO)

        return (dx * dx + dy * dy) <= (max_center_distance * max_center_distance)

    @staticmethod
    def _average_box(
        boxes: list[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, int]:
        count = max(1, len(boxes))
        x = int(round(sum(box[0] for box in boxes) / count))
        y = int(round(sum(box[1] for box in boxes) / count))
        w = max(1, int(round(sum(box[2] for box in boxes) / count)))
        h = max(1, int(round(sum(box[3] for box in boxes) / count)))
        return (x, y, w, h)

    @classmethod
    def _stabilize_window(
        cls,
        window_boxes: list[list[tuple[int, int, int, int]]],
    ) -> list[list[tuple[int, int, int, int]]]:
        if not window_boxes:
            return []

        clusters: list[tuple[list[tuple[int, int, int, int]], set[int]]] = []

        for frame_idx, frame_boxes in enumerate(window_boxes):
            for box in frame_boxes:
                best_idx = -1
                best_iou = -1.0

                for idx, (cluster_boxes, _cluster_frames) in enumerate(clusters):
                    ref_box = cls._average_box(cluster_boxes)
                    if not cls._boxes_close(box, ref_box):
                        continue

                    iou = cls._box_iou(box, ref_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_idx < 0:
                    clusters.append(([box], {frame_idx}))
                else:
                    clusters[best_idx][0].append(box)
                    clusters[best_idx][1].add(frame_idx)

        stable_boxes: list[tuple[int, int, int, int]] = []
        for cluster_boxes, cluster_frames in clusters:
            if len(cluster_frames) >= cls.MIN_STABLE_HITS:
                stable_boxes.append(cls._average_box(cluster_boxes))

        stabilized = [list(frame_boxes) for frame_boxes in window_boxes]
        for frame_idx in range(len(stabilized)):
            for stable_box in stable_boxes:
                already_present = any(
                    cls._boxes_close(existing_box, stable_box)
                    for existing_box in stabilized[frame_idx]
                )
                if not already_present:
                    stabilized[frame_idx].append(stable_box)

        return stabilized

    def _emit_progress(self, processed: int, frames_to_process: int) -> None:
        if frames_to_process > 0:
            progress = int((processed / frames_to_process) * 100)
            self.progress_changed.emit(min(100, progress))

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
            frame_buffer: list = []
            boxes_buffer: list[list[tuple[int, int, int, int]]] = []

            while self._running:
                if end_frame is not None and current_frame >= end_frame:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                boxes = self._detector.detect_boxes(frame, self._settings)
                frame_buffer.append(frame)
                boxes_buffer.append(boxes)
                current_frame += 1

                if len(frame_buffer) < self.TEMPORAL_WINDOW_SIZE:
                    continue

                stabilized_window = self._stabilize_window(boxes_buffer)
                masked = self._detector.apply_mask(frame_buffer[0], stabilized_window[0])
                out.write(masked)

                processed += 1
                self._emit_progress(processed, frames_to_process)
                frame_buffer.pop(0)
                boxes_buffer.pop(0)

            if not self._running:
                self.status_changed.emit("Video processing stopped.")
                return

            while frame_buffer:
                stabilized_window = self._stabilize_window(boxes_buffer)
                masked = self._detector.apply_mask(frame_buffer[0], stabilized_window[0])
                out.write(masked)

                processed += 1
                self._emit_progress(processed, frames_to_process)
                frame_buffer.pop(0)
                boxes_buffer.pop(0)

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
