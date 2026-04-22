"""QThread worker that processes videos frame by frame."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from app.core.settings import DetectionSettings
from app.services.detection_service import DetectionService


@dataclass
class _TrackState:
    track_id: int
    class_id: int
    last_box: tuple[int, int, int, int]
    last_seen_frame: int
    consecutive_hits: int = 1
    confirmed: bool = False


class VideoWorker(QThread):
    TRUE_POSITIVE_CONSECUTIVE_HITS = 10
    TRACK_LOOKAHEAD_MS = 1000
    BOX_MATCH_MIN_IOU = 0.20
    BOX_MATCH_CENTER_RATIO = 0.65
    BOX_MATCH_MIN_PIXELS = 16.0

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
        self._tracks: list[_TrackState] = []
        self._next_track_id = 1
        self._max_missing_frames = 1

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

    @staticmethod
    def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        return (box[0] + (box[2] / 2.0), box[1] + (box[3] / 2.0))

    @classmethod
    def _is_same_object(
        cls,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> bool:
        if cls._box_iou(box_a, box_b) >= cls.BOX_MATCH_MIN_IOU:
            return True

        acx, acy = cls._box_center(box_a)
        bcx, bcy = cls._box_center(box_b)

        dx = acx - bcx
        dy = acy - bcy

        diag_a = (box_a[2] ** 2 + box_a[3] ** 2) ** 0.5
        diag_b = (box_b[2] ** 2 + box_b[3] ** 2) ** 0.5
        max_center_distance = max(
            cls.BOX_MATCH_MIN_PIXELS,
            min(diag_a, diag_b) * cls.BOX_MATCH_CENTER_RATIO,
        )

        return (dx * dx + dy * dy) <= (max_center_distance * max_center_distance)

    def _emit_progress(self, processed: int, frames_to_process: int) -> None:
        if frames_to_process > 0:
            progress = int((processed / frames_to_process) * 100)
            self.progress_changed.emit(min(100, progress))

    def _find_best_detection_match(
        self,
        track: _TrackState,
        detections: list[tuple[int, int, int, int, int]],
        unmatched_indices: set[int],
    ) -> int | None:
        best_index: int | None = None
        best_score = -1.0

        for idx in unmatched_indices:
            bx, by, bw, bh, class_id = detections[idx]
            if class_id != track.class_id:
                continue

            det_box = (bx, by, bw, bh)
            if not self._is_same_object(track.last_box, det_box):
                continue

            iou = self._box_iou(track.last_box, det_box)
            if iou > best_score:
                best_score = iou
                best_index = idx

        return best_index

    def _mask_boxes_for_frame(
        self,
        frame_index: int,
        detections: list[tuple[int, int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        mask_boxes: list[tuple[int, int, int, int]] = [
            (bx, by, bw, bh) for (bx, by, bw, bh, _class_id) in detections
        ]

        unmatched_indices = set(range(len(detections)))
        stale_track_ids: set[int] = set()

        ordered_tracks = sorted(
            self._tracks,
            key=lambda track: (track.confirmed, track.consecutive_hits),
            reverse=True,
        )

        for track in ordered_tracks:
            matched_index = self._find_best_detection_match(track, detections, unmatched_indices)

            if matched_index is not None:
                unmatched_indices.remove(matched_index)
                bx, by, bw, bh, _class_id = detections[matched_index]
                det_box = (bx, by, bw, bh)

                gap = frame_index - track.last_seen_frame
                if gap == 1:
                    track.consecutive_hits += 1
                else:
                    track.consecutive_hits = 1

                track.last_box = det_box
                track.last_seen_frame = frame_index

                if (
                    not track.confirmed
                    and track.consecutive_hits >= self.TRUE_POSITIVE_CONSECUTIVE_HITS
                ):
                    track.confirmed = True
                continue

            missing_frames = frame_index - track.last_seen_frame

            if track.confirmed and missing_frames <= self._max_missing_frames:
                # Pessimistic mask-hold while waiting for possible reappearance.
                mask_boxes.append(track.last_box)

            if track.confirmed:
                if missing_frames > self._max_missing_frames:
                    stale_track_ids.add(track.track_id)
            elif missing_frames > 0:
                # Before confirmation, require uninterrupted detections.
                # One missed frame resets candidate tracking immediately.
                stale_track_ids.add(track.track_id)

        for idx in sorted(unmatched_indices):
            bx, by, bw, bh, class_id = detections[idx]
            self._tracks.append(
                _TrackState(
                    track_id=self._next_track_id,
                    class_id=class_id,
                    last_box=(bx, by, bw, bh),
                    last_seen_frame=frame_index,
                )
            )
            self._next_track_id += 1

        if stale_track_ids:
            self._tracks = [
                track
                for track in self._tracks
                if track.track_id not in stale_track_ids
            ]

        return mask_boxes

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

            self._tracks = []
            self._next_track_id = 1
            self._max_missing_frames = max(
                1,
                int(round((self.TRACK_LOOKAHEAD_MS / 1000.0) * fps)),
            )

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

                detections = self._detector.detect_boxes_with_classes(frame, self._settings)
                mask_boxes = self._mask_boxes_for_frame(current_frame, detections)
                masked = self._detector.apply_mask(frame, mask_boxes)
                out.write(masked)

                processed += 1
                self._emit_progress(processed, frames_to_process)
                current_frame += 1

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
