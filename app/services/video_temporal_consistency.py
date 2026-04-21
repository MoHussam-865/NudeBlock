"""Video-only temporal consistency post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from app.core.settings import VideoConsistencySettings

Box = tuple[int, int, int, int]


@dataclass
class _Track:
    track_id: int
    points: list[tuple[int, Box]] = field(default_factory=list)
    missed: int = 0


class VideoTemporalConsistency:
    """Stabilizes per-frame detections using future-aware interpolation."""

    def __init__(self, settings: VideoConsistencySettings):
        settings.validate()
        self._settings = settings

    def stabilize(self, frame_boxes: list[list[Box]]) -> list[list[Box]]:
        if not frame_boxes:
            return []

        if not self._settings.enabled:
            return [list(boxes) for boxes in frame_boxes]

        tracks = self._build_tracks(frame_boxes)
        return self._render_tracks(tracks, len(frame_boxes))

    def _build_tracks(self, frame_boxes: list[list[Box]]) -> list[_Track]:
        max_gap = self._settings.object_consistency_frames
        active: list[_Track] = []
        finished: list[_Track] = []
        next_track_id = 1

        for frame_idx, detections in enumerate(frame_boxes):
            detections = [self._sanitize_box(box) for box in detections if self._is_valid_box(box)]

            candidates: list[tuple[float, int, int]] = []
            for active_idx, track in enumerate(active):
                predicted_box = self._predict_box(track, frame_idx)
                for det_idx, det_box in enumerate(detections):
                    score = self._match_score(predicted_box, det_box)
                    if score is None:
                        continue
                    candidates.append((score, active_idx, det_idx))

            candidates.sort(reverse=True, key=lambda item: item[0])

            matched_tracks: set[int] = set()
            matched_dets: set[int] = set()

            for _score, active_idx, det_idx in candidates:
                if active_idx in matched_tracks or det_idx in matched_dets:
                    continue
                active[active_idx].points.append((frame_idx, detections[det_idx]))
                active[active_idx].missed = 0
                matched_tracks.add(active_idx)
                matched_dets.add(det_idx)

            retained: list[_Track] = []
            for active_idx, track in enumerate(active):
                if active_idx in matched_tracks:
                    retained.append(track)
                    continue

                track.missed += 1
                if track.missed > max_gap:
                    finished.append(track)
                else:
                    retained.append(track)

            for det_idx, det_box in enumerate(detections):
                if det_idx in matched_dets:
                    continue
                retained.append(_Track(track_id=next_track_id, points=[(frame_idx, det_box)]))
                next_track_id += 1

            active = retained

        finished.extend(active)
        return finished

    def _render_tracks(self, tracks: list[_Track], frame_count: int) -> list[list[Box]]:
        max_gap = self._settings.object_consistency_frames
        min_track_frames = self._settings.min_track_frames

        stabilized: list[list[Box]] = [[] for _ in range(frame_count)]

        for track in tracks:
            ordered_points = sorted(track.points, key=lambda item: item[0])
            if len(ordered_points) < min_track_frames:
                continue

            for frame_idx, box in ordered_points:
                if 0 <= frame_idx < frame_count:
                    stabilized[frame_idx].append(box)

            for point_idx in range(len(ordered_points) - 1):
                frame_a, box_a = ordered_points[point_idx]
                frame_b, box_b = ordered_points[point_idx + 1]

                gap = frame_b - frame_a - 1
                if gap <= 0 or gap > max_gap:
                    continue

                for step in range(1, gap + 1):
                    t = step / float(gap + 1)
                    interp = self._interpolate_box(box_a, box_b, t)
                    target_frame = frame_a + step
                    if 0 <= target_frame < frame_count and self._is_valid_box(interp):
                        stabilized[target_frame].append(interp)

        return [self._deduplicate_frame_boxes(frame) for frame in stabilized]

    def _match_score(self, predicted_box: Box, det_box: Box) -> float | None:
        iou = self._iou(predicted_box, det_box)
        dist_ratio = self._center_distance_ratio(predicted_box, det_box)

        if (
            iou < self._settings.match_iou_threshold
            and dist_ratio > self._settings.match_distance_ratio
        ):
            return None

        # Prefer overlap first, then center proximity.
        return iou + (max(0.0, 1.0 - dist_ratio) * 0.25)

    @staticmethod
    def _predict_box(track: _Track, frame_idx: int) -> Box:
        if not track.points:
            return (0, 0, 1, 1)
        if len(track.points) == 1:
            return track.points[-1][1]

        (prev_frame, prev_box), (last_frame, last_box) = track.points[-2], track.points[-1]
        dt = max(1, last_frame - prev_frame)
        delta = max(0, frame_idx - last_frame)

        prev_cx, prev_cy = VideoTemporalConsistency._box_center(prev_box)
        last_cx, last_cy = VideoTemporalConsistency._box_center(last_box)

        vx = (last_cx - prev_cx) / dt
        vy = (last_cy - prev_cy) / dt
        vw = (last_box[2] - prev_box[2]) / dt
        vh = (last_box[3] - prev_box[3]) / dt

        pred_cx = last_cx + (vx * delta)
        pred_cy = last_cy + (vy * delta)
        pred_w = max(1.0, last_box[2] + (vw * delta))
        pred_h = max(1.0, last_box[3] + (vh * delta))

        x = int(round(pred_cx - (pred_w / 2.0)))
        y = int(round(pred_cy - (pred_h / 2.0)))
        w = int(round(pred_w))
        h = int(round(pred_h))
        return (x, y, max(1, w), max(1, h))

    @staticmethod
    def _interpolate_box(box_a: Box, box_b: Box, t: float) -> Box:
        x = int(round((1.0 - t) * box_a[0] + (t * box_b[0])))
        y = int(round((1.0 - t) * box_a[1] + (t * box_b[1])))
        w = int(round((1.0 - t) * box_a[2] + (t * box_b[2])))
        h = int(round((1.0 - t) * box_a[3] + (t * box_b[3])))
        return (x, y, max(1, w), max(1, h))

    @staticmethod
    def _deduplicate_frame_boxes(boxes: list[Box]) -> list[Box]:
        unique: list[Box] = []
        seen: set[Box] = set()
        for box in boxes:
            if box in seen:
                continue
            seen.add(box)
            unique.append(box)
        return unique

    @staticmethod
    def _is_valid_box(box: Box) -> bool:
        return box[2] > 0 and box[3] > 0

    @staticmethod
    def _sanitize_box(box: Box) -> Box:
        x, y, w, h = box
        return (int(x), int(y), max(1, int(w)), max(1, int(h)))

    @staticmethod
    def _box_center(box: Box) -> tuple[float, float]:
        x, y, w, h = box
        return (x + (w / 2.0), y + (h / 2.0))

    @staticmethod
    def _center_distance_ratio(box_a: Box, box_b: Box) -> float:
        ax, ay = VideoTemporalConsistency._box_center(box_a)
        bx, by = VideoTemporalConsistency._box_center(box_b)
        dist = sqrt(((ax - bx) ** 2) + ((ay - by) ** 2))

        diag_a = sqrt((box_a[2] ** 2) + (box_a[3] ** 2))
        diag_b = sqrt((box_b[2] ** 2) + (box_b[3] ** 2))
        norm = max(1.0, (diag_a + diag_b) / 2.0)
        return dist / norm

    @staticmethod
    def _iou(box_a: Box, box_b: Box) -> float:
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

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union
