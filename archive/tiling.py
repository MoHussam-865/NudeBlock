"""
tiling.py – Tile-based inference for high-resolution frames.

Splits a large image into overlapping tiles, runs YOLO inference on each tile
independently, maps detections back to full-image coordinates, and merges
duplicates with a global NMS pass.
"""

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Tile generation
# ──────────────────────────────────────────────────────────────────────

def compute_tile_grid(img_w: int, img_h: int, model_size: int = 640,
                      overlap_ratio: float = 0.25):
    """
    Return a list of (x, y, w, h) tile regions that together cover the
    entire image.  Tiles overlap by *overlap_ratio* of the tile size so
    that objects sitting on a boundary appear fully inside at least one tile.

    The number of tiles along each axis is chosen so that each tile, when
    resized to *model_size*, does not down-scale by more than ~2×.  This
    keeps small objects large enough for the detector to see.

    Returns
    -------
    list[tuple[int, int, int, int]]
        Each element is (x_start, y_start, tile_w, tile_h) in pixel coords.
    """
    # Target: each tile should represent at most ~model_size * 2 real pixels
    # so the 640→real scale factor stays ≤ 2×.
    max_tile_px = model_size * 2  # 1280 px

    cols = max(1, int(np.ceil(img_w / max_tile_px)))
    rows = max(1, int(np.ceil(img_h / max_tile_px)))

    # Base tile size
    tile_w = int(np.ceil(img_w / cols))
    tile_h = int(np.ceil(img_h / rows))

    overlap_x = int(tile_w * overlap_ratio)
    overlap_y = int(tile_h * overlap_ratio)

    tiles = []
    for r in range(rows):
        for c in range(cols):
            # Compute the stride (tile size minus overlap) for interior tiles.
            # The last tile in each axis is always anchored to the image edge
            # so that every pixel is covered — this prevents the right/bottom
            # strip from being missed.
            if cols == 1:
                x = 0
            elif c == cols - 1:
                x = img_w - tile_w          # anchor last column to right edge
            else:
                x = c * (tile_w - overlap_x)

            if rows == 1:
                y = 0
            elif r == rows - 1:
                y = img_h - tile_h          # anchor last row to bottom edge
            else:
                y = r * (tile_h - overlap_y)

            x = max(0, x)
            y = max(0, y)
            w = min(tile_w, img_w - x)
            h = min(tile_h, img_h - y)
            tiles.append((x, y, w, h))

    return tiles


# ──────────────────────────────────────────────────────────────────────
# Per-tile inference
# ──────────────────────────────────────────────────────────────────────

def detect_on_tile(tile_img: np.ndarray, session, input_name: str,
                   model_size: int, classes_set: set,
                   conf_thresh: float):
    """
    Run YOLO inference on a single tile and return boxes in **tile-local**
    coordinates.

    Returns
    -------
    boxes : list[[left, top, width, height]]
    scores : list[float]
    class_ids : list[int]
    """
    tile_h, tile_w = tile_img.shape[:2]
    x_factor = tile_w / model_size
    y_factor = tile_h / model_size

    blob = cv2.dnn.blobFromImage(
        tile_img, 1 / 255.0, (model_size, model_size),
        swapRB=True, crop=False,
    )

    outputs = session.run(None, {input_name: blob})
    predictions = np.squeeze(outputs[0]).T  # [8400, 22]

    boxes = []
    scores = []
    class_ids = []

    for row in predictions:
        class_scores = row[4:]
        max_score = float(np.max(class_scores))
        if max_score < conf_thresh:
            continue
        class_id = int(np.argmax(class_scores))
        if class_id not in classes_set:
            continue

        cx, cy, w, h = row[0], row[1], row[2], row[3]
        left = int((cx - w / 2) * x_factor)
        top = int((cy - h / 2) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)

        boxes.append([left, top, width, height])
        scores.append(max_score)
        class_ids.append(class_id)

    return boxes, scores, class_ids


# ──────────────────────────────────────────────────────────────────────
# Full-frame tiled detection
# ──────────────────────────────────────────────────────────────────────

def detect_tiled(frame: np.ndarray, session, input_name: str,
                 model_size: int, classes_set: set,
                 conf_thresh: float, iou_thresh: float,
                 screen_w: int, screen_h: int):
    """
    Backward-compatible helper that returns boxes only.

    Returns
    -------
    list[tuple[int, int, int, int]]
        Final (x, y, w, h) bounding boxes clamped to screen bounds.
    """
    detections = detect_tiled_with_classes(
        frame,
        session,
        input_name,
        model_size,
        classes_set,
        conf_thresh,
        iou_thresh,
        screen_w,
        screen_h,
    )
    return [(bx, by, bw, bh) for (bx, by, bw, bh, _class_id) in detections]


def detect_tiled_with_classes(frame: np.ndarray, session, input_name: str,
                              model_size: int, classes_set: set,
                              conf_thresh: float, iou_thresh: float,
                              screen_w: int, screen_h: int):
    """
    Split *frame* into overlapping tiles, run inference on each, remap
    detections to full-frame coordinates, and apply a global NMS pass.

    Returns
    -------
    list[tuple[int, int, int, int, int]]
        Final (x, y, w, h, class_id) detections clamped to screen bounds.
    """
    img_h, img_w = frame.shape[:2]
    tiles = compute_tile_grid(img_w, img_h, model_size)

    all_boxes = []
    all_scores = []
    all_class_ids = []

    for (tx, ty, tw, th) in tiles:
        tile_img = frame[ty:ty + th, tx:tx + tw]
        boxes, scores, class_ids = detect_on_tile(
            tile_img, session, input_name,
            model_size, classes_set, conf_thresh,
        )
        # Remap tile-local coords → full-frame coords
        for box in boxes:
            box[0] += tx
            box[1] += ty
        all_boxes.extend(boxes)
        all_scores.extend(scores)
        all_class_ids.extend(class_ids)

    # Global NMS to merge duplicates from overlapping tiles
    final_boxes: list[tuple[int, int, int, int, int]] = []
    if all_boxes:
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, conf_thresh, iou_thresh)
        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = all_boxes[i]
                bx = max(0, bx)
                by = max(0, by)
                bw = min(bw, screen_w - bx)
                bh = min(bh, screen_h - by)
                if bw > 0 and bh > 0:
                    final_boxes.append((bx, by, bw, bh, all_class_ids[i]))

    return final_boxes
