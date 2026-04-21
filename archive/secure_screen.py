"""
NudeBlock – Real-time screen overlay that detects and masks nudity.

Captures the primary monitor, runs YOLOv8-based NudeNet inference via ONNX,
and draws opaque black rectangles on a transparent, click-through PyQt6 overlay.

Exit: Press  Ctrl+Shift+Q  (global hotkey via the `keyboard` module).
"""

import ctypes
import os
import sys
import threading
import time

# Disable Qt's own high-DPI scaling so it draws in raw physical pixels,
# matching the coordinates from mss.  Must be set before QApplication.
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

import cv2
import keyboard
import mss
import numpy as np
import onnxruntime as ort
from PyQt6.QtCore import QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter
from PyQt6.QtWidgets import QApplication, QWidget

from archive.tiling import detect_tiled

# ──────────────────────────────────────────────────────────────────────
# 1.  Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "nudenet_v8.onnx"
CONF_THRESHOLD = 0.1          # Minimum class-score confidence
IOU_THRESHOLD = 0.45           # NMS overlap threshold
INFER_INPUT_SIZE = 640         # YOLOv8 expected input dimension

# 18 NudeNet classes (indices 0-17)

# The 18 classes supported by the model, in the correct index order
CLASSES = [
    "FEMALE_GENITALIA_COVERED", 
    "FACE_FEMALE", 
    "BUTTOCKS_EXPOSED", 
    "FEMALE_BREAST_EXPOSED", 
    "FEMALE_GENITALIA_EXPOSED", 
    "MALE_BREAST_EXPOSED", 
    "ANUS_EXPOSED", 
    "FEET_EXPOSED", 
    "BELLY_COVERED", 
    "FEET_COVERED", 
    "ARMPITS_COVERED", 
    "ARMPITS_EXPOSED", 
    "FACE_MALE", 
    "BELLY_EXPOSED", 
    "MALE_GENITALIA_EXPOSED", 
    "ANUS_COVERED", 
    "FEMALE_BREAST_COVERED", 
    "BUTTOCKS_COVERED"
]
# Indices of classes to black-out.  Edit this list to taste.
CLASSES_TO_MASK = list(range(18))  # Mask all classes by default (0-17)

# How often (ms) the overlay repaints – 33 ms ≈ 30 FPS
OVERLAY_REFRESH_MS = 33

# Number of consecutive empty-box frames to tolerate before actually clearing
# the overlay.  Prevents single-frame detection drops from causing flashes.
BOX_HOLD_FRAMES = 8

# ──────────────────────────────────────────────────────────────────────
# 2.  Overlay window  (the "Hand")
# ──────────────────────────────────────────────────────────────────────
class Overlay(QWidget):
    """
    Full-screen, frameless, always-on-top, transparent, click-through window
    that simply paints solid-black rectangles wherever the inference thread
    tells it to.

    Window flags explained:
      • FramelessWindowHint ─ no title bar / borders
      • WindowStaysOnTopHint ─ keeps overlay above every other window
      • Tool ─ hides from the taskbar & Alt-Tab list
      • WA_TranslucentBackground ─ un-painted pixels are fully transparent
      • WA_TransparentForMouseEvents ─ all clicks pass through to windows below
    """

    # Qt signal carrying a list of (x, y, w, h) tuples from the worker thread
    boxes_ready = pyqtSignal(list)

    def __init__(self, screen_w: int, screen_h: int):
        super().__init__()

        self._boxes: list[tuple[int, int, int, int]] = []
        self._empty_streak = 0        # consecutive frames with no detections
        self._needs_repaint = False   # dirty flag to avoid redundant repaints

        # --- window flags ---
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Cover the entire primary monitor
        self.setGeometry(0, 0, screen_w, screen_h)

        # Connect the cross-thread signal to a slot that stores boxes & repaints
        self.boxes_ready.connect(self._on_boxes)

        # Safety-net timer: only triggers a repaint if the dirty flag is set
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(OVERLAY_REFRESH_MS)

    # ── slots / painting ──────────────────────────────────────────────

    def _tick(self):
        """Called by QTimer.  Only repaint when something actually changed."""
        if self._needs_repaint:
            self._needs_repaint = False
            self.update()

    def _on_boxes(self, boxes: list):
        if boxes:
            # New detections arrived — use them immediately
            self._empty_streak = 0
            if boxes != self._boxes:
                self._boxes = boxes
                self._needs_repaint = True
        else:
            # No detections this frame.  Keep previous boxes for a few frames
            # to ride out momentary detection drops and avoid flashing.
            self._empty_streak += 1
            if self._empty_streak >= BOX_HOLD_FRAMES and self._boxes:
                self._boxes = []
                self._needs_repaint = True

    def paintEvent(self, _event):
        """Draw opaque black rectangles over every detected region."""
        if not self._boxes:
            return  # nothing to mask → overlay is fully transparent
        painter = QPainter(self)
        painter.setBrush(QBrush(QColor(0, 0, 0)))
        painter.setPen(Qt.PenStyle.NoPen)
        for (x, y, w, h) in self._boxes:
            painter.drawRect(QRect(x, y, w, h))
        painter.end()


# ──────────────────────────────────────────────────────────────────────
# 3.  Inference thread  (the "Eye")
# ──────────────────────────────────────────────────────────────────────
class InferenceWorker(threading.Thread):
    """
    Continuously grabs the primary monitor with *mss*, runs the ONNX model,
    post-processes detections, and emits results to the overlay via a Qt signal.
    """

    def __init__(self, overlay: Overlay, monitor: dict):
        super().__init__(daemon=True)
        self._overlay = overlay
        self._monitor = monitor          # mss monitor dict
        self._running = True

        self._screen_w = monitor["width"]
        self._screen_h = monitor["height"]

        # --- load ONNX model (GPU first, fall back to CPU) ---
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("[NudeBlock] Using CUDA GPU for inference.")
        else:
            chosen = ["CPUExecutionProvider"]
            print("[NudeBlock] CUDA not available – falling back to CPU.")
        self._session = ort.InferenceSession(MODEL_PATH, providers=chosen)
        self._input_name = self._session.get_inputs()[0].name

    # ── public ────────────────────────────────────────────────────────

    def stop(self):
        self._running = False

    # ── main loop ─────────────────────────────────────────────────────

    def run(self):
        classes_set = set(CLASSES_TO_MASK)  # faster membership test

        with mss.mss() as sct:
            while self._running:
                # --- grab screen ---
                img = sct.grab(self._monitor)
                # mss returns BGRA; convert to a contiguous numpy BGR array
                frame = np.array(img, dtype=np.uint8)[:, :, :3].copy()

                # --- tiled inference ---
                # Splits frame into overlapping tiles, runs the model on each,
                # remaps to screen coordinates, and merges with global NMS.
                final_boxes = detect_tiled(
                    frame, self._session, self._input_name,
                    INFER_INPUT_SIZE, classes_set,
                    CONF_THRESHOLD, IOU_THRESHOLD,
                    self._screen_w, self._screen_h,
                )

                # Thread-safe: emit via Qt signal (queued connection by default
                # because emitter thread ≠ receiver thread).
                self._overlay.boxes_ready.emit(final_boxes)


# ──────────────────────────────────────────────────────────────────────
# 4.  Entry point
# ──────────────────────────────────────────────────────────────────────
def main():
    # --- Force Windows Per-Monitor DPI Awareness ---
    # mss captures in physical pixels.  Without this, PyQt uses logical
    # (scaled) pixels, so the overlay is the wrong size and boxes don't
    # line up.  Must be called BEFORE QApplication is created.
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Per-Monitor V2
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()    # fallback
            except Exception:
                pass

    app = QApplication(sys.argv)

    # Identify the primary monitor via mss (index 1 = first real monitor)
    with mss.mss() as sct:
        monitor = sct.monitors[1]
    screen_w = monitor["width"]
    screen_h = monitor["height"]
    screen_left = monitor["left"]
    screen_top = monitor["top"]
    print(f"[NudeBlock] Primary monitor: {screen_w}×{screen_h} at offset ({screen_left}, {screen_top})")

    # -- overlay --
    overlay = Overlay(screen_w, screen_h)
    # Position at the monitor's actual offset (matters for multi-monitor setups)
    overlay.setGeometry(screen_left, screen_top, screen_w, screen_h)
    overlay.show()

    # --- Exclude overlay from screen capture (Windows 10 2004+) ---
    # Without this the overlay's own black boxes get captured by mss,
    # the model no longer sees nudity (it's covered), removes the boxes,
    # nudity becomes visible again, gets detected, boxes drawn … = flash.
    if sys.platform == "win32":
        try:
            hwnd = int(overlay.winId())
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            print("[NudeBlock] Overlay excluded from screen capture.")
        except Exception as e:
            print(f"[NudeBlock] Could not exclude overlay from capture: {e}")

    # -- inference thread --
    worker = InferenceWorker(overlay, monitor)
    worker.start()

    # -- global hotkey: Ctrl+Shift+Q to quit --
    def _quit():
        print("\n[NudeBlock] Ctrl+Shift+Q pressed – shutting down.")
        worker.stop()
        app.quit()

    keyboard.add_hotkey("ctrl+shift+q", _quit)
    print("[NudeBlock] Running.  Press Ctrl+Shift+Q to exit.")

    exit_code = app.exec()
    worker.stop()
    worker.join(timeout=2)
    keyboard.unhook_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
