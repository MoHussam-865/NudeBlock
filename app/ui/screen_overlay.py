"""Transparent full-screen overlay used by secure screen mode."""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter
from PyQt6.QtWidgets import QWidget

from app.core.constants import DEFAULT_SCREEN_BOX_HOLD_FRAMES


class ScreenOverlay(QWidget):
    def __init__(self, monitor: dict):
        super().__init__()
        self._boxes: list[tuple[int, int, int, int]] = []
        self._empty_streak = 0
        self._hold_frames = DEFAULT_SCREEN_BOX_HOLD_FRAMES

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self.setGeometry(
            monitor["left"],
            monitor["top"],
            monitor["width"],
            monitor["height"],
        )

    def set_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        if boxes:
            self._empty_streak = 0
            if boxes != self._boxes:
                self._boxes = boxes
                self.update()
            return

        self._empty_streak += 1
        if self._empty_streak >= self._hold_frames and self._boxes:
            self._boxes = []
            self.update()

    def clear_boxes(self) -> None:
        self._empty_streak = 0
        self._boxes = []
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self._boxes:
            return

        painter = QPainter(self)
        painter.setBrush(QBrush(QColor(0, 0, 0)))
        painter.setPen(Qt.PenStyle.NoPen)

        for (x, y, w, h) in self._boxes:
            painter.drawRect(QRect(x, y, w, h))

        painter.end()
