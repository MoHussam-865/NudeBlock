"""Transparent full-screen overlay used by secure screen mode."""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter
from PyQt6.QtWidgets import QWidget


class ScreenOverlay(QWidget):
    def __init__(self, monitor: dict):
        super().__init__()
        self._boxes: list[tuple[int, int, int, int]] = []

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
        self._boxes = boxes
        self.update()

    def clear_boxes(self) -> None:
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
