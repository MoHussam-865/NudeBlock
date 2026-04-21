"""Entry point for the NudeBlock desktop application."""

from __future__ import annotations

import ctypes
import os
import sys

# Keep Qt pixel behavior aligned with mss screen capture coordinates.
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

from PyQt6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.ui.theme import MODERN_THEME_QSS


def _enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:  # noqa: BLE001
        pass

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    _enable_windows_dpi_awareness()

    app = QApplication(sys.argv)
    app.setStyleSheet(MODERN_THEME_QSS)
    app.setQuitOnLastWindowClosed(False)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
