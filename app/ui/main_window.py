"""Main desktop window for NudeBlock."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import mss
from PyQt6.QtCore import QTime, Qt
from PyQt6.QtGui import QAction, QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QStyle,
    QSystemTrayIcon,
    QTabWidget,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.constants import (
    APP_NAME,
    DEFAULT_BOX_SCALE,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCREEN_SCANS_PER_SECOND,
    NUDENET_CLASSES,
)
from app.core.settings import DetectionSettings, ScreenSettings
from app.services.detection_service import DetectionService
from app.services.image_service import ImageService
from app.ui.screen_overlay import ScreenOverlay
from app.workers.screen_worker import ScreenWorker
from app.workers.video_worker import VideoWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"{APP_NAME} Desktop")
        self.resize(960, 680)

        self._detector = DetectionService()
        self._image_service = ImageService(self._detector)

        self._video_worker: VideoWorker | None = None
        self._screen_worker: ScreenWorker | None = None
        self._screen_overlay: ScreenOverlay | None = None
        self._allow_close = False
        self._tray_notice_shown = False

        self._build_ui()
        self._setup_tray()
        self._update_tray_tooltip()

    # -----------------------------------------------------------------
    # UI setup
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        page = QVBoxLayout(root)
        page.setContentsMargins(20, 20, 20, 20)
        page.setSpacing(16)

        settings_group = QGroupBox("Detection Settings")
        settings_layout = QHBoxLayout(settings_group)

        form_left = QFormLayout()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(DEFAULT_CONF_THRESHOLD)
        form_left.addRow("Confidence threshold", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(DEFAULT_IOU_THRESHOLD)
        form_left.addRow("IoU threshold", self.iou_spin)

        self.box_scale_spin = QSpinBox()
        self.box_scale_spin.setRange(100, 300)
        self.box_scale_spin.setSingleStep(25)
        self.box_scale_spin.setSuffix("%")
        self.box_scale_spin.setValue(int(DEFAULT_BOX_SCALE * 100))
        form_left.addRow("Box scale", self.box_scale_spin)

        self.screen_rate_spin = QSpinBox()
        self.screen_rate_spin.setRange(1, 30)
        self.screen_rate_spin.setValue(DEFAULT_SCREEN_SCANS_PER_SECOND)
        form_left.addRow("Screen scans per second", self.screen_rate_spin)

        self.show_labels_checkbox = QCheckBox("Show class label and score")
        self.show_labels_checkbox.setChecked(True)
        form_left.addRow("Video/Image labels", self.show_labels_checkbox)

        settings_layout.addLayout(form_left, stretch=1)

        class_panel = QVBoxLayout()
        class_panel.addWidget(QLabel("Classes to block"))

        self.classes_list = QListWidget()
        self.classes_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for class_id, label in enumerate(NUDENET_CLASSES):
            item = QListWidgetItem(f"{class_id:02d}  {label}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.classes_list.addItem(item)
        class_panel.addWidget(self.classes_list)

        class_actions = QHBoxLayout()
        self.select_all_btn = QPushButton("Select all")
        self.select_all_btn.clicked.connect(self._select_all_classes)
        class_actions.addWidget(self.select_all_btn)

        self.clear_all_btn = QPushButton("Clear all")
        self.clear_all_btn.clicked.connect(self._clear_all_classes)
        class_actions.addWidget(self.clear_all_btn)

        class_panel.addLayout(class_actions)
        settings_layout.addLayout(class_panel, stretch=2)

        page.addWidget(settings_group)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_image_tab(), "Secure Image")
        self.tabs.addTab(self._build_video_tab(), "Secure Video")
        self.tabs.addTab(self._build_screen_tab(), "Secure Screen")
        page.addWidget(self.tabs)

        self.footer_status = QLabel(f"Model provider: {self._detector.provider_name}")
        self.footer_status.setObjectName("footerStatus")
        self.footer_status.setAlignment(Qt.AlignmentFlag.AlignRight)
        page.addWidget(self.footer_status)

    def _build_image_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        input_row = QHBoxLayout()
        self.image_input_edit = QLineEdit()
        input_row.addWidget(self.image_input_edit)
        browse_input = QPushButton("Browse image")
        browse_input.clicked.connect(self._browse_image_input)
        input_row.addWidget(browse_input)
        layout.addLayout(input_row)

        output_row = QHBoxLayout()
        self.image_output_edit = QLineEdit()
        self.image_output_edit.setPlaceholderText("Optional output path")
        output_row.addWidget(self.image_output_edit)
        browse_output = QPushButton("Browse output")
        browse_output.clicked.connect(self._browse_image_output)
        output_row.addWidget(browse_output)
        layout.addLayout(output_row)

        self.image_process_btn = QPushButton("Process image")
        self.image_process_btn.clicked.connect(self._process_image)
        layout.addWidget(self.image_process_btn)

        self.image_status = QLabel("Idle")
        layout.addWidget(self.image_status)

        self.image_preview = QLabel("Preview appears here")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumHeight(360)
        self.image_preview.setStyleSheet("border: 1px solid #909090;")
        layout.addWidget(self.image_preview)

        return tab

    def _build_video_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        input_row = QHBoxLayout()
        self.video_input_edit = QLineEdit()
        input_row.addWidget(self.video_input_edit)
        browse_input = QPushButton("Browse video")
        browse_input.clicked.connect(self._browse_video_input)
        input_row.addWidget(browse_input)
        layout.addLayout(input_row)

        output_row = QHBoxLayout()
        self.video_output_edit = QLineEdit()
        self.video_output_edit.setPlaceholderText("Optional output path")
        output_row.addWidget(self.video_output_edit)
        browse_output = QPushButton("Browse output")
        browse_output.clicked.connect(self._browse_video_output)
        output_row.addWidget(browse_output)
        layout.addLayout(output_row)

        trim_group = QGroupBox("Trim window (optional)")
        trim_layout = QHBoxLayout(trim_group)

        self.video_trim_enable = QCheckBox("Process only selected time range")
        self.video_trim_enable.toggled.connect(self._on_video_trim_toggled)
        trim_layout.addWidget(self.video_trim_enable)

        trim_layout.addWidget(QLabel("Start"))
        self.video_trim_start = QTimeEdit()
        self.video_trim_start.setDisplayFormat("HH:mm:ss")
        self.video_trim_start.setTime(QTime(0, 0, 0))
        self.video_trim_start.setEnabled(False)
        trim_layout.addWidget(self.video_trim_start)

        trim_layout.addWidget(QLabel("End"))
        self.video_trim_end = QTimeEdit()
        self.video_trim_end.setDisplayFormat("HH:mm:ss")
        self.video_trim_end.setTime(QTime(0, 0, 0))
        self.video_trim_end.setEnabled(False)
        trim_layout.addWidget(self.video_trim_end)
        trim_layout.addStretch(1)

        layout.addWidget(trim_group)

        actions = QHBoxLayout()
        self.video_start_btn = QPushButton("Start processing")
        self.video_start_btn.clicked.connect(self._start_video_processing)
        actions.addWidget(self.video_start_btn)

        self.video_stop_btn = QPushButton("Stop")
        self.video_stop_btn.clicked.connect(self._stop_video_processing)
        self.video_stop_btn.setEnabled(False)
        actions.addWidget(self.video_stop_btn)

        layout.addLayout(actions)

        self.video_progress = QProgressBar()
        self.video_progress.setValue(0)
        layout.addWidget(self.video_progress)

        self.video_status = QLabel("Idle")
        layout.addWidget(self.video_status)

        return tab

    def _build_screen_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        actions = QHBoxLayout()
        self.screen_start_btn = QPushButton("Start secure screen")
        self.screen_start_btn.clicked.connect(self._start_screen_protection)
        actions.addWidget(self.screen_start_btn)

        self.screen_stop_btn = QPushButton("Stop secure screen")
        self.screen_stop_btn.clicked.connect(self._stop_screen_protection)
        self.screen_stop_btn.setEnabled(False)
        actions.addWidget(self.screen_stop_btn)

        layout.addLayout(actions)

        self.screen_status = QLabel("Idle")
        layout.addWidget(self.screen_status)

        note = QLabel(
            "Close button hides to tray. Secure screen can keep running in the background."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addStretch(1)
        return tab

    # -----------------------------------------------------------------
    # Settings helpers
    # -----------------------------------------------------------------
    def _select_all_classes(self) -> None:
        for i in range(self.classes_list.count()):
            self.classes_list.item(i).setCheckState(Qt.CheckState.Checked)

    def _clear_all_classes(self) -> None:
        for i in range(self.classes_list.count()):
            self.classes_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _selected_class_ids(self) -> frozenset[int]:
        selected: set[int] = set()
        for i in range(self.classes_list.count()):
            if self.classes_list.item(i).checkState() == Qt.CheckState.Checked:
                selected.add(i)
        return frozenset(selected)

    def _build_detection_settings(self) -> DetectionSettings:
        selected = self._selected_class_ids()
        if not selected:
            raise ValueError("Select at least one class to block.")
        return DetectionSettings(
            conf_threshold=float(self.conf_spin.value()),
            iou_threshold=float(self.iou_spin.value()),
            box_scale=float(self.box_scale_spin.value()) / 100.0,
            show_labels_and_scores=self.show_labels_checkbox.isChecked(),
            selected_class_ids=selected,
        )

    def _build_screen_settings(self) -> ScreenSettings:
        return ScreenSettings(scans_per_second=int(self.screen_rate_spin.value()))

    # -----------------------------------------------------------------
    # Image feature
    # -----------------------------------------------------------------
    def _browse_image_input(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if file_path:
            self.image_input_edit.setText(file_path)

    def _browse_image_output(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save processed image",
            self._default_image_output(),
            "PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if file_path:
            self.image_output_edit.setText(file_path)

    def _default_image_output(self) -> str:
        src = self.image_input_edit.text().strip()
        if not src:
            return ""
        p = Path(src)
        return str(p.with_name(f"{p.stem}_secure.png"))

    def _process_image(self) -> None:
        try:
            settings = self._build_detection_settings()
        except ValueError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return

        input_path = self.image_input_edit.text().strip()
        if not input_path:
            QMessageBox.warning(self, APP_NAME, "Choose an input image first.")
            return

        output_path = self.image_output_edit.text().strip() or self._default_image_output()

        try:
            masked, boxes = self._image_service.process_image(input_path, settings, output_path)
            self.image_status.setText(
                f"Masked {len(boxes)} regions. Saved: {output_path}"
            )
            self._show_preview(masked)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, str(exc))

    def _show_preview(self, bgr_image) -> None:
        rgb = bgr_image[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        target_w = max(220, self.image_preview.width() - 20)
        target_h = max(220, self.image_preview.height() - 20)
        pixmap = pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_preview.setPixmap(pixmap)

    # -----------------------------------------------------------------
    # Video feature
    # -----------------------------------------------------------------
    def _browse_video_input(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        if file_path:
            self.video_input_edit.setText(file_path)

    def _browse_video_output(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save processed video",
            self._default_video_output(),
            "MP4 (*.mp4)",
        )
        if file_path:
            self.video_output_edit.setText(file_path)

    def _default_video_output(self) -> str:
        src = self.video_input_edit.text().strip()
        if not src:
            return ""
        p = Path(src)
        return str(p.with_name(f"{p.stem}_secure.mp4"))

    @staticmethod
    def _qtime_to_seconds(value: QTime) -> int:
        return (value.hour() * 3600) + (value.minute() * 60) + value.second()

    def _on_video_trim_toggled(self, enabled: bool) -> None:
        self.video_trim_start.setEnabled(enabled)
        self.video_trim_end.setEnabled(enabled)

    def _start_video_processing(self) -> None:
        if self._video_worker is not None and self._video_worker.isRunning():
            QMessageBox.information(self, APP_NAME, "Video processing is already running.")
            return

        try:
            settings = self._build_detection_settings()
        except ValueError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return

        input_path = self.video_input_edit.text().strip()
        if not input_path:
            QMessageBox.warning(self, APP_NAME, "Choose an input video first.")
            return

        output_path = self.video_output_edit.text().strip() or self._default_video_output()
        self.video_output_edit.setText(output_path)

        trim_start_sec: float | None = None
        trim_end_sec: float | None = None
        if self.video_trim_enable.isChecked():
            trim_start_sec = float(self._qtime_to_seconds(self.video_trim_start.time()))
            trim_end_sec = float(self._qtime_to_seconds(self.video_trim_end.time()))
            if trim_end_sec <= trim_start_sec:
                QMessageBox.warning(
                    self,
                    APP_NAME,
                    "Video trim end time must be greater than start time.",
                )
                return

        self._video_worker = VideoWorker(
            self._detector,
            input_path,
            output_path,
            settings,
            start_time_sec=trim_start_sec,
            end_time_sec=trim_end_sec,
        )
        self._video_worker.progress_changed.connect(self.video_progress.setValue)
        self._video_worker.status_changed.connect(self.video_status.setText)
        self._video_worker.completed.connect(self._on_video_completed)
        self._video_worker.failed.connect(self._on_video_failed)
        self._video_worker.finished.connect(self._on_video_finished)

        self.video_progress.setValue(0)
        self.video_start_btn.setEnabled(False)
        self.video_stop_btn.setEnabled(True)
        self._video_worker.start()
        self._update_tray_tooltip()

    def _stop_video_processing(self) -> None:
        if self._video_worker is not None and self._video_worker.isRunning():
            self._video_worker.stop()
            self.video_status.setText("Stopping video processing...")

    def _on_video_completed(self, output_path: str) -> None:
        self.video_status.setText(f"Video saved: {output_path}")

    def _on_video_failed(self, message: str) -> None:
        self.video_status.setText(f"Video failed: {message}")
        QMessageBox.critical(self, APP_NAME, message)

    def _on_video_finished(self) -> None:
        self.video_start_btn.setEnabled(True)
        self.video_stop_btn.setEnabled(False)
        self._update_tray_tooltip()

    # -----------------------------------------------------------------
    # Screen feature
    # -----------------------------------------------------------------
    def _start_screen_protection(self) -> None:
        if self._screen_worker is not None and self._screen_worker.isRunning():
            QMessageBox.information(self, APP_NAME, "Secure screen is already running.")
            return

        try:
            detection_settings = self._build_detection_settings()
            screen_settings = self._build_screen_settings()
        except ValueError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return

        with mss.mss() as sct:
            if len(sct.monitors) < 2:
                QMessageBox.critical(self, APP_NAME, "No monitor found for screen capture.")
                return
            monitor = sct.monitors[1]

        self._screen_overlay = ScreenOverlay(monitor)
        self._screen_overlay.show()
        self._exclude_overlay_from_capture()

        self._screen_worker = ScreenWorker(
            self._detector,
            monitor,
            detection_settings,
            screen_settings,
        )
        self._screen_worker.boxes_ready.connect(self._screen_overlay.set_boxes)
        self._screen_worker.status_changed.connect(self.screen_status.setText)
        self._screen_worker.failed.connect(self._on_screen_failed)
        self._screen_worker.finished.connect(self._on_screen_finished)

        self._screen_worker.start()
        self.screen_start_btn.setEnabled(False)
        self.screen_stop_btn.setEnabled(True)
        self._update_tray_tooltip()

    def _stop_screen_protection(self) -> None:
        if self._screen_worker is not None and self._screen_worker.isRunning():
            self.screen_status.setText("Stopping secure screen...")
            self._screen_worker.stop()

    def _on_screen_failed(self, message: str) -> None:
        self.screen_status.setText(f"Screen failed: {message}")
        QMessageBox.critical(self, APP_NAME, message)

    def _on_screen_finished(self) -> None:
        if self._screen_overlay is not None:
            self._screen_overlay.clear_boxes()
            self._screen_overlay.hide()
            self._screen_overlay.deleteLater()
            self._screen_overlay = None

        self.screen_start_btn.setEnabled(True)
        self.screen_stop_btn.setEnabled(False)
        self._update_tray_tooltip()

    def _exclude_overlay_from_capture(self) -> None:
        if sys.platform != "win32" or self._screen_overlay is None:
            return

        try:
            hwnd = int(self._screen_overlay.winId())
            # Windows 10 2004+ constant to hide a window from screen capture APIs.
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
        except Exception:  # noqa: BLE001
            pass

    # -----------------------------------------------------------------
    # Tray behavior
    # -----------------------------------------------------------------
    def _setup_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            self.footer_status.setText(
                self.footer_status.text()
                + " | System tray unavailable on this platform"
            )
            return

        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self._tray = QSystemTrayIcon(icon, self)

        menu = QMenu(self)
        show_action = QAction("Show", self)
        show_action.triggered.connect(self._show_window)
        menu.addAction(show_action)

        hide_action = QAction("Hide", self)
        hide_action.triggered.connect(self.hide)
        menu.addAction(hide_action)

        menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit_from_tray)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _show_window(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._show_window()

    def _quit_from_tray(self) -> None:
        self._allow_close = True
        self.close()

    def _update_tray_tooltip(self) -> None:
        if getattr(self, "_tray", None) is None:
            return

        states = []
        if self._screen_worker is not None and self._screen_worker.isRunning():
            states.append("Screen protection active")
        if self._video_worker is not None and self._video_worker.isRunning():
            states.append("Video processing active")
        if not states:
            states.append("Idle")

        self._tray.setToolTip(f"{APP_NAME}: {' | '.join(states)}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._allow_close or self._tray is None:
            self._shutdown_services()
            event.accept()
            return

        event.ignore()
        self.hide()

        if not self._tray_notice_shown:
            self._tray.showMessage(
                APP_NAME,
                "Application is still running in the system tray.",
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            )
            self._tray_notice_shown = True

    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------
    def _shutdown_services(self) -> None:
        if self._video_worker is not None and self._video_worker.isRunning():
            self._video_worker.stop()
            self._video_worker.wait(1500)

        if self._screen_worker is not None and self._screen_worker.isRunning():
            self._screen_worker.stop()
            self._screen_worker.wait(1500)

        if self._screen_overlay is not None:
            self._screen_overlay.hide()
