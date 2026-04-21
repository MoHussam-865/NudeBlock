MODERN_THEME_QSS = """
/* Modern Dark Theme for NudeBlock */

QWidget {
    background-color: #1e1e24;
    color: #e0e0e0;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #1e1e24;
}

/* Group Boxes */
QGroupBox {
    border: 1px solid #3a3a42;
    border-radius: 6px;
    margin-top: 16px;
    padding-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
    color: #9ba1a6;
    font-weight: bold;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #3a3a42;
    border-radius: 6px;
    background-color: #25252b;
}
QTabBar::tab {
    background-color: #1e1e24;
    color: #9ba1a6;
    padding: 8px 16px;
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
}
QTabBar::tab:selected {
    color: #ffffff;
    background-color: #25252b;
    border-bottom: 2px solid #3a7cff;
}
QTabBar::tab:hover {
    background-color: #2a2a32;
    color: #ffffff;
}

/* Buttons */
QPushButton {
    background-color: #3a7cff;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #558eff;
}
QPushButton:pressed {
    background-color: #2663e0;
}
QPushButton:disabled {
    background-color: #44444c;
    color: #888888;
}

/* Line Edits & SpinBoxes */
QLineEdit, QDoubleSpinBox, QSpinBox, QTimeEdit {
    background-color: #25252b;
    border: 1px solid #3a3a42;
    border-radius: 4px;
    padding: 6px 8px;
    color: #e0e0e0;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QTimeEdit:focus {
    border: 1px solid #3a7cff;
}
QDoubleSpinBox::up-button, QSpinBox::up-button, 
QDoubleSpinBox::down-button, QSpinBox::down-button {
    background-color: #33333b;
    border: none;
    border-radius: 2px;
    margin: 1px;
}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover, 
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {
    background-color: #4a4a55;
}

/* List Widget */
QListWidget {
    background-color: #25252b;
    border: 1px solid #3a3a42;
    border-radius: 4px;
    padding: 4px;
}
QListWidget::item {
    padding: 4px;
    border-radius: 2px;
}
QListWidget::item:hover {
    background-color: #2a2a32;
}
QListWidget::item:selected {
    background-color: #3a7cff;
    color: white;
}

/* Labels */
QLabel {
    color: #e0e0e0;
}
QLabel#footerStatus {
    color: #9ba1a6;
    font-size: 11px;
}

/* Progress Bar */
QProgressBar {
    background-color: #25252b;
    border: 1px solid #3a3a42;
    border-radius: 4px;
    text-align: center;
    color: white;
}
QProgressBar::chunk {
    background-color: #3a7cff;
    border-radius: 3px;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #1e1e24;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #4a4a55;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #5a5a66;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
}
"""