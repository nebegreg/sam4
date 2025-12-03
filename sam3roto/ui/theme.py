"""
Professional theme and styling for SAM3 Roto Ultimate
Modern dark theme with professional color palette
"""

# Professional Dark Theme
PROFESSIONAL_THEME = """
/* Main Application Style */
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Segoe UI", "San Francisco", "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
}

/* Menu Bar */
QMenuBar {
    background-color: #252525;
    color: #e0e0e0;
    border-bottom: 1px solid #3d3d3d;
    padding: 4px;
}

QMenuBar::item {
    background-color: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #0d7377;
}

QMenu {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #0d7377;
}

/* Buttons */
QPushButton {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    min-height: 32px;
}

QPushButton:hover {
    background-color: #3d3d3d;
    border: 1px solid: #4d4d4d;
}

QPushButton:pressed {
    background-color: #0d7377;
    border: 1px solid #14919b;
}

QPushButton:disabled {
    background-color: #252525;
    color: #666666;
    border: 1px solid #2d2d2d;
}

/* Primary Action Buttons */
QPushButton[primary="true"] {
    background-color: #0d7377;
    border: 1px solid #14919b;
    font-weight: 600;
}

QPushButton[primary="true"]:hover {
    background-color: #14919b;
    border: 1px solid #1aa6b2;
}

QPushButton[primary="true"]:pressed {
    background-color: #0a5a5d;
}

/* Danger Buttons */
QPushButton[danger="true"] {
    background-color: #8b1e1e;
    border: 1px solid #a32929;
}

QPushButton[danger="true"]:hover {
    background-color: #a32929;
}

/* Line Edits */
QLineEdit {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 8px 12px;
    min-height: 32px;
}

QLineEdit:focus {
    border: 1px solid #0d7377;
    background-color: #323232;
}

QLineEdit::placeholder {
    color: #666666;
}

/* Combo Boxes */
QComboBox {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 8px 12px;
    min-height: 32px;
}

QComboBox:hover {
    border: 1px solid #4d4d4d;
}

QComboBox:focus {
    border: 1px solid #0d7377;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #e0e0e0;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    selection-background-color: #0d7377;
    padding: 4px;
}

/* List Widget */
QListWidget {
    background-color: #252525;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 4px;
}

QListWidget::item {
    padding: 8px;
    border-radius: 4px;
    margin: 2px;
}

QListWidget::item:hover {
    background-color: #2d2d2d;
}

QListWidget::item:selected {
    background-color: #0d7377;
    color: #ffffff;
}

/* Checkboxes */
QCheckBox {
    spacing: 8px;
    padding: 4px;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid #3d3d3d;
    background-color: #2d2d2d;
}

QCheckBox::indicator:hover {
    border: 2px solid #4d4d4d;
}

QCheckBox::indicator:checked {
    background-color: #0d7377;
    border: 2px solid #14919b;
    image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzLjMzMzMgNEw2IDExLjMzMzNMMi42NjY2NyA4IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K);
}

/* Sliders */
QSlider::groove:horizontal {
    background-color: #2d2d2d;
    height: 6px;
    border-radius: 3px;
    border: 1px solid #3d3d3d;
}

QSlider::handle:horizontal {
    background-color: #0d7377;
    border: 2px solid #14919b;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background-color: #14919b;
    border: 2px solid #1aa6b2;
}

QSlider::sub-page:horizontal {
    background-color: #0d7377;
    border-radius: 3px;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    background-color: #252525;
    top: -1px;
}

QTabBar::tab {
    background-color: #2d2d2d;
    color: #9d9d9d;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    min-width: 100px;
    font-weight: 500;
}

QTabBar::tab:hover {
    background-color: #3d3d3d;
    color: #c0c0c0;
}

QTabBar::tab:selected {
    background-color: #0d7377;
    color: #ffffff;
    font-weight: 600;
}

/* Group Boxes */
QGroupBox {
    border: 1px solid #3d3d3d;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    background-color: #252525;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 0px;
    padding: 2px 8px;
    background-color: #252525;
    color: #0d7377;
}

/* Labels */
QLabel {
    color: #e0e0e0;
    background-color: transparent;
}

QLabel[heading="true"] {
    font-size: 14pt;
    font-weight: 600;
    color: #0d7377;
}

QLabel[subheading="true"] {
    font-size: 12pt;
    font-weight: 500;
    color: #b0b0b0;
}

/* Scrollbars */
QScrollBar:vertical {
    background-color: #252525;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #3d3d3d;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4d4d4d;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #252525;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #3d3d3d;
    border-radius: 6px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #4d4d4d;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Status Bar */
QStatusBar {
    background-color: #252525;
    color: #e0e0e0;
    border-top: 1px solid #3d3d3d;
}

QStatusBar::item {
    border: none;
}

/* Tool Bar */
QToolBar {
    background-color: #252525;
    border-bottom: 1px solid #3d3d3d;
    spacing: 4px;
    padding: 4px;
}

QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 6px;
}

QToolButton:hover {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
}

QToolButton:pressed {
    background-color: #0d7377;
}

/* Progress Bar */
QProgressBar {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    text-align: center;
    height: 24px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #0d7377, stop:1 #14919b);
    border-radius: 5px;
}

/* Tooltips */
QToolTip {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10pt;
}

/* Splitter */
QSplitter::handle {
    background-color: #3d3d3d;
}

QSplitter::handle:hover {
    background-color: #0d7377;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}
"""

# Color palette
COLORS = {
    "primary": "#0d7377",
    "primary_hover": "#14919b",
    "primary_pressed": "#0a5a5d",
    "danger": "#8b1e1e",
    "danger_hover": "#a32929",
    "success": "#2d6a3d",
    "warning": "#b87503",
    "background": "#1e1e1e",
    "surface": "#252525",
    "surface_elevated": "#2d2d2d",
    "border": "#3d3d3d",
    "text": "#e0e0e0",
    "text_secondary": "#9d9d9d",
    "text_disabled": "#666666",
}

# Icons (using Unicode symbols and emojis)
ICONS = {
    "video": "🎬",
    "images": "🖼️",
    "load": "📂",
    "save": "💾",
    "settings": "⚙️",
    "play": "▶️",
    "pause": "⏸️",
    "stop": "⏹️",
    "segment": "✂️",
    "track": "🎯",
    "preview": "👁️",
    "export": "📤",
    "depth": "🌊",
    "camera": "📷",
    "add": "➕",
    "remove": "➖",
    "clear": "🗑️",
    "check": "✓",
    "close": "✕",
    "info": "ℹ️",
    "warning": "⚠️",
    "error": "❌",
    "success": "✅",
}
