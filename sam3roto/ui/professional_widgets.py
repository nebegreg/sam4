"""
Professional custom widgets for SAM3 Roto Ultimate
Enhanced UI components with modern styling
"""

from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
from typing import Optional


class ModernSlider(QtWidgets.QWidget):
    """Modern slider with label and value display"""

    valueChanged = QtCore.Signal(int)

    def __init__(self, label: str, minimum: int, maximum: int, value: int,
                 suffix: str = "", description: str = ""):
        super().__init__()
        self.suffix = suffix

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(4)

        # Header with label and value
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        self.lbl = QtWidgets.QLabel(label)
        self.lbl.setProperty("subheading", True)
        header.addWidget(self.lbl)

        header.addStretch()

        self.val = QtWidgets.QLabel(f"{value}{suffix}")
        self.val.setStyleSheet("color: #0d7377; font-weight: 600; font-size: 11pt;")
        self.val.setMinimumWidth(70)
        self.val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        header.addWidget(self.val)

        layout.addLayout(header)

        # Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(value)
        self.slider.setMinimumHeight(24)
        layout.addWidget(self.slider)

        # Description
        if description:
            desc = QtWidgets.QLabel(description)
            desc.setStyleSheet("color: #9d9d9d; font-size: 9pt;")
            desc.setWordWrap(True)
            layout.addWidget(desc)

        # Connections
        self.slider.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, v: int):
        self.val.setText(f"{v}{self.suffix}")
        self.valueChanged.emit(v)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, v: int):
        self.slider.setValue(int(v))


class IconButton(QtWidgets.QPushButton):
    """Button with icon and optional text"""

    def __init__(self, icon: str, text: str = "", tooltip: str = "",
                 primary: bool = False, danger: bool = False):
        if text:
            super().__init__(f"{icon} {text}")
        else:
            super().__init__(icon)

        if tooltip:
            self.setToolTip(tooltip)

        if primary:
            self.setProperty("primary", True)
        elif danger:
            self.setProperty("danger", True)

        self.setMinimumHeight(36)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)


class StatusLabel(QtWidgets.QLabel):
    """Status label with icon and color coding"""

    def __init__(self, text: str = "Ready"):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMinimumHeight(60)
        self.setStyleSheet("""
            padding: 12px;
            background-color: #252525;
            border-radius: 6px;
            border: 1px solid #3d3d3d;
            font-size: 10pt;
        """)

    def setStatus(self, text: str, status: str = "info"):
        """Set status with icon and color

        Args:
            text: Status message
            status: One of 'info', 'success', 'warning', 'error'
        """
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
        }

        colors = {
            "info": "#0d7377",
            "success": "#2d6a3d",
            "warning": "#b87503",
            "error": "#8b1e1e",
        }

        icon = icons.get(status, "ℹ️")
        color = colors.get(status, "#0d7377")

        self.setText(f"{icon} {text}")
        self.setStyleSheet(f"""
            padding: 12px;
            background-color: #252525;
            border-radius: 6px;
            border-left: 4px solid {color};
            font-size: 10pt;
        """)


class CollapsibleSection(QtWidgets.QWidget):
    """Collapsible section with title"""

    def __init__(self, title: str, collapsed: bool = False):
        super().__init__()

        self.is_collapsed = collapsed

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        self.header = QtWidgets.QPushButton(f"{'▶' if collapsed else '▼'} {title}")
        self.header.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 12px;
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                font-weight: 600;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
        """)
        self.header.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.header.clicked.connect(self.toggle)
        main_layout.addWidget(self.header)

        # Content
        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addWidget(self.content)

        if collapsed:
            self.content.hide()

    def toggle(self):
        """Toggle collapsed state"""
        self.is_collapsed = not self.is_collapsed

        if self.is_collapsed:
            self.content.hide()
            # Update arrow
            text = self.header.text()
            self.header.setText(text.replace('▼', '▶'))
        else:
            self.content.show()
            # Update arrow
            text = self.header.text()
            self.header.setText(text.replace('▶', '▼'))

    def addWidget(self, widget: QtWidgets.QWidget):
        """Add widget to content area"""
        self.content_layout.addWidget(widget)

    def addLayout(self, layout: QtWidgets.QLayout):
        """Add layout to content area"""
        self.content_layout.addLayout(layout)


class ObjectListItem(QtWidgets.QWidget):
    """Custom list item for objects with visibility toggle and color"""

    visibilityChanged = QtCore.Signal(int, bool)
    selected = QtCore.Signal(int)
    deleted = QtCore.Signal(int)

    def __init__(self, obj_id: int, name: str, color: tuple, visible: bool = True):
        super().__init__()
        self.obj_id = obj_id

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Visibility toggle
        self.vis_btn = QtWidgets.QPushButton("👁️" if visible else "👁️‍🗨️")
        self.vis_btn.setMaximumWidth(40)
        self.vis_btn.setCheckable(True)
        self.vis_btn.setChecked(visible)
        self.vis_btn.clicked.connect(lambda: self.visibilityChanged.emit(obj_id, self.vis_btn.isChecked()))
        layout.addWidget(self.vis_btn)

        # Color indicator
        color_label = QtWidgets.QLabel("●")
        color_label.setStyleSheet(f"color: rgb{color}; font-size: 16pt;")
        layout.addWidget(color_label)

        # Name
        name_label = QtWidgets.QLabel(name)
        name_label.setStyleSheet("font-weight: 500;")
        layout.addWidget(name_label, 1)

        # Delete button
        del_btn = QtWidgets.QPushButton("🗑️")
        del_btn.setMaximumWidth(40)
        del_btn.setProperty("danger", True)
        del_btn.clicked.connect(lambda: self.deleted.emit(obj_id))
        layout.addWidget(del_btn)

        # Make clickable
        self.setStyleSheet("""
            ObjectListItem {
                background-color: #2d2d2d;
                border-radius: 6px;
                margin: 2px;
            }
            ObjectListItem:hover {
                background-color: #3d3d3d;
            }
        """)

        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.selected.emit(self.obj_id)
        super().mousePressEvent(event)


class ParameterGroup(QtWidgets.QGroupBox):
    """Group box for parameters with consistent styling"""

    def __init__(self, title: str, description: str = ""):
        super().__init__(title)

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setSpacing(8)

        if description:
            desc = QtWidgets.QLabel(description)
            desc.setStyleSheet("color: #9d9d9d; font-size: 9pt; font-weight: normal;")
            desc.setWordWrap(True)
            self.main_layout.addWidget(desc)


class ModernComboBox(QtWidgets.QComboBox):
    """Styled combo box with better appearance"""

    def __init__(self, items: list = None):
        super().__init__()
        self.setMinimumHeight(36)

        if items:
            self.addItems(items)


class ModernProgressBar(QtWidgets.QWidget):
    """Custom progress bar with percentage and status"""

    def __init__(self):
        super().__init__()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready")
        header.addWidget(self.status_label)

        header.addStretch()

        self.percent_label = QtWidgets.QLabel("0%")
        self.percent_label.setStyleSheet("color: #0d7377; font-weight: 600;")
        header.addWidget(self.percent_label)

        layout.addLayout(header)

        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

    def setValue(self, value: int, status: str = ""):
        """Set progress value and optional status"""
        self.progress.setValue(value)
        self.percent_label.setText(f"{value}%")

        if status:
            self.status_label.setText(status)

    def setStatus(self, status: str):
        """Set status text"""
        self.status_label.setText(status)

    def reset(self):
        """Reset progress"""
        self.setValue(0, "Ready")
