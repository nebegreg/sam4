from __future__ import annotations
from PySide6 import QtCore, QtWidgets

class LabeledSlider(QtWidgets.QWidget):
    def __init__(self, label: str, mn: int, mx: int, value: int, suffix: str = ""):
        super().__init__()
        self.suffix = suffix
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.lbl = QtWidgets.QLabel(label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(mn)
        self.slider.setMaximum(mx)
        self.slider.setValue(value)
        self.val = QtWidgets.QLabel(f"{value}{suffix}")
        self.val.setMinimumWidth(80)
        layout.addWidget(self.lbl, 1)
        layout.addWidget(self.slider, 2)
        layout.addWidget(self.val, 0)
        self.slider.valueChanged.connect(self._on)

    def _on(self, v: int):
        self.val.setText(f"{v}{self.suffix}")

    def value(self) -> int:
        return int(self.slider.value())

    def setValue(self, v: int):
        self.slider.setValue(int(v))
