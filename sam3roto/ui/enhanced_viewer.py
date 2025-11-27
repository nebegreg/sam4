"""
Viewer amélioré avec timeline, comparaison et modes d'overlay multiples
Basé sur les best practices de video segmentation UX
"""
from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
from typing import Optional
import numpy as np

class EnhancedViewer(QtWidgets.QWidget):
    """Viewer amélioré avec timeline et comparaison"""
    pointAdded = QtCore.Signal(int, int, int)  # x,y,label
    boxAdded = QtCore.Signal(int, int, int, int, int)  # x1,y1,x2,y2,label
    frameChanged = QtCore.Signal(int)  # frame_idx

    def __init__(self):
        super().__init__()
        self._build_ui()

        self._img_pix: Optional[QtGui.QPixmap] = None
        self._ov_img: Optional[QtGui.QImage] = None

        self._mode = "point"
        self._point_label = 1
        self._box_label = 1
        self._box_start = None

        # Overlay settings
        self._overlay_mode = "blend"  # blend, contour, side_by_side, onion_skin
        self._overlay_opacity = 0.55
        self._show_overlay = True

        # Split view
        self._split_position = 0.5  # For side-by-side mode

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Graphics view
        self.view = QtWidgets.QGraphicsView()
        self.view.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing |
                                QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

        self._scene = QtWidgets.QGraphicsScene(self.view)
        self.view.setScene(self._scene)

        self._img_item = QtWidgets.QGraphicsPixmapItem()
        self._ov_item = QtWidgets.QGraphicsPixmapItem()
        self._contour_item = None

        self._scene.addItem(self._img_item)
        self._scene.addItem(self._ov_item)

        layout.addWidget(self.view, 1)

        # Overlay controls
        controls = QtWidgets.QHBoxLayout()

        # Overlay mode
        controls.addWidget(QtWidgets.QLabel("Overlay:"))
        self.cb_overlay_mode = QtWidgets.QComboBox()
        self.cb_overlay_mode.addItems([
            "Blend", "Contour Only", "Side-by-Side", "Onion Skin", "Checkerboard"
        ])
        self.cb_overlay_mode.currentTextChanged.connect(self._on_overlay_mode_changed)
        controls.addWidget(self.cb_overlay_mode)

        # Opacity slider
        controls.addWidget(QtWidgets.QLabel("Opacity:"))
        self.slider_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(55)
        self.slider_opacity.setMaximumWidth(150)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        controls.addWidget(self.slider_opacity)

        self.lbl_opacity = QtWidgets.QLabel("55%")
        controls.addWidget(self.lbl_opacity)

        # Toggle overlay
        self.chk_show_overlay = QtWidgets.QCheckBox("Show")
        self.chk_show_overlay.setChecked(True)
        self.chk_show_overlay.toggled.connect(self._on_toggle_overlay)
        controls.addWidget(self.chk_show_overlay)

        controls.addStretch()
        layout.addLayout(controls)

        # Mouse events
        self.view.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.view.viewport():
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                return self._handle_mouse_press(event)
            elif event.type() == QtCore.QEvent.Type.MouseMove:
                return self._handle_mouse_move(event)
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                return self._handle_mouse_release(event)
        return super().eventFilter(obj, event)

    def _handle_mouse_press(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton and self._mode == "box":
            pos = self.view.mapToScene(e.pos())
            self._box_start = (int(pos.x()), int(pos.y()))
            return True
        return False

    def _handle_mouse_move(self, e: QtGui.QMouseEvent):
        if self._mode == "box" and self._box_start is not None:
            pos = self.view.mapToScene(e.pos())
            # Update box preview
            return True
        return False

    def _handle_mouse_release(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.view.mapToScene(e.pos())

            if self._mode == "point":
                self.pointAdded.emit(int(pos.x()), int(pos.y()), int(self._point_label))
                return True

            elif self._mode == "box" and self._box_start is not None:
                x1, y1 = self._box_start
                x2, y2 = int(pos.x()), int(pos.y())
                self._box_start = None
                self.boxAdded.emit(min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2), int(self._box_label))
                return True
        return False

    def set_mode(self, mode: str):
        self._mode = mode

    def set_point_label(self, label: int):
        self._point_label = int(label)

    def set_box_label(self, label: int):
        self._box_label = int(label)

    def set_image(self, pix: QtGui.QPixmap):
        self._img_pix = pix
        self._img_item.setPixmap(pix)
        self._img_item.setPos(0, 0)
        self._scene.setSceneRect(self._img_item.boundingRect())
        self.view.fitInView(self._img_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._update_overlay_display()

    def set_overlay(self, qimg: QtGui.QImage, opacity: float = 0.55):
        self._ov_img = qimg
        self._overlay_opacity = opacity
        self.slider_opacity.setValue(int(opacity * 100))
        self._update_overlay_display()

    def clear_overlay(self):
        self._ov_img = None
        self._ov_item.setPixmap(QtGui.QPixmap())
        if self._contour_item:
            self._scene.removeItem(self._contour_item)
            self._contour_item = None

    def _on_overlay_mode_changed(self, mode: str):
        mode_map = {
            "Blend": "blend",
            "Contour Only": "contour",
            "Side-by-Side": "side_by_side",
            "Onion Skin": "onion_skin",
            "Checkerboard": "checkerboard"
        }
        self._overlay_mode = mode_map.get(mode, "blend")
        self._update_overlay_display()

    def _on_opacity_changed(self, value: int):
        self._overlay_opacity = value / 100.0
        self.lbl_opacity.setText(f"{value}%")
        self._update_overlay_display()

    def _on_toggle_overlay(self, checked: bool):
        self._show_overlay = checked
        self._update_overlay_display()

    def _update_overlay_display(self):
        if not self._show_overlay or self._ov_img is None or self._ov_img.isNull():
            self.clear_overlay()
            return

        if self._overlay_mode == "blend":
            self._display_blend()
        elif self._overlay_mode == "contour":
            self._display_contour()
        elif self._overlay_mode == "side_by_side":
            self._display_side_by_side()
        elif self._overlay_mode == "onion_skin":
            self._display_onion_skin()
        elif self._overlay_mode == "checkerboard":
            self._display_checkerboard()

    def _display_blend(self):
        """Mode blend classique"""
        self._ov_item.setPixmap(QtGui.QPixmap.fromImage(self._ov_img))
        self._ov_item.setOpacity(self._overlay_opacity)
        self._ov_item.setPos(0, 0)
        if self._contour_item:
            self._scene.removeItem(self._contour_item)
            self._contour_item = None

    def _display_contour(self):
        """Affiche uniquement les contours du masque"""
        # Convert overlay to grayscale mask
        img = self._ov_img.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        ptr = img.bits()
        h, w = img.height(), img.width()

        # Extract alpha channel
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
        alpha = arr[:, :, 3]

        # Find contours
        import cv2
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on transparent image
        contour_img = np.zeros((h, w, 4), dtype=np.uint8)
        cv2.drawContours(contour_img, contours, -1, (255, 255, 0, 255), 2)

        # Convert back to QImage
        qimg = QtGui.QImage(contour_img.data, w, h, 4*w, QtGui.QImage.Format.Format_RGBA8888).copy()
        self._ov_item.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self._ov_item.setOpacity(1.0)
        self._ov_item.setPos(0, 0)

    def _display_side_by_side(self):
        """Split screen: original | masked"""
        if self._img_pix is None:
            return

        w, h = self._img_pix.width(), self._img_pix.height()
        split_x = int(w * self._split_position)

        # Create composite
        composite = QtGui.QImage(w, h, QtGui.QImage.Format.Format_RGBA8888)
        composite.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(composite)

        # Left side: original
        painter.drawPixmap(0, 0, split_x, h, self._img_pix, 0, 0, split_x, h)

        # Right side: overlay
        painter.setOpacity(self._overlay_opacity)
        painter.drawImage(split_x, 0, self._ov_img, split_x, 0, w - split_x, h)

        # Draw split line
        painter.setOpacity(1.0)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 2))
        painter.drawLine(split_x, 0, split_x, h)

        painter.end()

        self._ov_item.setPixmap(QtGui.QPixmap.fromImage(composite))
        self._ov_item.setOpacity(1.0)
        self._ov_item.setPos(0, 0)

    def _display_onion_skin(self):
        """Onion skin: blend with reduced opacity for temporal comparison"""
        self._ov_item.setPixmap(QtGui.QPixmap.fromImage(self._ov_img))
        self._ov_item.setOpacity(self._overlay_opacity * 0.5)  # Plus transparent pour onion skin
        self._ov_item.setPos(0, 0)

    def _display_checkerboard(self):
        """Checkerboard background pour voir la transparence"""
        if self._img_pix is None:
            return

        w, h = self._img_pix.width(), self._img_pix.height()

        # Create checkerboard
        checker = QtGui.QImage(w, h, QtGui.QImage.Format.Format_RGB888)
        checker_painter = QtGui.QPainter(checker)

        tile_size = 20
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                color = QtGui.QColor(200, 200, 200) if ((x // tile_size) + (y // tile_size)) % 2 == 0 else QtGui.QColor(150, 150, 150)
                checker_painter.fillRect(x, y, tile_size, tile_size, color)
        checker_painter.end()

        # Composite
        composite = QtGui.QImage(w, h, QtGui.QImage.Format.Format_RGBA8888)
        composite_painter = QtGui.QPainter(composite)
        composite_painter.drawImage(0, 0, checker)
        composite_painter.setOpacity(self._overlay_opacity)
        composite_painter.drawImage(0, 0, self._ov_img)
        composite_painter.end()

        self._ov_item.setPixmap(QtGui.QPixmap.fromImage(composite))
        self._ov_item.setOpacity(1.0)
        self._ov_item.setPos(0, 0)


class Timeline(QtWidgets.QWidget):
    """Timeline widget avec aperçu des frames"""
    frameChanged = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.setMaximumHeight(150)

        self._num_frames = 0
        self._current_frame = 0
        self._thumbnails = []  # List of QPixmap thumbnails
        self._hover_frame = -1

        self.setMouseTracking(True)

    def set_num_frames(self, n: int):
        self._num_frames = n
        self.update()

    def set_current_frame(self, idx: int):
        self._current_frame = idx
        self.update()

    def add_thumbnail(self, frame_idx: int, pixmap: QtGui.QPixmap):
        """Ajoute un thumbnail pour une frame"""
        while len(self._thumbnails) <= frame_idx:
            self._thumbnails.append(None)
        # Scale to thumbnail size
        thumb = pixmap.scaled(120, 68, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)
        self._thumbnails[frame_idx] = thumb
        self.update()

    def paintEvent(self, e: QtGui.QPaintEvent):
        if self._num_frames == 0:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(self.rect(), QtGui.QColor(40, 40, 40))

        # Frame indicators
        frame_width = max(2, w / self._num_frames)

        for i in range(self._num_frames):
            x = int(i * frame_width)

            # Draw frame rect
            if i == self._current_frame:
                painter.fillRect(x, h-20, int(frame_width), 20, QtGui.QColor(0, 120, 215))
            elif i == self._hover_frame:
                painter.fillRect(x, h-20, int(frame_width), 20, QtGui.QColor(100, 100, 100))
            else:
                painter.fillRect(x, h-20, int(frame_width), 20, QtGui.QColor(60, 60, 60))

            # Draw thumbnail if available
            if i < len(self._thumbnails) and self._thumbnails[i] is not None:
                thumb = self._thumbnails[i]
                thumb_x = x + (frame_width - thumb.width()) / 2
                thumb_y = 10
                painter.drawPixmap(int(thumb_x), int(thumb_y), thumb)

        # Frame numbers
        painter.setPen(QtGui.QColor(200, 200, 200))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        # Show frame numbers every N frames
        step = max(1, self._num_frames // 10)
        for i in range(0, self._num_frames, step):
            x = int(i * frame_width)
            painter.drawText(x + 2, h - 5, str(i + 1))

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._num_frames == 0:
            return
        frame_width = self.width() / self._num_frames
        self._hover_frame = int(e.pos().x() / frame_width)
        self.update()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if self._num_frames == 0:
            return
        frame_width = self.width() / self._num_frames
        frame_idx = int(e.pos().x() / frame_width)
        frame_idx = max(0, min(self._num_frames - 1, frame_idx))
        self.frameChanged.emit(frame_idx)

    def leaveEvent(self, e):
        self._hover_frame = -1
        self.update()
