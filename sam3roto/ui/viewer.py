from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets

class Viewer(QtWidgets.QGraphicsView):
    pointAdded = QtCore.Signal(int, int, int)  # x,y,label
    boxAdded = QtCore.Signal(int, int, int, int, int)  # x1,y1,x2,y2,label

    def __init__(self):
        super().__init__()
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self._img_item = QtWidgets.QGraphicsPixmapItem()
        self._ov_item = QtWidgets.QGraphicsPixmapItem()
        self._ov_item.setOpacity(0.55)

        self._scene.addItem(self._img_item)
        self._scene.addItem(self._ov_item)

        self._mode = "point"
        self._point_label = 1
        self._box_label = 1
        self._box_start = None
        self._box_rect_item = None

    def set_mode(self, mode: str):
        self._mode = mode

    def set_point_label(self, label: int):
        self._point_label = int(label)

    def set_box_label(self, label: int):
        self._box_label = int(label)

    def set_image(self, pix: QtGui.QPixmap):
        self._img_item.setPixmap(pix)
        self._img_item.setPos(0,0)
        self._scene.setSceneRect(self._img_item.boundingRect())
        self.fitInView(self._img_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def set_overlay(self, qimg: QtGui.QImage, opacity: float = 0.55):
        self._ov_item.setOpacity(float(opacity))
        self._ov_item.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self._ov_item.setPos(0,0)

    def clear_overlay(self):
        self._ov_item.setPixmap(QtGui.QPixmap())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._img_item.pixmap().isNull():
            self.fitInView(self._img_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton and self._mode == "box":
            pos = self.mapToScene(e.pos())
            self._box_start = (int(pos.x()), int(pos.y()))
            # draw rect preview
            if self._box_rect_item is None:
                pen = QtGui.QPen(QtGui.QColor(255,255,0), 2)
                self._box_rect_item = self._scene.addRect(0,0,0,0, pen)
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._mode == "box" and self._box_start is not None and self._box_rect_item is not None:
            pos = self.mapToScene(e.pos())
            x1,y1 = self._box_start
            x2,y2 = int(pos.x()), int(pos.y())
            rect = QtCore.QRectF(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))
            self._box_rect_item.setRect(rect)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton and self._mode == "point":
            pos = self.mapToScene(e.pos())
            self.pointAdded.emit(int(pos.x()), int(pos.y()), int(self._point_label))

        if e.button() == QtCore.Qt.MouseButton.LeftButton and self._mode == "box" and self._box_start is not None:
            pos = self.mapToScene(e.pos())
            x1,y1 = self._box_start
            x2,y2 = int(pos.x()), int(pos.y())
            self._box_start = None
            if self._box_rect_item is not None:
                self._scene.removeItem(self._box_rect_item)
                self._box_rect_item = None
            self.boxAdded.emit(min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2), int(self._box_label))
        super().mouseReleaseEvent(e)
