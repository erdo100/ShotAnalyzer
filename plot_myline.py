from PyQt5.QtWidgets import QGraphicsLineItem, QGraphicsScene
from PyQt5.QtGui import QPen, QColor

def plot_myline(color, x0, y0, scene):
    pen = QPen(QColor(color))
    pen.setWidth(4)

    line = QGraphicsLineItem(x0, y0, x0, y0)
    line.setPen(pen)
    scene.addItem(line)

    def mouse_move_event(event):
        line.setLine(line.line().x1(), line.line().y1(), event.scenePos().x(), event.scenePos().y())

    def mouse_release_event(event):
        scene.removeEventFilter(line)

    scene.installEventFilter(line)
    line.mouseMoveEvent = mouse_move_event
    line.mouseReleaseEvent = mouse_release_event