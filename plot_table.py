from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView
from PyQt6.QtGui import QPen, QBrush, QColor
from PyQt6.QtCore import Qt

def plot_table(param, player):
    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    view.setWindowTitle("Billard Table")
    view.setGeometry(param['TablePosition'][0], param['TablePosition'][1], param['TablePosition'][2], param['TablePosition'][3])

    if not player['setting'].get('plot_only_blue_table', False):
        # Draw table frame
        frame_pen = QPen(QColor(128, 128, 0))
        frame_brush = QBrush(QColor(128, 128, 0))
        scene.addRect(-param['framewidth'], -param['framewidth'], param['size'][1] + 2 * param['framewidth'], param['size'][0] + 2 * param['framewidth'], frame_pen, frame_brush)

        # Draw cushion line
        cushion_pen = QPen(QColor(0, 0, 255))
        scene.addRect(0, 0, param['size'][1], param['size'][0], cushion_pen)

        # Draw diamonds
        diamond_pen = QPen(QColor(0, 0, 0))
        diamond_brush = QBrush(QColor(0, 0, 0))
        for i in range(9):
            x = param['size'][1] / 8 * i
            scene.addEllipse(x - param['rdiam'], -param['diamdist'] - param['rdiam'], 2 * param['rdiam'], 2 * param['rdiam'], diamond_pen, diamond_brush)
            scene.addEllipse(x - param['rdiam'], param['size'][0] + param['diamdist'] - param['rdiam'], 2 * param['rdiam'], 2 * param['rdiam'], diamond_pen, diamond_brush)

        for i in range(5):
            y = param['size'][0] / 4 * i
            scene.addEllipse(-param['diamdist'] - param['rdiam'], y - param['rdiam'], 2 * param['rdiam'], 2 * param['rdiam'], diamond_pen, diamond_brush)
            scene.addEllipse(param['size'][1] + param['diamdist'] - param['rdiam'], y - param['rdiam'], 2 * param['rdiam'], 2 * param['rdiam'], diamond_pen, diamond_brush)

    view.show()
    return scene