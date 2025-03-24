from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem
from PyQt6.QtGui import QPen, QBrush, QColor

def plot_shot(scene, route, lw, param, plot_setting):
    ball_colors = ['white', 'yellow', 'red']
    line_colors = [QColor(255, 255, 255), QColor(204, 204, 0), QColor(255, 0, 0)]

    for bi in range(3):
        # Draw ball
        if plot_setting['ball'][bi]['ball']:
            ball = QGraphicsEllipseItem(
                route[bi]['x'][0] - param['ballR'],
                route[bi]['y'][0] - param['ballR'],
                param['ballR'] * 2,
                param['ballR'] * 2
            )
            ball.setBrush(QBrush(QColor(ball_colors[bi])))
            ball.setPen(QPen(Qt.NoPen))
            scene.addItem(ball)

        # Draw line
        if plot_setting['ball'][bi]['line'] or plot_setting['ball'][bi]['marker']:
            pen = QPen(line_colors[bi])
            pen.setWidth(lw[bi])

            for i in range(len(route[bi]['x']) - 1):
                line = QGraphicsLineItem(
                    route[bi]['x'][i], route[bi]['y'][i],
                    route[bi]['x'][i + 1], route[bi]['y'][i + 1]
                )
                line.setPen(pen)
                scene.addItem(line)