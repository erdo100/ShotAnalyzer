from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
from PyQt6.QtGui import QPen, QBrush, QColor

def plot_player(scene, ball, hit, lw, ti, plotlast, param, player):
    ball_colors = ['white', 'yellow', 'red']
    line_colors = [QColor(255, 255, 255), QColor(204, 204, 0), QColor(255, 0, 0)]

    for bi in range(3):
        # Draw ball
        if player['setting']['ball'][bi]['ball']:
            x = ball[bi]['x'][0 if plotlast == 'last' else ti[bi]]
            y = ball[bi]['y'][0 if plotlast == 'last' else ti[bi]]
            ellipse = QGraphicsEllipseItem(
                x - param['ballR'],
                y - param['ballR'],
                param['ballR'] * 2,
                param['ballR'] * 2
            )
            ellipse.setBrush(QBrush(QColor(ball_colors[bi])))
            ellipse.setPen(QPen(Qt.NoPen))
            scene.addItem(ellipse)

        # Draw line
        if player['setting']['ball'][bi]['line'] or player['setting']['ball'][bi]['marker']:
            pen = QPen(line_colors[bi])
            pen.setWidth(lw[bi])

            for i in range(ti[bi] - 1):
                line = QGraphicsLineItem(
                    ball[bi]['x'][i], ball[bi]['y'][i],
                    ball[bi]['x'][i + 1], ball[bi]['y'][i + 1]
                )
                line.setPen(pen)
                scene.addItem(line)

        # Draw hits
        if isinstance(hit, dict):
            for hi, h in enumerate(hit[bi]['t']):
                x = hit[bi]['XPos'][hi]
                y = hit[bi]['YPos'][hi]
                text = QGraphicsTextItem(str(hi))
                text.setPos(x, y)
                scene.addItem(text)