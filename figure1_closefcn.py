from PyQt5.QtWidgets import QApplication

def figure1_closefcn(h):
    app = QApplication.instance()
    for widget in app.allWidgets():
        if widget.objectName() == 'Table_figure':
            widget.close()
    h.close()