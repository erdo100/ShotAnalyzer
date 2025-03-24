from PyQt6.QtWidgets import QApplication

def close_table_figure(param):
    app = QApplication.instance()
    for widget in app.allWidgets():
        if widget.objectName() == 'Table_figure':
            param['TablePosition'] = widget.geometry()
            widget.close()
            break