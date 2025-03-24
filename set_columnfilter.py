from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView

def set_columnfilter(SA):
    class ColumnFilterDialog(QDialog):
        def __init__(self, SA):
            super().__init__()
            self.setWindowTitle("Select data to display shotlist")
            self.resize(450, 300)

            layout = QVBoxLayout()
            self.table = QTableWidget()
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["Enable", "Variable"])
            self.table.setRowCount(len(SA['Table'].columns))
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

            colvisible = [col in SA['ColumnsVisible'] for col in SA['Table'].columns]

            for i, col in enumerate(SA['Table'].columns):
                enable_item = QTableWidgetItem()
                enable_item.setCheckState(2 if colvisible[i] else 0)
                self.table.setItem(i, 0, enable_item)
                self.table.setItem(i, 1, QTableWidgetItem(col))

            layout.addWidget(self.table)
            self.setLayout(layout)

        def accept(self):
            checked = [self.table.item(i, 0).checkState() == 2 for i in range(self.table.rowCount())]
            SA['ColumnsVisible'] = [col for i, col in enumerate(SA['Table'].columns) if checked[i]]
            update_shot_list(SA)
            super().accept()

    dialog = ColumnFilterDialog(SA)
    dialog.exec()