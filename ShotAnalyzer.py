import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QMenuBar, QAction, QFileDialog
from extract_b1b2b3_start import extract_b1b2b3_start
from extract_data_quality_start import extract_data_quality_start
from selection_menu_function import selection_menu_function
from save_fcn import save_function
from read_gamefile import read_gamefile

class ShotAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Shot Analyzer v0.43i")
        self.setGeometry(100, 100, 800, 600)

        # Create a menu bar
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # File menu
        file_menu = menu_bar.addMenu("File")
        load_action = QAction("Load new file", self)
        load_action.triggered.connect(self.load_new_file)
        file_menu.addAction(load_action)

        append_action = QAction("Load & Append file", self)
        append_action.triggered.connect(self.append_file)
        file_menu.addAction(append_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        save_all_action = QAction("Save all to new file", self)
        save_all_action.triggered.connect(self.save_all_to_new_file)
        file_menu.addAction(save_all_action)

        save_selected_action = QAction("Save selected to new file", self)
        save_selected_action.triggered.connect(self.save_selected_to_new_file)
        file_menu.addAction(save_selected_action)

        # Interpreter menu
        interpreter_menu = menu_bar.addMenu("Interpreter")
        run_all_action = QAction("Run All Interpreters", self)
        run_all_action.triggered.connect(self.run_all_interpreters)
        interpreter_menu.addAction(run_all_action)

        analyze_quality_action = QAction("Analyze data quality", self)
        analyze_quality_action.triggered.connect(self.analyze_data_quality)
        interpreter_menu.addAction(analyze_quality_action)

        # Columns menu
        columns_menu = menu_bar.addMenu("Columns")
        add_blank_column_action = QAction("Add blank column", self)
        add_blank_column_action.triggered.connect(self.add_blank_column)
        columns_menu.addAction(add_blank_column_action)

        hide_columns_action = QAction("Hide all columns", self)
        hide_columns_action.triggered.connect(self.hide_all_columns)
        columns_menu.addAction(hide_columns_action)

        show_columns_action = QAction("Show all columns", self)
        show_columns_action.triggered.connect(self.show_all_columns)
        columns_menu.addAction(show_columns_action)

        # Shots Selection menu
        selection_menu = menu_bar.addMenu("Shots Selection")
        select_marked_action = QAction("Select marked shots", self)
        select_marked_action.triggered.connect(self.select_marked_shots)
        selection_menu.addAction(select_marked_action)

        unselect_marked_action = QAction("Unselect marked shots", self)
        unselect_marked_action.triggered.connect(self.unselect_marked_shots)
        selection_menu.addAction(unselect_marked_action)

        invert_selection_action = QAction("Invert Selection", self)
        invert_selection_action.triggered.connect(self.invert_selection)
        selection_menu.addAction(invert_selection_action)

        select_all_action = QAction("Select all shots", self)
        select_all_action.triggered.connect(self.select_all_shots)
        selection_menu.addAction(select_all_action)

        unselect_all_action = QAction("Unselect all shots", self)
        unselect_all_action.triggered.connect(self.unselect_all_shots)
        selection_menu.addAction(unselect_all_action)

        # Create a table widget
        self.table = QTableWidget(self)
        self.table.setGeometry(10, 10, 780, 580)
        self.table.setColumnCount(5)  # Example column count
        self.table.setRowCount(10)   # Example row count
        self.table.setHorizontalHeaderLabels(["ShotID", "Player", "ErrorID", "ErrorText", "Selected"])

        # Example data population
        for row in range(10):
            for col in range(5):
                self.table.setItem(row, col, QTableWidgetItem(f"Data {row},{col}"))

    # Placeholder methods for menu actions
    def process_gamefile(self, file_path):
        # Use the existing read_gamefile function
        print(f"Reading game data from {file_path}")
        game_data = read_gamefile(file_path)
        print("Game data loaded successfully.")
        # Further processing of game_data can be added here
        pass

    def load_new_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;ShotAnalyzer Files (*.saf)", options=options)
        if file_path:
            print(f"Selected file: {file_path}")
            self.process_gamefile(file_path)

    def append_file(self):
        print("Load & Append file")
        # Implement append logic if needed

    def save_file(self):
        print("Save file")
        save_function(0)

    def save_all_to_new_file(self):
        print("Save all to new file")
        save_function(1)

    def save_selected_to_new_file(self):
        print("Save selected to new file")
        save_function(2)

    def run_all_interpreters(self):
        print("Run All Interpreters")
        extract_b1b2b3_start(self.SA, self.player)

    def analyze_data_quality(self):
        print("Analyze data quality")
        extract_data_quality_start(self.SA, self.param, self.player)

    def add_blank_column(self):
        print("Add blank column")
        # Implement logic for adding a blank column

    def hide_all_columns(self):
        print("Hide all columns")
        selection_menu_function('Hide all columns', self.SA)

    def show_all_columns(self):
        print("Show all columns")
        selection_menu_function('Show all columns', self.SA)

    def select_marked_shots(self):
        print("Select marked shots")
        selection_menu_function('Select marked shots', self.SA)

    def unselect_marked_shots(self):
        print("Unselect marked shots")
        selection_menu_function('Unselect marked shots', self.SA)

    def invert_selection(self):
        print("Invert Selection")
        selection_menu_function('Invert Selection', self.SA)

    def select_all_shots(self):
        print("Select all shots")
        selection_menu_function('Select all shots', self.SA)

    def unselect_all_shots(self):
        print("Unselect all shots")
        selection_menu_function('Unselect all shots', self.SA)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    analyzer = ShotAnalyzer()
    analyzer.show()
    sys.exit(app.exec())