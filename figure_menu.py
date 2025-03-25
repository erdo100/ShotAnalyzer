from PyQt5.QtWidgets import QMenu, QAction

def create_figure_menu(main_window):
    menu_bar = main_window.menuBar()

    # File Menu
    file_menu = menu_bar.addMenu("File")

    load_new_action = QAction("Load new file", main_window)
    load_new_action.triggered.connect(lambda: print("Load new file"))
    file_menu.addAction(load_new_action)

    append_file_action = QAction("Load & Append file", main_window)
    append_file_action.triggered.connect(lambda: print("Load & Append file"))
    file_menu.addAction(append_file_action)

    save_action = QAction("Save", main_window)
    save_action.triggered.connect(lambda: print("Save"))
    file_menu.addAction(save_action)

    save_all_action = QAction("Save all to new file", main_window)
    save_all_action.triggered.connect(lambda: print("Save all to new file"))
    file_menu.addAction(save_all_action)

    save_selected_action = QAction("Save selected to new file", main_window)
    save_selected_action.triggered.connect(lambda: print("Save selected to new file"))
    file_menu.addAction(save_selected_action)

    export_png_action = QAction("Export selected shots to PNG", main_window)
    export_png_action.triggered.connect(lambda: print("Export to PNG"))
    file_menu.addAction(export_png_action)

    export_mp4_action = QAction("Export selected shots to MP4", main_window)
    export_mp4_action.triggered.connect(lambda: print("Export to MP4"))
    file_menu.addAction(export_mp4_action)

    # Additional menus can be added similarly.