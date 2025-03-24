from PyQt6.QtWidgets import QMenu, QAction

def figure_menu(hfigure, SA):
    # Files Menu
    mfile = QMenu("File", hfigure)
    hfigure.menuBar().addMenu(mfile)

    mfile.addAction(QAction("Load new file", hfigure, triggered=lambda: Read_All_GameData(0, SA)))
    mfile.addAction(QAction("Load & Append file", hfigure, triggered=lambda: Read_All_GameData(1, SA)))
    mfile.addAction(QAction("Save", hfigure, triggered=lambda: SaveFcn(0, SA)))
    mfile.addAction(QAction("Save all to new file", hfigure, triggered=lambda: SaveFcn(1, SA)))
    mfile.addAction(QAction("Save selected to new file", hfigure, triggered=lambda: SaveFcn(2, SA)))
    mfile.addAction(QAction("Export selected shots to PNG", hfigure, triggered=lambda: export_pdf_fcn(SA)))
    mfile.addAction(QAction("Export selected shots to MP4", hfigure, triggered=lambda: export_mp4_fcn(SA)))
    mfile.addAction(QAction("Export selected shots to BBS", hfigure, triggered=lambda: export_bbs_fcn(SA)))
    mfile.addSeparator()
    mfile.addAction(QAction("Export full shots table to XLS", hfigure, triggered=lambda: XLS_ExportALL_Fcn(SA)))
    mfile.addAction(QAction("Export visible columns to XLS", hfigure, triggered=lambda: XLS_Export_Fcn(SA)))
    mfile.addAction(QAction("Import all shots table from XLS", hfigure, triggered=lambda: XLS_Import_Fcn(SA)))

    # Interpreter Menu
    mextract = QMenu("Interpreter", hfigure)
    hfigure.menuBar().addMenu(mextract)

    mextract.addAction(QAction("Run All Interpreters", hfigure, triggered=lambda: extract_all_at_once(SA)))
    mextract.addAction(QAction("Analyze data quality", hfigure, triggered=lambda: extract_data_quality_start(SA)))
    mextract.addAction(QAction("Add mirrored positions", hfigure, triggered=lambda: AddMirroredPositions(SA).execute()))
    mextract.addAction(QAction("Identify B1B2B3", hfigure, triggered=lambda: extract_b1b2b3_start(SA)))
    mextract.addAction(QAction("Identify B1B2B3 Position", hfigure, triggered=lambda: extract_b1b2b3_position(SA)))
    mextract.addAction(QAction("Identify Events", hfigure, triggered=lambda: extract_events_start(SA)))
    mextract.addAction(QAction("ReIdentify Events", hfigure, triggered=lambda: extract2_events_start(SA)))

    # Columns Selection Menu
    mcol = QMenu("Columns", hfigure)
    hfigure.menuBar().addMenu(mcol)

    mcol.addAction(QAction("Add blank column", hfigure, triggered=lambda: user_defined_column(SA)))
    mcol.addAction(QAction("Hide all columns", hfigure, triggered=lambda: shotlist_menu_function(SA)))
    mcol.addAction(QAction("Show all columns", hfigure, triggered=lambda: shotlist_menu_function(SA)))
    mcol.addAction(QAction("Choose columns", hfigure, triggered=lambda: shotlist_menu_function(SA)))
    mcol.addAction(QAction("Import column names to display from XLS", hfigure, triggered=lambda: shotlist_menu_function(SA)))

    # Selection Menu
    msel = QMenu("Shots Selection", hfigure)
    hfigure.menuBar().addMenu(msel)

    msel.addAction(QAction("Select marked shots", hfigure, triggered=lambda: selection_menu_function(SA)))
    msel.addAction(QAction("Unselect marked shots", hfigure, triggered=lambda: selection_menu_function(SA)))
    msel.addAction(QAction("Invert Selection", hfigure, triggered=lambda: selection_menu_function(SA)))
    msel.addAction(QAction("Select all shots", hfigure, triggered=lambda: selection_menu_function(SA)))
    msel.addAction(QAction("Unselect all shots", hfigure, triggered=lambda: selection_menu_function(SA)))
    msel.addAction(QAction("Hide selected shots", hfigure, triggered=lambda: selection_menu_function(SA)))
    msel.addAction(QAction("Delete selected shots", hfigure, triggered=lambda: selection_menu_function(SA)))