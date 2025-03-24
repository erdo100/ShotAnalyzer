import pandas as pd
from PyQt6.QtWidgets import QFileDialog

def shotlist_menu_function(action, SA):
    if action == 'Hide all columns':
        SA['ColumnsVisible'] = SA['Table'].columns[:2].tolist()
        print('Done with hiding Columns')

    elif action == 'Show all columns':
        SA['ColumnsVisible'] = SA['Table'].columns.tolist()
        print('Done with showing all Columns')

    elif action == 'Choose columns':
        set_columnfilter(SA)

    elif action == 'Import column names to display from XLS':
        file, _ = QFileDialog.getOpenFileName(None, "Load columns setting", "", "Excel Files (*.xlsx)")
        if not file:
            print('Abort')
            return

        xls_data = pd.read_excel(file, header=None)
        new_names = xls_data.iloc[0].tolist()

        err = 0
        warn = 0

        for ci, name in enumerate(new_names):
            if pd.isna(name):
                print(f"Error: Column number {ci + 1} has no name entry.")
                err += 1

            if name not in SA['Table'].columns:
                print(f"WARNING: Column {name} is not available in the database.")
                warn += 1

        if err > 0:
            print("Check the XLS file carefully.")
            print("Import aborted.")
            return

        SA['ColumnsVisible'] = [name for name in new_names if name in SA['Table'].columns]
        print('Done with Loading Columns from XLS')

    update_shot_list(SA)