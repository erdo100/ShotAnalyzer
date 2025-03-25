import pandas as pd
from PyQt5.QtWidgets import QFileDialog

def xls_import_fcn(SA):
    print("Importing data from XLS...")

    # Prompt user to select an Excel file to import
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(None, "Import from XLS", "", "Excel Files (*.xlsx)", options=options)

    if not file_path:
        print("Import aborted.")
        return

    try:
        # Read the Excel file
        imported_data = pd.read_excel(file_path)

        # Check for column name conflicts
        for column in imported_data.columns:
            if column not in SA['Table'].columns:
                print(f"Adding new column: {column}")
                SA['Table'][column] = None

        # Merge imported data into the existing table
        for index, row in imported_data.iterrows():
            if index < len(SA['Table']):
                for column in imported_data.columns:
                    SA['Table'].at[index, column] = row[column]
            else:
                # Append new rows if necessary
                SA['Table'] = pd.concat([SA['Table'], pd.DataFrame([row])], ignore_index=True)

        # Update the GUI
        print("Data imported successfully.")
        update_shot_list(SA)

    except Exception as e:
        print(f"Error during import: {str(e)}")