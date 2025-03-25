import pandas as pd
from PyQt5.QtWidgets import QFileDialog

def xls_export_fcn(SA):
    print("Writing visible data to XLS ...")

    # Prompt user to select a file to save
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getSaveFileName(None, "Export to XLS", "", "Excel Files (*.xlsx)", options=options)

    if not file_path:
        print("Export aborted.")
        return

    # Prepare data for export
    data = [SA['Table'].columns.tolist()] + SA['Table'].values.tolist()

    # Write data to an Excel file
    df = pd.DataFrame(data[1:], columns=data[0])
    df.to_excel(file_path, index=False)

    print(f"Done with XLS writing. File saved to: {file_path}")