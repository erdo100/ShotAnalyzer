import pandas as pd
import os
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime

def xls_export_all_fcn(SA, param=None):
    """
    Exports all data from the SA structure to an Excel file, including
    columns that are not currently visible.
    """
    print("Writing all data to XLS...")

    # Add default filename based on current date/time
    default_filename = f"SA_Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Get directory from SA structure if available
    initial_dir = os.path.dirname(SA.get('fullfilename', '')) or '.'
    
    # Prompt user to select save location
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getSaveFileName(
        None,
        "Export ALL data to XLS",
        os.path.join(initial_dir, default_filename),
        "Excel Files (*.xlsx);;All Files (*)",
        options=options
    )

    if not file_path:
        print("Export aborted.")
        return
        
    try:
        # Create a DataFrame with all columns
        df = pd.DataFrame(SA['Table'])
        
        # Write to Excel
        df.to_excel(file_path, index=False, sheet_name='ShotAnalyzer Data')
        
        print(f"Successfully exported {len(df)} rows and {len(df.columns)} columns")
        print(f"File saved to: {file_path}")
        
    except Exception as e:
        print(f"Error during export: {str(e)}")
        return