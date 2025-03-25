from PyQt5.QtWidgets import QInputDialog

def user_defined_column(SA):
    # Prompt the user to input a column name
    column_name, ok = QInputDialog.getText(None, "Add Blank Column", "Enter column name:")

    if ok and column_name:
        # Add the new column to the table with default values
        SA['Table'][column_name] = [None] * len(SA['Table'])

        # Update the GUI
        print(f"Added blank column: {column_name}")
        update_shot_list(SA)
    else:
        print("No column added.")