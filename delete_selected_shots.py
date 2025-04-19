import pandas as pd

def delete_selected_shots(SA):
    """
    Deletes shots marked as 'Selected' (True) from SA['Data'] and SA['Shot'].

    Checks for consistency between the table and shot list lengths.
    Modifies the SA dictionary in place and resets the DataFrame index.

    Args:
        SA (dict): The main data structure containing 'Shot' list and 'Data' DataFrame.

    Returns:
        dict: The modified SA dictionary. Returns the original SA if errors occur.
    """
    print("Attempting to delete selected shots...")

    # --- Input Validation ---
    if SA is None or 'Data' not in SA or SA['Data'] is None or 'Shot' not in SA:
        print("Error: SA structure is invalid or empty. Cannot delete shots.")
        return SA
    if not isinstance(SA['Data'], pd.DataFrame):
         print("Error: SA['Data'] is not a Pandas DataFrame.")
         return SA
    if 'Selected' not in SA['Data'].columns:
        print("Error: 'Selected' column not found in SA['Data']. Cannot determine which shots to delete.")
        return SA
    # Critical Check: Ensure table rows and shot list items correspond before deletion
    if len(SA['Data']) != len(SA['Shot']):
        print(f"CRITICAL Error: Mismatch between number of rows in Table ({len(SA['Data'])}) "
              f"and entries in Shot ({len(SA['Shot'])}). Deletion aborted for data integrity.")
        return SA
    # --- End Validation ---

    initial_shot_count = len(SA['Data'])
    if initial_shot_count == 0:
         print("Table is empty. No shots to delete.")
         return SA

    # Find the original indices of the rows/shots to KEEP
    # We use the DataFrame's index directly. Assumes it aligns 1:1 with SA['Shot'] list.
    indices_to_keep = SA['Data'].index[SA['Data']['Selected'] == False].tolist()

    shots_to_delete = initial_shot_count - len(indices_to_keep)

    if shots_to_delete == 0:
        print("No shots marked for deletion ('Selected' is False for all).")
        return SA
    elif shots_to_delete == initial_shot_count:
         print("All shots are marked for deletion. Clearing SA['Data'] and SA['Shot'].")
         # Create an empty DataFrame with the same columns
         SA['Data'] = pd.DataFrame(columns=SA['Data'].columns)
         SA['Shot'] = []
         return SA
    else:
        print(f"Deleting {shots_to_delete} selected shots...")

        # --- Perform Deletion ---
        # 1. Create the new list of shot data using the original indices to keep
        # This relies on the initial length check ensuring indices are valid for SA['Shot']
        try:
            # List comprehension is efficient for building the new list
            new_shot_list = [SA['Shot'][i] for i in indices_to_keep]
        except IndexError:
             print(f"CRITICAL Error: Index mismatch during SA['Shot'] reconstruction. "
                   f"Max index to keep: {max(indices_to_keep) if indices_to_keep else 'N/A'}, "
                   f"Shot list length: {len(SA['Shot'])}. Aborting deletion.")
             # This error suggests the initial length check might have been insufficient
             # or something modified SA['Shot'] unexpectedly.
             return SA

        # 2. Filter the DataFrame to keep only the corresponding rows
        # Use .loc with the list of indices to keep. This preserves the selection.
        SA['Data'] = SA['Data'].loc[indices_to_keep]

        # 3. Replace the old shot list with the new one
        SA['Shot'] = new_shot_list

        # 4. Reset the DataFrame index to be sequential (0, 1, 2, ...)
        # drop=True prevents the old index from being added as a new column.
        SA['Data'] = SA['Data'].reset_index(drop=True)

        # --- Verification ---
        if len(SA['Data']) == len(SA['Shot']) == len(indices_to_keep):
             print(f"Deletion successful. Remaining shots: {len(SA['Data'])}.")
        else:
            # This indicates a major internal inconsistency
             print(f"CRITICAL Error after deletion: Table length ({len(SA['Data'])}) "
                   f"and Shot list length ({len(SA['Shot'])}) do not match the expected count ({len(indices_to_keep)}).")
             # Depending on requirements, might want to revert changes or flag heavily.

        return SA
