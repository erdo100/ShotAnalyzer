import json
import os
import pandas as pd
import numpy as np
import copy

def read_gamefile(filepath):
    """
    Reads game data from a JSON file, extracts relevant information,
    and structures it for analysis.

    Args:
        filepath (str): Path to the JSON game file.

    Returns:
        dict: A dictionary (SA) containing 'Shot' data (list of dicts)
              and 'Table' data (pandas DataFrame), or None if no valid
              shot data is found.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

    # Scaling factors
    tableX = 2840.0
    tableY = 1420.0

    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(filepath))[0]

    SA = {'Shot': [], 'Table': None}
    check = 0 # Flag to check if any valid shot data was processed

    # Lists to build the DataFrame columns
    Selected_list = []
    Filename_list = []
    GameType_list = []
    Player1_list = []
    Player2_list = []
    Set_list = []
    ShotID_list = []
    CurrentInning_list = []
    CurrentSeries_list = []
    CurrentTotalPoints_list = []
    Point_list = []
    Player_list = []
    ErrorID_list = []
    ErrorText_list = []
    Interpreted_list = []
    Mirrored_list = []

    num_shots_processed = 0

    if 'Match' not in data or 'Sets' not in data['Match']:
        print(f"Warning: 'Match' or 'Sets' not found in the JSON structure for {filepath}")
        return None

    sets_data = data['Match']['Sets']
    if not isinstance(sets_data, list):
        print(f"Warning: 'Sets' is not a list in {filepath}")
        return None

    for seti, current_set in enumerate(sets_data):
        # Handle potential inconsistencies where Entries might be a list of lists/dicts
        # The MATLAB code seems to handle a case where Entries is a cell array containing a single struct.
        # In JSON, this might appear as a list containing a single object.
        # This Python code assumes 'Entries' is expected to be a list of shot entry dictionaries.
        if isinstance(current_set.get('Entries'), list) and len(current_set['Entries']) > 0:
            # The MATLAB code had a peculiar block:
            # if iscell(json.Match.Sets(seti).Entries) ... end
            # This block copied *only the first element* of the cell array into a new structure array.
            # Translating this literally: if the first element is itself a list/dict containing 'PathTrackingId',
            # maybe it implies the structure is nested differently than expected?
            # Let's assume the standard structure is a list of dictionaries directly under 'Entries'.
            # If you encounter files where this MATLAB block was necessary, the JSON structure needs closer examination.
            pass # Assuming standard structure, no special handling needed like the MATLAB cell check

        entries = current_set.get('Entries')
        if not isinstance(entries, list):
            # print(f"Warning: 'Entries' is not a list or is missing in set {seti+1} for {filepath}")
            continue # Skip this set if entries are not as expected

        for entryi, current_entry in enumerate(entries):
            if not isinstance(current_entry, dict):
                # print(f"Warning: Entry {entryi+1} in set {seti+1} is not a dictionary for {filepath}")
                continue # Skip malformed entry

            path_tracking_id = current_entry.get('PathTrackingId')
            # Ensure PathTrackingId is a valid number and > 0
            if isinstance(path_tracking_id, (int, float)) and path_tracking_id > 0:

                # Check for necessary nested structures
                if 'Scoring' not in current_entry or 'PathTracking' not in current_entry or \
                   'DataSets' not in current_entry['PathTracking'] or \
                   len(current_entry['PathTracking']['DataSets']) != 3:
                    # print(f"Warning: Skipping entry {entryi+1} in set {seti+1} due to missing data structures.")
                    continue

                num_shots_processed += 1
                si = num_shots_processed # Use 1-based index for conceptual clarity if needed, but list index is si-1

                # Append data for the table
                Selected_list.append(False)
                Filename_list.append(filename)
                GameType_list.append(data['Match'].get('GameType', 'Unknown'))
                player1_name = data.get('Player1', 'Player1')
                player2_name = data.get('Player2', 'Player2')
                Player1_list.append(player1_name)
                Player2_list.append(player2_name)
                Set_list.append(seti + 1) # Store 1-based set index
                ShotID_list.append(path_tracking_id)

                scoring_data = current_entry['Scoring']
                CurrentInning_list.append(scoring_data.get('CurrentInning'))
                CurrentSeries_list.append(scoring_data.get('CurrentSeries'))
                CurrentTotalPoints_list.append(scoring_data.get('CurrentTotalPoints'))
                Point_list.append(scoring_data.get('EntryType')) # 'Point' in MATLAB was EntryType

                playernum = scoring_data.get('Player')
                if playernum == 1:
                    Player_list.append(player1_name)
                elif playernum == 2:
                    Player_list.append(player2_name)
                else:
                    Player_list.append(f"UnknownPlayer_{playernum}")

                ErrorID_list.append(-1) # Initial state
                ErrorText_list.append('only read in')
                Interpreted_list.append(0)
                Mirrored_list.append(0) # Assuming 0 initially, MATLAB code had this

                shot_data = {'Route': [{}, {}, {}], 'Route0': [{}, {}, {}], 'hit': 0}
                valid_route_data = True
                for bi in range(3): # For each ball (0, 1, 2)
                    coords = current_entry['PathTracking']['DataSets'][bi].get('Coords')
                    if not isinstance(coords, list):
                        # print(f"Warning: Coords missing or not a list for ball {bi+1}, shot {si}")
                        valid_route_data = False
                        break # Stop processing this shot if data is bad

                    clen = len(coords)
                    t_list, x_list, y_list = [], [], []

                    for ci in range(clen):
                        coord_point = coords[ci]
                        if not isinstance(coord_point, dict) or \
                           'DeltaT_500us' not in coord_point or \
                           'X' not in coord_point or 'Y' not in coord_point:
                            # print(f"Warning: Invalid coord point {ci+1} for ball {bi+1}, shot {si}")
                            valid_route_data = False
                            break # Stop processing this ball

                        t_list.append(coord_point['DeltaT_500us'] * 0.0005)
                        x_list.append(coord_point['X'] * tableX)
                        y_list.append(coord_point['Y'] * tableY)

                    if not valid_route_data: break # Stop processing this shot

                    # Store route data as numpy arrays
                    current_route = {
                        't': np.array(t_list, dtype=float),
                        'x': np.array(x_list, dtype=float),
                        'y': np.array(y_list, dtype=float)
                    }
                    shot_data['Route'][bi] = current_route
                    # Store initial state before modifications in quality check
                    shot_data['Route0'][bi] = copy.deepcopy(current_route)

                if valid_route_data:
                    SA['Shot'].append(shot_data)
                    check = 1 # Mark that we have processed at least one valid shot
                else:
                    # If route data was invalid, we need to remove the corresponding entries
                    # from the table lists that were added before the inner loop.
                    Selected_list.pop()
                    Filename_list.pop()
                    GameType_list.pop()
                    Player1_list.pop()
                    Player2_list.pop()
                    Set_list.pop()
                    ShotID_list.pop()
                    CurrentInning_list.pop()
                    CurrentSeries_list.pop()
                    CurrentTotalPoints_list.pop()
                    Point_list.pop()
                    Player_list.pop()
                    ErrorID_list.pop()
                    ErrorText_list.pop()
                    Interpreted_list.pop()
                    Mirrored_list.pop()
                    num_shots_processed -= 1 # Decrement counter as this shot is invalid


    if check:
        # Create the Pandas DataFrame
        df_data = {
            'Selected': Selected_list,
            'ShotID': ShotID_list,
            'Mirrored': Mirrored_list, # From MATLAB code
            'Filename': Filename_list,
            'GameType': GameType_list,
            'Interpreted': Interpreted_list, # From MATLAB code
            'Player': Player_list,
            'ErrorID': ErrorID_list, # From MATLAB code
            'ErrorText': ErrorText_list, # From MATLAB code
            'Set': Set_list,
            'CurrentInning': CurrentInning_list,
            'CurrentSeries': CurrentSeries_list,
            'CurrentTotalPoints': CurrentTotalPoints_list,
            'Point': Point_list # Mapped from EntryType
            # Adding Player1 and Player2 might be useful for context
            #'Player1': Player1_list,
            #'Player2': Player2_list
        }
        SA['Table'] = pd.DataFrame(df_data)
        # Ensure ShotID is integer if possible
        SA['Table']['ShotID'] = pd.to_numeric(SA['Table']['ShotID'], errors='coerce').fillna(0).astype(int)
        print(f"Successfully read {len(SA['Table'])} shots from {filepath}")
        return SA
    else:
        print(f"No valid shot data found or processed in {filepath}")
        return None

def read_gamefile(filepath):
    """
    Reads game data from a JSON file, extracts relevant information,
    and structures it for analysis.

    Args:
        filepath (str): Path to the JSON game file.

    Returns:
        dict: A dictionary (SA) containing 'Shot' data (list of dicts)
              and 'Table' data (pandas DataFrame), or None if no valid
              shot data is found.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

    # Scaling factors
    tableX = 2840.0
    tableY = 1420.0

    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(filepath))[0]

    SA = {'Shot': [], 'Table': None}
    check = 0 # Flag to check if any valid shot data was processed

    # Lists to build the DataFrame columns
    Selected_list = []
    Filename_list = []
    GameType_list = []
    Player1_list = []
    Player2_list = []
    Set_list = []
    ShotID_list = []
    CurrentInning_list = []
    CurrentSeries_list = []
    CurrentTotalPoints_list = []
    Point_list = []
    Player_list = []
    ErrorID_list = []
    ErrorText_list = []
    Interpreted_list = []
    Mirrored_list = []

    num_shots_processed = 0

    if 'Match' not in data or 'Sets' not in data['Match']:
        print(f"Warning: 'Match' or 'Sets' not found in the JSON structure for {filepath}")
        return None

    sets_data = data['Match']['Sets']
    if not isinstance(sets_data, list):
        print(f"Warning: 'Sets' is not a list in {filepath}")
        return None

    for seti, current_set in enumerate(sets_data):
        # Handle potential inconsistencies where Entries might be a list of lists/dicts
        # The MATLAB code seems to handle a case where Entries is a cell array containing a single struct.
        # In JSON, this might appear as a list containing a single object.
        # This Python code assumes 'Entries' is expected to be a list of shot entry dictionaries.
        if isinstance(current_set.get('Entries'), list) and len(current_set['Entries']) > 0:
            # The MATLAB code had a peculiar block:
            # if iscell(json.Match.Sets(seti).Entries) ... end
            # This block copied *only the first element* of the cell array into a new structure array.
            # Translating this literally: if the first element is itself a list/dict containing 'PathTrackingId',
            # maybe it implies the structure is nested differently than expected?
            # Let's assume the standard structure is a list of dictionaries directly under 'Entries'.
            # If you encounter files where this MATLAB block was necessary, the JSON structure needs closer examination.
            pass # Assuming standard structure, no special handling needed like the MATLAB cell check

        entries = current_set.get('Entries')
        if not isinstance(entries, list):
            # print(f"Warning: 'Entries' is not a list or is missing in set {seti+1} for {filepath}")
            continue # Skip this set if entries are not as expected

        for entryi, current_entry in enumerate(entries):
            if not isinstance(current_entry, dict):
                # print(f"Warning: Entry {entryi+1} in set {seti+1} is not a dictionary for {filepath}")
                continue # Skip malformed entry

            path_tracking_id = current_entry.get('PathTrackingId')
            # Ensure PathTrackingId is a valid number and > 0
            if isinstance(path_tracking_id, (int, float)) and path_tracking_id > 0:

                # Check for necessary nested structures
                if 'Scoring' not in current_entry or 'PathTracking' not in current_entry or \
                   'DataSets' not in current_entry['PathTracking'] or \
                   len(current_entry['PathTracking']['DataSets']) != 3:
                    # print(f"Warning: Skipping entry {entryi+1} in set {seti+1} due to missing data structures.")
                    continue

                num_shots_processed += 1
                si = num_shots_processed # Use 1-based index for conceptual clarity if needed, but list index is si-1

                # Append data for the table
                Selected_list.append(False)
                Filename_list.append(filename)
                GameType_list.append(data['Match'].get('GameType', 'Unknown'))
                player1_name = data.get('Player1', 'Player1')
                player2_name = data.get('Player2', 'Player2')
                Player1_list.append(player1_name)
                Player2_list.append(player2_name)
                Set_list.append(seti + 1) # Store 1-based set index
                ShotID_list.append(path_tracking_id)

                scoring_data = current_entry['Scoring']
                CurrentInning_list.append(scoring_data.get('CurrentInning'))
                CurrentSeries_list.append(scoring_data.get('CurrentSeries'))
                CurrentTotalPoints_list.append(scoring_data.get('CurrentTotalPoints'))
                Point_list.append(scoring_data.get('EntryType')) # 'Point' in MATLAB was EntryType

                playernum = scoring_data.get('Player')
                if playernum == 1:
                    Player_list.append(player1_name)
                elif playernum == 2:
                    Player_list.append(player2_name)
                else:
                    Player_list.append(f"UnknownPlayer_{playernum}")

                ErrorID_list.append(-1) # Initial state
                ErrorText_list.append('only read in')
                Interpreted_list.append(0)
                Mirrored_list.append(0) # Assuming 0 initially, MATLAB code had this

                shot_data = {'Route': [{}, {}, {}], 'Route0': [{}, {}, {}], 'hit': 0}
                valid_route_data = True
                for bi in range(3): # For each ball (0, 1, 2)
                    coords = current_entry['PathTracking']['DataSets'][bi].get('Coords')
                    if not isinstance(coords, list):
                        # print(f"Warning: Coords missing or not a list for ball {bi+1}, shot {si}")
                        valid_route_data = False
                        break # Stop processing this shot if data is bad

                    clen = len(coords)
                    t_list, x_list, y_list = [], [], []

                    for ci in range(clen):
                        coord_point = coords[ci]
                        if not isinstance(coord_point, dict) or \
                           'DeltaT_500us' not in coord_point or \
                           'X' not in coord_point or 'Y' not in coord_point:
                            # print(f"Warning: Invalid coord point {ci+1} for ball {bi+1}, shot {si}")
                            valid_route_data = False
                            break # Stop processing this ball

                        t_list.append(coord_point['DeltaT_500us'] * 0.0005)
                        x_list.append(coord_point['X'] * tableX)
                        y_list.append(coord_point['Y'] * tableY)

                    if not valid_route_data: break # Stop processing this shot

                    # Store route data as numpy arrays
                    current_route = {
                        't': np.array(t_list, dtype=float),
                        'x': np.array(x_list, dtype=float),
                        'y': np.array(y_list, dtype=float)
                    }
                    shot_data['Route'][bi] = current_route
                    # Store initial state before modifications in quality check
                    shot_data['Route0'][bi] = copy.deepcopy(current_route)

                if valid_route_data:
                    SA['Shot'].append(shot_data)
                    check = 1 # Mark that we have processed at least one valid shot
                else:
                    # If route data was invalid, we need to remove the corresponding entries
                    # from the table lists that were added before the inner loop.
                    Selected_list.pop()
                    Filename_list.pop()
                    GameType_list.pop()
                    Player1_list.pop()
                    Player2_list.pop()
                    Set_list.pop()
                    ShotID_list.pop()
                    CurrentInning_list.pop()
                    CurrentSeries_list.pop()
                    CurrentTotalPoints_list.pop()
                    Point_list.pop()
                    Player_list.pop()
                    ErrorID_list.pop()
                    ErrorText_list.pop()
                    Interpreted_list.pop()
                    Mirrored_list.pop()
                    num_shots_processed -= 1 # Decrement counter as this shot is invalid


    if check:
        # Create the Pandas DataFrame
        df_data = {
            'Selected': Selected_list,
            'ShotID': ShotID_list,
            'Mirrored': Mirrored_list, # From MATLAB code
            'Filename': Filename_list,
            'GameType': GameType_list,
            'Interpreted': Interpreted_list, # From MATLAB code
            'Player': Player_list,
            'ErrorID': ErrorID_list, # From MATLAB code
            'ErrorText': ErrorText_list, # From MATLAB code
            'Set': Set_list,
            'CurrentInning': CurrentInning_list,
            'CurrentSeries': CurrentSeries_list,
            'CurrentTotalPoints': CurrentTotalPoints_list,
            'Point': Point_list # Mapped from EntryType
            # Adding Player1 and Player2 might be useful for context
            #'Player1': Player1_list,
            #'Player2': Player2_list
        }
        SA['Table'] = pd.DataFrame(df_data)
        # Ensure ShotID is integer if possible
        SA['Table']['ShotID'] = pd.to_numeric(SA['Table']['ShotID'], errors='coerce').fillna(0).astype(int)
        print(f"Successfully read {len(SA['Table'])} shots from {filepath}")
        return SA
    else:
        print(f"No valid shot data found or processed in {filepath}")
        return None

