import json
import pandas as pd
import os
import copy

def read_gamefile(filepath):
    # Read the file in json format
    with open(filepath, 'r') as f:
        json_data = json.load(f)
    
    # Scaling factors
    tableX = 2840
    tableY = 1420
    
    # extract filename
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    # Initialize DataFrame for shot data
    shot_data = []
    route_data = []
    
    for seti, set_data in enumerate(json_data['Match']['Sets'], 1):
        # Handle cases where Entries might be a cell array (represented as list in JSON)
        entries = set_data['Entries']
        if isinstance(entries, list) and entries and isinstance(entries[0], dict):
            # Process each entry in the set
            for entry in entries:
                if entry['PathTrackingId'] > 0:
                    # Determine player name
                    playernum = entry['Scoring']['Player']
                    player = json_data['Player1'] if playernum == 1 else json_data['Player2']
                    
                    # Create shot record
                    shot_record = {
                        'Selected': False,
                        'Filename': filename,
                        'GameType': json_data['Match']['GameType'],
                        'Player1': json_data['Player1'],
                        'Player2': json_data['Player2'],
                        'Player': player,
                        'Set': seti,
                        'ShotID': entry['PathTrackingId'],
                        'CurrentInning': entry['Scoring']['CurrentInning'],
                        'CurrentSeries': entry['Scoring']['CurrentSeries'],
                        'CurrentTotalPoints': entry['Scoring']['CurrentTotalPoints'],
                        'Point': entry['Scoring']['EntryType'],
                        'ErrorID': -1,
                        'ErrorText': 'only read in',
                        'Interpreted': 0,
                        'Mirrored': 0
                    }
                    
                    # Process route data for each ball (assuming 3 balls)
                    for bi in range(3):
                        coords = entry['PathTracking']['DataSets'][bi]['Coords']
                        route_df = pd.DataFrame({
                            'ShotID': entry['PathTrackingId'],
                            'BallID': bi,
                            't': [coord['DeltaT_500us'] * 0.0005 for coord in coords],
                            'x': [coord['X'] * tableX for coord in coords],
                            'y': [coord['Y'] * tableY for coord in coords]
                        })
                        route_data.append(route_df)
                    
                    shot_data.append(shot_record)
    
    if not shot_data:
        return None
    
    # Create DataFrames
    # Build the nested dictionary structure for Route
    route_dict = {}
    for route_df in route_data:
        shot_id = route_df['ShotID'].iloc[0]
        ball_id = route_df['BallID'].iloc[0]
        if shot_id not in route_dict:
            route_dict[shot_id] = {}
        route_dict[shot_id][ball_id] = route_df  # Store DataFrame per BallID

    # Create SA structure with identical Route and Route0
    SA = {
        'Table': pd.DataFrame(shot_data),
        'Route': route_dict,
        'Route0': copy.deepcopy(route_dict)  # Identical nested structure
    }
    
    return SA