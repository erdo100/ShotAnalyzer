import json
import pandas as pd

def read_gamefile(filepath):
    # Read JSON file
    with open(filepath, 'r') as f:
        json_data = json.load(f)

    # Table dimensions (scaling factors)
    tableX = 2840
    tableY = 1420
    filename = filepath.split('/')[-1].split('.')[0]

    # Initialize lists to store shot data
    shot_data = []

    for game_set in json_data['Match']['Sets']:
        for entry in game_set['Entries']:
            if entry['PathTrackingId'] > 0:
                routes = []
                
                # Extract all 3 routes (before, during, after hit)
                for bi in range(3):
                    coords = entry['PathTracking']['DataSets'][bi]['Coords']
                    route_df = pd.DataFrame({
                        't': [c['DeltaT_500us'] * 0.0005 for c in coords],
                        'x': [c['X'] * tableX for c in coords],
                        'y': [c['Y'] * tableY for c in coords]
                    })
                    routes.append(route_df)
                
                # Append shot data to list
                shot_data.append({
                    'ShotID': entry['PathTrackingId'],
                    'Filename': filename,
                    'Route0': routes[0],  # Before hit
                    'Route1': routes[1],  # During hit
                    'Route2': routes[2],  # After hit
                })

    # Convert to DataFrame
    SA = pd.DataFrame(shot_data)
    return SA
