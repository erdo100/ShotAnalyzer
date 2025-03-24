import json
import pandas as pd

def read_gamefile(filepath):
    # Read the file in JSON format
    with open(filepath, 'r') as f:
        json_data = json.load(f)

    # Scaling factors
    tableX = 2840
    tableY = 1420

    # Extract filename
    filename = filepath.split('/')[-1].split('.')[0]

    SA = {'Shot': []}
    si = 0

    for seti, game_set in enumerate(json_data['Match']['Sets']):
        for entry in game_set['Entries']:
            if entry['PathTrackingId'] > 0:
                si += 1
                shot = {
                    'Route': [],
                    'Route0': [],
                    'hit': 0
                }

                for bi in range(3):
                    coords = entry['PathTracking']['DataSets'][bi]['Coords']
                    route = {
                        't': [c['DeltaT_500us'] * 0.0005 for c in coords],
                        'x': [c['X'] * tableX for c in coords],
                        'y': [c['Y'] * tableY for c in coords]
                    }
                    shot['Route'].append(route)
                    shot['Route0'].append(route)

                SA['Shot'].append(shot)

    SA['Table'] = pd.DataFrame({
        'Selected': [False] * si,
        'ShotID': [entry['PathTrackingId'] for game_set in json_data['Match']['Sets'] for entry in game_set['Entries'] if entry['PathTrackingId'] > 0],
        'Filename': [filename] * si,
        'GameType': [json_data['Match']['GameType']] * si,
        'Player': [entry['Scoring']['Player'] for game_set in json_data['Match']['Sets'] for entry in game_set['Entries'] if entry['PathTrackingId'] > 0],
        'ErrorID': [-1] * si,
        'ErrorText': ['only read in'] * si,
        'Interpreted': [0] * si,
        'Mirrored': [0] * si
    })

    return SA