import pandas as pd
import numpy as np
from datetime import datetime

def read_bbtfile(filepath):
    # Read the file in CSV format
    data = pd.read_csv(filepath, header=None).to_numpy()

    # Extract filename
    filename = filepath.split('/')[-1].split('.')[0]

    SA = {'Shot': []}
    si = 0

    shot = {
        'Route': [],
        'Route0': [],
        'hit': 0
    }

    for bi in range(3):
        route = {
            't': data[:, 0] / 1000,
            'x': data[:, (bi * 2) + 1],
            'y': data[:, (bi * 2) + 2]
        }

        # Calculate movement distance
        dL = np.sqrt((route['x'] - route['x'][0])**2 + (route['y'] - route['y'][0])**2)
        ind = np.where(dL > 5)[0]

        if ind.size == 0:
            route['t'] = np.delete(route['t'], slice(1, -1))
            route['x'] = np.delete(route['x'], slice(1, -1))
            route['y'] = np.delete(route['y'], slice(1, -1))
        else:
            route['t'] = np.delete(route['t'], slice(1, ind[0] + 1))
            route['x'] = np.delete(route['x'], slice(1, ind[0] + 1))
            route['y'] = np.delete(route['y'], slice(1, ind[0] + 1))

        shot['Route'].append(route)
        shot['Route0'].append(route)

    SA['Shot'].append(shot)

    SA['Table'] = pd.DataFrame({
        'Selected': [False],
        'ShotID': [int(datetime.now().strftime('%y%m%d%H%M%S%f'))],
        'Mirrored': [0],
        'Filename': [filename],
        'GameType': ['3'],
        'Interpreted': [0],
        'Player': [0],
        'ErrorID': [-1],
        'ErrorText': ['only read in']
    })

    return SA