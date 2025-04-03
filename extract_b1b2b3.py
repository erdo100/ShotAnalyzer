import numpy as np
from angle_vector import angle_vector

def extract_b1b2b3(shot):
    b1b2b3num = []
    err = {'code': None, 'text': ''}
    col = 'WYR'  # Default ordering (W=0, Y=1, R=2)

    # Get all three routes first for cleaner access
    route0 = shot['Route0']
    route1 = shot['Route1']
    route2 = shot['Route2']

    # Check if routes have enough points
    if len(route0) < 2 or len(route1) < 2 or len(route2) < 2:
        return 'WYR', {'code': 2, 'text': f"Empty Data ({__name__})"}

    # Get sorted routes by their second timestamp
    sorted_routes = sorted(
        [(route0['t'].iloc[1], 0),
         (route1['t'].iloc[1], 1),
         (route2['t'].iloc[1], 2)],
        key=lambda x: x[0]
    )
    b1it, b2it, b3it = [idx for (t, idx) in sorted_routes]

    # Get the actual route DataFrames in order
    b1_route = shot[f'Route{b1it}']
    b2_route = shot[f'Route{b2it}']
    b3_route = shot[f'Route{b3it}']

    # Early return if not enough points in b1
    if len(b1_route) < 3:
        return col, err

    # Find movement indices
    b1_t1 = b1_route['t'].iloc[1]
    b1_t2 = b1_route['t'].iloc[2]
    
    def get_last_movement(route, threshold):
        mov_idx = np.where(route['t'].iloc[1:] <= threshold)[0]
        return mov_idx[-1] if len(mov_idx) > 0 else None

    tb2i2 = get_last_movement(b2_route, b1_t1)
    tb3i2 = get_last_movement(b3_route, b1_t1)
    tb2i3 = get_last_movement(b2_route, b1_t2)
    tb3i3 = get_last_movement(b3_route, b1_t2)

    # Case 1: Only b1 has moved
    if not any([tb2i2, tb3i2, tb2i3, tb3i3]):
        return ''.join(col[i] for i in [b1it, b2it, b3it]), err

    # Case 2: b1 and b2 have moved
    if any([tb2i2, tb2i3]) and not any([tb3i2, tb3i3]):
        vec_b1b2 = [
            b2_route['x'].iloc[0] - b1_route['x'].iloc[0],
            b2_route['y'].iloc[0] - b1_route['y'].iloc[0]
        ]
        vec_b1dir = [
            b1_route['x'].iloc[1] - b1_route['x'].iloc[0],
            b1_route['y'].iloc[1] - b1_route['y'].iloc[0]
        ]
        vec_b2dir = [
            b2_route['x'].iloc[1] - b2_route['x'].iloc[0],
            b2_route['y'].iloc[1] - b2_route['y'].iloc[0]
        ]

        if angle_vector(vec_b1b2, vec_b2dir) > 90:
            return 'WYR'[b1it] + 'WYR'[b2it] + 'WYR'[b3it], err

    # Case 3: b1 and b3 have moved
    if not any([tb2i2, tb2i3]) and any([tb3i2, tb3i3]):
        vec_b1b3 = [
            b3_route['x'].iloc[0] - b1_route['x'].iloc[0],
            b3_route['y'].iloc[0] - b1_route['y'].iloc[0]
        ]
        vec_b1dir = [
            b1_route['x'].iloc[1] - b1_route['x'].iloc[0],
            b1_route['y'].iloc[1] - b1_route['y'].iloc[0]
        ]
        vec_b3dir = [
            b3_route['x'].iloc[1] - b3_route['x'].iloc[0],
            b3_route['y'].iloc[1] - b3_route['y'].iloc[0]
        ]

        if angle_vector(vec_b1b3, vec_b3dir) > 90:
            return 'WYR'[b1it] + 'WYR'[b3it] + 'WYR'[b2it], err

    # Default case
    b1b2b3 = ''.join(col[i] for i in [b1it, b2it, b3it])

    # Error case if all balls moved
    if any([tb2i2, tb2i3]) and any([tb3i2, tb3i3]):
        err.update({'code': 2, 'text': f"all balls moved at same time ({__name__})"})

    return b1b2b3, err