import numpy as np
from ball_velocity import ball_velocity
from ball_direction import ball_direction
from CushionAngle import cushion_angle

def eval_hit_events(hit, si, b1b2b3, SA, param):
    """Evaluate hit events for a shot and calculate related metrics."""
    # Initialize output variables for each ball
    for bi in range(3):
        # Initialize list-type metrics
        for metric in ['AimOffsetLL', 'AimOffsetSS', 'B1B2OffsetLL', 
                      'B1B2OffsetSS', 'B1B3OffsetLL', 'B1B3OffsetSS']:
            hit[bi][metric] = []
        
        # Initialize array-type metrics
        for metric in ['Type', 'FromC', 'ToC', 'V1', 'V2', 'Offset', 
                      'Fraction', 'DefAngle', 'CutAngle', 'CInAngle', 
                      'COutAngle', 'FromCPos', 'ToCPos', 'FromDPos', 'ToDPos']:
            hit[bi][metric] = np.full(8, np.nan)

    # Process hits for each ball in the b1b2b3 sequence
    for bi in b1b2b3:
        route = SA.iloc[si][f'Route{bi}']  # Get route data from DataFrame
        
        for hi, with_event in enumerate(hit[bi]['with']):
            if with_event == '-':
                continue  # Skip non-hit events

            # Calculate ball velocity and related metrics
            v, v1, v2, alpha, offset = ball_velocity(route, hit[bi], hi)
            hit[bi]['V1'][hi] = v[0]
            hit[bi]['V2'][hi] = v[1]
            hit[bi]['Offset'][hi] = offset

            # Calculate ball direction if there was movement
            if v[0] > 0 or v[1] > 0:
                direction = ball_direction(route, hit[bi], hi, param)
            else:
                direction = np.full((2, 6), np.nan)

            # Store direction results
            hit[bi]['FromC'][hi] = direction[1, 0]
            hit[bi]['FromCPos'][hi] = direction[1, 1]
            hit[bi]['FromDPos'][hi] = direction[1, 2]
            hit[bi]['ToC'][hi] = direction[1, 3]
            hit[bi]['ToCPos'][hi] = direction[1, 4]
            hit[bi]['ToDPos'][hi] = direction[1, 5]

            # Determine hit type (0=unknown, 1=ball-ball, 2=cushion)
            hit_type = 0
            if with_event in '1234':  # Cushion hit
                hit_type = 2
                angle = cushion_angle(int(with_event), v1, v2)
                hit[bi]['CInAngle'][hi] = angle[0]
                hit[bi]['COutAngle'][hi] = angle[1]
            elif with_event in 'WYR':  # Ball-ball hit
                hit_type = 1

            hit[bi]['Type'][hi] = hit_type

            # Process ball-ball collision metrics
            if hit_type == 1 and v[0] > 0:
                process_ball_collision(hit, bi, hi, b1b2b3, SA, si, param)

    return hit

def process_ball_collision(hit, bi, hi, b1b2b3, SA, si, param):
    """Calculate detailed metrics for ball-ball collisions."""
    # Find matching time indices in the routes
    try:
        tb10i = np.where(SA.iloc[si][f'Route{bi}']['t'] == hit[bi]['t'][hi])[0][0]
        tb11i = np.where(SA.iloc[si][f'Route{b1b2b3[1]}']['t'] == hit[bi]['t'][hi])[0][0]
    except IndexError:
        print('No time point found matching to hit event')
        return

    # Get ball positions at collision
    pb1 = np.array([
        SA.iloc[si][f'Route{bi}']['x'].iloc[tb10i],
        SA.iloc[si][f'Route{bi}']['y'].iloc[tb10i]
    ])
    pb2 = np.array([
        SA.iloc[si][f'Route{b1b2b3[1]}']['x'].iloc[tb11i],
        SA.iloc[si][f'Route{b1b2b3[1]}']['y'].iloc[tb11i]
    ])

    # Get velocity of second ball
    hib2 = np.where(hit[b1b2b3[1]]['t'] == hit[bi]['t'][hi])[0]
    if hib2.size > 0:
        _, v1b2, _, _, _ = ball_velocity(SA.iloc[si][f'Route{b1b2b3[1]}'], hit[b1b2b3[1]], hib2[0])
    else:
        v1b2 = np.zeros(2)

    # Calculate relative velocity and collision metrics
    velrel = np.append(hit[bi]['V1'][hi] - v1b2, 0)
    pb_diff = np.append(pb2 - pb1, 0)

    try:
        hit[bi]['Fraction'][hi] = 1 - (np.linalg.norm(np.cross(pb_diff, velrel)) / 
                                      np.linalg.norm(velrel) / param['ballR'] / 2)
        hit[bi]['DefAngle'][hi] = (np.arccos(np.dot(hit[bi]['V1'][hi], hit[bi]['V2'][hi]) / 
                                  (np.linalg.norm(hit[bi]['V1'][hi]) * np.linalg.norm(hit[bi]['V2'][hi])))) * 180 / np.pi
    except (ValueError, ZeroDivisionError):
        hit[bi]['Fraction'][hi] = 0
        hit[bi]['DefAngle'][hi] = 0