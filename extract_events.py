import numpy as np
from str2num_b1b2b3 import str2num_b1b2b3
from angle_vector import angle_vector

def extract_events(si, SA):
    err = {'code': None, 'text': ''}

    b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(SA['Table']['B1B2B3'][si])

    ball0 = [
        {
            't': SA['Shot'][si]['Route0'][bi]['t'],
            'x': SA['Shot'][si]['Route0'][bi]['x'],
            'y': SA['Shot'][si]['Route0'][bi]['y']
        }
        for bi in range(3)
    ]

    for bi in range(3):
        SA['Shot'][si]['Route'][bi] = {
            't': ball0[bi]['t'],
            'x': ball0[bi]['x'],
            'y': ball0[bi]['y']
        }

    hit = {
        b1i: {'with': 'S', 't': [0], 'XPos': [ball0[b1i]['x'][0]], 'YPos': [ball0[b1i]['y'][0]], 'Kiss': 0, 'Point': 0, 'Fuchs': 0, 'PointDist': 3000, 'KissDistB1': 3000, 'Tpoint': 3000, 'Tkiss': 3000, 'Tready': 3000, 'TB2hit': 3000, 'Tfailure': 3000},
        b2i: {'with': '-', 't': [0], 'XPos': [ball0[b2i]['x'][0]], 'YPos': [ball0[b2i]['y'][0]]},
        b3i: {'with': '-', 't': [0], 'XPos': [ball0[b3i]['x'][0]], 'YPos': [ball0[b3i]['y'][0]]}
    }

    for bi in range(3):
        ball0[bi]['dt'] = np.diff(ball0[bi]['t'], append=ball0[bi]['t'][-1])
        ball0[bi]['vx'] = np.diff(ball0[bi]['x'], append=0) / ball0[bi]['dt']
        ball0[bi]['vy'] = np.diff(ball0[bi]['y'], append=0) / ball0[bi]['dt']
        ball0[bi]['v'] = np.sqrt(ball0[bi]['vx']**2 + ball0[bi]['vy']**2)

    ball0[b2i]['vx'][0] = ball0[b2i]['vy'][0] = ball0[b2i]['v'][0] = 0
    ball0[b3i]['vx'][0] = ball0[b3i]['vy'][0] = ball0[b3i]['v'][0] = 0

    Tall0 = np.unique(np.concatenate([ball0[0]['t'], ball0[1]['t'], ball0[2]['t']]))

    do_scan = True

    # Initialize hit list and other variables
    hitlist = []
    lasthit_ball = None
    lasthit_t = None

    while do_scan:
        dT = np.diff(Tall0[:2])[0]
        tappr = Tall0[0] + dT * np.linspace(0.01, 1, 101)

        # Approximate positions for the next time step
        for bi in range(3):
            if len(ball0[bi]['t']) >= 2:
                vnext = ball0[bi]['v'][0]
                if vnext > 0:
                    ball0[bi]['x'][0] += ball0[bi]['vx'][0] * dT
                    ball0[bi]['y'][0] += ball0[bi]['vy'][0] * dT

        # Evaluate ball-ball collisions
        for bbi in [(0, 1), (0, 2), (1, 2)]:
            b1, b2 = bbi
            dist = np.sqrt((ball0[b1]['x'][0] - ball0[b2]['x'][0])**2 + (ball0[b1]['y'][0] - ball0[b2]['y'][0])**2)
            if dist < 2 * SA['param']['ballR']:
                tc = tappr[np.argmin(dist)]
                hitlist.append({'time': tc, 'type': 'ball-ball', 'b1': b1, 'b2': b2})

        # Evaluate ball-cushion collisions
        for bi in range(3):
            if ball0[bi]['x'][0] < SA['param']['ballR'] or ball0[bi]['x'][0] > SA['param']['size'][0] - SA['param']['ballR']:
                hitlist.append({'time': Tall0[0], 'type': 'cushion', 'ball': bi, 'side': 'x'})
            if ball0[bi]['y'][0] < SA['param']['ballR'] or ball0[bi]['y'][0] > SA['param']['size'][1] - SA['param']['ballR']:
                hitlist.append({'time': Tall0[0], 'type': 'cushion', 'ball': bi, 'side': 'y'})

        # Process hits
        for hit_event in hitlist:
            if hit_event['type'] == 'ball-ball':
                b1, b2 = hit_event['b1'], hit_event['b2']
                # Update velocities after collision (simplified elastic collision)
                ball0[b1]['vx'][0], ball0[b2]['vx'][0] = ball0[b2]['vx'][0], ball0[b1]['vx'][0]
                ball0[b1]['vy'][0], ball0[b2]['vy'][0] = ball0[b2]['vy'][0], ball0[b1]['vy'][0]
            elif hit_event['type'] == 'cushion':
                bi = hit_event['ball']
                if hit_event['side'] == 'x':
                    ball0[bi]['vx'][0] *= -1
                elif hit_event['side'] == 'y':
                    ball0[bi]['vy'][0] *= -1

        # Update time and check if scanning should continue
        Tall0 = Tall0[1:]
        do_scan = len(Tall0) >= 3

    # Assign updated routes back to SA
    for bi in range(3):
        SA['Shot'][si]['Route'][bi] = {
            't': ball0[bi]['t'],
            'x': ball0[bi]['x'],
            'y': ball0[bi]['y']
        }

    # Calculate Ball trajectory angle change
    for bi in range(3):
        ball0[bi]['a12'] = angle_vector(ball0[bi]['v1'], ball0[bi]['v2'])

    # Calculate Ball-Ball Distance
    BB = [(0, 1), (0, 2), (1, 2)]
    d = []
    for bbi in BB:
        bx1, bx2 = bbi
        dist = np.sqrt((ball0[bx1]['xa'] - ball0[bx2]['xa'])**2 + (ball0[bx1]['ya'] - ball0[bx2]['ya'])**2) - 2 * SA['param']['ballR']
        d.append({'BB': dist})

    # Calculate Cushion distance
    for bi in range(3):
        ball0[bi]['cd'] = [
            ball0[bi]['ya'] - SA['param']['ballR'],
            SA['param']['size'][1] - SA['param']['ballR'] - ball0[bi]['xa'],
            SA['param']['size'][0] - SA['param']['ballR'] - ball0[bi]['ya'],
            ball0[bi]['xa'] - SA['param']['ballR']
        ]

    hitlist = []

    # Evaluate cushion hit
    for bi in range(3):
        for cii in range(4):
            checkdist = np.min(ball0[bi]['cd'][cii]) <= 0
            checkangle = ball0[bi]['a12'] > 1 or ball0[bi]['a12'] == -1
            velx, vely = ball0[bi]['v1']

            if checkdist and checkangle:
                if cii == 0 and vely < 0:
                    tc = np.interp(0, ball0[bi]['cd'][cii], tappr)
                    cushx = np.interp(tc, tappr, ball0[bi]['xa'])
                    cushy = SA['param']['ballR']
                elif cii == 1 and velx > 0:
                    tc = np.interp(0, ball0[bi]['cd'][cii], tappr)
                    cushx = SA['param']['size'][1] - SA['param']['ballR']
                    cushy = np.interp(tc, tappr, ball0[bi]['ya'])
                elif cii == 2 and vely > 0:
                    tc = np.interp(0, ball0[bi]['cd'][cii], tappr)
                    cushx = np.interp(tc, tappr, ball0[bi]['xa'])
                    cushy = SA['param']['size'][0] - SA['param']['ballR']
                elif cii == 3 and velx < 0:
                    tc = np.interp(0, ball0[bi]['cd'][cii], tappr)
                    cushx = SA['param']['ballR']
                    cushy = np.interp(tc, tappr, ball0[bi]['ya'])
                else:
                    continue

                hitlist.append({'time': tc, 'type': 'cushion', 'ball': bi, 'side': cii, 'x': cushx, 'y': cushy})

    # Evaluate Ball-Ball hit
    for bbi in BB:
        bx1, bx2 = bbi
        dist = d[BB.index(bbi)]['BB']

        # Check if distance transitions from positive to negative
        if np.any(dist <= 0) and np.any(dist > 0):
            tc = np.interp(0, dist, tappr)
            hitlist.append({
                'time': tc,
                'type': 'ball-ball',
                'b1': bx1,
                'b2': bx2,
                'x1': np.interp(tc, tappr, b[bx1]['xa']),
                'y1': np.interp(tc, tappr, b[bx1]['ya']),
                'x2': np.interp(tc, tappr, b[bx2]['xa']),
                'y2': np.interp(tc, tappr, b[bx2]['ya'])
            })

        # Handle cases where velocity or angle changes are significant
        checkangleb1 = b[bx1]['a12'] > 10 or b[bx1]['a12'] == -1
        checkangleb2 = b[bx2]['a12'] > 10 or b[bx2]['a12'] == -1

        checkvelb1 = abs(b[bx1]['vt2'] - b[bx1]['vt1']) > 50 if abs(b[bx1]['vt1']) > 0 else False
        checkvelb2 = abs(b[bx2]['vt2'] - b[bx2]['vt1']) > 50 if abs(b[bx2]['vt1']) > 0 else False

        if checkangleb1 and checkangleb2 and checkvelb1 and checkvelb2:
            tc = np.interp(0, dist, tappr)
            hitlist.append({
                'time': tc,
                'type': 'ball-ball',
                'b1': bx1,
                'b2': bx2,
                'x1': np.interp(tc, tappr, b[bx1]['xa']),
                'y1': np.interp(tc, tappr, b[bx1]['ya']),
                'x2': np.interp(tc, tappr, b[bx2]['xa']),
                'y2': np.interp(tc, tappr, b[bx2]['ya'])
            })

    # Assign new hit events or next timestep
    for hit_event in hitlist:
        if hit_event['type'] == 'cushion':
            bi = hit_event['ball']
            ball0[bi]['t'][0] = hit_event['time']
            ball0[bi]['x'][0] = hit_event['x']
            ball0[bi]['y'][0] = hit_event['y']
        elif hit_event['type'] == 'ball-ball':
            b1, b2 = hit_event['b1'], hit_event['b2']
            ball0[b1]['vx'][0], ball0[b2]['vx'][0] = ball0[b2]['vx'][0], ball0[b1]['vx'][0]
            ball0[b1]['vy'][0], ball0[b2]['vy'][0] = ball0[b2]['vy'][0], ball0[b1]['vy'][0]

    return hit, err