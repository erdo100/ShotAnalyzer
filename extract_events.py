import numpy as np
from str2num_B1B2B3 import str2num_B1B2B3
from angle_vector import angle_vector

def extract_events(si, SA):
    err = {'code': None, 'text': ''}

    b1b2b3, b1i, b2i, b3i = str2num_B1B2B3(SA['Table']['B1B2B3'][si])

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
    while do_scan:
        dT = np.diff(Tall0[:2])[0]
        tappr = Tall0[0] + dT * np.linspace(0.01, 1, 101)

        for bi in range(3):
            if len(ball0[bi]['t']) >= 2:
                vnext = ball0[bi]['v'][0]
                if vnext > 0:
                    ball0[bi]['x'][0] += ball0[bi]['vx'][0] * dT
                    ball0[bi]['y'][0] += ball0[bi]['vy'][0] * dT

        Tall0 = Tall0[1:]
        do_scan = len(Tall0) >= 3

    for bi in range(3):
        SA['Shot'][si]['Route'][bi] = {
            't': ball0[bi]['t'],
            'x': ball0[bi]['x'],
            'y': ball0[bi]['y']
        }

    return hit, err