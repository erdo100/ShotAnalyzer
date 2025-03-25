import numpy as np
from ball_velocity import ball_velocity
from ball_direction import ball_direction
from CushionAngle import cushion_angle

def eval_hit_events(hit, si, b1b2b3, SA, param):
    b1i = b1b2b3[0]

    # Initialize output variables
    for bi in range(3):
        hit[bi]['AimOffsetLL'] = []
        hit[bi]['AimOffsetSS'] = []
        hit[bi]['B1B2OffsetLL'] = []
        hit[bi]['B1B2OffsetSS'] = []
        hit[bi]['B1B3OffsetLL'] = []
        hit[bi]['B1B3OffsetSS'] = []

        for hi in range(8):
            for key in ['Type', 'FromC', 'ToC', 'V1', 'V2', 'Offset', 'Fraction', 'DefAngle', 'CutAngle', 'CInAngle', 'COutAngle', 'FromCPos', 'ToCPos', 'FromDPos', 'ToDPos']:
                hit[bi][key] = [np.nan] * 8

    # Evaluate hits for each ball
    for bi in b1b2b3:
        for hi, with_event in enumerate(hit[bi]['with']):
            if with_event == '-':
                continue

            v, v1, v2, alpha, offset = ball_velocity(SA['Shot'][si]['Route'][bi], hit[bi], hi)

            hit[bi]['V1'][hi] = v[0]
            hit[bi]['V2'][hi] = v[1]
            hit[bi]['Offset'][hi] = offset

            if v[0] > 0 or v[1] > 0:
                direction = ball_direction(SA['Shot'][si]['Route'][bi], hit[bi], hi, param)
            else:
                direction = np.full((2, 6), np.nan)

            hit[bi]['FromC'][hi] = direction[1, 0]
            hit[bi]['FromCPos'][hi] = direction[1, 1]
            hit[bi]['FromDPos'][hi] = direction[1, 2]
            hit[bi]['ToC'][hi] = direction[1, 3]
            hit[bi]['ToCPos'][hi] = direction[1, 4]
            hit[bi]['ToDPos'][hi] = direction[1, 5]

            hit[bi]['Type'][hi] = 0

            # Check for cushion hit
            if with_event in '1234':
                hit[bi]['Type'][hi] = 2
                angle = cushion_angle(int(with_event), v1, v2)
                hit[bi]['CInAngle'][hi] = angle[0]
                hit[bi]['COutAngle'][hi] = angle[1]

            # Check for ball-ball hit
            if with_event in 'WYR':
                hit[bi]['Type'][hi] = 1

                if v[0] > 0:
                    tb10i = np.where(hit[bi]['t'][hi] == SA['Shot'][si]['Route'][bi]['t'])[0]
                    tb11i = np.where(hit[bi]['t'][hi] == SA['Shot'][si]['Route'][b1b2b3[1]]['t'])[0]

                    if tb10i.size == 0:
                        print('No time point found matching to hit event')
                        continue

                    pb1 = np.array([SA['Shot'][si]['Route'][bi]['x'][tb10i[0]], SA['Shot'][si]['Route'][bi]['y'][tb10i[0]]])
                    pb2 = np.array([SA['Shot'][si]['Route'][b1b2b3[1]]['x'][tb11i[0]], SA['Shot'][si]['Route'][b1b2b3[1]]['y'][tb11i[0]]])

                    hib2 = np.where(hit[b1b2b3[1]]['t'] == hit[bi]['t'][hi])[0]
                    _, v1b2, _, _, _ = ball_velocity(SA['Shot'][si]['Route'][b1b2b3[1]], hit[b1b2b3[1]], hib2[0])

                    velrel = np.append(v1 - v1b2, 0)

                    try:
                        hit[bi]['Fraction'][hi] = 1 - np.linalg.norm(np.cross(np.append(pb2 - pb1, 0), velrel)) / np.linalg.norm(velrel) / param['ballR'] / 2
                        hit[bi]['DefAngle'][hi] = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
                    except:
                        hit[bi]['Fraction'][hi] = 0
                        hit[bi]['DefAngle'][hi] = 0

    return hit