import numpy as np
from eval_kiss import eval_kiss

def eval_point_and_kiss_control(si, hit, SA, param):
    cols = 'WYR'

    b1b2b3, b1i, b2i, b3i = str2num_B1B2B3(SA['Table']['B1B2B3'][si])

    ball = [SA['Shot'][si]['Route0'][b1b2b3[bi]] for bi in range(3)]

    hit = eval_kiss(hit, b1i, b2i, b3i)

    BB = [[1, 2], [1, 3], [2, 3]]
    Tall = np.unique(np.concatenate([ball[0]['t'], ball[1]['t'], ball[2]['t']]))

    CC = []
    for bbi in range(3):
        bx1, bx2 = BB[bbi]
        B1x = np.interp(Tall, ball[bx1 - 1]['t'], ball[bx1 - 1]['x'])
        B1y = np.interp(Tall, ball[bx1 - 1]['t'], ball[bx1 - 1]['y'])
        B2x = np.interp(Tall, ball[bx2 - 1]['t'], ball[bx2 - 1]['x'])
        B2y = np.interp(Tall, ball[bx2 - 1]['t'], ball[bx2 - 1]['y'])
        CC.append({'dist': np.sqrt((B2x - B1x)**2 + (B2y - B1y)**2)})

    if hit[b1i]['Kiss'] == 1:
        ei = np.where(hit[b1i]['t'] == hit[b1i]['Tkiss'])[0][0]
        hit[b1i]['KissDistB1'] = hit[b1i]['Fraction'][ei] * param['ballR'] * 2
    else:
        ind1 = np.where(Tall > hit[b1i]['TB2hit'])[0][0]
        ind2 = ind1 + np.where(np.diff(CC[0]['dist'][ind1:]) < 0)[0][0]
        ind3 = min([
            np.where(Tall >= hit[b1i]['Tkiss'])[0][0] if hit[b1i]['Tkiss'] < 1000 else len(Tall),
            np.where(Tall >= hit[b1i]['Tpoint'])[0][0] if hit[b1i]['Tpoint'] < 1000 else len(Tall),
            np.where(Tall >= hit[b1i]['Tfailure'])[0][0] if hit[b1i]['Tfailure'] < 1000 else len(Tall),
            len(Tall)
        ])
        hit[b1i]['KissDistB1'] = np.min(CC[0]['dist'][ind2:ind3])

    if hit[b1i]['Point']:
        hi1 = np.where(hit[b1i]['with'] == SA['Table']['B1B2B3'][si][2])[0][0]
        hi3 = np.where(hit[b3i]['with'] == SA['Table']['B1B2B3'][si][0])[0][0]

        if hit[b1i]['Fraction'][hi1] == 1:
            hitsign = 0
        else:
            t1i = np.where(ball[b3i]['t'] < hit[b1i]['t'][hi1])[0][-1]
            t3i = np.where(ball[b3i]['t'] < hit[b1i]['t'][hi1])[0][-1]

            v1 = [
                hit[b1i]['XPos'][hi1] - ball[0]['x'][t1i],
                hit[b1i]['YPos'][hi1] - ball[0]['y'][t1i],
                0
            ]
            v2 = [
                hit[b3i]['XPos'][hi3] - hit[b1i]['XPos'][hi1],
                hit[b3i]['YPos'][hi3] - hit[b1i]['YPos'][hi1],
                0
            ]
            v3 = np.cross(v2, v1)
            hitsign = 1 if v3[2] > 0 else -1

        hit[b1i]['PointDist'] = hitsign * (1 - hit[b1i]['Fraction'][hi1]) * param['ballR'] * 2
    else:
        ind = np.where(Tall > hit[b1i]['Tready'])[0]
        if ind.size > 0:
            PointDist = np.min(CC[1]['dist'][ind])
            imin = np.argmin(CC[1]['dist'][ind])

            t1i = np.where(ball[0]['t'] <= Tall[ind[imin]])[0][-1]
            t3i = np.where(ball[2]['t'] <= Tall[ind[imin]])[0][-1]

            v1 = [
                ball[0]['x'][t1i] - ball[0]['x'][t1i - 1],
                ball[0]['y'][t1i] - ball[0]['y'][t1i - 1],
                0
            ]
            v2 = [
                ball[2]['x'][t3i] - ball[0]['x'][t1i],
                ball[2]['y'][t3i] - ball[0]['y'][t1i],
                0
            ]
            v3 = np.cross(v2, v1)
            hitsign = 1 if v3[2] > 0 else -1

            hit[b1i]['PointDist'] = hitsign * PointDist
        else:
            hit[b1i]['PointDist'] = 3000

    return hit