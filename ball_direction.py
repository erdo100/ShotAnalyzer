import numpy as np

def ball_direction(route, hit, ei, param):
    direction = np.zeros((2, 6))  # fromCushion# PosOn PosThrough toCushion# PosOn PosThrough

    imax = 10  # max node number to be considered
    if ei == 0 and len(hit['t']) > 1:
        t1i = np.where((route['t'] >= hit['t'][ei]) & (route['t'] <= hit['t'][ei + 1]))[0][:imax]
        t2i = np.where((route['t'] >= hit['t'][ei]) & (route['t'] <= hit['t'][ei + 1]))[0][-imax:]
    elif ei < len(hit['t']) - 1:
        t1i = np.where((route['t'] >= hit['t'][ei - 1]) & (route['t'] <= hit['t'][ei]))[0][-imax:]
        t2i = np.where((route['t'] >= hit['t'][ei]) & (route['t'] <= hit['t'][ei + 1]))[0][-imax:]
    else:
        t1i = np.where(route['t'] >= hit['t'][ei])[0][:imax]
        t2i = np.where(route['t'] >= hit['t'][ei])[0][-imax:]

    pstart = [
        [route['x'][t1i[0]], route['y'][t1i[0]]],
        [route['x'][t2i[0]], route['y'][t2i[0]]]
    ]
    pend = [
        [route['x'][t1i[-1]], route['y'][t1i[-1]]],
        [route['x'][t2i[-1]], route['y'][t2i[-1]]]
    ]

    for i in range(2):
        p1 = np.array(pstart[i])
        p2 = np.array(pend[i])

        if not np.any(np.isnan([p1, p2])):
            if p1[1] != p2[1]:
                xon1 = np.interp(param['ballR'], [p1[1], p2[1]], [p1[0], p2[0]])
                xthrough1 = np.interp(-param['diamdist'], [p1[1], p2[1]], [p1[0], p2[0]])
                xon3 = np.interp(param['size'][0] - param['ballR'], [p1[1], p2[1]], [p1[0], p2[0]])
                xthrough3 = np.interp(param['size'][0] + param['diamdist'], [p1[1], p2[1]], [p1[0], p2[0]])
            else:
                xon1 = xon3 = xthrough1 = xthrough3 = p1[0]

            if p1[0] != p2[0]:
                yon2 = np.interp(param['size'][1] - param['ballR'], [p1[0], p2[0]], [p1[1], p2[1]])
                yon4 = np.interp(param['ballR'], [p1[0], p2[0]], [p1[1], p2[1]])
                ythrough2 = np.interp(param['size'][1] + param['diamdist'], [p1[0], p2[0]], [p1[1], p2[1]])
                ythrough4 = np.interp(-param['diamdist'], [p1[0], p2[0]], [p1[1], p2[1]])
            else:
                yon2 = yon4 = ythrough2 = ythrough4 = p1[1]

            if param['ballR'] <= xon1 <= param['size'][1] - param['ballR']:
                if p2[1] < p1[1]:
                    direction[i, 4] = 1
                    direction[i, 5] = xon1
                    direction[i, 6] = xthrough1
                else:
                    direction[i, 0] = 1
                    direction[i, 1] = xon1
                    direction[i, 2] = xthrough1

            if param['ballR'] <= xon3 <= param['size'][1] - param['ballR']:
                if p1[1] < p2[1]:
                    direction[i, 4] = 3
                    direction[i, 5] = xon3
                    direction[i, 6] = xthrough3
                else:
                    direction[i, 0] = 3
                    direction[i, 1] = xon3
                    direction[i, 2] = xthrough3

            if param['ballR'] <= yon2 <= param['size'][0] - param['ballR']:
                if p1[0] < p2[0]:
                    direction[i, 4] = 2
                    direction[i, 5] = yon2
                    direction[i, 6] = ythrough2
                else:
                    direction[i, 0] = 2
                    direction[i, 1] = yon2
                    direction[i, 2] = ythrough2

            if param['ballR'] <= yon4 <= param['size'][0] - param['ballR']:
                if p2[0] < p1[0]:
                    direction[i, 4] = 4
                    direction[i, 5] = yon4
                    direction[i, 6] = ythrough4
                else:
                    direction[i, 0] = 4
                    direction[i, 1] = yon4
                    direction[i, 2] = ythrough4

    return direction