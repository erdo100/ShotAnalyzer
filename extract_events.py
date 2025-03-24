import numpy as np
import matplotlib.pyplot as plt

def extract_events(si):
    # Extract all ball-ball hit events and ball-cushion hit events

    global SA, param

    plot_flag = plt.get_current_fig_manager().findobj('Tag', 'PlotAnalytics').get_checked() == 'on'
    ti_plot_start = 1

    # Initialize outputs
    err = {'code': None, 'text': ''}

    col = 'WYR'
    ax = plt.gca()

    # Get B1, B2, B3 from ShotList
    b1b2b3, b1i, b2i, b3i = str2num_B1B2B3(SA.Table.loc[si, 'B1B2B3'])

    # Get the copy from original data
    ball0 = [None] * 3
    for bi in range(3):
        ball0[bi] = SA.Shot[si].Route0[bi]
        SA.Shot[si].Route[bi].t = SA.Shot[si].Route0[bi].t
        SA.Shot[si].Route[bi].x = SA.Shot[si].Route0[bi].x
        SA.Shot[si].Route[bi].y = SA.Shot[si].Route0[bi].y

    ball = [None] * 3
    for bi in range(3):
        ball[bi] = {'x': SA.Shot[si].Route0[bi].x[0],
                    'y': SA.Shot[si].Route0[bi].y[0],
                    't': 0}

    # Create common Time points
    Tall0 = np.unique(np.concatenate([ball0[0].t, ball0[1].t, ball0[2].t]))
    ti = 0

    # Discretization of dT
    tvec = np.linspace(0.01, 1, 101)

    # Initialize hit structure
    hit = [None] * 3
    for bi in range(3):
        hit[bi] = {'with': '-', 't': [0], 'XPos': [ball0[bi].x[0]], 'YPos': [ball0[bi].y[0]],
                   'Kiss': 0, 'Point': 0, 'Fuchs': 0, 'PointDist': 3000, 'KissDistB1': 3000,
                   'Tpoint': 3000, 'Tkiss': 3000, 'Tready': 3000, 'TB2hit': 3000, 'Tfailure': 3000}

    # Calculate ball velocities
    for bi in range(3):
        ball0[bi].dt = np.diff(ball0[bi].t)
        ball0[bi].vx = np.concatenate([np.diff(ball0[bi].x) / ball0[bi].dt, [0]])
        ball0[bi].vy = np.concatenate([np.diff(ball0[bi].y) / ball0[bi].dt, [0]])
        ball0[bi].v = np.concatenate([np.sqrt(ball0[bi].vx**2 + ball0[bi].vy**2), [0]])

    # Set initial ball speeds for B2 and B3 = 0
    ball0[b2i].vx[0] = 0
    ball0[b2i].vy[0] = 0
    ball0[b2i].v[0] = 0
    ball0[b3i].vx[0] = 0
    ball0[b3i].vy[0] = 0
    ball0[b3i].v[0] = 0

    if plot_flag:
        ax = plot_table()
        lw = [0.5] * 3
        plot_shot(ax, ball0, lw)

    # Main loop
    do_scan = True
    bi_list = [0, 1, 2]

    while do_scan:
        ti += 1

        # Approximate Position of Ball at next time step
        dT = np.diff(Tall0[:2])
        tappr = Tall0[0] + dT * tvec
        b = [None] * 3
        for bi in range(3):
            if len(ball0[bi].t) >= 2:
                # Travel distance in original data
                if len(ball[bi]['x']) >= 2:
                    ds0 = np.sqrt((ball[bi]['x'][1] - ball[bi]['x'][0])**2 + 
                          (ball[bi]['y'][1] - ball[bi]['y'][0])**2)
                    v0 = ds0 / dT
                # Current Speed
                b[bi] = {'vt1': ball0[bi].v[0]}

                # Was last hit in last time step?
                iprev = np.where(ball[bi]['t'] >= hit[bi]['t'][-1])[0][-param.timax_appr:]

                # Speed Components in the next ti
                if hit[bi]['with'][-1] == '-':
                    # Ball was not moving previously
                    b[bi]['v1'] = [0, 0]
                    b[bi]['v2'] = [ball0[bi].vx[1], ball0[bi].vy[1]]
                elif len(iprev) == 1:
                    # When last hit was on this ball
                    b[bi]['v1'] = [ball0[bi].vx[0], ball0[bi].vy[0]]
                    b[bi]['v2'] = [ball0[bi].vx[1], ball0[bi].vy[1]]
                else:
                    # Extrapolate from last points
                    CoefVX = np.linalg.lstsq(np.vstack([ball[bi]['t'][iprev], np.ones(len(iprev))]).T, 
                                            ball[bi]['x'][iprev], rcond=None)[0]
                    CoefVY = np.linalg.lstsq(np.vstack([ball[bi]['t'][iprev], np.ones(len(iprev))]).T, 
                                            ball[bi]['y'][iprev], rcond=None)[0]
                    b[bi]['v1'] = [CoefVX[0], CoefVY[0]]
                    b[bi]['v2'] = [ball0[bi].vx[1], ball0[bi].vy[1]]

                # Speed in data for next ti
                b[bi]['vt1'] = np.linalg.norm(b[bi]['v1'])
                b[bi]['vt2'] = np.linalg.norm(b[bi]['v2'])
                vtnext = max(b[bi]['vt1'], ball0[bi].v[0])

                if np.linalg.norm(b[bi]['v1']) > 0:
                    vnext = b[bi]['v1'] / np.linalg.norm(b[bi]['v1']) * vtnext
                else:
                    vnext = [0, 0]

                # Approximation of next ti
                b[bi]['xa'] = ball[bi]['x'][-1] + vnext[0] * dT * tvec
                b[bi]['ya'] = ball[bi]['y'][-1] + vnext[1] * dT * tvec

        # Plot
        if plot_flag and ti >= ti_plot_start:
            for bi in range(3):
                plt.plot(ax, ball[bi]['x'][-1], ball[bi]['y'][-1], 'ok', tag='hlast')
                plt.plot(ax, b[bi]['xa'][[0, -1]], b[bi]['ya'][[0, -1]], '--or', tag='hlast', 
                         markersize=7, linewidth=2, markerfacecolor='r')
                plt.plot(ax, ball[bi]['x'][-1] + param.ballcirc[0, :], 
                         ball[bi]['y'][-1] + param.ballcirc[1, :], '-k', tag='hlast')
                plt.plot(ax, b[bi]['xa'][-1] + param.ballcirc[0, :], 
                         b[bi]['ya'][-1] + param.ballcirc[1, :], '-r', tag='hlast')
                plt.draw()

        # Calculate Ball trajectory angle change
        for bi in range(3):
            b[bi]['a12'] = angle_vector(b[bi]['v1'], b[bi]['v2'])

        # Calculate the Ball-Ball Distance
        BB = [[0, 1], [0, 2], [1, 2]]
        d = [None] * 3
        for bbi in range(3):
            bx1 = b1b2b3[BB[bbi][0]]
            bx2 = b1b2b3[BB[bbi][1]]
            d[bbi] = {'BB': np.sqrt((b[bx1]['xa'] - b[bx2]['xa'])**2 + 
                                 (b[bx1]['ya'] - b[bx2]['ya'])**2) - 2 * param.ballR}

        # Calculate Cushion distance
        for bi in range(3):
            b[bi]['cd'] = np.column_stack([
                b[bi]['ya'] - param.ballR,
                param.size[1] - param.ballR - b[bi]['xa'],
                param.size[0] - param.ballR - b[bi]['ya'],
                b[bi]['xa'] - param.ballR
            ])

        hitlist = []
        lasthitball = 0

        # Evaluate cushion hit
        for bi in range(3):
            for cii in range(4):
                checkdist = np.any(b[bi]['cd'][:, cii] <= 0)
                checkangle = b[bi]['a12'] > 1 or b[bi]['a12'] == -1
                velx = b[bi]['v1'][0]
                vely = b[bi]['v1'][1]

                checkcush = False
                tc = 0
                if checkdist and checkangle and vely < 0 and cii == 0 and b[bi]['v1'][1] != 0:  # bottom cushion
                    checkcush = True
                    tc = np.interp(0, b[bi]['cd'][:, cii], tappr)
                    cushx = np.interp(tc, tappr, b[bi]['xa'])
                    cushy = param.ballR
                elif checkdist and checkangle and velx > 0 and cii == 1 and b[bi]['v1'][0] != 0:  # right cushions
                    checkcush = True
                    tc = np.interp(0, b[bi]['cd'][:, cii], tappr)
                    cushx = param.size[1] - param.ballR
                    cushy = np.interp(tc, tappr, b[bi]['ya'])
                elif checkdist and checkangle and vely > 0 and cii == 2 and b[bi]['v1'][1] != 0:  # top Cushion
                    checkcush = True
                    tc = np.interp(0, b[bi]['cd'][:, cii], tappr)
                    cushy = param.size[0] - param.ballR
                    cushx = np.interp(tc, tappr, b[bi]['xa'])
                elif checkdist and checkangle and velx < 0 and cii == 3 and b[bi]['v1'][0] != 0:  # left Cushion
                    checkcush = True
                    tc = np.interp(0, b[bi]['cd'][:, cii], tappr)
                    cushx = param.ballR
                    cushy = np.interp(tc, tappr, b[bi]['ya'])

                if checkcush:
                    hitlist.append([tc, bi, 2, cii, cushx, cushy])

        # Evaluate Ball-Ball hit
        for bbi in range(3):
            bx1 = b1b2b3[BB[bbi][0]]
            bx2 = b1b2b3[BB[bbi][1]]

            checkdist = np.any(d[bbi]['BB'] <= 0) and np.any(d[bbi]['BB'] > 0) and 
                       ((d[bbi]['BB'][0] >= 0 and d[bbi]['BB'][-1] < 0) or 
                        (d[bbi]['BB'][0] < 0 and d[bbi]['BB'][-1] >= 0))

            if checkdist:
                tc = np.interp(0, d[bbi]['BB'], tappr)

            checkangleb1 = b[bx1]['a12'] > 10 or b[bx1]['a12'] == -1
            checkangleb2 = b[bx2]['a12'] > 10 or b[bx2]['a12'] == -1

            checkvelb1 = abs(b[bx1]['vt2'] - b[bx1]['vt1']) > 50 if abs(b[bx1]['vt1']) > np.finfo(float).eps or abs(b[bx1]['vt2']) > np.finfo(float).eps else False
            checkvelb2 = abs(b[bx2]['vt2'] - b[bx2]['vt1']) > 50 if abs(b[bx2]['vt1']) > np.finfo(float).eps or abs(b[bx2]['vt2']) > np.finfo(float).eps else False

            checkdouble = tc >= hit[bx1]['t'][-1] + 0.01 and tc >= hit[bx2]['t'][-1] + 0.01

            if checkdouble and checkdist:
                hitlist.append([tc, bx1, 1, bx2, np.interp(tc, tappr, b[bx1]['xa']), np.interp(tc, tappr, b[bx1]['ya'])])
                hitlist.append([tc, bx2, 1, bx1, np.interp(tc, tappr, b[bx2]['xa']), np.interp(tc, tappr, b[bx2]['ya'])])

        # Assign new hit event or next timestep into the ball route history
        if hitlist and Tall0[1] >= tc:
            tc = min([h[0] for h in hitlist])
            if Tall0[1] < tc:
                print('Warning: hit is after next time step')

            # Assign new data for hits
            for hi in [h for h in hitlist if h[0] == tc]:
                bi = hi[1]
                bi_list.remove(bi)
                lasthitball = bi
                lasthit_t = tc

                checkdist = False
                if len(ball0[bi].x) > 1:
                    checkdist = np.sqrt((ball0[bi].x[1] - hi[4])**2 + (ball0[bi].y[1] - hi[5])**2) < 5

                Tall0[0] = tc
                ball0[bi].t[0] = tc
                ball0[bi].x[0] = hi[4]
                ball0[bi].y[0] = hi[5]

                ball[bi]['t'].append(tc)
                ball[bi]['x'].append(hi[4])
                ball[bi]['y'].append(hi[5])

                hit[bi]['t'].append(hi[0])
                if hi[2] == 1:
                    hit[bi]['with'] += col[hi[3]]
                elif hi[2] == 2:
                    hit[bi]['with'] += str(hi[3])
                hit[bi]['XPos'].append(hi[4])
                hit[bi]['YPos'].append(hi[5])

                if plot_flag and ti >= ti_plot_start:
                    plt.plot(ax, ball[bi]['x'][-1], ball[bi]['y'][-1], 'oc', tag='hlast2')
                    plt.plot(ax, ball[bi]['x'][-1] + param.ballcirc[0, :], 
                             ball[bi]['y'][-1] + param.ballcirc[1, :], '-w', tag='hlast2')
                    plt.draw()

            # Assign time to ball without hit event
            for bi in bi_list:
                ball[bi]['t'].append(Tall0[0])
                ball[bi]['x'].append(np.interp(Tall0[0], ball0[bi].t, ball0[bi].x))
                ball[bi]['y'].append(np.interp(Tall0[0], ball0[bi].t, ball0[bi].y))

                if b[bi]['vt1'] > 0:
                    ball0[bi].t[0] = Tall0[0]
                    ball0[bi].x[0] = ball[bi]['x'][-1]
                    ball0[bi].y[0] = ball[bi]['y'][-1]

                if plot_flag and ti >= ti_plot_start:
                    plt.plot(ax, ball[bi]['x'][-1], ball[bi]['y'][-1], 'oc', tag='hlast2')
                    plt.draw()

        else:
            # Assign new data for no Ball hit
            Tall0 = np.delete(Tall0, 0)

            for bi in range(3):
                ball[bi]['t'].append(Tall0[0])
                if b[bi]['vt1'] > 0:
                    ball[bi]['x'].append(np.interp(Tall0[0], ball0[bi].t, ball0[bi].x))
                    ball[bi]['y'].append(np.interp(Tall0[0], ball0[bi].t, ball0[bi].y))
                else:
                    ball[bi]['x'].append(ball[bi]['x'][-1])
                    ball[bi]['y'].append(ball[bi]['y'][-1])

                ind = np.where(ball0[bi].t < Tall0[0])[0]
                if ball0[bi].t[1] <= Tall0[0]:
                    ball0[bi].t = np.delete(ball0[bi].t, ind)
                    ball0[bi].x = np.delete(ball0[bi].x, ind)
                    ball0[bi].y = np.delete(ball0[bi].y, ind)
                else:
                    ball0[bi].t[0] = Tall0[0]
                    ball0[bi].x[0] = ball[bi]['x'][-1]
                    ball0[bi].y[0] = ball[bi]['y'][-1]

        # Update derivatives
        for bi in range(3):
            if hit[bi]['with'][-1] != '-':
                ind = np.argsort(ball0[bi].t)
                ball0[bi].t = ball0[bi].t[ind]
                ball0[bi].x = ball0[bi].x[ind]
                ball0[bi].y = ball0[bi].y[ind]

                ball0[bi].dt = np.diff(ball0[bi].t)
                ball0[bi].vx = np.concatenate([np.diff(ball0[bi].x) / ball0[bi].dt, [0]])
                ball0[bi].vy = np.concatenate([np.diff(ball0[bi].y) / ball0[bi].dt, [0]])
                ball0[bi].v = np.concatenate([np.sqrt(ball0[bi].vx**2 + ball0[bi].vy**2), [0]])

        # Plot
        if plot_flag and ti >= ti_plot_start:
            for bi in range(3):
                plt.plot(ax, ball[bi]['x'], ball[bi]['y'], 'cs', markersize=8, tag='hlast')
            hlast = plt.gca().findobj(tag='hlast')
            for h in hlast:
                h.remove()

        # Check whether time is over
        do_scan = len(Tall0) >= 3

    # Plot
    if plot_flag:
        hlast2 = plt.gca().findobj(tag='hlast2')
        for h in hlast2:
            h.remove()

    # Assign back the shot
    for bi in range(3):
        SA.Shot[si].Route[bi].t = ball[bi]['t']
        SA.Shot[si].Route[bi].x = ball[bi]['x']
        SA.Shot[si].Route[bi].y = ball[bi]['y']

    return hit, err

# Helper functions (placeholders, implement as needed)
def str2num_B1B2B3(B1B2B3):
    """
    Convert a 3-character string of ball identifiers to numerical indices.
    
    Parameters:
    B1B2B3 (str): 3-character string where each character is one of 'W', 'Y', or 'R'
    
    Returns:
    tuple: (b1b2b3, b1i, b2i, b3i) where:
        - b1b2b3 is a list of indices [b1, b2, b3]
        - b1i, b2i, b3i are the individual indices
    """
    # Convert each character to its index in 'WYR' (1-based)
    b1b2b3 = [
        'WYR'.index(B1B2B3[0]) + 1,
        'WYR'.index(B1B2B3[1]) + 1,
        'WYR'.index(B1B2B3[2]) + 1
    ]
    
    b1i = b1b2b3[0]
    b2i = b1b2b3[1]
    b3i = b1b2b3[2]
    
    return b1b2b3, b1i, b2i, b3i


def plot_table():
    # Placeholder for plotting the table
    pass

def plot_shot(ax, ball0, lw):
    # Placeholder for plotting the shot
    pass


def angle_vector(a, b):
    """
    Calculate the angle between two vectors in degrees.
    
    Parameters:
    a, b : array-like
        Input vectors (must be same length)
    
    Returns:
    float: Angle between vectors in degrees, or -1/-2 for special cases
    """
    a = np.array(a)
    b = np.array(b)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a > 0 and norm_b > 0:
        # Calculate angle in radians and convert to degrees
        angle = np.arccos(np.dot(a, b) / (norm_a * norm_b)) * 180 / np.pi
    elif norm_a > 0 or norm_b > 0:
        # One vector is zero-length
        angle = -1
    else:
        # Both vectors are zero-length
        angle = -2
    
    return angle
