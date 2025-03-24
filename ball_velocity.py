import numpy as np

def ball_velocity(ball, hit, ei):
    imax = 10

    if hit['with'][ei] != '-':
        ti = np.interp(hit['t'][ei], ball['t'], np.arange(len(ball['t'])))

        if ei > 0:
            ti_before = np.interp(hit['t'][ei - 1], ball['t'], np.arange(len(ball['t'])))
        else:
            ti_before = 0

        if ei + 1 < len(hit['t']):
            ti_after = np.interp(hit['t'][ei + 1], ball['t'], np.arange(len(ball['t'])))
        else:
            ti_after = len(ball['t']) - 1

        # Before condition
        it = min(int(ti - ti_before), imax)
        if it >= 1 or ei == 0:
            if ei == 0:
                ind = [1, 4]
            else:
                ind = [int(ti - it), int(ti)]

            dx = np.sqrt(np.diff(ball['x'][ind])**2 + np.diff(ball['y'][ind])**2)
            dt = np.diff(ball['t'][ind])

            vt0 = dx / dt
            vx = np.diff(ball['x'][ind]) / dt
            vy = np.diff(ball['y'][ind]) / dt

            if vt0 > 0:
                vt1 = vt0 / 1000
                v1 = [vx / 1000, vy / 1000]
                alpha1 = np.arctan2(vx, vy) * 180 / np.pi
            else:
                vt1 = vt0 / 1000
                v1 = [0, 0]
                alpha1 = np.nan
        else:
            vt1 = 0
            v1 = [0, 0]
            alpha1 = np.nan

        # After condition
        it = min(int(ti_after - ti), imax)
        if it >= 1:
            ind = [int(ti_after - it), int(ti_after)]
            dx = np.sqrt(np.diff(ball['x'][ind])**2 + np.diff(ball['y'][ind])**2)
            dt = np.diff(ball['t'][ind])

            vt0 = dx / dt
            vx = np.diff(ball['x'][ind]) / dt
            vy = np.diff(ball['y'][ind]) / dt

            if vt0 > 0:
                vt2 = vt0 / 1000
                v2 = [vx / 1000, vy / 1000]
                alpha2 = np.arctan2(vx, vy) * 180 / np.pi
            else:
                vt2 = vt0 / 1000
                v2 = [0, 0]
                alpha2 = np.nan
        else:
            vt2 = 0
            v2 = [0, 0]
            alpha2 = np.nan

        # Calculate offset
        p1 = np.array([ball['x'][int(ti)], ball['y'][int(ti)]])
        p2 = np.array([ball['x'][int(ti_after)], ball['y'][int(ti_after)]])

        if np.linalg.norm(v1) > 0:
            offset = np.linalg.norm(np.cross(np.append(p2 - p1, 0), np.append(v2, 0))) / np.linalg.norm(np.append(v2, 0))
        else:
            offset = np.nan

        return [vt1, vt2], v1, v2, [alpha1, alpha2], offset

    else:
        return [0, 0], [0, 0], [0, 0], [0, 0], 0