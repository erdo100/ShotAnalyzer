import numpy as np
import matplotlib.pyplot as plt

def correct_velocity(shot):
    print(shot)

    # Calculate differences and velocities
    dx = np.diff(shot['x'])
    dy = np.diff(shot['y'])
    dt = np.diff(shot['t'])

    ds = np.sqrt((shot['x'][1:] - shot['x'][:-1])**2 + (shot['y'][1:] - shot['y'][:-1])**2) / 1000
    vel = ds / dt

    ind = np.where(dt > np.mean(dt))[0]

    plt.figure()
    plt.plot(shot['t'], shot['x'], '-')
    plt.plot(shot['t'], shot['y'], '-')
    plt.plot(shot['t'][ind], shot['y'][ind], 'o')
    plt.grid()
    plt.show()

    check = True
    while check:
        # Identify single extremes
        vel_mvmean = np.convolve(vel, np.ones(5)/5, mode='same')  # Moving average
        dev = vel_mvmean - vel
        ind = np.where(dev < -0.5)[0]

        if len(ind) > 0:
            ind = ind[0]
            plt.subplot(3, 1, 1)
            plt.plot(shot['t'][ind], dx[ind], 'o')
            plt.plot(shot['t'][ind], dy[ind], 'o')

            plt.subplot(3, 1, 2)
            plt.plot(shot['t'][ind], vel[ind], 'o')

            plt.subplot(3, 1, 3)
            plt.plot(shot['t'][ind], shot['x'][ind], 'o')
            plt.plot(shot['t'][ind], shot['y'][ind], 'o')

            if abs((dx[ind+1] - dx[ind-1]) / dx[ind-1]) < 0.1 or abs((dy[ind+1] - dy[ind-1]) / dy[ind-1]) < 0.1:
                dx1 = (dx[ind+1] + dx[ind-1]) / 2
                dy1 = (dy[ind+1] + dy[ind-1]) / 2
                shot['x'][ind] = shot['x'][ind-1] + dx1
                shot['y'][ind] = shot['y'][ind-1] + dy1

                dx = np.diff(shot['x'])
                dy = np.diff(shot['y'])

                ds = np.sqrt((shot['x'][1:] - shot['x'][:-1])**2 + (shot['y'][1:] - shot['y'][:-1])**2) / 1000
                vel = ds / dt

                plt.subplot(3, 1, 1)
                plt.plot(shot['t'][ind], dx[ind], 'x')
                plt.plot(shot['t'][ind], dy[ind], 'x')

                plt.subplot(3, 1, 2)
                plt.plot(shot['t'][ind], vel[ind], 'x')

                plt.subplot(3, 1, 3)
                plt.plot(shot['t'][ind], shot['x'][ind], 'x')
                plt.plot(shot['t'][ind], shot['y'][ind], 'x')

                plt.figure()
                plt.plot(shot['t'], shot['y'], '-x')
                plt.show()

    return shot