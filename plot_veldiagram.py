import matplotlib.pyplot as plt
import numpy as np

def plot_veldiagram(SA, param):
    cols = ['b', 'g', 'r']

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    fig.canvas.manager.set_window_title("Velocity Plot")

    legtxt = []

    for si in SA['Current_si']:
        b1b2b3, b1i, b2i, b3i = str2num_B1B2B3(SA['Table']['B1B2B3'][si])
        ball = []

        for bi in range(3):
            vx = np.diff(SA['Shot'][si]['Route0'][bi]['x'], append=0) / np.diff(SA['Shot'][si]['Route0'][bi]['t'], append=1)
            vy = np.diff(SA['Shot'][si]['Route0'][bi]['y'], append=0) / np.diff(SA['Shot'][si]['Route0'][bi]['t'], append=1)
            t = SA['Shot'][si]['Route0'][bi]['t']

            if b1b2b3[bi] != 1:
                t = np.insert(t, 1, t[1])
                vx = np.insert(vx, 1, 0)
                vy = np.insert(vy, 1, 0)

            v = np.sqrt(vx**2 + vy**2) / 1000
            ball.append({'t': t, 'v': v})

        for bi in range(3):
            ax.plot(ball[b1b2b3[bi] - 1]['t'], ball[b1b2b3[bi] - 1]['v'], f'-{cols[bi]}', label=f'Shot {si} B{bi + 1}')

    ax.grid(True)
    ax.set_title("Balls Speeds")
    ax.set_xlabel("Time in s")
    ax.set_ylabel("Velocity in m/s")
    ax.legend()

    plt.show()