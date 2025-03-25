import numpy as np
from angle_vector import angle_vector

def cushion_angle(ci, v1, v2):
    v1 = np.append(v1, 0)
    v2 = np.append(v2, 0)

    if ci == 1:
        b = np.array([0, 1, 0])
    elif ci == 2:
        b = np.array([-1, 0, 0])
    elif ci == 3:
        b = np.array([0, -1, 0])
    elif ci == 4:
        b = np.array([1, 0, 0])

    angle = [
        angle_vector(v1, -b),
        angle_vector(v2, b)
    ]

    # Check whether we have a negative angle
    if ci == 1 and np.sign(v1[0]) != np.sign(v2[0]):
        angle[0] = -angle[0]
    elif ci == 2 and np.sign(v1[1]) != np.sign(v2[1]):
        angle[0] = -angle[0]
    elif ci == 3 and np.sign(v1[0]) != np.sign(v2[0]):
        angle[0] = -angle[0]
    elif ci == 4 and np.sign(v1[1]) != np.sign(v2[1]):
        angle[0] = -angle[0]

    return angle