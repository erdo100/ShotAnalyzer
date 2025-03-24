import numpy as np

def angle_vector(a, b):
    if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
        angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) * 180 / np.pi
    elif np.linalg.norm(a) > 0 or np.linalg.norm(b) > 0:
        angle = -1
    else:
        angle = -2
    return angle