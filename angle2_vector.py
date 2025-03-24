import numpy as np

def angle2_vector(a, b):
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) * 180 / np.pi