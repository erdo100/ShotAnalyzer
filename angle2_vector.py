import numpy as np

def angle2_vector(vec1, vec2):
    """Calculate angle between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    angle_rad = np.arccos(dot_product / (norm_a * norm_b))
    angle_deg = np.degrees(angle_rad)
    return angle_deg