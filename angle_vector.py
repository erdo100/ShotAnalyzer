import numpy as np

# Use the updated angle_vector function provided previously
def angle_vector(a, b):
    """
    Calculates the angle between two vectors in degrees, based on angle_vector.txt.

    Args:
        a (np.ndarray): The first vector (1D NumPy array).
        b (np.ndarray): The second vector (1D NumPy array).

    Returns:
        float: The angle in degrees. Returns -1 if only one vector has a
               non-zero norm, and -2 if both vectors have zero norm.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a > 0 and norm_b > 0:
        dot_product = np.dot(a, b)
        cos_theta = np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        angle_deg = angle_rad * 180 / np.pi
        return angle_deg
    elif norm_a > 0 or norm_b > 0:
        return -1.0
    else:
        return -2.0

