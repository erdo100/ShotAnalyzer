import numpy as np

def angle_vector(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the angle between two vectors in degrees, with the same
    behavior as the MATLAB version.
    
    Args:
        a: First vector (numpy array)
        b: Second vector (numpy array)
        
    Returns:
        float: Angle in degrees, or -1/-2 for special cases
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a > 0 and norm_b > 0:
        # Both vectors have non-zero length
        dot_product = np.dot(a, b)
        angle_rad = np.arccos(dot_product / (norm_a * norm_b))
        return np.degrees(angle_rad)
    elif norm_a > 0 or norm_b > 0:
        # Only one vector has non-zero length
        return -1.0
    else:
        # Both vectors are zero-length
        return -2.0