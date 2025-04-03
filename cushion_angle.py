import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from angle_vector import angle_vector


def cushion_angle(ci: int, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate angles between ball velocities and cushion normal vectors
    
    Args:
        ci: Cushion index (1=bottom, 2=right, 3=top, 4=left)
        v1: Velocity vector before hit [vx, vy]
        v2: Velocity vector after hit [vx, vy]
        
    Returns:
        numpy.ndarray: [angle_in, angle_out] in degrees
    """
    # Convert to 3D vectors
    v1_3d = np.array([v1[0], v1[1], 0])
    v2_3d = np.array([v2[0], v2[1], 0])
    
    # Get cushion normal vector
    if ci == 1:
        # Bottom cushion (positive y normal)
        b = np.array([0, 1, 0])
    elif ci == 2:
        # Right cushion (negative x normal)
        b = np.array([-1, 0, 0])
    elif ci == 3:
        # Top cushion (negative y normal)
        b = np.array([0, -1, 0])
    elif ci == 4:
        # Left cushion (positive x normal)
        b = np.array([1, 0, 0])
    else:
        raise ValueError("Invalid cushion index (must be 1-4)")
    
    # Calculate angles
    angle_in = angle_vector(v1_3d, -b)
    angle_out = angle_vector(v2_3d, b)
    
    angle = np.array([angle_in, angle_out])
    
    # Check for negative angles based on tangent direction
    if ci == 1:
        # Bottom cushion - check x component
        if np.sign(v1[0]) != np.sign(v2[0]):
            angle[0] = -angle[0]
    elif ci == 2:
        # Right cushion - check y component
        if np.sign(v1[1]) != np.sign(v2[1]):
            angle[0] = -angle[0]
    elif ci == 3:
        # Top cushion - check x component
        if np.sign(v1[0]) != np.sign(v2[0]):
            angle[0] = -angle[0]
    elif ci == 4:
        # Left cushion - check y component
        if np.sign(v1[1]) != np.sign(v2[1]):
            angle[0] = -angle[0]
    
    return angle