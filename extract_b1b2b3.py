import numpy as np
import os
from math import acos, degrees
from numpy.linalg import norm
from angle_vector import angle_vector


def extract_b1b2b3(shot):
    """
    Translates the MATLAB function Extract_b1b2b3 from Extract_b1b2b3.txt.
    Determines the likely order of balls (Cue, Object, Other) based on initial movement.

    Args:
        shot (dict): Dictionary containing shot data, specifically shot['Route']
                     which is a list of 3 dicts, each with 't', 'x', 'y' numpy arrays.
        angle_vector_func (callable): The function to calculate the angle between vectors.

    Returns:
        tuple: (b1b2b3 (str), err (dict))
               b1b2b3: String representing the assumed ball order ('WYR', 'YWR', etc.)
               err: Dictionary with 'code' and 'text' for errors.
    """

    # initialize outputs
    # b1b2b3num is not directly used for the final output string in MATLAB,
    # but helps determine the order. We'll use indices 0, 1, 2 instead.
    err = {'code': None, 'text': ''}

    col = 'WYR' # Represents Ball 0 (White), Ball 1 (Yellow), Ball 2 (Red)

    # Check if any ball has less than 2 data points
    # MATLAB: length(shot.Route(i).t) < 2 corresponds to len(shot['Route'][i-1]['t']) < 2
    # MATLAB's | is logical OR
    if len(shot['Route'][0]['t']) < 2 or \
       len(shot['Route'][1]['t']) < 2 or \
       len(shot['Route'][2]['t']) < 2:
        # The MATLAB comments seem slightly contradictory/confusing here
        b1b2b3 = 'WYR' # Default order if insufficient data
        err['code'] = 2 # Error code from MATLAB
        # Using os.path.basename(__file__) is the Python equivalent of mfilename
        err['text'] = f'Empty Data ({os.path.basename(__file__)} Extract_b1b2b3.py)' # Adjusted filename
        return b1b2b3, err # Corresponds to MATLAB's return

    # Get the time of the *second* point for each ball (index 1 in Python)
    # MATLAB: shot.Route(i).t(2) corresponds to shot['Route'][i-1]['t'][1]
    times_at_step2 = [shot['Route'][0]['t'][1], shot['Route'][1]['t'][1], shot['Route'][2]['t'][1]]

    # Sort the times and get the original indices (0, 1, 2)
    # MATLAB: [t, b1b2b3num] = sort(...)
    # np.argsort gives the indices that would sort the array
    b1b2b3_num = np.argsort(times_at_step2)
    # t = np.sort(times_at_step2) # Sorted times (variable 't' in MATLAB, not used later)

    # Temporary assumed B1B2B3 order based on the second time step
    # Indices corresponding to the fastest (b1it), second (b2it), third (b3it) moving balls
    b1it = b1b2b3_num[0] # Index of the first ball to move (based on t[1])
    b2it = b1b2b3_num[1] # Index of the second ball to move
    b3it = b1b2b3_num[2] # Index of the third ball to move

    # Check if the fastest moving ball (b1it) has at least 3 points
    # MATLAB: length(shot.Route(b1it).t) >= 3
    if len(shot['Route'][b1it]['t']) >= 3:

        # Find indices for balls b2it and b3it moving *before or at* the time of
        # the 2nd and 3rd steps of the first moving ball (b1it).

        t_b1_step2 = shot['Route'][b1it]['t'][1]
        t_b1_step3 = shot['Route'][b1it]['t'][2]
        t_b2_from_step2 = shot['Route'][b2it]['t'][1:]
        t_b3_from_step2 = shot['Route'][b3it]['t'][1:]

        # Find last index in t_b2_from_step2 <= t_b1_step2
        idx_b2_le_t_b1_s2 = np.where(t_b2_from_step2 <= t_b1_step2)[0]
        tb2i2 = idx_b2_le_t_b1_s2[-1] if len(idx_b2_le_t_b1_s2) > 0 else None

        # Find last index in t_b3_from_step2 <= t_b1_step2
        idx_b3_le_t_b1_s2 = np.where(t_b3_from_step2 <= t_b1_step2)[0]
        tb3i2 = idx_b3_le_t_b1_s2[-1] if len(idx_b3_le_t_b1_s2) > 0 else None

        # Find last index in t_b2_from_step2 <= t_b1_step3
        idx_b2_le_t_b1_s3 = np.where(t_b2_from_step2 <= t_b1_step3)[0]
        tb2i3 = idx_b2_le_t_b1_s3[-1] if len(idx_b2_le_t_b1_s3) > 0 else None

        # Find last index in t_b3_from_step2 <= t_b1_step3
        idx_b3_le_t_b1_s3 = np.where(t_b3_from_step2 <= t_b1_step3)[0]
        tb3i3 = idx_b3_le_t_b1_s3[-1] if len(idx_b3_le_t_b1_s3) > 0 else None

        if tb2i2 is None and tb3i2 is None and tb2i3 is None and tb3i3 is None:
            # Only B1 moved for sure, the initial sort is likely correct.
            b1b2b3 = "".join([col[i] for i in b1b2b3_num])
            return b1b2b3, err

    else:
        b1b2b3 = col
        return b1b2b3, err


    # Check if B1 and B2 moved, but B3 didn't (early on)
    if (tb2i2 is not None or tb2i3 is not None) and \
       (tb3i2 is None and tb3i3 is None):
        # B1 and B2 moved

        # Get initial positions (index 0)
        x_b1_0, y_b1_0 = shot['Route'][b1it]['x'][0], shot['Route'][b1it]['y'][0]
        x_b2_0, y_b2_0 = shot['Route'][b2it]['x'][0], shot['Route'][b2it]['y'][0]

        # Get second positions (index 1)
        x_b1_1, y_b1_1 = shot['Route'][b1it]['x'][1], shot['Route'][b1it]['y'][1]
        x_b2_1, y_b2_1 = shot['Route'][b2it]['x'][1], shot['Route'][b2it]['y'][1]

        # B1-B2 vector (from B1 towards B2)
        vec_b1b2 = np.array([x_b2_0 - x_b1_0, y_b2_0 - y_b1_0])

        # B1 direction vector (initial step)
        vec_b1dir = np.array([x_b1_1 - x_b1_0, y_b1_1 - y_b1_0])

        # B2 direction vector (initial step)
        vec_b2dir = np.array([x_b2_1 - x_b2_0, y_b2_1 - y_b2_0])

        # Calculate angles relative to the B1-B2 connection line
        angle_b1 = angle_vector(vec_b1b2, vec_b1dir) # Angle of B1's movement
        angle_b2 = angle_vector(vec_b1b2, vec_b2dir) # Angle of B2's movement

        if angle_b2 > 90:
             b1b2b3_num = [1 , 0, 2] # Swap first two elements

    # Check if B1 and B3 moved, but B2 didn't (early on)
    # Symmetrical case to the above
    # MATLAB: (isempty(tb2i2) & isempty(tb2i3)) & (~isempty(tb3i2) | ~isempty(tb3i3))
    if (tb2i2 is None and tb2i3 is None) and \
         (tb3i2 is not None or tb3i3 is not None):
         # B1 and B3 moved

        # Get initial positions (index 0)
        x_b1_0, y_b1_0 = shot['Route'][b1it]['x'][0], shot['Route'][b1it]['y'][0]
        x_b3_0, y_b3_0 = shot['Route'][b3it]['x'][0], shot['Route'][b3it]['y'][0]

        # Get second positions (index 1)
        x_b1_1, y_b1_1 = shot['Route'][b1it]['x'][1], shot['Route'][b1it]['y'][1]
        x_b3_1, y_b3_1 = shot['Route'][b3it]['x'][1], shot['Route'][b3it]['y'][1]

        # B1-B3 vector (from B1 towards B3)
        vec_b1b3 = np.array([x_b3_0 - x_b1_0, y_b3_0 - y_b1_0])

        # B1 direction vector (initial step)
        vec_b1dir = np.array([x_b1_1 - x_b1_0, y_b1_1 - y_b1_0])

        # B3 direction vector (initial step)
        vec_b3dir = np.array([x_b3_1 - x_b3_0, y_b3_1 - y_b3_0])

        # Calculate angles relative to the B1-B3 connection line
        angle_b1 = angle_vector(vec_b1b3, vec_b1dir) # Angle of B1's movement
        angle_b3 = angle_vector(vec_b1b3, vec_b3dir) # Angle of B3's movement

        if angle_b3 > 90:
             b1b2b3_num = [1, 2, 0] # Reorder elements

    b1b2b3 = "".join([col[i] for i in b1b2b3_num])

    if (tb2i2 is not None and tb2i3 is not None) and \
         (tb3i2 is not None or tb3i3 is not None):
        # B1, B2 , B3 moved
        err['code'] = 2 # Error code from MATLAB
        # Using os.path.basename(__file__) is the Python equivalent of mfilename
        err['text'] = f'all balls moved at same time  ({os.path.basename(__file__)} Extract_b1b2b3.py)'
        
        return b1b2b3, err # Return the determined order and error status

    # The final b1b2b3 string is determined by the logic above.
    return b1b2b3, err # Return the determined order and error status
