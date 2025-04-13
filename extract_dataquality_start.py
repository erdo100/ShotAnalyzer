import os
import copy
import numpy as np
import pandas as pd
import warnings

from delete_selected_shots import delete_selected_shots

def extract_dataquality_start(self):
    """
    Performs data quality checks and corrections on shot data.

    Modifies SA['Shot'][si]['Route'] and updates SA['Table'] with error codes/text.

    Args:
        SA (dict): The main data structure containing 'Shot' and 'Table'.
        param (dict): Dictionary of parameters for table dimensions, tolerances, etc.
    """
    print(f'start ({os.path.basename(__file__)})') # Using file name
    SA = self.SA
    param = self.param

    if SA is None or 'Table' not in SA or SA['Table'] is None or 'Shot' not in SA:
        print("Error: SA structure is invalid or empty.")
        return

    num_shots = len(SA['Table'])
    if num_shots == 0:
        print("No shots to process for data quality.")
        return

    if num_shots != len(SA['Shot']):
         print(f"Warning: Mismatch between number of rows in Table ({num_shots}) and entries in Shot ({len(SA['Shot'])}). Check read_gamefile.")
         # Decide how to handle: stop, or process min(num_shots, len(SA['Shot']))?
         # Let's process based on Table index, assuming Shot list corresponds.
         num_shots = min(num_shots, len(SA['Shot']))


    selected_count_start = SA['Table']['Selected'].sum()

    # Use DataFrame index directly (si represents the index in the DataFrame and the list SA['Shot'])
    for si in range(num_shots): # Iterate using 0-based index
        # Access table data using .loc for safety if index isn't sequential 0..N-1
        # However, since we build it sequentially, direct index 'si' should work.
        # Using .iloc[si] is safer if index isn't guaranteed 0..N-1
        try:
            interpreted_status = SA['Table'].iloc[si]['Interpreted']
        except IndexError:
             print(f"Error: Index {si} out of bounds for SA['Table'] or SA['Shot'].")
             continue # Skip this iteration

        if interpreted_status == 0:

            # Initialize error state for this shot
            err_code = None
            err_text = None
            base_errcode = 100 # Start error codes from 100

            # --- Quality Checks ---

            # Check 1 & 2: Ball data missing (0 or 1 points)
            current_errcode = base_errcode + 1
            for bi in range(3):
                if len(SA['Shot'][si]['Route'][bi]['t']) == 0:
                    err_code = current_errcode
                    err_text = f'Ball data is missing (0 points) (ball {bi+1})'
                    print(f"ShotIndex {si}: {err_text}")
                    SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                    break # Found error, stop checking this block
            if err_code is None:
                current_errcode = base_errcode + 2
                for bi in range(3):
                    if len(SA['Shot'][si]['Route'][bi]['t']) == 1:
                        err_code = current_errcode
                        err_text = f'Ball data is missing (1 point) (ball {bi+1})'
                        print(f"ShotIndex {si}: {err_text}")
                        SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                        break

            # Check 3 & 4: (Duplicate of 1 & 2 in MATLAB) - Check if points remain after potential deletion (which was commented out)
            # This check is useful if points outside the table were deleted (the disabled 'if 0' block in MATLAB)
            # If that block remains disabled, this check might be redundant, but let's keep it for robustness.
            if err_code is None:
                current_errcode = base_errcode + 3
                for bi in range(3):
                    if len(SA['Shot'][si]['Route'][bi]['t']) == 0:
                        err_code = current_errcode
                        err_text = f'Ball data missing after processing (0 points) (ball {bi+1})'
                        print(f"ShotIndex {si}: {err_text}")
                        SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                        break
            if err_code is None:
                current_errcode = base_errcode + 4
                for bi in range(3):
                    if len(SA['Shot'][si]['Route'][bi]['t']) == 1:
                        err_code = current_errcode
                        err_text = f'Ball data missing after processing (1 point) (ball {bi+1})'
                        print(f"ShotIndex {si}: {err_text}")
                        SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                        break

            # Check 5: Project Points slightly outside cushion back onto cushion edge
            if err_code is None:
                # Note: The MATLAB code had 'tol = param.BallProjecttoCushionLimit;' but didn't use it here.
                # It used hardcoded oncushion values based on ballR.
                ballR = param['ballR']
                sizeX = param['size'][0] # Typically width? MATLAB used size(2)
                sizeY = param['size'][1] # Typically height? MATLAB used size(1) -> Check convention! Assuming size = [height, width]
                
                # Assuming param['size'] = [height, width] like typical image/matrix dims
                # If param['size'] = [width, height] -> swap sizeX/sizeY
                # Let's assume param['size'] = [height, width] based on common conventions. 
                # But MATLAB code used size(2) for X limits and size(1) for Y limits, suggesting [Y, X] or [Height, Width]
                # Let's follow the MATLAB indexing: size(2) -> X, size(1) -> Y
                sizeX = param['size'][1] # Width
                sizeY = param['size'][0] # Height
                
                # Define the boundaries where the *center* of the ball should be
                min_x = ballR + 0.1
                max_x = sizeX - ballR - 0.1
                min_y = ballR + 0.1
                max_y = sizeY - ballR - 0.1

                for bi in range(3):
                    x_coords = SA['Shot'][si]['Route'][bi]['x']
                    y_coords = SA['Shot'][si]['Route'][bi]['y']

                    # Clip coordinates to be within the boundaries
                    x_coords = np.clip(x_coords, min_x, max_x)
                    y_coords = np.clip(y_coords, min_y, max_y)

                    SA['Shot'][si]['Route'][bi]['x'] = x_coords
                    SA['Shot'][si]['Route'][bi]['y'] = y_coords


            # Check 6: Initial ball distance check and correction
            if err_code is None:
                current_errcode = base_errcode + 5
                BB = [(0, 1), (0, 2), (1, 2)] # Ball pairs (0-based)
                ballR = param['ballR']
                correction_applied = False
                for b1i, b2i in BB:
                    # Ensure there's at least one point for each ball in the pair
                    if len(SA['Shot'][si]['Route'][b1i]['x']) == 0 or len(SA['Shot'][si]['Route'][b2i]['x']) == 0:
                       continue # Cannot compare if a ball has no data

                    x1, y1 = SA['Shot'][si]['Route'][b1i]['x'][0], SA['Shot'][si]['Route'][b1i]['y'][0]
                    x2, y2 = SA['Shot'][si]['Route'][b2i]['x'][0], SA['Shot'][si]['Route'][b2i]['y'][0]

                    dx, dy = x1 - x2, y1 - y2
                    dist_sq = dx**2 + dy**2
                    min_dist_sq = (2 * ballR)**2
                    
                    # Check for overlap (dist < 2*ballR), allow for small tolerance
                    if dist_sq < min_dist_sq - 1e-6: # Added tolerance for float comparison
                        dist = np.sqrt(dist_sq)
                        overlap = (2 * ballR) - dist
                        correction_applied = True
                        #print(f"ShotIndex {si}: Correcting initial overlap between balls {b1i+1} and {b2i+1}. Overlap: {overlap:.2f}")

                        # Calculate vector from b1 to b2
                        vec_x, vec_y = x2 - x1, y2 - y1

                        # Normalize vector
                        norm = np.sqrt(vec_x**2 + vec_y**2)
                        if norm > 1e-6: # Avoid division by zero
                            unit_vec_x, unit_vec_y = vec_x / norm, vec_y / norm

                            # Move balls apart along the connection line by half the overlap each
                            move_dist = overlap / 2.0
                            SA['Shot'][si]['Route'][b1i]['x'][0] -= unit_vec_x * move_dist
                            SA['Shot'][si]['Route'][b1i]['y'][0] -= unit_vec_y * move_dist
                            SA['Shot'][si]['Route'][b2i]['x'][0] += unit_vec_x * move_dist
                            SA['Shot'][si]['Route'][b2i]['y'][0] += unit_vec_y * move_dist
                        else:
                            # Balls are exactly at the same spot, move them slightly apart arbitrarily (e.g., along x-axis)
                            SA['Shot'][si]['Route'][b1i]['x'][0] -= ballR
                            SA['Shot'][si]['Route'][b2i]['x'][0] += ballR

                # Re-project positions onto cushion after correction, as correction might push them out
                if correction_applied:
                     for bi_reproj in range(3):
                        if len(SA['Shot'][si]['Route'][bi_reproj]['x']) > 0: # Check if ball has data
                            x_coords_reproj = SA['Shot'][si]['Route'][bi_reproj]['x']
                            y_coords_reproj = SA['Shot'][si]['Route'][bi_reproj]['y']
                            x_coords_reproj[0] = np.clip(x_coords_reproj[0], min_x, max_x)
                            y_coords_reproj[0] = np.clip(y_coords_reproj[0], min_y, max_y)
                            SA['Shot'][si]['Route'][bi_reproj]['x'] = x_coords_reproj
                            SA['Shot'][si]['Route'][bi_reproj]['y'] = y_coords_reproj


            # Check 7: Time linearity check and correction
            if err_code is None:
                current_errcode = base_errcode + 6
                for bi in range(3):
                    t = SA['Shot'][si]['Route'][bi]['t']
                    if len(t) < 2: continue # Need at least two points to check diff

                    dt = np.diff(t)
                    ind = np.where(dt <= 0)[0] # Indices where dt is non-positive

                    if len(ind) > 0:
                        print(f"ShotIndex {si}: Try to fix time linearity for ball {bi+1}....")
                        # Complex correction from MATLAB: find all points after index 'idx'
                        # that have a timestamp <= t[idx].
                        delind_set = set()
                        for idx in ind: # idx is the index *before* the non-positive dt
                            problematic_t = t[idx]
                            # Find indices in the rest of the array (from idx+1 onwards)
                            indices_after = np.where(t[idx+1:] <= problematic_t)[0]
                            # Adjust these indices to be relative to the original array 't'
                            absolute_indices = indices_after + (idx + 1)
                            delind_set.update(absolute_indices)

                        if not delind_set: # Should not happen if len(ind)>0, but safety check
                             print(f"ShotIndex {si}: Time linearity issue detected but no points identified for deletion.")
                             continue # Move to next ball


                        delind_list = sorted(list(delind_set))
                        #print(f"   Indices to delete: {delind_list}")

                        # Create a mask to keep elements NOT in delind_list
                        mask = np.ones(len(t), dtype=bool)
                        if max(delind_list) < len(mask):
                            mask[delind_list] = False
                        else:
                            print(f"   Error: Deletion index out of bounds. Max index: {max(delind_list)}, Array len: {len(mask)}")
                            # Handle error - maybe mark shot as bad?
                            err_code = current_errcode
                            err_text = f'Time linearity correction failed (index OOB) (ball {bi+1})'
                            SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                            break # Stop checking this shot


                        t_new = t[mask]

                        # Check if the correction worked
                        if len(t_new) < 2 or np.all(np.diff(t_new) > 0):
                            print(f"ShotIndex {si}: Successfully fixed time linearity for ball {bi+1} :)")
                            SA['Shot'][si]['Route'][bi]['t'] = t_new
                            SA['Shot'][si]['Route'][bi]['x'] = SA['Shot'][si]['Route'][bi]['x'][mask]
                            SA['Shot'][si]['Route'][bi]['y'] = SA['Shot'][si]['Route'][bi]['y'][mask]
                            ind = [] # Reset ind as the issue is fixed
                        else:
                            # Correction failed, mark error
                            print(f"ShotIndex {si}: Could not fix time linearity for ball {bi+1} :(")
                            err_code = current_errcode
                            err_text = f'Non-linear time detected and unfixable (ball {bi+1})'
                            SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                            # No need to store err.region, it was for GUI
                            break # Stop checking balls for this shot

            # Check 8 & 9: Check start and end times consistency
            if err_code is None:
                 t_starts = []
                 t_ends = []
                 valid_times = True
                 for bi_t in range(3):
                     t_vec = SA['Shot'][si]['Route'][bi_t]['t']
                     if len(t_vec) > 0:
                         t_starts.append(t_vec[0])
                         t_ends.append(t_vec[-1])
                     else:
                         valid_times = False # Cannot check if a ball has no data
                         break # No need to check further if one ball is empty

                 if valid_times and len(t_starts) == 3: # Ensure all 3 balls had data
                     current_errcode = base_errcode + 7
                     # Check if all start times are close enough (allow for float precision)
                     if not np.allclose(t_starts, t_starts[0], atol=1e-6):
                         err_code = current_errcode
                         err_text = f'Start times differ between balls ({t_starts})'
                         print(f"ShotIndex {si}: {err_text}")
                         SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True

                     if err_code is None: # Only check end times if start times were ok
                        current_errcode = base_errcode + 8
                        # Check if all end times are close enough
                        if not np.allclose(t_ends, t_ends[0], atol=1e-6):
                            err_code = current_errcode
                            err_text = f'End times differ between balls ({t_ends})'
                            print(f"ShotIndex {si}: {err_text}")
                            SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                 elif not valid_times:
                     # If we couldn't check because a ball was empty, this was already flagged earlier.
                     pass


            # Check 10: Check for point reflections/jumps and remove intermediate point
            # This check looks for patterns like A -> B -> C where A->C is much shorter
            # than A->B + B->C, suggesting B is an erroneous point.
            if err_code is None:
                current_errcode = base_errcode + 9 # MATLAB used errcode + 1 for this block ID
                for bi in range(3):
                    corrected = True # Flag to re-iterate if a point is deleted
                    while corrected:
                         corrected = False
                         t = SA['Shot'][si]['Route'][bi]['t']
                         x = SA['Shot'][si]['Route'][bi]['x']
                         y = SA['Shot'][si]['Route'][bi]['y']
                         n_points = len(t)

                         if n_points < 3: break # Need at least 3 points

                         indices_to_delete = []
                         i = 0
                         while i <= n_points - 3: # Iterate up to the third-to-last point
                            p0 = np.array([x[i], y[i]])
                            p1 = np.array([x[i+1], y[i+1]])
                            p2 = np.array([x[i+2], y[i+2]])

                            # Calculate distances
                            dl0 = np.linalg.norm(p2 - p0) # Dist p0 -> p2
                            dl1 = np.linalg.norm(p1 - p0) # Dist p0 -> p1
                            dl2 = np.linalg.norm(p2 - p1) # Dist p1 -> p2

                            # Conditions from MATLAB: dl0 < dl1*0.5 & dl0 < dl2*0.5 & dl1 > 100
                            # The dl1 > 100 condition seems context specific, maybe minimum segment length?
                            if dl1 > 1e-6 and dl2 > 1e-6: # Avoid division by zero / check valid segments
                                if dl0 < dl1 * 0.5 and dl0 < dl2 * 0.5 and dl1 > 100: # Using 100 from MATLAB
                                     # Mark point i+1 for deletion
                                     indices_to_delete.append(i + 1)
                                     # Skip checking the next segment involving the deleted point
                                     i += 2
                                else:
                                     i += 1
                            else:
                                i += 1

                         if indices_to_delete:
                             print(f"ShotIndex {si}: Removing {len(indices_to_delete)} jump/reflection points for ball {bi+1}.")
                             mask = np.ones(n_points, dtype=bool)
                             mask[indices_to_delete] = False
                             SA['Shot'][si]['Route'][bi]['t'] = t[mask]
                             SA['Shot'][si]['Route'][bi]['x'] = x[mask]
                             SA['Shot'][si]['Route'][bi]['y'] = y[mask]
                             corrected = True # Signal to re-run the check on the modified data

            # Check 11 & 12: Check gaps in tracking (distance and velocity)
            if err_code is None:
                current_errcode = base_errcode + 10 # MATLAB used errcode + 1 for this block ID
                max_dist_limit = param.get('NoDataDistanceDetectionLimit', 600)
                max_vel_limit = param.get('MaxVelocity', 12000)

                for bi in range(3):
                    t = SA['Shot'][si]['Route'][bi]['t']
                    x = SA['Shot'][si]['Route'][bi]['x']
                    y = SA['Shot'][si]['Route'][bi]['y']

                    if len(t) < 2: continue # Need at least two points

                    dx = np.diff(x)
                    dy = np.diff(y)
                    dt = np.diff(t)

                    ds = np.sqrt(dx**2 + dy**2)

                    # Check distance gap
                    if np.any(ds > max_dist_limit):
                        err_code = current_errcode
                        problem_indices = np.where(ds > max_dist_limit)[0]
                        err_text = f'Gap in data too big ({np.max(ds[problem_indices]):.1f} > {max_dist_limit}) (ball {bi+1}, step {problem_indices[0]+1})'
                        print(f"ShotIndex {si}: {err_text}")
                        SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                        break # Stop checking balls for this shot

                    # Check velocity
                    # Suppress division by zero warnings and handle dt=0
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore", category=RuntimeWarning)
                         vabs = np.divide(ds, dt, out=np.zeros_like(ds), where=dt!=0)
                    
                    # Handle cases where dt was zero explicitly - assign infinite velocity? Or flag?
                    if np.any(dt <= 1e-9): # Check for zero or very small dt
                         # Option 1: Assign a very high velocity
                         vabs[dt <= 1e-9] = np.inf 
                         # Option 2: Flag as error directly
                         # err_code = current_errcode + 1 # Or use a specific code
                         # err_text = f'Zero time difference detected (ball {bi+1})'
                         # print(f"ShotIndex {si}: {err_text}")
                         # SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                         # break # Stop checking balls for this shot
                    
                    # Now check velocity limit
                    if np.any(vabs > max_vel_limit):
                        err_code = current_errcode + 1 # Use next code for velocity error
                        problem_indices = np.where(vabs > max_vel_limit)[0]
                        err_text = f'Velocity too high ({np.max(vabs[problem_indices]):.1f} > {max_vel_limit}) (ball {bi+1}, step {problem_indices[0]+1})'
                        print(f"ShotIndex {si}: {err_text}")
                        SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                        break # Stop checking balls for this shot
            
            # Update the error status in the table
            SA['Table'].iloc[si, SA['Table'].columns.get_loc('ErrorID')] = err_code if err_code is not None else 0 # Use 0 for no error found
            SA['Table'].iloc[si, SA['Table'].columns.get_loc('ErrorText')] = err_text if err_text is not None else 'Quality checks passed'

            # --- End of Quality Checks ---

    selected_count_end = SA['Table']['Selected'].sum()
    print(f'{selected_count_end}/{num_shots} shots selected (marked with errors/warnings)')

    print("\nRunning deletion of selected shots...")
    delete_selected_shots(SA) # Modify SA in place

    self.SA = SA # Update the main SA structure with the modified one

    print(f'done ({os.path.basename(__file__)})')