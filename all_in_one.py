import numpy as np
import json
import os
import pandas as pd
import copy
import warnings # Used to suppress potential division-by-zero warnings if dt is zero

# ===== Functions to be provided later (placeholders) =====
def extract_events_start(SA, param):
    print("Placeholder: extract_events_start called")
    # Add implementation later
    pass
# ==========================================================

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


def read_gamefile(filepath):
    """
    Reads game data from a JSON file, extracts relevant information,
    and structures it for analysis.

    Args:
        filepath (str): Path to the JSON game file.

    Returns:
        dict: A dictionary (SA) containing 'Shot' data (list of dicts)
              and 'Table' data (pandas DataFrame), or None if no valid
              shot data is found.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

    # Scaling factors
    tableX = 2840.0
    tableY = 1420.0

    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(filepath))[0]

    SA = {'Shot': [], 'Table': None}
    check = 0 # Flag to check if any valid shot data was processed

    # Lists to build the DataFrame columns
    Selected_list = []
    Filename_list = []
    GameType_list = []
    Player1_list = []
    Player2_list = []
    Set_list = []
    ShotID_list = []
    CurrentInning_list = []
    CurrentSeries_list = []
    CurrentTotalPoints_list = []
    Point_list = []
    Player_list = []
    ErrorID_list = []
    ErrorText_list = []
    Interpreted_list = []
    Mirrored_list = []

    num_shots_processed = 0

    if 'Match' not in data or 'Sets' not in data['Match']:
        print(f"Warning: 'Match' or 'Sets' not found in the JSON structure for {filepath}")
        return None

    sets_data = data['Match']['Sets']
    if not isinstance(sets_data, list):
        print(f"Warning: 'Sets' is not a list in {filepath}")
        return None

    for seti, current_set in enumerate(sets_data):
        # Handle potential inconsistencies where Entries might be a list of lists/dicts
        # The MATLAB code seems to handle a case where Entries is a cell array containing a single struct.
        # In JSON, this might appear as a list containing a single object.
        # This Python code assumes 'Entries' is expected to be a list of shot entry dictionaries.
        if isinstance(current_set.get('Entries'), list) and len(current_set['Entries']) > 0:
            # The MATLAB code had a peculiar block:
            # if iscell(json.Match.Sets(seti).Entries) ... end
            # This block copied *only the first element* of the cell array into a new structure array.
            # Translating this literally: if the first element is itself a list/dict containing 'PathTrackingId',
            # maybe it implies the structure is nested differently than expected?
            # Let's assume the standard structure is a list of dictionaries directly under 'Entries'.
            # If you encounter files where this MATLAB block was necessary, the JSON structure needs closer examination.
            pass # Assuming standard structure, no special handling needed like the MATLAB cell check

        entries = current_set.get('Entries')
        if not isinstance(entries, list):
            # print(f"Warning: 'Entries' is not a list or is missing in set {seti+1} for {filepath}")
            continue # Skip this set if entries are not as expected

        for entryi, current_entry in enumerate(entries):
            if not isinstance(current_entry, dict):
                # print(f"Warning: Entry {entryi+1} in set {seti+1} is not a dictionary for {filepath}")
                continue # Skip malformed entry

            path_tracking_id = current_entry.get('PathTrackingId')
            # Ensure PathTrackingId is a valid number and > 0
            if isinstance(path_tracking_id, (int, float)) and path_tracking_id > 0:

                # Check for necessary nested structures
                if 'Scoring' not in current_entry or 'PathTracking' not in current_entry or \
                   'DataSets' not in current_entry['PathTracking'] or \
                   len(current_entry['PathTracking']['DataSets']) != 3:
                    # print(f"Warning: Skipping entry {entryi+1} in set {seti+1} due to missing data structures.")
                    continue

                num_shots_processed += 1
                si = num_shots_processed # Use 1-based index for conceptual clarity if needed, but list index is si-1

                # Append data for the table
                Selected_list.append(False)
                Filename_list.append(filename)
                GameType_list.append(data['Match'].get('GameType', 'Unknown'))
                player1_name = data.get('Player1', 'Player1')
                player2_name = data.get('Player2', 'Player2')
                Player1_list.append(player1_name)
                Player2_list.append(player2_name)
                Set_list.append(seti + 1) # Store 1-based set index
                ShotID_list.append(path_tracking_id)

                scoring_data = current_entry['Scoring']
                CurrentInning_list.append(scoring_data.get('CurrentInning'))
                CurrentSeries_list.append(scoring_data.get('CurrentSeries'))
                CurrentTotalPoints_list.append(scoring_data.get('CurrentTotalPoints'))
                Point_list.append(scoring_data.get('EntryType')) # 'Point' in MATLAB was EntryType

                playernum = scoring_data.get('Player')
                if playernum == 1:
                    Player_list.append(player1_name)
                elif playernum == 2:
                    Player_list.append(player2_name)
                else:
                    Player_list.append(f"UnknownPlayer_{playernum}")

                ErrorID_list.append(-1) # Initial state
                ErrorText_list.append('only read in')
                Interpreted_list.append(0)
                Mirrored_list.append(0) # Assuming 0 initially, MATLAB code had this

                shot_data = {'Route': [{}, {}, {}], 'Route0': [{}, {}, {}], 'hit': 0}
                valid_route_data = True
                for bi in range(3): # For each ball (0, 1, 2)
                    coords = current_entry['PathTracking']['DataSets'][bi].get('Coords')
                    if not isinstance(coords, list):
                        # print(f"Warning: Coords missing or not a list for ball {bi+1}, shot {si}")
                        valid_route_data = False
                        break # Stop processing this shot if data is bad

                    clen = len(coords)
                    t_list, x_list, y_list = [], [], []

                    for ci in range(clen):
                        coord_point = coords[ci]
                        if not isinstance(coord_point, dict) or \
                           'DeltaT_500us' not in coord_point or \
                           'X' not in coord_point or 'Y' not in coord_point:
                            # print(f"Warning: Invalid coord point {ci+1} for ball {bi+1}, shot {si}")
                            valid_route_data = False
                            break # Stop processing this ball

                        t_list.append(coord_point['DeltaT_500us'] * 0.0005)
                        x_list.append(coord_point['X'] * tableX)
                        y_list.append(coord_point['Y'] * tableY)

                    if not valid_route_data: break # Stop processing this shot

                    # Store route data as numpy arrays
                    current_route = {
                        't': np.array(t_list, dtype=float),
                        'x': np.array(x_list, dtype=float),
                        'y': np.array(y_list, dtype=float)
                    }
                    shot_data['Route'][bi] = current_route
                    # Store initial state before modifications in quality check
                    shot_data['Route0'][bi] = copy.deepcopy(current_route)

                if valid_route_data:
                    SA['Shot'].append(shot_data)
                    check = 1 # Mark that we have processed at least one valid shot
                else:
                    # If route data was invalid, we need to remove the corresponding entries
                    # from the table lists that were added before the inner loop.
                    Selected_list.pop()
                    Filename_list.pop()
                    GameType_list.pop()
                    Player1_list.pop()
                    Player2_list.pop()
                    Set_list.pop()
                    ShotID_list.pop()
                    CurrentInning_list.pop()
                    CurrentSeries_list.pop()
                    CurrentTotalPoints_list.pop()
                    Point_list.pop()
                    Player_list.pop()
                    ErrorID_list.pop()
                    ErrorText_list.pop()
                    Interpreted_list.pop()
                    Mirrored_list.pop()
                    num_shots_processed -= 1 # Decrement counter as this shot is invalid


    if check:
        # Create the Pandas DataFrame
        df_data = {
            'Selected': Selected_list,
            'ShotID': ShotID_list,
            'Mirrored': Mirrored_list, # From MATLAB code
            'Filename': Filename_list,
            'GameType': GameType_list,
            'Interpreted': Interpreted_list, # From MATLAB code
            'Player': Player_list,
            'ErrorID': ErrorID_list, # From MATLAB code
            'ErrorText': ErrorText_list, # From MATLAB code
            'Set': Set_list,
            'CurrentInning': CurrentInning_list,
            'CurrentSeries': CurrentSeries_list,
            'CurrentTotalPoints': CurrentTotalPoints_list,
            'Point': Point_list # Mapped from EntryType
            # Adding Player1 and Player2 might be useful for context
            #'Player1': Player1_list,
            #'Player2': Player2_list
        }
        SA['Table'] = pd.DataFrame(df_data)
        # Ensure ShotID is integer if possible
        SA['Table']['ShotID'] = pd.to_numeric(SA['Table']['ShotID'], errors='coerce').fillna(0).astype(int)
        print(f"Successfully read {len(SA['Table'])} shots from {filepath}")
        return SA
    else:
        print(f"No valid shot data found or processed in {filepath}")
        return None

def extract_dataquality_start(SA, param):
    """
    Performs data quality checks and corrections on shot data.

    Modifies SA['Shot'][si]['Route'] and updates SA['Table'] with error codes/text.

    Args:
        SA (dict): The main data structure containing 'Shot' and 'Table'.
        param (dict): Dictionary of parameters for table dimensions, tolerances, etc.
    """
    print(f'start ({os.path.basename(__file__)})') # Using file name

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
            # Reset route data from original Route0 before applying checks
            # Ensure Route0 exists and has the correct structure
            if 'Route0' not in SA['Shot'][si] or len(SA['Shot'][si]['Route0']) != 3:
                 print(f"Warning: Route0 missing or invalid for shot index {si}. Skipping quality check.")
                 SA['Table'].iloc[si, SA['Table'].columns.get_loc('ErrorID')] = 99 # Custom error code
                 SA['Table'].iloc[si, SA['Table'].columns.get_loc('ErrorText')] = 'Internal error: Route0 missing'
                 SA['Table'].iloc[si, SA['Table'].columns.get_loc('Selected')] = True
                 continue
            SA['Shot'][si]['Route'] = copy.deepcopy(SA['Shot'][si]['Route0'])

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

    print(f'done ({os.path.basename(__file__)})')

def delete_selected_shots(SA):
    """
    Deletes shots marked as 'Selected' (True) from SA['Table'] and SA['Shot'].

    Checks for consistency between the table and shot list lengths.
    Modifies the SA dictionary in place and resets the DataFrame index.

    Args:
        SA (dict): The main data structure containing 'Shot' list and 'Table' DataFrame.

    Returns:
        dict: The modified SA dictionary. Returns the original SA if errors occur.
    """
    print("Attempting to delete selected shots...")

    # --- Input Validation ---
    if SA is None or 'Table' not in SA or SA['Table'] is None or 'Shot' not in SA:
        print("Error: SA structure is invalid or empty. Cannot delete shots.")
        return SA
    if not isinstance(SA['Table'], pd.DataFrame):
         print("Error: SA['Table'] is not a Pandas DataFrame.")
         return SA
    if 'Selected' not in SA['Table'].columns:
        print("Error: 'Selected' column not found in SA['Table']. Cannot determine which shots to delete.")
        return SA
    # Critical Check: Ensure table rows and shot list items correspond before deletion
    if len(SA['Table']) != len(SA['Shot']):
        print(f"CRITICAL Error: Mismatch between number of rows in Table ({len(SA['Table'])}) "
              f"and entries in Shot ({len(SA['Shot'])}). Deletion aborted for data integrity.")
        return SA
    # --- End Validation ---

    initial_shot_count = len(SA['Table'])
    if initial_shot_count == 0:
         print("Table is empty. No shots to delete.")
         return SA

    # Find the original indices of the rows/shots to KEEP
    # We use the DataFrame's index directly. Assumes it aligns 1:1 with SA['Shot'] list.
    indices_to_keep = SA['Table'].index[SA['Table']['Selected'] == False].tolist()

    shots_to_delete = initial_shot_count - len(indices_to_keep)

    if shots_to_delete == 0:
        print("No shots marked for deletion ('Selected' is False for all).")
        return SA
    elif shots_to_delete == initial_shot_count:
         print("All shots are marked for deletion. Clearing SA['Table'] and SA['Shot'].")
         # Create an empty DataFrame with the same columns
         SA['Table'] = pd.DataFrame(columns=SA['Table'].columns)
         SA['Shot'] = []
         return SA
    else:
        print(f"Deleting {shots_to_delete} selected shots...")

        # --- Perform Deletion ---
        # 1. Create the new list of shot data using the original indices to keep
        # This relies on the initial length check ensuring indices are valid for SA['Shot']
        try:
            # List comprehension is efficient for building the new list
            new_shot_list = [SA['Shot'][i] for i in indices_to_keep]
        except IndexError:
             print(f"CRITICAL Error: Index mismatch during SA['Shot'] reconstruction. "
                   f"Max index to keep: {max(indices_to_keep) if indices_to_keep else 'N/A'}, "
                   f"Shot list length: {len(SA['Shot'])}. Aborting deletion.")
             # This error suggests the initial length check might have been insufficient
             # or something modified SA['Shot'] unexpectedly.
             return SA

        # 2. Filter the DataFrame to keep only the corresponding rows
        # Use .loc with the list of indices to keep. This preserves the selection.
        SA['Table'] = SA['Table'].loc[indices_to_keep]

        # 3. Replace the old shot list with the new one
        SA['Shot'] = new_shot_list

        # 4. Reset the DataFrame index to be sequential (0, 1, 2, ...)
        # drop=True prevents the old index from being added as a new column.
        SA['Table'] = SA['Table'].reset_index(drop=True)

        # --- Verification ---
        if len(SA['Table']) == len(SA['Shot']) == len(indices_to_keep):
             print(f"Deletion successful. Remaining shots: {len(SA['Table'])}.")
        else:
            # This indicates a major internal inconsistency
             print(f"CRITICAL Error after deletion: Table length ({len(SA['Table'])}) "
                   f"and Shot list length ({len(SA['Shot'])}) do not match the expected count ({len(indices_to_keep)}).")
             # Depending on requirements, might want to revert changes or flag heavily.

        return SA


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



# Translated function from Extract_b1b2b3_start.m
# Needs the main data structure SA (containing 'Shot' list and 'Table' DataFrame)
# and param dictionary (though param seems unused in the MATLAB version of this specific function)
def extract_b1b2b3_start(SA, param=None): # param added for consistency, but unused here
    """
    Iterates through shots, determines the B1B2B3 order using extract_b1b2b3,
    and updates the SA['Table'].

    Args:
        SA (dict): The main data structure containing 'Shot' list and 'Table' DataFrame.
        param (dict): Parameters dictionary (unused in this function's core logic).
    """
    print(f'start ({os.path.basename(__file__)} calling extract_b1b2b3)') # Indicate function start

    if SA is None or 'Table' not in SA or SA['Table'] is None or 'Shot' not in SA:
        print("Error: SA structure is invalid or empty.")
        return

    num_shots = len(SA['Table'])
    if num_shots == 0:
        print("No shots to process for B1B2B3 extraction.")
        return

    if num_shots != len(SA['Shot']):
         print(f"Warning: Mismatch between Table rows ({num_shots}) and Shot entries ({len(SA['Shot'])}).")
         num_shots = min(num_shots, len(SA['Shot'])) # Process only matching entries

    varname = 'B1B2B3' # Column name to store the result in the table
    if varname not in SA['Table'].columns:
        # Add the column if it doesn't exist, initialize with None or empty string
        SA['Table'][varname] = None # Or np.nan, or ''

    selected_count = 0
    processed_count = 0


    # Iterate through shots using the DataFrame index
    for si, current_shot_id in enumerate(SA['Table']['ShotID']):
        print(f"Processing shot index {si} (ShotID: {current_shot_id})...")
        try:
            # Check if the shot has already been interpreted (skip if so)
            if SA['Table'].iloc[si]['Interpreted'] == 0:
                processed_count += 1
                # Call the function to determine B1B2B3 order
                current_shot_data = SA['Shot'][si]
                b1b2b3_result, err_info = extract_b1b2b3(current_shot_data)

                if err_info['code'] is not None:
                    print(f"ShotIndex {si}: Error {err_info['code']} - {err_info['text']}")
                # Update the table with the result
                SA['Table'].loc[SA['Table'].index[si], varname] = b1b2b3_result

                # Update error info and selection status if an error occurred
                if err_info['code'] is not None:
                    SA['Table'].loc[SA['Table'].index[si], 'ErrorID'] = err_info['code']
                    SA['Table'].loc[SA['Table'].index[si], 'ErrorText'] = err_info['text']
                    SA['Table'].loc[SA['Table'].index[si], 'Selected'] = True
                    selected_count += 1
                    # Optional: print error message like in MATLAB
                    # print(f"ShotIndex {si}: {err_info['text']}")
                else:
                     # If calculation was successful and no previous error, ensure ErrorID/Text reflect this
                     # This depends on whether other functions might set errors; be careful here.
                     # For now, we only update if extract_b1b2b3 finds an error.
                     pass


        except IndexError:
             print(f"Error: Index {si} out of bounds for SA['Table'] or SA['Shot'] during B1B2B3.")
             # Optionally mark this as an error in the table
             SA['Table'].loc[SA['Table'].index[si], 'ErrorID'] = 98 # Custom error code
             SA['Table'].loc[SA['Table'].index[si], 'ErrorText'] = 'Internal index error during B1B2B3 processing'
             SA['Table'].loc[SA['Table'].index[si], 'Selected'] = True
             selected_count +=1
        except Exception as e:
             print(f"Error processing shot index {si} for B1B2B3: {e}")
             SA['Table'].loc[SA['Table'].index[si], 'ErrorID'] = 97 # Custom error code
             SA['Table'].loc[SA['Table'].index[si], 'ErrorText'] = f'Unexpected error during B1B2B3: {e}'
             SA['Table'].loc[SA['Table'].index[si], 'Selected'] = True
             selected_count +=1


    # Update the total selected count based on the current state of the 'Selected' column
    final_selected_count = SA['Table']['Selected'].sum()

    print(f"Processed {processed_count} uninterpreted shots for B1B2B3.")
    # print(f"{selected_count} shots newly marked as selected due to B1B2B3 errors.")
    print(f'{final_selected_count}/{num_shots} total shots selected (marked with errors/warnings)')

    print("\nRunning deletion of selected shots...")
    delete_selected_shots(SA) # Modify SA in place

    print(f'done ({os.path.basename(__file__)} finished extract_b1b2b3_start)')




def extract_events(SA, si, param):
    # Extract all ball-ball hit events and ball-Cushion hit events
    # Revised version using 0-based indexing throughout and removed plotting.

    # initialize outputs
    err = {'code': None, 'text': ''} # Matching Python context structure

    col = 'WYR'

    # Get B1,B2,B3 from ShotList (using 0-based indices now)
    b1b2b3_str = SA['Table'].iloc[si]['B1B2B3']
    b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(b1b2b3_str) # b1i, b2i, b3i are now 0-based

    # Check if str2num_b1b2b3 returned valid indices
    if b1i is None:
        err['code'] = 3 # Assign a new error code for invalid B1B2B3 string
        err['text'] = f'ERROR: Invalid B1B2B3 string "{b1b2b3_str}" for shot index {si}.'
        print(err['text'])
        # Return empty hit structure? Or handle differently?
        # Let's return an empty list for hit structure consistency.
        empty_hit = [{'with': [], 't': [], 'XPos': [], 'YPos': [], 'Kiss': [],
                      'Point': [], 'Fuchs': [], 'PointDist': [], 'KissDistB1': [],
                      'Tpoint': [], 'Tkiss': [], 'Tready': [], 'TB2hit': [], 'Tfailure': []}
                     for _ in range(3)]
        return empty_hit, err


    # Get the copy from original data
    ball0 = [None] * 3 # Use list for ball0, indices 0, 1, 2
    for bi_idx in range(3): # Python 0-based index
        ball0[bi_idx] = copy.deepcopy(SA['Shot'][si]['Route0'][bi_idx])

    ball = [{} for _ in range(3)] # Use list for ball, indices 0, 1, 2
    for bi_idx in range(3):
        ball[bi_idx]['x'] = np.array([SA['Shot'][si]['Route0'][bi_idx]['x'][0]])
        ball[bi_idx]['y'] = np.array([SA['Shot'][si]['Route0'][bi_idx]['y'][0]])
        ball[bi_idx]['t'] = np.array([0.0])

    # Create common Time points
    t_all = np.concatenate([
        np.ravel(ball0[0].get('t', np.array([]))),
        np.ravel(ball0[1].get('t', np.array([]))),
        np.ravel(ball0[2].get('t', np.array([])))
    ])
    Tall0 = np.unique(t_all)
    Tall0 = Tall0[Tall0 >= 0]
    Tall0 = np.sort(Tall0)

    ti=0

    # discretization of dT
    tvec = np.linspace(0.01, 1, 101)

    ## Scanning all balls
    # Initialize hit structure as a list (index 0, 1, 2)
    hit = [
        {'with': [], 't': [], 'XPos': [], 'YPos': [], 'Kiss': [],
         'Point': [], 'Fuchs': [], 'PointDist': [], 'KissDistB1': [],
         'Tpoint': [], 'Tkiss': [], 'Tready': [], 'TB2hit': [], 'Tfailure': []}
        for _ in range(3)
    ]

    # Initialize hit structure for each ball (using 0-based indices b1i, b2i, b3i)
    hit[b1i]['with'].append('S')
    hit[b1i]['t'].append(0.0)
    hit[b1i]['XPos'].append(ball0[b1i]['x'][0])
    hit[b1i]['YPos'].append(ball0[b1i]['y'][0])
    hit[b1i]['Kiss'].append(0)
    hit[b1i]['Point'].append(0)
    hit[b1i]['Fuchs'].append(0)
    hit[b1i]['PointDist'].append(3000.0)
    hit[b1i]['KissDistB1'].append(3000.0)
    hit[b1i]['Tpoint'].append(3000.0)
    hit[b1i]['Tkiss'].append(3000.0)
    hit[b1i]['Tready'].append(3000.0)
    hit[b1i]['TB2hit'].append(3000.0)
    hit[b1i]['Tfailure'].append(3000.0)

    hit[b2i]['with'].append('-')
    hit[b2i]['t'].append(0.0)
    hit[b2i]['XPos'].append(ball0[b2i]['x'][0])
    hit[b2i]['YPos'].append(ball0[b2i]['y'][0])
    hit[b2i]['Kiss'].append(0)
    hit[b2i]['Point'].append(0)
    hit[b2i]['Fuchs'].append(0)
    hit[b2i]['PointDist'].append(3000.0)
    hit[b2i]['KissDistB1'].append(3000.0)
    hit[b2i]['Tpoint'].append(3000.0)
    hit[b2i]['Tkiss'].append(3000.0)
    hit[b2i]['Tready'].append(3000.0)
    hit[b2i]['TB2hit'].append(3000.0)
    hit[b2i]['Tfailure'].append(3000.0)

    hit[b3i]['with'].append('-')
    hit[b3i]['t'].append(0.0)
    hit[b3i]['XPos'].append(ball0[b3i]['x'][0])
    hit[b3i]['YPos'].append(ball0[b3i]['y'][0])
    hit[b3i]['Kiss'].append(0)
    hit[b3i]['Point'].append(0)
    hit[b3i]['Fuchs'].append(0)
    hit[b3i]['PointDist'].append(3000.0)
    hit[b3i]['KissDistB1'].append(3000.0)
    hit[b3i]['Tpoint'].append(3000.0)
    hit[b3i]['Tkiss'].append(3000.0)
    hit[b3i]['Tready'].append(3000.0)
    hit[b3i]['TB2hit'].append(3000.0)
    hit[b3i]['Tfailure'].append(3000.0)

    # Calculate initial ball velocities
    for bi_idx in range(3): # Python 0-based index
        if len(ball0[bi_idx]['t']) >= 2:
            dt = np.diff(ball0[bi_idx]['t'])
            dt[dt == 0] = np.finfo(float).eps # Replace 0 with a very small number

            vx = np.diff(ball0[bi_idx]['x']) / dt
            vy = np.diff(ball0[bi_idx]['y']) / dt

            ball0[bi_idx]['dt'] = np.append(dt, dt[-1] if len(dt) > 0 else 0)
            ball0[bi_idx]['vx'] = np.append(vx, 0.0)
            ball0[bi_idx]['vy'] = np.append(vy, 0.0)
            v = np.sqrt(ball0[bi_idx]['vx']**2 + ball0[bi_idx]['vy']**2)
            ball0[bi_idx]['v'] = v

        else: # Handle cases with less than 2 points
            ball0[bi_idx]['dt'] = np.array([0.0])
            ball0[bi_idx]['vx'] = np.array([0.0])
            ball0[bi_idx]['vy'] = np.array([0.0])
            ball0[bi_idx]['v']  = np.array([0.0])

    # Set initial ball speeds for B2 and B3 = 0 (using 0-based indices)
    if len(ball0[b2i]['vx']) > 0: ball0[b2i]['vx'][0] = 0.0
    if len(ball0[b2i]['vy']) > 0: ball0[b2i]['vy'][0] = 0.0
    if len(ball0[b2i]['v']) > 0:  ball0[b2i]['v'][0]  = 0.0
    if len(ball0[b3i]['vx']) > 0: ball0[b3i]['vx'][0] = 0.0
    if len(ball0[b3i]['vy']) > 0: ball0[b3i]['vy'][0] = 0.0
    if len(ball0[b3i]['v']) > 0:  ball0[b3i]['v'][0]  = 0.0


    ## Lets start
    do_scan = True # Initialize loop condition
    b = [{} for _ in range(3)] # Initialize list for temporary ball data 'b'

    while do_scan:
        # print(f'length off Tall0: {len(Tall0)}')

        if len(Tall0) < 2: # Need at least two points to get dT
             # print("Warning: Not enough time points in Tall0 to continue scan.") # Removed for less verbosity
             break

        ## Approximate Position of Ball at next time step
        dT = np.diff(Tall0[0:2])[0]
        if dT <= 0: dT = np.finfo(float).eps # Avoid non-positive dT for approximation
        tappr = Tall0[0] + dT * tvec

        for bi_idx in range(3): # Python 0-based index

            if len(ball0[bi_idx]['t']) >= 2:

                if len(ball[bi_idx]['x']) >= 2:
                    ds0 = np.sqrt((ball[bi_idx]['x'][1]-ball[bi_idx]['x'][0])**2 + (ball[bi_idx]['y'][1]-ball[bi_idx]['y'][0])**2)
                    v0 = ds0/dT if dT > 0 else 0
                b[bi_idx]['vt1'] = ball0[bi_idx]['v'][0] if len(ball0[bi_idx]['v']) > 0 else 0

                last_hit_time = hit[bi_idx]['t'][-1] if len(hit[bi_idx]['t']) > 0 else -np.inf
                indices_after_hit = np.where(ball[bi_idx]['t'] >= last_hit_time)[0]

                num_to_take = min(param.get('timax_appr', 5), len(indices_after_hit))
                iprev = indices_after_hit[-num_to_take:] if num_to_take > 0 else np.array([], dtype=int)

                last_hit_with = hit[bi_idx]['with'][-1] if len(hit[bi_idx]['with']) > 0 else None

                if last_hit_with == '-':
                    b[bi_idx]['v1'] = np.array([0.0, 0.0])
                    vx2 = ball0[bi_idx]['vx'][1] if len(ball0[bi_idx]['vx']) > 1 else 0.0
                    vy2 = ball0[bi_idx]['vy'][1] if len(ball0[bi_idx]['vy']) > 1 else 0.0
                    b[bi_idx]['v2'] = np.array([vx2, vy2])
                elif len(iprev) == 1:
                    vx1 = ball0[bi_idx]['vx'][0] if len(ball0[bi_idx]['vx']) > 0 else 0.0
                    vy1 = ball0[bi_idx]['vy'][0] if len(ball0[bi_idx]['vy']) > 0 else 0.0
                    b[bi_idx]['v1'] = np.array([vx1, vy1])
                    vx2 = ball0[bi_idx]['vx'][1] if len(ball0[bi_idx]['vx']) > 1 else 0.0
                    vy2 = ball0[bi_idx]['vy'][1] if len(ball0[bi_idx]['vy']) > 1 else 0.0
                    b[bi_idx]['v2'] = np.array([vx2, vy2])

                elif len(iprev) > 1 :
                    t_prev = ball[bi_idx]['t'][iprev]
                    x_prev = ball[bi_idx]['x'][iprev]
                    y_prev = ball[bi_idx]['y'][iprev]
                    A = np.vstack([t_prev, np.ones(len(t_prev))]).T

                    try:
                       res_vx = np.linalg.lstsq(A, x_prev, rcond=None)
                       CoefVX = res_vx[0]
                       res_vy = np.linalg.lstsq(A, y_prev, rcond=None)
                       CoefVY = res_vy[0]
                       b[bi_idx]['v1'] = np.array([CoefVX[0], CoefVY[0]])
                    except np.linalg.LinAlgError:
                       # print(f"Warning: Least squares failed for ball {bi_idx+1}. Using previous velocity.") # Removed for less verbosity
                       vx1 = ball0[bi_idx]['vx'][0] if len(ball0[bi_idx]['vx']) > 0 else 0.0
                       vy1 = ball0[bi_idx]['vy'][0] if len(ball0[bi_idx]['vy']) > 0 else 0.0
                       b[bi_idx]['v1'] = np.array([vx1, vy1])

                    vx2 = ball0[bi_idx]['vx'][1] if len(ball0[bi_idx]['vx']) > 1 else 0.0
                    vy2 = ball0[bi_idx]['vy'][1] if len(ball0[bi_idx]['vy']) > 1 else 0.0
                    b[bi_idx]['v2'] = np.array([vx2, vy2])
                else: # Handle case where iprev is empty
                    vx1 = ball0[bi_idx]['vx'][0] if len(ball0[bi_idx]['vx']) > 0 else 0.0
                    vy1 = ball0[bi_idx]['vy'][0] if len(ball0[bi_idx]['vy']) > 0 else 0.0
                    b[bi_idx]['v1'] = np.array([vx1, vy1])
                    vx2 = ball0[bi_idx]['vx'][1] if len(ball0[bi_idx]['vx']) > 1 else 0.0
                    vy2 = ball0[bi_idx]['vy'][1] if len(ball0[bi_idx]['vy']) > 1 else 0.0
                    b[bi_idx]['v2'] = np.array([vx2, vy2])

                b[bi_idx]['vt1'] = np.linalg.norm(b[bi_idx]['v1'])
                b[bi_idx]['vt2'] = np.linalg.norm(b[bi_idx]['v2'])

                vt_next_recorded = ball0[bi_idx]['v'][0] if len(ball0[bi_idx]['v']) > 0 else 0.0
                vtnext = max(b[bi_idx]['vt1'], vt_next_recorded)

                norm_v1 = np.linalg.norm(b[bi_idx]['v1'])
                if norm_v1 > np.finfo(float).eps:
                    vnext = b[bi_idx]['v1'] / norm_v1 * vtnext
                else:
                    vnext = np.array([0.0, 0.0])

                last_x = ball[bi_idx]['x'][-1] if len(ball[bi_idx]['x']) > 0 else 0.0
                last_y = ball[bi_idx]['y'][-1] if len(ball[bi_idx]['y']) > 0 else 0.0
                time_increments = dT * tvec
                b[bi_idx]['xa'] = last_x + vnext[0] * time_increments
                b[bi_idx]['ya'] = last_y + vnext[1] * time_increments

            else: # Handle case where len(ball0[bi_idx]['t']) < 2
                 b[bi_idx]['v1'] = np.array([0.0, 0.0])
                 b[bi_idx]['v2'] = np.array([0.0, 0.0])
                 b[bi_idx]['vt1'] = 0.0
                 b[bi_idx]['vt2'] = 0.0
                 last_x = ball[bi_idx]['x'][-1] if len(ball[bi_idx]['x']) > 0 else 0.0
                 last_y = ball[bi_idx]['y'][-1] if len(ball[bi_idx]['y']) > 0 else 0.0
                 b[bi_idx]['xa'] = np.full_like(tvec, last_x)
                 b[bi_idx]['ya'] = np.full_like(tvec, last_y)

        # print(f"calculate angle change and ball ball distance...")
        ## Calculate Ball trajectory angle change
        for bi_idx in range(3):
             v1 = b[bi_idx].get('v1', np.array([0.0, 0.0]))
             v2 = b[bi_idx].get('v2', np.array([0.0, 0.0]))
             b[bi_idx]['a12'] = angle_vector(v1, v2)


        ## Calculate the Ball Ball Distance
        BB = np.array([[1, 2], [1, 3], [2, 3]]) # Roles (1st, 2nd, 3rd ball)

        d = [{} for _ in range(3)] # List for distances, index matches BB rows
        for bbi in range(3): # Iterate through BB pairs
            # Map role (1, 2, 3) to 0-based ball index ('W'=0, 'Y'=1, 'R'=2) using b1b2b3
            idx1 = b1b2b3[BB[bbi, 0]-1] # 0-based index of first ball in pair
            idx2 = b1b2b3[BB[bbi, 1]-1] # 0-based index of second ball in pair

            xa1 = b[idx1].get('xa', np.array([np.nan]))
            ya1 = b[idx1].get('ya', np.array([np.nan]))
            xa2 = b[idx2].get('xa', np.array([np.nan]))
            ya2 = b[idx2].get('ya', np.array([np.nan]))

            if xa1.shape == xa2.shape and ya1.shape == ya2.shape:
                 d[bbi]['BB'] = np.sqrt((xa1 - xa2)**2 + (ya1 - ya2)**2) - 2 * param.get('ballR', 0)
            else:
                 # print(f"Warning: Shape mismatch for distance calculation pair {bbi}") # Removed for less verbosity
                 d[bbi]['BB'] = np.full_like(tvec, np.nan)

        # print(f"calculate ball-cushion distance...")
        ## Calculate Cushion distance
        for bi_idx in range(3):
            xa = b[bi_idx].get('xa', np.array([np.nan]))
            ya = b[bi_idx].get('ya', np.array([np.nan]))
            table_size = param.get('size', [np.nan, np.nan])
            ball_radius = param.get('ballR', 0)

            b[bi_idx]['cd'] = np.full((len(tvec), 4), np.nan)
            b[bi_idx]['cd'][:,0] = ya - ball_radius # Bottom cushion distance
            b[bi_idx]['cd'][:,1] = table_size[1] - ball_radius - xa # Right cushion distance
            b[bi_idx]['cd'][:,2] = table_size[0] - ball_radius - ya # Top cushion distance
            b[bi_idx]['cd'][:,3] = xa - ball_radius # Left cushion distance



        #####################################################
        # Evaluate cushion hit
        # REFERENZ TO MATLAB LINE 229


        hitlist = [] # Initialize as list of lists
        lasthitball_idx = -1 # Use 0-based index, -1 for none

        # print(f"evaluate ball-cushion hits...")
        ## Evaluate cushion hit
        for bi_idx in range(3): # Python 0-based index
            for cii in range(4): # Python 0-based index (0=Bottom, 1=Right, 2=Top, 3=Left)

                cd_col = b[bi_idx].get('cd', np.full((len(tvec), 4), np.nan))[:, cii]
                a12_val = b[bi_idx].get('a12', np.nan)
                v1_val = b[bi_idx].get('v1', np.array([0.0, 0.0]))

                checkdist = np.any(cd_col <= 0)
                checkangle = a12_val > 1 or a12_val == -1
                velx = v1_val[0]
                vely = v1_val[1]

                checkcush = False
                tc = np.nan
                cushx = np.nan
                cushy = np.nan

                if checkdist and checkangle:
                    interp_possible = len(tappr) == len(cd_col) and np.any(np.isfinite(tappr)) and np.any(np.isfinite(cd_col))

                    if vely < 0 and cii == 0 and abs(v1_val[1]) > np.finfo(float).eps: # bottom cushion
                        if interp_possible:
                             try:
                                 tc = np.interp(0, cd_col, tappr)
                                 if tc >= tappr[0] and tc <= tappr[-1]:
                                     checkcush = True
                                     cushx = np.interp(tc, tappr, b[bi_idx].get('xa', np.full_like(tappr, np.nan)))
                                     cushy = param.get('ballR', 0)
                                 else: tc = np.nan
                             except Exception: tc = np.nan
                        else: tc = np.nan

                    elif velx > 0 and cii == 1 and abs(v1_val[0]) > np.finfo(float).eps: # right cushion
                         if interp_possible:
                             try:
                                 tc = np.interp(0, cd_col, tappr)
                                 if tc >= tappr[0] and tc <= tappr[-1]:
                                     checkcush = True
                                     cushx = param.get('size', [np.nan, np.nan])[1] - param.get('ballR', 0)
                                     cushy = np.interp(tc, tappr, b[bi_idx].get('ya', np.full_like(tappr, np.nan)))
                                 else: tc = np.nan
                             except Exception: tc = np.nan
                         else: tc = np.nan

                    elif vely > 0 and cii == 2 and abs(v1_val[1]) > np.finfo(float).eps: # top Cushion
                         if interp_possible:
                             try:
                                 tc = np.interp(0, cd_col, tappr)
                                 if tc >= tappr[0] and tc <= tappr[-1]:
                                     checkcush = True
                                     cushy = param.get('size', [np.nan, np.nan])[0] - param.get('ballR', 0)
                                     cushx = np.interp(tc, tappr, b[bi_idx].get('xa', np.full_like(tappr, np.nan)))
                                 else: tc = np.nan
                             except Exception: tc = np.nan
                         else: tc = np.nan

                    elif velx < 0 and cii == 3 and abs(v1_val[0]) > np.finfo(float).eps: # left Cushion
                         if interp_possible:
                             try:
                                 tc = np.interp(0, cd_col, tappr)
                                 if tc >= tappr[0] and tc <= tappr[-1]:
                                     checkcush = True
                                     cushx = param.get('ballR', 0)
                                     cushy = np.interp(tc, tappr, b[bi_idx].get('ya', np.full_like(tappr, np.nan)))
                                 else: tc = np.nan
                             except Exception: tc = np.nan
                         else: tc = np.nan

                if checkcush and not np.isnan(tc):
                    current_bi_idx = bi_idx  # 0-based ball index
                    cushion_id = cii     # 0-based cushion index

                    hitlist.append([
                        tc,            # What Time
                        current_bi_idx,# Which Ball (0-based index)
                        2,             # Contact with 1=Ball, 2=cushion
                        cushion_id,    # Contact with ID (Cushion No. 0-3)
                        cushx,         # Contact location X
                        cushy          # Contact location Y
                    ])


        #####################################################
        # Evaluate ball-ball hit
        # REFERENZ TO MATLAB LINE 285


        # print(f"evaluate ball-ball hits...")
        ## Evaluate Ball-Ball hit
        for bbi in range(3): # Index for BB pairs and 'd' list
            # Map role (1, 2, 3) to 0-based ball index ('W'=0, 'Y'=1, 'R'=2)
            ball1_idx = b1b2b3[BB[bbi, 0]-1] # 0-based index
            ball2_idx = b1b2b3[BB[bbi, 1]-1] # 0-based index


            # Check whether distance has values smaller and bigger 0
            # But take care, B1 can go through B2. Therefore use transition
            # positiv gap ==> negative gap.

            dist_bb = d[bbi].get('BB', np.array([np.nan]))

            if not isinstance(dist_bb, np.ndarray) or np.all(np.isnan(dist_bb)):
                 continue

            checkdist = False
            
            print(f"min dist_bb: {np.min(dist_bb)}")
            if np.any(dist_bb <= 0) and np.any(dist_bb > 0) and \
                (dist_bb[0] >= 0 and dist_bb[-1] < 0) or \
                (dist_bb[0] < 0 and dist_bb[-1] >= 0):
                # Balls are going to be in contact or are already in contact. Previous contact not detected
                checkdist = True
                tc = np.interp(0, dist_bb, tappr)

            elif np.any(dist_bb <= 0) and np.any(dist_bb > 0):
                # here the ball-ball contact is going through the balls
                checkdist = True
                decreasing_indices = np.where(np.diff(dist_bb) <= 0)[0]

                valid_indices = decreasing_indices[decreasing_indices < len(dist_bb)-1]
                indices_to_use = np.union1d(valid_indices, valid_indices + 1)
                indices_to_use = indices_to_use[indices_to_use < len(dist_bb)]

                tc = np.interp(0, dist_bb[indices_to_use], tappr[indices_to_use])



            a12_b1 = b[ball1_idx].get('a12', np.nan)
            a12_b2 = b[ball2_idx].get('a12', np.nan)
            vt1_b1 = b[ball1_idx].get('vt1', 0.0)
            vt2_b1 = b[ball1_idx].get('vt2', 0.0)
            vt1_b2 = b[ball2_idx].get('vt1', 0.0)
            vt2_b2 = b[ball2_idx].get('vt2', 0.0)

            checkangleb1 = a12_b1 > 10 or a12_b1 == -1
            checkangleb2 = a12_b2 > 10 or a12_b2 == -1

            if abs(vt1_b1) > np.finfo(float).eps or abs(vt2_b1) > np.finfo(float).eps:
                 checkvelb1 = abs(vt2_b1 - vt1_b1) > 50
            else: checkvelb1 = False

            if abs(vt1_b2) > np.finfo(float).eps or abs(vt2_b2) > np.finfo(float).eps:
                 checkvelb2 = abs(vt2_b2 - vt1_b2) > 50
            else: checkvelb2 = False

            # Use 0-based indices ball1_idx, ball2_idx for accessing 'hit' list
            last_hit_t_b1 = hit[ball1_idx]['t'][-1] if len(hit[ball1_idx]['t']) > 0 else -np.inf
            last_hit_t_b2 = hit[ball2_idx]['t'][-1] if len(hit[ball2_idx]['t']) > 0 else -np.inf

            if not np.isnan(tc) and tc >= last_hit_t_b1 + 0.01 and tc >= last_hit_t_b2 + 0.01:
                 checkdouble = True
            else: checkdouble = False

            if checkdouble and checkdist and not np.isnan(tc):
                 xa_b1 = b[ball1_idx].get('xa', np.full_like(tappr, np.nan))
                 ya_b1 = b[ball1_idx].get('ya', np.full_like(tappr, np.nan))
                 xa_b2 = b[ball2_idx].get('xa', np.full_like(tappr, np.nan))
                 ya_b2 = b[ball2_idx].get('ya', np.full_like(tappr, np.nan))

                 interp_possible_b1 = len(tappr) == len(xa_b1) and len(tappr) == len(ya_b1)
                 interp_possible_b2 = len(tappr) == len(xa_b2) and len(tappr) == len(ya_b2)

                 if interp_possible_b1 and interp_possible_b2:
                     try:
                         hit_x_b1 = np.interp(tc, tappr, xa_b1)
                         hit_y_b1 = np.interp(tc, tappr, ya_b1)
                         hit_x_b2 = np.interp(tc, tappr, xa_b2)
                         hit_y_b2 = np.interp(tc, tappr, ya_b2)

                         # Add hits using 0-based ball indices
                         # Contact with ID is the *other* ball's 0-based index
                         hitlist.append([ tc, ball1_idx, 1, ball2_idx, hit_x_b1, hit_y_b1 ])
                         hitlist.append([ tc, ball2_idx, 1, ball1_idx, hit_x_b2, hit_y_b2 ])

                     except Exception:
                         # print(f"Interpolation failed for ball-ball hit between {ball1_idx} and {ball2_idx}: {e}") # Removed for less verbosity
                         pass
                 # else: print(f"Warning: Cannot interpolate ball-ball hit, input array mismatch for balls {ball1_idx}, {ball2_idx}.") # Removed


        ## When Just before the Ball-Ball hit velocity is too small...
        # print(f"condition when ball hit velocity is small")
        len_hit_b2i = len(hit[b2i]['t'])
        len_t_b1i = len(ball0[b1i]['t'])
        len_t_b2i = len(ball0[b2i]['t'])
        t2_b1i = ball0[b1i]['t'][1] if len_t_b1i > 1 else np.inf
        t2_b2i = ball0[b2i]['t'][1] if len_t_b2i > 1 else np.inf
        x1_b2i = ball0[b2i]['x'][0] if len_t_b2i > 0 else np.nan
        x2_b2i = ball0[b2i]['x'][1] if len_t_b2i > 1 else np.nan
        y1_b2i = ball0[b2i]['y'][0] if len_t_b2i > 0 else np.nan
        y2_b2i = ball0[b2i]['y'][1] if len_t_b2i > 1 else np.nan

        if ti > 0 and len_hit_b2i == 1 and not hitlist and t2_b1i >= t2_b2i and \
           (x1_b2i != x2_b2i or y1_b2i != y2_b2i):

            # Find the bbi index corresponding to B1 role hitting B2 role
            bbi = -1
            for i in range(len(BB)):
                role1 = BB[i, 0] # 1, 2, or 3
                role2 = BB[i, 1]
                # Check if the ball indices corresponding to these roles match b1i and b2i
                if b1b2b3[role1-1] == b1i and b1b2b3[role2-1] == b2i:
                     bbi = i
                     break

            if bbi != -1:
                dist_bb = d[bbi].get('BB', np.array([np.nan]))
                if isinstance(dist_bb, np.ndarray) and not np.all(np.isnan(dist_bb)) and len(dist_bb) == len(tappr):
                     try:
                          tc = np.interp(0, dist_bb, tappr)
                          if not (tappr[0] <= tc <= tappr[-1]): tc = np.nan
                     except Exception: tc = np.nan

                     if np.isnan(tc) or tc < 0:
                          if not np.all(np.isnan(dist_bb)):
                               min_dist_idx = np.nanargmin(dist_bb)
                               tc = tappr[min_dist_idx]
                          else: tc = np.nan

                     if not np.isnan(tc):
                         xa_b1i = b[b1i].get('xa', np.full_like(tappr, np.nan))
                         ya_b1i = b[b1i].get('ya', np.full_like(tappr, np.nan))
                         interp_possible_b1 = len(tappr) == len(xa_b1i) and len(tappr) == len(ya_b1i)

                         if interp_possible_b1:
                              try:
                                  hit_x_b1 = np.interp(tc, tappr, xa_b1i)
                                  hit_y_b1 = np.interp(tc, tappr, ya_b1i)
                                  hit_x_b2 = x1_b2i
                                  hit_y_b2 = y1_b2i

                                  # Add hits using 0-based ball indices
                                  hitlist.append([ tc, b1i, 1, b2i, hit_x_b1, hit_y_b1 ])
                                  hitlist.append([ tc, b2i, 1, b1i, hit_x_b2, hit_y_b2 ])
                              except Exception:
                                   # print(f"Interpolation failed for special B1-B2 hit positions: {e}") # Removed
                                   pass
                         # else: print(f"Warning: Cannot interpolate special B1-B2 hit positions.") # Removed
                # else: print("Warning: Distance data invalid for special B1-B2 case.") # Removed
            # else: print("Warning: Could not find BB index for B1 role hitting B2 role.") # Removed


        ## Check first Time step for Ball-Ball hit
        if ti == 0 and not hitlist and t2_b1i == t2_b2i and \
            (x1_b2i != x2_b2i or y1_b2i != y2_b2i):

            vec_b2dir = np.array([x2_b2i - x1_b2i, y2_b2i - y1_b2i])
            norm_vec_b2dir = np.linalg.norm(vec_b2dir)
            if norm_vec_b2dir > np.finfo(float).eps:
                 b1pos2 = np.array([x1_b2i, y1_b2i]) - vec_b2dir / norm_vec_b2dir * param.get('ballR', 0) * 2
                 tc = t2_b1i / 2.0
                 # Add hits using 0-based ball indices
                 hitlist.append([tc, b1i, 1, b2i, b1pos2[0], b1pos2[1]])
                 hitlist.append([tc, b2i, 1, b1i, x1_b2i, y1_b2i])
            # else: print("Warning: B2 direction vector has zero norm in first timestep check.") # Removed

        ####################################################################
        # REFERENZ TO MATLAB LINE 430

        ## Assign new hit event or next timestep in to the ball route history
        # print(f"assign new hit event or next timestep in to the ball route history")
        bi_list_indices = list(range(3)) # Balls without hit (use 0-based index)
        hit_processed = False

        valid_hits = [h for h in hitlist if not np.isnan(h[0])]
        if valid_hits and Tall0[1] >= tc:
            tc = min(h[0] for h in valid_hits)
            

            if Tall0[1] < tc:
                print('warning: hit is after next time step') # Removed
            
            
            hit_processed = True
            balls_hit_at_tc_idx = [h[1] for h in valid_hits if h[0] == tc] # 0-based indices
            for bi_check_idx in range(3):
                hits_on_ball = balls_hit_at_tc_idx.count(bi_check_idx)
                if hits_on_ball > 1:
                    err['code'] = 1
                    err_text = f'ERROR: Ball index {bi_check_idx} has {hits_on_ball} hits at same time {tc}.'
                    err['text'] = err_text if not err['text'] else err['text'] + "; " + err_text
                    print(err_text)
                    # Decide how to handle - maybe return? For now, just log.

            indices_at_tc = [i for i, h in enumerate(valid_hits) if h[0] == tc]

            for hit_idx in indices_at_tc:
                current_hit = valid_hits[hit_idx]
                bi_idx = int(current_hit[1]) # 0-based ball index for this hit

                if bi_idx in bi_list_indices: bi_list_indices.remove(bi_idx)
                lasthitball_idx = bi_idx
                lasthit_t = tc

                Tall0[0] = tc # Update Tall0 to the time of the hit event

                ball0[bi_idx]['t'][0] = tc
                ball0[bi_idx]['x'][0] = current_hit[4]
                ball0[bi_idx]['y'][0] = current_hit[5]

                ball[bi_idx]['t'] = np.append(ball[bi_idx]['t'], tc)
                ball[bi_idx]['x'] = np.append(ball[bi_idx]['x'], current_hit[4])
                ball[bi_idx]['y'] = np.append(ball[bi_idx]['y'], current_hit[5])

                ## Hit data structure update (using 0-based bi_idx)
                hit[bi_idx]['t'].append(current_hit[0])
                contact_type = int(current_hit[2]) # 1=Ball, 2=Cushion
                contact_id = int(current_hit[3])   # Ball(0-based)/Cushion(1-based) ID

                if contact_type == 1: # Ball
                    # contact_id is the 0-based index of the other ball
                    hit[bi_idx]['with'].append(col[contact_id])
                elif contact_type == 2: # Cushion
                    # contact_id is the 1-based cushion number
                    hit[bi_idx]['with'].append(str(contact_id))
                else:
                    hit[bi_idx]['with'].append('?')

                hit[bi_idx]['XPos'].append(current_hit[4])
                hit[bi_idx]['YPos'].append(current_hit[5])



            #####################################################
            # REFERENZ TO MATLAB LINE 536

            ## Assign time to ball without hit event
            for bi_idx_no_hit in bi_list_indices: # Use 0-based index
                x_at_tc = np.nan
                y_at_tc = np.nan
                if len(ball0[bi_idx_no_hit]['t']) > 0 and len(ball0[bi_idx_no_hit]['x']) > 0 and len(ball0[bi_idx_no_hit]['y']) > 0:
                    try:
                        x_at_tc = np.interp(tc, ball0[bi_idx_no_hit]['t'], ball0[bi_idx_no_hit]['x'])
                        y_at_tc = np.interp(tc, ball0[bi_idx_no_hit]['t'], ball0[bi_idx_no_hit]['y'])
                    except Exception: pass # Ignore interpolation errors

                ball[bi_idx_no_hit]['t'] = np.append(ball[bi_idx_no_hit]['t'], tc)
                ball[bi_idx_no_hit]['x'] = np.append(ball[bi_idx_no_hit]['x'], x_at_tc)
                ball[bi_idx_no_hit]['y'] = np.append(ball[bi_idx_no_hit]['y'], y_at_tc)

                if len(ball0[bi_idx_no_hit]['t']) > 0:
                    if b[bi_idx_no_hit].get('vt1', 0) > np.finfo(float).eps:
                        ball0[bi_idx_no_hit]['t'][0] = tc
                        ball0[bi_idx_no_hit]['x'][0] = x_at_tc
                        ball0[bi_idx_no_hit]['y'][0] = y_at_tc


        #####################################################
        # REFERENZ TO MATLAB LINE 559


        else: # if not hit_processed:

            next_t = Tall0[1]
            for bi_idx in range(3):
                x_at_next_t = np.nan
                y_at_next_t = np.nan

                if b[bi_idx].get('vt1', 0) > np.finfo(float).eps:
                    x_at_next_t = np.interp(next_t, ball0[bi_idx]['t'], ball0[bi_idx]['x'])
                    y_at_next_t = np.interp(next_t, ball0[bi_idx]['t'], ball0[bi_idx]['y'])
                            
                else:
                    x_at_next_t = ball[bi_idx]['x'][-1]
                    y_at_next_t = ball[bi_idx]['y'][-1]

                ball[bi_idx]['t'] = np.append(ball[bi_idx]['t'], next_t)
                ball[bi_idx]['x'] = np.append(ball[bi_idx]['x'], x_at_next_t)
                ball[bi_idx]['y'] = np.append(ball[bi_idx]['y'], y_at_next_t)

                if len(ball0[bi_idx]['t']) > 0:
                    indices_to_keep = ball0[bi_idx]['t'] >= next_t
                    ball0[bi_idx]['t'] = ball0[bi_idx]['t'][indices_to_keep]
                    ball0[bi_idx]['x'] = ball0[bi_idx]['x'][indices_to_keep]
                    ball0[bi_idx]['y'] = ball0[bi_idx]['y'][indices_to_keep]

                    if len(ball0[bi_idx]['t']) > 0 and ball0[bi_idx]['t'][0] != next_t:
                        ball0[bi_idx]['t'][0] = next_t
                        ball0[bi_idx]['x'][0] = x_at_next_t
                        ball0[bi_idx]['y'][0] = y_at_next_t
                    elif len(ball0[bi_idx]['t']) == 0 :
                        ball0[bi_idx]['t'] = np.array([next_t])
                        ball0[bi_idx]['x'] = np.array([x_at_next_t])
                        ball0[bi_idx]['y'] = np.array([y_at_next_t])

            Tall0 = Tall0[1:] # Advance Tall0



        # Refined Tall0 Update Logic
        if hit_processed:
            Tall0 = Tall0[Tall0 >= tc]
            if len(Tall0) == 0 or Tall0[0] != tc:
                Tall0 = np.unique(np.insert(Tall0, 0, tc))
        # If no hit, Tall0 was already advanced above


        #####################################################
        # REFERENZ TO MATLAB LINE 592


        ## Update derivatives in ball0
        # print(f"update derivatives in ball0")
        for bi_idx in range(3): # Use 0-based index
            last_hit_with = hit[bi_idx]['with'][-1] if len(hit[bi_idx]['with']) > 0 else '-'
            if last_hit_with != '-':

                 if len(ball0[bi_idx]['t']) > 1:
                      # Data should be sorted by time now due to processing logic
                      dt = np.diff(ball0[bi_idx]['t'])
                      dt[dt == 0] = np.finfo(float).eps

                      vx = np.diff(ball0[bi_idx]['x']) / dt
                      vy = np.diff(ball0[bi_idx]['y']) / dt

                      ball0[bi_idx]['dt'] = np.append(dt, dt[-1] if len(dt) > 0 else 0)
                      ball0[bi_idx]['vx'] = np.append(vx, 0.0)
                      ball0[bi_idx]['vy'] = np.append(vy, 0.0)
                      v = np.sqrt(ball0[bi_idx]['vx']**2 + ball0[bi_idx]['vy']**2)
                      ball0[bi_idx]['v'] = v

                 else: # If only one point left, derivatives are zero
                      ball0[bi_idx]['dt'] = np.array([0.0])
                      ball0[bi_idx]['vx'] = np.array([0.0])
                      ball0[bi_idx]['vy'] = np.array([0.0])
                      ball0[bi_idx]['v']  = np.array([0.0])

        # Sanity checks (optional, kept for now)
        for bi_idx in range(3): # Use 0-based index
            for hit_time in hit[bi_idx]['t']:
                 if not np.any(np.isclose(hit_time, ball[bi_idx]['t'])):
                      # print(f'Warning: Hit time {hit_time} for ball index {bi_idx} not found in final trajectory.') # Removed
                      pass

            if len(Tall0) > 0 and len(ball0[bi_idx]['t']) > 0:
                 indices_before = np.where(ball0[bi_idx]['t'] < Tall0[0])[0]
                 if len(indices_before) > 0:
                      # print(f"Internal Warning: Ball index {bi_idx} has times in ball0 before {Tall0[0]}") # Removed
                      pass

        # Check whether time is over
        do_scan = len(Tall0) >= 2 # Need at least 2 points for next dT
        ti=ti+1


    # Assign back the processed shot data to SA['Shot'][si]['Route']
    for bi_idx in range(3):
        # Ensure Route structure exists if it wasn't guaranteed before
        if 'Route' not in SA['Shot'][si] or len(SA['Shot'][si]['Route']) <= bi_idx:
             # Handle error - cannot assign back if structure is missing
             err['code'] = 4 # New error code
             err['text'] = f"ERROR: SA['Shot'][{si}]['Route'][{bi_idx}] structure missing for assignment."
             print(err['text'])
             # Return potentially incomplete hit data?
             return hit, err
        SA['Shot'][si]['Route'][bi_idx]['t'] = ball[bi_idx].get('t', np.array([]))
        SA['Shot'][si]['Route'][bi_idx]['x'] = ball[bi_idx].get('x', np.array([]))
        SA['Shot'][si]['Route'][bi_idx]['y'] = ball[bi_idx].get('y', np.array([]))

    # Convert hit dictionary values from lists to numpy arrays
    for bi_idx in range(3):
         for key in hit[bi_idx]:
              hit[bi_idx][key] = np.array(hit[bi_idx][key])

    return hit, err




def extract_events_start(SA, param=None): # param added for consistency, but unused here
    """Extract all ball-ball hit events and ball-Cushion hit events
    """
    print(f'start ({os.path.basename(__file__)} calling extract_b1b2b3)') # Indicate function start
    num_shots = len(SA['Table'])
    if num_shots == 0:
        print("No shots to process for B1B2B3 extraction.")
        return

    if num_shots != len(SA['Shot']):
         print(f"Warning: Mismatch between Table rows ({num_shots}) and Shot entries ({len(SA['Shot'])}).")
         num_shots = min(num_shots, len(SA['Shot'])) # Process only matching entries

    err = {'code': None, 'text': ''}

    # Iterate through shots using the DataFrame index
    for si, current_shot_id in enumerate(SA['Table']['ShotID']):
        print(f"Processing shot index {si} (ShotID: {current_shot_id})...")

        b1b2b3_num, b1i, b2i, b3i = str2num_b1b2b3(SA['Table'].iloc[si]['B1B2B3'])
        
        # extract all events
        hit = extract_events(SA, si, param)
        print(f"Hit data extracted for shot index {si}.")


def str2num_b1b2b3(b1b2b3_str):
    base_string = 'WYR'
    b1b2b3_num = [] # To store the resulting indices

    b1b2b3_num = [base_string.index(char) for char in b1b2b3_str]

    # Unpack the list into individual variables
    b1i = b1b2b3_num[0]
    b2i = b1b2b3_num[1]
    b3i = b1b2b3_num[2]

    return b1b2b3_num, b1i, b2i, b3i



# Main execution function (similar to the original script)
def extract_shotdata_cmd(filepath):
    """
    Extract shot data by sequentially calling the required functions.

    Args:
        filepath (str): Path to the game file.
    """

    # Table properties (adjusted for Python syntax)
    param = {
        "ver": "Shot Analyzer v0.43i_Python",
        # Assuming [height, width] based on typical usage, but MATLAB code suggested [Y, X] -> [1420, 2840]
        # Let's stick to MATLAB's apparent [height, width] based on index usage (size(1)=y, size(2)=x)
        "size": [1420, 2840],
        "ballR": 61.5 / 2,
        # 'ballcirc' not directly used in translated functions, kept for reference
        "ballcirc": {
            "x": np.sin(np.linspace(0, 2 * np.pi, 36)) * (61.5 / 2),
            "y": np.cos(np.linspace(0, 2 * np.pi, 36)) * (61.5 / 2),
        },
        "rdiam": 7,
        "cushionwidth": 50,
        "diamdist": 97,
        "framewidth": 147,
        "colors": "wyr",
        "BallOutOfTableDetectionLimit": 30, # Used in commented MATLAB code
        "BallCushionHitDetectionRange": 50, # Not used in translated code
        "BallProjecttoCushionLimit": 10,   # Not used in translated code (clipping used instead)
        "NoDataDistanceDetectionLimit": 600, # Used
        "MaxTravelDistancePerDt": 0.1,      # Not used in translated code
        "MaxVelocity": 12000,               # Used
        "timax_appr": 5,                    # Not used in translated code
    }

    print("Starting shot data extraction...")

    # Step 1: Read game file
    print("Reading game file...")
    SA = read_gamefile(filepath)

    if SA is None:
        print("Failed to read game file or no data found. Aborting.")
        return

    # Step 2: Extract data quality
    print("Extracting data quality...")
    extract_dataquality_start(SA, param)

    # Step 3: Extract B1B2B3 start (Placeholder)
    print("Extracting B1B2B3 start...")
    extract_b1b2b3_start(SA, param)

    # Step 4: Extract events (Placeholder)
    print("Extracting events...")
    extract_events_start(SA, param)

    print("Shot data extraction process completed.")

    # You can now access the processed data in SA
    # Example: print(SA['Table'].head())
    # Example: print(SA['Shot'][0]['Route'][0]['x']) # X coordinates of ball 1, shot 1


# Example Usage:
if __name__ == "__main__":
    # Choose the file path to test
    # filepath = "D:/Programming/Shotdata/JSON/20210704_Ersin_Cemal.json" # Use forward slashes or raw string
    filepath = r"D:\Billard\0_AllDatabase\WebSport\20170906_match_01_Ersin_Cemal.txt" # Use raw string for Windows paths

    # Check if file exists before running
    if os.path.exists(filepath):
        extract_shotdata_cmd(filepath)
    else:
        print(f"Error: Test file not found at {filepath}")

    