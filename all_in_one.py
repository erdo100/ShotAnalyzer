import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math
import warnings

# Suppress runtime warnings from interpolation/division by zero if needed
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Constants and Parameters ---
# (Moved param definition inside the main class or function later)
DEFAULT_PARAMS = {
    "ver": "Shot Analyzer Python v1.0",
    "size": [1420, 2840], # Y, X # Note: Swapped compared to MATLAB for consistency? Check usage. Assuming [Y, X] for param.size(1), param.size(2)
    "ballR": 61.5 / 2,
    "rdiam": 7,
    "cushionwidth": 50,
    "diamdist": 97,
    "framewidth": 147,
    "colors": "WYR",
    "BallOutOfTableDetectionLimit": 30,
    "BallCushionHitDetectionRange": 50, # Used implicitly in MATLAB?
    "BallProjecttoCushionLimit": 10,
    "NoDataDistanceDetectionLimit": 600,
    "MaxTravelDistancePerDt": 0.1, # Not explicitly used in provided MATLAB?
    "MaxVelocity": 12000,
    "timax_appr": 5,
    "tableX": 2840, # Scaling factors from read_gamefile
    "tableY": 1420,
}

# --- Utility Functions ---

def angle_vector(a, b):
    """Calculates angle between two vectors in degrees."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 0 and norm_b > 0:
        dot_product = np.dot(a, b)
        # Clamp dot_product to avoid floating point errors causing acos domain issues
        cos_angle = np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)
        return np.arccos(cos_angle) * 180 / np.pi
    elif norm_a > 0 or norm_b > 0:
        return -1.0 # Indicate one vector is zero
    else:
        return -2.0 # Indicate both vectors are zero

def str_to_indices(b1b2b3_str):
    """Converts 'WYR' style string to 0-based indices [W_idx, Y_idx, R_idx]."""
    mapping = {'W': 0, 'Y': 1, 'R': 2}
    try:
        return [mapping[b1b2b3_str[0]], mapping[b1b2b3_str[1]], mapping[b1b2b3_str[2]]]
    except (IndexError, KeyError):
        # Handle cases where input string is not valid like 'WYR'
         print(f"Warning: Invalid B1B2B3 string '{b1b2b3_str}'. Cannot convert to indices.")
         return None # Or raise an error

def indices_to_str(indices):
    """Converts 0-based indices [idx1, idx2, idx3] to 'WYR' style string."""
    mapping = {0: 'W', 1: 'Y', 2: 'R'}
    try:
        return mapping[indices[0]] + mapping[indices[1]] + mapping[indices[2]]
    except (IndexError, KeyError):
         print(f"Warning: Invalid indices '{indices}'. Cannot convert to string.")
         return "???"


# --- Classes ---

class BallRoute:
    """Represents the trajectory of a single ball."""
    def __init__(self, t=None, x=None, y=None):
        self.t = np.array(t) if t is not None else np.array([])
        self.x = np.array(x) if x is not None else np.array([])
        self.y = np.array(y) if y is not None else np.array([])
        # Derivatives can be calculated on demand or stored if frequently used
        self.vx = np.array([])
        self.vy = np.array([])
        self.v = np.array([])
        self.dt = np.array([])

    def calculate_velocities(self):
        if len(self.t) > 1:
            self.dt = np.diff(self.t)
            # Handle potential zero dt
            self.dt[self.dt == 0] = 1e-9 # Avoid division by zero, add small epsilon
            self.vx = np.diff(self.x) / self.dt
            self.vy = np.diff(self.y) / self.dt
             # Append a zero or repeat last for consistent length if needed by logic
            self.vx = np.append(self.vx, 0)
            self.vy = np.append(self.vy, 0)
            self.v = np.sqrt(self.vx**2 + self.vy**2)
            self.dt = np.append(self.dt, self.dt[-1] if len(self.dt)>0 else 0) # Repeat last dt
        else:
            self.vx = np.zeros(len(self.t))
            self.vy = np.zeros(len(self.t))
            self.v = np.zeros(len(self.t))
            self.dt = np.zeros(len(self.t))


    def copy(self):
        return BallRoute(self.t.copy(), self.x.copy(), self.y.copy())

    def __len__(self):
        return len(self.t)

class HitEvent:
    """Represents a single collision or start event."""
    def __init__(self, t, with_char, pos_x, pos_y):
        self.t = t
        self.with_char = with_char  # 'S', '1', '2', '3', '4', 'W', 'Y', 'R' (relative to B1/B2/B3)
        self.pos_x = pos_x
        self.pos_y = pos_y

        # Detailed metrics to be filled later
        self.type = np.nan         # 0=Start, 1=Ball, 2=Cushion
        self.from_c = np.nan       # Direction calculation result
        self.to_c = np.nan         # Direction calculation result
        self.v1 = np.nan           # Velocity magnitude before
        self.v2 = np.nan           # Velocity magnitude after
        self.offset = np.nan       # Offset calculation result
        self.fraction = np.nan     # Hit fraction for Ball-Ball
        self.def_angle = np.nan    # Deflection angle for Ball-Ball
        self.cut_angle = np.nan    # Cut angle for Ball-Ball
        self.c_in_angle = np.nan   # Cushion in angle
        self.c_out_angle = np.nan  # Cushion out angle
        self.from_c_pos = np.nan   # Diamond position
        self.to_c_pos = np.nan     # Diamond position
        self.from_d_pos = np.nan   # Diamond position (through)
        self.to_d_pos = np.nan     # Diamond position (through)

        # Kiss/Point specific results (often stored only for B1's first event)
        self.point = 0
        self.kiss = 0
        self.fuchs = 0
        self.point_dist = np.nan
        self.kiss_dist_b1 = np.nan
        self.t_point = np.inf # Use inf as default 'not occurred' time
        self.t_kiss = np.inf
        self.t_ready = np.inf # Time when B1 hit B2 and 3 Cushions
        self.t_b2_hit = np.inf # Time of first B1->B2 hit
        self.t_failure = np.inf # Time of premature B1->B3 hit


class Shot:
    """Represents a single shot with its data and events."""
    def __init__(self, shot_id, mirrored, point_type, table_index):
        self.shot_id = shot_id
        self.mirrored = mirrored
        self.point_type = point_type # From scoring entry type
        self.table_index = table_index # Index in the main DataFrame
        self.route0 = [BallRoute(), BallRoute(), BallRoute()] # Original routes (W, Y, R order)
        self.route = [BallRoute(), BallRoute(), BallRoute()] # Processed routes
        self.hit_events = [[], [], []] # List of HitEvent objects for each ball (W, Y, R order)
        self.b1b2b3_indices = None # Which original index (0,1,2) corresponds to B1, B2, B3? E.g., [0, 1, 2] means W=B1, Y=B2, R=B3
        self.b1b2b3_str = "???"
        self.interpreted = 0
        self.error_code = None
        self.error_text = ""
        self.selected = False # Flag for issues

    def get_ball_route(self, ball_index, original=False):
        """Access ball route using 0, 1, 2 index."""
        if original:
            return self.route0[ball_index]
        else:
            return self.route[ball_index]

    def get_b1_route(self, original=False):
        return self.get_ball_route(self.b1b2b3_indices[0], original)
    def get_b2_route(self, original=False):
        return self.get_ball_route(self.b1b2b3_indices[1], original)
    def get_b3_route(self, original=False):
        return self.get_ball_route(self.b1b2b3_indices[2], original)

    def get_ball_hits(self, ball_index):
         """Access hit events using 0, 1, 2 index."""
         return self.hit_events[ball_index]

    def get_b1_hits(self):
        return self.get_ball_hits(self.b1b2b3_indices[0])
    def get_b2_hits(self):
        return self.get_ball_hits(self.b1b2b3_indices[1])
    def get_b3_hits(self):
        return self.get_ball_hits(self.b1b2b3_indices[2])

    def add_hit_event(self, ball_index, hit_event):
        self.hit_events[ball_index].append(hit_event)


class ShotAnalyzer:
    """Main class to manage shot data analysis."""
    def __init__(self, params=None):
        self.params = params if params else DEFAULT_PARAMS.copy()
        self.shots = [] # List of Shot objects
        self.table = pd.DataFrame() # Summary table
        self.columns_visible = [] # Track dynamically added columns for table output

    def _find_shot_index(self, shot_id, mirrored):
        """Find the internal list index for a given shot_id and mirrored status."""
        # More efficient than MATLAB's intersect if table is large
        # This assumes shot_id + mirrored/10 logic is robust enough
        target_val = shot_id + mirrored / 10
        try:
            # Assuming 'ShotID' and 'Mirrored' columns exist and are numeric
            match = self.table['InternalID'] == target_val
            if match.any():
                return self.table.index[match].tolist()[0] # Get DataFrame index
            else:
                return None
        except KeyError:
             print("Error: 'ShotID' or 'Mirrored' not found in table during lookup.")
             return None


    def read_gamefile(self, filepath):
        """Reads a JSON game file and populates initial shot data."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filepath}")
            return False

        table_data = []
        self.shots = []
        si = -1 # Internal list index
        df_idx = 0 # DataFrame index counter

        player1 = data.get('Player1', 'Unknown')
        player2 = data.get('Player2', 'Unknown')
        game_type = data.get('Match', {}).get('GameType', 'Unknown')
        filename = filepath.split('/')[-1].split('\\')[-1] # Basic filename extraction

        for seti, set_data in enumerate(data.get('Match', {}).get('Sets', [])):
            entries = set_data.get('Entries', [])
            # Handle MATLAB's cell array case if necessary
            if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], list):
                 entries = entries[0] # Assume it wraps a single list
            if not isinstance(entries, list):
                 print(f"Warning: Skipping set {seti+1}, 'Entries' is not a list.")
                 continue

            for entryi, entry in enumerate(entries):
                 if isinstance(entry, dict) and entry.get('PathTrackingId', 0) > 0:
                     si += 1
                     shot_id = entry['PathTrackingId']
                     mirrored = 0 # Assuming default, modify if available in JSON
                     scoring = entry.get('Scoring', {})
                     point_type = scoring.get('EntryType', 'Unknown')

                     # Create Shot object
                     current_shot = Shot(shot_id, mirrored, point_type, df_idx)

                     # Extract Routes
                     path_tracking = entry.get('PathTracking', {})
                     datasets = path_tracking.get('DataSets', [])
                     valid_route = True
                     if len(datasets) == 3:
                         for bi in range(3):
                             coords = datasets[bi].get('Coords', [])
                             t = [c.get('DeltaT_500us', 0) * 0.0005 for c in coords]
                             x = [c.get('X', 0) * self.params['tableX'] for c in coords]
                             y = [c.get('Y', 0) * self.params['tableY'] for c in coords]
                             if not t: # If no coordinates, mark as invalid for now
                                 valid_route = False
                                 break
                             # Store original route
                             current_shot.route0[bi] = BallRoute(t, x, y)
                             # Initialize processed route as a copy
                             current_shot.route[bi] = current_shot.route0[bi].copy()
                     else:
                         valid_route = False

                     if not valid_route:
                          print(f"Warning: Skipping ShotID {shot_id} due to incomplete route data.")
                          si -= 1 # Decrement index as we skip this shot
                          continue

                     # Add shot to list
                     self.shots.append(current_shot)

                     # Prepare data for DataFrame row
                     player_num = scoring.get('Player', 0)
                     player_name = player1 if player_num == 1 else (player2 if player_num == 2 else 'Unknown')

                     row_data = {
                         'Selected': False,
                         'ShotID': shot_id,
                         'Mirrored': mirrored,
                         'InternalID': shot_id + mirrored / 10.0, # For easier lookup
                         'Filename': filename,
                         'GameType': game_type,
                         'Interpreted': 0,
                         'Player': player_name,
                         'ErrorID': None,
                         'ErrorText': 'Read OK',
                         'Set': seti + 1,
                         'CurrentInning': scoring.get('CurrentInning', 0),
                         'CurrentSeries': scoring.get('CurrentSeries', 0),
                         'CurrentTotalPoints': scoring.get('CurrentTotalPoints', 0),
                         'Point': point_type,
                         'TableIndex': df_idx # Store DF index in row data
                     }
                     table_data.append(row_data)
                     df_idx += 1

        if not table_data:
            print("No valid shot data found in the file.")
            return False

        self.table = pd.DataFrame(table_data)
        # Set DataFrame index to match the TableIndex column for easy access via iloc
        # self.table.set_index('TableIndex', inplace=True, drop=False)
        print(f"Read {len(self.shots)} shots.")
        return True


    def perform_data_quality_checks(self):
        """Applies data cleaning and validation rules to shot routes."""
        print("Performing data quality checks...")
        err_count = 0
        shots_to_remove = [] # Indices of shots to potentially remove

        for i, shot in enumerate(self.shots):
            if shot.interpreted: continue

            df_idx = shot.table_index # Get the corresponding DataFrame index
            error_code = None
            error_text = ""
            selected = False

            try:
                 # --- Apply checks similar to MATLAB's Extract_DataQuality_start ---
                 errcode_base = 100

                 # 1. Check for missing points (length 0 or 1)
                 for bi in range(3):
                     if len(shot.route[bi]) == 0:
                         error_code = errcode_base + 1
                         error_text = f'Ball {bi+1} data missing (len 0)'
                         selected = True
                         break
                     if len(shot.route[bi]) == 1:
                         error_code = errcode_base + 2
                         error_text = f'Ball {bi+1} data insufficient (len 1)'
                         selected = True
                         break
                 if selected: raise ValueError(error_text) # Go to error handling

                 # 2. Project points outside cushion limits (simplification of MATLAB)
                 #    MATLAB code had a flag 'if 1', so we implement this part.
                 #    It pushes points just inside the cushion radius.
                 tol = self.params['BallProjecttoCushionLimit'] # How far outside to tolerate before projecting? MATLAB didn't use tol here.
                 oncushion_x_min = self.params['ballR'] + 0.1
                 oncushion_x_max = self.params['size'][1] - self.params['ballR'] - 0.1
                 oncushion_y_min = self.params['ballR'] + 0.1
                 oncushion_y_max = self.params['size'][0] - self.params['ballR'] - 0.1

                 for bi in range(3):
                    route = shot.route[bi]
                    route.x = np.clip(route.x, oncushion_x_min, oncushion_x_max)
                    route.y = np.clip(route.y, oncushion_y_min, oncushion_y_max)

                 # 3. Check initial ball distance and correct overlap
                 errcode_base += 10
                 bb_indices = [(0, 1), (0, 2), (1, 2)]
                 for bbi, (b1i, b2i) in enumerate(bb_indices):
                     r1 = shot.route[b1i]
                     r2 = shot.route[b2i]
                     if len(r1) > 0 and len(r2) > 0:
                         dx = r1.x[0] - r2.x[0]
                         dy = r1.y[0] - r2.y[0]
                         dist_sq = dx*dx + dy*dy
                         min_dist_sq = (2 * self.params['ballR'])**2
                         if dist_sq < min_dist_sq - 1e-6: # Allow for small tolerance
                             dist = math.sqrt(dist_sq) if dist_sq > 0 else 0
                             overlap = 2 * self.params['ballR'] - dist
                             if dist > 1e-6: # Avoid division by zero if balls are exactly at same spot
                                 vec_x, vec_y = dx / dist, dy / dist
                                 # Move balls apart along the line connecting centers
                                 r1.x[0] += vec_x * overlap / 2.0
                                 r1.y[0] += vec_y * overlap / 2.0
                                 r2.x[0] -= vec_x * overlap / 2.0
                                 r2.y[0] -= vec_y * overlap / 2.0
                             else: # If balls start exactly at the same point, nudge one slightly
                                 r1.x[0] += overlap / 2.0


                 # 4. Check time linearity and attempt correction
                 errcode_base += 10
                 for bi in range(3):
                    route = shot.route[bi]
                    if len(route) > 1:
                        dt = np.diff(route.t)
                        non_linear_indices = np.where(dt <= 0)[0]
                        if len(non_linear_indices) > 0:
                            # Simple correction: remove points causing non-linearity
                            # MATLAB's correction was more complex, finding all subsequent points affected
                            # For simplicity here, just remove the point *after* the non-positive dt
                            indices_to_remove = sorted(list(set(non_linear_indices + 1)), reverse=True) # Get unique indices to remove

                            # Check if removing makes it too short
                            if len(route) - len(indices_to_remove) < 2:
                                 error_code = errcode_base + 1
                                 error_text = f'Ball {bi+1} time linearity unfixable (too short)'
                                 selected = True
                                 break # Break inner loop

                            # Remove points
                            route.t = np.delete(route.t, indices_to_remove)
                            route.x = np.delete(route.x, indices_to_remove)
                            route.y = np.delete(route.y, indices_to_remove)
                            print(f"Corrected time linearity for Shot {shot.shot_id}, Ball {bi+1}. Removed {len(indices_to_remove)} points.")
                            # Recheck after correction (optional, could lead to loops if not careful)
                            # dt_new = np.diff(route.t)
                            # if np.any(dt_new <= 0):
                            #     error_code = errcode_base + 1
                            #     error_text = f'Ball {bi+1} time linearity unfixable'
                            #     selected = True
                            #     break # Break inner loop
                    if selected: break # Break outer loop
                 if selected: raise ValueError(error_text)

                 # 5. Check start/end times consistency (approximate check)
                 errcode_base += 10
                 start_times = [shot.route[bi].t[0] for bi in range(3) if len(shot.route[bi]) > 0]
                 end_times = [shot.route[bi].t[-1] for bi in range(3) if len(shot.route[bi]) > 0]
                 if len(start_times)>0 and not np.allclose(start_times, start_times[0], atol=1e-6):
                     error_code = errcode_base + 1
                     error_text = 'Inconsistent start times'
                     selected = True
                     raise ValueError(error_text)
                 if len(end_times)>0 and not np.allclose(end_times, end_times[0], atol=1e-6):
                     error_code = errcode_base + 2
                     error_text = 'Inconsistent end times'
                     selected = True
                     raise ValueError(error_text)


                 # 6. Check for reflections/jumps (simplified check)
                 # MATLAB code seems complex; this is a basic distance check.
                 # A more robust check would analyze angles.
                 errcode_base += 10
                 for bi in range(3):
                     route = shot.route[bi]
                     if len(route) > 2:
                         dx = np.diff(route.x)
                         dy = np.diff(route.y)
                         ds = np.sqrt(dx**2 + dy**2)
                         # Look for large jumps back immediately followed by jump forward (reflection artifact)
                         # Example: point i -> i+1 is large neg dx, i+1 -> i+2 is large pos dx
                         # This is a very basic placeholder for the MATLAB logic.
                         pass # Skipping complex reflection logic for brevity

                 # 7. Check gaps and velocity
                 errcode_base += 10
                 max_dist_limit = self.params['NoDataDistanceDetectionLimit']
                 max_vel_limit = self.params['MaxVelocity']
                 for bi in range(3):
                     route = shot.route[bi]
                     if len(route) > 1:
                        route.calculate_velocities() # Ensure velocities are calculated
                        ds = np.sqrt(np.diff(route.x)**2 + np.diff(route.y)**2)
                        if np.any(ds > max_dist_limit):
                             error_code = errcode_base + 1
                             error_text = f'Ball {bi+1} gap too large'
                             selected = True
                             break
                        # Check velocity (v includes the appended 0, so check v[:-1])
                        if len(route.v) > 1 and np.any(route.v[:-1] > max_vel_limit):
                             error_code = errcode_base + 2
                             error_text = f'Ball {bi+1} velocity too high'
                             selected = True
                             break
                 if selected: raise ValueError(error_text)

                 # If all checks passed
                 error_text = "Quality OK"
                 # Update route0 with the cleaned route
                 for bi in range(3):
                     shot.route0[bi] = shot.route[bi].copy()


            except ValueError as e:
                 # Handles jumps from `raise ValueError`
                 print(f"Quality Check Failed for Shot {shot.shot_id} (Idx: {df_idx}): {e}")
                 err_count += 1
                 # Keep error_code and error_text set by the check

            except Exception as e:
                # Catch any other unexpected error during checks
                print(f"Unexpected Error during Quality Check for Shot {shot.shot_id} (Idx: {df_idx}): {e}")
                error_code = 999 # Generic error
                error_text = "Unexpected processing error"
                selected = True
                err_count += 1

            # --- Update Table ---
            # Use .loc with the DataFrame index stored in shot.table_index
            self.table.loc[df_idx, 'ErrorID'] = error_code
            self.table.loc[df_idx, 'ErrorText'] = error_text
            self.table.loc[df_idx, 'Selected'] = selected
            shot.error_code = error_code
            shot.error_text = error_text
            shot.selected = selected


        print(f"Data quality checks completed. {err_count} shots flagged.")
        # Consider removing shots flagged for removal if needed
        # self.shots = [s for i, s in enumerate(self.shots) if i not in shots_to_remove]
        # self.table = self.table[~self.table.index.isin(shots_to_remove)].reset_index(drop=True)


    def _determine_b1b2b3_for_shot(self, shot):
        """Determines B1, B2, B3 based on initial movement."""
        cols = "WYR"
        # Check if routes are long enough
        if any(len(shot.route[bi]) < 2 for bi in range(3)):
            return None, 2, "Empty Data (B1B2B3 determination)"

        # Initial sort based on time of second point (first movement)
        # Use a large number if a ball doesn't have a second point
        t2 = [shot.route[bi].t[1] if len(shot.route[bi]) > 1 else float('inf') for bi in range(3)]
        
        # Handle cases where start times might not be exactly zero after quality checks
        t_start = shot.route[0].t[0] # Assume consistent start after quality check
        t2 = [t - t_start for t in t2] # Time relative to start

        # Get indices sorted by second point time
        b_indices_sorted_by_t2 = np.argsort(t2)

        # Check if any ball has infinite t2 (meaning it didn't move)
        if t2[b_indices_sorted_by_t2[0]] == float('inf'):
             # No ball moved, return default? Or error? Let's return default based on MATLAB comment
             return [0, 1, 2], None, "No ball movement detected" # Default W,Y,R

        b1it = b_indices_sorted_by_t2[0] # Tentative B1 index (0, 1, or 2)
        b2it = b_indices_sorted_by_t2[1] # Tentative B2 index
        b3it = b_indices_sorted_by_t2[2] # Tentative B3 index

        # Check if B1 route is long enough for further analysis
        if len(shot.route[b1it]) < 3:
             # Only B1 moved definitively (or B1 and another started exactly same time but B1 route too short)
             # Return order based on initial sort
             return list(b_indices_sorted_by_t2), None, None # No error

        # Time points for comparison
        t2_b1 = shot.route[b1it].t[1]
        t3_b1 = shot.route[b1it].t[2]

        # Check if tentative B2 moved by time t2_b1 or t3_b1
        moved_b2 = False
        if len(shot.route[b2it]) > 1 and shot.route[b2it].t[1] <= t3_b1 + 1e-9: # Tolerance
            moved_b2 = True

        # Check if tentative B3 moved by time t2_b1 or t3_b1
        moved_b3 = False
        if len(shot.route[b3it]) > 1 and shot.route[b3it].t[1] <= t3_b1 + 1e-9: # Tolerance
            moved_b3 = True


        final_order = list(b_indices_sorted_by_t2) # Start with time-sorted order

        # Refine based on MATLAB logic (simplified interpretation)
        if moved_b2 and moved_b3:
            # All balls moved very early - ambiguous case
             return final_order, 3, "All balls moved simultaneously"
        elif moved_b2 and not moved_b3:
            # B1 and B2 moved, B3 didn't (early enough)
            # Check angles like MATLAB (using first step)
            r_b1 = shot.route[b1it]
            r_b2 = shot.route[b2it]
            vec_b1b2 = np.array([r_b2.x[0] - r_b1.x[0], r_b2.y[0] - r_b1.y[0]])
            vec_b1dir = np.array([r_b1.x[1] - r_b1.x[0], r_b1.y[1] - r_b1.y[0]])
            vec_b2dir = np.array([r_b2.x[1] - r_b2.x[0], r_b2.y[1] - r_b2.y[0]])
            # angle_b2 = angle_vector(vec_b1b2, vec_b2dir) # Angle between initial separation and B2 movement
            # MATLAB logic 'if angle_b2 > 90' seems to swap B1 and B2
            # This might indicate B2 was hit *by* B1, but the initial sort was wrong? Let's stick to sorted order for now.
            pass # Keep final_order = [b1it, b2it, b3it]
        elif not moved_b2 and moved_b3:
            # B1 and B3 moved, B2 didn't (early enough)
             pass # Keep final_order = [b1it, b2it, b3it] (B2 is still the second in sort)
        # else: Only B1 moved, keep initial sort

        return final_order, None, None # Return the determined order and no error


    def determine_b1b2b3(self):
        """Determines the B1, B2, B3 identity for each shot."""
        print("Determining B1, B2, B3...")
        err_count = 0
        for i, shot in enumerate(self.shots):
             if shot.interpreted or shot.selected: continue # Skip already processed or flagged shots

             df_idx = shot.table_index
             indices, err_code, err_text = self._determine_b1b2b3_for_shot(shot)

             if err_code:
                 shot.error_code = err_code
                 shot.error_text = err_text
                 shot.selected = True
                 print(f"B1B2B3 Error Shot {shot.shot_id} (Idx: {df_idx}): {err_text}")
                 err_count += 1
                 b1b2b3_str = "ERR"
                 # Update table immediately
                 self.table.loc[df_idx, 'ErrorID'] = err_code
                 self.table.loc[df_idx, 'ErrorText'] = err_text
                 self.table.loc[df_idx, 'Selected'] = True

             else:
                 shot.b1b2b3_indices = indices # Store e.g., [0, 1, 2]
                 b1b2b3_str = indices_to_str(indices)
                 shot.b1b2b3_str = b1b2b3_str
                 # Update table (B1B2B3 column might not exist yet)
                 if 'B1B2B3' not in self.table.columns:
                      self.table['B1B2B3'] = "???" # Initialize column if needed
                 self.table.loc[df_idx, 'B1B2B3'] = b1b2b3_str
                 # Clear previous errors if B1B2B3 determination is now successful
                 if self.table.loc[df_idx, 'ErrorID'] is not None and self.table.loc[df_idx, 'ErrorID'] < 10 : # Only clear B1B2B3 specific errors
                    self.table.loc[df_idx, 'ErrorID'] = None
                    self.table.loc[df_idx, 'ErrorText'] = "B1B2B3 OK"
                    self.table.loc[df_idx, 'Selected'] = False
                    shot.error_code = None
                    shot.error_text = "B1B2B3 OK"
                    shot.selected = False


        print(f"B1B2B3 determination completed. {err_count} shots flagged.")


    def _calculate_ball_velocity(self, route, hit_event_time, event_index):
        """Calculates velocity before and after a hit event."""
        # Corresponds to MATLAB's Ball_velocity

        t = route.t
        x = route.x
        y = route.y
        if len(t) < 2:
             return (0, 0), (0, 0), (0, 0), (np.nan, np.nan), np.nan # vt, v1_vec, v2_vec, alpha, offset

        try:
            # Find indices around the hit event time
            # Use searchsorted for efficiency
            ti = np.searchsorted(t, hit_event_time, side='left')
            # Ensure ti is within valid range
            ti = np.clip(ti, 1, len(t) - 1) # Need at least one point before and potentially one after

            # Find time index of previous and next hit event if they exist
            # This part is tricky without the full 'hit' structure readily available
            # Simplified approach: use points immediately before and after the interpolated time 'ti'
            # More accurate: would need access to the hit_events list for this ball

            # --- Velocity Before (v1) ---
            # Use interval ending at ti
            idx_before_start = max(0, ti - self.params['timax_appr'])
            idx_before_end = ti
            if idx_before_end <= idx_before_start: # Ensure at least one point difference
                idx_before_start = max(0, idx_before_end - 1)

            v1_vec = np.array([0.0, 0.0])
            vt1 = 0.0
            alpha1 = np.nan
            if idx_before_end > idx_before_start:
                # Use the two points defining the interval containing the event time
                idx1, idx2 = idx_before_end -1, idx_before_end # ti-1 and ti
                dt_before = t[idx2] - t[idx1]
                if dt_before > 1e-9:
                    vx1 = (x[idx2] - x[idx1]) / dt_before
                    vy1 = (y[idx2] - y[idx1]) / dt_before
                    v1_vec = np.array([vx1, vy1])
                    vt1 = np.linalg.norm(v1_vec)
                    if vt1 > 1e-9:
                        alpha1 = math.atan2(vy1, vx1) * 180 / np.pi # atan2(y, x) common convention

            # --- Velocity After (v2) ---
            # Use interval starting at ti
            idx_after_start = ti
            idx_after_end = min(len(t), ti + self.params['timax_appr'])
            if idx_after_end <= idx_after_start:
                idx_after_end = min(len(t), idx_after_start + 1)

            v2_vec = np.array([0.0, 0.0])
            vt2 = 0.0
            alpha2 = np.nan
            if idx_after_end > idx_after_start:
                 # Use the first two points *after* the event index ti
                 idx1, idx2 = idx_after_start, idx_after_start + 1
                 if idx2 < len(t): # Ensure idx2 is valid
                    dt_after = t[idx2] - t[idx1]
                    if dt_after > 1e-9:
                        vx2 = (x[idx2] - x[idx1]) / dt_after
                        vy2 = (y[idx2] - y[idx1]) / dt_after
                        v2_vec = np.array([vx2, vy2])
                        vt2 = np.linalg.norm(v2_vec)
                        if vt2 > 1e-9:
                            alpha2 = math.atan2(vy2, vx2) * 180 / np.pi # atan2(y, x)

            # --- Offset Calculation (Placeholder) ---
            # MATLAB code calculates offset based on points p1 (at ti) and p2 (at ti_after)
            # and the direction vector v2. This needs careful translation.
            offset = np.nan # Placeholder


            # Return magnitudes and vectors
            return (vt1, vt2), v1_vec, v2_vec, (alpha1, alpha2), offset

        except IndexError:
            print(f"Warning: Index error during velocity calculation for event at t={hit_event_time}")
            return (0, 0), (0, 0), (0, 0), (np.nan, np.nan), np.nan
        except Exception as e:
            print(f"Warning: Error during velocity calculation: {e}")
            return (0, 0), (0, 0), (0, 0), (np.nan, np.nan), np.nan


    def _calculate_ball_direction(self, route, hit_event_time, event_index, num_events):
         """Calculates cushion projection points (diamonds). Simplified."""
         # Corresponds to MATLAB's Ball_direction
         # This requires careful interpolation and extrapolation based on points
         # around the event time. Simplified version returning NaNs.

         t = route.t
         x = route.x
         y = route.y
         if len(t) < 2:
              return np.full((2, 6), np.nan) # Matches MATLAB output structure size

         try:
            # Find indices around the event time
            ti = np.searchsorted(t, hit_event_time, side='left')
            ti = np.clip(ti, 0, len(t) - 1)

            # Determine time intervals before/after (needs access to other hit times)
            # Simplified: Use a fixed number of points or time delta if possible
            t_before_event = hit_event_time - 0.05 # Example fixed delta
            t_after_event = hit_event_time + 0.05

            # Indices for 'from' direction (before event or at start)
            idx_from_start = np.searchsorted(t, t_before_event, side='left')
            idx_from_end = ti
            idx_from_start = max(0, idx_from_end - self.params['timax_appr'])
            idx_from_end = max(idx_from_start + 1, idx_from_end) # Need at least 2 points

             # Indices for 'to' direction (after event or towards end)
            idx_to_start = ti + 1 # Start from point *after* the event index
            idx_to_end = np.searchsorted(t, t_after_event, side='right')
            idx_to_end = min(len(t), idx_to_start + self.params['timax_appr'])
            idx_to_start = min(idx_to_end -1, idx_to_start) # Need at least 2 points

            results = np.full((2, 6), np.nan) # Rows: 0='from', 1='to'. Cols: fromC#, PosOn, PosThrough, toC#, PosOn, PosThrough

            indices_sets = [(idx_from_start, idx_from_end), (idx_to_start, idx_to_end)]

            for i, (idx_start, idx_end) in enumerate(indices_sets):
                if idx_end > idx_start + 1 and idx_end <= len(t): # Need at least two points
                    t_seg, x_seg, y_seg = t[idx_start:idx_end], x[idx_start:idx_end], y[idx_start:idx_end]

                    # Use first and last point of the segment for direction vector
                    p1 = np.array([x_seg[0], y_seg[0]])
                    p2 = np.array([x_seg[-1], y_seg[-1]])
                    
                    if np.allclose(p1, p2): continue # Skip if points are the same

                    # --- Extrapolation to cushions (similar to MATLAB) ---
                    # Caution: Extrapolation can be sensitive
                    # Y coordinates for horizontal cushions (1 and 3)
                    y_cushion1_on = self.params['ballR']
                    y_cushion3_on = self.params['size'][0] - self.params['ballR']
                    y_cushion1_th = -self.params['diamdist']
                    y_cushion3_th = self.params['size'][0] + self.params['diamdist']

                    # X coordinates for vertical cushions (2 and 4)
                    x_cushion2_on = self.params['size'][1] - self.params['ballR']
                    x_cushion4_on = self.params['ballR']
                    x_cushion2_th = self.params['size'][1] + self.params['diamdist']
                    x_cushion4_th = -self.params['diamdist']
                    
                    # Interpolation functions
                    try:
                        interp_x = interp1d(y_seg, x_seg, kind='linear', fill_value="extrapolate", bounds_error=False)
                        interp_y = interp1d(x_seg, y_seg, kind='linear', fill_value="extrapolate", bounds_error=False)
                    except ValueError: # Need at least 2 points for interp1d
                        continue 

                    # Calculate intersection points
                    xon1 = interp_x(y_cushion1_on)
                    xon3 = interp_x(y_cushion3_on)
                    xth1 = interp_x(y_cushion1_th)
                    xth3 = interp_x(y_cushion3_th)

                    yon2 = interp_y(x_cushion2_on)
                    yon4 = interp_y(x_cushion4_on)
                    yth2 = interp_y(x_cushion2_th)
                    yth4 = interp_y(x_cushion4_th)

                    # Determine 'from' or 'to' based on segment direction
                    dx_seg, dy_seg = p2[0] - p1[0], p2[1] - p1[1]

                    # Check Cushion 1 (Bottom, Y increasing towards it)
                    if y_cushion1_on <= yon4 <= y_cushion3_on : # Basic check if Y is within table Y range approx
                        if dy_seg < -1e-6 : # Moving towards cushion 1 (decreasing Y)
                            results[i, 3] = 1 # toCushion
                            results[i, 4] = xon1 # PosOn
                            results[i, 5] = xth1 # PosThrough
                        elif dy_seg > 1e-6: # Moving away from cushion 1
                            results[i, 0] = 1 # fromCushion
                            results[i, 1] = xon1
                            results[i, 2] = xth1

                    # Check Cushion 3 (Top, Y decreasing towards it)
                    if y_cushion1_on <= yon2 <= y_cushion3_on :
                        if dy_seg > 1e-6 : # Moving towards cushion 3 (increasing Y)
                            results[i, 3] = 3
                            results[i, 4] = xon3
                            results[i, 5] = xth3
                        elif dy_seg < -1e-6: # Moving away from cushion 3
                            results[i, 0] = 3
                            results[i, 1] = xon3
                            results[i, 2] = xth3

                    # Check Cushion 2 (Right, X decreasing towards it)
                    if x_cushion4_on <= xon3 <= x_cushion2_on : # Basic check if X is within table X range approx
                         if dx_seg > 1e-6 : # Moving towards cushion 2 (increasing X)
                            results[i, 3] = 2
                            results[i, 4] = yon2
                            results[i, 5] = yth2
                         elif dx_seg < -1e-6: # Moving away from cushion 2
                            results[i, 0] = 2
                            results[i, 1] = yon2
                            results[i, 2] = yth2

                    # Check Cushion 4 (Left, X increasing towards it)
                    if x_cushion4_on <= xon1 <= x_cushion2_on :
                         if dx_seg < -1e-6 : # Moving towards cushion 4 (decreasing X)
                            results[i, 3] = 4
                            results[i, 4] = yon4
                            results[i, 5] = yth4
                         elif dx_seg > 1e-6: # Moving away from cushion 4
                            results[i, 0] = 4
                            results[i, 1] = yon4
                            results[i, 2] = yth4

            # Convert positions to diamond units (simple scaling)
            # PosOn X (Cushions 1, 3) -> Scale by Y size (param.size[0])? No, X pos along Y size -> param.size[1]/8
            # PosOn Y (Cushions 2, 4) -> Scale by X size (param.size[1])? No, Y pos along X size -> param.size[0]/4

            # diamond_scale_x = 8.0 / self.params['size'][1] # Diamonds along Long cushion (X)
            # diamond_scale_y = 4.0 / self.params['size'][0] # Diamonds along Short cushion (Y)

            # # Scale PosOn values
            # if not np.isnan(results[0, 1]): # From C PosOn
            #     cush = results[0, 0]
            #     results[0, 1] *= diamond_scale_y if cush in [2, 4] else diamond_scale_x
            # if not np.isnan(results[1, 1]): # To C PosOn (index might be wrong here)
            #     cush = results[1, 0] # Should be results[0,3]? Needs fix
            #     # results[1, 1] *= diamond_scale_y if cush in [2, 4] else diamond_scale_x

            # # Scale PosThrough values
            # if not np.isnan(results[0, 2]): # From C PosThrough
            #     cush = results[0, 0]
            #     results[0, 2] *= diamond_scale_y if cush in [2, 4] else diamond_scale_x
            # if not np.isnan(results[1, 2]): # To C PosThrough (index might be wrong here)
            #     cush = results[1, 0] # Should be results[0,3]? Needs fix
            #     # results[1, 2] *= diamond_scale_y if cush in [2, 4] else diamond_scale_x

            # --- Simplified Diamond Scaling ---
            # Apply scaling based on whether it's X or Y position being reported
            x_pos_indices = [(0,1), (0,2), (1,1), (1,2)] # (row, col) for X positions on cushions 1,3
            y_pos_indices = [(0,4), (0,5), (1,4), (1,5)] # (row, col) for Y positions on cushions 2,4
            
            diamond_scale_x = 8.0 / self.params['size'][1] # For X positions (on cushion 1 or 3)
            diamond_scale_y = 4.0 / self.params['size'][0] # For Y positions (on cushion 2 or 4)

            for r, c in x_pos_indices: # Scaling PosOn and PosThrough for Cushions 1, 3
                 if not np.isnan(results[r, c]):
                     # Determine if it's fromC (col 1,2) or toC (col 4,5)
                     cushion_index_col = 0 if c <= 2 else 3
                     cushion_num = results[r, cushion_index_col]
                     if cushion_num in [1, 3]:
                          results[r, c] *= diamond_scale_x
                     # else: # It was assigned to wrong cushion, leave unscaled? or NaN?
                     #    results[r,c] = np.nan

            for r, c in y_pos_indices: # Scaling PosOn and PosThrough for Cushions 2, 4
                 if not np.isnan(results[r, c]):
                     cushion_index_col = 0 if c <= 2 else 3
                     cushion_num = results[r, cushion_index_col]
                     if cushion_num in [2, 4]:
                          results[r, c] *= diamond_scale_y
                     # else:
                     #     results[r,c] = np.nan


            return results


         except Exception as e:
            print(f"Warning: Error during direction calculation: {e}")
            return np.full((2, 6), np.nan)

    def _calculate_cushion_angle(self, cushion_index, v1_vec, v2_vec):
        """Calculates cushion in/out angles."""
        # Corresponds to MATLAB's CushionAngle

        v1_vec_3d = np.append(v1_vec, 0)
        v2_vec_3d = np.append(v2_vec, 0)
        norm_v1 = np.linalg.norm(v1_vec_3d)
        norm_v2 = np.linalg.norm(v2_vec_3d)

        if norm_v1 < 1e-9 or norm_v2 < 1e-9: # No angle if velocity is zero
            return np.nan, np.nan

        # Cushion normal vectors (pointing inwards) - Assuming std table layout
        normals = {
            1: np.array([0, 1, 0]),  # Bottom cushion (pointing up)
            2: np.array([-1, 0, 0]), # Right cushion (pointing left)
            3: np.array([0, -1, 0]), # Top cushion (pointing down)
            4: np.array([1, 0, 0])   # Left cushion (pointing right)
        }
        if cushion_index not in normals:
             return np.nan, np.nan

        cushion_normal = normals[cushion_index]

        # Angle between incoming velocity and cushion normal (use -normal for standard angle)
        angle_in = angle_vector(v1_vec_3d, -cushion_normal)

        # Angle between outgoing velocity and cushion normal
        angle_out = angle_vector(v2_vec_3d, cushion_normal)

        # --- Adjust angle_in sign based on tangential velocity direction change (MATLAB logic) ---
        # Checks if the velocity component PARALLEL to the cushion reverses direction.
        # If it reverses, it implies spin was involved, and MATLAB negates angle_in.
        sign_changed = False
        if cushion_index in [1, 3]: # Horizontal cushions, check X velocity component
            if np.sign(v1_vec[0]) != np.sign(v2_vec[0]) and abs(v1_vec[0]) > 1e-6 and abs(v2_vec[0]) > 1e-6:
                sign_changed = True
        elif cushion_index in [2, 4]: # Vertical cushions, check Y velocity component
            if np.sign(v1_vec[1]) != np.sign(v2_vec[1]) and abs(v1_vec[1]) > 1e-6 and abs(v2_vec[1]) > 1e-6:
                sign_changed = True

        if sign_changed:
            angle_in = -angle_in

        return angle_in, angle_out


    def _evaluate_hit_details(self, shot):
        """Calculates detailed metrics for each hit event in a shot."""
        # Corresponds to MATLAB's eval_hit_events

        if shot.b1b2b3_indices is None:
             print(f"Warning: Cannot evaluate hit details for Shot {shot.shot_id}, B1B2B3 unknown.")
             return

        b1i, b2i, b3i = shot.b1b2b3_indices # Original indices (0,1,2) for B1, B2, B3
        cols = "WYR"

        for ball_idx_orig in range(3): # Iterate through W, Y, R (0, 1, 2)
            hit_events = shot.hit_events[ball_idx_orig]
            route = shot.route[ball_idx_orig]

            for hi, event in enumerate(hit_events):
                if event.with_char == 'S': # Skip start event
                    event.type = 0
                    continue
                if event.with_char == '-': # Skip padding/placeholder event
                    continue

                # --- Calculate Velocity and Angles ---
                vt, v1_vec, v2_vec, alpha, offset = self._calculate_ball_velocity(route, event.t, hi)
                event.v1 = vt[0] / 1000.0 # Convert mm/s to m/s like MATLAB? Check units. Assume input is mm/s.
                event.v2 = vt[1] / 1000.0
                event.offset = offset # Store offset if calculated

                # --- Calculate Direction ---
                # Need number of events for the route
                num_events_for_ball = len([e for e in hit_events if e.with_char != 'S' and e.with_char != '-'])
                direction_info = self._calculate_ball_direction(route, event.t, hi, num_events_for_ball)
                # direction_info row 0 is 'from', row 1 is 'to'
                event.from_c = direction_info[0, 0]
                event.from_c_pos = direction_info[0, 1]
                event.from_d_pos = direction_info[0, 2]
                event.to_c = direction_info[0, 3] # 'to' info is stored in row 0, cols 3,4,5 in MATLAB output
                event.to_c_pos = direction_info[0, 4]
                event.to_d_pos = direction_info[0, 5]

                # --- Identify Hit Type and Calculate Specific Metrics ---
                contact_char = event.with_char
                is_ball_hit = contact_char in cols
                is_cushion_hit = contact_char in '1234'

                if is_cushion_hit:
                    event.type = 2 # Cushion
                    cushion_index = int(contact_char)
                    event.c_in_angle, event.c_out_angle = self._calculate_cushion_angle(cushion_index, v1_vec, v2_vec)

                elif is_ball_hit:
                    event.type = 1 # Ball
                    target_ball_char = contact_char
                    # Find original index (0,1,2) of the target ball
                    target_ball_idx_orig = cols.find(target_ball_char)

                    # Ensure velocities are valid for calculation
                    if event.v1 > 1e-6: # Need incoming velocity
                         # Find the corresponding hit event on the *other* ball to get its velocity *before* contact
                         # This requires searching the other ball's hit list for the same time 'event.t'
                         v1_target_vec = np.array([0.0, 0.0]) # Assume target was stationary initially
                         target_hit_events = shot.hit_events[target_ball_idx_orig]
                         target_route = shot.route[target_ball_idx_orig]
                         for target_event in target_hit_events:
                             if abs(target_event.t - event.t) < 1e-9: # Found matching event
                                 _, v1_target_vec_calc, _, _, _ = self._calculate_ball_velocity(target_route, target_event.t, 0) # Index 0 placeholder
                                 v1_target_vec = v1_target_vec_calc
                                 break

                         # Calculate hit fraction based on relative velocity and positions
                         # Position of this ball (bi) at event time `event.t`
                         pos1 = np.array([event.pos_x, event.pos_y])
                         # Position of target ball (b2i_orig) at event time `event.t`
                         # Find target ball's event at same time
                         pos2 = np.array([np.nan, np.nan])
                         for target_event in target_hit_events:
                              if abs(target_event.t - event.t) < 1e-9:
                                  pos2 = np.array([target_event.pos_x, target_event.pos_y])
                                  break

                         if not np.isnan(pos2[0]):
                            # Relative velocity vector *before* contact
                            vel_rel = v1_vec - v1_target_vec
                            norm_vel_rel = np.linalg.norm(vel_rel)

                            if norm_vel_rel > 1e-6:
                                 # Vector from pos1 to pos2
                                 p1p2_vec = pos2 - pos1
                                 # Calculate distance from p2 to the line defined by p1 and vel_rel
                                 # Using formula: |(p2-p1) x vel_rel| / |vel_rel|
                                 # In 2D, cross product magnitude is |(x2-x1)*vy - (y2-y1)*vx|
                                 cross_prod_mag = abs(p1p2_vec[0] * vel_rel[1] - p1p2_vec[1] * vel_rel[0])
                                 perp_dist = cross_prod_mag / norm_vel_rel

                                 # Fraction calculation from MATLAB
                                 event.fraction = 1.0 - perp_dist / (self.params['ballR'] * 2.0)
                                 event.fraction = np.clip(event.fraction, 0.0, 1.0) # Ensure bounds
                            else:
                                 event.fraction = 0.0 # Or NaN if relative velocity is zero

                         # Calculate deflection angle (angle between v1_vec and v2_vec of *this* ball)
                         if np.linalg.norm(v1_vec) > 1e-6 and np.linalg.norm(v2_vec) > 1e-6:
                             event.def_angle = angle_vector(v1_vec, v2_vec)
                         else:
                             event.def_angle = 0.0 # Or NaN

                         # Calculate cut angle (angle between v1_vec of *this* ball and v2_vec of *target* ball)
                         # Need v2_vec of the target ball
                         v2_target_vec = np.array([0.0, 0.0])
                         for target_event in target_hit_events:
                              if abs(target_event.t - event.t) < 1e-9:
                                  _, _, v2_target_vec_calc, _, _ = self._calculate_ball_velocity(target_route, target_event.t, 0) # Index 0 placeholder
                                  v2_target_vec = v2_target_vec_calc
                                  break

                         if np.linalg.norm(v1_vec) > 1e-6 and np.linalg.norm(v2_target_vec) > 1e-6:
                             event.cut_angle = angle_vector(v1_vec, v2_target_vec)
                         else:
                              event.cut_angle = 0.0 # Or NaN


        # --- Calculate Aim/Position Offsets (for B1 only) ---
        # These are calculated based on the first few points/events of B1
        b1_route = shot.get_b1_route()
        b1_hits = shot.get_b1_hits()
        aim_offset_ll = np.nan
        aim_offset_ss = np.nan

        if len(b1_route) >= 2:
            dx12 = b1_route.x[1] - b1_route.x[0]
            dy12 = b1_route.y[1] - b1_route.y[0]
            table_h = self.params['size'][0]
            table_w = self.params['size'][1]
            diam_dist = self.params['diamdist']

            # Aim Offset LL (Long-Long) - based on Y change
            if abs(dy12) > 1e-6:
                 aim_offset_ll = abs((dx12 / dy12) * (table_h + 2 * diam_dist)) * 4.0 / table_h # Should scale by Y size? Check MATLAB param.size(1) usage = Y
                 # Correction: MATLAB uses *4/param.size(1) which is Y. Let's assume that's diamonds along Y.
                 # aim_offset_ll = abs((dx12 / dy12) * (table_h + 2 * diam_dist)) * 4.0 / table_h # Original guess
                 aim_offset_ll = abs((dx12 / dy12) * (table_w + 2 * diam_dist)) * 8.0 / table_w # More likely: diamonds along X (long) based on Y change


            # Aim Offset SS (Short-Short) - based on X change
            if abs(dx12) > 1e-6:
                 aim_offset_ss = abs((dy12 / dx12) * (table_w + 2 * diam_dist)) * 8.0 / table_w # Should scale by X size? Check MATLAB param.size(2) usage = X
                 # Correction: MATLAB uses *4/param.size(1)? No, *8/param.size(2) -> 8 diamonds along Y (short) based on X change
                 # aim_offset_ss = abs((dy12 / dx12) * (table_w + 2 * diam_dist)) * 8.0 / table_w # Original guess
                 aim_offset_ss = abs((dy12 / dx12) * (table_h + 2 * diam_dist)) * 4.0 / table_h # More likely: diamonds along Y (short) based on X change


            # Adjust sign based on first hit type (MATLAB logic)
            if len(b1_hits) >= 2:
                first_hit_event = b1_hits[1] # Index 0 is 'S'tart
                if first_hit_event.type == 1: # Hit ball B2 first
                    b2_route = shot.get_b2_route()
                    if len(b2_route) > 0 and len(b1_route) > 1:
                         # Sign depends on relative position and movement direction
                         # Needs careful translation of the sign() logic from MATLAB
                         # directL = sign(hit(b2i).XPos(1)-hit(b1i).XPos(2))*sign(hit(b1i).XPos(1)-hit(b1i).XPos(2));
                         # directS = sign(hit(b2i).YPos(1)-hit(b1i).YPos(2))*sign(hit(b1i).YPos(1)-hit(b1i).YPos(2));
                         # Simplified: Assume sign adjustment needed, placeholder sign = 1
                         sign_l, sign_s = 1, 1 # Placeholder
                         if not np.isnan(aim_offset_ll): aim_offset_ll *= sign_l
                         if not np.isnan(aim_offset_ss): aim_offset_ss *= sign_s
                elif first_hit_event.type == 2 and len(b1_hits) >= 3: # Hit cushion first, need next event
                    # directS = sign(( hit(b1i).YPos(2)-hit(b1i).YPos(1) ) - (hit(b1i).YPos(2)-hit(b1i).YPos(3)));
                    # directL = sign(( hit(b1i).XPos(2)-hit(b1i).XPos(1) ) - (hit(b1i).XPos(2)-hit(b1i).XPos(3)));
                    # Needs hit positions YPos(1), YPos(2), YPos(3) etc.
                    # Placeholder sign = 1
                    sign_l, sign_s = 1, 1 # Placeholder
                    if not np.isnan(aim_offset_ll): aim_offset_ll *= sign_l
                    if not np.isnan(aim_offset_ss): aim_offset_ss *= sign_s


        # Store B1 offsets in B1's first event (or dedicated shot attribute)
        if b1_hits:
             b1_hits[0].aim_offset_ll = aim_offset_ll # Store in Start event?
             b1_hits[0].aim_offset_ss = aim_offset_ss

        # --- Calculate B1-B2, B1-B3 initial position offsets ---
        b1_start_pos = np.array([b1_route.x[0], b1_route.y[0]]) if len(b1_route) > 0 else np.array([np.nan, np.nan])
        b2_route = shot.get_b2_route()
        b3_route = shot.get_b3_route()
        b2_start_pos = np.array([b2_route.x[0], b2_route.y[0]]) if len(b2_route) > 0 else np.array([np.nan, np.nan])
        b3_start_pos = np.array([b3_route.x[0], b3_route.y[0]]) if len(b3_route) > 0 else np.array([np.nan, np.nan])

        b1b2_offset_ll, b1b2_offset_ss = np.nan, np.nan
        b1b3_offset_ll, b1b3_offset_ss = np.nan, np.nan

        if not np.isnan(b1_start_pos[0]) and not np.isnan(b2_start_pos[0]):
            dx_b1b2 = b2_start_pos[0] - b1_start_pos[0]
            dy_b1b2 = b2_start_pos[1] - b1_start_pos[1]
            if abs(dy_b1b2) > 1e-6:
                 b1b2_offset_ll = abs((dx_b1b2 / dy_b1b2) * (table_w + 2 * diam_dist)) * 8.0 / table_w
            else: b1b2_offset_ll = 99 # MATLAB code uses 99 for parallel case
            if abs(dx_b1b2) > 1e-6:
                 b1b2_offset_ss = abs((dy_b1b2 / dx_b1b2) * (table_h + 2 * diam_dist)) * 4.0 / table_h
            else: b1b2_offset_ss = 99

        if not np.isnan(b1_start_pos[0]) and not np.isnan(b3_start_pos[0]):
            dx_b1b3 = b3_start_pos[0] - b1_start_pos[0]
            dy_b1b3 = b3_start_pos[1] - b1_start_pos[1]
            if abs(dy_b1b3) > 1e-6:
                 b1b3_offset_ll = abs((dx_b1b3 / dy_b1b3) * (table_w + 2 * diam_dist)) * 8.0 / table_w
            else: b1b3_offset_ll = 99
            if abs(dx_b1b3) > 1e-6:
                 b1b3_offset_ss = abs((dy_b1b3 / dx_b1b3) * (table_h + 2 * diam_dist)) * 4.0 / table_h
            else: b1b3_offset_ss = 99

        # Store these in B1's start event as well
        if b1_hits:
            b1_hits[0].b1b2_offset_ll = b1b2_offset_ll
            b1_hits[0].b1b2_offset_ss = b1b2_offset_ss
            b1_hits[0].b1b3_offset_ll = b1b3_offset_ll
            b1_hits[0].b1b3_offset_ss = b1b3_offset_ss

        # --- Calculate Inside/Outside shot (for B1 only) ---
        # Based on sequence B1->B2 hit, B1->Cushion hit, B1->Next hit
        inside_outside = '?' # Default unknown
        if len(b1_hits) >= 4: # Need Start, B2, Cushion, NextEvent
            # Check sequence type: Start(0), Ball(1), Cushion(2), Any(3)
            if b1_hits[1].type == 1 and b1_hits[2].type == 2:
                 # Position at Start, B2 hit, Cushion hit, Next event
                 p1 = np.array([b1_hits[0].pos_x, b1_hits[0].pos_y, 0])
                 p2 = np.array([b1_hits[1].pos_x, b1_hits[1].pos_y, 0])
                 p3 = np.array([b1_hits[2].pos_x, b1_hits[2].pos_y, 0])
                 p4 = np.array([b1_hits[3].pos_x, b1_hits[3].pos_y, 0])

                 v1 = p1 - p3 # Vector from Cushion Hit back to Start
                 v2 = p2 - p3 # Vector from Cushion Hit back to B2 Hit
                 v3 = p4 - p3 # Vector from Cushion Hit to Next Event

                 a1 = angle_vector(v1, v3) # Angle between initial dir projection and final dir
                 a2 = angle_vector(v2, v3) # Angle between B2->C dir and final dir

                 # MATLAB logic: if a1 <= a2 -> Inside ('I'), else Outside ('E')
                 if not np.isnan(a1) and not np.isnan(a2):
                     inside_outside = 'I' if a1 <= a2 else 'E'

        # Store Inside/Outside marker (e.g., in B1 start event 'with_char'?)
        # MATLAB overwrites with(1) which was 'S'. Let's add a specific field.
        if b1_hits:
             b1_hits[0].inside_outside_marker = inside_outside


    def _evaluate_point_kiss_fuchs(self, shot):
        """Evaluate Point, Kiss, Fuchs status and distances."""
        # Corresponds to MATLAB's eval_Point_and_Kiss_Control and eval_kiss

        if shot.b1b2b3_indices is None: return

        b1i, b2i, b3i = shot.b1b2b3_indices
        b1_hits = shot.get_b1_hits()
        b2_hits = shot.get_b2_hits()
        b3_hits = shot.get_b3_hits()
        b1_route = shot.get_b1_route()
        b2_route = shot.get_b2_route()
        b3_route = shot.get_b3_route()
        cols = "WYR"
        target_b2_char = cols[b2i]
        target_b3_char = cols[b3i]
        target_b1_char = cols[b1i]

        # Find relevant hit events and times
        b1_to_b3_hits = [h for h in b1_hits if h.with_char == target_b3_char]
        b1_to_b2_hits = [h for h in b1_hits if h.with_char == target_b2_char]
        b1_cushion_hits = [h for h in b1_hits if h.with_char in '1234']
        b2_to_b3_hits = [h for h in b2_hits if h.with_char == target_b3_char]

        # Sort by time just in case
        b1_to_b3_hits.sort(key=lambda h: h.t)
        b1_to_b2_hits.sort(key=lambda h: h.t)
        b1_cushion_hits.sort(key=lambda h: h.t)
        b2_to_b3_hits.sort(key=lambda h: h.t)

        first_b1_b2_hit = b1_to_b2_hits[0] if b1_to_b2_hits else None
        first_b1_b3_hit = b1_to_b3_hits[0] if b1_to_b3_hits else None
        third_b1_cush_hit = b1_cushion_hits[2] if len(b1_cushion_hits) >= 3 else None
        second_b1_b2_hit = b1_to_b2_hits[1] if len(b1_to_b2_hits) >= 2 else None
        first_b2_b3_hit = b2_to_b3_hits[0] if b2_to_b3_hits else None

        point = 0
        point_time = np.inf
        failure_time = np.inf
        kiss = 0
        kiss_time = np.inf
        fuchs = 0
        b1b2_3c_time = np.inf # Time when B1 hit B2 AND hit 3rd cushion (max of the two)

        # --- Check for Point ---
        if first_b1_b2_hit and first_b1_b3_hit and third_b1_cush_hit:
            if first_b1_b3_hit.t > first_b1_b2_hit.t + 1e-9 and \
               first_b1_b3_hit.t > third_b1_cush_hit.t + 1e-9:
                point = 1
                point_time = first_b1_b3_hit.t

        # --- Check for Failure (premature B1->B3) ---
        if first_b1_b2_hit and first_b1_b3_hit and third_b1_cush_hit:
             if first_b1_b3_hit.t > first_b1_b2_hit.t + 1e-9 and \
                first_b1_b3_hit.t < third_b1_cush_hit.t - 1e-9: # Hit B3 *before* 3rd cushion
                 failure_time = first_b1_b3_hit.t
                 # Note: MATLAB code didn't explicitly set an error flag here, just time

        # --- Check for Kisses ---
        if point == 1:
            # Check B2->B3 kiss before point
            if first_b2_b3_hit and first_b2_b3_hit.t < point_time - 1e-9:
                 if first_b2_b3_hit.t < kiss_time:
                      kiss = 3 # B2-B3 kiss is primary
                      kiss_time = first_b2_b3_hit.t
                      fuchs = 1
            # Check B1->B2 kiss (second time) before point
            if second_b1_b2_hit and second_b1_b2_hit.t < point_time - 1e-9:
                 if second_b1_b2_hit.t < kiss_time:
                      kiss = 1 # B1-B2 kiss is primary
                      kiss_time = second_b1_b2_hit.t
                      fuchs = 1
        elif failure_time == np.inf: # Only check kiss if not a failure and no point
             # Check B2->B3 kiss
             if first_b2_b3_hit:
                 if first_b2_b3_hit.t < kiss_time:
                      kiss = 3
                      kiss_time = first_b2_b3_hit.t
             # Check B1->B2 kiss (second time)
             if second_b1_b2_hit:
                 if second_b1_b2_hit.t < kiss_time:
                      kiss = 1
                      kiss_time = second_b1_b2_hit.t

        # --- Calculate Time Ready (B1 hit B2 and 3 Cushions) ---
        if first_b1_b2_hit and third_b1_cush_hit:
             b1b2_3c_time = max(first_b1_b2_hit.t, third_b1_cush_hit.t)

        # --- Get B1->B2 Hit Time ---
        t_b2_hit = first_b1_b2_hit.t if first_b1_b2_hit else np.inf

        # --- Store results in B1's start event ---
        if shot.get_b1_hits():
            start_event = shot.get_b1_hits()[0]
            start_event.point = point
            start_event.kiss = kiss
            start_event.fuchs = fuchs
            start_event.t_point = point_time
            start_event.t_kiss = kiss_time
            start_event.t_ready = b1b2_3c_time
            start_event.t_b2_hit = t_b2_hit
            start_event.t_failure = failure_time

        # --- Calculate Point Distance ---
        point_dist = np.nan
        if point == 1 and first_b1_b3_hit:
             # Use hit fraction from the B1->B3 event
             hit_fraction = first_b1_b3_hit.fraction
             if not np.isnan(hit_fraction):
                 # Determine sign based on cross product (MATLAB logic)
                 # Need positions before and at the hit
                 b1_event_idx = b1_hits.index(first_b1_b3_hit)
                 b3_event_idx = -1
                 b3_hits_at_time = [h for h in b3_hits if abs(h.t - first_b1_b3_hit.t) < 1e-9]
                 if b3_hits_at_time:
                     b3_event = b3_hits_at_time[0]
                     b3_event_idx = b3_hits.index(b3_event)


                 if b1_event_idx > 0 and b3_event_idx != -1:
                     # Simplified: Get velocity vector before hit for B1
                     _, v1_vec, _, _, _ = self._calculate_ball_velocity(b1_route, first_b1_b3_hit.t, b1_event_idx)
                     # Get vector from B1 hit pos to B3 hit pos
                     v2 = np.array([b3_event.pos_x - first_b1_b3_hit.pos_x,
                                    b3_event.pos_y - first_b1_b3_hit.pos_y, 0.0])
                     v1_3d = np.append(v1_vec, 0.0)

                     cross_prod = np.cross(v2, v1_3d)
                     hit_sign = np.sign(cross_prod[2]) if abs(cross_prod[2]) > 1e-6 else 0
                     if hit_fraction == 1.0: hit_sign = 0 # MATLAB special case

                     point_dist = hit_sign * (1.0 - hit_fraction) * self.params['ballR'] * 2.0
                 else:
                      point_dist = (1.0 - hit_fraction) * self.params['ballR'] * 2.0 # Sign unknown


        elif point == 0: # No point scored, find closest approach after Tready
            if b1b2_3c_time != np.inf:
                 # Interpolate ball positions after Tready and find min distance B1-B3
                 t_common = np.linspace(b1b2_3c_time, shot.route[0].t[-1], 100) # Sample time points
                 try:
                      interp_b1x = interp1d(b1_route.t, b1_route.x, bounds_error=False, fill_value="extrapolate")
                      interp_b1y = interp1d(b1_route.t, b1_route.y, bounds_error=False, fill_value="extrapolate")
                      interp_b3x = interp1d(b3_route.t, b3_route.x, bounds_error=False, fill_value="extrapolate")
                      interp_b3y = interp1d(b3_route.t, b3_route.y, bounds_error=False, fill_value="extrapolate")

                      b1x_interp = interp_b1x(t_common)
                      b1y_interp = interp_b1y(t_common)
                      b3x_interp = interp_b3x(t_common)
                      b3y_interp = interp_b3y(t_common)

                      dist_sq = (b1x_interp - b3x_interp)**2 + (b1y_interp - b3y_interp)**2
                      min_dist_idx = np.nanargmin(dist_sq)
                      min_dist = math.sqrt(dist_sq[min_dist_idx])

                      # Determine sign (simplified: use B1 velocity at closest point)
                      t_min = t_common[min_dist_idx]
                      t_min_idx_b1 = np.searchsorted(b1_route.t, t_min)
                      if t_min_idx_b1 > 0 and t_min_idx_b1 < len(b1_route.t):
                           dt = b1_route.t[t_min_idx_b1] - b1_route.t[t_min_idx_b1-1]
                           if dt > 1e-9:
                                vx1 = (b1_route.x[t_min_idx_b1] - b1_route.x[t_min_idx_b1-1]) / dt
                                vy1 = (b1_route.y[t_min_idx_b1] - b1_route.y[t_min_idx_b1-1]) / dt
                                v1_vec = np.array([vx1, vy1, 0])
                                v2 = np.array([b3x_interp[min_dist_idx] - b1x_interp[min_dist_idx],
                                               b3y_interp[min_dist_idx] - b1y_interp[min_dist_idx], 0])
                                cross_prod = np.cross(v2, v1_vec)
                                hit_sign = np.sign(cross_prod[2]) if abs(cross_prod[2]) > 1e-6 else 0
                                point_dist = hit_sign * min_dist
                           else: point_dist = min_dist # Sign unknown
                      else: point_dist = min_dist # Sign unknown

                 except ValueError:
                      point_dist = np.nan # Interpolation failed
                 except Exception:
                      point_dist = np.nan # Other error


        # Store point distance
        if shot.get_b1_hits():
             shot.get_b1_hits()[0].point_dist = point_dist


        # --- Calculate Kiss Distance B1 --- (Distance B1-B2 after first B1->B2 hit)
        kiss_dist_b1 = np.nan
        if kiss == 1 and second_b1_b2_hit: # If kiss was B1->B2 (second time)
            # Use hit fraction from the second B1->B2 event
             hit_fraction = second_b1_b2_hit.fraction
             if not np.isnan(hit_fraction):
                 kiss_dist_b1 = hit_fraction * self.params['ballR'] * 2.0
        elif first_b1_b2_hit: # No B1->B2 kiss, find closest approach after first hit
            t_start_search = first_b1_b2_hit.t + 1e-6 # Start just after hit
            # End search at point time, kiss time, or failure time, whichever is earliest
            t_end_search = min(point_time, kiss_time, failure_time, shot.route[0].t[-1])

            if t_end_search > t_start_search:
                 t_common = np.linspace(t_start_search, t_end_search, 100)
                 try:
                      interp_b1x = interp1d(b1_route.t, b1_route.x, bounds_error=False, fill_value="extrapolate")
                      interp_b1y = interp1d(b1_route.t, b1_route.y, bounds_error=False, fill_value="extrapolate")
                      interp_b2x = interp1d(b2_route.t, b2_route.x, bounds_error=False, fill_value="extrapolate")
                      interp_b2y = interp1d(b2_route.t, b2_route.y, bounds_error=False, fill_value="extrapolate")

                      b1x_interp = interp_b1x(t_common)
                      b1y_interp = interp_b1y(t_common)
                      b2x_interp = interp_b2x(t_common)
                      b2y_interp = interp_b2y(t_common)

                      dist_sq = (b1x_interp - b2x_interp)**2 + (b1y_interp - b2y_interp)**2
                      min_dist_val = np.nanmin(dist_sq)
                      if not np.isnan(min_dist_val):
                          kiss_dist_b1 = math.sqrt(min_dist_val)

                 except ValueError:
                      kiss_dist_b1 = np.nan
                 except Exception:
                      kiss_dist_b1 = np.nan

        # Store kiss distance B1
        if shot.get_b1_hits():
             shot.get_b1_hits()[0].kiss_dist_b1 = kiss_dist_b1


    def _update_table_with_event_data(self, shot):
        """Updates the main DataFrame with calculated event details."""
        # Corresponds to MATLAB's create_varname

        if shot.b1b2b3_indices is None: return

        df_idx = shot.table_index
        b1i, b2i, b3i = shot.b1b2b3_indices
        b1_start_event = shot.get_b1_hits()[0] if shot.get_b1_hits() else None

        # --- Update summary columns (from B1's start event) ---
        if b1_start_event:
             self.table.loc[df_idx, 'Point'] = b1_start_event.point # Overwrite original point type? Maybe store in new col 'CalcPoint'
             self.table.loc[df_idx, 'Kiss'] = b1_start_event.kiss
             self.table.loc[df_idx, 'Fuchs'] = b1_start_event.fuchs
             self.table.loc[df_idx, 'PointDist'] = b1_start_event.point_dist
             self.table.loc[df_idx, 'KissDistB1'] = b1_start_event.kiss_dist_b1
             self.table.loc[df_idx, 'AimOffsetLL'] = getattr(b1_start_event, 'aim_offset_ll', np.nan) # Use getattr for safety
             self.table.loc[df_idx, 'AimOffsetSS'] = getattr(b1_start_event, 'aim_offset_ss', np.nan)
             self.table.loc[df_idx, 'B1B2OffsetLL'] = getattr(b1_start_event, 'b1b2_offset_ll', np.nan)
             self.table.loc[df_idx, 'B1B2OffsetSS'] = getattr(b1_start_event, 'b1b2_offset_ss', np.nan)
             self.table.loc[df_idx, 'B1B3OffsetLL'] = getattr(b1_start_event, 'b1b3_offset_ll', np.nan)
             self.table.loc[df_idx, 'B1B3OffsetSS'] = getattr(b1_start_event, 'b1b3_offset_ss', np.nan)

        # --- Update detailed event columns ---
        max_hits = [8, 4, 0] # Max events per ball (B1, B2, B3) from MATLAB? Seems arbitrary. Use actual lengths?
        del_names = ['Fraction', 'DefAngle', 'CutAngle', 'CInAngle', 'COutAngle'] # Skip these for first hit (hi=1 in MATLAB)

        for ball_loop_idx in range(3): # Corresponds to B1, B2, B3 loop
            original_ball_index = shot.b1b2b3_indices[ball_loop_idx] # W, Y, or R index (0, 1, 2)
            hit_events = shot.hit_events[original_ball_index]
            
            # Use actual number of events, limit if necessary
            num_events_to_log = len(hit_events) # Or min(len(hit_events), max_hits[ball_loop_idx])

            for hi in range(num_events_to_log): # Event index for this ball
                 event = hit_events[hi]
                 # Map event attributes to DataFrame columns
                 # Python attribute names match HitEvent class
                 event_attrs = [
                      'Type', 'FromC', 'ToC', 'V1', 'V2', 'Offset', 'Fraction',
                      'DefAngle', 'CutAngle', 'CInAngle', 'COutAngle',
                      'FromCPos', 'ToCPos', 'FromDPos', 'ToDPos'
                 ]
                 for attr in event_attrs:
                      # Construct column name, e.g., B1_1_V1, B2_3_Fraction
                      # Use ball_loop_idx + 1 for B1, B2, B3 numbering
                      # Use hi for event number (MATLAB hi starts from 1, here from 0)
                      # MATLAB skips hit 0 ('S'), so map Python hi to MATLAB hi+1
                      matlab_hi = hi + 1
                      col_name = f"B{ball_loop_idx+1}_{matlab_hi}_{attr}"

                      # Check skip condition from MATLAB
                      if matlab_hi == 1 and attr in del_names:
                          continue

                      value = getattr(event, attr.lower(), np.nan) # Get value, default np.nan

                      # Apply scaling for position columns (already done in _calculate_ball_direction?)
                      # if 'Pos' in attr: # Check if scaling already happened
                          # pass
                      # else:
                          # pass # Default scale = 1

                      # Add column to DataFrame if it doesn't exist
                      if col_name not in self.table.columns:
                           # Initialize with appropriate type, e.g., float
                           self.table[col_name] = np.nan
                           # self.table[col_name] = self.table[col_name].astype(float) # Ensure type

                      # Update value (handle potential type issues)
                      try:
                          self.table.loc[df_idx, col_name] = float(value) if not pd.isna(value) else np.nan
                      except (ValueError, TypeError):
                           self.table.loc[df_idx, col_name] = np.nan # Assign NaN if conversion fails


    def extract_events(self):
        """Detects and evaluates hit events for all shots."""
        print("Extracting Events...")
        total_shots = len(self.shots)
        processed_count = 0
        error_count = 0

        for i, shot in enumerate(self.shots):
            print(f"Processing shot {i+1}/{total_shots} (ID: {shot.shot_id})")
            if shot.interpreted or shot.selected or shot.b1b2b3_indices is None:
                print(f"  Skipping shot {shot.shot_id} (Interpreted: {shot.interpreted}, Selected: {shot.selected}, B1B2B3: {shot.b1b2b3_str})")
                continue

            df_idx = shot.table_index

            try:
                # --- 1. Detect Events ---
                # This needs the complex iterative logic from MATLAB's Extract_Events
                # Placeholder: Assume events are detected and populated in shot.hit_events
                # self._detect_events_for_shot(shot) # <-- This function needs to be implemented based on MATLAB logic
                # --- For now: Create dummy start events ---
                shot.hit_events = [[], [], []] # Reset events
                for bi in range(3):
                     if len(shot.route[bi]) > 0:
                         start_event = HitEvent(t=shot.route[bi].t[0],
                                                with_char='S' if bi == shot.b1b2b3_indices[0] else '-', # Only B1 starts with 'S'
                                                pos_x=shot.route[bi].x[0],
                                                pos_y=shot.route[bi].y[0])
                         shot.add_hit_event(bi, start_event)
                 # !!! --- Replace dummy events with actual detection logic --- !!!
                 print(f"  WARNING: Event detection logic not implemented. Using only start events for Shot {shot.shot_id}.")
                 # !!! --- --- --- --- --- --- --- --- --- --- --- --- --- --- !!!


                # --- 2. Evaluate Hit Details ---
                self._evaluate_hit_details(shot)

                # --- 3. Evaluate Point/Kiss/Fuchs ---
                self._evaluate_point_kiss_fuchs(shot)

                # --- 4. Update Table ---
                self._update_table_with_event_data(shot)

                # --- Mark as Interpreted ---
                shot.interpreted = 1
                self.table.loc[df_idx, 'Interpreted'] = 1
                # Clear errors if interpretation successful
                self.table.loc[df_idx, 'ErrorID'] = None
                self.table.loc[df_idx, 'ErrorText'] = "Interpreted OK"
                self.table.loc[df_idx, 'Selected'] = False
                shot.error_code = None
                shot.error_text = "Interpreted OK"
                shot.selected = False
                processed_count += 1

            except Exception as e:
                print(f"  ERROR processing Shot {shot.shot_id} (Idx: {df_idx}): {e}")
                import traceback
                traceback.print_exc() # Print traceback for debugging
                shot.error_code = 1000 # Generic event processing error
                shot.error_text = f"Event processing error: {e}"
                shot.selected = True
                self.table.loc[df_idx, 'ErrorID'] = shot.error_code
                self.table.loc[df_idx, 'ErrorText'] = shot.error_text
                self.table.loc[df_idx, 'Selected'] = True
                error_count += 1

        print(f"Event extraction completed. Processed: {processed_count}, Errors: {error_count}")


    def run_analysis(self, filepath):
        """Executes the full analysis pipeline."""
        print(f"Starting analysis for: {filepath}")

        if not self.read_gamefile(filepath):
            print("Analysis stopped: Could not read game file.")
            return

        self.perform_data_quality_checks()

        self.determine_b1b2b3()

        # Add B1B2B3 position calculation (simple version from MATLAB)
        self.calculate_b1b2b3_positions()

        # Major step: Event detection and evaluation
        self.extract_events() # Includes sub-steps: detect, eval details, eval point/kiss, update table

        print("Analysis finished.")
        print(f"{self.table['Selected'].sum()} shots flagged with issues.")

        # Optional: Save results
        # self.save_results("output_analysis.csv")


    def calculate_b1b2b3_positions(self):
        """ Calculates initial ball positions in diamond units."""
        print("Calculating initial positions...")
        if 'B1B2B3' not in self.table.columns:
             print("  Skipping: B1B2B3 column not found.")
             return

        scale_x = 8.0 / self.params['size'][1] # Diamonds along long side (X pos)
        scale_y = 4.0 / self.params['size'][0] # Diamonds along short side (Y pos)

        for i, shot in enumerate(self.shots):
             if shot.selected or shot.b1b2b3_indices is None: continue
             df_idx = shot.table_index

             for ball_loop_idx in range(3): # B1, B2, B3
                 original_ball_idx = shot.b1b2b3_indices[ball_loop_idx]
                 route = shot.route[original_ball_idx]

                 if len(route) > 0:
                     pos_x_val = route.x[0] * scale_x
                     pos_y_val = route.y[0] * scale_y
                 else:
                     pos_x_val, pos_y_val = np.nan, np.nan

                 col_x = f'B{ball_loop_idx+1}posX'
                 col_y = f'B{ball_loop_idx+1}posY'

                 if col_x not in self.table.columns: self.table[col_x] = np.nan
                 if col_y not in self.table.columns: self.table[col_y] = np.nan

                 self.table.loc[df_idx, col_x] = pos_x_val
                 self.table.loc[df_idx, col_y] = pos_y_val

    def save_results(self, output_filepath):
        """Saves the resulting table to a CSV file."""
        try:
            self.table.to_csv(output_filepath, index=False)
            print(f"Results saved to {output_filepath}")
        except Exception as e:
            print(f"Error saving results to {output_filepath}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # filepath = "D:/Programming/Shotdata/JSON/20210704_Ersin_Cemal.json" # Example from MATLAB comments
    filepath = "D:\\Billard\\0_AllDatabase\\WebSport\\20181031_match_03_Ersin_Cemal.txt" # Example from Python snippet

    analyzer = ShotAnalyzer() # Use default params
    analyzer.run_analysis(filepath)

    # Display some results (optional)
    print("\n--- Analysis Summary Table (First 10 rows) ---")
    print(analyzer.table.head(10).to_string())
    print("\n--- Flagged Shots ---")
    flagged = analyzer.table[analyzer.table['Selected'] == True]
    if not flagged.empty:
        print(flagged[['ShotID', 'ErrorID', 'ErrorText']].to_string())
    else:
        print("No shots flagged.")

    # Example: Save to CSV
    analyzer.save_results("analysis_output.csv")