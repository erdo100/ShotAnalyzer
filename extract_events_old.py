import numpy as np
from scipy.interpolate import interp1d
import json
import os
import pandas as pd
import copy
import warnings # Used to suppress potential division-by-zero warnings if dt is zero
from str2num_b1b2b3 import str2num_b1b2b3
from angle_vector import angle_vector
import matplotlib.pyplot as plt




def extract_events(SA, si, param):
    """
    Extracts all ball-ball hit events and ball-Cushion hit events for a given shot.
    Orchestrates the event extraction process using helper functions.

    Args:
        SA (dict): The main data structure containing 'Shot' and 'Table'.
        si (int): The index of the shot to process.
        param (dict): Dictionary of parameters.

    Returns:
        tuple: (hit (list), err (dict))
               hit: List of dictionaries, one for each ball, containing hit events.
               err: Dictionary with error code and text.
    """
    col = 'WYR'
    BB_pairs = np.array([[1, 2], [1, 3], [2, 3]]) # Roles (1st, 2nd, 3rd ball)

    # Initialize the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ball0_line, = ax.plot([], [], 'bo-', label='Ball 0')  # Line for Ball 0
    appr_line, = ax.plot([], [], 'k+', label='Ball 0')  # Line for Ball 0
    
    ax.set_ylim(0, 2840)  # Set appropriate limits for your data
    ax.set_xlim(0, 1420)
    #show axis real size
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    
    # --- Initialization Helper ---
    def _initialize_event_data(SA, si, param):
        """Initializes data structures needed for event extraction."""
        err_init = {'code': None, 'text': ''}
        hit_init = [
            {'with': [], 't': [], 'XPos': [], 'YPos': [], 'Kiss': [], 'Point': [], 'Fuchs': [],
                'PointDist': [], 'KissDistB1': [], 'Tpoint': [], 'Tkiss': [], 'Tready': [],
                'TB2hit': [], 'Tfailure': []} for _ in range(3)
        ]
        ball0_init = [None] * 3
        ball_init = [{} for _ in range(3)]
        Tall0_init = np.array([])

        # Get B1,B2,B3 order (0-based indices)
        b1b2b3_str = SA['Table'].iloc[si]['B1B2B3']
        b1b2b3_num, b1i, b2i, b3i = str2num_b1b2b3(b1b2b3_str)
        if b1i is None: # Error in str2num_b1b2b3
            err_init['code'] = 3
            err_init['text'] = f'ERROR: Invalid B1B2B3 string "{b1b2b3_str}" for shot index {si}.'
            print(err_init['text'])
            # Return initialized but potentially empty structures and the error
            return ball0_init, ball_init, hit_init, Tall0_init, err_init, (None, None, None, None)

        # Deep copy original data and set up initial state
        t_all_list = []
        for bi_idx in range(3):
            ball0_init[bi_idx] = copy.deepcopy(SA['Shot'][si]['Route0'][bi_idx])
            if 't' in ball0_init[bi_idx] and len(ball0_init[bi_idx]['t']) > 0:
                    ball_init[bi_idx]['t'] = np.array([ball0_init[bi_idx]['t'][0]]) # Start time only initially
                    ball_init[bi_idx]['x'] = np.array([ball0_init[bi_idx]['x'][0]])
                    ball_init[bi_idx]['y'] = np.array([ball0_init[bi_idx]['y'][0]])
                    t_all_list.append(np.ravel(ball0_init[bi_idx]['t']))
            else: # Handle empty initial data for a ball
                    ball_init[bi_idx]['t'] = np.array([])
                    ball_init[bi_idx]['x'] = np.array([])
                    ball_init[bi_idx]['y'] = np.array([])
                    # Add a placeholder 't' array to ball0 if missing, maybe with start time if available elsewhere?
                    # For now, assume read_gamefile ensures structure exists, but maybe empty arrays.
                    if 't' not in ball0_init[bi_idx]: ball0_init[bi_idx]['t'] = np.array([])
                    if 'x' not in ball0_init[bi_idx]: ball0_init[bi_idx]['x'] = np.array([])
                    if 'y' not in ball0_init[bi_idx]: ball0_init[bi_idx]['y'] = np.array([])


        # Create common Time points from valid data
        if t_all_list:
                t_all = np.concatenate(t_all_list)
                Tall0_init = np.unique(t_all)
                Tall0_init = Tall0_init[Tall0_init >= Tall0_init[0]] # Ensure non-negative relative to start
                Tall0_init = np.sort(Tall0_init)
        else:
                # Handle case where NO ball has any time data
                err_init['code'] = 5 # New error code
                err_init['text'] = f'ERROR: No time data found for any ball in shot index {si}.'
                print(err_init['text'])
                # Fallback: maybe create a dummy Tall0? Or just return error.
                Tall0_init = np.array([0.0]) # Minimal time vector

        # Initialize hit structure lists (start event)
        # Ensure initial positions exist before accessing
        if len(ball0_init[b1i]['x']) > 0:
            hit_init[b1i]['with'].append('S'); hit_init[b1i]['t'].append(Tall0_init[0]);
            hit_init[b1i]['XPos'].append(ball0_init[b1i]['x'][0]); hit_init[b1i]['YPos'].append(ball0_init[b1i]['y'][0])
        if len(ball0_init[b2i]['x']) > 0:
            hit_init[b2i]['with'].append('-'); hit_init[b2i]['t'].append(Tall0_init[0]);
            hit_init[b2i]['XPos'].append(ball0_init[b2i]['x'][0]); hit_init[b2i]['YPos'].append(ball0_init[b2i]['y'][0])
        if len(ball0_init[b3i]['x']) > 0:
            hit_init[b3i]['with'].append('-'); hit_init[b3i]['t'].append(Tall0_init[0]);
            hit_init[b3i]['XPos'].append(ball0_init[b3i]['x'][0]); hit_init[b3i]['YPos'].append(ball0_init[b3i]['y'][0])
        # Add default values for other fields if needed, matching original code...
        for bi_idx in [b1i, b2i, b3i]:
            if hit_init[bi_idx]['t']: # Only if start was added
                for key in ['Kiss', 'Point', 'Fuchs']: hit_init[bi_idx][key].append(0)
                for key in ['PointDist', 'KissDistB1', 'Tpoint', 'Tkiss', 'Tready', 'TB2hit', 'Tfailure']: hit_init[bi_idx][key].append(3000.0)


        # Calculate initial velocities/derivatives
        _update_ball0_derivatives(ball0_init)

        # Set initial velocities for B2, B3 to zero
        for idx_zero in [b2i, b3i]:
            if len(ball0_init[idx_zero].get('vx', [])) > 0: ball0_init[idx_zero]['vx'][0] = 0.0
            if len(ball0_init[idx_zero].get('vy', [])) > 0: ball0_init[idx_zero]['vy'][0] = 0.0
            if len(ball0_init[idx_zero].get('v', [])) > 0:  ball0_init[idx_zero]['v'][0] = 0.0

        return ball0_init, ball_init, hit_init, Tall0_init, err_init, (b1b2b3_num, b1i, b2i, b3i)
    # --- End Initialization Helper ---

    # --- Derivative Calculation Helper ---
    def _update_ball0_derivatives(ball0_data):
        """Calculates/updates vx, vy, v, dt in the ball0_data structure for a ball."""
        for bi_idx in range(3):

            if len(ball0_data[bi_idx].get('t', [])) >= 2:
                t = ball0_data[bi_idx]['t']
                x = ball0_data[bi_idx]['x']
                y = ball0_data[bi_idx]['y']
                dt = np.diff(t)
                # Prevent division by zero or negative dt
                valid_dt = dt > np.finfo(float).eps
                dt_safe = np.where(valid_dt, dt, np.finfo(float).eps)

                vx = np.zeros_like(t)
                vy = np.zeros_like(t)
                vx[:-1] = np.diff(x) / dt_safe
                vy[:-1] = np.diff(y) / dt_safe

                # Append last dt for consistent array length, handle single diff case
                dt_full = np.append(dt, dt[-1] if len(dt) > 0 else 0)

                ball0_data[bi_idx]['dt'] = dt_full
                ball0_data[bi_idx]['vx'] = vx
                ball0_data[bi_idx]['vy'] = vy
                ball0_data[bi_idx]['v'] = np.sqrt(vx**2 + vy**2)
            else: # Handle cases with less than 2 points
                ball0_data[bi_idx]['dt'] = np.zeros(len(ball0_data[bi_idx].get('t', [])))
                ball0_data[bi_idx]['vx'] = np.zeros(len(ball0_data[bi_idx].get('t', [])))
                ball0_data[bi_idx]['vy'] = np.zeros(len(ball0_data[bi_idx].get('t', [])))
                ball0_data[bi_idx]['v']  = np.zeros(len(ball0_data[bi_idx].get('t', [])))
        # --- End Derivative Calculation Helper ---

    # --- Approximation Helper ---
    def _calculate_approximations(ball, ball0, hit, param, Tall0, tvec):
        # Initialize output structure with 0-based indexing
        b = {0: {}, 1: {}, 2: {}}
        
        dT = np.diff(Tall0[:2])[0]  # Time difference between first two elements
        tappr = Tall0[0] + dT * tvec  # Approximate time vector
        
        for bi in range(3):  # 0-based ball indices (0, 1, 2)
            if len(ball0[bi]['t']) >= 2:
                # Travel distance calculation (using first two elements)
                if len(ball[bi]['x']) >= 2:
                    dx = ball[bi]['x'][1] - ball[bi]['x'][0]
                    dy = ball[bi]['y'][1] - ball[bi]['y'][0]
                    ds0 = np.sqrt(dx**2 + dy**2)
                    v0 = ds0 / dT
                
                # Initialize velocity components
                b[bi]['vt1'] = ball0[bi]['v'][0]  # First velocity component
                
                # Find previous hits (0-based indices)
                last_hit_time = hit[bi]['t'][-1]
                valid_indices = np.where(ball[bi]['t'] >= last_hit_time)[0]
                
                # Get up to timax_appr last indices
                iprev = valid_indices[-param['timax_appr']:] if valid_indices.size > 0 else []
                
                # Velocity component determination
                if hit[bi]['with'] == '-':
                    b[bi]['v1'] = [0, 0]
                    b[bi]['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
                elif len(iprev) == 1:
                    b[bi]['v1'] = [ball0[bi]['vx'][0], ball0[bi]['vy'][0]]
                    b[bi]['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
                else:
                    # Linear regression for velocity extrapolation
                    if len(iprev) > 0:
                        A = np.vstack([ball[bi]['t'][iprev], np.ones_like(ball[bi]['t'][iprev])]).T
                        CoefVX = np.linalg.lstsq(A, ball[bi]['x'][iprev], rcond=None)[0]
                        CoefVY = np.linalg.lstsq(A, ball[bi]['y'][iprev], rcond=None)[0]
                    else:
                        CoefVX = [0, 0]
                        CoefVY = [0, 0]
                    
                    b[bi]['v1'] = [CoefVX[0], CoefVY[0]]
                    b[bi]['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
                
                # Velocity magnitude calculations
                b[bi]['vt1'] = np.linalg.norm(b[bi]['v1'])
                b[bi]['vt2'] = np.linalg.norm(b[bi]['v2'])
                vtnext = max(b[bi]['vt1'], ball0[bi]['v'][0])
                
                # Normalize velocity vector if needed
                if b[bi]['vt1'] > 1e-6:  # Avoid division by zero
                    vnext = (np.array(b[bi]['v1']) / b[bi]['vt1']) * vtnext
                else:
                    vnext = np.zeros(2)
                
                # Position approximation using last known position
                last_x = ball[bi]['x'][-1]
                last_y = ball[bi]['y'][-1]
                b[bi]['xa'] = last_x + vnext[0] * dT * tvec
                b[bi]['ya'] = last_y + vnext[1] * dT * tvec
        
        return b
    # --- End Approximation Helper ---

    # --- Distance Calculation Helpers ---
    def _calculate_ball_ball_distances(b_approx, b1b2b3_indices, BB_pairs, param, tvec):
        """Calculates ball-ball distances based on approximated positions."""
        distances = [{} for _ in range(len(BB_pairs))]
        ballR = param.get('ballR', 0)
        b1i, b2i, b3i = b1b2b3_indices # Unpack 0-based indices

        for bbi, roles in enumerate(BB_pairs):
            # Map role (1, 2, 3) to actual 0-based ball index (W=0, Y=1, R=2)
            # This needs the b1b2b3_num mapping if roles are 1,2,3
            # Let's assume b1b2b3_indices contains the mapping: e.g., (0, 1, 2) for WYR
            idx1 = b1b2b3_indices[roles[0]-1] # Role 1 -> index idx1
            idx2 = b1b2b3_indices[roles[1]-1] # Role 2 -> index idx2

            xa1 = b_approx[idx1].get('xa', np.full_like(tvec, np.nan))
            ya1 = b_approx[idx1].get('ya', np.full_like(tvec, np.nan))
            xa2 = b_approx[idx2].get('xa', np.full_like(tvec, np.nan))
            ya2 = b_approx[idx2].get('ya', np.full_like(tvec, np.nan))

            if xa1.shape == xa2.shape and ya1.shape == ya2.shape:
                 # Distance between centers minus diameter
                 dist_centers = np.sqrt((xa1 - xa2)**2 + (ya1 - ya2)**2)
                 distances[bbi]['BB'] = dist_centers - (2 * ballR)
            else:
                 distances[bbi]['BB'] = np.full_like(tvec, np.nan)
        return distances

    def _calculate_cushion_distances(b_approx, param, tvec):
        """Calculates ball-cushion distances and adds them to b_approx."""
        ballR = param.get('ballR', 0)
        table_size = param.get('size', [np.nan, np.nan]) # Expects [height, width]
        height, width = table_size[0], table_size[1]

        for bi_idx in range(3):
            xa = b_approx[bi_idx].get('xa', np.full_like(tvec, np.nan))
            ya = b_approx[bi_idx].get('ya', np.full_like(tvec, np.nan))

            # cd shape: (n_points, 4) -> [Bottom, Right, Top, Left] distances
            cd = np.full((len(tvec), 4), np.nan)
            cd[:,0] = ya - ballR                      # Dist to Bottom edge
            cd[:,1] = (width - ballR) - xa            # Dist to Right edge
            cd[:,2] = (height - ballR) - ya           # Dist to Top edge
            cd[:,3] = xa - ballR                      # Dist to Left edge
            b_approx[bi_idx]['cd'] = cd
        # No return needed as b_approx is modified in place
    # --- End Distance Calculation Helpers ---

    # --- Hit Detection Helpers ---
    def _find_potential_cushion_hits(bi_idx, b_data, tappr, param):
        """Finds potential cushion hits for a single ball."""
        hits = []
        ballR = param.get('ballR', 0)
        table_size = param.get('size', [np.nan, np.nan]) # [height, width]
        height, width = table_size[0], table_size[1]
        cd_all = b_data.get('cd')
        a12 = b_data.get('a12', np.nan)
        v1 = b_data.get('v1', np.array([0.0, 0.0]))

        # Check if angle indicates change or start, and if cushion data exists
        if (a12 > 1 or a12 == -1) and cd_all is not None:
            for cii in range(4): # 0:Bottom, 1:Right, 2:Top, 3:Left
                cd_col = cd_all[:, cii]
                # Check if distance crosses zero (potential hit)
                if np.any(cd_col <= 0): # Simplified check (original was just <=0)
                    # Check velocity direction towards cushion
                    vel_towards_cushion = (
                        (cii == 0 and v1[1] < -np.finfo(float).eps) or # Moving Down
                        (cii == 1 and v1[0] > np.finfo(float).eps)  or # Moving Right
                        (cii == 2 and v1[1] > np.finfo(float).eps)  or # Moving Up
                        (cii == 3 and v1[0] < -np.finfo(float).eps)     # Moving Left
                    )

                    if vel_towards_cushion:
                        # Interpolate time of hit (where distance is 0)
                        # Ensure valid inputs for interpolation
                        finite_mask = np.isfinite(cd_col) & np.isfinite(tappr)
                        if np.any(finite_mask) and len(tappr[finite_mask]) > 1 and len(cd_col[finite_mask]) > 1 and \
                           (np.min(cd_col[finite_mask]) <= 0 <= np.max(cd_col[finite_mask])): # Check if 0 is within range
                            try:
                                # Sort by cd_col to ensure monotonicity for interp
                                sort_idx = np.argsort(cd_col[finite_mask])
                                tc = np.interp(0, cd_col[finite_mask][sort_idx], tappr[finite_mask][sort_idx])

                                # Check if interpolated time is within the approximation interval
                                if tappr[0] <= tc <= tappr[-1]:
                                    # Interpolate position at hit time tc
                                    xa_interp = np.interp(tc, tappr[finite_mask], b_data['xa'][finite_mask])
                                    ya_interp = np.interp(tc, tappr[finite_mask], b_data['ya'][finite_mask])

                                    # Define hit position precisely on cushion edge
                                    cushx, cushy = np.nan, np.nan
                                    if cii == 0: cushx, cushy = xa_interp, ballR         # Bottom
                                    elif cii == 1: cushx, cushy = width - ballR, ya_interp # Right
                                    elif cii == 2: cushx, cushy = xa_interp, height - ballR # Top
                                    elif cii == 3: cushx, cushy = ballR, ya_interp      # Left

                                    if not (np.isnan(cushx) or np.isnan(cushy)):
                                        # [time, ball_idx, type(2=cush), id(0-3), x, y]
                                        hits.append([tc, bi_idx, 2, cii, cushx, cushy])
                                        print(f"Potential cushion hit detected: {hits[-1]}") # Debugging output
                                        
                            except Exception as e:
                                # print(f"Warning: Cushion hit interpolation failed for ball {bi_idx}, cushion {cii}: {e}") # Removed verbosity
                                pass # Ignore interpolation errors
        return hits

    def _find_potential_ball_ball_hits(b_approx, b, d, tappr, hit, tc):
        """Finds potential ball-ball hits for a single pair."""
        BB = np.array([[0, 1], [0, 2], [1, 2]])  # Ball pairs (0-based)
        hitlist = []

        for bbi in range(3):  # Iterate through all ball pairs
            bx1, bx2 = BB[bbi]
            d_bbi = d[bbi]['BB']
            checkdist = 0
            tc = None

            # Check distance crossing conditions
            has_negative = np.any(d_bbi <= 0)
            has_positive = np.any(d_bbi > 0)
            
            if has_negative and has_positive:
                # Case 1: Check first and last elements for zero crossing
                if (d_bbi[0] >= 0 and d_bbi[-1] < 0) or (d_bbi[0] < 0 and d_bbi[-1] >= 0):
                    f = interp1d(d_bbi, tappr, kind='linear', fill_value='extrapolate')
                    tc = float(f(0))
                    checkdist = 1
                else:
                    # Case 2: Find internal zero crossings
                    sign_changes = np.diff(np.sign(d_bbi)) != 0
                    if np.any(sign_changes):
                        ind = np.where(np.diff(d_bbi) <= 0)[0]
                        if ind.size > 0:
                            f = interp1d(d_bbi[ind], tappr[ind], kind='linear', fill_value='extrapolate')
                            tc = float(f(0))
                            checkdist = 1

            # Angle checks (using get() for safety)
            a12_b1 = b[bx1].get('a12', -1)
            a12_b2 = b[bx2].get('a12', -1)
            checkangle_b1 = (a12_b1 > 10) or (a12_b1 == -1)
            checkangle_b2 = (a12_b2 > 10) or (a12_b2 == -1)

            # Velocity checks
            eps = np.finfo(float).eps
            vt1_b1, vt2_b1 = b[bx1].get('vt1', 0), b[bx1].get('vt2', 0)
            vt1_b2, vt2_b2 = b[bx2].get('vt1', 0), b[bx2].get('vt2', 0)
            
            checkvel_b1 = abs(vt2_b1 - vt1_b1) > 50 if any(abs(v) > eps for v in [vt1_b1, vt2_b1]) else False
            checkvel_b2 = abs(vt2_b2 - vt1_b2) > 50 if any(abs(v) > eps for v in [vt1_b2, vt2_b2]) else False

            # Time collision check
            checkdouble = False
            if tc is not None:
                last_hit_b1 = hit[bx1]['t'][-1] if hit[bx1]['t'] else -np.inf
                last_hit_b2 = hit[bx2]['t'][-1] if hit[bx2]['t'] else -np.inf
                checkdouble = (tc >= last_hit_b1 + 0.01) and (tc >= last_hit_b2 + 0.01)

            # Final collision check
            if checkdist and checkdouble:
                # Create interpolators for positions
                interp_x1 = interp1d(tappr, b[bx1]['xa'], fill_value='extrapolate')
                interp_y1 = interp1d(tappr, b[bx1]['ya'], fill_value='extrapolate')
                interp_x2 = interp1d(tappr, b[bx2]['xa'], fill_value='extrapolate')
                interp_y2 = interp1d(tappr, b[bx2]['ya'], fill_value='extrapolate')

                # Add entries to hitlist
                hitlist.append([
                    tc, bx1, 1, bx2,
                    float(interp_x1(tc)),
                    float(interp_y1(tc))
                ])
                hitlist.append([
                    tc, bx2, 1, bx1,
                    float(interp_x2(tc)),
                    float(interp_y2(tc))
                ])
                    
        # --- Special Case: Low Velocity B1 Hit (Original MATLAB Logic Section) ---
        len_hit_b2i = len(hit[b2i]['t']) if b2i is not None else 0
        len_t_b1i = len(ball0[b1i]['t']) if b1i is not None else 0
        len_t_b2i = len(ball0[b2i]['t']) if b2i is not None else 0
        t2_b1i = ball0[b1i]['t'][1] if len_t_b1i > 1 else np.inf
        t2_b2i = ball0[b2i]['t'][1] if len_t_b2i > 1 else np.inf
        x1_b2i = ball0[b2i]['x'][0] if len_t_b2i > 0 else np.nan
        y1_b2i = ball0[b2i]['y'][0] if len_t_b2i > 0 else np.nan
        x2_b2i = ball0[b2i]['x'][1] if len_t_b2i > 1 else np.nan
        y2_b2i = ball0[b2i]['y'][1] if len_t_b2i > 1 else np.nan

        # Check if this pair is B1 hitting B2 (roles 1 and 2)
        is_b1_hitting_b2 = (idx1 == b1i and idx2 == b2i) or (idx1 == b2i and idx2 == b1i)

        if is_b1_hitting_b2 and not first_step and len_hit_b2i == 1 and not np.isnan(x1_b2i) and \
           (x1_b2i != x2_b2i or y1_b2i != y2_b2i): # B2 moved slightly but no hit detected yet

             # Original logic used interpolation, but if tc is already nan, find minimum distance time
             if np.isnan(tc):
                 if np.any(np.isfinite(dist_bb)):
                      min_dist_idx = np.nanargmin(dist_bb)
                      if dist_bb[min_dist_idx] < ballR: # Check if minimum distance is reasonably close
                          tc = tappr[min_dist_idx]
                          if not (tappr[0] <= tc <= tappr[-1]): tc = np.nan # Reset if outside bounds

        # --- Special Case: First Time Step Equal Start Times ---
        if first_step and t2_b1i == t2_b2i and not np.isnan(x1_b2i) and \
           (x1_b2i != x2_b2i or y1_b2i != y2_b2i):
             vec_b2dir = np.array([x2_b2i - x1_b2i, y2_b2i - y1_b2i])
             norm_vec_b2dir = np.linalg.norm(vec_b2dir)
             if norm_vec_b2dir > np.finfo(float).eps:
                  # Estimate B1 position at contact based on B2's first step
                  b1_hitpos_est = np.array([x1_b2i, y1_b2i]) - vec_b2dir / norm_vec_b2dir * (2 * ballR)
                  # Estimate time as halfway through the first step
                  tc_first_step = (ball0[b1i]['t'][0] + t2_b1i) / 2.0 # Use absolute time if available

                  # If standard interpolation didn't find a hit, use this estimate
                  if np.isnan(tc) or tc_first_step < tc:
                       tc = tc_first_step
                       # We need hit positions for B1 and B2 at this estimated time
                       hit_x_b1_est, hit_y_b1_est = b1_hitpos_est[0], b1_hitpos_est[1]
                       hit_x_b2_est, hit_y_b2_est = x1_b2i, y1_b2i # B2 hasn't moved yet at contact

                       # Add estimated hit - ensure correct ball indices (idx1, idx2) are used
                       # Find which index (idx1 or idx2) corresponds to b1i and b2i
                       if idx1 == b1i: # idx1 is B1, idx2 is B2
                            hits.append([ tc, idx1, 1, idx2, hit_x_b1_est, hit_y_b1_est ]) # B1 hits B2
                            hits.append([ tc, idx2, 1, idx1, hit_x_b2_est, hit_y_b2_est ]) # B2 is hit by B1
                       else: # idx1 is B2, idx2 is B1
                            hits.append([ tc, idx2, 1, idx1, hit_x_b1_est, hit_y_b1_est ]) # B1 hits B2
                            hits.append([ tc, idx1, 1, idx2, hit_x_b2_est, hit_y_b2_est ]) # B2 is hit by B1
                       return hits # Return early as we have the first step hit


        # --- Finalize Standard Hit ---
        if not np.isnan(tc):
             # Check if hit time is significantly after the last hit for *both* balls involved
             last_hit_t_idx1 = hit[idx1]['t'][-1] if len(hit[idx1]['t']) > 0 else -np.inf
             last_hit_t_idx2 = hit[idx2]['t'][-1] if len(hit[idx2]['t']) > 0 else -np.inf
             time_threshold = 0.001 # Small threshold to prevent re-detecting same hit

             if tc > last_hit_t_idx1 + time_threshold and tc > last_hit_t_idx2 + time_threshold:
                 # Interpolate positions for both balls at time tc
                 xa_idx1 = b_approx[idx1].get('xa', np.full_like(tappr, np.nan))
                 ya_idx1 = b_approx[idx1].get('ya', np.full_like(tappr, np.nan))
                 xa_idx2 = b_approx[idx2].get('xa', np.full_like(tappr, np.nan))
                 ya_idx2 = b_approx[idx2].get('ya', np.full_like(tappr, np.nan))

                 # Ensure interpolation inputs are valid
                 mask1 = np.isfinite(tappr) & np.isfinite(xa_idx1) & np.isfinite(ya_idx1)
                 mask2 = np.isfinite(tappr) & np.isfinite(xa_idx2) & np.isfinite(ya_idx2)

                 if np.any(mask1) and np.any(mask2):
                      try:
                          hit_x1 = np.interp(tc, tappr[mask1], xa_idx1[mask1])
                          hit_y1 = np.interp(tc, tappr[mask1], ya_idx1[mask1])
                          hit_x2 = np.interp(tc, tappr[mask2], xa_idx2[mask2])
                          hit_y2 = np.interp(tc, tappr[mask2], ya_idx2[mask2])

                          # Add hits: [time, hitting_ball_idx, type(1=Ball), hit_ball_idx, x, y]
                          hits.append([ tc, idx1, 1, idx2, hit_x1, hit_y1 ]) # Ball idx1 hits idx2
                          hits.append([ tc, idx2, 1, idx1, hit_x2, hit_y2 ]) # Ball idx2 is hit by idx1
                      except Exception as e:
                          # print(f"Warning: Ball-ball hit interpolation failed between {idx1} and {idx2}: {e}") # Removed verbosity
                          pass
        return hits
    # --- End Hit Detection Helpers ---

    # --- Event Processing Helper ---
    def _select_next_event(hitlist_all, Tall0):
        """Selects the earliest valid hit event from the list."""
        earliest_tc = np.inf
        next_event_time = Tall0[1] if len(Tall0) > 1 else np.inf
        valid_hits_at_tc = []

        # Find the minimum valid hit time from the list
        valid_hit_times = [h[0] for h in hitlist_all if np.isfinite(h[0]) and h[0] >= Tall0[0]] # Must be >= current time
        if valid_hit_times:
            earliest_tc = min(valid_hit_times)

        # Decide whether the next event is a hit or just advancing time
        if earliest_tc <= next_event_time: # Hit event occurs before or at the next data point time
            # Collect all hits occurring exactly at earliest_tc
            valid_hits_at_tc = [h for h in hitlist_all if np.isclose(h[0], earliest_tc)]
            # Ensure earliest_tc is not before current time due to float issues
            earliest_tc = max(earliest_tc, Tall0[0])
            return earliest_tc, valid_hits_at_tc, True # Return time, hits, and True (hit processed)
        else:
            # No hit before the next data point, so advance time
            return next_event_time, [], False # Return next data time, empty list, and False (no hit processed)

    def _apply_hit_event(tc, hits_at_tc, ball, ball0, hit, err):
        """Updates ball, ball0, and hit structures for a detected hit event.
        that means:
        - if hit time is not already in the time, add the hit to involved ball time, positions
        - add the hit time to ball not involved in the hit, interpolating position
        - remove from ball0
        -"""
        balls_hit_this_step = set()
        processed_balls = set() # Track balls updated in this function call

        for current_hit in hits_at_tc:
            # [time, ball_idx, type(1=Ball, 2=cushion), id, x, y]
            bi_idx = int(current_hit[1]) # 0-based ball index for this hit record
            contact_type = int(current_hit[2])
            contact_id = int(current_hit[3])
            hit_x = current_hit[4]
            hit_y = current_hit[5]

            balls_hit_this_step.add(bi_idx)

            # Skip if this ball's hit at this time was already processed (e.g., reciprocal ball-ball hits)
            #if bi_idx in processed_balls:
            #    continue

            # Update ball0 (remaining raw data) - set first point to hit location/time
            # Find index in ball0 corresponding to tc or just after
            idx_in_ball0 = np.searchsorted(ball0[bi_idx]['t'], tc, side='left')
            # Remove points before tc (keeping tc itself if it exists)
            keep_mask = ball0[bi_idx]['t'] >= tc
            # Ensure we don't lose the point immediately before tc if tc isn't in the array
            if idx_in_ball0 > 0 and tc not in ball0[bi_idx]['t']:
                 keep_mask[idx_in_ball0-1] = True # Keep the point just before

            for key in ['t', 'x', 'y']:
                if key in ball0[bi_idx] and len(ball0[bi_idx][key]) > 0:
                     ball0[bi_idx][key] = ball0[bi_idx][key][keep_mask]

            # Insert/update the exact hit point at the beginning of ball0
            if len(ball0[bi_idx]['t']) == 0 or not np.isclose(ball0[bi_idx]['t'][0], tc):
                 ball0[bi_idx]['t'] = np.insert(ball0[bi_idx]['t'], 0, tc)
                 ball0[bi_idx]['x'] = np.insert(ball0[bi_idx]['x'], 0, hit_x)
                 ball0[bi_idx]['y'] = np.insert(ball0[bi_idx]['y'], 0, hit_y)
                 # Need to recalculate derivatives after modification
            else: # If tc already exists, just update position
                ball0[bi_idx]['x'][0] = hit_x
                ball0[bi_idx]['y'][0] = hit_y

            # Update ball (processed trajectory)
            ball[bi_idx]['t'] = np.append(ball[bi_idx]['t'], tc)
            ball[bi_idx]['x'] = np.append(ball[bi_idx]['x'], hit_x)
            ball[bi_idx]['y'] = np.append(ball[bi_idx]['y'], hit_y)

            # Update hit data structure
            hit[bi_idx]['t'].append(tc)
            hit[bi_idx]['XPos'].append(hit_x)
            hit[bi_idx]['YPos'].append(hit_y)
            if contact_type == 1: # Ball
                hit[bi_idx]['with'].append(col[contact_id]) # Other ball's color char
            elif contact_type == 2: # Cushion
                hit[bi_idx]['with'].append(str(contact_id + 1)) # Cushion ID (1-4)
            else: hit[bi_idx]['with'].append('?')
            # Append default values for other fields if needed
            for key in ['Kiss', 'Point', 'Fuchs']:
                if key in hit[bi_idx]: hit[bi_idx][key].append(0) # Default 0 for new hits
                # else: print(f"Warning: Key {key} missing in hit structure for ball {bi_idx}")
            for key in ['PointDist', 'KissDistB1', 'Tpoint', 'Tkiss', 'Tready', 'TB2hit', 'Tfailure']:
                 if key in hit[bi_idx]: hit[bi_idx][key].append(np.nan) # Default NaN for new hits
                 # else: print(f"Warning: Key {key} missing in hit structure for ball {bi_idx}")

            processed_balls.add(bi_idx)


        # Update balls that were NOT hit at this exact time 'tc'
        for bi_idx_no_hit in range(3):
            if bi_idx_no_hit not in balls_hit_this_step:
                # Interpolate position at time tc using original data (ball0)
                x_at_tc, y_at_tc = np.nan, np.nan
                # Need reliable data in ball0 for interpolation
                t0 = ball0[bi_idx_no_hit].get('t', [])
                x0 = ball0[bi_idx_no_hit].get('x', [])
                y0 = ball0[bi_idx_no_hit].get('y', [])

                # Use the 'b_approx' data calculated earlier for interpolation
                # This uses the projected path *before* the hit tc
                xa_appr = b_approx[bi_idx_no_hit].get('xa', [])
                ya_appr = b_approx[bi_idx_no_hit].get('ya', [])
                t_appr_vec = b_approx[bi_idx_no_hit].get('tappr', []) # Need tappr here

                if len(t_appr_vec) > 1 and len(xa_appr) == len(t_appr_vec) and len(ya_appr) == len(t_appr_vec):
                     finite_mask = np.isfinite(t_appr_vec) & np.isfinite(xa_appr) & np.isfinite(ya_appr)
                     if np.sum(finite_mask) >= 2 and t_appr_vec[0] <= tc <= t_appr_vec[-1]:
                         try:
                              x_at_tc = np.interp(tc, t_appr_vec[finite_mask], xa_appr[finite_mask])
                              y_at_tc = np.interp(tc, t_appr_vec[finite_mask], ya_appr[finite_mask])
                         except Exception: pass # Keep NaN if interp fails

                # Update ball trajectory
                ball[bi_idx_no_hit]['t'] = np.append(ball[bi_idx_no_hit]['t'], tc)
                ball[bi_idx_no_hit]['x'] = np.append(ball[bi_idx_no_hit]['x'], x_at_tc)
                ball[bi_idx_no_hit]['y'] = np.append(ball[bi_idx_no_hit]['y'], y_at_tc)

                # Update ball0 - similar to hit balls, remove points before tc and update first point
                idx_in_ball0 = np.searchsorted(t0, tc, side='left')
                keep_mask = t0 >= tc
                if idx_in_ball0 > 0 and tc not in t0:
                    keep_mask[idx_in_ball0-1] = True

                for key in ['t', 'x', 'y']:
                     if key in ball0[bi_idx_no_hit] and len(ball0[bi_idx_no_hit][key]) > 0:
                          ball0[bi_idx_no_hit][key] = ball0[bi_idx_no_hit][key][keep_mask]

                if len(ball0[bi_idx_no_hit]['t']) == 0 or not np.isclose(ball0[bi_idx_no_hit]['t'][0], tc):
                     ball0[bi_idx_no_hit]['t'] = np.insert(ball0[bi_idx_no_hit]['t'], 0, tc)
                     ball0[bi_idx_no_hit]['x'] = np.insert(ball0[bi_idx_no_hit]['x'], 0, x_at_tc)
                     ball0[bi_idx_no_hit]['y'] = np.insert(ball0[bi_idx_no_hit]['y'], 0, y_at_tc)
                else:
                     ball0[bi_idx_no_hit]['x'][0] = x_at_tc
                     ball0[bi_idx_no_hit]['y'][0] = y_at_tc

                processed_balls.add(bi_idx_no_hit)


        # Check for duplicate hits (multiple events assigned to the same ball at the same time)
        # This check might be redundant if _select_next_event ensures uniqueness, but good for safety
        for bi_check in balls_hit_this_step:
             hit_times_for_ball = [h[0] for h in hits_at_tc if h[1] == bi_check]
             if len(hit_times_for_ball) > 1:
                  err['code'] = 1 # Use original error code
                  err_text = f'ERROR: Ball index {bi_check} involved in {len(hit_times_for_ball)} simultaneous hits at time {tc:.4f}.'
                  err['text'] = err_text if not err['text'] else err['text'] + "; " + err_text
                  print(err_text) # Log the error

        return balls_hit_this_step # Return set of balls involved in hits at this step

    def _advance_timestep(next_t, ball, ball0, b_approx):
        """Updates ball and ball0 structures when advancing to the next timestep without a hit."""
        processed_balls = set()
        for bi_idx in range(3):
            t0 = ball0[bi_idx].get('t', [])
            x0 = ball0[bi_idx].get('x', [])
            y0 = ball0[bi_idx].get('y', [])

            x_at_next_t, y_at_next_t = np.nan, np.nan

            if b_approx[bi_idx]['vt1'] > 0:
            # Interpolate position from original data if possible
                x_at_next_t = np.interp(next_t, t0, x0)
                y_at_next_t = np.interp(next_t, t0, y0)
            else:
                x_at_next_t = ball[bi_idx]['x'][-1] # Use last known position if ball stopped
                y_at_next_t = ball[bi_idx]['y'][-1]

            # Update ball (processed trajectory)
            ball[bi_idx]['t'] = np.append(ball[bi_idx]['t'], next_t)
            ball[bi_idx]['x'] = np.append(ball[bi_idx]['x'], x_at_next_t)
            ball[bi_idx]['y'] = np.append(ball[bi_idx]['y'], y_at_next_t)

            # Update ball0 (remaining raw data) remove values smaller next_t
            keep_mask = ball0[bi_idx]['t'] >= next_t

            if ball0[bi_idx]['t'][1] <= next_t: # If first point is after next_t, keep it
                for key in ['t', 'x', 'y']:
                    ball0[bi_idx][key] = ball0[bi_idx][key][keep_mask]
            else:
                ball0[bi_idx]['t'][0] = next_t # Set first point to next_t
                ball0[bi_idx]['x'][0] = x_at_next_t
                ball0[bi_idx]['y'][0] = y_at_next_t

            processed_balls.add(bi_idx)

        return processed_balls # All balls are processed in a timestep advance

    # --- Finalization Helper ---
    def _finalize_data(SA, si, ball, hit):
        """Assigns processed data back to SA and converts hit lists to arrays."""
        for bi_idx in range(3):
            # Assign processed trajectories back to SA['Shot'][si]['Route']
            # Check structure validity before assignment
            if 'Route' in SA['Shot'][si] and isinstance(SA['Shot'][si]['Route'], list) and len(SA['Shot'][si]['Route']) > bi_idx \
               and isinstance(SA['Shot'][si]['Route'][bi_idx], dict):
                 SA['Shot'][si]['Route'][bi_idx]['t'] = ball[bi_idx].get('t', np.array([]))
                 SA['Shot'][si]['Route'][bi_idx]['x'] = ball[bi_idx].get('x', np.array([]))
                 SA['Shot'][si]['Route'][bi_idx]['y'] = ball[bi_idx].get('y', np.array([]))
                 # Optionally clear other fields if they exist (vx, vy etc.) or recalculate them
                 if 'vx' in SA['Shot'][si]['Route'][bi_idx]: SA['Shot'][si]['Route'][bi_idx]['vx'] = np.array([])
                 if 'vy' in SA['Shot'][si]['Route'][bi_idx]: SA['Shot'][si]['Route'][bi_idx]['vy'] = np.array([])
                 if 'v' in SA['Shot'][si]['Route'][bi_idx]: SA['Shot'][si]['Route'][bi_idx]['v'] = np.array([])
                 if 'dt' in SA['Shot'][si]['Route'][bi_idx]: SA['Shot'][si]['Route'][bi_idx]['dt'] = np.array([])

            else:
                 print(f"Warning: SA['Shot'][{si}]['Route'][{bi_idx}] structure invalid. Cannot assign final trajectory.")
                 # Potentially set an error flag here?

            # Convert hit dictionary value lists to numpy arrays
            if isinstance(hit, list) and len(hit) > bi_idx and isinstance(hit[bi_idx], dict):
                 for key in hit[bi_idx]:
                     hit[bi_idx][key] = np.array(hit[bi_idx][key])
            else:
                 print(f"Warning: Final hit structure invalid for ball index {bi_idx}.")


    # ========================================
    # ===== Main Execution of extract_events =====
    # ========================================

    # 1. Initialization
    ball0, ball, hit, Tall0, err, b1b2b3_data = _initialize_event_data(SA, si, param)
    if err['code'] is not None:
        return hit, err # Return early if initialization failed
    b1b2b3_num, b1i, b2i, b3i = b1b2b3_data

    ball0_full, = ax.plot(ball0[0]['y'], ball0[0]['x'], 'rd-', label='Ball')  # Line for Ball 0

    # discretization steps for interpolation within dT
    tvec = np.linspace(0.0, 1.0, 51) # 0 to 1 relative time within dT

    do_scan = len(Tall0) >= 2
    ti = 0 # Step counter (mainly for debugging/first step logic)

    # Main processing loop
    while do_scan:
        current_t = Tall0[0]
        next_t = Tall0[1]
        dT = next_t - current_t

        if dT <= 0: # Should not happen with unique/sorted Tall0, but safety check
             # print(f"Warning: Non-positive dT ({dT}) encountered at t={current_t}. Skipping step.") # Removed verbosity
             Tall0 = Tall0[1:]
             do_scan = len(Tall0) >= 2
             continue

        # Absolute time vector for approximation interval
        tappr = current_t + dT * tvec

        # 2. Calculate Approximations & Intermediate Velocities
        b_approx = [{} for _ in range(3)]
             # Pass tappr to approximation function bi, ball, ball0, hit, param, Tall0, tvec):
        b_approx = _calculate_approximations(ball, ball0, hit, param, Tall0, tvec)
        
        # Update the plot
        appr_line.set_data(b_approx[0]['ya'], b_approx[0]['xa'])
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot

        # 3. Calculate Distances
        ball_distances = _calculate_ball_ball_distances(b_approx, b1b2b3_num, BB_pairs, param, tvec)
        _calculate_cushion_distances(b_approx, param, tvec) # Modifies b_approx in place

        # 4. Detect Potential Hits
        hitlist_all = []
        # Cushions
        for bi_idx in range(3):
             hitlist_all.extend(_find_potential_cushion_hits(bi_idx, b_approx[bi_idx], tappr, param))
        # Ball-Ball
        for bbi in range(len(BB_pairs)):
             hitlist_all.extend(_find_potential_ball_ball_hits(bbi, b_approx, ball_distances, tappr, hit, b1b2b3_num, BB_pairs, param, current_t, ti==0, ball0))

        # 5. Select Next Event (Hit or Timestep Advance)
        event_time, hits_at_event_time, is_hit_event = _select_next_event(hitlist_all, Tall0)

        if np.isinf(event_time): # No more events or next steps possible
             # print("Info: No further events or time steps.") # Removed verbosity
             break

        # 6. Process Event
        processed_balls_indices = set()
        if is_hit_event:
            # Update ball, ball0, hit based on the hits at event_time
            processed_balls_indices = _apply_hit_event(event_time, hits_at_event_time, ball, ball0, hit, err) # err is modified in place
            ax.plot(hit[0]['YPos'], hit[0]['XPos'], 'ko', markersize=20)  # Plot hit point for Ball 0

            # Update Derivatives in ball0
            _update_ball0_derivatives(ball0)

            # Update Tall0: remove times before event_time, ensure event_time is present
            Tall0 = Tall0[Tall0 >= event_time]
            if len(Tall0) == 0 or not np.isclose(Tall0[0], event_time):
                 Tall0 = np.unique(np.insert(Tall0, 0, event_time))

        else: # No hit, advance to next timestep
            processed_balls_indices = _advance_timestep(event_time, ball, ball0, b_approx)
            # Update Tall0: simply move to the next time step
            Tall0 = Tall0[1:]

        # Update the plot
        ball0_line.set_data(ball[0]['y'], ball[0]['x'])
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot

        # 7. update for affected balls
        for bi_idx in processed_balls_indices:
             # Only update if the ball was actually involved in a hit or moved
             last_hit_with = hit[bi_idx]['with'][-1] if len(hit[bi_idx]['with']) > 0 else '-'
             # Update needed if hit occurred OR if timestep advanced and ball was moving
             # Needs careful check: derivatives should reflect state *after* event_time
             


        # 8. Loop Condition
        do_scan = len(Tall0) >= 2
        ti += 1
        print(f"Step {ti}: Tall0 = {Tall0[0:4]}, dT = {dT}, Hits = {len(hits_at_event_time)}, Event Time = {event_time:.4f}") # Debug output



    # 9. Finalization
    _finalize_data(SA, si, ball, hit) # Assigns back to SA, converts hit lists

    return hit, err
