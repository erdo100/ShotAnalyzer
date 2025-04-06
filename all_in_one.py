# <<< CODE STARTS >>>
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math
import warnings
import copy
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None

# --- Constants and Parameters ---
# Keep parameters needed for reading, quality checks, and event detection
DEFAULT_PARAMS = {
    "ver": "Shot Analyzer Python v1.0 - Raw Hits",
    "size": [1420, 2840], # Y, X
    "ballR": 61.5 / 2,
    "colors": "WYR",
    "BallProjecttoCushionLimit": 10, # Keep for basic quality
    "minvel": 10, # Keep for basic event detection logic
    "tc_precision": 1e-9,
    "dist_precision": 1e-6,
    "tableX": 2840,
    "tableY": 1420,
}

# --- Utility Functions ---
# (str_to_indices, indices_to_str, interpolate_pos - unchanged)
def str_to_indices(b1b2b3_str):
    mapping = {'W': 0, 'Y': 1, 'R': 2}
    try: return [mapping[b1b2b3_str[0]], mapping[b1b2b3_str[1]], mapping[b1b2b3_str[2]]]
    except (IndexError, KeyError, TypeError): return None

def indices_to_str(indices):
    mapping = {0: 'W', 1: 'Y', 2: 'R'}
    try: return mapping[indices[0]] + mapping[indices[1]] + mapping[indices[2]]
    except (IndexError, KeyError, TypeError): return "???"

def interpolate_pos(t, t1, t2, x1, x2, y1, y2):
    if abs(t2 - t1) < 1e-12:
        return x1, y1
    
    ratio = (t - t1) / (t2 - t1)
    x = x1 + ratio * (x2 - x1)
    y = y1 + ratio * (y2 - y1)
    return x, y

# --- Classes ---

class BallRoute:
    # (BallRoute class - unchanged, but velocity calculation might be less critical)
    def __init__(self, t=None, x=None, y=None):
        self.t = np.array(t, dtype=float) if t is not None else np.array([], dtype=float)
        self.x = np.array(x, dtype=float) if x is not None else np.array([], dtype=float)
        self.y = np.array(y, dtype=float) if y is not None else np.array([], dtype=float)
        self.vx = np.array([], dtype=float); self.vy = np.array([], dtype=float)
        self.v = np.array([], dtype=float); self.dt = np.array([], dtype=float)
        if len(self.t) > 0: self.calculate_velocities()
    
    def calculate_velocities(self): # Keep for event detection logic needs
        n = len(self.t)
        if n > 1:
            self.dt = np.diff(self.t)
            self.vx = np.zeros(n, dtype=float)
            self.vy = np.zeros(n, dtype=float)
            valid_dt = self.dt > 1e-9
            dx = np.diff(self.x)
            dy = np.diff(self.y)
            self.vx[:-1][valid_dt] = dx[valid_dt] / self.dt[valid_dt]
            self.vy[:-1][valid_dt] = dy[valid_dt] / self.dt[valid_dt]

            if n > 1: 
                self.vx[-1] = self.vx[-2]
                self.vy[-1] = self.vy[-2]
            
            self.v = np.sqrt(self.vx**2 + self.vy**2)
            self.dt = np.append(self.dt, self.dt[-1] if len(self.dt)>0 else 0)

        elif n == 1: 
            self.vx = np.zeros(1)
            self.vy = np.zeros(1)
            self.v = np.zeros(1)
            self.dt = np.zeros(1)
        else: 
            self.vx = np.array([])
            self.vy = np.array([])
            self.v = np.array([])
            self.dt = np.array([])

    def copy(self): 
        return copy.deepcopy(self)
    
    def __len__(self): 
        return len(self.t)
    
    def append_point(self, t, x, y):
        self.t = np.append(self.t, t)
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)

    def remove_first_point(self):
        if len(self) > 0: self.t = self.t[1:]
        self.x = self.x[1:]
        self.y = self.y[1:]
        self.calculate_velocities()
        
    def update_first_point(self, t, x, y):
        if len(self) > 0: 
            self.t[0] = t
            self.x[0] = x
            self.y[0] = y
            self.calculate_velocities()

    def get_pos_at_time(self, time_target):
        if len(self.t) == 0: 
            return np.nan, np.nan
        if len(self.t) == 1: 
            return self.x[0], self.y[0]
        try:
            ix = interp1d(self.t, self.x, kind='linear', bounds_error=False, fill_value=(self.x[0], self.x[-1]))
            iy = interp1d(self.t, self.y, kind='linear', bounds_error=False, fill_value=(self.y[0], self.y[-1]))
            return float(ix(time_target)), float(iy(time_target))
        except ValueError:
             idx = np.searchsorted(self.t, time_target, side='right')
             if idx == 0: 
                 return self.x[0], self.y[0]
             if idx >= len(self.t): 
                 return self.x[-1], self.y[-1]
             t1, t2 = self.t[idx-1], self.t[idx]
             x1, x2 = self.x[idx-1], self.x[idx]
             y1, y2 = self.y[idx-1], self.y[idx]

             return interpolate_pos(time_target, t1, t2, x1, x2, y1, y2)

class HitEvent:
    """Represents a basic detected collision or start event."""
    def __init__(self, t, with_char, pos_x, pos_y):
        self.t = t
        self.with_char = with_char # S, 1, 2, 3, 4 (cushions), W, Y, R (target balls)
        self.pos_x = pos_x
        self.pos_y = pos_y
        # No other calculated fields needed for this request

class Shot:
    """Represents a single shot with its basic data and raw events."""
    def __init__(self, shot_id, mirrored, point_type, table_index):
        self.shot_id = shot_id; self.mirrored = mirrored
        self.point_type = point_type
        self.table_index = table_index
        self.route0 = [BallRoute(), BallRoute(), BallRoute()] # Original (cleaned)
        self.route = [BallRoute(), BallRoute(), BallRoute()]  # Refined by events
        self.hit_events = [[], [], []] # Lists of basic HitEvent objects [W, Y, R]
        self.b1b2b3_indices = None
        self.b1b2b3_str = "???"
        self.interpreted = 0
        self.error_code = None
        self.error_text = ""
        self.selected = False

    # Add helper to get the hit sequence string relative to B1/B2/B3 roles
    def get_hit_sequence_str(self, ball_role_index): # 0=B1, 1=B2, 2=B3
        if not self.b1b2b3_indices:
            return ""
        original_ball_idx = self.b1b2b3_indices[ball_role_index]
        hits = self.hit_events[original_ball_idx]
        if not hits:
            return ""

        # Map target ball chars (W,Y,R) to their role (B1,B2,B3) for output clarity
        role_map = {idx: f"B{i+1}" for i, idx in enumerate(self.b1b2b3_indices)}
        output_chars = []
        for event in hits:
             char = event.with_char
             if char in 'WYR': # If it's a ball hit
                  target_original_idx = 'WYR'.find(char)
                  # Find the role (B1, B2, or B3) of the target ball
                  try:
                      target_role_idx = self.b1b2b3_indices.index(target_original_idx)
                      output_chars.append(f"B{target_role_idx + 1}")
                  except (ValueError, TypeError):
                      output_chars.append('?') # Should not happen if indices valid
             else: # Start ('S'), Cushion ('1'-'4'), or Placeholder ('-')
                  output_chars.append(char)
        return "-".join(output_chars)

    # Keep basic route accessors if needed
    def get_ball_route(self, ball_index, original=False):
        if ball_index not in [0, 1, 2]:
            return BallRoute()
        return self.route0[ball_index] if original else self.route[ball_index]
    def add_hit_event(self, ball_index, hit_event):
        if ball_index in [0, 1, 2]: 
            self.hit_events[ball_index].append(hit_event)


class ShotAnalyzer:
    """Simplified analyzer to extract B1B2B3 and raw hit sequences."""
    def __init__(self, params=None):
        self.params = {**DEFAULT_PARAMS, **(params if params else {})}
        self.shots = []
        self.table = pd.DataFrame()

    def read_gamefile(self, filepath):
        # (read_gamefile implementation - unchanged)
        try:
            with open(filepath, 'r') as f: data = json.load(f)
        except FileNotFoundError: print(f"Error: File not found at {filepath}"); return False
        except json.JSONDecodeError: print(f"Error: Could not decode JSON from {filepath}"); return False
        table_data = []
        self.shots = []
        si = -1
        df_idx = 0
        player1 = data.get('Player1', 'Unknown')
        player2 = data.get('Player2', 'Unknown')
        game_type = data.get('Match', {}).get('GameType', 'Unknown'); filename = filepath.split('/')[-1].split('\\')[-1]
        for seti, set_data in enumerate(data.get('Match', {}).get('Sets', [])):
            entries = set_data.get('Entries', []);
            if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], list): entries = entries[0]
            if not isinstance(entries, list): continue
            for entryi, entry in enumerate(entries):
                 if isinstance(entry, dict) and entry.get('PathTrackingId', 0) > 0:
                     si += 1; shot_id = entry['PathTrackingId']; mirrored = 0
                     scoring = entry.get('Scoring', {})
                     point_type = scoring.get('EntryType', 'Unknown')
                     current_shot = Shot(shot_id, mirrored, point_type, df_idx) # Store initial df_idx
                     path_tracking = entry.get('PathTracking', {})
                     datasets = path_tracking.get('DataSets', [])
                     valid_route = True
                     if len(datasets) == 3:
                         for bi in range(3):
                             coords = datasets[bi].get('Coords', [])
                             t = [c.get('DeltaT_500us', 0) * 0.0005 for c in coords]
                             x = [c.get('X', 0) * self.params['tableX'] for c in coords]
                             y = [c.get('Y', 0) * self.params['tableY'] for c in coords]
                             if not t or len(t) < 2: valid_route = False; break # Ensure at least 2 points
                             current_shot.route0[bi] = BallRoute(t, x, y)
                     else: valid_route = False
                     if not valid_route: print(f"Warning: Skipping ShotID {shot_id} due to incomplete/short route data."); si -= 1; continue
                     self.shots.append(current_shot)
                     player_num = scoring.get('Player', 0)
                     player_name = player1 if player_num == 1 else (player2 if player_num == 2 else 'Unknown')
                     row_data = {'Selected': False, 'ShotID': shot_id, 'Mirrored': mirrored, 'InternalID': shot_id + mirrored / 10.0, 'Filename': filename, 'GameType': game_type, 'Interpreted': 0, 'Player': player_name, 'ErrorID': None, 'ErrorText': 'Read OK', 'Set': seti + 1, 'CurrentInning': scoring.get('CurrentInning', 0), 'CurrentSeries': scoring.get('CurrentSeries', 0), 'CurrentTotalPoints': scoring.get('CurrentTotalPoints', 0), 'Point': point_type, 'TableIndex': df_idx }
                     table_data.append(row_data)
                     df_idx += 1

        if not table_data: print("No valid shot data found in file."); return False
        self.table = pd.DataFrame(table_data)
        print(f"Read {len(self.shots)} shots.")
        return True

    def perform_data_quality_checks(self):
        """Basic data cleaning needed for B1B2B3 and event detection."""
        print("Performing basic data quality checks...")
        err_count = 0
        for i, shot in enumerate(self.shots):
            if shot.interpreted: continue
            df_idx = shot.table_index
            error_code = None; error_text = ""; selected = False
            try:
                 # Ensure at least 2 points per ball (checked in read_gamefile now)
                 # Keep projection and initial distance checks
                 temp_routes = [r.copy() for r in shot.route0]
                 xmin = self.params['ballR'] + 0.1
                 xmax = self.params['size'][1] - self.params['ballR'] - 0.1
                 ymin = self.params['ballR'] + 0.1; ymax = self.params['size'][0] - self.params['ballR'] - 0.1
                 for bi in range(3): 
                    temp_routes[bi].x = np.clip(temp_routes[bi].x, xmin, xmax)
                    temp_routes[bi].y = np.clip(temp_routes[bi].y, ymin, ymax)

                 bb_idx = [(0, 1), (0, 2), (1, 2)]
                 ballR = self.params['ballR']
                 for bbi, (b1i, b2i) in enumerate(bb_idx):
                     r1 = temp_routes[b1i]; r2 = temp_routes[b2i]
                     if len(r1) > 0 and len(r2) > 0:
                         dx = r1.x[0] - r2.x[0]
                         dy = r1.y[0] - r2.y[0]
                         dsq = dx*dx + dy*dy
                         mindsq = (2 * ballR)**2
                         
                         if dsq < mindsq - 1e-6:
                             dist = math.sqrt(dsq) if dsq > 0 else 0
                             ov = 2 * ballR - dist
                             if dist > 1e-6: 
                                vx, vy = dx / dist, dy / dist
                                r1.x[0] += vx * ov / 2.0
                                r1.y[0] += vy * ov / 2.0
                                r2.x[0] -= vx * ov / 2.0
                                r2.y[0] -= vy * ov / 2.0
                             else:
                                 r1.x[0] += ov / 2.0

                 # Keep basic time linearity check/fix
                 for bi in range(3):
                    route = temp_routes[bi]
                    if len(route) > 1:
                        dt = np.diff(route.t)
                        nonlin = np.where(dt <= 0)[0]
                        if len(nonlin) > 0:
                            to_rem = sorted(list(set(nonlin + 1)), reverse=True)
                            if len(route) - len(to_rem) < 2:
                                error_code = 111
                                error_text = f'Ball {bi+1} time linearity unfixable (too short)'
                                selected = True; break
                            route.t = np.delete(route.t, to_rem)
                            route.x = np.delete(route.x, to_rem)
                            route.y = np.delete(route.y, to_rem)
                    if selected: break
                 if selected: raise ValueError(error_text)

                 # Keep start/end time consistency check
                 starts = [temp_routes[bi].t[0] for bi in range(3) if len(temp_routes[bi]) > 0]
                 ends = [temp_routes[bi].t[-1] for bi in range(3) if len(temp_routes[bi]) > 0]
                 if len(starts)>0 and not np.allclose(starts, starts[0], atol=1e-6):
                    error_code = 121
                    error_text = 'Inconsistent start times'
                    selected = True; raise ValueError(error_text)
                 if len(ends)>0 and not np.allclose(ends, ends[0], atol=1e-6):
                    error_code = 122
                    error_text = 'Inconsistent end times'
                    selected = True; raise ValueError(error_text)

                 error_text = "Quality OK"
                 shot.route0 = temp_routes # Update route0

            except ValueError as e:
                print(f"Quality Check Failed for Shot {shot.shot_id} (Idx: {df_idx}): {e}")
                err_count += 1
            except Exception as e:
                print(f"Unexpected Error during Quality Check for Shot {shot.shot_id} (Idx: {df_idx}): {e}")
                error_code = 999
                error_text = "Unexpected processing error"
                selected = True; err_count += 1

            if df_idx is not None and df_idx < len(self.table):
                self.table.loc[df_idx, 'ErrorID'] = error_code
                self.table.loc[df_idx, 'ErrorText'] = error_text
                self.table.loc[df_idx, 'Selected'] = selected
            shot.error_code = error_code
            shot.error_text = error_text
            shot.selected = selected
        print(f"Basic quality checks completed. {err_count} shots flagged.")

    def _determine_b1b2b3_for_shot(self, shot):
        # (_determine_b1b2b3_for_shot implementation - unchanged)
        if any(len(shot.route0[bi]) < 2 for bi in range(3)): return None, 2, "Empty Data (B1B2B3 determination)"
        try: t2 = [shot.route0[bi].t[1] if len(shot.route0[bi]) > 1 else float('inf') for bi in range(3)]; t_start = shot.route0[0].t[0]
        except IndexError: return None, 2, "Empty Data"
        t2 = [t - t_start for t in t2]; b_indices = np.argsort(t2)
        if t2[b_indices[0]] == float('inf'): return [0, 1, 2], None, "No ball movement detected" # Default W,Y,R if no move
        b1it = b_indices[0]; b2it = b_indices[1]; b3it = b_indices[2]
        if len(shot.route0[b1it]) < 3: return list(b_indices), None, None # Not enough data to refine further
        t2_b1 = shot.route0[b1it].t[1]; t3_b1 = shot.route0[b1it].t[2]; moved_b2 = False; moved_b3 = False
        if len(shot.route0[b2it]) > 1 and shot.route0[b2it].t[1] <= t3_b1 + 1e-9: moved_b2 = True
        if len(shot.route0[b3it]) > 1 and shot.route0[b3it].t[1] <= t3_b1 + 1e-9: moved_b3 = True
        final_order = list(b_indices)
        if moved_b2 and moved_b3: return final_order, 3, "All balls moved simultaneously"
        # Skip complex angle refinement from original MATLAB as it wasn't fully clear/translated
        return final_order, None, None

    def determine_b1b2b3(self):
        # (determine_b1b2b3 implementation - unchanged)
        print("Determining B1, B2, B3..."); err_count = 0
        for i, shot in enumerate(self.shots):
             if shot.interpreted or shot.selected: continue
             df_idx = shot.table_index; indices, err_code, err_text = self._determine_b1b2b3_for_shot(shot)
             if df_idx is None or df_idx >= len(self.table): continue
             if err_code:
                 shot.error_code = err_code; shot.error_text = err_text; shot.selected = True; print(f"B1B2B3 Error Shot {shot.shot_id} (Idx: {df_idx}): {err_text}"); err_count += 1; b1b2b3_str = "ERR"
                 self.table.loc[df_idx, 'ErrorID'] = err_code; self.table.loc[df_idx, 'ErrorText'] = err_text; self.table.loc[df_idx, 'Selected'] = True
             else:
                 shot.b1b2b3_indices = indices; b1b2b3_str = indices_to_str(indices); shot.b1b2b3_str = b1b2b3_str
                 if 'B1B2B3' not in self.table.columns: self.table['B1B2B3'] = "???"
                 self.table.loc[df_idx, 'B1B2B3'] = b1b2b3_str
                 current_err = self.table.loc[df_idx, 'ErrorID']
                 if pd.notna(current_err) and current_err < 10 : # Clear only B1B2B3 errors
                    self.table.loc[df_idx, 'ErrorID'] = None; self.table.loc[df_idx, 'ErrorText'] = "B1B2B3 OK"; self.table.loc[df_idx, 'Selected'] = False
                    shot.error_code = None; shot.error_text = "B1B2B3 OK"; shot.selected = False
        print(f"B1B2B3 determination completed. {err_count} shots flagged.")


    def _detect_raw_events_for_shot(self, shot):
        """
        Detects raw collision events (time, type, position) for a single shot.
        Populates shot.route (refined trajectories) and shot.hit_events.
        Simplified version focusing only on sequence detection.
        """
        try:
            cols = "WYR"
            ballR = self.params['ballR']
            minvel = self.params['minvel']
            tc_precision = self.params['tc_precision']
            dist_precision = self.params['dist_precision']
            table_size_y, table_size_x = self.params['size']

            # --- Initialization ---
            shot.hit_events = [[], [], []]
            shot.route = [BallRoute(), BallRoute(), BallRoute()]
            ball0 = [r.copy() for r in shot.route0]

            if any(len(b) < 2 for b in ball0): return False # Already checked? Redundant check

            for bi in range(3):
                shot.route[bi].append_point(ball0[bi].t[0], ball0[bi].x[0], ball0[bi].y[0])
                start_char = 'S' if bi == shot.b1b2b3_indices[0] else '-'
                start_event = HitEvent(t=ball0[bi].t[0], with_char=start_char, pos_x=ball0[bi].x[0], pos_y=ball0[bi].y[0])
                shot.add_hit_event(bi, start_event)
                ball0[bi].calculate_velocities()

            Tall0 = ball0[0].t.copy()
            Tall0 = Tall0[Tall0 >= shot.route[0].t[0]]

            loop_count = 0; max_loops = len(Tall0) * 5

            # --- Main Iterative Loop ---
            while len(Tall0) >= 2 and loop_count < max_loops:
                loop_count += 1
                t1 = Tall0[0]; t2 = Tall0[1]; dt = t2 - t1

                if dt < tc_precision:
                    Tall0 = Tall0[1:]
                    for bi in range(3):
                        idx = np.searchsorted(ball0[bi].t, t1, side='left')
                        if idx > 0: ball0[bi].t = ball0[bi].t[idx:]; ball0[bi].x = ball0[bi].x[idx:]; ball0[bi].y = ball0[bi].y[idx:]; ball0[bi].calculate_velocities()
                    continue

                pos1 = np.array([b.get_pos_at_time(t1) for b in ball0])
                vel1 = np.zeros((3, 2)); vmag1 = np.zeros(3)
                for bi in range(3):
                    idx1 = np.searchsorted(ball0[bi].t, t1, side='right') - 1; idx1 = max(0, idx1)
                    if idx1 < len(ball0[bi].vx): vel1[bi, 0] = ball0[bi].vx[idx1]; vel1[bi, 1] = ball0[bi].vy[idx1]; vmag1[bi] = ball0[bi].v[idx1]
                pos2_pred = pos1 + vel1 * dt

                # --- Calculate Distances ---
                distBB1 = np.full((3, 3), np.inf); distBB2 = np.full((3, 3), np.inf)
                distC1 = np.full((3, 4), np.inf); distC2 = np.full((3, 4), np.inf)
                clim = [ ballR, table_size_x - ballR, table_size_y - ballR, ballR ]

                for bi in range(3):
                    distC1[bi, 0]=pos1[bi, 1]-clim[0]; distC1[bi, 1]=clim[1]-pos1[bi, 0]; distC1[bi, 2]=clim[2]-pos1[bi, 1]; distC1[bi, 3]=pos1[bi, 0]-clim[3]
                    distC2[bi, 0]=pos2_pred[bi, 1]-clim[0]; distC2[bi, 1]=clim[1]-pos2_pred[bi, 0]; distC2[bi, 2]=clim[2]-pos2_pred[bi, 1]; distC2[bi, 3]=pos2_pred[bi, 0]-clim[3]
                    for bj in range(bi + 1, 3):
                        dx1=pos1[bi, 0]-pos1[bj, 0]; dy1=pos1[bi, 1]-pos1[bj, 1]; distBB1[bi, bj]=distBB1[bj, bi]=math.sqrt(dx1*dx1 + dy1*dy1)
                        dx2=pos2_pred[bi, 0]-pos2_pred[bj, 0]; dy2=pos2_pred[bi, 1]-pos2_pred[bj, 1]; distBB2[bi, bj]=distBB2[bj, bi]=math.sqrt(dx2*dx2 + dy2*dy2)

                # --- Hit Detection (Simplified - focus on crossing threshold) ---
                hitlist = []; min_tc = t2 + tc_precision

                # Cushion Hits
                for bi in range(3):
                    if vmag1[bi] < minvel/2.0: continue # Less strict velocity check
                    for ci in range(4):
                        moving_towards = False
                        if ci == 0 and vel1[bi, 1] < -minvel/10.0: moving_towards = True
                        elif ci == 1 and vel1[bi, 0] > minvel/10.0: moving_towards = True
                        elif ci == 2 and vel1[bi, 1] > minvel/10.0: moving_towards = True
                        elif ci == 3 and vel1[bi, 0] < -minvel/10.0: moving_towards = True
                        if not moving_towards: continue

                        crosses_zero = distC1[bi, ci] >= -dist_precision and distC2[bi, ci] < dist_precision
                        if crosses_zero: # Simplified check: just look for crossing zero
                             tc = t1; denom = distC1[bi, ci] - distC2[bi, ci]
                             if abs(denom) > dist_precision: tc = t1 + dt * distC1[bi, ci] / denom
                             tc = np.clip(tc, t1 - tc_precision, t2 + tc_precision)
                             if t1 - tc_precision <= tc < min_tc:
                                 cushX, cushY = interpolate_pos(tc, t1, t2, pos1[bi, 0], pos2_pred[bi, 0], pos1[bi, 1], pos2_pred[bi, 1])
                                 if 0 <= cushX <= table_size_x and 0 <= cushY <= table_size_y:
                                     hitlist.append({'tc': tc, 'type': 'C', 'bi': bi, 'other': ci + 1, 'x': cushX, 'y': cushY})
                                     min_tc = min(min_tc, tc)

                # Ball-Ball Hits
                for bi in range(3):
                    for bj in range(bi + 1, 3):
                        crosses_thresh = distBB1[bi, bj] >= 2*ballR - dist_precision and distBB2[bi, bj] < 2*ballR + dist_precision
                        if crosses_thresh: # Simplified check: just look for crossing threshold
                            vel_rel = vel1[bi] - vel1[bj]; pos_rel = pos1[bi] - pos1[bj]
                            a = np.dot(vel_rel, vel_rel); b = 2 * np.dot(pos_rel, vel_rel); c = np.dot(pos_rel, pos_rel) - (2*ballR)**2
                            tc_sol = np.inf
                            if abs(a) > 1e-9:
                                delta = b*b - 4*a*c
                                if delta >= 0:
                                    s_delta = math.sqrt(delta); t1s = (-b + s_delta)/(2*a); t2s = (-b - s_delta)/(2*a)
                                    vsols = [s for s in [t1s, t2s] if 0 - tc_precision <= s <= dt + tc_precision]
                                    if vsols: tc_sol = t1 + min(vsols)
                            elif abs(b) > 1e-9: t_s = -c / b;
                            if 0 - tc_precision <= t_s <= dt + tc_precision: tc_sol = t1 + t_s

                            if t1 - tc_precision <= tc_sol < min_tc:
                                hX_bi, hY_bi = interpolate_pos(tc_sol, t1, t2, pos1[bi, 0], pos2_pred[bi, 0], pos1[bi, 1], pos2_pred[bi, 1])
                                hX_bj, hY_bj = interpolate_pos(tc_sol, t1, t2, pos1[bj, 0], pos2_pred[bj, 0], pos1[bj, 1], pos2_pred[bj, 1])
                                hitlist.append({'tc': tc_sol, 'type': 'B', 'bi': bi, 'other': bj, 'x': hX_bi, 'y': hY_bi})
                                hitlist.append({'tc': tc_sol, 'type': 'B', 'bi': bj, 'other': bi, 'x': hX_bj, 'y': hY_bj})
                                min_tc = min(min_tc, tc_sol)

                # --- Trajectory Refinement ---
                actual_hits_at_tc = [h for h in hitlist if abs(h['tc'] - min_tc) < tc_precision]

                if actual_hits_at_tc: # Hit occurred
                    tc = min_tc
                    balls_hit = set(h['bi'] for h in actual_hits_at_tc)
                    processed_pairs = set()

                    for hit_info in actual_hits_at_tc:
                        bi = hit_info['bi']; h_type = hit_info['type']; other = hit_info['other']
                        hX = hit_info['x']; hY = hit_info['y']
                        if h_type == 'B': pair = tuple(sorted((bi, other)));
                        if h_type == 'B' and pair in processed_pairs: continue;
                        if h_type == 'B': processed_pairs.add(pair)

                        if abs(shot.route[bi].t[-1] - tc) > tc_precision: shot.route[bi].append_point(tc, hX, hY)
                        wc = str(other) if h_type == 'C' else cols[other]
                        event = HitEvent(t=tc, with_char=wc, pos_x=hX, pos_y=hY)
                        shot.add_hit_event(bi, event)

                        if tc > ball0[bi].t[0] + tc_precision:
                             idx_s = np.searchsorted(ball0[bi].t, tc, side='left')
                             ball0[bi].t = np.insert(ball0[bi].t[idx_s:], 0, tc); ball0[bi].x = np.insert(ball0[bi].x[idx_s:], 0, hX); ball0[bi].y = np.insert(ball0[bi].y[idx_s:], 0, hY)
                        elif abs(tc - ball0[bi].t[0]) < tc_precision: ball0[bi].x[0] = hX; ball0[bi].y[0] = hY

                    for bi in range(3):
                        if bi not in balls_hit:
                            if abs(shot.route[bi].t[-1] - tc) > tc_precision: iX, iY = ball0[bi].get_pos_at_time(tc); shot.route[bi].append_point(tc, iX, iY)
                            if tc > ball0[bi].t[0] + tc_precision:
                                iX0, iY0 = ball0[bi].get_pos_at_time(tc); idx_s = np.searchsorted(ball0[bi].t, tc, side='left')
                                ball0[bi].t = np.insert(ball0[bi].t[idx_s:], 0, tc); ball0[bi].x = np.insert(ball0[bi].x[idx_s:], 0, iX0); ball0[bi].y = np.insert(ball0[bi].y[idx_s:], 0, iY0)

                    if abs(Tall0[0] - tc) > tc_precision: Tall0 = np.insert(Tall0, 1, tc)
                    Tall0 = Tall0[1:]

                else: # No hit
                    t_next = t2
                    for bi in range(3):
                         idx_next = np.searchsorted(ball0[bi].t, t_next, side='left')
                         if idx_next < len(ball0[bi].t) and abs(ball0[bi].t[idx_next] - t_next) < tc_precision:
                            x_n, y_n = ball0[bi].x[idx_next], ball0[bi].y[idx_next]
                            if abs(shot.route[bi].t[-1] - t_next) > tc_precision: shot.route[bi].append_point(t_next, x_n, y_n)
                            ball0[bi].t = ball0[bi].t[idx_next+1:]; ball0[bi].x = ball0[bi].x[idx_next+1:]; ball0[bi].y = ball0[bi].y[idx_next+1:]
                         elif idx_next >= 1:
                             if abs(shot.route[bi].t[-1] - t_next) > tc_precision: iX, iY = ball0[bi].get_pos_at_time(t_next); shot.route[bi].append_point(t_next, iX, iY)
                             ball0[bi].t = ball0[bi].t[idx_next:]; ball0[bi].x = ball0[bi].x[idx_next:]; ball0[bi].y = ball0[bi].y[idx_next:]
                    Tall0 = Tall0[1:]

                for bi in range(3): ball0[bi].calculate_velocities()
                if loop_count >= max_loops: print(f"Warning Shot {shot.shot_id}: Event detection loop limit reached."); break

            # --- Finalization ---
            for bi in range(3):
                 if len(shot.route0[bi]) > 0 and len(shot.route[bi]) > 0:
                     if shot.route[bi].t[-1] < shot.route0[bi].t[-1] - tc_precision:
                         shot.route[bi].append_point(shot.route0[bi].t[-1], shot.route0[bi].x[-1], shot.route0[bi].y[-1])
                 shot.route[bi].calculate_velocities() # Final velocity calculation for refined route

            # print(f"  Raw event detection for Shot {shot.shot_id} finished.")
            return True

        except Exception as e:
             print(f"  ERROR during raw event detection for Shot {shot.shot_id}: {e}")
             import traceback; traceback.print_exc()
             shot.error_code = 298; shot.error_text = f"Raw event detection error: {e}"; shot.selected = True
             return False


    def extract_raw_events(self):
        """Detects raw hit events and updates B1B2B3 strings."""
        print("Extracting Raw Events...")
        total_shots = len(self.shots); processed_count = 0; error_count = 0

        # Initialize hit sequence columns if they don't exist
        for i in range(1, 4):
            col_name = f"B{i}_Hits"
            if col_name not in self.table.columns:
                self.table[col_name] = ""

        for i, shot in enumerate(self.shots):
            # print(f"Processing shot {i+1}/{total_shots} (ID: {shot.shot_id})") # Verbose
            if shot.interpreted or shot.selected or shot.b1b2b3_indices is None:
                continue # Skip already processed, flagged, or shots without B1B2B3
            df_idx = shot.table_index
            if df_idx is None or df_idx >= len(self.table): continue

            try:
                # --- 1. Detect Raw Events ---
                if not self._detect_raw_events_for_shot(shot):
                     error_count += 1
                     # Error flags set within the function
                     if df_idx is not None and df_idx < len(self.table):
                         self.table.loc[df_idx, 'ErrorID'] = shot.error_code
                         self.table.loc[df_idx, 'ErrorText'] = shot.error_text
                         self.table.loc[df_idx, 'Selected'] = True
                     continue

                # --- 2. Store Hit Sequence String in Table ---
                for role_idx in range(3): # B1, B2, B3
                    col_name = f"B{role_idx+1}_Hits"
                    sequence_str = shot.get_hit_sequence_str(role_idx)
                    self.table.loc[df_idx, col_name] = sequence_str

                # --- Mark as Interpreted (basic interpretation done) ---
                shot.interpreted = 1; self.table.loc[df_idx, 'Interpreted'] = 1
                # Clear non-critical errors if basic interpretation successful
                if shot.error_code is None or shot.error_code >= 200: # Keep critical read/quality/B1B2B3 errors
                    self.table.loc[df_idx, 'ErrorID'] = None; self.table.loc[df_idx, 'ErrorText'] = "Raw Hits OK"; self.table.loc[df_idx, 'Selected'] = False
                    shot.error_code = None; shot.error_text = "Raw Hits OK"; shot.selected = False
                processed_count += 1

            except Exception as e:
                print(f"  ERROR processing Shot {shot.shot_id} (Idx: {df_idx}): {e}")
                import traceback; traceback.print_exc()
                shot.error_code = 1001; shot.error_text = f"Raw Event processing error: {e}"; shot.selected = True
                if df_idx is not None and df_idx < len(self.table):
                    self.table.loc[df_idx, 'ErrorID'] = shot.error_code; self.table.loc[df_idx, 'ErrorText'] = shot.error_text; self.table.loc[df_idx, 'Selected'] = True
                error_count += 1
        print(f"Raw Event extraction completed. Processed: {processed_count}, Errors: {error_count}")


    def remove_flagged_shots(self):
        # (remove_flagged_shots implementation - unchanged)
        if not isinstance(self.table, pd.DataFrame) or 'Selected' not in self.table.columns:
            print("Cannot remove flagged shots: Table not initialized or 'Selected' column missing."); return
        initial_shot_count = len(self.shots); initial_table_rows = len(self.table)
        keep_mask = self.table['Selected'] == False
        if keep_mask.all(): print("No shots flagged for removal."); return
        kept_table = self.table[keep_mask].copy(); kept_internal_ids = set(kept_table['InternalID'].unique())
        kept_shots = [ shot for shot in self.shots if (shot.shot_id + shot.mirrored / 10.0) in kept_internal_ids ]
        num_removed = initial_shot_count - len(kept_shots); print(f"Removing {num_removed} flagged shots...")
        self.table = kept_table.reset_index(drop=True); self.shots = kept_shots
        if not self.table.empty:
            id_to_new_index = pd.Series(self.table.index, index=self.table['InternalID'])
            for shot in self.shots:
                internal_id = shot.shot_id + shot.mirrored / 10.0; new_index = id_to_new_index.get(internal_id)
                if new_index is not None: shot.table_index = new_index
                else: print(f"Warning: Could not find new table index for ShotID {shot.shot_id}. Setting index to None."); shot.table_index = None
        else:
             for shot in self.shots: shot.table_index = None
        print(f"Removal complete. Remaining shots: {len(self.shots)}. Remaining table rows: {len(self.table)}")


    def run_analysis(self, filepath, remove_flagged=False):
        """Executes the simplified analysis pipeline for raw hits."""
        print(f"Starting analysis for: {filepath}")
        if not self.read_gamefile(filepath): print("Analysis stopped: Could not read game file."); return

        self.perform_data_quality_checks() # Basic checks
        self.determine_b1b2b3() # Identify balls
        self.extract_raw_events() # Detect hit sequence

        if remove_flagged:
            self.remove_flagged_shots()

        print("Analysis finished.")
        if 'Selected' in self.table.columns:
             flagged_count = self.table['Selected'].sum()
             print(f"{flagged_count} shots currently flagged with issues.")
        else: print("Selected column not found in results.")

    # Remove methods no longer needed
    # def _evaluate_hit_details(self, shot): pass
    # def _evaluate_point_kiss_fuchs(self, shot): pass
    # def _update_table_with_event_data(self, shot): pass
    # def _calculate_ball_velocity(self, route, hit_event_time, event_index): pass
    # def _calculate_ball_direction(self, route, hit_event_time, event_index, num_events): pass
    # def _calculate_cushion_angle(self, cushion_index, v1_vec, v2_vec): pass
    # def calculate_b1b2b3_positions(self): pass
    # def _initialize_event_columns(self, max_events=8): pass


    def save_results(self, output_filepath, separator=';', decimal_sep=','):
        """Saves the resulting table to a CSV file with specified formatting."""
        try:
            all_cols = self.table.columns.tolist()
            print(f"Saving results to {output_filepath} (Separator='{separator}', Decimal='{decimal_sep}')")
            self.table.to_csv(
                output_filepath,
                index=False,
                columns=all_cols,
                sep=separator,
                decimal=decimal_sep
            )
            print(f"Results saved successfully.")
        except Exception as e:
            print(f"Error saving results to {output_filepath}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    filepath = "D:\\Billard\\0_AllDatabase\\WebSport\\20170906_match_01_Ersin_Cemal.txt"
    if not os.path.exists(filepath): print(f"Error: Input file not found at {filepath}"); exit()

    analyzer = ShotAnalyzer()
    analyzer.run_analysis(filepath, remove_flagged=True) # Set remove_flagged=True if desired

    print("\n--- Analysis Summary Table (First 10 rows) ---")
    pd.set_option('display.max_rows', 20); pd.set_option('display.max_columns', 30); pd.set_option('display.width', 150) # Adjust display
    print(analyzer.table.head(10).to_string())

    # Save results with specified format
    analyzer.save_results("analysis_output_raw_hits.csv", separator=';', decimal_sep=',')

# <<< CODE ENDS >>>