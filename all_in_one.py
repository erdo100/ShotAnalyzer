# <<< CODE STARTS >>>
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math
import warnings
import copy # Needed for deep copies
import os # For checking file existence

# Suppress runtime warnings from interpolation/division by zero if needed
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Ignore warnings from pandas about SettingWithCopyWarning if necessary
pd.options.mode.chained_assignment = None # default='warn'

# --- Constants and Parameters ---
DEFAULT_PARAMS = {
    "ver": "Shot Analyzer Python v1.0",
    "size": [1420, 2840], # Y, X
    "ballR": 61.5 / 2,
    "rdiam": 7,
    "cushionwidth": 50,
    "diamdist": 97,
    "framewidth": 147,
    "colors": "WYR",
    "BallOutOfTableDetectionLimit": 30,
    "BallCushionHitDetectionRange": 50,
    "BallProjecttoCushionLimit": 10,
    "NoDataDistanceDetectionLimit": 600,
    "MaxTravelDistancePerDt": 0.1,
    "MaxVelocity": 12000,
    "timax_appr": 5,
    "tableX": 2840, # Scaling factors from read_gamefile
    "tableY": 1420,
    "minangle": 1,
    "minvelchange": 0.1,
    "minvel": 10,
    "tc_precision": 1e-9,
    "dist_precision": 1e-6,
    "refine_iterations": 3
}

# --- Utility Functions ---
# (angle_vector, str_to_indices, indices_to_str, interpolate_pos - unchanged)
def angle_vector(a, b):
    norm_a = np.linalg.norm(a); norm_b = np.linalg.norm(b)
    if norm_a > 0 and norm_b > 0:
        dot_product = np.dot(a, b); cos_angle = np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)
        return np.arccos(cos_angle) * 180 / np.pi
    elif norm_a > 0 or norm_b > 0: return -1.0
    else: return -2.0

def str_to_indices(b1b2b3_str):
    mapping = {'W': 0, 'Y': 1, 'R': 2}
    try: return [mapping[b1b2b3_str[0]], mapping[b1b2b3_str[1]], mapping[b1b2b3_str[2]]]
    except (IndexError, KeyError, TypeError): return None

def indices_to_str(indices):
    mapping = {0: 'W', 1: 'Y', 2: 'R'}
    try: return mapping[indices[0]] + mapping[indices[1]] + mapping[indices[2]]
    except (IndexError, KeyError, TypeError): return "???"

def interpolate_pos(t, t1, t2, x1, x2, y1, y2):
    if abs(t2 - t1) < 1e-12: return x1, y1
    ratio = (t - t1) / (t2 - t1); x = x1 + ratio * (x2 - x1); y = y1 + ratio * (y2 - y1)
    return x, y

# --- Classes ---
# (BallRoute, HitEvent, Shot - unchanged)
class BallRoute:
    def __init__(self, t=None, x=None, y=None):
        self.t = np.array(t, dtype=float) if t is not None else np.array([], dtype=float)
        self.x = np.array(x, dtype=float) if x is not None else np.array([], dtype=float)
        self.y = np.array(y, dtype=float) if y is not None else np.array([], dtype=float)
        self.vx = np.array([], dtype=float); self.vy = np.array([], dtype=float)
        self.v = np.array([], dtype=float); self.dt = np.array([], dtype=float)
        if len(self.t) > 0: self.calculate_velocities()
    def calculate_velocities(self):
        n = len(self.t)
        if n > 1:
            self.dt = np.diff(self.t); self.vx = np.zeros(n, dtype=float); self.vy = np.zeros(n, dtype=float)
            valid_dt = self.dt > 1e-9; dx = np.diff(self.x); dy = np.diff(self.y)
            self.vx[:-1][valid_dt] = dx[valid_dt] / self.dt[valid_dt]; self.vy[:-1][valid_dt] = dy[valid_dt] / self.dt[valid_dt]
            if n > 1: self.vx[-1] = self.vx[-2]; self.vy[-1] = self.vy[-2]
            self.v = np.sqrt(self.vx**2 + self.vy**2); self.dt = np.append(self.dt, self.dt[-1] if len(self.dt)>0 else 0)
        elif n == 1: self.vx = np.zeros(1); self.vy = np.zeros(1); self.v = np.zeros(1); self.dt = np.zeros(1)
        else: self.vx = np.array([]); self.vy = np.array([]); self.v = np.array([]); self.dt = np.array([])
    def copy(self): return copy.deepcopy(self)
    def __len__(self): return len(self.t)
    def append_point(self, t, x, y): self.t = np.append(self.t, t); self.x = np.append(self.x, x); self.y = np.append(self.y, y)
    def remove_first_point(self):
        if len(self) > 0: self.t = self.t[1:]; self.x = self.x[1:]; self.y = self.y[1:]; self.calculate_velocities()
    def update_first_point(self, t, x, y):
         if len(self) > 0: self.t[0] = t; self.x[0] = x; self.y[0] = y; self.calculate_velocities()
    def get_pos_at_time(self, time_target):
        if len(self.t) == 0: return np.nan, np.nan
        if len(self.t) == 1: return self.x[0], self.y[0]
        try:
            interp_x = interp1d(self.t, self.x, kind='linear', bounds_error=False, fill_value=(self.x[0], self.x[-1]))
            interp_y = interp1d(self.t, self.y, kind='linear', bounds_error=False, fill_value=(self.y[0], self.y[-1]))
            return float(interp_x(time_target)), float(interp_y(time_target))
        except ValueError:
             idx = np.searchsorted(self.t, time_target, side='right')
             if idx == 0: return self.x[0], self.y[0]
             if idx >= len(self.t): return self.x[-1], self.y[-1]
             t1, t2 = self.t[idx-1], self.t[idx]; x1, x2 = self.x[idx-1], self.x[idx]; y1, y2 = self.y[idx-1], self.y[idx]
             return interpolate_pos(time_target, t1, t2, x1, x2, y1, y2)

class HitEvent:
    def __init__(self, t, with_char, pos_x, pos_y):
        self.t = t; self.with_char = with_char; self.pos_x = pos_x; self.pos_y = pos_y
        self.type = np.nan; self.from_c = np.nan; self.to_c = np.nan; self.v1 = np.nan; self.v2 = np.nan
        self.offset = np.nan; self.fraction = np.nan; self.def_angle = np.nan; self.cut_angle = np.nan
        self.c_in_angle = np.nan; self.c_out_angle = np.nan; self.from_c_pos = np.nan; self.to_c_pos = np.nan
        self.from_d_pos = np.nan; self.to_d_pos = np.nan; self.point = 0; self.kiss = 0; self.fuchs = 0
        self.point_dist = np.nan; self.kiss_dist_b1 = np.nan; self.t_point = np.inf; self.t_kiss = np.inf
        self.t_ready = np.inf; self.t_b2_hit = np.inf; self.t_failure = np.inf; self.aim_offset_ll = np.nan
        self.aim_offset_ss = np.nan; self.b1b2_offset_ll = np.nan; self.b1b2_offset_ss = np.nan
        self.b1b3_offset_ll = np.nan; self.b1b3_offset_ss = np.nan; self.inside_outside_marker = '?'

class Shot:
    def __init__(self, shot_id, mirrored, point_type, table_index):
        self.shot_id = shot_id; self.mirrored = mirrored; self.point_type = point_type; self.table_index = table_index
        self.route0 = [BallRoute(), BallRoute(), BallRoute()]; self.route = [BallRoute(), BallRoute(), BallRoute()]
        self.hit_events = [[], [], []]; self.b1b2b3_indices = None; self.b1b2b3_str = "???"
        self.interpreted = 0; self.error_code = None; self.error_text = ""; self.selected = False
    def get_ball_route(self, ball_index, original=False):
        if ball_index not in [0, 1, 2]: return BallRoute()
        return self.route0[ball_index] if original else self.route[ball_index]
    def get_b1_route(self, original=False): return self.get_ball_route(self.b1b2b3_indices[0], original) if self.b1b2b3_indices else BallRoute()
    def get_b2_route(self, original=False): return self.get_ball_route(self.b1b2b3_indices[1], original) if self.b1b2b3_indices else BallRoute()
    def get_b3_route(self, original=False): return self.get_ball_route(self.b1b2b3_indices[2], original) if self.b1b2b3_indices else BallRoute()
    def get_ball_hits(self, ball_index): return self.hit_events[ball_index] if ball_index in [0,1,2] else []
    def get_b1_hits(self): return self.get_ball_hits(self.b1b2b3_indices[0]) if self.b1b2b3_indices else []
    def get_b2_hits(self): return self.get_ball_hits(self.b1b2b3_indices[1]) if self.b1b2b3_indices else []
    def get_b3_hits(self): return self.get_ball_hits(self.b1b2b3_indices[2]) if self.b1b2b3_indices else []
    def add_hit_event(self, ball_index, hit_event):
        if ball_index in [0, 1, 2]: self.hit_events[ball_index].append(hit_event)


class ShotAnalyzer:
    """Main class to manage shot data analysis."""
    def __init__(self, params=None):
        self.params = {**DEFAULT_PARAMS, **(params if params else {})}
        self.shots = []
        self.table = pd.DataFrame()

    def _find_shot_index(self, shot_id, mirrored):
        target_val = shot_id + mirrored / 10.0
        try:
            match = self.table['InternalID'] == target_val
            if match.any(): return self.table.index[match].tolist()[0]
            else: return None
        except KeyError: return None

    def read_gamefile(self, filepath):
        # ... (read_gamefile implementation - unchanged) ...
        try:
            with open(filepath, 'r') as f: data = json.load(f)
        except FileNotFoundError: print(f"Error: File not found at {filepath}"); return False
        except json.JSONDecodeError: print(f"Error: Could not decode JSON from {filepath}"); return False
        table_data = []; self.shots = []; si = -1; df_idx = 0
        player1 = data.get('Player1', 'Unknown'); player2 = data.get('Player2', 'Unknown')
        game_type = data.get('Match', {}).get('GameType', 'Unknown'); filename = filepath.split('/')[-1].split('\\')[-1]
        for seti, set_data in enumerate(data.get('Match', {}).get('Sets', [])):
            entries = set_data.get('Entries', []);
            if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], list): entries = entries[0]
            if not isinstance(entries, list): continue
            for entryi, entry in enumerate(entries):
                 if isinstance(entry, dict) and entry.get('PathTrackingId', 0) > 0:
                     si += 1; shot_id = entry['PathTrackingId']; mirrored = 0
                     scoring = entry.get('Scoring', {}); point_type = scoring.get('EntryType', 'Unknown')
                     current_shot = Shot(shot_id, mirrored, point_type, df_idx) # Store initial df_idx
                     path_tracking = entry.get('PathTracking', {}); datasets = path_tracking.get('DataSets', [])
                     valid_route = True
                     if len(datasets) == 3:
                         for bi in range(3):
                             coords = datasets[bi].get('Coords', [])
                             t = [c.get('DeltaT_500us', 0) * 0.0005 for c in coords]
                             x = [c.get('X', 0) * self.params['tableX'] for c in coords]
                             y = [c.get('Y', 0) * self.params['tableY'] for c in coords]
                             if not t: valid_route = False; break
                             current_shot.route0[bi] = BallRoute(t, x, y)
                     else: valid_route = False
                     if not valid_route: print(f"Warning: Skipping ShotID {shot_id} due to incomplete route data."); si -= 1; continue
                     self.shots.append(current_shot)
                     player_num = scoring.get('Player', 0); player_name = player1 if player_num == 1 else (player2 if player_num == 2 else 'Unknown')
                     # Store InternalID using original df_idx - this is crucial
                     row_data = {'Selected': False, 'ShotID': shot_id, 'Mirrored': mirrored, 'InternalID': shot_id + mirrored / 10.0, 'Filename': filename, 'GameType': game_type, 'Interpreted': 0, 'Player': player_name, 'ErrorID': None, 'ErrorText': 'Read OK', 'Set': seti + 1, 'CurrentInning': scoring.get('CurrentInning', 0), 'CurrentSeries': scoring.get('CurrentSeries', 0), 'CurrentTotalPoints': scoring.get('CurrentTotalPoints', 0), 'Point': point_type, 'TableIndex': df_idx }
                     table_data.append(row_data); df_idx += 1
        if not table_data: print("No valid shot data found in file."); return False
        self.table = pd.DataFrame(table_data)
        print(f"Read {len(self.shots)} shots.")
        return True

    def perform_data_quality_checks(self):
        # ... (perform_data_quality_checks implementation - unchanged) ...
        print("Performing data quality checks..."); err_count = 0
        for i, shot in enumerate(self.shots):
            if shot.interpreted: continue
            df_idx = shot.table_index # Use stored index
            error_code = None; error_text = ""; selected = False
            try:
                 errcode_base = 100
                 for bi in range(3):
                     if len(shot.route0[bi]) == 0: error_code = errcode_base + 1; error_text = f'Ball {bi+1} data missing (len 0)'; selected = True; break
                     if len(shot.route0[bi]) == 1: error_code = errcode_base + 2; error_text = f'Ball {bi+1} data insufficient (len 1)'; selected = True; break
                 if selected: raise ValueError(error_text)
                 temp_routes = [r.copy() for r in shot.route0]
                 xmin = self.params['ballR'] + 0.1; xmax = self.params['size'][1] - self.params['ballR'] - 0.1
                 ymin = self.params['ballR'] + 0.1; ymax = self.params['size'][0] - self.params['ballR'] - 0.1
                 for bi in range(3): temp_routes[bi].x = np.clip(temp_routes[bi].x, xmin, xmax); temp_routes[bi].y = np.clip(temp_routes[bi].y, ymin, ymax)
                 errcode_base += 10; bb_idx = [(0, 1), (0, 2), (1, 2)]
                 for bbi, (b1i, b2i) in enumerate(bb_idx):
                     r1 = temp_routes[b1i]; r2 = temp_routes[b2i]
                     if len(r1) > 0 and len(r2) > 0:
                         dx = r1.x[0] - r2.x[0]; dy = r1.y[0] - r2.y[0]; dsq = dx*dx + dy*dy; mindsq = (2 * self.params['ballR'])**2
                         if dsq < mindsq - 1e-6:
                             dist = math.sqrt(dsq) if dsq > 0 else 0; ov = 2 * self.params['ballR'] - dist
                             if dist > 1e-6: vx, vy = dx / dist, dy / dist; r1.x[0] += vx * ov / 2.0; r1.y[0] += vy * ov / 2.0; r2.x[0] -= vx * ov / 2.0; r2.y[0] -= vy * ov / 2.0
                             else: r1.x[0] += ov / 2.0
                 errcode_base += 10
                 for bi in range(3):
                    route = temp_routes[bi]
                    if len(route) > 1:
                        dt = np.diff(route.t); nonlin = np.where(dt <= 0)[0]
                        if len(nonlin) > 0:
                            to_rem = sorted(list(set(nonlin + 1)), reverse=True)
                            if len(route) - len(to_rem) < 2: error_code = errcode_base + 1; error_text = f'Ball {bi+1} time linearity unfixable (too short)'; selected = True; break
                            route.t = np.delete(route.t, to_rem); route.x = np.delete(route.x, to_rem); route.y = np.delete(route.y, to_rem)
                    if selected: break
                 if selected: raise ValueError(error_text)
                 errcode_base += 10
                 starts = [temp_routes[bi].t[0] for bi in range(3) if len(temp_routes[bi]) > 0]
                 ends = [temp_routes[bi].t[-1] for bi in range(3) if len(temp_routes[bi]) > 0]
                 if len(starts)>0 and not np.allclose(starts, starts[0], atol=1e-6): error_code = errcode_base + 1; error_text = 'Inconsistent start times'; selected = True; raise ValueError(error_text)
                 if len(ends)>0 and not np.allclose(ends, ends[0], atol=1e-6): error_code = errcode_base + 2; error_text = 'Inconsistent end times'; selected = True; raise ValueError(error_text)
                 errcode_base += 10 # Reflect skip
                 errcode_base += 10; max_d = self.params['NoDataDistanceDetectionLimit']; max_v = self.params['MaxVelocity']
                 for bi in range(3):
                     route = temp_routes[bi]
                     if len(route) > 1:
                        route.calculate_velocities(); ds = np.sqrt(np.diff(route.x)**2 + np.diff(route.y)**2)
                        if np.any(ds > max_d): error_code = errcode_base + 1; error_text = f'Ball {bi+1} gap too large'; selected = True; break
                        if len(route.v) > 1 and np.any(route.v[:-1] > max_v): error_code = errcode_base + 2; error_text = f'Ball {bi+1} velocity too high'; selected = True; break
                 if selected: raise ValueError(error_text)
                 error_text = "Quality OK"; shot.route0 = temp_routes # IMPORTANT: Update route0 with cleaned routes
            except ValueError as e: print(f"Quality Check Failed for Shot {shot.shot_id} (Idx: {df_idx}): {e}"); err_count += 1
            except Exception as e: print(f"Unexpected Error during Quality Check for Shot {shot.shot_id} (Idx: {df_idx}): {e}"); error_code = 999; error_text = "Unexpected processing error"; selected = True; err_count += 1
            # Update table using the stored df_idx
            if df_idx is not None and df_idx < len(self.table):
                self.table.loc[df_idx, 'ErrorID'] = error_code; self.table.loc[df_idx, 'ErrorText'] = error_text; self.table.loc[df_idx, 'Selected'] = selected
            shot.error_code = error_code; shot.error_text = error_text; shot.selected = selected
        print(f"Data quality checks completed. {err_count} shots flagged.")

    def _determine_b1b2b3_for_shot(self, shot):
        # ... (_determine_b1b2b3_for_shot implementation - unchanged) ...
        if any(len(shot.route0[bi]) < 2 for bi in range(3)): return None, 2, "Empty Data (B1B2B3 determination)"
        try: t2 = [shot.route0[bi].t[1] if len(shot.route0[bi]) > 1 else float('inf') for bi in range(3)]; t_start = shot.route0[0].t[0]
        except IndexError: return None, 2, "Empty Data"
        t2 = [t - t_start for t in t2]; b_indices = np.argsort(t2)
        if t2[b_indices[0]] == float('inf'): return [0, 1, 2], None, "No ball movement detected"
        b1it = b_indices[0]; b2it = b_indices[1]; b3it = b_indices[2]
        if len(shot.route0[b1it]) < 3: return list(b_indices), None, None
        t2_b1 = shot.route0[b1it].t[1]; t3_b1 = shot.route0[b1it].t[2]; moved_b2 = False; moved_b3 = False
        if len(shot.route0[b2it]) > 1 and shot.route0[b2it].t[1] <= t3_b1 + 1e-9: moved_b2 = True
        if len(shot.route0[b3it]) > 1 and shot.route0[b3it].t[1] <= t3_b1 + 1e-9: moved_b3 = True
        final_order = list(b_indices)
        if moved_b2 and moved_b3: return final_order, 3, "All balls moved simultaneously"
        return final_order, None, None

    def determine_b1b2b3(self):
        # ... (determine_b1b2b3 implementation - unchanged) ...
        print("Determining B1, B2, B3..."); err_count = 0
        for i, shot in enumerate(self.shots):
             if shot.interpreted or shot.selected: continue
             df_idx = shot.table_index; indices, err_code, err_text = self._determine_b1b2b3_for_shot(shot)
             if df_idx is None or df_idx >= len(self.table): continue # Skip if index invalid
             if err_code:
                 shot.error_code = err_code; shot.error_text = err_text; shot.selected = True; print(f"B1B2B3 Error Shot {shot.shot_id} (Idx: {df_idx}): {err_text}"); err_count += 1; b1b2b3_str = "ERR"
                 self.table.loc[df_idx, 'ErrorID'] = err_code; self.table.loc[df_idx, 'ErrorText'] = err_text; self.table.loc[df_idx, 'Selected'] = True
             else:
                 shot.b1b2b3_indices = indices; b1b2b3_str = indices_to_str(indices); shot.b1b2b3_str = b1b2b3_str
                 if 'B1B2B3' not in self.table.columns: self.table['B1B2B3'] = "???"
                 self.table.loc[df_idx, 'B1B2B3'] = b1b2b3_str
                 current_err = self.table.loc[df_idx, 'ErrorID']
                 if pd.notna(current_err) and current_err < 10 :
                    self.table.loc[df_idx, 'ErrorID'] = None; self.table.loc[df_idx, 'ErrorText'] = "B1B2B3 OK"; self.table.loc[df_idx, 'Selected'] = False
                    shot.error_code = None; shot.error_text = "B1B2B3 OK"; shot.selected = False
        print(f"B1B2B3 determination completed. {err_count} shots flagged.")

    def _calculate_ball_velocity(self, route, hit_event_time, event_index):
        # ... (_calculate_ball_velocity implementation - unchanged) ...
        t = route.t; x = route.x; y = route.y
        if len(t) < 2: return (0, 0), np.array([0.,0.]), np.array([0.,0.]), (np.nan, np.nan), np.nan
        try:
            ti = np.searchsorted(t, hit_event_time, side='left'); ti = np.clip(ti, 1, len(t) - 1)
            idx1, idx2 = ti-1, ti; v1_vec = np.array([0.0, 0.0]); vt1 = 0.0; alpha1 = np.nan
            if idx2 > idx1:
                dt_b = t[idx2] - t[idx1];
                if dt_b > 1e-9: vx1 = (x[idx2] - x[idx1]) / dt_b; vy1 = (y[idx2] - y[idx1]) / dt_b; v1_vec = np.array([vx1, vy1]); vt1 = np.linalg.norm(v1_vec);
                if vt1 > 1e-9: alpha1 = math.atan2(vy1, vx1) * 180 / np.pi
            idx1, idx2 = ti, ti + 1; v2_vec = np.array([0.0, 0.0]); vt2 = 0.0; alpha2 = np.nan
            if idx2 < len(t):
                 dt_a = t[idx2] - t[idx1];
                 if dt_a > 1e-9: vx2 = (x[idx2] - x[idx1]) / dt_a; vy2 = (y[idx2] - y[idx1]) / dt_a; v2_vec = np.array([vx2, vy2]); vt2 = np.linalg.norm(v2_vec);
                 if vt2 > 1e-9: alpha2 = math.atan2(vy2, vx2) * 180 / np.pi
            offset = np.nan
            return (vt1, vt2), v1_vec, v2_vec, (alpha1, alpha2), offset
        except IndexError: return (0, 0), np.array([0.,0.]), np.array([0.,0.]), (np.nan, np.nan), np.nan
        except Exception: return (0, 0), np.array([0.,0.]), np.array([0.,0.]), (np.nan, np.nan), np.nan

    def _calculate_ball_direction(self, route, hit_event_time, event_index, num_events):
        # ... (_calculate_ball_direction implementation - unchanged) ...
         t = route.t; x = route.x; y = route.y
         if len(t) < 2: return np.full((2, 6), np.nan)
         try:
            ti = np.searchsorted(t, hit_event_time, side='left'); ti = np.clip(ti, 0, len(t) - 1)
            idx_f_s = max(0, ti - self.params['timax_appr']); idx_f_e = max(idx_f_s + 2, ti + 1)
            idx_t_s = min(len(t)-2, ti); idx_t_e = min(len(t), idx_t_s + self.params['timax_appr'])
            results = np.full((2, 6), np.nan); indices = [(idx_f_s, idx_f_e), (idx_t_s, idx_t_e)]
            for i, (ids, ide) in enumerate(indices):
                if ide > ids + 1 and ide <= len(t):
                    ts, xs, ys = t[ids:ide], x[ids:ide], y[ids:ide]; p1 = np.array([xs[0], ys[0]]); p2 = np.array([xs[-1], ys[-1]])
                    if np.allclose(p1, p2): continue
                    y1 = self.params['ballR']; y3 = self.params['size'][0] - y1; y1th = -self.params['diamdist']; y3th = self.params['size'][0] + self.params['diamdist']
                    x2 = self.params['size'][1] - y1; x4 = y1; x2th = self.params['size'][1] + self.params['diamdist']; x4th = -self.params['diamdist']
                    try: ix = interp1d(ys, xs, kind='linear', fill_value="extrapolate", bounds_error=False); iy = interp1d(xs, ys, kind='linear', fill_value="extrapolate", bounds_error=False)
                    except ValueError: continue
                    xn1 = ix(y1); xn3 = ix(y3); xth1 = ix(y1th); xth3 = ix(y3th); yn2 = iy(x2); yn4 = iy(x4); yth2 = iy(x2th); yth4 = iy(x4th)
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    c1v = x4 <= xn1 <= x2; c3v = x4 <= xn3 <= x2; c2v = y1 <= yn2 <= y3; c4v = y1 <= yn4 <= y3
                    if c1v:
                         if dy < -1e-6 : results[i, 3] = 1; results[i, 4] = xn1; results[i, 5] = xth1
                         elif dy > 1e-6: results[i, 0] = 1; results[i, 1] = xn1; results[i, 2] = xth1
                    if c3v:
                         if dy > 1e-6 : results[i, 3] = 3; results[i, 4] = xn3; results[i, 5] = xth3
                         elif dy < -1e-6: results[i, 0] = 3; results[i, 1] = xn3; results[i, 2] = xth3
                    if c2v:
                         if dx > 1e-6 : results[i, 3] = 2; results[i, 4] = yn2; results[i, 5] = yth2
                         elif dx < -1e-6: results[i, 0] = 2; results[i, 1] = yn2; results[i, 2] = yth2
                    if c4v:
                         if dx < -1e-6 : results[i, 3] = 4; results[i, 4] = yn4; results[i, 5] = yth4
                         elif dx > 1e-6: results[i, 0] = 4; results[i, 1] = yn4; results[i, 2] = yth4
            sx = 8.0 / self.params['size'][1]; sy = 4.0 / self.params['size'][0]
            for r in range(2):
                 for c_idx, c in enumerate([1, 2, 4, 5]):
                      if not np.isnan(results[r, c]):
                           col = 0 if c <= 2 else 3; cn = results[r, col]
                           if cn in [1, 3]: results[r, c] *= sx
                           elif cn in [2, 4]: results[r, c] *= sy
            return results
         except Exception: return np.full((2, 6), np.nan)

    def _calculate_cushion_angle(self, cushion_index, v1_vec, v2_vec):
        # ... (_calculate_cushion_angle implementation - unchanged) ...
        v1 = np.append(v1_vec, 0); v2 = np.append(v2_vec, 0); n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9: return np.nan, np.nan
        norms = {1: np.array([0, 1, 0]), 2: np.array([-1, 0, 0]), 3: np.array([0, -1, 0]), 4: np.array([1, 0, 0])}
        if cushion_index not in norms: return np.nan, np.nan
        cn = norms[cushion_index]; a_in = angle_vector(v1, -cn); a_out = angle_vector(v2, cn); sc = False
        if cushion_index in [1, 3] and np.sign(v1[0]) != np.sign(v2[0]) and abs(v1[0]) > 1e-6 and abs(v2[0]) > 1e-6: sc = True
        elif cushion_index in [2, 4] and np.sign(v1[1]) != np.sign(v2[1]) and abs(v1[1]) > 1e-6 and abs(v2[1]) > 1e-6: sc = True
        if sc: a_in = -a_in
        return a_in, a_out

    def _evaluate_hit_details(self, shot):
        # ... (_evaluate_hit_details implementation - unchanged) ...
        if shot.b1b2b3_indices is None: return
        b1i, b2i, b3i = shot.b1b2b3_indices; cols = "WYR"
        for b_idx in range(3):
            hits = shot.hit_events[b_idx]; route = shot.route[b_idx]
            for hi, evt in enumerate(hits):
                if evt.with_char in ['S', '-']: evt.type = 0; continue
                vt, v1v, v2v, _, off = self._calculate_ball_velocity(route, evt.t, hi)
                evt.v1 = vt[0] / 1000.0; evt.v2 = vt[1] / 1000.0; evt.offset = off
                n_evts = len([e for e in hits if e.with_char not in ['S', '-']])
                d_info = self._calculate_ball_direction(route, evt.t, hi, n_evts)
                evt.from_c = d_info[0, 0]; evt.from_c_pos = d_info[0, 1]; evt.from_d_pos = d_info[0, 2]; evt.to_c = d_info[0, 3]; evt.to_c_pos = d_info[0, 4]; evt.to_d_pos = d_info[0, 5]
                contact = evt.with_char; is_c = contact in '1234'; is_b = contact in cols
                if is_c: evt.type = 2; c_idx = int(contact); evt.c_in_angle, evt.c_out_angle = self._calculate_cushion_angle(c_idx, v1v, v2v)
                elif is_b:
                    evt.type = 1; t_char = contact; t_idx = cols.find(t_char)
                    if evt.v1 > 1e-6:
                         v1_t = np.array([0.,0.]); t_hits = shot.hit_events[t_idx]; t_route = shot.route[t_idx]
                         for t_ev in t_hits:
                             if abs(t_ev.t - evt.t) < 1e-9: _, v1_t_c, _, _, _ = self._calculate_ball_velocity(t_route, t_ev.t, 0); v1_t = v1_t_c; break
                         p1 = np.array([evt.pos_x, evt.pos_y]); p2 = np.array([np.nan, np.nan])
                         for t_ev in t_hits:
                              if abs(t_ev.t - evt.t) < 1e-9: p2 = np.array([t_ev.pos_x, t_ev.pos_y]); break
                         if not np.isnan(p2[0]):
                            v_rel = v1v - v1_t; nvr = np.linalg.norm(v_rel)
                            if nvr > 1e-6: p12 = p2 - p1; cr = abs(p12[0] * v_rel[1] - p12[1] * v_rel[0]); pe = cr / nvr; evt.fraction = np.clip(1.0 - pe / (self.params['ballR'] * 2.0), 0.0, 1.0)
                            else: evt.fraction = 0.0
                         nv1 = np.linalg.norm(v1v); nv2 = np.linalg.norm(v2v); evt.def_angle = angle_vector(v1v, v2v) if nv1 > 1e-6 and nv2 > 1e-6 else 0.0
                         v2_t = np.array([0.,0.])
                         for t_ev in t_hits:
                              if abs(t_ev.t - evt.t) < 1e-9: _, _, v2_t_c, _, _ = self._calculate_ball_velocity(t_route, t_ev.t, 0); v2_t = v2_t_c; break
                         nv2t = np.linalg.norm(v2_t); evt.cut_angle = angle_vector(v1v, v2_t) if nv1 > 1e-6 and nv2t > 1e-6 else 0.0
        # Offsets
        b1_route = shot.get_b1_route(); b1_hits = shot.get_b1_hits()
        aim_offset_ll = np.nan
        aim_offset_ss = np.nan

        if len(b1_route) >= 2:
            dx12 = b1_route.x[1] - b1_route.x[0]; dy12 = b1_route.y[1] - b1_route.y[0]
            table_h = self.params['size'][0]; table_w = self.params['size'][1]; diam_dist = self.params['diamdist']

            # Calculate raw offset magnitudes
            if abs(dy12) > 1e-6: aim_offset_ll = abs((dx12 / dy12) * (table_w + 2 * diam_dist)) * 8.0 / table_w
            if abs(dx12) > 1e-6: aim_offset_ss = abs((dy12 / dx12) * (table_h + 2 * diam_dist)) * 4.0 / table_h

            # ***** CORRECTED PART *****
            # Initialize sign factors BEFORE the conditional block
            sign_l, sign_s = 1, 1 # Default sign is positive (no change)

            # Adjust sign based on first hit type (MATLAB logic)
            if len(b1_hits) >= 2: # Check if there's at least one hit after 'Start'
                first_hit_event = b1_hits[1] # Index 0 is 'S'tart
                if first_hit_event.type == 1: # Hit ball B2 first
                    # Placeholder for complex MATLAB sign logic based on relative positions/directions
                    # sign_l, sign_s = calculated_sign_l, calculated_sign_s # Update signs if needed
                    pass # Using default signs for now
                elif first_hit_event.type == 2 and len(b1_hits) >= 3: # Hit cushion first
                    # Placeholder for complex MATLAB sign logic based on path before/after cushion
                    # sign_l, sign_s = calculated_sign_l, calculated_sign_s # Update signs if needed
                    pass # Using default signs for now
                # else: Hit was neither ball nor cushion or not enough events? Keep default signs.
            # ***** END CORRECTED PART *****

            # Apply the sign factors (now guaranteed to be defined)
            if not np.isnan(aim_offset_ll): aim_offset_ll *= sign_l
            if not np.isnan(aim_offset_ss): aim_offset_ss *= sign_s

        # Store B1 offsets in B1's first event (or dedicated shot attribute)
        if b1_hits:
             # Use setattr for safety in case attribute wasn't pre-defined on HitEvent
             setattr(b1_hits[0], 'aim_offset_ll', aim_offset_ll)
             setattr(b1_hits[0], 'aim_offset_ss', aim_offset_ss)

        # --- Calculate B1-B2, B1-B3 initial position offsets ---
        # ... (rest of the B1B2/B1B3 offset calculation remains the same) ...
        p1 = np.array([b1_route.x[0], b1_route.y[0]]) if len(b1_route) > 0 else np.array([np.nan, np.nan])
        b2_route = shot.get_b2_route(); b3_route = shot.get_b3_route()
        p2 = np.array([b2_route.x[0], b2_route.y[0]]) if len(b2_route) > 0 else np.array([np.nan, np.nan])
        p3 = np.array([b3_route.x[0], b3_route.y[0]]) if len(b3_route) > 0 else np.array([np.nan, np.nan])
        b1b2_offset_ll, b1b2_offset_ss = np.nan, np.nan; b1b3_offset_ll, b1b3_offset_ss = np.nan, np.nan
        if not np.isnan(p1[0]) and not np.isnan(p2[0]):
            dx_b1b2 = p2[0] - p1[0]; dy_b1b2 = p2[1] - p1[1]; tw = self.params['size'][1]; th = self.params['size'][0]; dd = self.params['diamdist']
            b1b2_offset_ll = abs((dx_b1b2 / dy_b1b2) * (tw + 2 * dd)) * 8.0 / tw if abs(dy_b1b2) > 1e-6 else 99
            b1b2_offset_ss = abs((dy_b1b2 / dx_b1b2) * (th + 2 * dd)) * 4.0 / th if abs(dx_b1b2) > 1e-6 else 99
        if not np.isnan(p1[0]) and not np.isnan(p3[0]):
            dx_b1b3 = p3[0] - p1[0]; dy_b1b3 = p3[1] - p1[1]; tw = self.params['size'][1]; th = self.params['size'][0]; dd = self.params['diamdist']
            b1b3_offset_ll = abs((dx_b1b3 / dy_b1b3) * (tw + 2 * dd)) * 8.0 / tw if abs(dy_b1b3) > 1e-6 else 99
            b1b3_offset_ss = abs((dy_b1b3 / dx_b1b3) * (th + 2 * dd)) * 4.0 / th if abs(dx_b1b3) > 1e-6 else 99
        if b1_hits:
            setattr(b1_hits[0], 'b1b2_offset_ll', b1b2_offset_ll); setattr(b1_hits[0], 'b1b2_offset_ss', b1b2_offset_ss)
            setattr(b1_hits[0], 'b1b3_offset_ll', b1b3_offset_ll); setattr(b1_hits[0], 'b1b3_offset_ss', b1b3_offset_ss)

        # --- Calculate Inside/Outside shot (for B1 only) ---
        # ... (inside/outside calculation remains the same) ...
        inside_outside = '?'
        if len(b1_hits) >= 4:
            if b1_hits[1].type == 1 and b1_hits[2].type == 2:
                 p1_io = np.array([b1_hits[0].pos_x, b1_hits[0].pos_y, 0]); p2_io = np.array([b1_hits[1].pos_x, b1_hits[1].pos_y, 0])
                 p3_io = np.array([b1_hits[2].pos_x, b1_hits[2].pos_y, 0]); p4_io = np.array([b1_hits[3].pos_x, b1_hits[3].pos_y, 0])
                 v1_io = p1_io - p3_io; v2_io = p2_io - p3_io; v3_io = p4_io - p3_io
                 a1_io = angle_vector(v1_io, v3_io); a2_io = angle_vector(v2_io, v3_io)
                 if not np.isnan(a1_io) and not np.isnan(a2_io): inside_outside = 'I' if a1_io <= a2_io else 'E'
        if b1_hits: setattr(b1_hits[0], 'inside_outside_marker', inside_outside)

    def _evaluate_point_kiss_fuchs(self, shot):
        # ... (_evaluate_point_kiss_fuchs implementation - unchanged) ...
        if shot.b1b2b3_indices is None: return
        b1i, b2i, b3i = shot.b1b2b3_indices; cols = "WYR"; t2c = cols[b2i]; t3c = cols[b3i]
        b1h = shot.get_b1_hits(); b2h = shot.get_b2_hits(); b3h = shot.get_b3_hits(); b1r = shot.get_b1_route(); b2r = shot.get_b2_route(); b3r = shot.get_b3_route()
        b13h = sorted([h for h in b1h if h.with_char == t3c], key=lambda h: h.t); b12h = sorted([h for h in b1h if h.with_char == t2c], key=lambda h: h.t); b1ch = sorted([h for h in b1h if h.with_char in '1234'], key=lambda h: h.t); b23h = sorted([h for h in b2h if h.with_char == t3c], key=lambda h: h.t)
        h12_1 = b12h[0] if b12h else None; h13_1 = b13h[0] if b13h else None; h1c_3 = b1ch[2] if len(b1ch) >= 3 else None; h12_2 = b12h[1] if len(b12h) >= 2 else None; h23_1 = b23h[0] if b23h else None
        pt = 0; ptt = np.inf; ft = np.inf; k = 0; kt = np.inf; f = 0; b123ct = np.inf
        if h12_1 and h13_1 and h1c_3:
            if h13_1.t > h12_1.t + 1e-9 and h13_1.t > h1c_3.t + 1e-9: pt = 1; ptt = h13_1.t
            if h13_1.t > h12_1.t + 1e-9 and h13_1.t < h1c_3.t - 1e-9: ft = h13_1.t
        if pt == 1:
            if h23_1 and h23_1.t < ptt - 1e-9:
                 if h23_1.t < kt: k = 3; kt = h23_1.t; f = 1
            if h12_2 and h12_2.t < ptt - 1e-9:
                 if h12_2.t < kt: k = 1; kt = h12_2.t; f = 1
        elif ft == np.inf:
             if h23_1 and h23_1.t < kt: k = 3; kt = h23_1.t
             if h12_2 and h12_2.t < kt: k = 1; kt = h12_2.t
        if h12_1 and h1c_3: b123ct = max(h12_1.t, h1c_3.t)
        tb2h = h12_1.t if h12_1 else np.inf
        if shot.get_b1_hits():
            se = shot.get_b1_hits()[0]; se.point = pt; se.kiss = k; se.fuchs = f; se.t_point = ptt; se.t_kiss = kt; se.t_ready = b123ct; se.t_b2_hit = tb2h; se.t_failure = ft
        pdist = np.nan
        if pt == 1 and h13_1 and pd.notna(h13_1.fraction): hf = h13_1.fraction; hs = 0; pdist = hs * (1.0 - hf) * self.params['ballR'] * 2.0 if hf != 1.0 else 0
        elif pt == 0 and b123ct != np.inf: pdist = np.nan
        if shot.get_b1_hits(): shot.get_b1_hits()[0].point_dist = pdist
        kdistb1 = np.nan
        if k == 1 and h12_2 and pd.notna(h12_2.fraction): kdistb1 = h12_2.fraction * self.params['ballR'] * 2.0
        elif h12_1: kdistb1 = np.nan
        if shot.get_b1_hits(): shot.get_b1_hits()[0].kiss_dist_b1 = kdistb1

    def _update_table_with_event_data(self, shot):
        """Updates the main DataFrame with calculated event details."""
        if shot.b1b2b3_indices is None: return
        df_idx = shot.table_index
        b1i, b2i, b3i = shot.b1b2b3_indices
        b1_start_event = shot.get_b1_hits()[0] if shot.get_b1_hits() else None

        # Check if DataFrame index is valid
        if df_idx is None or df_idx >= len(self.table):
             print(f"Warning: Invalid table index {df_idx} for Shot {shot.shot_id}. Cannot update table.")
             return # Skip updating this shot's data in the table

        # --- Update summary columns (from B1's start event) ---
        if b1_start_event:
             # Use pre-initialized summary columns
             self.table.loc[df_idx, 'Kiss'] = b1_start_event.kiss
             self.table.loc[df_idx, 'Fuchs'] = b1_start_event.fuchs
             self.table.loc[df_idx, 'PointDist'] = b1_start_event.point_dist
             self.table.loc[df_idx, 'KissDistB1'] = b1_start_event.kiss_dist_b1
             self.table.loc[df_idx, 'AimOffsetLL'] = getattr(b1_start_event, 'aim_offset_ll', np.nan)
             self.table.loc[df_idx, 'AimOffsetSS'] = getattr(b1_start_event, 'aim_offset_ss', np.nan)
             self.table.loc[df_idx, 'B1B2OffsetLL'] = getattr(b1_start_event, 'b1b2_offset_ll', np.nan)
             self.table.loc[df_idx, 'B1B2OffsetSS'] = getattr(b1_start_event, 'b1b2_offset_ss', np.nan)
             self.table.loc[df_idx, 'B1B3OffsetLL'] = getattr(b1_start_event, 'b1b3_offset_ll', np.nan)
             self.table.loc[df_idx, 'B1B3OffsetSS'] = getattr(b1_start_event, 'b1b3_offset_ss', np.nan)
             # Overwrite 'Point' column with calculated value? Or use a new column 'CalcPoint'?
             # self.table.loc[df_idx, 'Point'] = b1_start_event.point # Overwrites input file value
             if 'CalcPoint' not in self.table.columns: self.table['CalcPoint'] = np.nan
             self.table.loc[df_idx, 'CalcPoint'] = b1_start_event.point


        # --- Update detailed event columns ---
        max_events_to_log = 8 # Use the same max used for initialization
        del_names = ['Fraction', 'DefAngle', 'CutAngle', 'CInAngle', 'COutAngle']
        event_attrs = [
            'Type', 'FromC', 'ToC', 'V1', 'V2', 'Offset', 'Fraction',
            'DefAngle', 'CutAngle', 'CInAngle', 'COutAngle',
            'FromCPos', 'ToCPos', 'FromDPos', 'ToDPos'
        ]

        for ball_loop_idx in range(3): # Corresponds to B1, B2, B3 loop
            original_ball_index = shot.b1b2b3_indices[ball_loop_idx]
            hit_events = shot.hit_events[original_ball_index]
            # Skip the first event ('S' or '-') for detailed columns
            num_events_to_process = min(len(hit_events) - 1, max_events_to_log)

            for hi in range(num_events_to_process): # Event index (0 to N-1), maps to event number 1 to N
                 event_no = hi + 1 # Event number 1, 2, ...
                 event = hit_events[event_no] # Get event data (event at index 1 is event #1)

                 for attr in event_attrs:
                      col_name = f"B{ball_loop_idx+1}_{event_no}_{attr}"

                      # Check skip condition from MATLAB (event_no=1 corresponds to matlab_hi=1 ?)
                      # The MATLAB code skips hi=1, which is the *first* non-start hit.
                      # So, skip if event_no == 1
                      if event_no == 1 and attr in del_names:
                          continue

                      value = getattr(event, attr.lower(), np.nan)

                      # *** REMOVED CHECK: Columns are pre-allocated ***
                      # if col_name not in self.table.columns:
                      #    self.table[col_name] = np.nan

                      # Update value (handle potential type issues)
                      # Check if column actually exists (safety for mismatched max_events)
                      if col_name in self.table.columns:
                          try:
                              self.table.loc[df_idx, col_name] = float(value) if not pd.isna(value) else np.nan
                          except (ValueError, TypeError):
                              self.table.loc[df_idx, col_name] = np.nan
                      # else:
                      #     print(f"Warning: Column {col_name} not pre-allocated. Skipping update.")

    def _initialize_event_columns(self, max_events=8):
        """Initializes potential event columns in the DataFrame to prevent fragmentation."""
        print("Initializing event columns in DataFrame...")
        if self.table.empty:
            print("  Table is empty, skipping column initialization.")
            return

        event_attrs = [
            'Type', 'FromC', 'ToC', 'V1', 'V2', 'Offset', 'Fraction',
            'DefAngle', 'CutAngle', 'CInAngle', 'COutAngle',
            'FromCPos', 'ToCPos', 'FromDPos', 'ToDPos'
        ]
        # Attributes from B1 start event (summary values)
        summary_attrs = [
             'Point', 'Kiss', 'Fuchs', 'PointDist', 'KissDistB1',
             'AimOffsetLL', 'AimOffsetSS', 'B1B2OffsetLL', 'B1B2OffsetSS',
             'B1B3OffsetLL', 'B1B3OffsetSS'
        ]

        new_cols = []
        # Add summary columns if they don't exist
        for attr in summary_attrs:
            if attr not in self.table.columns:
                new_cols.append(attr)

        # Add detailed event columns if they don't exist
        # Skip event 0 ('S'tart event - details not typically stored like this)
        for ball_loop_idx in range(1, 4): # B1, B2, B3
            for event_no in range(1, max_events + 1): # Event no 1 to max_events
                for attr in event_attrs:
                    col_name = f"B{ball_loop_idx}_{event_no}_{attr}"
                    if col_name not in self.table.columns:
                        new_cols.append(col_name)

        if new_cols:
            # Add all new columns at once
            current_cols = self.table.columns.tolist()
            # Create a DataFrame of NaNs for the new columns matching the index
            nan_df = pd.DataFrame(np.nan, index=self.table.index, columns=new_cols)
            # Concatenate along columns axis=1
            self.table = pd.concat([self.table, nan_df], axis=1)
            print(f"  Added {len(new_cols)} potential event columns.")
        else:
            print("  All potential event columns already exist.")
    
    
    
    # ==============================================================
    # ===== VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV =====
    # =====   IMPLEMENTED EVENT DETECTION LOGIC START          =====
    # ==============================================================
    def _detect_events_for_shot(self, shot):
        """
        Detects collision events for a single shot using iterative prediction
        and interpolation, based on MATLAB's Extract_Events logic.
        Populates shot.route (refined trajectories) and shot.hit_events.
        """
        try:
            cols = "WYR"
            ballR = self.params['ballR']
            minvel = self.params['minvel']
            tc_precision = self.params['tc_precision']
            dist_precision = self.params['dist_precision']
            table_size_y, table_size_x = self.params['size']

            # --- Initialization ---
            shot.hit_events = [[], [], []] # Reset events
            shot.route = [BallRoute(), BallRoute(), BallRoute()] # Reset refined routes
            # Use route0 (cleaned original data) as the source for remaining trajectory
            ball0 = [r.copy() for r in shot.route0]

            if any(len(b) < 2 for b in ball0):
                print(f"  Skipping event detection for Shot {shot.shot_id}: Insufficient route points after cleaning.");
                shot.error_code = 201; shot.error_text = "Insufficient route points for event detection"; shot.selected = True
                return False

            # Initialize refined routes and add start events
            for bi in range(3):
                shot.route[bi].append_point(ball0[bi].t[0], ball0[bi].x[0], ball0[bi].y[0])
                start_char = 'S' if bi == shot.b1b2b3_indices[0] else '-'
                start_event = HitEvent(t=ball0[bi].t[0], with_char=start_char, pos_x=ball0[bi].x[0], pos_y=ball0[bi].y[0])
                shot.add_hit_event(bi, start_event)
                ball0[bi].calculate_velocities() # Ensure initial velocities are calculated

            # Master time vector (use union of all time points for accuracy if needed, but start with ball 0)
            # Tall0_all = sorted(list(set(np.concatenate([b.t for b in ball0])))) # More robust but complex
            Tall0 = ball0[0].t.copy() # Simpler: Use B1's time (assuming reasonable consistency)
            Tall0 = Tall0[Tall0 >= shot.route[0].t[0]] # Ensure we start from the initial time

            # --- Main Iterative Loop ---
            loop_count = 0
            max_loops = len(Tall0) * 5 # Safety break

            while len(Tall0) >= 2 and loop_count < max_loops:
                loop_count += 1
                t1 = Tall0[0]
                t2 = Tall0[1]
                dt = t2 - t1

                if dt < tc_precision: # Skip steps with negligible time difference
                    Tall0 = Tall0[1:]
                    # Sync ball0 starting points if needed (might have different indices)
                    for bi in range(3):
                        idx = np.searchsorted(ball0[bi].t, t1, side='left')
                        if idx > 0: # Remove preceding points up to t1
                            ball0[bi].t = ball0[bi].t[idx:]
                            ball0[bi].x = ball0[bi].x[idx:]
                            ball0[bi].y = ball0[bi].y[idx:]
                            ball0[bi].calculate_velocities()
                    continue

                # Current state at t1
                pos1 = np.array([b.get_pos_at_time(t1) for b in ball0])
                vel1 = np.zeros((3, 2))
                vmag1 = np.zeros(3)
                for bi in range(3):
                    # Find velocity valid at or just before t1
                    idx1 = np.searchsorted(ball0[bi].t, t1, side='right') - 1
                    idx1 = max(0, idx1) # Ensure index is not negative
                    if idx1 < len(ball0[bi].vx):
                        vel1[bi, 0] = ball0[bi].vx[idx1]
                        vel1[bi, 1] = ball0[bi].vy[idx1]
                        vmag1[bi] = ball0[bi].v[idx1]


                # Predicted positions at t2 (linear extrapolation from t1)
                pos2_pred = pos1 + vel1 * dt

                # --- Calculate Distances (at t1 and predicted t2) ---
                distBB1 = np.full((3, 3), np.inf); distBB2 = np.full((3, 3), np.inf)
                distC1 = np.full((3, 4), np.inf); distC2 = np.full((3, 4), np.inf)
                cushion_limits = [ ballR, table_size_x - ballR, table_size_y - ballR, ballR ] # Ymin, Xmax, Ymax, Xmin

                for bi in range(3):
                    distC1[bi, 0] = pos1[bi, 1] - cushion_limits[0] # Y - Ymin (Cush 1)
                    distC1[bi, 1] = cushion_limits[1] - pos1[bi, 0] # Xmax - X (Cush 2)
                    distC1[bi, 2] = cushion_limits[2] - pos1[bi, 1] # Ymax - Y (Cush 3)
                    distC1[bi, 3] = pos1[bi, 0] - cushion_limits[3] # X - Xmin (Cush 4)
                    distC2[bi, 0] = pos2_pred[bi, 1] - cushion_limits[0]
                    distC2[bi, 1] = cushion_limits[1] - pos2_pred[bi, 0]
                    distC2[bi, 2] = cushion_limits[2] - pos2_pred[bi, 1]
                    distC2[bi, 3] = pos2_pred[bi, 0] - cushion_limits[3]
                    for bj in range(bi + 1, 3):
                        dx1 = pos1[bi, 0] - pos1[bj, 0]; dy1 = pos1[bi, 1] - pos1[bj, 1]; distBB1[bi, bj] = distBB1[bj, bi] = math.sqrt(dx1*dx1 + dy1*dy1)
                        dx2 = pos2_pred[bi, 0] - pos2_pred[bj, 0]; dy2 = pos2_pred[bi, 1] - pos2_pred[bj, 1]; distBB2[bi, bj] = distBB2[bj, bi] = math.sqrt(dx2*dx2 + dy2*dy2)

                # --- Hit Detection ---
                hitlist = [] # [ {tc, type, bi, other, x, y}, ... ]
                min_tc = t2 + tc_precision

                # Cushion Hits
                for bi in range(3):
                    if vmag1[bi] < minvel: continue
                    for ci in range(4):
                        moving_towards = False
                        if ci == 0 and vel1[bi, 1] < -minvel/10.0: moving_towards = True # Down
                        elif ci == 1 and vel1[bi, 0] > minvel/10.0: moving_towards = True # Right
                        elif ci == 2 and vel1[bi, 1] > minvel/10.0: moving_towards = True # Up
                        elif ci == 3 and vel1[bi, 0] < -minvel/10.0: moving_towards = True # Left

                        if not moving_towards: continue
                        # Check if distance crosses zero or is within range and decreasing
                        crosses_zero = distC1[bi, ci] >= -dist_precision and distC2[bi, ci] < dist_precision
                        decreasing_in_range = abs(distC1[bi, ci]) < self.params['BallCushionHitDetectionRange'] and distC1[bi, ci] > distC2[bi, ci] - dist_precision

                        if crosses_zero or decreasing_in_range:
                             # Interpolate time tc
                             tc = t1
                             denominator = distC1[bi, ci] - distC2[bi, ci]
                             if abs(denominator) > dist_precision:
                                 tc = t1 + dt * distC1[bi, ci] / denominator
                             tc = np.clip(tc, t1 - tc_precision, t2 + tc_precision) # Clip within interval (allow slight overshoot)

                             if t1 - tc_precision <= tc < min_tc: # Check if earliest
                                 cushX, cushY = interpolate_pos(tc, t1, t2, pos1[bi, 0], pos2_pred[bi, 0], pos1[bi, 1], pos2_pred[bi, 1])
                                 # Check if interpolated position is valid (optional)
                                 if 0 <= cushX <= table_size_x and 0 <= cushY <= table_size_y:
                                     hitlist.append({'tc': tc, 'type': 'C', 'bi': bi, 'other': ci + 1, 'x': cushX, 'y': cushY})
                                     min_tc = min(min_tc, tc)


                # Ball-Ball Hits
                for bi in range(3):
                    for bj in range(bi + 1, 3):
                        # Check if distance crosses threshold or is decreasing within range
                        crosses_thresh = distBB1[bi, bj] >= 2*ballR - dist_precision and distBB2[bi, bj] < 2*ballR + dist_precision
                        decreasing_in_range = abs(distBB1[bi, bj] - 2*ballR) < ballR and distBB1[bi, bj] > distBB2[bi, bj] - dist_precision

                        if crosses_thresh or decreasing_in_range:
                            # Basic check if balls are moving towards each other
                            vel_rel = vel1[bi] - vel1[bj]
                            pos_rel = pos1[bi] - pos1[bj]
                            if np.dot(vel_rel, pos_rel) < 0 or np.linalg.norm(vel_rel) < minvel / 5.0: # Moving apart or too slow relative

                               # --- Special Case Check (from MATLAB comments) ---
                               # Check if B2 or B3 *just* started moving - indicates missed interpolation
                               b1i_glob, b2i_glob, b3i_glob = shot.b1b2b3_indices
                               is_initial_b1b2 = (bi == b1i_glob and bj == b2i_glob and len(shot.hit_events[b2i_glob]) == 1 and vmag1[bj] > minvel)
                               is_initial_b1b3 = (bi == b1i_glob and bj == b3i_glob and len(shot.hit_events[b3i_glob]) == 1 and vmag1[bj] > minvel)

                               if not (is_initial_b1b2 or is_initial_b1b3) : continue # Skip if not moving towards or special case
                               # --- End Special Case Check ---


                            # Calculate tc using quadratic solver for ||pos_rel + vel_rel*t||^2 = (2*ballR)^2
                            a = np.dot(vel_rel, vel_rel)
                            b = 2 * np.dot(pos_rel, vel_rel)
                            c = np.dot(pos_rel, pos_rel) - (2*ballR)**2
                            tc_sol = np.inf

                            if abs(a) > 1e-9:
                                delta = b*b - 4*a*c
                                if delta >= 0:
                                    sqrt_delta = math.sqrt(delta)
                                    t_sol1 = (-b + sqrt_delta) / (2*a); t_sol2 = (-b - sqrt_delta) / (2*a)
                                    valid_sols = [s for s in [t_sol1, t_sol2] if 0 - tc_precision <= s <= dt + tc_precision]
                                    if valid_sols: tc_sol = t1 + min(valid_sols)
                            elif abs(b) > 1e-9:
                                t_sol = -c / b
                                if 0 - tc_precision <= t_sol <= dt + tc_precision: tc_sol = t1 + t_sol

                            # Check if this tc is the earliest valid hit
                            if t1 - tc_precision <= tc_sol < min_tc:
                                hitX_bi, hitY_bi = interpolate_pos(tc_sol, t1, t2, pos1[bi, 0], pos2_pred[bi, 0], pos1[bi, 1], pos2_pred[bi, 1])
                                hitX_bj, hitY_bj = interpolate_pos(tc_sol, t1, t2, pos1[bj, 0], pos2_pred[bj, 0], pos1[bj, 1], pos2_pred[bj, 1])
                                hitlist.append({'tc': tc_sol, 'type': 'B', 'bi': bi, 'other': bj, 'x': hitX_bi, 'y': hitY_bi})
                                hitlist.append({'tc': tc_sol, 'type': 'B', 'bi': bj, 'other': bi, 'x': hitX_bj, 'y': hitY_bj})
                                min_tc = min(min_tc, tc_sol)


                # --- Trajectory Refinement ---
                actual_hits_at_tc = [h for h in hitlist if abs(h['tc'] - min_tc) < tc_precision]

                if actual_hits_at_tc: # Hit occurred
                    tc = min_tc
                    balls_hit_this_step = set(h['bi'] for h in actual_hits_at_tc)

                    processed_hit_pairs = set() # Avoid duplicate B-B events

                    for hit_info in actual_hits_at_tc:
                        bi = hit_info['bi']
                        hit_type = hit_info['type']
                        other_idx = hit_info['other']
                        hitX = hit_info['x']
                        hitY = hit_info['y']

                        # Avoid adding same B-B event twice
                        if hit_type == 'B':
                            pair = tuple(sorted((bi, other_idx)))
                            if pair in processed_hit_pairs: continue
                            processed_hit_pairs.add(pair)


                        # Append hit point to refined route if time is new
                        if abs(shot.route[bi].t[-1] - tc) > tc_precision:
                             shot.route[bi].append_point(tc, hitX, hitY)

                        # Create HitEvent
                        if hit_type == 'C': with_char = str(other_idx)
                        elif hit_type == 'B': with_char = cols[other_idx]
                        else: with_char = '?'
                        event = HitEvent(t=tc, with_char=with_char, pos_x=hitX, pos_y=hitY)
                        shot.add_hit_event(bi, event)

                        # Update ball0 start point only if tc is ahead of current start
                        if tc > ball0[bi].t[0] + tc_precision:
                             idx_start = np.searchsorted(ball0[bi].t, tc, side='left')
                             ball0[bi].t = np.insert(ball0[bi].t[idx_start:], 0, tc)
                             ball0[bi].x = np.insert(ball0[bi].x[idx_start:], 0, hitX)
                             ball0[bi].y = np.insert(ball0[bi].y[idx_start:], 0, hitY)
                        elif abs(tc - ball0[bi].t[0]) < tc_precision:
                             ball0[bi].x[0] = hitX # Update position if time matches
                             ball0[bi].y[0] = hitY
                        # Else: tc is before current ball0 start? Should not happen if logic is correct


                    # Update balls that *didn't* have a hit at tc
                    for bi in range(3):
                        if bi not in balls_hit_this_step:
                            if abs(shot.route[bi].t[-1] - tc) > tc_precision:
                                # Interpolate position at tc
                                interpX, interpY = ball0[bi].get_pos_at_time(tc) # Interpolate from original data copy
                                shot.route[bi].append_point(tc, interpX, interpY)

                            # Update ball0 start point
                            if tc > ball0[bi].t[0] + tc_precision:
                                interpX_b0, interpY_b0 = ball0[bi].get_pos_at_time(tc)
                                idx_start = np.searchsorted(ball0[bi].t, tc, side='left')
                                ball0[bi].t = np.insert(ball0[bi].t[idx_start:], 0, tc)
                                ball0[bi].x = np.insert(ball0[bi].x[idx_start:], 0, interpX_b0)
                                ball0[bi].y = np.insert(ball0[bi].y[idx_start:], 0, interpY_b0)
                            elif abs(tc - ball0[bi].t[0]) < tc_precision:
                                pass # Start time already matches


                    # Update master time vector only if tc is new
                    if abs(Tall0[0] - tc) > tc_precision:
                         Tall0 = np.insert(Tall0, 1, tc) # Insert tc
                    Tall0 = Tall0[1:] # Advance past original t1 or inserted tc


                else: # No hit detected, advance to next original time step t2
                    t_next = t2
                    processed_indices = set() # Track which ball0 have been advanced

                    # Find the corresponding index for t_next in each ball0
                    for bi in range(3):
                         idx_next = np.searchsorted(ball0[bi].t, t_next, side='left')

                         if idx_next < len(ball0[bi].t) and abs(ball0[bi].t[idx_next] - t_next) < tc_precision:
                            # Found the exact time step t2 in this ball's route
                            x_next, y_next = ball0[bi].x[idx_next], ball0[bi].y[idx_next]
                            if abs(shot.route[bi].t[-1] - t_next) > tc_precision:
                                shot.route[bi].append_point(t_next, x_next, y_next)

                            # Remove points from ball0 up to and including t_next
                            ball0[bi].t = ball0[bi].t[idx_next+1:]
                            ball0[bi].x = ball0[bi].x[idx_next+1:]
                            ball0[bi].y = ball0[bi].y[idx_next+1:]
                            processed_indices.add(bi)

                         elif idx_next >= 1: # t_next falls between points
                             if abs(shot.route[bi].t[-1] - t_next) > tc_precision:
                                 interpX, interpY = ball0[bi].get_pos_at_time(t_next)
                                 shot.route[bi].append_point(t_next, interpX, interpY)
                             # Remove points before t_next, keep the one just after
                             ball0[bi].t = ball0[bi].t[idx_next:]
                             ball0[bi].x = ball0[bi].x[idx_next:]
                             ball0[bi].y = ball0[bi].y[idx_next:]
                             # Update the first point to be exactly at t_next? Might cause issues.
                             processed_indices.add(bi)


                    # Remove t1 from Tall0
                    Tall0 = Tall0[1:]

                # Recalculate velocities for the remaining parts of ball0
                for bi in range(3):
                    ball0[bi].calculate_velocities()


            # --- Finalization ---
            for bi in range(3):
                 if len(shot.route0[bi]) > 0 and len(shot.route[bi]) > 0:
                     if shot.route[bi].t[-1] < shot.route0[bi].t[-1] - tc_precision:
                         shot.route[bi].append_point(shot.route0[bi].t[-1], shot.route0[bi].x[-1], shot.route0[bi].y[-1])
                 shot.route[bi].calculate_velocities()

            # print(f"  Event detection for Shot {shot.shot_id} finished. Found {sum(len(h) for h in shot.hit_events)} raw events.")
            return True

        except Exception as e:
             print(f"  ERROR during event detection for Shot {shot.shot_id}: {e}")
             import traceback; traceback.print_exc()
             shot.error_code = 299; shot.error_text = f"Event detection error: {e}"; shot.selected = True
             return False
    # ==============================================================
    # =====   IMPLEMENTED EVENT DETECTION LOGIC END            =====
    # ===== ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ =====
    # ==============================================================

    def extract_events(self):
        # ... (extract_events implementation - unchanged, calls _detect_events...) ...
        print("Extracting Events..."); total_shots = len(self.shots); pc = 0; ec = 0
        for i, shot in enumerate(self.shots):
            print(f"Processing shot {i+1}/{total_shots} (ID: {shot.shot_id})")
            if shot.interpreted or shot.selected or shot.b1b2b3_indices is None:
                print(f"  Skipping shot {shot.shot_id} (Interpreted: {shot.interpreted}, Selected: {shot.selected}, B1B2B3: {shot.b1b2b3_str})")
                if shot.selected and not shot.error_code: shot.error_code = 998; shot.error_text = "Skipped (pre-selected or missing B1B2B3)"
                if shot.table_index is not None and shot.table_index < len(self.table): self.table.loc[shot.table_index, 'ErrorID'] = shot.error_code; self.table.loc[shot.table_index, 'ErrorText'] = shot.error_text
                continue
            df_idx = shot.table_index
            try:
                if not self._detect_events_for_shot(shot):
                     ec += 1
                     if df_idx is not None and df_idx < len(self.table): self.table.loc[df_idx, 'ErrorID'] = shot.error_code; self.table.loc[df_idx, 'ErrorText'] = shot.error_text; self.table.loc[df_idx, 'Selected'] = True
                     continue
                self._evaluate_hit_details(shot)
                self._evaluate_point_kiss_fuchs(shot)
                self._update_table_with_event_data(shot)
                shot.interpreted = 1; self.table.loc[df_idx, 'Interpreted'] = 1; self.table.loc[df_idx, 'ErrorID'] = None; self.table.loc[df_idx, 'ErrorText'] = "Interpreted OK"; self.table.loc[df_idx, 'Selected'] = False
                shot.error_code = None; shot.error_text = "Interpreted OK"; shot.selected = False; pc += 1
            except Exception as e:
                print(f"  ERROR processing Shot {shot.shot_id} (Idx: {df_idx}): {e}"); import traceback; traceback.print_exc()
                shot.error_code = 1000; shot.error_text = f"Event processing error: {e}"; shot.selected = True
                if df_idx is not None and df_idx < len(self.table): self.table.loc[df_idx, 'ErrorID'] = shot.error_code; self.table.loc[df_idx, 'ErrorText'] = shot.error_text; self.table.loc[df_idx, 'Selected'] = True
                ec += 1
        print(f"Event extraction completed. Processed: {pc}, Errors: {ec}")


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++ NEW FUNCTION START +++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def remove_flagged_shots(self):
        """Removes shots flagged as 'Selected' from the analysis."""
        if not isinstance(self.table, pd.DataFrame) or 'Selected' not in self.table.columns:
            print("Cannot remove flagged shots: Table is not initialized or 'Selected' column missing.")
            return

        initial_shot_count = len(self.shots)
        initial_table_rows = len(self.table)

        # 1. Identify shots/rows to keep
        keep_mask = self.table['Selected'] == False
        if keep_mask.all():
            print("No shots flagged for removal.")
            return

        kept_table = self.table[keep_mask].copy()
        kept_internal_ids = set(kept_table['InternalID'].unique()) # Use InternalID for matching

        # 2. Filter the shots list
        kept_shots = [
            shot for shot in self.shots
            if (shot.shot_id + shot.mirrored / 10.0) in kept_internal_ids
        ]

        num_removed = initial_shot_count - len(kept_shots)
        print(f"Removing {num_removed} flagged shots...")

        # 3. Update the main table and shots list
        self.table = kept_table.reset_index(drop=True)
        self.shots = kept_shots

        # 4. Update the 'table_index' in the remaining Shot objects
        if not self.table.empty:
            # Create a mapping from InternalID to new DataFrame index
            id_to_new_index = pd.Series(self.table.index, index=self.table['InternalID'])
            for shot in self.shots:
                internal_id = shot.shot_id + shot.mirrored / 10.0
                new_index = id_to_new_index.get(internal_id)
                if new_index is not None:
                    shot.table_index = new_index
                else:
                    # This case should ideally not happen if filtering was correct
                    print(f"Warning: Could not find new table index for ShotID {shot.shot_id}, Mirrored {shot.mirrored}. Setting index to None.")
                    shot.table_index = None
        else:
             # If table is empty, set all indices to None
             for shot in self.shots:
                 shot.table_index = None


        print(f"Removal complete. Remaining shots: {len(self.shots)}. Remaining table rows: {len(self.table)}")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++ NEW FUNCTION END +++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    def run_analysis(self, filepath, remove_flagged=False): # Added flag
        """Executes the full analysis pipeline."""
        print(f"Starting analysis for: {filepath}")
        if not self.read_gamefile(filepath): print("Analysis stopped: Could not read game file."); return
        
        self._initialize_event_columns() # Pre-allocate columns
        self.perform_data_quality_checks()
        self.determine_b1b2b3()
        self.calculate_b1b2b3_positions()
        self.extract_events()

        # --- Optionally remove flagged shots ---
        if remove_flagged:
            self.remove_flagged_shots()
        # --- ---

        print("Analysis finished.")
        if 'Selected' in self.table.columns:
             flagged_count = self.table['Selected'].sum() # Count remaining flagged (should be 0 if remove_flagged=True)
             print(f"{flagged_count} shots currently flagged with issues.")
        else: print("Selected column not found in results.")

    def calculate_b1b2b3_positions(self):
        # ... (calculate_b1b2b3_positions implementation - unchanged) ...
        print("Calculating initial positions...");
        if 'B1B2B3' not in self.table.columns: print("  Skipping: B1B2B3 column not found."); return
        sx = 8.0 / self.params['size'][1]; sy = 4.0 / self.params['size'][0]
        for i, shot in enumerate(self.shots):
             if shot.selected or shot.b1b2b3_indices is None: continue
             df_idx = shot.table_index
             if df_idx is None or df_idx >= len(self.table): continue
             for bl_idx in range(3):
                 orig_idx = shot.b1b2b3_indices[bl_idx]; r = shot.route0[orig_idx]
                 px, py = (r.x[0] * sx, r.y[0] * sy) if len(r) > 0 else (np.nan, np.nan)
                 cx = f'B{bl_idx+1}posX'; cy = f'B{bl_idx+1}posY'
                 if cx not in self.table.columns: self.table[cx] = np.nan
                 if cy not in self.table.columns: self.table[cy] = np.nan
                 self.table.loc[df_idx, cx] = px; self.table.loc[df_idx, cy] = py

    def save_results(self, output_filepath):
        """
        Saves the resulting table to a CSV file with semicolon separators
        and comma decimal separators.
        """
        try:
            all_cols = self.table.columns.tolist()
            print(f"Saving results to {output_filepath} (Separator=';', Decimal=',')")
            # Add sep=';' and decimal=',' parameters
            self.table.to_csv(
                output_filepath,
                index=False,
                columns=all_cols,
                sep=';',      # Use semicolon as separator
                decimal=','   # Use comma as decimal separator
            )
            print(f"Results saved successfully.")
        except Exception as e:
            print(f"Error saving results to {output_filepath}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    filepath = "D:\\Billard\\0_AllDatabase\\WebSport\\20181031_match_03_Ersin_Cemal.txt"
    if not os.path.exists(filepath): print(f"Error: Input file not found at {filepath}"); exit()

    analyzer = ShotAnalyzer()
    # Set remove_flagged=True to remove flagged shots after analysis
    analyzer.run_analysis(filepath, remove_flagged=True)

    #print("\n--- Analysis Summary Table (First 10 rows) ---")
    pd.set_option('display.max_rows', 20); pd.set_option('display.max_columns', 20); pd.set_option('display.width', 120)
    #print(analyzer.table.head(10).to_string())

    #print("\n--- Flagged Shots (Should be empty if remove_flagged=True) ---")
    if 'Selected' in analyzer.table.columns:
        flagged = analyzer.table[analyzer.table['Selected'] == True]
        if not flagged.empty: print(flagged[['ShotID', 'ErrorID', 'ErrorText']].to_string(index=False))
        else: print("No shots flagged.")
    else: print("Selected column not found.")

    analyzer.save_results("analysis_output_cleaned.csv") # Save cleaned results

# <<< CODE ENDS >>>