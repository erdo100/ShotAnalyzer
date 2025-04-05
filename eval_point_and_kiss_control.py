import numpy as np
from typing import Dict, Any
import pandas as pd
from str2num_b1b2b3 import str2num_b1b2b3
from eval_kiss import eval_kiss

def eval_point_and_kiss_control(si: int, hit: Dict[int, Dict[str, Any]], SA: Dict[str, Any], param: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Evaluate point and kiss control metrics for a shot
    
    Args:
        si: Shot index
        hit: Dictionary containing hit data for each ball
        SA: Dictionary containing shot data with nested Route structure
        param: Dictionary containing parameters
        
    Returns:
        Dictionary with updated hit data including point and kiss metrics
    """
    cols = 'WYR'
    
    # Get shot information
    shot_id = SA['Table'].loc[si, 'ShotID']
    b1b2b3_str = SA['Table'].loc[si, 'B1B2B3']
    b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(b1b2b3_str)
    
    # Get ball trajectories from nested structure
    ball = {}
    for bi in [0, 1, 2]:  # Using 0-based indexing
        ball_id = b1b2b3[bi]  # Get actual ball ID from order
        if shot_id in SA['Route'] and ball_id in SA['Route'][shot_id]:
            ball_df = SA['Route'][shot_id][ball_id]
            ball[bi] = {
                't': ball_df['t'].values,
                'x': ball_df['x'].values,
                'y': ball_df['y'].values
            }
        else:
            # Handle missing ball data
            ball[bi] = {'t': np.array([]), 'x': np.array([]), 'y': np.array([])}
            hit[bi]['PointDist'] = 3000
            hit[bi]['KissDistB1'] = 3000
            continue
    
    # Evaluate kiss events
    hit = eval_kiss(hit, b1i, b2i, b3i)
    
    # Calculate ball-ball distances
    BB = [(0, 1), (0, 2), (1, 2)]  # Ball pairs
    
    # Get common time points only if all balls have data
    if all(len(ball[bi]['t']) > 0 for bi in [0, 1, 2]):
        Tall = np.unique(np.concatenate([ball[bi]['t'] for bi in [0, 1, 2]]))
    else:
        Tall = np.array([])
    
    CC = {}
    for bbi, (bx1, bx2) in enumerate(BB):
        # Only calculate if both balls have data
        if len(ball[bx1]['t']) > 0 and len(ball[bx2]['t']) > 0:
            # Interpolate positions to common time points
            B1x = np.interp(Tall, ball[bx1]['t'], ball[bx1]['x'])
            B1y = np.interp(Tall, ball[bx1]['t'], ball[bx1]['y'])
            B2x = np.interp(Tall, ball[bx2]['t'], ball[bx2]['x'])
            B2y = np.interp(Tall, ball[bx2]['t'], ball[bx2]['y'])
            
            CC[bbi] = {
                'dist': np.sqrt((B2x - B1x)**2 + (B2y - B1y)**2)
            }
        else:
            CC[bbi] = {'dist': np.full(len(Tall), np.nan)}
    
    # Evaluate kiss thickness for B1
    if hit[b1i]['Kiss'] == 1 and len(ball[b1i]['t']) > 0 and len(ball[b2i]['t']) > 0:
        ei = np.where(hit[b1i]['t'] == hit[b1i]['Tkiss'])[0][0] if np.any(hit[b1i]['t'] == hit[b1i]['Tkiss']) else -1
        if ei >= 0:
            hit[b1i]['KissDistB1'] = hit[b1i]['Fraction'][ei] * param['ballR'] * 2
        else:
            hit[b1i]['KissDistB1'] = 3000
    else:
        # No B1-B2 kiss, evaluate closest passage if data exists
        if len(Tall) > 0 and 0 in CC and len(CC[0]['dist']) > 0:
            ind1 = np.where(Tall > hit[b1i]['TB2hit'])[0][0] if np.any(Tall > hit[b1i]['TB2hit']) else len(Tall)-1
            diff_dist = np.diff(CC[0]['dist'][ind1:])
            ind2 = ind1 + np.where(diff_dist < 0)[0][0] if np.any(diff_dist < 0) else len(Tall)-1
            
            # Find limiting time indices
            ind3_options = []
            if 'Tkiss' in hit[b1i]:
                ind3_options.append(np.where(Tall >= hit[b1i]['Tkiss'])[0][0] if np.any(Tall >= hit[b1i]['Tkiss']) else len(Tall)-1)
            if 'Tpoint' in hit[b1i]:
                ind3_options.append(np.where(Tall >= hit[b1i]['Tpoint'])[0][0] if np.any(Tall >= hit[b1i]['Tpoint']) else len(Tall)-1)
            if 'Tfailure' in hit[b1i]:
                ind3_options.append(np.where(Tall >= hit[b1i]['Tfailure'])[0][0] if np.any(Tall >= hit[b1i]['Tfailure']) else len(Tall)-1)
            
            ind3 = min(ind3_options) if ind3_options else len(Tall)-1
            
            # Find minimum distance in the range
            hit[b1i]['KissDistB1'] = np.min(CC[0]['dist'][ind2:ind3+1]) if ind2 <= ind3 else 3000
        else:
            hit[b1i]['KissDistB1'] = 3000
    
    # Evaluate point accuracy
    if hit[b1i]['Point'] and len(ball[b1i]['t']) > 0 and len(ball[b3i]['t']) > 0:
        # Find hit indices
        hi1 = np.where(np.array(hit[b1i]['with']) == b1b2b3_str[2])[0][0] if np.any(np.array(hit[b1i]['with']) == b1b2b3_str[2]) else -1
        hi3 = np.where(np.array(hit[b3i]['with']) == b1b2b3_str[0])[0][0] if np.any(np.array(hit[b3i]['with']) == b1b2b3_str[0]) else -1
        
        if hi1 >= 0 and hi3 >= 0:
            if hit[b1i]['Fraction'][hi1] == 1:
                hitsign = 0
            else:
                # Find time indices just before hit
                t1i = np.where(ball[b1i]['t'] < hit[b1i]['t'][hi1])[0][-1] if np.any(ball[b1i]['t'] < hit[b1i]['t'][hi1]) else -1
                t3i = np.where(ball[b3i]['t'] < hit[b1i]['t'][hi1])[0][-1] if np.any(ball[b3i]['t'] < hit[b1i]['t'][hi1]) else -1
                
                if t1i >= 0 and t3i >= 0:
                    # Calculate vectors
                    v1 = np.array([
                        hit[b1i]['XPos'][hi1] - ball[b1i]['x'][t1i],
                        hit[b1i]['YPos'][hi1] - ball[b1i]['y'][t1i],
                        0
                    ])
                    
                    v2 = np.array([
                        hit[b3i]['XPos'][hi3] - hit[b1i]['XPos'][hi1],
                        hit[b3i]['YPos'][hi3] - hit[b1i]['YPos'][hi1],
                        0
                    ])
                    
                    v3 = np.cross(v2, v1)
                    hitsign = 1 if v3[2] > 0 else -1
                else:
                    hitsign = 0
            
            hit[b1i]['PointDist'] = hitsign * (1 - hit[b1i]['Fraction'][hi1]) * param['ballR'] * 2
        else:
            hit[b1i]['PointDist'] = 3000
    else:
        # No point - find closest approach after B2 hit and 3rd cushion if data exists
        if len(Tall) > 0 and 1 in CC and len(CC[1]['dist']) > 0 and 'Tready' in hit[b1i]:
            ind = np.where(Tall > hit[b1i]['Tready'])[0] if np.any(Tall > hit[b1i]['Tready']) else np.array([])
            
            if len(ind) > 0:
                PointDist = np.min(CC[1]['dist'][ind])
                imin = np.argmin(CC[1]['dist'][ind])
                
                t1i = np.where(ball[b1i]['t'] <= Tall[ind[imin]])[0][-1] if np.any(ball[b1i]['t'] <= Tall[ind[imin]]) else -1
                t3i = np.where(ball[b3i]['t'] <= Tall[ind[imin]])[0][-1] if np.any(ball[b3i]['t'] <= Tall[ind[imin]]) else -1
                
                if t1i >= 0 and t3i >= 0 and t1i > 0:
                    # Calculate vectors
                    v1 = np.array([
                        ball[b1i]['x'][t1i] - ball[b1i]['x'][t1i-1],
                        ball[b1i]['y'][t1i] - ball[b1i]['y'][t1i-1],
                        0
                    ])
                    
                    v2 = np.array([
                        ball[b3i]['x'][t3i] - ball[b1i]['x'][t1i],
                        ball[b3i]['y'][t3i] - ball[b1i]['y'][t1i],
                        0
                    ])
                    
                    v3 = np.cross(v2, v1)
                    hitsign = 1 if v3[2] > 0 else -1
                    
                    hit[b1i]['PointDist'] = hitsign * PointDist
                else:
                    hit[b1i]['PointDist'] = 3000
            else:
                hit[b1i]['PointDist'] = 3000
        else:
            hit[b1i]['PointDist'] = 3000
    
    return hit