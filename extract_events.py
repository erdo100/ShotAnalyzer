import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from scipy.interpolate import interp1d
from str2num_B1B2B3 import str2num_B1B2B3


def extract_events(si: int, SA: Dict[str, Any], param: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract all ball-ball hit events and ball-cushion hit events
    
    Args:
        si: Shot index
        SA: Dictionary containing shot data with nested Route structure
        param: Dictionary containing parameters
        
    Returns:
        Tuple: (hit dictionary, error dictionary)
    """
    # Initialize outputs
    err = {'code': None, 'text': ''}
    col = 'WYR'  # Ball colors: White=0, Yellow=1, Red=2
    
    # Get shot information
    shot_id = SA['Table'].loc[si, 'ShotID']
    b1b2b3_str = SA['Table'].loc[si, 'B1B2B3']
    b1b2b3, b1i, b2i, b3i = str2num_B1B2B3(b1b2b3_str)
    
    # Initialize ball data structures
    ball0 = {}
    ball = {}
    hit = {}
    
    # Load original route data from nested structure
    for bi in [0, 1, 2]:  # Python uses 0-based indexing
        if shot_id not in SA['Route'] or bi not in SA['Route'][shot_id]:
            err['code'] = 3
            err['text'] = f'Missing data for ball {bi} in shot {shot_id}'
            return {}, err
            
        ball_df = SA['Route'][shot_id][bi]
        ball0[bi] = {
            't': ball_df['t'].values,
            'x': ball_df['x'].values,
            'y': ball_df['y'].values
        }
        
        # Initialize processed ball data
        ball[bi] = {
            't': [ball0[bi]['t'][0]],
            'x': [ball0[bi]['x'][0]],
            'y': [ball0[bi]['y'][0]]
        }
        
        # Initialize hit tracking
        hit[bi] = {
            'with': ['S' if bi == b1i else '-'],
            't': [0],
            'XPos': [ball0[bi]['x'][0]],
            'YPos': [ball0[bi]['y'][0]],
            'Kiss': 0,
            'Point': 0,
            'Fuchs': 0,
            'PointDist': 3000,
            'KissDistB1': 3000,
            'Tpoint': 3000,
            'Tkiss': 3000,
            'Tready': 3000,
            'TB2hit': 3000,
            'Tfailure': 3000
        }
    
    # Create common time points
    Tall0 = np.unique(np.concatenate([ball0[bi]['t'] for bi in [0, 1, 2]]))
    ti = 0
    
    # Time vector for approximation
    tvec = np.linspace(0.01, 1, 101)
    
    # Main processing loop
    do_scan = True
    while do_scan and len(Tall0) >= 3:
        ti += 1
        dT = np.diff(Tall0[:2])[0]
        tappr = Tall0[0] + dT * tvec
        
        # Initialize ball approximations
        b = {}
        for bi in [0, 1, 2]:
            b[bi] = {
                'xa': np.zeros_like(tappr),
                'ya': np.zeros_like(tappr),
                'cd': np.zeros((len(tappr), 4))  # Cushion distances
            }
            
            # Calculate velocities
            if len(ball0[bi]['t']) >= 2:
                dt = np.diff(ball0[bi]['t'])
                vx = np.diff(ball0[bi]['x']) / dt
                vy = np.diff(ball0[bi]['y']) / dt
                v = np.sqrt(vx**2 + vy**2)
                
                ball0[bi]['dt'] = dt
                ball0[bi]['vx'] = np.append(vx, 0)
                ball0[bi]['vy'] = np.append(vy, 0)
                ball0[bi]['v'] = np.append(v, 0)
                
                # Set initial ball speeds for B2 and B3 = 0
                if bi in [b2i, b3i]:
                    ball0[bi]['vx'][0] = 0
                    ball0[bi]['vy'][0] = 0
                    ball0[bi]['v'][0] = 0
                
                # Approximate positions
                if len(ball[bi]['x']) >= 2:
                    ds0 = np.sqrt((ball[bi]['x'][-1]-ball[bi]['x'][-2])**2 + 
                                 (ball[bi]['y'][-1]-ball[bi]['y'][-2])**2)
                    v0 = ds0 / dT
                
                # Velocity components
                if hit[bi]['with'][-1] == '-':
                    b[bi]['v1'] = np.array([0, 0])
                    b[bi]['v2'] = np.array([ball0[bi]['vx'][1], ball0[bi]['vy'][1]])
                else:
                    b[bi]['v1'] = np.array([ball0[bi]['vx'][0], ball0[bi]['vy'][0]])
                    b[bi]['v2'] = np.array([ball0[bi]['vx'][1], ball0[bi]['vy'][1]])
                
                # Calculate approximate trajectories
                vtnext = max([np.linalg.norm(b[bi]['v1']), ball0[bi]['v'][0]])
                if np.linalg.norm(b[bi]['v1']) > 0:
                    vnext = b[bi]['v1'] / np.linalg.norm(b[bi]['v1']) * vtnext
                else:
                    vnext = np.array([0, 0])
                
                b[bi]['xa'] = ball[bi]['x'][-1] + vnext[0] * dT * tvec
                b[bi]['ya'] = ball[bi]['y'][-1] + vnext[1] * dT * tvec
        
        # Calculate ball-ball distances
        BB = [(0, 1), (0, 2), (1, 2)]  # Ball pairs
        d = {}
        for bbi, (bx1, bx2) in enumerate(BB):
            d[bbi] = {
                'BB': np.sqrt((b[bx1]['xa'] - b[bx2]['xa'])**2 + 
                       (b[bx1]['ya'] - b[bx2]['ya'])**2) - 2 * param['ballR']
            }
        
        # Calculate cushion distances
        for bi in [0, 1, 2]:
            b[bi]['cd'][:, 0] = b[bi]['ya'] - param['ballR']  # Bottom cushion
            b[bi]['cd'][:, 1] = param['size'][1] - param['ballR'] - b[bi]['xa']  # Right cushion
            b[bi]['cd'][:, 2] = param['size'][0] - param['ballR'] - b[bi]['ya']  # Top cushion
            b[bi]['cd'][:, 3] = b[bi]['xa'] - param['ballR']  # Left cushion
        
        # Detect hits
        hitlist = []
        
        # Check cushion hits
        for bi in [0, 1, 2]:
            for cii in range(4):  # 4 cushions
                if np.any(b[bi]['cd'][:, cii] <= 0):
                    # Find exact contact time
                    f = interp1d(b[bi]['cd'][:, cii], tappr, kind='linear', fill_value='extrap')
                    tc = f(0)
                    
                    # Get contact position
                    fx = interp1d(tappr, b[bi]['xa'], kind='linear', fill_value='extrap')
                    fy = interp1d(tappr, b[bi]['ya'], kind='linear', fill_value='extrap')
                    
                    if cii == 0:  # Bottom cushion
                        cushx = fx(tc)
                        cushy = param['ballR']
                    elif cii == 1:  # Right cushion
                        cushx = param['size'][1] - param['ballR']
                        cushy = fy(tc)
                    elif cii == 2:  # Top cushion
                        cushx = fx(tc)
                        cushy = param['size'][0] - param['ballR']
                    else:  # Left cushion
                        cushx = param['ballR']
                        cushy = fy(tc)
                    
                    hitlist.append({
                        'time': tc,
                        'ball': bi,
                        'type': 2,  # Cushion
                        'partner': cii,
                        'x': cushx,
                        'y': cushy
                    })
        
        # Check ball-ball hits
        for bbi, (bx1, bx2) in enumerate(BB):
            if np.any(d[bbi]['BB'] <= 0) and np.any(d[bbi]['BB'] > 0):
                # Find exact contact time
                f = interp1d(d[bbi]['BB'], tappr, kind='linear', fill_value='extrap')
                tc = f(0)
                
                # Get contact positions
                fx1 = interp1d(tappr, b[bx1]['xa'], kind='linear', fill_value='extrap')
                fy1 = interp1d(tappr, b[bx1]['ya'], kind='linear', fill_value='extrap')
                fx2 = interp1d(tappr, b[bx2]['xa'], kind='linear', fill_value='extrap')
                fy2 = interp1d(tappr, b[bx2]['ya'], kind='linear', fill_value='extrap')
                
                hitlist.append({
                    'time': tc,
                    'ball': bx1,
                    'type': 1,  # Ball-ball
                    'partner': bx2,
                    'x': fx1(tc),
                    'y': fy1(tc)
                })
                
                hitlist.append({
                    'time': tc,
                    'ball': bx2,
                    'type': 1,  # Ball-ball
                    'partner': bx1,
                    'x': fx2(tc),
                    'y': fy2(tc)
                })
        
        # Process hits
        if hitlist:
            # Find earliest hit time
            tc = min(h['time'] for h in hitlist)
            
            for h in hitlist:
                if h['time'] == tc:
                    bi = h['ball']
                    
                    # Update ball position
                    ball[bi]['t'].append(tc)
                    ball[bi]['x'].append(h['x'])
                    ball[bi]['y'].append(h['y'])
                    
                    # Update hit tracking
                    hit[bi]['t'].append(tc)
                    if h['type'] == 1:  # Ball-ball
                        hit[bi]['with'].append(col[h['partner']])
                    else:  # Cushion
                        hit[bi]['with'].append(str(h['partner']))
                    hit[bi]['XPos'].append(h['x'])
                    hit[bi]['YPos'].append(h['y'])
        
        # Update time points
        Tall0 = Tall0[1:]
        
        # Check termination condition
        do_scan = len(Tall0) >= 3
    
    # Update velocities for plotting
    for bi in [0, 1, 2]:
        if len(ball[bi]['t']) > 1:
            dt = np.diff(ball[bi]['t'])
            vx = np.diff(ball[bi]['x']) / dt
            vy = np.diff(ball[bi]['y']) / dt
            v = np.sqrt(vx**2 + vy**2)
            
            ball[bi]['vx'] = np.append(vx, 0)
            ball[bi]['vy'] = np.append(vy, 0)
            ball[bi]['v'] = np.append(v, 0)
    
    return hit, err