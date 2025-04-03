import numpy as np
import pandas as pd
from str2num_b1b2b3 import str2num_b1b2b3
from angle_vector import angle_vector


def process_hits(hitlist, ball, ball0, hit, Tall0, tappr, b):
    if not hitlist:
        return
        
    tc = min(hit['time'] for hit in hitlist)
    processed_balls = set()
    
    for hit_event in hitlist:
        if hit_event['time'] == tc:
            bi = hit_event['ball1']
            processed_balls.add(bi)
            
            # Update positions
            ball[bi]['t'].append(tc)
            ball[bi]['x'].append(hit_event['contact_x'])
            ball[bi]['y'].append(hit_event['contact_y'])
            
            # Update hit tracking
            hit[bi]['t'].append(tc)
            hit[bi]['with'].append(str(hit_event['type']))
            hit[bi]['XPos'].append(hit_event['contact_x'])
            hit[bi]['YPos'].append(hit_event['contact_y'])
    
    # Update non-hit balls
    for bi in set(range(3)) - processed_balls:
        ball[bi]['t'].append(Tall0[0])
        ball[bi]['x'].append(np.interp(Tall0[0], ball0[bi]['t'], ball0[bi]['x']))
        ball[bi]['y'].append(np.interp(Tall0[0], ball0[bi]['t'], ball0[bi]['y']))

def update_ball_derivatives(ball_data):
    sorted_idx = np.argsort(ball_data['t'])
    ball_data['t'] = np.array(ball_data['t'])[sorted_idx]
    ball_data['x'] = np.array(ball_data['x'])[sorted_idx]
    ball_data['y'] = np.array(ball_data['y'])[sorted_idx]
    
    ball_data['dt'] = np.diff(ball_data['t'], append=ball_data['t'][-1])
    ball_data['vx'] = np.diff(ball_data['x'], append=0) / ball_data['dt']
    ball_data['vy'] = np.diff(ball_data['y'], append=0) / ball_data['dt']
    ball_data['v'] = np.sqrt(ball_data['vx']**2 + ball_data['vy']**2)

# Helper functions
def calculate_ball_kinematics(bi, ball, ball0, hit, tappr, dT):
    b_data = {}
    
    if len(ball0[bi]['t']) >= 2:
        # Current speed
        b_data['vt1'] = ball0[bi]['v'][0]
        
        # Determine velocity components
        if hit[bi]['with'] == ['-']:
            b_data['v1'] = [0, 0]
            b_data['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
        else:
            iprev = next((i for i, t in enumerate(ball[bi]['t']) if t >= hit[bi]['t'][-1]), None)
            
            if iprev is not None:
                b_data['v1'] = [ball0[bi]['vx'][0], ball0[bi]['vy'][0]]
                b_data['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
            else:
                CoefVX = np.polyfit(ball[bi]['t'][-2:], ball[bi]['x'][-2:], 1)
                CoefVY = np.polyfit(ball[bi]['t'][-2:], ball[bi]['y'][-2:], 1)
                b_data['v1'] = [CoefVX[0], CoefVY[0]]
                b_data['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
        
        # Calculate trajectory
        b_data['vt1'] = np.linalg.norm(b_data['v1'])
        b_data['vt2'] = np.linalg.norm(b_data['v2'])
        vtnext = max(b_data['vt1'], ball0[bi]['v'][0])
        
        if np.linalg.norm(b_data['v1']) > 0:
            vnext = (b_data['v1'] / np.linalg.norm(b_data['v1'])) * vtnext
        else:
            vnext = [0, 0]
        
        b_data['xa'] = ball[bi]['x'][-1] + vnext[0] * dT * tappr
        b_data['ya'] = ball[bi]['y'][-1] + vnext[1] * dT * tappr
        
        # Calculate angle change
        b_data['a12'] = angle_vector(b_data['v1'], b_data['v2'])
    
    return b_data


def detect_collisions(b, b1i, b2i, b3i, tappr, param, ti, ball0):
    hitlist = []
    BB = [(b1i, b2i), (b1i, b3i), (b2i, b3i)]
    
    # Ball-Ball collisions
    for bx1, bx2 in BB:
        if bx1 in b and bx2 in b and 'xa' in b[bx1] and 'xa' in b[bx2]:
            d_BB = np.sqrt((b[bx1]['xa'] - b[bx2]['xa'])**2 + 
                          (b[bx1]['ya'] - b[bx2]['ya'])**2) - 2 * param['ballR']
            
            if (np.any(d_BB <= 0) and np.any(d_BB > 0)):
                tc = np.interp(0, d_BB, tappr)
                hitlist.append({
                    'time': tc,
                    'ball1': bx1,
                    'type': 1,
                    'ball2': bx2,
                    'contact_x': np.interp(tc, tappr, b[bx1]['xa']),
                    'contact_y': np.interp(tc, tappr, b[bx1]['ya'])
                })
    
    # Cushion collisions
    for bi in b:
        if 'xa' not in b[bi]:
            continue
            
        cd = [
            b[bi]['ya'] - param['ballR'],
            param['size'][1] - param['ballR'] - b[bi]['xa'],
            param['size'][0] - param['ballR'] - b[bi]['ya'],
            b[bi]['xa'] - param['ballR']
        ]
        
        for cii in range(4):
            if np.any(cd[cii] <= 0) and b[bi].get('a12', 0) > 1:
                tc = np.interp(0, cd[cii], tappr)
                hitlist.append({
                    'time': tc,
                    'ball1': bi,
                    'type': 2,
                    'ball2': cii,
                    'contact_x': np.interp(tc, tappr, b[bi]['xa']),
                    'contact_y': np.interp(tc, tappr, b[bi]['ya'])
                })
    
    return hitlist


def extract_events(SA, si, param):
    err = {'code': None, 'text': ''}
    
    # Initialize data structures
    b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(SA.iloc[si]['B1B2B3'])
    ball_indices = {'b1': b1i, 'b2': b2i, 'b3': b3i}
    
    # Load route data
    routes = {bi: SA.iloc[si][f'Route{bi}'] for bi in range(3)}
    
    # Initialize ball tracking structures
    ball0 = {
        bi: {
            't': routes[bi]['t'].copy(),
            'x': routes[bi]['x'].copy(),
            'y': routes[bi]['y'].copy(),
            'dt': None,
            'vx': None,
            'vy': None,
            'v': None
        }
        for bi in range(3)
    }
    
    ball = {
        bi: {
            'x': [routes[bi]['x'].iloc[0]],
            'y': [routes[bi]['y'].iloc[0]],
            't': [0]
        }
        for bi in range(3)
    }
    
    # Initialize hit tracking
    hit = {
        bi: {
            'with': ['S' if bi == b1i else '-'],
            't': [0],
            'XPos': [routes[bi]['x'].iloc[0]],
            'YPos': [routes[bi]['y'].iloc[0]],
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
        for bi in range(3)
    }
    
    # Calculate velocities
    for bi in range(3):
        ball0[bi]['dt'] = np.diff(ball0[bi]['t'], append=ball0[bi]['t'].iloc[-1])
        ball0[bi]['vx'] = np.diff(ball0[bi]['x'], append=0) / ball0[bi]['dt']
        ball0[bi]['vy'] = np.diff(ball0[bi]['y'], append=0) / ball0[bi]['dt']
        ball0[bi]['v'] = np.sqrt(ball0[bi]['vx']**2 + ball0[bi]['vy']**2)
    
    # Set initial velocities for stationary balls
    for bi in [b2i, b3i]:
        ball0[bi]['vx'][0] = ball0[bi]['vy'][0] = ball0[bi]['v'][0] = 0
    
    # Main processing loop
    Tall0 = np.unique(np.concatenate([ball0[bi]['t'] for bi in range(3)]))
    tvec = np.linspace(0.01, 1, 101)
    ti = 0
    
    while len(Tall0) >= 3:
        ti += 1
        dT = np.diff(Tall0[:2])[0]
        tappr = Tall0[0] + dT * tvec
        
        # Calculate ball trajectories and velocities
        b = {}
        for bi in range(3):
            b[bi] = calculate_ball_kinematics(bi, ball, ball0, hit, tappr, dT)
        
        # Detect collisions
        hitlist = detect_collisions(b, b1i, b2i, b3i, tappr, param, ti, ball0)
        
        # Process hits and update ball positions
        process_hits(hitlist, ball, ball0, hit, Tall0, tappr, b)
        
        # Update derivatives
        for bi in range(3):
            if hit[bi]['with'][-1] != '-':
                update_ball_derivatives(ball0[bi])
        
        # Prepare for next iteration
        Tall0 = Tall0[1:]
    
    # Update the shot data with processed routes
    for bi in range(3):
        SA.at[si, f'Route{bi}'] = pd.DataFrame({
            't': ball[bi]['t'],
            'x': ball[bi]['x'],
            'y': ball[bi]['y']
        })
    
    return hit, err

