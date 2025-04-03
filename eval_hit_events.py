import numpy as np
from typing import Dict, Any
from ball_velocity import ball_velocity
from ball_direction import ball_direction
from cushion_angle import cushion_angle

def eval_hit_events(si: int, hit: Dict[int, Dict[str, Any]], b1b2b3: list, SA: Dict[str, Any], param: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Evaluate hit events and calculate various metrics
    
    Args:
        si: Shot index
        hit: Dictionary containing hit data for each ball
        b1b2b3: List of ball indices [b1, b2, b3]
        SA: Dictionary containing shot data with nested Route structure
        param: Dictionary containing parameters
        
    Returns:
        Dictionary with evaluated hit data
    """
    # Initialize output variables
    b1i, b2i, b3i = b1b2b3
    shot_id = SA['Table'].loc[si, 'ShotID']
    
    # Initialize additional fields
    for bi in [0, 1, 2]:  # Using 0-based indexing
        hit[bi]['AimOffsetLL'] = np.nan
        hit[bi]['AimOffsetSS'] = np.nan
        hit[bi]['B1B2OffsetLL'] = np.nan
        hit[bi]['B1B2OffsetSS'] = np.nan
        hit[bi]['B1B3OffsetLL'] = np.nan
        hit[bi]['B1B3OffsetSS'] = np.nan
        
        for hi in range(8):  # Initialize arrays for up to 8 hits
            hit[bi]['Type'] = hit[bi].get('Type', [np.nan]*8)
            hit[bi]['FromC'] = hit[bi].get('FromC', [np.nan]*8)
            hit[bi]['ToC'] = hit[bi].get('ToC', [np.nan]*8)
            hit[bi]['V1'] = hit[bi].get('V1', [np.nan]*8)
            hit[bi]['V2'] = hit[bi].get('V2', [np.nan]*8)
            hit[bi]['Offset'] = hit[bi].get('Offset', [np.nan]*8)
            hit[bi]['Fraction'] = hit[bi].get('Fraction', [np.nan]*8)
            hit[bi]['DefAngle'] = hit[bi].get('DefAngle', [np.nan]*8)
            hit[bi]['CutAngle'] = hit[bi].get('CutAngle', [np.nan]*8)
            hit[bi]['CInAngle'] = hit[bi].get('CInAngle', [np.nan]*8)
            hit[bi]['COutAngle'] = hit[bi].get('COutAngle', [np.nan]*8)
            hit[bi]['FromCPos'] = hit[bi].get('FromCPos', [np.nan]*8)
            hit[bi]['ToCPos'] = hit[bi].get('ToCPos', [np.nan]*8)
            hit[bi]['FromDPos'] = hit[bi].get('FromDPos', [np.nan]*8)
            hit[bi]['ToDPos'] = hit[bi].get('ToDPos', [np.nan]*8)

    # Evaluate all hits for each ball
    for bi in b1b2b3:
        # Skip if ball data is missing
        if shot_id not in SA['Route'] or bi not in SA['Route'][shot_id]:
            continue
            
        # Go through all hits of this ball
        for hi in range(len(hit[bi]['with'])):
            if hit[bi]['with'][hi] == '-':
                continue  # Skip first entries with '-'
            
            # Get ball velocity and direction using nested route data
            v, v1, v2, alpha, offset = ball_velocity(SA['Route'][shot_id][bi], hit[bi], hi)
            
            # Store velocity data
            hit[bi]['V1'][hi] = v[0]
            hit[bi]['V2'][hi] = v[1]
            hit[bi]['Offset'][hi] = offset
            
            if v[0] > 0 or v[1] > 0:
                direction = ball_direction(SA['Route'][shot_id][bi], hit[bi], hi, param)
            else:
                direction = np.full((2, 6), np.nan)
            
            # Store direction data
            hit[bi]['FromC'][hi] = direction[1, 0]
            hit[bi]['FromCPos'][hi] = direction[1, 1]
            hit[bi]['FromDPos'][hi] = direction[1, 2]
            hit[bi]['ToC'][hi] = direction[1, 3]
            hit[bi]['ToCPos'][hi] = direction[1, 4]
            hit[bi]['ToDPos'][hi] = direction[1, 5]
            
            hit[bi]['Type'][hi] = 0  # 0=Start, 1=Ball, 2=Cushion
            
            # Check for cushion hits
            c2i = next((i for i, c in enumerate(hit[bi]['with'][hi]) if c in '1234'), None)
            if c2i is not None:
                hit[bi]['Type'][hi] = 2  # Cushion hit
                angle = cushion_angle(int(hit[bi]['with'][hi][c2i]), v1, v2)
                hit[bi]['CInAngle'][hi] = angle[0]
                hit[bi]['COutAngle'][hi] = angle[1]
            
            # Check for ball-ball hits
            b2i_char = next((c for c in hit[bi]['with'][hi] if c in 'WYR'), None)
            if b2i_char is not None:
                hit[bi]['Type'][hi] = 1  # Ball hit
                b2i = {'W': 0, 'Y': 1, 'R': 2}[b2i_char]  # Convert to ball index
                
                if v[0] > 0 and shot_id in SA['Route'] and b2i in SA['Route'][shot_id]:
                    # Find matching time indices
                    ball_df = SA['Route'][shot_id][bi]
                    partner_df = SA['Route'][shot_id][b2i]
                    
                    tb10i = np.where(ball_df['t'] == hit[bi]['t'][hi])[0][0]
                    tb11i = np.where(partner_df['t'] == hit[bi]['t'][hi])[0][0]
                    
                    pb1 = np.array([ball_df['x'].iloc[tb10i], 
                                   ball_df['y'].iloc[tb10i], 0])
                    pb2 = np.array([partner_df['x'].iloc[tb11i], 
                                   partner_df['y'].iloc[tb11i], 0])
                    
                    # Find matching hit in partner ball
                    hib2 = np.where(hit[b2i]['t'] == hit[bi]['t'][hi])[0][0]
                    _, v1b2, _, _, _ = ball_velocity(partner_df, hit[b2i], hib2)
                    
                    # Calculate fraction and deflection angle
                    velrel = np.array([v1[0]-v1b2[0], v1[1]-v1b2[1], 0])
                    try:
                        hit[bi]['Fraction'][hi] = 1 - np.linalg.norm(np.cross(pb2-pb1, velrel)) / \
                            np.linalg.norm(velrel) / param['ballR'] / 2
                        hit[bi]['DefAngle'][hi] = np.degrees(np.arccos(np.dot(v1, v2) / \
                            (np.linalg.norm(v1) * np.linalg.norm(v2))))
                    except:
                        hit[bi]['Fraction'][hi] = 0
                        hit[bi]['DefAngle'][hi] = 0
                    
                    # Calculate cut angle
                    _, _, b2v2, _, _ = ball_velocity(partner_df, hit[b2i], 2)
                    hit[bi]['CutAngle'][hi] = np.degrees(np.arccos(np.dot(v1, b2v2) / \
                        (np.linalg.norm(v1) * np.linalg.norm(b2v2))))

    # Calculate shot offset angles
    if len(hit[b1i]['YPos']) >= 2:
        if abs(hit[b1i]['YPos'][0] - hit[b1i]['YPos'][1]) > np.finfo(float).eps:
            hit[b1i]['AimOffsetLL'] = abs((hit[b1i]['XPos'][1] - hit[b1i]['XPos'][0]) / \
                (hit[b1i]['YPos'][1] - hit[b1i]['YPos'][0]) * \
                (param['size'][0] + 2 * param['diamdist'])) * 4 / param['size'][0]
    
    if len(hit[b1i]['XPos']) >= 2:
        if abs(hit[b1i]['XPos'][0] - hit[b1i]['XPos'][1]) > np.finfo(float).eps:
            hit[b1i]['AimOffsetSS'] = abs((hit[b1i]['YPos'][1] - hit[b1i]['YPos'][0]) / \
                (hit[b1i]['XPos'][1] - hit[b1i]['XPos'][0]) * \
                (param['size'][1] + 2 * param['diamdist'])) * 4 / param['size'][0]
    
    # Apply direction signs to offsets
    if len(hit[b1i]['with']) >= 2:
        if hit[b1i]['with'][1] in 'WYR':
            directL = np.sign(hit[b2i]['XPos'][0] - hit[b1i]['XPos'][1]) * \
                     np.sign(hit[b1i]['XPos'][0] - hit[b1i]['XPos'][1])
            directS = np.sign(hit[b2i]['YPos'][0] - hit[b1i]['YPos'][1]) * \
                     np.sign(hit[b1i]['YPos'][0] - hit[b1i]['YPos'][1])
            hit[b1i]['AimOffsetLL'] *= directL
            hit[b1i]['AimOffsetSS'] *= directS
        
        elif hit[b1i]['with'][1] in '1234':
            directS = np.sign((hit[b1i]['YPos'][1] - hit[b1i]['YPos'][0]) - \
                            (hit[b1i]['YPos'][1] - hit[b1i]['YPos'][2]))
            directL = np.sign((hit[b1i]['XPos'][1] - hit[b1i]['XPos'][0]) - \
                            (hit[b1i]['XPos'][1] - hit[b1i]['XPos'][2]))
            hit[b1i]['AimOffsetLL'] *= directL
            hit[b1i]['AimOffsetSS'] *= directS
    
    # Calculate position offsets
    if abs(hit[b2i]['YPos'][0] - hit[b1i]['YPos'][0]) > np.finfo(float).eps:
        hit[b1i]['B1B2OffsetLL'] = abs((hit[b2i]['XPos'][0] - hit[b1i]['XPos'][0]) / \
            (hit[b2i]['YPos'][0] - hit[b1i]['YPos'][0]) * \
            (param['size'][0] + 2 * param['diamdist'])) * 4 / param['size'][0]
    else:
        hit[b1i]['B1B2OffsetLL'] = 99
    
    if abs(hit[b2i]['XPos'][0] - hit[b1i]['XPos'][0]) > np.finfo(float).eps:
        hit[b1i]['B1B2OffsetSS'] = abs((hit[b2i]['YPos'][0] - hit[b1i]['YPos'][0]) / \
            (hit[b2i]['XPos'][0] - hit[b1i]['XPos'][0]) * \
            (param['size'][1] + 2 * param['diamdist'])) * 4 / param['size'][0]
    else:
        hit[b1i]['B1B2OffsetSS'] = 99
    
    if abs(hit[b3i]['YPos'][0] - hit[b1i]['YPos'][0]) > np.finfo(float).eps:
        hit[b1i]['B1B3OffsetLL'] = abs((hit[b3i]['XPos'][0] - hit[b1i]['XPos'][0]) / \
            (hit[b3i]['YPos'][0] - hit[b1i]['YPos'][0]) * \
            (param['size'][0] + 2 * param['diamdist'])) * 4 / param['size'][0]
    else:
        hit[b1i]['B1B3OffsetLL'] = 99
    
    if abs(hit[b3i]['XPos'][0] - hit[b1i]['XPos'][0]) > np.finfo(float).eps:
        hit[b1i]['B1B3OffsetSS'] = abs((hit[b3i]['YPos'][0] - hit[b1i]['YPos'][0]) / \
            (hit[b3i]['XPos'][0] - hit[b1i]['XPos'][0]) * \
            (param['size'][1] + 2 * param['diamdist'])) * 4 / param['size'][0]
    else:
        hit[b1i]['B1B3OffsetSS'] = 99
    
    # Calculate inside/outside shot classification
    if len(hit[b1i]['Type']) >= 2:
        if hit[b1i]['Type'][1] == 2:  # Cushion first
            hit[b1i]['with'][0] = 'B'
        elif len(hit[b1i]['t']) >= 4 and hit[b1i]['Type'][1] == 1 and hit[b1i]['Type'][2] == 2:
            P1 = np.array([hit[b1i]['XPos'][0], hit[b1i]['YPos'][0], 0])
            P2 = np.array([hit[b1i]['XPos'][1], hit[b1i]['YPos'][1], 0])
            P3 = np.array([hit[b1i]['XPos'][2], hit[b1i]['YPos'][2], 0])
            P4 = np.array([hit[b1i]['XPos'][3], hit[b1i]['YPos'][3], 0])
            
            v1 = P1 - P3
            v2 = P2 - P3
            v3 = P4 - P3
            
            a1 = np.degrees(np.arccos(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))))
            a2 = np.degrees(np.arccos(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))))
            
            hit[b1i]['with'][0] = 'I' if a1 <= a2 else 'E'
    
    return hit