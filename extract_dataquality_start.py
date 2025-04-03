import pandas as pd
import numpy as np
from math import sqrt
from typing import Dict, Any

def extract_dataquality_start(SA: Dict[str, Any], param: Dict[str, Any]):
    print(f"start (extract_data_quality_start)")

    # Initialize error tracking
    err = {'code': None, 'text': '', 'region': {1: {'ti': []}, 2: {'ti': []}, 3: {'ti': []}}}

    # Process each shot in the DataFrame
    for si, row in SA['Table'].iterrows():
        if row['Interpreted'] == 0:
            shot_id = row['ShotID']
            
            # Reset to original route data (copy from Route0 to Route)
            for ball_id in [0, 1, 2]:
                if shot_id in SA['Route0'] and ball_id in SA['Route0'][shot_id]:
                    SA['Route'][shot_id][ball_id] = SA['Route0'][shot_id][ball_id].copy()

            # Initialize error for this shot
            errcode = 100
            err['code'] = None
            err['text'] = ''
            SA['Table'].at[si, 'Selected'] = False

            # Check ball data exists
            for bi in range(3):
                if shot_id not in SA['Route'] or bi not in SA['Route'][shot_id]:
                    err['code'] = errcode + 1
                    err['text'] = f'Ball data is missing 1 (extract_data_quality_start)'
                    print(f"{si}: Ball data is missing 1 (extract_data_quality_start)")
                    SA['Table'].at[si, 'Selected'] = True
                    break
                elif len(SA['Route'][shot_id][bi]) == 1:
                    err['code'] = errcode + 2
                    err['text'] = f'Ball data is missing 2 (extract_data_quality_start)'
                    print(f"{si}: Ball data is missing 2 (extract_data_quality_start)")
                    SA['Table'].at[si, 'Selected'] = True
                    break

            # Project points out of cushion onto cushion
            if err['code'] is None:
                tol = param['BallProjecttoCushionLimit']
                cushion = [
                    param['ballR'] - tol,
                    param['size'][1] - param['ballR'] + tol,
                    param['size'][0] - param['ballR'] + tol,
                    param['ballR'] - tol
                ]
                oncushion = [
                    param['ballR'] + 0.1,
                    param['size'][1] - param['ballR'] - 0.1,
                    param['size'][0] - param['ballR'] - 0.1,
                    param['ballR'] + 0.1
                ]

                for bi in range(3):
                    ball_df = SA['Route'][shot_id][bi]
                    # Project x coordinates
                    ball_df.loc[ball_df['x'] > oncushion[1], 'x'] = oncushion[1]
                    ball_df.loc[ball_df['x'] < oncushion[3], 'x'] = oncushion[3]
                    # Project y coordinates
                    ball_df.loc[ball_df['y'] < oncushion[0], 'y'] = oncushion[0]
                    ball_df.loc[ball_df['y'] > oncushion[2], 'y'] = oncushion[2]

            # Check initial ball distance
            if err['code'] is None:
                errcode += 1
                BB = [(0, 1), (0, 2), (1, 2)]
                
                for b1i, b2i in BB:
                    ball1 = SA['Route'][shot_id][b1i].iloc[0]
                    ball2 = SA['Route'][shot_id][b2i].iloc[0]
                    
                    distance = sqrt((ball1['x'] - ball2['x'])**2 + 
                                (ball1['y'] - ball2['y'])**2) - 2 * param['ballR']
                    
                    if distance < 0:
                        # Move balls apart along their center line
                        vec = np.array([ball2['x'] - ball1['x'], 
                                       ball2['y'] - ball1['y']])
                        vec = -vec / np.linalg.norm(vec) * distance
                        
                        # Update positions
                        SA['Route'][shot_id][b1i].loc[SA['Route'][shot_id][b1i].index[0], 'x'] -= vec[0] / 2
                        SA['Route'][shot_id][b1i].loc[SA['Route'][shot_id][b1i].index[0], 'y'] -= vec[1] / 2
                        SA['Route'][shot_id][b2i].loc[SA['Route'][shot_id][b2i].index[0], 'x'] += vec[0] / 2
                        SA['Route'][shot_id][b2i].loc[SA['Route'][shot_id][b2i].index[0], 'y'] += vec[1] / 2

            # Check time linearity
            if err['code'] is None:
                errcode += 1
                for bi in range(3):
                    ball_df = SA['Route'][shot_id][bi].sort_values('t')
                    dt = ball_df['t'].diff().dropna()
                    
                    # Find non-increasing time points
                    ind = dt[dt <= 0].index
                    delind = []
                    
                    if not ind.empty:
                        print(f"{si}: Try to fix time linearity...")
                        t = ball_df['t'].values
                        
                        for idx in ind:
                            delind_new = np.where(t[idx+1:] <= t[idx])[0] + idx
                            delind.extend(delind_new)
                        
                        if delind:
                            # Remove problematic indices
                            SA['Route'][shot_id][bi] = ball_df.drop(ball_df.index[delind])
                            
                            # Check again
                            ball_df = SA['Route'][shot_id][bi].sort_values('t')
                            dt = ball_df['t'].diff().dropna()
                            ind_new = dt[dt <= 0].index
                            
                            if ind_new.empty:
                                print(f"{si}: Successfully fixed time linearity :)")
                            else:
                                err['code'] = errcode
                                err['text'] = f'no time linearity (extract_data_quality_start)'
                                print(f"{si}: no time linearity (extract_data_quality_start)")
                                err['region'][bi]['ti'] = ind_new.tolist()
                                SA['Table'].at[si, 'Selected'] = True

            # Check time synchronization across balls
            if err['code'] is None:
                errcode += 1
                ball_times = []
                for bi in range(3):
                    ball_times.append(SA['Route'][shot_id][bi]['t'].values)
                
                if not all(np.allclose(times[0], ball_times[0][0]) for times in ball_times):
                    err['code'] = errcode
                    err['text'] = 'time doesnt start from 0 (extract_data_quality_start)'
                    print(f"{si}: time doesnt start from 0 (extract_data_quality_start)")
                    SA['Table'].at[si, 'Selected'] = True
                
                errcode += 1
                if not all(np.allclose(times[-1], ball_times[0][-1]) for times in ball_times):
                    err['code'] = errcode
                    err['text'] = 'time doesnt end equal for all balls (extract_data_quality_start)'
                    print(f"{si}: time doesnt end equal for all balls (extract_data_quality_start)")
                    SA['Table'].at[si, 'Selected'] = True

            # Check for tracking gaps and velocity limits
            if err['code'] is None:
                errcode += 1
                for bi in range(3):
                    ball_df = SA['Route'][shot_id][bi].sort_values('t')
                    
                    dx = ball_df['x'].diff().values[1:]
                    dy = ball_df['y'].diff().values[1:]
                    dt = ball_df['t'].diff().values[1:]
                    ds = np.sqrt(dx**2 + dy**2)
                    vabs = ds / dt
                    
                    if np.any(ds > param['NoDataDistanceDetectionLimit']):
                        err['code'] = errcode
                        err['text'] = 'gap in data is too big (extract_data_quality_start)'
                        print(f"{si}: gap in data is too big (extract_data_quality_start)")
                        SA['Table'].at[si, 'Selected'] = True
                        break
                    
                    if np.any(vabs > param['MaxVelocity']):
                        err['code'] = errcode
                        err['text'] = 'Velocity is too high (extract_data_quality_start)'
                        print(f"{si}: Velocity is too high (extract_data_quality_start)")
                        SA['Table'].at[si, 'Selected'] = True
                        break

            # Update Route0 with cleaned data
            for bi in range(3):
                SA['Route0'][shot_id][bi] = SA['Route'][shot_id][bi].copy()

            # Update error information
            SA['Table'].at[si, 'ErrorID'] = err['code']
            SA['Table'].at[si, 'ErrorText'] = err['text']

    selected_count = SA['Table']['Selected'].sum()
    total_count = len(SA['Table'])
    print(f"{selected_count}/{total_count} shots selected")
    print("done (extract_data_quality_start)")