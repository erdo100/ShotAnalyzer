import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from angle_vector import angle_vector


def extract_b1b2b3_start(SA: dict, param: dict):
    """
    Processes ball position data for each shot using nested Route structure
    """
    print(f"start (extract_b1b2b3_start)")
    
    varname = 'B1B2B3'
    selected_count = 0
    color_to_idx = {'W': 0, 'Y': 1, 'R': 2}  # Map color letters back to BallIDs

    # Process each shot in the DataFrame
    for si, row in SA['Table'].iterrows():
        if row['Interpreted'] == 0:
            shot_id = row['ShotID']
            
            # Skip if shot data is missing
            if shot_id not in SA['Route']:
                err = {'code': 3, 'text': 'Missing shot data in Route'}
                SA['Table'].at[si, 'Selected'] = True
                SA['Table'].at[si, 'ErrorID'] = err['code']
                SA['Table'].at[si, 'ErrorText'] = err['text']
                selected_count += 1
                continue
                
            # Call the calculation function (now takes nested dict structure)
            b1b2b3_str, err = extract_b1b2b3(SA['Route'][shot_id])
            
            if err['code'] is not None:
                SA['Table'].at[si, 'Selected'] = True
                print(f"{si}: {err['text']}")
                selected_count += 1
            
            # Update the DataFrame with the order string
            SA['Table'].at[si, varname] = b1b2b3_str

            # Calculate and store positions for each ball (B1, B2, B3)
            for bi in range(1, 4):  # B1, B2, B3
                # Get the actual ball index from the color string
                ball_color = b1b2b3_str[bi-1]  # Get W/Y/R for B1/B2/B3
                ball_idx = color_to_idx[ball_color]  # Convert to 0/1/2
                
                # Get the ball's initial position
                if ball_idx not in SA['Route'][shot_id] or len(SA['Route'][shot_id][ball_idx]) == 0:
                    continue
                
                ball_data = SA['Route'][shot_id][ball_idx]
                first_row = ball_data.iloc[0]
                
                # Calculate normalized positions
                x_pos = first_row['x'] / param['size'][1] * 8
                y_pos = first_row['y'] / param['size'][0] * 4
                
                # Create or update columns
                posx_col = f'B{bi}posX'
                posy_col = f'B{bi}posY'
                
                if posx_col not in SA['Table'].columns:
                    SA['Table'][posx_col] = None
                if posy_col not in SA['Table'].columns:
                    SA['Table'][posy_col] = None

                SA['Table'].at[si, posx_col] = x_pos
                SA['Table'].at[si, posy_col] = y_pos

            SA['Table'].at[si, 'ErrorID'] = err['code']
            SA['Table'].at[si, 'ErrorText'] = err['text']

    total_shots = len(SA['Table'])
    print(f"{selected_count}/{total_shots} shots selected")
    print("done (extract_b1b2b3_start)")


def extract_b1b2b3(shot_route: Dict[int, pd.DataFrame]) -> Tuple[str, Dict[str, Any]]:
    """
    Determine ball movement order from nested shot route data
    Returns: (order string like "WYR", error dict)
    """
    # Initialize outputs
    b1b2b3 = "WYR"  # Default color order
    err = {'code': None, 'text': ''}
    color_map = {0: 'W', 1: 'Y', 2: 'R'}  # Map BallID to color

    # Verify all balls exist in the shot data
    balls = {}
    for ball_id in [0, 1, 2]:
        if ball_id not in shot_route or len(shot_route[ball_id]) < 2:
            err['code'] = 2
            err['text'] = f'Missing data for ball {ball_id} (extract_b1b2b3)'
            return b1b2b3, err
        
        balls[ball_id] = {
            't': shot_route[ball_id]['t'].values,
            'x': shot_route[ball_id]['x'].values,
            'y': shot_route[ball_id]['y'].values
        }

    # Sort balls by their second timestamp to determine initial order
    second_times = [(ball_id, balls[ball_id]['t'][1]) for ball_id in balls]
    sorted_balls = sorted(second_times, key=lambda x: x[1])
    ball_order = [x[0] for x in sorted_balls]  # BallIDs in order of movement

    # Map BallIDs to colors (e.g., [0, 1, 2] -> "WYR")
    b1b2b3 = ''.join([color_map[ball_id] for ball_id in ball_order])

    # Early exit if only B1 moved
    if len(balls[ball_order[0]]['t']) >= 3:
        b1t2 = balls[ball_order[0]]['t'][1]
        b1t3 = balls[ball_order[0]]['t'][2]

        # Check if other balls moved before b1t2/b1t3
        moved_b2 = any(t <= b1t2 for t in balls[ball_order[1]]['t'][1:])
        moved_b3 = any(t <= b1t2 for t in balls[ball_order[2]]['t'][1:])

        if moved_b2 and not moved_b3:
            # Check angles between B1 and B2
            vec_b1b2 = np.array([
                balls[ball_order[1]]['x'][0] - balls[ball_order[0]]['x'][0],
                balls[ball_order[1]]['y'][0] - balls[ball_order[0]]['y'][0]
            ])
            vec_b1dir = np.array([
                balls[ball_order[0]]['x'][1] - balls[ball_order[0]]['x'][0],
                balls[ball_order[0]]['y'][1] - balls[ball_order[0]]['y'][0]
            ])
            angle = angle_vector(vec_b1b2, vec_b1dir)
            if angle > 90:  # Swap B1 and B2
                b1b2b3 = color_map[ball_order[1]] + color_map[ball_order[0]] + color_map[ball_order[2]]

        elif not moved_b2 and moved_b3:
            # Check angles between B1 and B3
            vec_b1b3 = np.array([
                balls[ball_order[2]]['x'][0] - balls[ball_order[0]]['x'][0],
                balls[ball_order[2]]['y'][0] - balls[ball_order[0]]['y'][0]
            ])
            vec_b1dir = np.array([
                balls[ball_order[0]]['x'][1] - balls[ball_order[0]]['x'][0],
                balls[ball_order[0]]['y'][1] - balls[ball_order[0]]['y'][0]
            ])
            angle = angle_vector(vec_b1b3, vec_b1dir)
            if angle > 90:  # Swap B1 and B3
                b1b2b3 = color_map[ball_order[2]] + color_map[ball_order[1]] + color_map[ball_order[0]]

        elif moved_b2 and moved_b3:
            err['code'] = 2
            err['text'] = 'All balls moved simultaneously (extract_b1b2b3)'

    return b1b2b3, err