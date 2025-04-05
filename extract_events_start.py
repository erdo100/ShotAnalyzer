import pandas as pd
from typing import Dict, Any
from str2num_b1b2b3 import str2num_b1b2b3
from eval_hit_events import eval_hit_events
from extract_events import extract_events
from eval_point_and_kiss_control import eval_point_and_kiss_control
from create_varname import create_varname


def extract_events_start(SA: dict, param: dict):
    """
    Extract all ball-ball hit events and ball-cushion hit events
    
    Args:
        SA: Dictionary containing shot data with DataFrames
        player: Dictionary containing player information
    """
    print(f"start (extract_events_start)")

    # Check if B1B2B3 column exists
    if 'B1B2B3' not in SA['Table'].columns:
        print("B1B2B3 is not identified yet (extract_events_start)")
        return

    err = 0
    err_shots = []

    shot_length = len(SA['Table'])
    for si, row in SA['Table'].iterrows():
        if row['Interpreted'] == 0:
            print(f"Shot {si+1}/{shot_length}")
            if len(row['B1B2B3']) == 3:
                # try:
                # Convert B1B2B3 string to indices
                b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(row['B1B2B3'])
                
                # Extract all events
                hit, _ = extract_events(si, SA, param)
                
                # Evaluate hit events
                hit = eval_hit_events(si, hit, b1b2b3, SA, param)
                
                # Evaluate point and kiss control
                hit = eval_point_and_kiss_control(si, hit, SA, param)
                
                # Create variables in SA Table
                SA = create_varname(si, hit, SA, param)
                
                # Store hit data
                if 'Shot' not in SA:
                    SA['Shot'] = {}
                if si not in SA['Shot']:
                    SA['Shot'][si] = {}
                SA['Shot'][si]['hit'] = hit
                
                # Update flags
                SA['Table'].at[si, 'Interpreted'] = 1
                SA['Table'].at[si, 'ErrorID'] = None
                SA['Table'].at[si, 'ErrorText'] = ''
                SA['Table'].at[si, 'Selected'] = False
                
                # Clear route data
                if 'Route' in SA and si in SA['Route']:
                    SA['Route'].drop(SA['Route'][SA['Route']['ShotID'] == row['ShotID']].index, inplace=True)
                    
                # except Exception as e:
                #     SA['Table'].at[si, 'ErrorID'] = 100
                #     SA['Table'].at[si, 'ErrorText'] = 'Check diagram, correct or delete.'
                #     SA['Table'].at[si, 'Selected'] = True
                    
                #     print("Some error occurred, probably ball routes are not continuous. Check diagram, correct or delete")
                #     print(f"Error: {str(e)}")
                #     err += 1
                #     err_shots.append(si)
            else:
                print(f"B1B2B3 has not 3 letters, skipping Shot {si+1}")

    print("These Shots are not interpreted:")
    for shot in err_shots:
        print(shot)
    
    print("done (extract_events_start)")

