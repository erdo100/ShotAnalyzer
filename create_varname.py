import pandas as pd
from typing import Dict, Any
from str2num_B1B2B3 import str2num_B1B2B3
from replace_colors_b1b2b3 import replace_colors_b1b2b3


def create_varname(si: int, hit: dict, SA: dict, param: dict) -> dict:
    """
    Create and populate variables in the SA table based on hit data
    
    Args:
        SA: Dictionary containing shot data with DataFrames
        hit: Dictionary containing hit data for each ball
        si: Shot index
        param: Dictionary containing parameters
        
    Returns:
        Updated SA dictionary with new variables
    """
    varparts = list(hit[0].keys())  # Get field names from first ball's hit data
    
    # Get ball order
    b1b2b3_str = SA['Table'].loc[si, 'B1B2B3']
    b1b2b3, b1i, b2i, b3i = str2num_B1B2B3(b1b2b3_str)
    
    # Write BX_with variables
    for balli in [1, 2, 3]:
        varname = f'B{balli}_with'
        
        # Check if column exists, create if needed
        if varname not in SA['Table'].columns:
            SA['Table'][varname] = None  # Initialize column
        
        # Update value for this shot
        with_new = replace_colors_b1b2b3(hit[b1b2b3[balli-1]]['with'][0], b1b2b3)
        SA['Table'].at[si, varname] = with_new
    
    # Write basic metrics
    SA['Table'].at[si, 'Point0'] = hit[b1i]['Point']
    SA['Table'].at[si, 'Kiss'] = hit[b1i]['Kiss']
    SA['Table'].at[si, 'Fuchs'] = hit[b1i]['Fuchs']
    SA['Table'].at[si, 'PointDist'] = hit[b1i]['PointDist']
    SA['Table'].at[si, 'KissDistB1'] = hit[b1i]['KissDistB1']
    SA['Table'].at[si, 'AimOffsetSS'] = hit[b1i]['AimOffsetSS']
    SA['Table'].at[si, 'AimOffsetLL'] = hit[b1i]['AimOffsetLL']
    SA['Table'].at[si, 'B1B2OffsetSS'] = hit[b1i]['B1B2OffsetSS']
    SA['Table'].at[si, 'B1B2OffsetLL'] = hit[b1i]['B1B2OffsetLL']
    SA['Table'].at[si, 'B1B3OffsetSS'] = hit[b1i]['B1B3OffsetSS']
    SA['Table'].at[si, 'B1B3OffsetLL'] = hit[b1i]['B1B3OffsetLL']
    
    # Variables to exclude for first hit
    delname = ['Fraction', 'DefAngle', 'CutAngle', 'CInAngle', 'COutAngle']
    
    # Maximum hits to process per ball
    himax = [8, 4, 0]
    
    # Write detailed hit metrics
    for bi in [1, 2, 3]:
        ball_idx = b1b2b3[bi-1]
        for hi in range(1, himax[bi-1] + 1):
            for vi in range(20, len(varparts)):  # Starting from index 21 in MATLAB (20 in Python)
                var = varparts[vi]
                
                # Skip certain variables for first hit
                if hi == 1 and var in delname:
                    continue
                
                # Check if data exists for this hit
                if var in hit[ball_idx] and len(hit[ball_idx][var]) >= hi:
                    # Scale position variables to diamonds
                    scale = param['size'][1]/8 if 'Pos' in var else 1
                    
                    # Create variable name and store value
                    varname = f'B{bi}_{hi}_{var}'
                    SA['Table'].at[si, varname] = hit[ball_idx][var][hi-1]/scale
    
    return SA

