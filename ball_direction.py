import numpy as np
import pandas as pd
from typing import Tuple

def ball_direction(ball_df: pd.DataFrame, hit: dict, ei: int, param: dict) -> np.ndarray:
    """
    Calculate ball direction and cushion projection points
    
    Args:
        ball_df: DataFrame containing ball trajectory with columns:
                 't': array of time points
                 'x': array of x positions
                 'y': array of y positions
        hit: Dictionary containing hit event data
        ei: Event index (0-based in Python)
        param: Dictionary containing table parameters:
               'ballR': ball radius
               'size': [table_length, table_width]
               'diamdist': diamond distance
    
    Returns:
        numpy.ndarray: 2x6 array containing:
        - direction[0,:]: Start direction info [fromCushion, PosOn, PosThrough, 0,0,0]
        - direction[1,:]: End direction info [0,0,0, toCushion, PosOn, PosThrough]
    """
    direction = np.zeros((2, 6))  # fromCushion# PosOn PosThrough toCushion# PosOn PosThrough
    imax = 10  # max node number to be considered

    # Get time indices for analysis
    if ei == 0 and len(hit['t']) > 1:
        # First event, look forward
        t1i = np.where((ball_df['t'] >= hit['t'][ei]) & (ball_df['t'] <= hit['t'][ei+1]))[0][:imax]
        t2i = np.where((ball_df['t'] >= hit['t'][ei]) & (ball_df['t'] <= hit['t'][ei+1]))[0][-imax:]
    elif ei < len(hit['t']) - 1:
        # Middle event, look before and after
        t1i = np.where((ball_df['t'] >= hit['t'][ei-1]) & (ball_df['t'] <= hit['t'][ei]))[0][-imax:]
        t2i = np.where((ball_df['t'] >= hit['t'][ei]) & (ball_df['t'] <= hit['t'][ei+1]))[0][-imax:]
    else:
        # Last event, just look at remaining points
        t1i = np.where(ball_df['t'] >= hit['t'][ei])[0][:imax]
        t2i = np.where(ball_df['t'] >= hit['t'][ei])[0][-imax:]

    # Points for extrapolation to cushions
    pstart = np.zeros((2, 2))
    pend = np.zeros((2, 2))
    
    # Initial direction directly after hit event
    pstart[0, :] = [ball_df['x'].iloc[t1i[0]], ball_df['y'].iloc[t1i[0]]]  # X/Y at start
    pend[0, :] = [ball_df['x'].iloc[t1i[-1]], ball_df['y'].iloc[t1i[-1]]]   # X/Y just after start

    # Direction just before next event
    pstart[1, :] = [ball_df['x'].iloc[t2i[0]], ball_df['y'].iloc[t2i[0]]]   # X/Y just before end
    pend[1, :] = [ball_df['x'].iloc[t2i[-1]], ball_df['y'].iloc[t2i[-1]]]    # X/Y at end

    for i in range(2):
        p1 = pstart[i, :]
        p2 = pend[i, :]
        
        if not np.any(np.isnan([p1, p2])):
            # Initialize cushion projection values
            xon1 = xon3 = xthrough1 = xthrough3 = p1[0]
            yon2 = yon4 = ythrough2 = ythrough4 = p1[1]
            
            # Calculate projections for cushions 1 and 3 (bottom and top)
            if p1[1] != p2[1]:
                # Linear interpolation functions
                f_x = lambda y: np.interp(y, [p1[1], p2[1]], [p1[0], p2[0]], 
                                         left=np.nan, right=np.nan)
                
                xon1 = f_x(param['ballR'])
                xthrough1 = f_x(-param['diamdist'])
                xon3 = f_x(param['size'][0] - param['ballR'])
                xthrough3 = f_x(param['size'][0] + param['diamdist'])
            
            # Calculate projections for cushions 2 and 4 (right and left)
            if p1[0] != p2[0]:
                # Linear interpolation functions
                f_y = lambda x: np.interp(x, [p1[0], p2[0]], [p1[1], p2[1]], 
                                         left=np.nan, right=np.nan)
                
                yon2 = f_y(param['size'][1] - param['ballR'])
                ythrough2 = f_y(param['size'][1] + param['diamdist'])
                yon4 = f_y(param['ballR'])
                ythrough4 = f_y(-param['diamdist'])
            
            # Check cushion 1 (bottom)
            if (param['ballR'] <= xon1 <= param['size'][1] - param['ballR']):
                if p2[1] < p1[1]:  # Moving toward cushion 1
                    direction[i, 3] = 1  # toCushion
                    direction[i, 4] = xon1
                    direction[i, 5] = xthrough1
                else:  # Moving away from cushion 1
                    direction[i, 0] = 1  # fromCushion
                    direction[i, 1] = xon1
                    direction[i, 2] = xthrough1
            
            # Check cushion 3 (top)
            if (param['ballR'] <= xon3 <= param['size'][1] - param['ballR']):
                if p1[1] < p2[1]:  # Moving toward cushion 3
                    direction[i, 3] = 3  # toCushion
                    direction[i, 4] = xon3
                    direction[i, 5] = xthrough3
                else:  # Moving away from cushion 3
                    direction[i, 0] = 3  # fromCushion
                    direction[i, 1] = xon3
                    direction[i, 2] = xthrough3
            
            # Check cushion 2 (right)
            if (param['ballR'] <= yon2 <= param['size'][0] - param['ballR']):
                if p1[0] < p2[0]:  # Moving toward cushion 2
                    direction[i, 3] = 2  # toCushion
                    direction[i, 4] = yon2
                    direction[i, 5] = ythrough2
                else:  # Moving away from cushion 2
                    direction[i, 0] = 2  # fromCushion
                    direction[i, 1] = yon2
                    direction[i, 2] = ythrough2
            
            # Check cushion 4 (left)
            if (param['ballR'] <= yon4 <= param['size'][0] - param['ballR']):
                if p2[0] < p1[0]:  # Moving toward cushion 4
                    direction[i, 3] = 4  # toCushion
                    direction[i, 4] = yon4
                    direction[i, 5] = ythrough4
                else:  # Moving away from cushion 4
                    direction[i, 0] = 4  # fromCushion
                    direction[i, 1] = yon4
                    direction[i, 2] = ythrough4
    
    return direction