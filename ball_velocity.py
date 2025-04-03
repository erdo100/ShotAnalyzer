import numpy as np

def ball_velocity(ball: dict, hit: dict, ei: int) -> tuple:
    """
    Calculates ball velocities and angles before and after a hit event
    
    Args:
        ball: Dictionary containing ball trajectory data with keys:
              't': array of time points
              'x': array of x positions
              'y': array of y positions
        hit: Dictionary containing hit event data
        ei: Event index to analyze
        
    Returns:
        tuple: (vt, v1, v2, alpha, offset)
        - vt: Tuple of velocities (before, after) in m/s
        - v1: Velocity vector before hit [vx, vy] in m/s
        - v2: Velocity vector after hit [vx, vy] in m/s  
        - alpha: Tuple of angles (before, after) in degrees
        - offset: Offset distance in mm
    """
    imax = 10
    vt = [0.0, 0.0]
    v1 = np.array([0.0, 0.0])
    v2 = np.array([0.0, 0.0])
    alpha = [np.nan, np.nan]
    offset = np.nan
    
    if hit['with'][ei] != '-':
        # Find nearest time index for the hit event
        ti = np.abs(ball['t'] - hit['t'][ei]).argmin()
        
        # Find indices for previous and next hits
        if ei > 0: # matlab: > 1
            ti_before = np.abs(ball['t'] - hit['t'][ei-1]).argmin() # matlab: [ei]
        else:
            ti_before = 0  # First point
            
        if ei+1 < len(hit['t']): # matlab : < length(hit.t)
            ti_after = np.abs(ball['t'] - hit['t'][ei+1]).argmin()
        else:
            ti_after = len(ball['t']) - 1  # Last point
            
        # Calculate velocity before hit (v1)
        it = min(ti - ti_before, imax)
        if it >= 1 or ei == 0:  # Note: ei is 0-based in Python
            if ei == 0:
                ind = [1, 3]  # Using 1 and 3 to get initial velocity
            else:
                ind = [ti-it, ti]
                
            # Calculate velocity components
            dx = np.sqrt(np.diff(ball['x'][ind])**2 + np.diff(ball['y'][ind])**2)
            dt = np.diff(ball['t'][ind])
            
            # Avoid division by zero
            valid = dt > 0
            if valid:
                vt0 = dx / dt
                vx = np.diff(ball['x'][ind]) / dt
                vy = np.diff(ball['y'][ind]) / dt
                
                if vt0 > 0:
                    vt[0] = vt0 / 1000  # Convert mm/s to m/s
                    v1 = np.array([vx / 1000, vy / 1000])  # Convert to m/s
                    alpha[0] = np.degrees(np.arctan2(vx, vy))
                else:
                    vt[0] = vt0 / 1000
                    v1 = np.array([0.0, 0.0])
                    alpha[0] = np.nan
                    
        # Calculate velocity after hit (v2)
        it = min(ti_after - ti, imax)
        if it >= 0: # matlab: >= 1
            ind = [ti, ti+it] if ti+it < len(ball['t']) else [ti, ti_after]
            dx = np.sqrt(np.diff(ball['x'][ind])**2 + np.diff(ball['y'][ind])**2)
            dt = np.diff(ball['t'][ind])
            
            valid = dt > 0
            if valid:
                vt0 = dx / dt
                vx = np.diff(ball['x'][ind]) / dt
                vy = np.diff(ball['y'][ind]) / dt
                
                if vt0 > 0:
                    vt[1] = vt0 / 1000  # Convert mm/s to m/s
                    v2 = np.array([vx / 1000, vy / 1000])  # Convert to m/s
                    alpha[1] = np.degrees(np.arctan2(vx, vy))
                else:
                    vt[1] = vt0 / 1000
                    v2 = np.array([0.0, 0.0])
                    alpha[1] = np.nan
                    
        # Calculate offset
        p1 = np.array([ball['x'][ti], ball['y'][ti], 0])
        p2 = np.array([ball['x'][ti_after], ball['y'][ti_after], 0])
        
        if np.linalg.norm(v2) > 0:
            offset = np.linalg.norm(np.cross(p2-p1, np.append(v2, 0))) / np.linalg.norm(v2)
            
    return tuple(vt), v1, v2, tuple(alpha), offset