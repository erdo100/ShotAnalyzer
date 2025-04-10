import numpy as np
import copy
from scipy.interpolate import interp1d

from str2num_b1b2b3 import str2num_b1b2b3
from angle_vector import angle_vector
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter  # For MP4

class plotshot:
    def __init__(self, param, ShotID):
        self.param = param
        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(7, 12))

        ax.set_ylim(0, 2840)  # Set appropriate limits for your data
        ax.set_xlim(0, 1420)
        
        #show axis real size
        ax.set_aspect('equal', adjustable='box')
        
        # make figure close around axis
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ax.set_yticks(np.linspace(0, 2840, 9))
        ax.set_xticks(np.linspace(0, 1420, 5))

        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.8, color='gray')

        ax.set_facecolor((0.95, 0.95, 0.95))
        ax.set_xticklabels([])  # Remove x-axis labels
        ax.set_yticklabels([])  # Remove y-axis labels
        ax.tick_params(axis='both', which='both', length=0)

        self.ball_line={}
        self.ball_line[0], = ax.plot([], [], 'wo-', label='Ball 0')  # Line for Ball 0
        self.ball_line[1], = ax.plot([], [], 'yo-', label='Ball 1')  # Line for Ball 0
        self.ball_line[2], = ax.plot([], [], 'ro-', label='Ball 2')  # Line for Ball 0
        
        self.appr_line={}
        self.appr_line[0], = ax.plot([], [], 'k+')  # Line for Ball 0
        self.appr_line[1], = ax.plot([], [], 'k+')  # Line for Ball 0
        self.appr_line[2], = ax.plot([], [], 'k+')  # Line for Ball 0
        
        self.ball_circ = {}
        self.ball_circ[0] = plt.Circle((200, 220), self.param['ballR'], color='w', linewidth=2, fill=False)
        self.ball_circ[1] = plt.Circle((100, 500), self.param['ballR'], color='y', linewidth=2, fill=False)
        self.ball_circ[2] = plt.Circle((800, 1000), self.param['ballR'], color='r', linewidth=2, fill=False)

        ax.add_patch(self.ball_circ[0])
        ax.add_patch(self.ball_circ[1])
        ax.add_patch(self.ball_circ[2])


        # Initialize video writer (for MP4)
        self.writer = FFMpegWriter(fps=15)  # Adjust FPS as needed
        self.writer.setup(fig, f"extract_events_{ShotID}.mp4", dpi=100)  # Start recording

        self.ax = ax
        self.fig = fig

    def close(self):
            plt.close(self.fig)
    
    def plot(self, ball, b):
        # Update the ball positions
        self.ball_line[0].set_data(ball[0]['y'], ball[0]['x'])
        self.ball_line[1].set_data(ball[1]['y'], ball[1]['x'])
        self.ball_line[2].set_data(ball[2]['y'], ball[2]['x'])

        # Update the circle patches for each ball
        self.ball_circ[0].center = (ball[0]['y'][-1], ball[0]['x'][-1])
        self.ball_circ[1].center = (ball[1]['y'][-1], ball[1]['x'][-1])
        self.ball_circ[2].center = (ball[2]['y'][-1], ball[2]['x'][-1])

        plt.draw()
        plt.pause(0.01)

    def plot_appr(self, b):
        self.appr_line[0].set_data([b[0]['ya']], [b[0]['xa']])
        self.appr_line[1].set_data([b[1]['ya']], [b[1]['xa']])
        self.appr_line[2].set_data([b[2]['ya']], [b[2]['xa']])

        plt.draw()
        plt.pause(0.01)

    def plot_hit(self, x, y):
        # Plot the hit points
        circ = plt.Circle((y, x),
                    self.param['ballR'], 
                    color='k', linestyle='--',linewidth=2, fill=False)
        self.ax.add_patch(circ)

        plt.draw()
        plt.pause(0.01)

    def update(self):
        plt.draw()
        self.writer.grab_frame() 
        plt.pause(0.01)

def extract_events(SA, si, param):
    """
    Extract events from the shot data.

    Args:
        SA (object): Shot Analyzer object containing shot data.
        param (dict): Parameters for extraction.
    """

    # Initiate Plot and video
    ShotID = SA['Table'].iloc[si]['ShotID']
    ps = plotshot(param, ShotID)


    b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(SA['Table'].iloc[si]['B1B2B3'])

    # Ball-Ball distance calculation
    BB = [[0, 1], [0, 2], [1, 2]]  # 0-based ball pairs (W=0, Y=1, R=2)
    eps = np.finfo(float).eps

    # Get copy from original data
    ball0 = [None]*3
    for bi in range(3):
        # Copy Route0 to Route
        ball0[bi] = copy.deepcopy(SA['Shot'][si]['Route'][bi])
        
    # Initialize ball positions
    ball = {bi: {'x': [], 'y': [], 't': []} for bi in range(3)}
    for bi in range(3):
        ball[bi]['x'].append(SA['Shot'][si]['Route'][bi]['x'][0])
        ball[bi]['y'].append(SA['Shot'][si]['Route'][bi]['y'][0])
        ball[bi]['t'].append(0.0)  # Initial time

    # Create common time points
    Tall0 = np.sort(np.unique(np.concatenate([b['t'] for b in ball0])))
    tvec = np.linspace(0.01, 1, 101)

    # Initialize hit structure
    hit = {
        bi: {
            'with': ['-'],  # List of strings
            't': [0.0],     # List of timestamps
            'XPos': [ball0[bi]['x'][0]],  # Initial X position
            'YPos': [ball0[bi]['y'][0]],  # Initial Y position
            'Kiss': [0],
            'Point': [0],
            'Fuchs': [0],
            'PointDist': [3000.0],
            'KissDistB1': [3000.0],
            'Tpoint': [3000.0],
            'Tkiss': [3000.0],
            'Tready': [3000.0],
            'TB2hit': [3000.0],
            'Tfailure': [3000.0]
        } for bi in range(3)
    }

    # Set initial hit values for each ball
    hit[b1i]['with'][0] = 'S'
    hit[b1i]['XPos'][0] = [ball0[b1i]['x'][0]]
    hit[b1i]['YPos'][0] = [ball0[b1i]['y'][0]]

    hit[b2i]['XPos'][0] = [ball0[b2i]['x'][0]]
    hit[b2i]['YPos'][0] = [ball0[b2i]['y'][0]]

    hit[b3i]['XPos'][0] = [ball0[b3i]['x'][0]]
    hit[b3i]['YPos'][0] = [ball0[b3i]['y'][0]]

    
    # Calculate velocities for each ball
    for bi in range(3):
        # Compute time differences
        ball0[bi]['dt'] = np.diff(ball0[bi]['t'])
        ball0[bi]['vx'] = np.diff(ball0[bi]['x']) / ball0[bi]['dt']
        ball0[bi]['vy'] = np.diff(ball0[bi]['y']) / ball0[bi]['dt']
        
        # Append 0 to velocities to match original time array length
        ball0[bi]['vx'] = np.concatenate([ball0[bi]['vx'], [0.0]])
        ball0[bi]['vy'] = np.concatenate([ball0[bi]['vy'], [0.0]])
        dt = np.concatenate([ball0[bi]['dt'] , [ball0[bi]['dt'][-1]]])

        # Compute velocity magnitude
        ball0[bi]['v']  = np.sqrt(ball0[bi]['vx']**2 + ball0[bi]['vy']**2)

    # Initialize b as a list of dictionaries with required fields
    b = [{'vt1': 0.0, 'v1': [0.0, 0.0], 'v2': [0.0, 0.0], 'vt2': 0.0, 'xa': 0.0, 'ya': 0.0} for _ in range(3)]

    ti = -1
    do_scan = True;
    while do_scan:
        ti += 1
        
        # Approximate Position of Ball at next time step
        tappr = Tall0[0] + np.diff(Tall0[0:2]) * tvec
        dT = np.diff(Tall0[0:2])
        
        for bi in range(3):
            # Check if it is last index
            if len(ball0[bi]['t']) >= 2:
                # Travel distance in original data
                if len(ball[bi]['x']) >= 2:
                    ds0 = np.sqrt((ball[bi]['x'][1] - ball[bi]['x'][0]) ** 2 + (ball[bi]['y'][1] - ball[bi]['y'][0]) ** 2)
                    v0 = ds0 / dT
                
                # current Speed
                b[bi]['vt1'] = ball0[bi]['v'][0]

                # was last hit in last time step?
                iprev = np.where(np.array(ball[bi]['t']) >= hit[bi]['t'][-1])[0]
                if len(iprev) > 0:
                    iprev = iprev[-param['timax_appr']:]  # Get last N indices
                else:
                    iprev = []  # No previous hits

                # Velocity Components in the next time step
                if hit[bi]['with'][-1] == '-':  # Check last element of 'with' list
                    # Ball was not moving previously
                    b[bi]['v1'] = [0.0, 0.0]
                    # Use second element (index 1 in 0-based) of velocity arrays
                    ball[bi]['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
                    
                elif len(iprev) == 1:
                    # When last hit was on this ball - use first element (index 0) of velocities
                    b[bi]['v1'] = [ball0[bi]['vx'][0], ball0[bi]['vy'][0]]
                    b[bi]['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
                    
                else:
                    # Extrapolate using linear regression
                    # Create design matrix [t, 1] for the selected previous points
                    A = np.column_stack([
                        np.array(ball[bi]['t'])[iprev],  # Time values
                        np.ones(len(iprev))  # Intercept term
                    ])
                    
                    # Solve linear systems using least squares
                    CoefVX, _ = np.linalg.lstsq(A, np.array(ball[bi]['x'])[iprev], rcond=None)[0]
                    CoefVY, _ = np.linalg.lstsq(A, np.array(ball[bi]['y'])[iprev], rcond=None)[0]
                    
                    # Velocity components are the coefficients (slopes)
                    b[bi]['v1'] = [CoefVX, CoefVY]
                    # Use second element (index 1) of velocity arrays
                    b[bi]['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
                    
                # Calculate velocity magnitudes
                b[bi]['vt1'] = np.linalg.norm(b[bi]['v1'])
                b[bi]['vt2'] = np.linalg.norm(b[bi]['v2'])

                # Determine maximum velocity
                vtnext = max(b[bi]['vt1'], ball0[bi]['v'][0])

                # Calculate normalized velocity vector
                if np.linalg.norm(b[bi]['v1']) > 1e-6:  # Avoid division by zero
                    vnext = (np.array(b[bi]['v1']) / np.linalg.norm(b[bi]['v1'])) * vtnext
                else:
                    vnext = np.array([0.0, 0.0])

                # Approximate positions for next time steps
                b[bi]['xa'] = ball[bi]['x'][-1] + vnext[0] * dT * tvec
                b[bi]['ya'] = ball[bi]['y'][-1] + vnext[1] * dT * tvec

        ps.plot_appr(b)
        # Update the plot
        ps.plot(ball, b)

        # Calculate Ball trajectory angle change
        for bi in range(3):  # 0-based indexing for balls 0, 1, 2
            # Convert velocity vectors to numpy arrays
            v1 = np.array(b[bi]['v1'])
            v2 = np.array(b[bi]['v2'])
            
            # Calculate angle between velocity vectors using the provided function
            b[bi]['a12'] = angle_vector(v1, v2)


        # Initialize distance storage
        d = [{} for _ in range(3)]

        for bbi in range(3):
            # Get ball indices for this pair (0-based)
            bx1 = b1b2b3[BB[bbi][0]]
            bx2 = b1b2b3[BB[bbi][1]]
            
            # Calculate Euclidean distance between ball centers minus 2*radius
            dx = np.array(b[bx1]['xa']) - np.array(b[bx2]['xa'])
            dy = np.array(b[bx1]['ya']) - np.array(b[bx2]['ya'])
            d[bbi]['BB'] = np.sqrt(dx**2 + dy**2) - 2 * param['ballR']

        # Cushion distance calculation
        for bi in range(3):
            # Initialize cushion distances dictionary with 4 directions
            b[bi]['cd'] = {
                0: np.array(b[bi]['ya']) - param['ballR'],         # Bottom cushion
                1: param['size'][1] - param['ballR'] - np.array(b[bi]['xa']),  # Right cushion
                2: param['size'][0] - param['ballR'] - np.array(b[bi]['ya']),  # Top cushion
                3: np.array(b[bi]['xa']) - param['ballR']          # Left cushion
            }

        # Initialize hit tracking
        hitlist = []
        lasthitball = 0


        # Evaluate cushion hits
        for bi in range(3):  # 0-based ball index (0, 1, 2)
            for cii in range(4):  # 0-based cushion index (0:bottom, 1:right, 2:top, 3:left)
                # Convert to numpy arrays for vector operations
                cushion_dists = np.array(b[bi]['cd'][cii])
                
                # Check distance condition (any point <= 0)
                checkdist = np.any(cushion_dists <= 0)
                
                # Check angle condition (angle > 1° or invalid measurement)
                checkangle = b[bi]['a12'] > 1 or b[bi]['a12'] == -1
                
                # Current velocity components
                velx, vely = b[bi]['v1'][0], b[bi]['v1'][1]
                
                checkcush = False
                tc = 0.0
                cushx = 0.0
                cushy = 0.0
                
                # Bottom cushion (cii=0)
                if checkdist and checkangle and cii == 0:
                    if vely < 0 and vely != 0:  # Moving downward
                        # Interpolate contact time
                        f = interp1d(cushion_dists, tappr, fill_value='extrapolate')
                        tc = float(f(0))
                        # Interpolate contact position
                        cushx = float(interp1d(tappr, b[bi]['xa'], fill_value='extrapolate')(tc))
                        cushy = param['ballR']
                        checkcush = True
                        
                # Right cushion (cii=1)
                elif checkdist and checkangle and cii == 1:
                    if velx > 0 and velx != 0:  # Moving right
                        f = interp1d(cushion_dists, tappr, fill_value='extrapolate')
                        tc = float(f(0))
                        cushx = param['size'][1] - param['ballR']
                        cushy = float(interp1d(tappr, b[bi]['ya'], fill_value='extrapolate')(tc))
                        checkcush = True
                        
                # Top cushion (cii=2)
                elif checkdist and checkangle and cii == 2:
                    if vely > 0 and vely != 0:  # Moving upward
                        f = interp1d(cushion_dists, tappr, fill_value='extrapolate')
                        tc = float(f(0))
                        cushy = param['size'][0] - param['ballR']
                        cushx = float(interp1d(tappr, b[bi]['xa'], fill_value='extrapolate')(tc))
                        checkcush = True
                        
                # Left cushion (cii=3)
                elif checkdist and checkangle and cii == 3:
                    if velx < 0 and velx != 0:  # Moving left
                        f = interp1d(cushion_dists, tappr, fill_value='extrapolate')
                        tc = float(f(0))
                        cushx = param['ballR']
                        cushy = float(interp1d(tappr, b[bi]['ya'], fill_value='extrapolate')(tc))
                        checkcush = True
                
                # Handle negative time collision
                if tc < 0:
                    print(f"Warning: Negative collision time detected for ball {bi} on cushion {cii}")
                
                if checkcush:
                    hitlist.append([
                        tc,            # Contact time
                        bi,            # Ball index
                        2,             # Contact type (2=cushion)
                        cii,           # Cushion ID (0-3)
                        cushx,         # Contact X position
                        cushy          # Contact Y position
                    ])


        # Evaluate ball-ball hits
        for bbi in range(3):
            bx1 = b1b2b3[BB[bbi][0]]  # First ball in pair
            bx2 = b1b2b3[BB[bbi][1]]  # Second ball in pair
            dist_data = d[bbi]['BB']
            
            checkdist = False
            tc = 0.0
              
            if (np.any(dist_data <= 0) and np.any(dist_data > 0) and
                (dist_data[0] >= 0 and dist_data[-1] < 0) or 
                (dist_data[0] < 0 and dist_data[-1] >= 0)):
                #Balls are going to be in contact or are already in contact. Previous contact not detected
                checkdist = True
                f = interp1d(dist_data, tappr, fill_value='extrapolate')
                tc = float(f(0))
            
            elif (np.any(dist_data <= 0) and np.any(dist_data > 0)):
                # here the ball-ball contact is going through the balls
                checkdist = True
                ind = np.where(np.diff(dist_data) <= 0)[0]
                f = interp1d(dist_data[ind], tappr[ind], fill_value='extrapolate')
                tc = float(f(0))

            # Angle checks (using modified angle_vector function)
            checkangle_b1 = (b[bx1]['a12'] > 10) or (b[bx1]['a12'] == -1)
            checkangle_b2 = (b[bx2]['a12'] > 10) or (b[bx2]['a12'] == -1)

            # Velocity checks
            checkvel_b1 = False
            if abs(b[bx1]['vt1']) > eps or abs(b[bx1]['vt2']) > eps:
                checkvel_b1 = abs(b[bx1]['vt2'] - b[bx1]['vt1']) > 50

            checkvel_b2 = False
            if abs(b[bx2]['vt1']) > eps or abs(b[bx2]['vt2']) > eps:
                checkvel_b2 = abs(b[bx2]['vt2'] - b[bx2]['vt1']) > 50

            # Time collision check
            checkdouble = False
            if (tc >= hit[bx1]['t'][-1]+0.01) and (tc>= hit[bx1]['t'][-1]+0.01):
                checkdouble = True
            
            # Final collision check
            if checkdouble and checkdist:
                # Create interpolators for both balls
                interp_x1 = interp1d(tappr, b[bx1]['xa'], fill_value='extrapolate')
                interp_y1 = interp1d(tappr, b[bx1]['ya'], fill_value='extrapolate')
                interp_x2 = interp1d(tappr, b[bx2]['xa'], fill_value='extrapolate')
                interp_y2 = interp1d(tappr, b[bx2]['ya'], fill_value='extrapolate')

                # Add entries for both directions
                hitlist.append([
                    tc,                # Contact time
                    bx1,               # Ball 1 index
                    1,                 # Contact type (1=ball-ball)
                    bx2,               # Ball 2 index
                    float(interp_x1(tc)),  # X position
                    float(interp_y1(tc))   # Y position
                ])
                
                hitlist.append([
                    tc,                # Contact time
                    bx2,               # Ball 1 index (reverse)
                    1,                 # Contact type (1=ball-ball)
                    bx1,               # Ball 2 index (reverse)
                    float(interp_x2(tc)),  # X position
                    float(interp_y2(tc))   # Y position
                ])



        # When Just before the Ball-Ball hit velocity is too small, then hit can be missed
        # therefore we check whether B2 is moving without hit
        # now only for first hit
        # then we have to calculate the past hit
        # Move back B1 so that B1 is touching B2
        if (ti == 0 and 
            len(hit[b2i]['t']) == 1 and 
            not hitlist and
            ball0[b1i]['t'][1] >= ball0[b2i]['t'][1] and 
            (ball0[b2i]['x'][0] != ball0[b2i]['x'][1] or 
            ball0[b2i]['y'][0] != ball0[b2i]['y'][1])):

            # Find the correct ball pair index
            target_pair = [b1i, b2i]
            for bbi, pair in enumerate(BB):
                if [b1b2b3[pair[0]], b1b2b3[pair[1]]] == target_pair:
                    break

            # Calculate collision time
            try:
                f = interp1d(d[bbi]['BB'], tappr, fill_value='extrapolate')
                tc = float(f(0))
            except:
                tc = tappr[np.argmin(d[bbi]['BB'])]

            if tc < 0:
                tc = tappr[-1]

            # Add missed collision entries
            interp_x = interp1d(tappr.ravel(), b[b1i]['xa'].ravel(), fill_value='extrapolate')
            interp_y = interp1d(tappr.ravel(), b[b1i]['ya'].ravel(), fill_value='extrapolate')
            
            hitlist.append([
                tc, b1i, 1, b2i,
                float(interp_x(tc)),
                float(interp_y(tc))
            ])
            
            hitlist.append([
                tc, b2i, 1, b1i,
                ball0[b2i]['x'][0],
                ball0[b2i]['y'][0]
            ])

        # Check first Time step for Ball-Ball hit
        # if Balls are too close, so that the direction change is not visible,
        # then the algorithm cant detect the Bal Ball hit. Therefore:
        # Check whether B2 is moving with without hit
        # If yes, then use the direction of B2, calculate the perpendicular
        # direction and assign to B1
        if (ti == 1 and 
            not hitlist and 
            ball0[b1i]['t'][1] == ball0[b2i]['t'][1] and
            (ball0[b2i]['x'][0] != ball0[b2i]['x'][1] or 
            ball0[b2i]['y'][0] != ball0[b2i]['y'][1])):

            # Calculate B2's direction vector (from first to second position)
            vec_b2dir = np.array([
                ball0[b2i]['x'][1] - ball0[b2i]['x'][0],
                ball0[b2i]['y'][1] - ball0[b2i]['y'][0]
            ])
            
            # Calculate contact position if B2 is moving
            norm_dir = np.linalg.norm(vec_b2dir)
            # Calculate perpendicular contact position
            tangent_dir = vec_b2dir / norm_dir
            contact_offset = tangent_dir * 2 * param['ballR']
            b1pos2 = np.array([
                ball0[b2i]['x'][0],
                ball0[b2i]['y'][0]
            ]) - contact_offset

            # Calculate collision time (midpoint of first interval)
            tc = ball0[b1i]['t'][1] / 2

            # Add entries for both balls
            hitlist.append([
                tc,          # Contact time
                b1i,         # Ball 1 index
                1,           # Contact type (ball-ball)
                b2i,         # Ball 2 index
                b1pos2[0],   # X position
                b1pos2[1]    # Y position
            ])
            
            hitlist.append([
                tc,          # Contact time
                b2i,         # Ball 2 index
                1,           # Contact type (ball-ball)
                b1i,         # Ball 1 index
                ball0[b2i]['x'][0],  # Original B2 position
                ball0[b2i]['y'][0]   # Original B2 position
            ])




        # Assign new hit event or next timestep in to the ball route history
        # if hit: replace current point with hit,
        #     - add current point of ball0 to ball
        #     - replace current point with new hit point
        # otherwise
        #     - add current point of ball0 to ball
        #     - delete current point in ball0
        bi_list = [0, 1, 2]  # List of balls to check for hits
        # Check if hitlist exists and next time step is valid
        if hitlist and Tall0[1] >= tc:
            # Get earliest collision time
            tc = min(hit[0] for hit in hitlist)
            
            # Check if collision is after next time step
            if Tall0[1] < tc:
                print("Warning: Hit is after next time step")
            
            # Validate single hit per ball at this time
            for bi in range(3):  # 0-based ball indices (0,1,2)
                # Count hits for this ball at this time
                hit_count = sum(1 for hit in hitlist if hit[0] == tc and hit[1] == bi)
                
                if hit_count > 1:
                    raise ValueError(f"Ball {bi} has {hit_count} hits at same time")
                    
            # Process hits occurring at current collision time (tc)
            for hi in [h for h in hitlist if h[0] == tc]:
                bi = hi[1]  # Ball index from hit data
                
                # Remove from unprocessed balls list (bi_list)
                if bi in bi_list:
                    bi_list.remove(bi)
                
                lasthit_ball = bi
                lasthit_t = tc
                
                # Update ball0 data if needed
                if len(ball0[bi]['x']) > 1:
                    # Check distance to next point (5mm threshold)
                    dx = ball0[bi]['x'][1] - hi[4]
                    dy = ball0[bi]['y'][1] - hi[5]
                    checkdist = np.sqrt(dx**2 + dy**2) < 5
                    
                    # (Commented MATLAB code about deleting points omitted)
                
                # Update timeline and ball data
                Tall0[0] = tc  # Update first timeline element
                
                # Update current ball state
                ball0[bi]['t'][0] = tc
                ball0[bi]['x'][0] = hi[4]
                ball0[bi]['y'][0] = hi[5]
                
                # Add to ball history
                ball[bi]['t'].append(tc)
                ball[bi]['x'].append(hi[4])
                ball[bi]['y'].append(hi[5])
                
                # Update hit records
                hit[bi]['t'].append(hi[0])
                if hi[2] == 1:  # Ball-ball collision
                    # Assuming 'col' maps to ball identifiers (e.g. 'WYR')
                    hit[bi]['with'].append('WYR'[hi[3]])
                elif hit[2] == 2:  # Cushion collision
                    hit[bi]['with'].append(str(hi[3]))
                
                hit[bi]['XPos'].append(hi[4])
                hit[bi]['YPos'].append(hi[5])

                ps.plot_hit(hi[4], hi[5])  # Plot hit point for Ball 0


            for bi in bi_list:
                # Append current time to ball's history
                ball[bi]['t'].append(Tall0[0])
                
                # Create interpolators for position
                interp_x = interp1d(ball0[bi]['t'], ball0[bi]['x'], fill_value='extrapolate')
                interp_y = interp1d(ball0[bi]['t'], ball0[bi]['y'], fill_value='extrapolate')
                
                # Get interpolated position
                x_pos = float(interp_x(Tall0[0]))
                y_pos = float(interp_y(Tall0[0]))
                
                # Append positions to history
                ball[bi]['x'].append(x_pos)
                ball[bi]['y'].append(y_pos)
                
                # Update current state if velocity is positive
                if b[bi]['vt1'] > 0:
                    ball0[bi]['t'][0] = Tall0[0]
                    ball0[bi]['x'][0] = x_pos
                    ball0[bi]['y'][0] = y_pos
            

        else:

            # Remove first element from timeline
            Tall0 = Tall0[1:]

            for bi in range(3):  # 0-based ball indices (0,1,2)
                # Append new time to ball history
                ball[bi]['t'].append(Tall0[0])
                
                if b[bi]['vt1'] > 0:
                    # Interpolate position
                    interp_x = interp1d(ball0[bi]['t'], ball0[bi]['x'], fill_value='extrapolate')
                    interp_y = interp1d(ball0[bi]['t'], ball0[bi]['y'], fill_value='extrapolate')
                    ball[bi]['x'].append(float(interp_x(Tall0[0])))
                    ball[bi]['y'].append(float(interp_y(Tall0[0])))
                else:
                    # Use last position
                    ball[bi]['x'].append(ball[bi]['x'][-1])
                    ball[bi]['y'].append(ball[bi]['y'][-1])

                # Find indices of times before current timeline
                ind = np.where(ball0[bi]['t'] < Tall0[0])[0]
                
                if len(ball0[bi]['t']) > 1 and ball0[bi]['t'][1] <= Tall0[0]:
                    # Remove old data points
                    ball0[bi]['t'] = np.delete(ball0[bi]['t'], ind)
                    ball0[bi]['x'] = np.delete(ball0[bi]['x'], ind)
                    ball0[bi]['y'] = np.delete(ball0[bi]['y'], ind)
                else:
                    # Update current position
                    ball0[bi]['t'][0] = Tall0[0]
                    ball0[bi]['x'][0] = ball[bi]['x'][-1]
                    ball0[bi]['y'][0] = ball[bi]['y'][-1]

        # Update derivatives
        for bi in range(3):  # 0-based ball indices (0,1,2)
            if hit[bi]['with'][-1] != '-':
                # Sort by time
                ind = np.argsort(ball0[bi]['t'])
                ball0[bi]['t'] = ball0[bi]['t'][ind]
                ball0[bi]['x'] = ball0[bi]['x'][ind]
                ball0[bi]['y'] = ball0[bi]['y'][ind]
                
                # Calculate derivatives
                dt = np.diff(ball0[bi]['t'])
                vx = np.diff(ball0[bi]['x']) / dt
                vy = np.diff(ball0[bi]['y']) / dt
                v = np.sqrt(vx**2 + vy**2)
                
                # Append zero to maintain array length
                ball0[bi]['dt'] = np.concatenate([dt, [0]])
                ball0[bi]['vx'] = np.concatenate([vx, [0]])
                ball0[bi]['vy'] = np.concatenate([vy, [0]])
                ball0[bi]['v'] = np.concatenate([v, [0]])

        # Verify hit times in ball data
        for bi in range(3):
            for iii in range(len(hit[bi]['t'])):
                if hit[bi]['t'][iii] not in ball[bi]['t']:
                    print(f'No hit time found in ball {bi} data')
            
            # Find indices before current timeline
            ind = np.where(ball0[bi]['t'] < Tall0[0])[0]
            if ind.size > 0:
                pass  # Optional: Add handling code here
                # print(f'{bi}:{ind}')  # MATLAB-compatible output

        ps.update()
        do_scan = len(Tall0) >= 3

    ps.writer.finish()
    ps.close()

    # Assign processed ball data back to SA structure
    for bi in range(3):  # 0-based indexing for 3 balls
        SA['Shot'][si]['Route'][bi]['t'] = ball[bi]['t']
        SA['Shot'][si]['Route'][bi]['x'] = ball[bi]['x']
        SA['Shot'][si]['Route'][bi]['y'] = ball[bi]['y']