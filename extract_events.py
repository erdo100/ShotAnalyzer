import numpy as np
from str2num_b1b2b3 import str2num_b1b2b3
from angle_vector import angle_vector

def extract_events(SA, si, param):
    err = {'code': None, 'text': ''}

    b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(SA['B1B2B3'][si])


    # Get the copy from original data
    ball0 = [{} for _ in range(3)]  # Initialize ball0 as a list of empty dictionaries
    for bi in range(3):
        ball0[bi] = SA['Shot'][si]['Route0'][bi]
        SA['Shot'][si]['Route'][bi] = {
            't': SA['Shot'][si]['Route0'][bi]['t'],
            'x': SA['Shot'][si]['Route0'][bi]['x'],
            'y': SA['Shot'][si]['Route0'][bi]['y']
        }

    ball = []
    for bi in range(3):
        ball.append({
            'x': [SA['Shot'][si]['Route0'][bi]['x'][0]],
            'y': [SA['Shot'][si]['Route0'][bi]['y'][0]],
            't': [0]
        })

    # Create common Time points
    Tall0 = np.unique(np.concatenate([ball0[0]['t'], ball0[1]['t'], ball0[2]['t']]))
    ti = 0

    # Discretization of dT
    tvec = np.linspace(0.01, 1, 101)

    # Scanning all balls
    hit = {
        b1i: {'with': ['S'], 't': [0], 'XPos': [ball0[b1i]['x'][0]], 'YPos': [ball0[b1i]['y'][0]], 'Kiss': 0, 'Point': 0, 'Fuchs': 0, 'PointDist': 3000, 'KissDistB1': 3000, 'Tpoint': 3000, 'Tkiss': 3000, 'Tready': 3000, 'TB2hit': 3000, 'Tfailure': 3000},
        b2i: {'with': ['-'], 't': [0], 'XPos': [ball0[b2i]['x'][0]], 'YPos': [ball0[b2i]['y'][0]]},
        b3i: {'with': ['-'], 't': [0], 'XPos': [ball0[b3i]['x'][0]], 'YPos': [ball0[b3i]['y'][0]]}
    }

    '''
    Targets
     - Make valid Ball routes which resolve every event exactly
     - Smoothen the trajectories, accelerations

    Idea
    make simulation/extrapolation for next time steps

    But what is the velicity of the balls. Very initially, B1 is moving, B2,B3 are not moving.
    How big is V1 velocity? Where can we find indications?

    Investigate forward
    integrate from current position/velocity and check distance

    Investigate backward
    use velocities after the collision and estimate previous condition
    check energy/momentum of sum of balls

    Case 1: No collision in ti=2
     - Ball distance is big in first time steps
     - Ball distance is small in first time steps

    Case 2: Collision in ti=2
     - Ball distance is big in first time steps
     - Ball distance is small in first time steps

    B2 velocity is 0 in ti=2
    '''

    # Initialize b as a list of dictionaries for each ball
    b = [{} for _ in range(3)]

    # ball velocities
    for bi in range(3):
        ball0[bi]['dt'] = np.diff(ball0[bi]['t'], append=ball0[bi]['t'][-1])
        ball0[bi]['vx'] = np.diff(ball0[bi]['x'], append=0) / ball0[bi]['dt']
        ball0[bi]['vy'] = np.diff(ball0[bi]['y'], append=0) / ball0[bi]['dt']
        ball0[bi]['v'] = np.sqrt(ball0[bi]['vx']**2 + ball0[bi]['vy']**2)

    ball0[b2i]['vx'][0] = ball0[b2i]['vy'][0] = ball0[b2i]['v'][0] = 0
    ball0[b3i]['vx'][0] = ball0[b3i]['vy'][0] = ball0[b3i]['v'][0] = 0

    
    do_scan = True

    # ball0 shall contain only remaining future data
    # ball1 shall contain only proccessed past data


    # Initialize hit list and other variables
    hitlist = []
    lasthit_ball = None
    lasthit_t = None

    # Initialize d as an empty list
    d = []

    while do_scan:

        ti += 1
        print(f"Time step {ti}")
        dT = np.diff(Tall0[:2])[0]
        tappr = Tall0[0] + dT * np.linspace(0.01, 1, 101)

        for bi in range(3):
            # Check if it is last index
            if len(ball0[bi]['t']) >= 2:

                # Current speed
                b[bi]['vt1'] = ball0[bi]['v'][0]

                # Was the last hit in the last time step?
                iprev = np.where(ball[bi]['t'] >= [hit[bi]['t'][-1]])[0][-1] if len(hit[bi]['t']) > 0 else None
                
                # Speed Components in the next time interval
                if hit[bi]['with'] == '-':
                    # Ball was not moving previously
                    # hit[bi]['with'] has only 1 element = '-'
                    # Ball 2 and 3 with ti = 1, ball is not moving ==> v=0
                    b[bi]['v1'] = [0, 0]
                    b[bi]['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
                elif iprev is not None:
                    # When last hit was on this ball
                    # Current X-Y Velocity components
                    b[bi]['v1'] = [ball0[bi]['vx'][0], ball0[bi]['vy'][0]]
                    b[bi]['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]
                else:
                    # Extrapolate from last points
                    CoefVX = np.polyfit(ball[bi]['t'][iprev], ball[bi]['x'][iprev], 1)
                    CoefVY = np.polyfit(ball[bi]['t'][iprev], ball[bi]['y'][iprev], 1)
                    b[bi]['v1'] = [CoefVX[0], CoefVY[0]]
                    b[bi]['v2'] = [ball0[bi]['vx'][1], ball0[bi]['vy'][1]]


                # Speed in data for next time interval
                b[bi]['vt1'] = np.linalg.norm(b[bi]['v1'])
                b[bi]['vt2'] = np.linalg.norm(b[bi]['v2'])

                vtnext = max(b[bi]['vt1'], ball0[bi]['v'][0])

                if np.linalg.norm(b[bi]['v1']) > 0:
                    vnext = (b[bi]['v1'] / np.linalg.norm(b[bi]['v1'])) * vtnext
                else:
                    vnext = [0, 0]

                # Approximation of next time interval
                b[bi]['xa'] = ball[bi]['x'][-1] + vnext[0] * dT * tvec
                b[bi]['ya'] = ball[bi]['y'][-1] + vnext[1] * dT * tvec






        # Calculate Ball trajectory angle change
        for bi in range(3):
            # Angle of trajectory in data for next time interval
            b[bi]['a12'] = angle_vector(b[bi]['v1'], b[bi]['v2'])
        
        # Calculate the Ball-Ball Distance
        # List of Ball-Ball collisions
        BB = [[1, 2], [1, 3], [2, 3]]
        
        # Get B1, B2, B3 from ShotList
        b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(SA['B1B2B3'][si])


        for bbi in range(3):
            bx1 = b1b2b3[BB[bbi][0] - 1]
            bx2 = b1b2b3[BB[bbi][1] - 1]
            
            d_BB = np.sqrt((b[bx1]['xa'] - b[bx2]['xa'])**2 + 
                            (b[bx1]['ya'] - b[bx2]['ya'])**2) - 2 * param['ballR']
            d.append({'BB': d_BB})
        
        # Calculate Cushion distance
        for bi in range(3):
            b[bi]['cd'] = [
                b[bi]['ya'] - param['ballR'],
                param['size'][1] - param['ballR'] - b[bi]['xa'],
                param['size'][0] - param['ballR'] - b[bi]['ya'],
                b[bi]['xa'] - param['ballR']
            ]
        
        hitlist = []
        lasthitball = None
            
        # Evaluate cushion hit
        # Criteria for cushion hit:
        #  - Ball was moving in Cushion direction
        #  - Ball coordinate is close to cushion
        #  - Ball is changing direction
        for bi in range(3):
            for cii in range(4):
                checkdist = any(b[bi]['cd'][cii] <= 0)
                checkangle = b[bi]['a12'] > 1 or b[bi]['a12'] == -1
                velx = b[bi]['v1'][0]
                vely = b[bi]['v1'][1]
                
                checkcush = False
                tc = 0
                if checkdist and checkangle and vely < 0 and cii == 0 and b[bi]['v1'][1] != 0:  # bottom cushion
                    checkcush = True
                    tc = np.interp(0, b[bi]['cd'][cii], tappr)  # Exact time of contact
                    cushx = np.interp(tc, tappr, b[bi]['xa'])
                    cushy = param['ballR']
                elif checkdist and checkangle and velx > 0 and cii == 1 and b[bi]['v1'][0] != 0:  # right cushion
                    checkcush = True
                    tc = np.interp(0, b[bi]['cd'][cii], tappr)  # Exact time of contact
                    cushx = param['size'][1] - param['ballR']
                    cushy = np.interp(tc, tappr, b[bi]['ya'])
                elif checkdist and checkangle and vely > 0 and cii == 2 and b[bi]['v1'][1] != 0:  # top cushion
                    checkcush = True
                    tc = np.interp(0, b[bi]['cd'][cii], tappr)  # Exact time of contact
                    cushy = param['size'][0] - param['ballR']
                    cushx = np.interp(tc, tappr, b[bi]['xa'])
                elif checkdist and checkangle and velx < 0 and cii == 3 and b[bi]['v1'][0] != 0:  # left cushion
                    checkcush = True
                    tc = np.interp(0, b[bi]['cd'][cii], tappr)  # Exact time of contact
                    cushx = param['ballR']
                    cushy = np.interp(tc, tappr, b[bi]['ya'])
                
                if checkcush:
                    hitlist.append([tc, bi, 2, cii, cushx, cushy])  # Append hit details


        
        
        # Evaluate Ball-Ball hit
        for bbi in range(3):
            bx1 = b1b2b3[BB[bbi][0] - 1]
            bx2 = b1b2b3[BB[bbi][1] - 1]
            
            # Check whether distance has values smaller and bigger than 0
            checkdist = False
            if (any(d[bbi]['BB'] <= 0) and any(d[bbi]['BB'] > 0) and
                ((d[bbi]['BB'][0] >= 0 and d[bbi]['BB'][-1] < 0) or
                (d[bbi]['BB'][0] < 0 and d[bbi]['BB'][-1] >= 0))):
                checkdist = True
                tc = np.interp(0, d[bbi]['BB'], tappr)
            elif any(d[bbi]['BB'] <= 0) and any(d[bbi]['BB'] > 0):
                checkdist = True
                ind = np.diff(d[bbi]['BB']) <= 0
                tc = np.interp(0, d[bbi]['BB'][ind], tappr[ind])
            
            checkangleb1 = b[bx1]['a12'] > 10 or b[bx1]['a12'] == -1
            checkangleb2 = b[bx2]['a12'] > 10 or b[bx2]['a12'] == -1
            
            checkvelb1 = abs(b[bx1]['vt2'] - b[bx1]['vt1']) > 50 if abs(b[bx1]['vt1']) > 0 or abs(b[bx1]['vt2']) > 0 else False
            checkvelb2 = abs(b[bx2]['vt2'] - b[bx2]['vt1']) > 50 if abs(b[bx2]['vt1']) > 0 or abs(b[bx2]['vt2']) > 0 else False
            
            checkdouble = tc >= hit[bx1]['t'][-1] + 0.01 and tc >= hit[bx2]['t'][-1] + 0.01
            
            if checkdouble and checkdist:
                hitlist.append([tc, bx1, 1, bx2, np.interp(tc, tappr, b[bx1]['xa']), np.interp(tc, tappr, b[bx1]['ya'])])
                hitlist.append([tc, bx2, 1, bx1, np.interp(tc, tappr, b[bx2]['xa']), np.interp(tc, tappr, b[bx2]['ya'])])
        

            
        # When just before the Ball-Ball hit velocity is too small, then hit can be missed
        # Therefore, we check whether B2 is moving without a hit
        # Now only for the first hit, then we have to calculate the past hit
        # Move back B1 so that B1 is touching B2

        if ti > 0 and len(hit[b2i]['t']) == 1 and not hitlist and ball0[b1i]['t'][1] >= ball0[b2i]['t'][1] and \
                (ball0[b2i]['x'][0] != ball0[b2i]['x'][1] or ball0[b2i]['y'][0] != ball0[b2i]['y'][1]):

            # Determine which bbi corresponds to b1i hitting b2i
            # First is B1, second is B2
            bbi = next((i for i, pair in enumerate(BB) if pair[0] == 1 and pair[1] == 2), None)

            if bbi is not None:
                tc = np.interp(0, d[bbi]['BB'], tappr)
                if tc < 0:
                    ti = np.argmin(d[bbi]['BB'])
                    tc = tappr[ti]

                hitlist.append({
                    'time': tc,  # Contact Time
                    'ball1': b1i,  # Ball 1
                    'type': 1,  # 1 = Ball-Ball, 2 = Cushion
                    'ball2': b2i,  # Ball 2
                    'contact_x': np.interp(tc, tappr, b[b1i]['xa']),
                    'contact_y': np.interp(tc, tappr, b[b1i]['ya'])
                })

                hitlist.append({
                    'time': tc,  # Contact Time
                    'ball1': b2i,  # Ball 1
                    'type': 1,  # 1 = Ball-Ball, 2 = Cushion
                    'ball2': b1i,  # Ball 2
                    'contact_x': ball0[b2i]['x'][0],  # Contact location x
                    'contact_y': ball0[b2i]['y'][0]  # Contact location y
                })


        # Check first Time step for Ball-Ball hit
        # If Balls are too close, so that the direction change is not visible,
        # then the algorithm can't detect the Ball-Ball hit. Therefore:
        # Check whether B2 is moving without a hit
        # If yes, then use the direction of B2, calculate the perpendicular
        # direction and assign to B1
        if (
            ti == 0
            and not hitlist
            and ball0[b1i]["t"][1] == ball0[b2i]["t"][1]
            and (
                ball0[b2i]["x"][0] != ball0[b2i]["x"][1]
                or ball0[b2i]["y"][0] != ball0[b2i]["y"][1]
            )
        ):
            vec_b2dir = [
                ball0[b2i]["x"][1] - ball0[b2i]["x"][0],
                ball0[b2i]["y"][1] - ball0[b2i]["y"][0],
            ]  # Direction vector of B1, this is reference

            # Contact position of B1. This is in tangential direction of B2
            # movement direction
            b1pos2 = [
                ball0[b2i]["x"][0],
                ball0[b2i]["y"][0],
            ] - np.array(vec_b2dir) / np.linalg.norm(vec_b2dir) * param["ballR"] * 2

            hitlist.append(
                {
                    "time": ball0[b1i]["t"][1] / 2,  # Contact Time
                    "ball1": b1i,  # Ball 1
                    "type": 1,  # 1= Ball-Ball, 2= Cushion
                    "ball2": b2i,  # Ball 2
                    "contact_x": b1pos2[0],
                    "contact_y": b1pos2[1],
                }
            )

            hitlist.append(
                {
                    "time": ball0[b1i]["t"][1] / 2,  # Contact Time
                    "ball1": b2i,  # Ball 2
                    "type": 1,  # 1= Ball-Ball, 2= Cushion
                    "ball2": b1i,  # Ball 1
                    "contact_x": ball0[b2i]["x"][0],  # Contact location x
                    "contact_y": ball0[b2i]["y"][0],  # Contact location y
                }
            )

        # Assign new hit event or next timestep into the ball route history
        bi_list = list(range(3))  # To store which ball didn't have a hit, so must set new time manually
        if hitlist and Tall0[1] >= tc:
            # Get the first time when a hit is detected
            tc = min(hit["time"] for hit in hitlist)
            if Tall0[1] < tc:
                print("Warning: hit is after next time step")

            # Check whether for current ball more than 1 event has happened
            for bi in range(3):
                if tc in hitlist[0] and bi in hitlist[1]:
                    raise ValueError(f"ERROR: Ball {bi} has multiple hits at the same time.")

            # Assign new data for hits
            for hi, hit_event in enumerate(hitlist):
                if hit_event['time'] == tc:
                    bi = hit_event['ball1']
                    bi_list.remove(bi)
                    lasthit_ball = bi
                    lasthit_t = tc

                    # Update ball0 and ball with hit data
                    Tall0[0] = tc
                    ball0[bi]['t'][0] = tc
                    ball0[bi]['x'][0] = hit_event['contact_x']
                    ball0[bi]['y'][0] = hit_event['contact_y']

                    ball[bi]['t'].append(tc)
                    ball[bi]['x'].append(hit_event['contact_x'])
                    ball[bi]['y'].append(hit_event['contact_y'])

                    # Update hit data
                    hit[bi]['t'].append(tc)
                    if hit_event['type'] == 1:  # Ball-Ball collision
                        hit[bi]['with'].append(hit_event['type'])
                    elif hit_event['type'] == 2:  # Cushion collision
                        hit[bi]['with'].append(str(hit_event['type']))
                    hit[bi]['XPos'].append(hit_event['contact_x'])
                    hit[bi]['YPos'].append(hit_event['contact_y'])

            # Assign time to balls without hit events
            for bi in bi_list:
                ball[bi]['t'].append(Tall0[0])
                ball[bi]['x'].append(np.interp(Tall0[0], ball0[bi]['t'], ball0[bi]['x']))
                ball[bi]['y'].append(np.interp(Tall0[0], ball0[bi]['t'], ball0[bi]['y']))

                if b[bi]['vt1'] > 0:
                    ball0[bi]['t'][0] = Tall0[0]
                    ball0[bi]['x'][0] = ball[bi]['x'][-1]
                    ball0[bi]['y'][0] = ball[bi]['y'][-1]
        else:
            # Assign new data for no ball hit
            Tall0 = Tall0[1:]
            for bi in range(3):
                ball[bi]['t'].append(Tall0[0])
                if b[bi]['vt1'] > 0:
                    ball[bi]['x'].append(np.interp(Tall0[0], ball0[bi]['t'], ball0[bi]['x']))
                    ball[bi]['y'].append(np.interp(Tall0[0], ball0[bi]['t'], ball0[bi]['y']))
                else:
                    ball[bi]['x'].append(ball[bi]['x'][-1])
                    ball[bi]['y'].append(ball[bi]['y'][-1])

                if ball0[bi]['t'][1] <= Tall0[0]:
                    ball0[bi]['t'] = ball0[bi]['t'][1:]
                    ball0[bi]['x'] = ball0[bi]['x'][1:]
                    ball0[bi]['y'] = ball0[bi]['y'][1:]
                else:
                    ball0[bi]['t'][0] = Tall0[0]
                    ball0[bi]['x'][0] = ball[bi]['x'][-1]
                    ball0[bi]['y'][0] = ball[bi]['y'][-1]


            
        # Update derivatives
        for bi in range(3):
            if hit[bi]['with'][-1] != '-':
                # First sort by time
                sorted_indices = np.argsort(ball0[bi]['t'])
                ball0[bi]['t'] = np.array(ball0[bi]['t'])[sorted_indices]
                ball0[bi]['x'] = np.array(ball0[bi]['x'])[sorted_indices]
                ball0[bi]['y'] = np.array(ball0[bi]['y'])[sorted_indices]
                
                ball0[bi]['dt'] = np.diff(ball0[bi]['t'], append=ball0[bi]['t'][-1])
                ball0[bi]['vx'] = np.diff(ball0[bi]['x'], append=0) / ball0[bi]['dt']
                ball0[bi]['vy'] = np.diff(ball0[bi]['y'], append=0) / ball0[bi]['dt']
                ball0[bi]['v'] = np.sqrt(ball0[bi]['vx']**2 + ball0[bi]['vy']**2)
        
        
        for bi in range(3):
            for iii in range(len(hit[bi]['t'])):
                if not any(hit[bi]['t'][iii] == t for t in ball[bi]['t']):
                    print('no hit time found in ball data')
            
            ind = [i for i, t in enumerate(ball0[bi]['t']) if t < Tall0[0]]
            if ind:
                # Uncomment the following line to display debug information
                # print(f"{bi}:{ind}")
                pass
        
        # Check whether time is over
        do_scan = len(Tall0) >= 3
        print(f"do_scan: {len(Tall0)}")


    # Assign back the shot
    for bi in range(3):
        SA['Shot'][si]['Route'][bi]['t'] = ball[bi]['t']
        SA['Shot'][si]['Route'][bi]['x'] = ball[bi]['x']
        SA['Shot'][si]['Route'][bi]['y'] = ball[bi]['y']
