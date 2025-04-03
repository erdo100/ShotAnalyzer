def eval_kiss(hit: dict, b1i: int, b2i: int, b3i: int) -> dict:
    """
    Evaluate kiss events and point/failure conditions for a shot
    
    Args:
        hit: Dictionary containing hit data for each ball
        b1i: Index of ball 1 (typically 0)
        b2i: Index of ball 2 (typically 1)
        b3i: Index of ball 3 (typically 2)
        
    Returns:
        Updated hit dictionary with kiss/point evaluation results
    """
    ballcolor = 'WYR'
    
    # Find hit indices
    b1b3i = [i for i, x in enumerate(hit[b1i]['with']) if x == ballcolor[b3i]]
    b1b2i = [i for i, x in enumerate(hit[b1i]['with']) if x == ballcolor[b2i]]
    b2b3i = [i for i, x in enumerate(hit[b2i]['with']) if x == ballcolor[b3i]]
    b1cushi = [i for i, x in enumerate(hit[b1i]['with']) if x in ['1', '2', '3', '4']]
    
    # Initialize timing variables
    failure = 0
    pointtime = 1000
    failtime = 1000
    kisstime = 1000
    b1b23C_time = 1000
    b1b2time = 1000
    
    # Check for Point condition
    if b1b3i and b1b2i and len(b1cushi) >= 3:
        if hit[b1i]['t'][b1b3i[0]] > hit[b1i]['t'][b1b2i[0]] and \
           hit[b1i]['t'][b1b3i[0]] > hit[b1i]['t'][b1cushi[2]]:  # 3rd cushion hit (0-based index 2)
            hit[b1i]['Point'] = 1
            pointtime = hit[b1i]['t'][b1b3i[0]]
    
    # Check for failure condition
    if b1b3i and b1b2i and len(b1cushi) >= 3:
        if hit[b1i]['t'][b1b3i[0]] > hit[b1i]['t'][b1b2i[0]] and \
           hit[b1i]['t'][b1b3i[0]] < hit[b1i]['t'][b1cushi[2]]:  # 3rd cushion hit (0-based index 2)
            failure = 1
            failtime = hit[b1i]['t'][b1b3i[0]]
    
    # Check for Kisses
    if hit[b1i].get('Point', 0) == 1:
        if b2b3i:  # Check for B2-B3 kiss
            if hit[b2i]['t'][b2b3i[0]] < pointtime and hit[b2i]['t'][b2b3i[0]] < kisstime:
                hit[b1i]['Kiss'] = 3
                hit[b1i]['Fuchs'] = 1
                kisstime = hit[b2i]['t'][b2b3i[0]]
        
        if len(b1b2i) >= 2:  # Check for B1-B2 kiss (second hit)
            if hit[b1i]['t'][b1b2i[1]] < pointtime and hit[b1i]['t'][b1b2i[1]] < kisstime:
                hit[b1i]['Kiss'] = 1
                hit[b1i]['Fuchs'] = 1
                kisstime = hit[b1i]['t'][b1b2i[1]]
    else:
        if failure == 0:
            if b2b3i and hit[b2i]['t'][b2b3i[0]] < kisstime:
                hit[b1i]['Kiss'] = 3
                kisstime = hit[b2i]['t'][b2b3i[0]]
            
            if len(b1b2i) >= 2 and hit[b1i]['t'][b1b2i[1]] < kisstime:
                hit[b1i]['Kiss'] = 1
                kisstime = hit[b1i]['t'][b1b2i[1]]
    
    # Calculate timing metrics
    if b1b2i and len(b1cushi) >= 3:
        b1b23C_time = max(hit[b1i]['t'][b1b2i[0]], hit[b1i]['t'][b1cushi[2]])  # 3rd cushion hit
    
    if b1b2i:
        b1b2time = hit[b1i]['t'][b1b2i[0]]
    
    # Update hit dictionary with timing information
    hit[b1i]['Tpoint'] = pointtime
    hit[b1i]['Tkiss'] = kisstime
    hit[b1i]['Tready'] = b1b23C_time
    hit[b1i]['TB2hit'] = b1b2time
    hit[b1i]['Tfailure'] = failtime
    
    return hit