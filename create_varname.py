def create_varname(SA, hit, si, param, str2num_B1B2B3, replace_colors_b1b2b3):
    """Create and update variable names in the SA table based on hit data."""
    # Initialize variables
    varparts = list(hit.keys())
    b1b2b3, b1i, b2i, b3i = str2num_B1B2B3(SA['Table']['B1B2B3'][si])
    
    # Update ball-with columns
    update_ball_with_columns(SA, hit, si, b1b2b3, replace_colors_b1b2b3)
    
    # Update main metrics
    update_main_metrics(SA, hit, si, b1i)
    
    # Update detailed hit metrics
    update_detailed_metrics(SA, hit, si, b1b2b3, varparts, param)
    
    return SA

def update_ball_with_columns(SA, hit, si, b1b2b3, replace_colors_b1b2b3):
    """Update the BX_with columns in the SA table."""
    for balli in range(1, 4):
        varname = f"B{balli}_with"
        if varname not in SA['Table']:
            SA['ColumnsVisible'].append(varname)
        with_new = replace_colors_b1b2b3(hit[b1b2b3[balli - 1]]['with'], b1b2b3)
        SA['Table'][varname][si] = with_new

def update_main_metrics(SA, hit, si, b1i):
    """Update the main metrics in the SA table."""
    main_metrics = {
        'Point0': hit[b1i]['Point'],
        'Kiss': hit[b1i]['Kiss'],
        'Fuchs': hit[b1i]['Fuchs'],
        'PointDist': hit[b1i]['PointDist'],
        'KissDistB1': hit[b1i]['KissDistB1'],
        'AimOffsetSS': hit[b1i]['AimOffsetSS'],
        'AimOffsetLL': hit[b1i]['AimOffsetLL'],
        'B1B2OffsetSS': hit[b1i]['B1B2OffsetSS'],
        'B1B2OffsetLL': hit[b1i]['B1B2OffsetLL'],
        'B1B3OffsetSS': hit[b1i]['B1B3OffsetSS'],
        'B1B3OffsetLL': hit[b1i]['B1B3OffsetLL']
    }
    
    for metric, value in main_metrics.items():
        SA['Table'][metric][si] = value

def update_detailed_metrics(SA, hit, si, b1b2b3, varparts, param):
    """Update detailed hit metrics in the SA table."""
    delname = ['Fraction', 'DefAngle', 'CutAngle', 'CInAngle', 'COutAngle']
    himax = [8, 4, 0]
    
    for bi in range(1, 4):
        for hi in range(1, himax[bi - 1] + 1):
            for vi in range(20, len(varparts)):
                varname = f"B{bi}_{hi}_{varparts[vi]}"
                
                if hi == 1 and varparts[vi] in delname:
                    continue
                
                if len(hit[b1b2b3[bi - 1]][varparts[vi]]) >= hi:
                    scale = param['size'][1] / 8 if 'Pos' in varparts[vi] else 1
                    SA['Table'][varname][si] = hit[b1b2b3[bi - 1]][varparts[vi]][hi - 1] / scale