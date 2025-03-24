def eval_kiss(hit, b1i, b2i, b3i):
    ballcolor = 'WYR'

    b1b3i = [i for i, w in enumerate(hit[b1i]['with']) if w == ballcolor[b3i]]
    b1b2i = [i for i, w in enumerate(hit[b1i]['with']) if w == ballcolor[b2i]]
    b2b3i = [i for i, w in enumerate(hit[b2i]['with']) if w == ballcolor[b3i]]
    b1cushi = [i for i, w in enumerate(hit[b1i]['with']) if w in '1234']

    failure = 0
    pointtime = 1000
    failtime = 1000
    kisstime = 1000
    b1b23C_time = 1000
    b1b2time = 1000

    if b1b3i and b1b2i and len(b1cushi) >= 3:
        if hit[b1i]['t'][b1b3i[0]] > hit[b1i]['t'][b1b2i[0]] and hit[b1i]['t'][b1b3i[0]] > hit[b1i]['t'][b1cushi[2]]:
            hit[b1i]['Point'] = 1
            pointtime = hit[b1i]['t'][b1b3i[0]]

    if b1b3i and b1b2i and len(b1cushi) >= 3:
        if hit[b1i]['t'][b1b3i[0]] > hit[b1i]['t'][b1b2i[0]] and hit[b1i]['t'][b1b3i[0]] < hit[b1i]['t'][b1cushi[2]]:
            failure = 1
            failtime = hit[b1i]['t'][b1b3i[0]]

    if hit[b1i]['Point'] == 1:
        if b2b3i and hit[b2i]['t'][b2b3i[0]] < pointtime:
            if kisstime > hit[b2i]['t'][b2b3i[0]]:
                hit[b1i]['Kiss'] = 3
                hit[b1i]['Fuchs'] = 1
                kisstime = hit[b2i]['t'][b2b3i[0]]
        if len(b1b2i) >= 2 and hit[b1i]['t'][b1b2i[1]] < pointtime:
            if kisstime > hit[b1i]['t'][b1b2i[1]]:
                hit[b1i]['Kiss'] = 1
                hit[b1i]['Fuchs'] = 1
                kisstime = hit[b1i]['t'][b1b2i[1]]
    else:
        if failure == 0:
            if b2b3i and kisstime > hit[b2i]['t'][b2b3i[0]]:
                hit[b1i]['Kiss'] = 3
                kisstime = hit[b2i]['t'][b2b3i[0]]
            if len(b1b2i) >= 2 and kisstime > hit[b1i]['t'][b1b2i[1]]:
                hit[b1i]['Kiss'] = 1
                kisstime = hit[b1i]['t'][b1b2i[1]]

    if b1b2i and len(b1cushi) >= 3:
        b1b23C_time = max(hit[b1i]['t'][b1b2i[0]], hit[b1i]['t'][b1cushi[2]])

    if b1b2i:
        b1b2time = hit[b1i]['t'][b1b2i[0]]

    hit[b1i]['Tpoint'] = pointtime
    hit[b1i]['Tkiss'] = kisstime
    hit[b1i]['Tready'] = b1b23C_time
    hit[b1i]['TB2hit'] = b1b2time
    hit[b1i]['Tfailure'] = failtime

    return hit