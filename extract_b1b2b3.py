import numpy as np
from angle_vector import angle_vector

def extract_b1b2b3(shot):
    b1b2b3num = []
    err = {'code': None, 'text': ''}

    col = 'WYR'

    if len(shot['Route'][0]['t']) < 2 or len(shot['Route'][1]['t']) < 2 or len(shot['Route'][2]['t']) < 2:
        b1b2b3 = 'WYR'
        err['code'] = 2
        err['text'] = f"Empty Data ({__name__})"
        return b1b2b3, err

    t, b1b2b3num = zip(*sorted((shot['Route'][i]['t'][1], i) for i in range(3)))

    # initiate b1, b2, b3
    b1it, b2it, b3it = b1b2b3num

    if len(shot['Route'][b1it]['t']) >= 3:
        # find balls moving until 3rd time step of b1
        tb2i2 = (np.where(np.array(shot['Route'][b2it]['t'][1:]) <= shot['Route'][b1it]['t'][1])[0][-1:] or [None])[-1]
        tb3i2 = (np.where(np.array(shot['Route'][b3it]['t'][1:]) <= shot['Route'][b1it]['t'][1])[0][-1:] or [None])[-1]
        tb2i3 = (np.where(np.array(shot['Route'][b2it]['t'][1:]) <= shot['Route'][b1it]['t'][2])[0][-1:] or [None])[-1]
        tb3i3 = (np.where(np.array(shot['Route'][b3it]['t'][1:]) <= shot['Route'][b1it]['t'][2])[0][-1:] or [None])[-1]
        
        if not any([tb2i2, tb3i2, tb2i3, tb3i3]):
            # only b1 has moved for sure
            b1b2b3 = ''.join(col[i] for i in b1b2b3num)
            return b1b2b3, err
    else:
        # no ball has moved
        b1b2b3 = col
        return b1b2b3, err

    if any([tb2i2, tb2i3]) and not any([tb3i2, tb3i3]):
        vec_b1b2 = [shot['Route'][b2it]['x'][0] - shot['Route'][b1it]['x'][0],
                    shot['Route'][b2it]['y'][0] - shot['Route'][b1it]['y'][0]]
        vec_b1dir = [shot['Route'][b1it]['x'][1] - shot['Route'][b1it]['x'][0],
                     shot['Route'][b1it]['y'][1] - shot['Route'][b1it]['y'][0]]
        vec_b2dir = [shot['Route'][b2it]['x'][1] - shot['Route'][b2it]['x'][0],
                     shot['Route'][b2it]['y'][1] - shot['Route'][b2it]['y'][0]]

        angle_b1 = angle_vector(vec_b1b2, vec_b1dir)
        angle_b2 = angle_vector(vec_b1b2, vec_b2dir)

        if angle_b2 > 90:
            b1b2b3num = [1, 0, 2]

    if not any([tb2i2, tb2i3]) and any([tb3i2, tb3i3]):
        vec_b1b3 = [shot['Route'][b3it]['x'][0] - shot['Route'][b1it]['x'][0],
                    shot['Route'][b3it]['y'][0] - shot['Route'][b1it]['y'][0]]
        vec_b1dir = [shot['Route'][b1it]['x'][1] - shot['Route'][b1it]['x'][0],
                     shot['Route'][b1it]['y'][1] - shot['Route'][b1it]['y'][0]]
        vec_b3dir = [shot['Route'][b3it]['x'][1] - shot['Route'][b3it]['x'][0],
                     shot['Route'][b3it]['y'][1] - shot['Route'][b3it]['y'][0]]

        angle_b1 = angle_vector(vec_b1b3, vec_b1dir)
        angle_b3 = angle_vector(vec_b1b3, vec_b3dir)

        if angle_b3 > 90:
            b1b2b3num = [1, 2, 0]

    b1b2b3 = ''.join(col[i] for i in b1b2b3num)

    if any([tb2i2, tb2i3]) and any([tb3i2, tb3i3]):
        err['code'] = 2
        err['text'] = f"all balls moved at same time ({__name__})"

    return b1b2b3, err