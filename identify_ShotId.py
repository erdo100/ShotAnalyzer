import numpy as np

def identify_shot_id(ti, data, column_name, SA):
    col_si = column_name.index('ShotID')
    col_mi = column_name.index('Mirrored')

    shot_id = data[ti][col_si]
    mirrored = data[ti][col_mi]

    combined = np.array(SA['Table']['ShotID']) + np.array(SA['Table']['Mirrored']) / 10
    target = shot_id + mirrored / 10

    si = np.where(combined == target)[0][0]

    SA['Current_ti'] = ti
    SA['Current_ShotID'] = SA['Table']['ShotID'][si]
    SA['Current_si'] = si