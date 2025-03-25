from identify_ShotId import identify_shot_id
from update_shot_list import update_shot_list
from player_function import player_function

def extract_b1b2b3_position(SA, param, player):
    print('Start identifying B1B2B3 Positions ...')

    hsl_data = SA['Table']['Data']
    hsl_columns = SA['Table']['ColumnName']

    for ti, row in enumerate(hsl_data):
        identify_shot_id(ti, hsl_data, hsl_columns, SA)
        si = SA['Current_si']

        if SA['Table']['Interpreted'][si] == 0:
            b1b2b3, b1i, b2i, b3i = str2num_B1B2B3(SA['Table']['B1B2B3'][si])

            for bi in range(1, 4):
                varname_x = f"B{bi}posX"
                SA['Table'][varname_x][si] = SA['Shot'][si]['Route'][b1b2b3[bi - 1]]['x'][0] / param['size'][1] * 8

                varname_y = f"B{bi}posY"
                SA['Table'][varname_y][si] = SA['Shot'][si]['Route'][b1b2b3[bi - 1]]['y'][0] / param['size'][0] * 4

                SA['Table']['ErrorID'][si] = None
                SA['Table']['ErrorText'][si] = None

    update_shot_list(SA)
    player['uptodate'] = False
    player_function('plotcurrent', player)

    print(f"done ({__name__})")