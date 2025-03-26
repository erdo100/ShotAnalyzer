from player_function import player_function
from str2num_b1b2b3 import str2num_b1b2b3

def extract_b1b2b3_position(SA, param):
    print('Start identifying B1B2B3 Positions ...')

    for si in range(len(SA['Shot'])):
        b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(SA['B1B2B3'][si])

        for bi in range(1, 4):
            varname_x = f"B{bi}posX"
            SA[varname_x][si] = SA['Shot'][si]['Route'][b1b2b3[bi - 1]]['x'][0] / param['size'][1] * 8

            varname_y = f"B{bi}posY"
            SA[varname_y][si] = SA['Shot'][si]['Route'][b1b2b3[bi - 1]]['y'][0] / param['size'][0] * 4

            SA['ErrorID'][si] = None
            SA['ErrorText'][si] = None

    print(f"done ({__name__})")