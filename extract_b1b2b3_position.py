from player_function import player_function
from str2num_b1b2b3 import str2num_b1b2b3

def extract_b1b2b3_position(SA, param):
    print(f"Start ({__name__})")
    for bi in range(0, 3):
        SA[f"B{bi+1}posX"] = None  # Initialize columns for ball positions
        SA[f"B{bi+1}posY"] = None

    for si in range(len(SA)):
        b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(SA.iloc[si]['B1B2B3'])

        for bi in range(0, 3):
            varname_x = f"B{bi+1}posX"
            SA.at[si, varname_x] = SA.iloc[si][f'Route{b1b2b3[bi]}']['x'][0] / param['size'][1] * 8

            varname_y = f"B{bi+1}posY"
            SA.at[si, varname_y] = SA.iloc[si][f'Route{b1b2b3[bi]}']['y'][0] / param['size'][0] * 4

        SA.at[si, 'ErrorID'] = None
        SA.at[si, 'ErrorText'] = None

    print(f"done ({__name__})")
    return SA
