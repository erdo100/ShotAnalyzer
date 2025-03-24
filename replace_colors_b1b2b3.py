def replace_colors_b1b2b3(with_str, b1b2b3):
    with_new = with_str.replace('W', 'V').replace('Y', 'Z').replace('R', 'T')

    if b1b2b3[0] == 1:  # B1 is white
        with_new = with_new.replace('V', 'W')
    if b1b2b3[1] == 1:  # B2 is white
        with_new = with_new.replace('V', 'Y')
    if b1b2b3[2] == 1:  # B3 is white
        with_new = with_new.replace('V', 'R')

    if b1b2b3[0] == 2:  # B1 is yellow
        with_new = with_new.replace('Z', 'W')
    if b1b2b3[1] == 2:
        with_new = with_new.replace('Z', 'Y')
    if b1b2b3[2] == 2:
        with_new = with_new.replace('Z', 'R')

    if b1b2b3[0] == 3:  # B1 is red
        with_new = with_new.replace('T', 'W')
    if b1b2b3[1] == 3:
        with_new = with_new.replace('T', 'Y')
    if b1b2b3[2] == 3:
        with_new = with_new.replace('T', 'R')

    return with_new