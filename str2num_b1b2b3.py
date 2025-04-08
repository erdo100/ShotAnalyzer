
def str2num_b1b2b3(b1b2b3_str):
    base_string = 'WYR'
    b1b2b3_num = [] # To store the resulting indices

    b1b2b3_num = [base_string.index(char) for char in b1b2b3_str]

    # Unpack the list into individual variables
    b1i = b1b2b3_num[0]
    b2i = b1b2b3_num[1]
    b3i = b1b2b3_num[2]

    return b1b2b3_num, b1i, b2i, b3i

