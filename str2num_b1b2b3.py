def str2num_b1b2b3(b1b2b3):
    """
    Converts a string representation of B1B2B3 (e.g., 'WYR')
    into numerical indices for each ball.

    Args:
        b1b2b3 (str): A string containing 'W', 'Y', and 'R' in any order.

    Returns:
        tuple: A tuple containing the numerical indices of B1, B2, and B3.
    """
    mapping = {'W': 0, 'Y': 1, 'R': 2}

    b1i = mapping[b1b2b3[0]]
    b2i = mapping[b1b2b3[1]]
    b3i = mapping[b1b2b3[2]]

    b1b2b3_indices = [b1i, b2i, b3i]
    
    return b1b2b3_indices, b1i, b2i, b3i