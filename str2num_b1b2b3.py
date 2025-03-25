def str2num_b1b2b3(b1b2b3):
    """
    Converts a string representation of B1B2B3 (e.g., 'WYR')
    into numerical indices for each ball.

    Args:
        b1b2b3 (str): A string containing 'W', 'Y', and 'R' in any order.

    Returns:
        tuple: A tuple containing the numerical indices of B1, B2, and B3.
    """
    mapping = {'W': 1, 'Y': 2, 'R': 3}
    b1b2b3_indices = [mapping[char] for char in b1b2b3]
    return tuple(b1b2b3_indices)