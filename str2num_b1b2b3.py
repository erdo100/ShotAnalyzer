def str2num_B1B2B3(B1B2B3_str: str) -> tuple:
    """
    Convert B1B2B3 string to numerical indices matching MATLAB's str2num_B1B2B3
    
    Args:
        B1B2B3_str: 3-character string containing ball order (e.g., 'WYR')
        
    Returns:
        tuple: (b1b2b3, b1i, b2i, b3i)
        where:
        - b1b2b3: list of indices [b1, b2, b3]
        - b1i, b2i, b3i: individual indices
    """
    # Define the reference order
    ref_order = 'WYR'
    
    # Convert each character to its position in 'WYR' (1-based in MATLAB, 0-based here)
    b1b2b3 = [ref_order.index(B1B2B3_str[0]),
              ref_order.index(B1B2B3_str[1]),
              ref_order.index(B1B2B3_str[2])]
    
    # Return both the list and individual components
    return b1b2b3, b1b2b3[0], b1b2b3[1], b1b2b3[2]