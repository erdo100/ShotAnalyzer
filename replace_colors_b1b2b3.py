def replace_colors_b1b2b3(with_str: str, b1b2b3: list) -> str:
    """
    Replace ball color codes (W/Y/R) with B1/B2/B3 notation based on ball order
    
    Args:
        with_str: String containing original color codes (W/Y/R)
        b1b2b3: List indicating ball order [B1_color, B2_color, B3_color]
               where 1=White(W), 2=Yellow(Y), 3=Red(R)
    
    Returns:
        String with color codes replaced by B1/B2/B3 notation
    """
    # First replace colors with temporary placeholders
    with_new = with_str.replace('W', 'V').replace('Y', 'Z').replace('R', 'T')
    
    # Replace placeholders based on actual ball order
    if b1b2b3[0] == 1:  # B1 is white (W)
        with_new = with_new.replace('V', 'W')
    if b1b2b3[1] == 1:  # B2 is white (W)
        with_new = with_new.replace('V', 'Y')
    if b1b2b3[2] == 1:  # B3 is white (W)
        with_new = with_new.replace('V', 'R')
        
    if b1b2b3[0] == 2:  # B1 is yellow (Y)
        with_new = with_new.replace('Z', 'W')
    if b1b2b3[1] == 2:  # B2 is yellow (Y)
        with_new = with_new.replace('Z', 'Y')
    if b1b2b3[2] == 2:  # B3 is yellow (Y)
        with_new = with_new.replace('Z', 'R')
        
    if b1b2b3[0] == 3:  # B1 is red (R)
        with_new = with_new.replace('T', 'W')
    if b1b2b3[1] == 3:  # B2 is red (R)
        with_new = with_new.replace('T', 'Y')
    if b1b2b3[2] == 3:  # B3 is red (R)
        with_new = with_new.replace('T', 'R')
        
    return with_new