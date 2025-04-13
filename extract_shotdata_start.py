import numpy as np

from read_gamefile import read_gamefile
from extract_b1b2b3_start import extract_b1b2b3_start
from extract_events_start import extract_events_start
from extract_dataquality_start import extract_dataquality_start


# Main execution function (similar to the original script)
def extract_shotdata_start(filepath):
    """
    Extract shot data by sequentially calling the required functions.

    Args:
        filepath (str): Path to the game file.
    """

    # Table properties (adjusted for Python syntax)
    param = {
        "ver": "Shot Analyzer v0.43i_Python",
        # Assuming [height, width] based on typical usage, but MATLAB code suggested [Y, X] -> [1420, 2840]
        # Let's stick to MATLAB's apparent [height, width] based on index usage (size(1)=y, size(2)=x)
        "size": [1420, 2840],
        "ballR": 61.5 / 2,
        # 'ballcirc' not directly used in translated functions, kept for reference
        "ballcirc": {
            "x": np.sin(np.linspace(0, 2 * np.pi, 36)) * (61.5 / 2),
            "y": np.cos(np.linspace(0, 2 * np.pi, 36)) * (61.5 / 2),
        },
        "rdiam": 7,
        "cushionwidth": 50,
        "diamdist": 97,
        "framewidth": 147,
        "colors": "wyr",
        "BallOutOfTableDetectionLimit": 30, # Used in commented MATLAB code
        "BallCushionHitDetectionRange": 50, # Not used in translated code
        "BallProjecttoCushionLimit": 10,   # Not used in translated code (clipping used instead)
        "NoDataDistanceDetectionLimit": 600, # Used
        "MaxTravelDistancePerDt": 0.1,      # Not used in translated code
        "MaxVelocity": 12000,               # Used
        "timax_appr": 5,                    # Not used in translated code
    }

    print("Starting shot data extraction...")

    # Step 1: Read game file
    print("Reading game file...")
    SA = read_gamefile(filepath)

    if SA is None:
        print("Failed to read game file or no data found. Aborting.")
        return

    # Step 2: Extract data quality
    print("Extracting data quality...")
    extract_dataquality_start(SA, param)

    # Step 3: Extract B1B2B3 start (Placeholder)
    print("Extracting B1B2B3 start...")
    extract_b1b2b3_start(SA, param)

    # Step 4: Extract events (Placeholder)
    print("Extracting events...")
    extract_events_start(SA, param)

    print("Shot data extraction process completed.")

    return SA

    # You can now access the processed data in SA
    # Example: print(SA['Table'].head())
    # Example: print(SA['Shot'][0]['Route'][0]['x']) # X coordinates of ball 1, shot 1