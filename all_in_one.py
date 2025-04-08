import numpy as np
import json
import os
import pandas as pd
import copy
import warnings # Used to suppress potential division-by-zero warnings if dt is zero

from read_gamefile import read_gamefile
from extract_b1b2b3_start import extract_b1b2b3_start
from extract_events import extract_events
from extract_dataquality_start import extract_dataquality_start
from delete_selected_shots import delete_selected_shots
from extract_b1b2b3 import extract_b1b2b3
from angle_vector import angle_vector
from extract_events import extract_events
from str2num_b1b2b3 import str2num_b1b2b3


def extract_events_start(SA, param=None): # param added for consistency, but unused here
    """Extract all ball-ball hit events and ball-Cushion hit events
    """
    print(f'start ({os.path.basename(__file__)} calling extract_b1b2b3)') # Indicate function start
    num_shots = len(SA['Table'])
    if num_shots == 0:
        print("No shots to process for B1B2B3 extraction.")
        return

    if num_shots != len(SA['Shot']):
         print(f"Warning: Mismatch between Table rows ({num_shots}) and Shot entries ({len(SA['Shot'])}).")
         num_shots = min(num_shots, len(SA['Shot'])) # Process only matching entries

    err = {'code': None, 'text': ''}

    # Iterate through shots using the DataFrame index
    for si, current_shot_id in enumerate(SA['Table']['ShotID']):
        print(f"Processing shot index {si} (ShotID: {current_shot_id})...")

        b1b2b3_num, b1i, b2i, b3i = str2num_b1b2b3(SA['Table'].iloc[si]['B1B2B3'])
        
        # extract all events
        hit = extract_events(SA, si, param)
        print(f"Hit data extracted for shot index {si}.")



# Main execution function (similar to the original script)
def extract_shotdata_cmd(filepath):
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

    # You can now access the processed data in SA
    # Example: print(SA['Table'].head())
    # Example: print(SA['Shot'][0]['Route'][0]['x']) # X coordinates of ball 1, shot 1


# Example Usage:
if __name__ == "__main__":
    # Choose the file path to test
    filepath = "D:/Programming/Shotdata/JSON/20210704_Ersin_Cemal.json" # Use forward slashes or raw string
    # filepath = r"D:\Billard\0_AllDatabase\WebSport\20170906_match_01_Ersin_Cemal.txt" # Use raw string for Windows paths

    # Check if file exists before running
    if os.path.exists(filepath):
        extract_shotdata_cmd(filepath)
    else:
        print(f"Error: Test file not found at {filepath}")

