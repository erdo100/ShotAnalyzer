from read_gamefile import read_gamefile
from extract_dataquality_start import extract_data_quality_start
from extract_b1b2b3_start import extract_b1b2b3_start
from extract_b1b2b3_position import extract_b1b2b3_position
from extract_events_start import extract_events_start
import numpy as np

def extract_shotdata_cmd(filepath):
    """
    Extract shot data by sequentially calling the required functions.

    Args:
        filepath (str): Path to the game file.
    """

    # Table properties
    param = {
        "ver": "Shot Analyzer v0.43i",
        "size": [1420, 2840],
        "ballR": 61.5 / 2,
        "ballcirc": {
            "x": np.sin(np.linspace(0, 2 * np.pi, 36)) * (61.5 / 2),
            "y": np.cos(np.linspace(0, 2 * np.pi, 36)) * (61.5 / 2),
        },
        "rdiam": 7,
        "cushionwidth": 50,
        "diamdist": 97,
        "framewidth": 147,
        "colors": "wyr",
        "BallOutOfTableDetectionLimit": 30,
        "BallCushionHitDetectionRange": 50,
        "BallProjecttoCushionLimit": 10,
        "NoDataDistanceDetectionLimit": 600,
        "MaxTravelDistancePerDt": 0.1,
        "MaxVelocity": 12000,
        "timax_appr": 5,
    }


    print("Starting shot data extraction...")

    # Step 1: Read game file
    print("Reading game file...")
    SA = read_gamefile(filepath)

    # Step 2: Extract data quality
    print("Extracting data quality...")
    SA = extract_data_quality_start(SA, param)

    # Step 3: Extract B1B2B3 start
    print("Extracting B1B2B3 start...")
    SA = extract_b1b2b3_start(SA)

    # Step 4: Extract B1B2B3 positions
    print("Extracting B1B2B3 positions...")
    SA = extract_b1b2b3_position(SA, param)

    # Step 5: Extract events
    print("Extracting events...")
    SA = extract_events_start(SA, param)

    print("Shot data extraction completed.")



filepath = "D:/Programming/Shotdata/JSON/20210704_Ersin_Cemal.json"
extract_shotdata_cmd(filepath)