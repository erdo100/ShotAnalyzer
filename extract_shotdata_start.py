import numpy as np

from read_gamefile import read_gamefile
from extract_b1b2b3_start import extract_b1b2b3_start
from extract_events_start import extract_events_start
from extract_dataquality_start import extract_dataquality_start


# Main execution function (similar to the original script)
def extract_shotdata_start(self):
    """
    Extract shot data by sequentially calling the required functions.

    Args:
        filepath (str): Path to the game file.
    """

    print("Starting shot data extraction...")

    # Step 2: Extract data quality
    print("Extracting data quality...")
    extract_dataquality_start(self)

    # Step 3: Extract B1B2B3 start (Placeholder)
    print("Extracting B1B2B3 start...")
    extract_b1b2b3_start(self)

    # Step 4: Extract events (Placeholder)
    print("Extracting events...")
    
    extract_events_start(self)  # Enable plotting by setting plotflag to True

    print("Shot data extraction process completed.")



    # You can now access the processed data in SA
    # Example: print(self.SA['Data'].head())
    # Example: print(self.SA['Shot'][0]['Ball'][0]['x']) # X coordinates of ball 1, shot 1