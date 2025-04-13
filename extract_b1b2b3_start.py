import os
import numpy as np
import pandas as pd
from angle_vector import angle_vector
from delete_selected_shots import delete_selected_shots
from extract_b1b2b3 import extract_b1b2b3



# Translated function from Extract_b1b2b3_start.m
# Needs the main data structure SA (containing 'Shot' list and 'Table' DataFrame)
# and param dictionary (though param seems unused in the MATLAB version of this specific function)
def extract_b1b2b3_start(self): # param added for consistency, but unused here
    """
    Iterates through shots, determines the B1B2B3 order using extract_b1b2b3,
    and updates the SA['Table'].

    Args:
        SA (dict): The main data structure containing 'Shot' list and 'Table' DataFrame.
        param (dict): Parameters dictionary (unused in this function's core logic).
    """
    print(f'start ({os.path.basename(__file__)} calling extract_b1b2b3)') # Indicate function start

    SA = self.SA
    param = self.param

    if SA is None or 'Table' not in SA or SA['Table'] is None or 'Shot' not in SA:
        print("Error: SA structure is invalid or empty.")
        return

    num_shots = len(SA['Table'])
    if num_shots == 0:
        print("No shots to process for B1B2B3 extraction.")
        return

    if num_shots != len(SA['Shot']):
         print(f"Warning: Mismatch between Table rows ({num_shots}) and Shot entries ({len(SA['Shot'])}).")
         num_shots = min(num_shots, len(SA['Shot'])) # Process only matching entries

    varname = 'B1B2B3' # Column name to store the result in the table
    if varname not in SA['Table'].columns:
        # Add the column if it doesn't exist, initialize with None or empty string
        SA['Table'][varname] = None # Or np.nan, or ''

    selected_count = 0
    processed_count = 0


    # Iterate through shots using the DataFrame index
    for si, current_shot_id in enumerate(SA['Table']['ShotID']):
        print(f"Processing shot index {si} (ShotID: {current_shot_id})...")
        try:
            # Check if the shot has already been interpreted (skip if so)
            if SA['Table'].iloc[si]['Interpreted'] == 0:
                processed_count += 1
                # Call the function to determine B1B2B3 order
                current_shot_data = SA['Shot'][si]
                b1b2b3_result, b1b2b3_num, err_info = extract_b1b2b3(current_shot_data)

                if err_info['code'] is not None:
                    print(f"ShotIndex {si}: Error {err_info['code']} - {err_info['text']}")
                # Update the table with the result
                SA['Table'].loc[SA['Table'].index[si], varname] = b1b2b3_result

                #correct for b2 and b3 the trajectory data so that time is added before the hit
                for bi in range(1, 3):
                    SA['Shot'][si]['Route'][b1b2b3_num[bi]]['t'] = \
                        np.insert(SA['Shot'][si]['Route'][b1b2b3_num[bi]]['t'], 1, SA['Shot'][si]['Route'][b1b2b3_num[bi]]['t'][1] - 0.01)
                    SA['Shot'][si]['Route'][b1b2b3_num[bi]]['x'] = \
                        np.insert(SA['Shot'][si]['Route'][b1b2b3_num[bi]]['x'], 1, SA['Shot'][si]['Route'][b1b2b3_num[bi]]['x'][0])
                    SA['Shot'][si]['Route'][b1b2b3_num[bi]]['y'] = \
                        np.insert(SA['Shot'][si]['Route'][b1b2b3_num[bi]]['y'], 1, SA['Shot'][si]['Route'][b1b2b3_num[bi]]['y'][0])
                    
                # Update error info and selection status if an error occurred
                if err_info['code'] is not None:
                    SA['Table'].loc[SA['Table'].index[si], 'ErrorID'] = err_info['code']
                    SA['Table'].loc[SA['Table'].index[si], 'ErrorText'] = err_info['text']
                    SA['Table'].loc[SA['Table'].index[si], 'Selected'] = True
                    selected_count += 1
                    # Optional: print error message like in MATLAB
                    # print(f"ShotIndex {si}: {err_info['text']}")
                else:
                     # If calculation was successful and no previous error, ensure ErrorID/Text reflect this
                     # This depends on whether other functions might set errors; be careful here.
                     # For now, we only update if extract_b1b2b3 finds an error.
                     pass


        except IndexError:
             print(f"Error: Index {si} out of bounds for SA['Table'] or SA['Shot'] during B1B2B3.")
             # Optionally mark this as an error in the table
             SA['Table'].loc[SA['Table'].index[si], 'ErrorID'] = 98 # Custom error code
             SA['Table'].loc[SA['Table'].index[si], 'ErrorText'] = 'Internal index error during B1B2B3 processing'
             SA['Table'].loc[SA['Table'].index[si], 'Selected'] = True
             selected_count +=1
        except Exception as e:
             print(f"Error processing shot index {si} for B1B2B3: {e}")
             SA['Table'].loc[SA['Table'].index[si], 'ErrorID'] = 97 # Custom error code
             SA['Table'].loc[SA['Table'].index[si], 'ErrorText'] = f'Unexpected error during B1B2B3: {e}'
             SA['Table'].loc[SA['Table'].index[si], 'Selected'] = True
             selected_count +=1


    # Update the total selected count based on the current state of the 'Selected' column
    final_selected_count = SA['Table']['Selected'].sum()

    print(f"Processed {processed_count} uninterpreted shots for B1B2B3.")
    # print(f"{selected_count} shots newly marked as selected due to B1B2B3 errors.")
    print(f'{final_selected_count}/{num_shots} total shots selected (marked with errors/warnings)')

    print("\nRunning deletion of selected shots...")
    delete_selected_shots(SA) # Modify SA in place

    print(f'done ({os.path.basename(__file__)} finished extract_b1b2b3_start)')

