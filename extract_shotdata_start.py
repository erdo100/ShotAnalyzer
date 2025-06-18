import numpy as np

from read_gamefile import read_gamefile
from extract_b1b2b3_start import extract_b1b2b3_start
from extract_events_start import extract_events_start
from extract_dataquality_start import extract_dataquality_start

def check_shot_evaluation_status(SA, si):
    """
    Check if a shot has been fully evaluated by examining populated columns.
    
    Args:
        SA: Shot Analyzer data structure
        si: Shot index
        
    Returns:
        dict: Status of different evaluation steps
    """
    status = {
        'data_quality': False,
        'b1b2b3': False,  
        'events': False,
        'fully_evaluated': False
    }
    
    try:
        # Check if data quality has been processed (has ErrorID/ErrorText or no errors)
        if 'ErrorID' in SA['Data'].columns and 'ErrorText' in SA['Data'].columns:
            import pandas as pd
            status['data_quality'] = (
                pd.notna(SA['Data'].iloc[si]['ErrorID']) or 
                pd.notna(SA['Data'].iloc[si]['ErrorText']) or
                SA['Data'].iloc[si]['Interpreted'] != 0
            )
        
        # Check if B1B2B3 has been processed
        if 'B1B2B3' in SA['Data'].columns:
            import pandas as pd
            b1b2b3_val = SA['Data'].iloc[si]['B1B2B3']
            status['b1b2b3'] = (
                pd.notna(b1b2b3_val) and 
                b1b2b3_val != '' and 
                b1b2b3_val is not None
            )
            
        # Check if events have been processed
        if all(col in SA['Data'].columns for col in ['B1hit', 'B2hit', 'B3hit']):
            import pandas as pd
            status['events'] = any([
                (pd.notna(SA['Data'].iloc[si]['B1hit']) and SA['Data'].iloc[si]['B1hit'] != ''),
                (pd.notna(SA['Data'].iloc[si]['B2hit']) and SA['Data'].iloc[si]['B2hit'] != ''), 
                (pd.notna(SA['Data'].iloc[si]['B3hit']) and SA['Data'].iloc[si]['B3hit'] != '')
            ])
            
        status['fully_evaluated'] = all([status['data_quality'], status['b1b2b3'], status['events']])
        
    except Exception as e:
        print(f"Error checking evaluation status for shot {si}: {e}")
        
    return status


# Main execution function (similar to the original script)
def extract_shotdata_start(self):
    """
    Extract shot data by sequentially calling the required functions.
    Only processes shots that haven't been fully evaluated yet.
    """

    print("Starting shot data extraction...")
    
    # Get initial statistics
    SA = self.SA
    if SA is None or 'Data' not in SA or len(SA['Data']) == 0:
        print("No shot data to process.")
        return
        
    total_shots = len(SA['Data'])
    fully_evaluated_count = 0
      # Check current evaluation status and store initial state
    initial_evaluated_shots = set()
    for si in range(total_shots):
        status = check_shot_evaluation_status(SA, si)
        if status['fully_evaluated']:
            fully_evaluated_count += 1
            initial_evaluated_shots.add(si)
            
    print(f"Found {total_shots} total shots, {fully_evaluated_count} already fully evaluated.")
    
    if fully_evaluated_count == total_shots:
        print("All shots are already fully evaluated. No processing needed.")
        return

    # Step 1: Extract data quality (only for shots that need it)
    print("Extracting data quality...")
    extract_dataquality_start(self)

    # Step 2: Extract B1B2B3 start (only for shots that need it)
    print("Extracting B1B2B3 start...")
    extract_b1b2b3_start(self)

    # Step 3: Extract events (only for shots that need it)
    print("Extracting events...")
    extract_events_start(self)    # Final status check and mark fully evaluated shots
    final_fully_evaluated_count = 0
    newly_evaluated_shots = []
    
    for si in range(total_shots):
        status = check_shot_evaluation_status(SA, si)
        if status['fully_evaluated']:
            final_fully_evaluated_count += 1
            # If this shot wasn't fully evaluated before, mark it as interpreted
            if si not in initial_evaluated_shots:
                try:
                    SA['Data'].iloc[si, SA['Data'].columns.get_loc('Interpreted')] = 1
                    newly_evaluated_shots.append(si)
                    print(f"Shot {si} marked as fully evaluated (Interpreted = 1)")
                except Exception as e:
                    print(f"Error setting Interpreted status for shot {si}: {e}")
            
    newly_evaluated = len(newly_evaluated_shots)
    print(f"Shot data extraction process completed.")
    print(f"Processed {newly_evaluated} new shots. Total evaluated: {final_fully_evaluated_count}/{total_shots}")
    
    if newly_evaluated_shots:
        print(f"Newly evaluated shots: {newly_evaluated_shots}")
        
    # Refresh the table display
    self.refresh_table()

    # You can now access the processed data in SA
    # Example: print(self.SA['Data'].head())
    # Example: print(self.SA['Shot'][0]['Ball'][0]['x']) # X coordinates of ball 1, shot 1