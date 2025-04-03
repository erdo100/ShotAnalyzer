from identify_ShotId import identify_shot_id
from extract_b1b2b3 import extract_b1b2b3
from update_shot_list import update_shot_list

def extract_b1b2b3_start(SA):
    print(f"start ({__name__})")
    varname = 'B1B2B3'
    SA[varname] = None  # Initialize column for B1B2B3 data

    for si in range(len(SA)):
        b1b2b3, err = extract_b1b2b3(SA.iloc[si])

        if err['code']:
            SA.at[si, 'Selected'] = True
            print(f"{si}: {err['text']}")

        SA.at[si, varname] = b1b2b3
        SA.at[si, 'ErrorID'] = err['code']
        SA.at[si, 'ErrorText'] = err['text']

    selected_count = SA['Selected'].sum()
    print(f"{selected_count}/{len(SA)} shots selected")
    print(f"done ({__name__})")

    return SA