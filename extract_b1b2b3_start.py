from identify_ShotId import identify_shot_id
from extract_b1b2b3 import extract_b1b2b3
from update_shot_list import update_shot_list

def extract_b1b2b3_start(SA):
    print(f"start ({__name__})")
    varname = 'B1B2B3'
    SA[varname] = [None] * len(SA['Shot'])

    for si in range(len(SA['Shot'])):
        b1b2b3, err = extract_b1b2b3(SA['Shot'][si])

        if err['code']:
            SA['Selected'][si] = True
            print(f"{si}: {err['text']}")

        SA[varname][si] = b1b2b3
        SA['ErrorID'][si] = err['code']
        SA['ErrorText'][si] = err['text']


    selected_count = sum(SA['Selected'])
    print(f"{selected_count}/{len(SA['Shot'])} shots selected")
    print(f"done ({__name__})")

    return SA