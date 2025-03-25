from identify_ShotId import identify_shot_id
from extract_b1b2b3 import extract_b1b2b3
from update_shot_list import update_shot_list

def extract_b1b2b3_start(SA, player):
    print(f"start ({__name__})")
    varname = 'B1B2B3'

    hsl_data = SA['Table']['Data']
    hsl_columns = SA['Table']['ColumnName']

    for ti, row in enumerate(hsl_data):
        identify_shot_id(ti, hsl_data, hsl_columns, SA)
        si = SA['Current_si']

        if SA['Table']['Interpreted'][si] == 0:
            b1b2b3, err = extract_b1b2b3(SA['Shot'][si])

            if err['code']:
                SA['Table']['Selected'][si] = True
                print(f"{si}: {err['text']}")

            SA['Table'][varname][si] = b1b2b3
            SA['Table']['ErrorID'][si] = err['code']
            SA['Table']['ErrorText'][si] = err['text']

    update_shot_list(SA)
    player['uptodate'] = False

    selected_count = sum(SA['Table']['Selected'])
    print(f"{selected_count}/{len(SA['Table']['Selected'])} shots selected")
    print(f"done ({__name__})")