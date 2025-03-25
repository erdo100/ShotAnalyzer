from identify_ShotId import identify_shot_id
from update_shot_list import update_shot_list

def extract_data_quality_start(SA, param, player):
    print(f"start ({__name__})")

    hsl_data = SA['Table']['Data']
    hsl_columns = SA['Table']['ColumnName']

    for ti, row in enumerate(hsl_data):
        identify_shot_id(ti, hsl_data, hsl_columns, SA)
        si = SA['Current_si']

        if SA['Table']['Interpreted'][si] == 0:
            SA['Shot'][si]['Route'] = SA['Shot'][si]['Route0']

            err = {'code': None, 'text': ''}
            SA['Table']['Selected'][si] = False

            for bi in range(3):
                if len(SA['Shot'][si]['Route'][bi]['t']) == 0:
                    err['code'] = 100
                    err['text'] = f"Ball data is missing 1 ({__name__})"
                    print(f"{si}: Ball data is missing 1 ({__name__})")
                    SA['Table']['Selected'][si] = True

                elif len(SA['Shot'][si]['Route'][bi]['t']) == 1:
                    err['code'] = 101
                    err['text'] = f"Ball data is missing 2 ({__name__})"
                    print(f"{si}: Ball data is missing 2 ({__name__})")
                    SA['Table']['Selected'][si] = True

            if err['code'] is None:
                tol = param['BallProjecttoCushionLimit']
                cushion = [
                    param['ballR'] - tol,
                    param['size'][1] - param['ballR'] + tol,
                    param['size'][0] - param['ballR'] + tol,
                    param['ballR'] - tol
                ]
                oncushion = [
                    param['ballR'] + 0.1,
                    param['size'][1] - param['ballR'] - 0.1,
                    param['size'][0] - param['ballR'] - 0.1,
                    param['ballR'] + 0.1
                ]

                for bi in range(3):
                    if any(SA['Shot'][si]['Route'][bi]['x'] > oncushion[1]):
                        SA['Shot'][si]['Route'][bi]['x'] = [
                            min(x, oncushion[1]) for x in SA['Shot'][si]['Route'][bi]['x']
                        ]

                    if any(SA['Shot'][si]['Route'][bi]['x'] < oncushion[3]):
                        SA['Shot'][si]['Route'][bi]['x'] = [
                            max(x, oncushion[3]) for x in SA['Shot'][si]['Route'][bi]['x']
                        ]

                    if any(SA['Shot'][si]['Route'][bi]['y'] < oncushion[0]):
                        SA['Shot'][si]['Route'][bi]['y'] = [
                            max(y, oncushion[0]) for y in SA['Shot'][si]['Route'][bi]['y']
                        ]

                    if any(SA['Shot'][si]['Route'][bi]['y'] > oncushion[2]):
                        SA['Shot'][si]['Route'][bi]['y'] = [
                            min(y, oncushion[2]) for y in SA['Shot'][si]['Route'][bi]['y']
                        ]

            SA['Table']['ErrorID'][si] = err['code']
            SA['Table']['ErrorText'][si] = err['text']

    update_shot_list(SA)
    player['uptodate'] = False

    selected_count = sum(SA['Table']['Selected'])
    print(f"{selected_count}/{len(SA['Table']['Selected'])} shots selected")
    print(f"done ({__name__})")