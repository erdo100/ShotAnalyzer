from identify_ShotId import identify_shot_id
from update_shot_list import update_shot_list

def extract_data_quality_start(SA, param):
    print(f"start ({__name__})")

    # Initialize ErrorID and ErrorText lists to match the number of shots
    SA['ErrorID'] = [None] * len(SA['Shot'])
    SA['ErrorText'] = [''] * len(SA['Shot'])

    for si in range(len(SA['Shot'])):

        err = {'code': None, 'text': ''}

        for bi in range(3):
            if len(SA['Shot'][si]['Route'][bi]['t']) == 0:
                err['code'] = 100
                err['text'] = f"Ball data is missing 1 ({__name__})"
                print(f"{si}: Ball data is missing 1 ({__name__})")
                SA['Selected'][si] = True

            elif len(SA['Shot'][si]['Route'][bi]['t']) == 1:
                err['code'] = 101
                err['text'] = f"Ball data is missing 2 ({__name__})"
                print(f"{si}: Ball data is missing 2 ({__name__})")
                SA['Selected'][si] = True

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
                if any(x > oncushion[1] for x in SA['Shot'][si]['Route'][bi]['x']):
                    SA['Shot'][si]['Route'][bi]['x'] = [
                        min(x, oncushion[1]) for x in SA['Shot'][si]['Route'][bi]['x']
                    ]

                if any(x < oncushion[3] for x in SA['Shot'][si]['Route'][bi]['x']):
                    SA['Shot'][si]['Route'][bi]['x'] = [
                        max(x, oncushion[3]) for x in SA['Shot'][si]['Route'][bi]['x']
                    ]

                if any(y < oncushion[0] for y in SA['Shot'][si]['Route'][bi]['y']):
                    SA['Shot'][si]['Route'][bi]['y'] = [
                        max(y, oncushion[0]) for y in SA['Shot'][si]['Route'][bi]['y']
                    ]

                if any(y > oncushion[2] for y in SA['Shot'][si]['Route'][bi]['y']):
                    SA['Shot'][si]['Route'][bi]['y'] = [
                        min(y, oncushion[2]) for y in SA['Shot'][si]['Route'][bi]['y']
                    ]

        SA['ErrorID'][si] = err['code']
        SA['ErrorText'][si] = err['text']

    selected_count = sum(SA['Selected'])
    print(f"{selected_count}/{len(SA['Shot'])} shots selected")
    print(f"done ({__name__})")