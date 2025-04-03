def extract_data_quality_start(SA, param):
    print(f"start ({__name__})")

    # Initialize ErrorID and ErrorText columns
    SA['ErrorID'] = None
    SA['ErrorText'] = ''

    # Initialize Selected column
    SA['Selected'] = False

    for si in range(len(SA)):
        err = {'code': None, 'text': ''}

        for bi in range(3):
            route_key = f"Route{bi}"
            if SA.iloc[si][route_key].empty:
                err['code'] = 100
                err['text'] = f"Ball data is missing 1 ({__name__})"
                print(f"{si}: Ball data is missing 1 ({__name__})")
                SA.at[si, 'Selected'] = True

            elif len(SA.iloc[si][route_key]['t']) == 1:
                err['code'] = 101
                err['text'] = f"Ball data is missing 2 ({__name__})"
                print(f"{si}: Ball data is missing 2 ({__name__})")
                SA.at[si, 'Selected'] = True

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
                route_key = f"Route{bi}"
                route = SA.iloc[si][route_key]

                if any(x > oncushion[1] for x in route['x']):
                    route['x'] = [min(x, oncushion[1]) for x in route['x']]

                if any(x < oncushion[3] for x in route['x']):
                    route['x'] = [max(x, oncushion[3]) for x in route['x']]

                if any(y < oncushion[0] for y in route['y']):
                    route['y'] = [max(y, oncushion[0]) for y in route['y']]

                if any(y > oncushion[2] for y in route['y']):
                    route['y'] = [min(y, oncushion[2]) for y in route['y']]

        SA.at[si, 'ErrorID'] = err['code']
        SA.at[si, 'ErrorText'] = err['text']

    selected_count = SA['Selected'].sum()
    print(f"{selected_count}/{len(SA)} shots selected")
    print(f"done ({__name__})")
    
    return SA
