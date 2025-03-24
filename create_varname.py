def create_varname(SA, hit, si, param, str2num_B1B2B3, replace_colors_b1b2b3):
    varparts = list(hit.keys())

    b1b2b3, b1i, b2i, b3i = str2num_B1B2B3(SA['Table']['B1B2B3'][si])

    # Write BX_with
    for balli in range(1, 4):
        varname = f"B{balli}_with"
        # Check if column already exists, then update visibility
        if varname not in SA['Table']:
            SA['ColumnsVisible'].append(varname)

        with_new = replace_colors_b1b2b3(hit[b1b2b3[balli - 1]]['with'], b1b2b3)
        SA['Table'][varname][si] = with_new

    SA['Table']['Point0'][si] = hit[b1i]['Point']
    SA['Table']['Kiss'][si] = hit[b1i]['Kiss']
    SA['Table']['Fuchs'][si] = hit[b1i]['Fuchs']
    SA['Table']['PointDist'][si] = hit[b1i]['PointDist']
    SA['Table']['KissDistB1'][si] = hit[b1i]['KissDistB1']
    SA['Table']['AimOffsetSS'][si] = hit[b1i]['AimOffsetSS']
    SA['Table']['AimOffsetLL'][si] = hit[b1i]['AimOffsetLL']
    SA['Table']['B1B2OffsetSS'][si] = hit[b1i]['B1B2OffsetSS']
    SA['Table']['B1B2OffsetLL'][si] = hit[b1i]['B1B2OffsetLL']
    SA['Table']['B1B3OffsetSS'][si] = hit[b1i]['B1B3OffsetSS']
    SA['Table']['B1B3OffsetLL'][si] = hit[b1i]['B1B3OffsetLL']

    delname = ['Fraction', 'DefAngle', 'CutAngle', 'CInAngle', 'COutAngle']

    himax = [8, 4, 0]

    for bi in range(1, 4):
        # Write other results
        for hi in range(1, himax[bi - 1] + 1):
            for vi in range(20, len(varparts)):
                varname = f"B{bi}_{hi}_{varparts[vi]}"

                if hi == 1 and varparts[vi] in delname:
                    continue

                if len(hit[b1b2b3[bi - 1]][varparts[vi]]) >= hi:
                    # Check if we have cushion position to store as diamond values
                    scale = param['size'][1] / 8 if 'Pos' in varparts[vi] else 1

                    # Add the column
                    SA['Table'][varname][si] = hit[b1b2b3[bi - 1]][varparts[vi]][hi - 1] / scale

    return SA