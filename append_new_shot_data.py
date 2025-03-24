def append_new_shot_data(current, new):
    """
    Append new shot data to the current dataset.
    """
    if not current:
        # If current dataset is empty, return the new dataset
        SA = new

        # Adjust visibility for new data
        SA['ShotIDsVisible'] = (
            new.get('ShotIDsVisible') or
            [sid + mir / 10 for sid, mir in zip(new['Table']['ShotID'], new['Table']['Mirrored'])]
        )
        SA['ColumnsVisible'] = new.get('ColumnsVisible') or list(new['Table'].keys())

    else:
        # Ensure both datasets have the same columns
        for name in set(current['Table'].keys()).union(new['Table'].keys()):
            if name not in current['Table']:
                current['Table'][name] = [None] * len(current['Table']['ShotID'])
            if name not in new['Table']:
                new['Table'][name] = [None] * len(new['Table']['ShotID'])

        # Remove duplicate ShotIDs from the new dataset
        current_ids = set(current['Table']['ShotID'])
        new_ids = set(new['Table']['ShotID'])
        duplicate_ids = current_ids.intersection(new_ids)

        new['Table'] = {
            key: [val for i, val in enumerate(values) if new['Table']['ShotID'][i] not in duplicate_ids]
            for key, values in new['Table'].items()
        }
        new['Shot'] = [shot for i, shot in enumerate(new['Shot']) if new['Table']['ShotID'][i] not in duplicate_ids]

        # Append new data to the current dataset
        SA = {
            'Table': {
                key: current['Table'][key] + new['Table'][key]
                for key in current['Table']
            },
            'Shot': current['Shot'] + new['Shot'],
            'ColumnsVisible': current['ColumnsVisible'] + new.get('ColumnsVisible', []),
            'ShotIDsVisible': current['ShotIDsVisible'] + new.get('ShotIDsVisible', [])
        }

    return SA