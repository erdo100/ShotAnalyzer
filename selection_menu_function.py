def selection_menu_function(action, SA):
    marked = [i for i, selected in enumerate(SA['Table']['Selected']) if selected]
    nonmarked = [i for i, selected in enumerate(SA['Table']['Selected']) if not selected]

    if action == 'Select marked shots':
        for i in marked:
            SA['Table']['Selected'][i] = True

    elif action == 'Unselect marked shots':
        for i in marked:
            SA['Table']['Selected'][i] = False

    elif action == 'Invert Selection':
        if marked:
            for i in marked:
                SA['Table']['Selected'][i] = False
            for i in nonmarked:
                SA['Table']['Selected'][i] = True
        else:
            print('Nothing marked, nothing to invert.')

    elif action == 'Unselect all shots':
        SA['Table']['Selected'] = [False] * len(SA['Table']['Selected'])

    elif action == 'Select all shots':
        SA['Table']['Selected'] = [True] * len(SA['Table']['Selected'])

    elif action == 'Delete selected shots':
        if marked:
            SA['Shot'] = [shot for i, shot in enumerate(SA['Shot']) if i not in marked]
            SA['Table'] = SA['Table'].drop(index=marked).reset_index(drop=True)
            print('Delete completed.')
        else:
            print('Nothing marked, nothing to delete.')

    elif action == 'Hide selected shots':
        if marked:
            print('Hide completed.')
        else:
            print('Nothing marked, nothing to hide.')

    # Update GUI
    update_shot_list(SA)

    # Plot selected things
    player_function('plotcurrent', SA)