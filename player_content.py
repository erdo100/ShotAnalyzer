def player_content(player, action):
    if action == 'initialize':
        player['setting'] = {
            'plot_selected': False,
            'plot_timediagram': False,
            'plot_check_extendline': False,
            'plot_only_blue_table': 'off',
            'lw': [7, 7, 7],
            'ball': [
                {'ball': True, 'line': True, 'initialpos': True, 'marker': False, 'ghostball': True, 'diamondpos': False},
                {'ball': True, 'line': True, 'initialpos': True, 'marker': False, 'ghostball': True, 'diamondpos': False},
                {'ball': True, 'line': True, 'initialpos': True, 'marker': False, 'ghostball': True, 'diamondpos': False}
            ]
        }
        player['uptodate'] = 0
        print("Player initialized.")

    elif action == 'update':
        # Example: Update player settings or state
        player['uptodate'] = 1
        print("Player updated.")

    elif action == 'reset':
        # Reset specific player settings
        player['setting']['plot_selected'] = False
        player['setting']['plot_timediagram'] = False
        print("Player settings reset.")

    else:
        print(f"Unknown action: {action}")

    return player