from PyQt6.QtWidgets import QMessageBox

def plot_menu_function(player):
    # Simulate toggling settings
    player['setting']['plot_selected'] = not player['setting'].get('plot_selected', False)
    player['setting']['plot_timediagram'] = not player['setting'].get('plot_timediagram', False)
    player['setting']['plot_check_extendline'] = not player['setting'].get('plot_check_extendline', False)
    player['setting']['plot_only_blue_table'] = not player['setting'].get('plot_only_blue_table', False)

    # Ball settings
    color = ['white', 'yellow', 'red']
    tags = ['ball', 'line', 'initialpos', 'marker', 'ghostball', 'diamondpos']

    for bi in range(3):
        player['setting']['ball'][bi] = {}
        for tag in tags:
            player['setting']['ball'][bi][tag] = True  # Simulate toggling

    QMessageBox.information(None, "Plot Menu", "Plot settings updated.")

    # Simulate calling PlayerFunction
    player_function('replot', player)