from PyQt5.QtWidgets import QMenu, QAction

def create_table_menu(main_window, player):
    """
    Creates the table menu for the main window.

    Args:
        main_window: The main application window.
        player: The player object containing settings and state.
    """
    menu_bar = main_window.menuBar()

    # Plot Menu
    plot_menu = QMenu("Plot Settings", main_window)
    menu_bar.addMenu(plot_menu)

    plot_selected_action = QAction("Plot Selected", main_window)
    plot_selected_action.setCheckable(True)
    plot_selected_action.setChecked(player['setting']['plot_selected'])
    plot_selected_action.triggered.connect(lambda: toggle_setting(player, 'plot_selected'))
    plot_menu.addAction(plot_selected_action)

    plot_time_diagram_action = QAction("Plot Time Diagram", main_window)
    plot_time_diagram_action.setCheckable(True)
    plot_time_diagram_action.setChecked(player['setting']['plot_timediagram'])
    plot_time_diagram_action.triggered.connect(lambda: toggle_setting(player, 'plot_timediagram'))
    plot_menu.addAction(plot_time_diagram_action)

    extend_shot_line_action = QAction("Extend Shot Line", main_window)
    extend_shot_line_action.setCheckable(True)
    extend_shot_line_action.setChecked(False)  # Default value
    extend_shot_line_action.triggered.connect(lambda: toggle_setting(player, 'plot_check_extendline'))
    plot_menu.addAction(extend_shot_line_action)

    plot_only_blue_action = QAction("Plot Only in Blue", main_window)
    plot_only_blue_action.setCheckable(True)
    plot_only_blue_action.setChecked(player['setting']['plot_only_blue_table'] == 'on')
    plot_only_blue_action.triggered.connect(lambda: toggle_setting(player, 'plot_only_blue_table', toggle_value='on'))
    plot_menu.addAction(plot_only_blue_action)

    # Ball Settings
    for color, index in zip(['White', 'Yellow', 'Red'], range(3)):
        ball_menu = QMenu(f"{color} Ball", main_window)
        plot_menu.addMenu(ball_menu)

        ball_action = QAction(f"{color} Ball", main_window)
        ball_action.setCheckable(True)
        ball_action.setChecked(player['setting']['ball'][index]['ball'])
        ball_action.triggered.connect(lambda checked, idx=index: toggle_ball_setting(player, idx, 'ball', checked))
        ball_menu.addAction(ball_action)

        line_action = QAction(f"{color} Ball Line", main_window)
        line_action.setCheckable(True)
        line_action.setChecked(player['setting']['ball'][index]['line'])
        line_action.triggered.connect(lambda checked, idx=index: toggle_ball_setting(player, idx, 'line', checked))
        ball_menu.addAction(line_action)

        initial_pos_action = QAction(f"{color} Initial Position", main_window)
        initial_pos_action.setCheckable(True)
        initial_pos_action.setChecked(player['setting']['ball'][index]['initialpos'])
        initial_pos_action.triggered.connect(lambda checked, idx=index: toggle_ball_setting(player, idx, 'initialpos', checked))
        ball_menu.addAction(initial_pos_action)

        marker_action = QAction(f"{color} Ball Marker", main_window)
        marker_action.setCheckable(True)
        marker_action.setChecked(player['setting']['ball'][index]['marker'])
        marker_action.triggered.connect(lambda checked, idx=index: toggle_ball_setting(player, idx, 'marker', checked))
        ball_menu.addAction(marker_action)

        ghostball_action = QAction(f"{color} Ghostball", main_window)
        ghostball_action.setCheckable(True)
        ghostball_action.setChecked(player['setting']['ball'][index]['ghostball'])
        ghostball_action.triggered.connect(lambda checked, idx=index: toggle_ball_setting(player, idx, 'ghostball', checked))
        ball_menu.addAction(ghostball_action)

        diamond_pos_action = QAction(f"{color} Ball Diamond Position", main_window)
        diamond_pos_action.setCheckable(True)
        diamond_pos_action.setChecked(player['setting']['ball'][index]['diamondpos'])
        diamond_pos_action.triggered.connect(lambda checked, idx=index: toggle_ball_setting(player, idx, 'diamondpos', checked))
        ball_menu.addAction(diamond_pos_action)

def toggle_setting(player, setting_key, toggle_value=True):
    """
    Toggles a boolean setting in the player object.

    Args:
        player: The player object.
        setting_key: The key of the setting to toggle.
        toggle_value: The value to toggle to (default is True).
    """
    current_value = player['setting'].get(setting_key, False)
    player['setting'][setting_key] = not current_value if toggle_value is True else toggle_value
    print(f"Toggled {setting_key} to {player['setting'][setting_key]}")

def toggle_ball_setting(player, ball_index, setting_key, value):
    """
    Toggles a setting for a specific ball in the player object.

    Args:
        player: The player object.
        ball_index: The index of the ball (0, 1, or 2).
        setting_key: The key of the setting to toggle.
        value: The value to set.
    """
    player['setting']['ball'][ball_index][setting_key] = value
    print(f"Set {setting_key} for ball {ball_index} to {value}")