from PyQt6.QtWidgets import QInputDialog

def line_width_menu_function(player):
    prompt = ['White:', 'Yellow:', 'Red:']
    dlgtitle = 'Line width'

    for i, color in enumerate(prompt):
        value, ok = QInputDialog.getText(None, dlgtitle, f"{color}", text=str(player['setting']['lw'][i]))
        if ok:
            player['setting']['lw'][i] = float(value)

    # Update Plot
    player_function('plotcurrent', player)