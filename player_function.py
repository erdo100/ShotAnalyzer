from PyQt6.QtWidgets import QMessageBox

def player_function(action, player):
    if action == 'plotcurrent':
        player['video'] = 0
        # Clear previous plots
        print("Clearing previous plots...")
        # Plot new things
        for shot in player['Shot']:
            print(f"Plotting shot with {len(shot['ball'][0]['t'])} time points.")

        QMessageBox.information(None, "Plot", "Current plot updated.")

    elif action == 'record':
        player['video'] = 1
        print("Recording started...")
        # Simulate recording logic
        QMessageBox.information(None, "Record", "Recording completed.")

    elif action == 'play':
        player['video'] = 0
        print("Playing animation...")
        QMessageBox.information(None, "Play", "Animation played.")

    elif action == 'stop':
        player['video'] = 0
        print("Stopping animation...")
        QMessageBox.information(None, "Stop", "Animation stopped.")

    else:
        print(f"Action '{action}' is not implemented.")