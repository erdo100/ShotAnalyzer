import os
from PyQt6.QtWidgets import QFileDialog, QMessageBox

def read_all_gamedata(mode, SA, param, player):
    # Handle to ShotList Table
    if 'fullfilename' not in SA:
        SA['fullfilename'] = '../Gamedata/'
    elif isinstance(SA['fullfilename'], list):
        SA['fullfilename'] = SA['fullfilename'][0]

    # Request the files to load
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
    dialog.setNameFilter("All Shotanalyser files (*.bbt,*.txt,*.saf);;BilliardBallTracker file (*.bbt);;MyWebSport file (*.txt);;ShotAnalyzer file (*.saf)")
    dialog.setDirectory(SA['fullfilename'])

    if dialog.exec():
        files = dialog.selectedFiles()
    else:
        print("Abort")
        return

    # Delete current data if not in append mode
    if mode == 0:
        SA.clear()

    # Read files depending on file types
    for file in files:
        print(f"Reading {file}")
        _, ext = os.path.splitext(file)

        if ext == '.bbt':
            SAnew = read_bbtfile(file)
        elif ext == '.txt':
            SAnew = read_gamefile(file)
        elif ext == '.saf':
            SAnew = load_saf(file)
        else:
            print(f"File not useful: {file}")
            continue

        if not SAnew:
            print("File empty")
            continue

        # Append to available data
        SA = append_new_shot_data(SA, SAnew)

    if not SA:
        print("Empty shot list, nothing to do")
        return

    # Update Window Title
    if len(files) == 1 and mode == 0:
        QMessageBox.information(None, "Title", f"{param['ver']} {os.path.basename(files[0])}")
    else:
        QMessageBox.information(None, "Title", f"{param['ver']} Multiple Files")

    SA['fullfilename'] = files

    # Update the GUI
    update_shot_list(SA)

    # Identify the first shot
    identify_shot_id(0, SA['Table']['Data'], SA['Table']['ColumnName'], SA)

    # Plot selected things
    player['uptodate'] = False
    player_function('plotcurrent', player)

    print(f"Done. Database has now {len(SA['Shot'])} shots.")