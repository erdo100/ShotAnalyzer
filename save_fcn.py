import os
import pickle
from PyQt6.QtWidgets import QFileDialog

def save_fcn(mode, SA):
    file0 = SA.get('fullfilename', '')

    if mode == 0:
        if not os.path.isfile(file0):
            file, _ = QFileDialog.getSaveFileName(None, "Save Shot Analyzer File", file0, "Shot Analyzer Files (*.saf)")
            if not file:
                print("Abort")
                return
        else:
            file = file0

        SA1 = SA

    elif mode == 1:
        file, _ = QFileDialog.getSaveFileName(None, "Save Shot Analyzer File", file0, "Shot Analyzer Files (*.saf)")
        if not file:
            print("Abort")
            return

        SA1 = SA

    elif mode == 2:
        file, _ = QFileDialog.getSaveFileName(None, "Save Shot Analyzer File", file0, "Shot Analyzer Files (*.saf)")
        if not file:
            print("Abort")
            return

        selected_indices = [i for i, selected in enumerate(SA['Table']['Selected']) if selected]
        SA1 = {
            'Shot': [SA['Shot'][i] for i in selected_indices],
            'Table': SA['Table'].iloc[selected_indices]
        }

    print("Start saving ...")
    with open(file, 'wb') as f:
        pickle.dump(SA1, f)

    SA['fullfilename'] = file
    print("Done with save")