import os
from shutil import copyfile

def export_bbs_fcn(SA, param):
    from PyQt6.QtWidgets import QFileDialog

    # Request the files to load
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
    dialog.setNameFilter("BBS Files (*.bbs)")
    dialog.setDirectory(os.path.dirname(SA['fullfilename']))

    if dialog.exec():
        files = dialog.selectedFiles()
    else:
        print("Abort")
        return

    for file in files:
        with open(file, 'r') as f:
            BBStxt = f.readlines()

        # Search the content for selected lines
        TF = [0]
        for si, selected in enumerate(SA['Table']['Selected']):
            if selected and any(SA['Table']['Filename'][si] in line for line in BBStxt):
                TF.append(next(i for i, line in enumerate(BBStxt) if SA['Table']['Filename'][si] in line))

        # Backup the original file
        backup_file = f"{file}_orig"
        copyfile(file, backup_file)

        # Write the new file
        with open(file, 'w') as f:
            for index in TF:
                f.write(BBStxt[index])