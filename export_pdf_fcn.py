import os
from close_table_figure import close_table_figure
from player_function import player_function
from PyQt6.QtWidgets import QApplication

def export_pdf_fcn(SA, param, player):
    pathname, filename = os.path.split(SA['fullfilename'])
    exportfolder = os.path.join(pathname, 'PNG_export')

    if not os.path.isdir(exportfolder):
        os.mkdir(exportfolder)

    si_sel = [i for i, selected in enumerate(SA['Table']['Selected']) if selected]

    app = QApplication.instance()

    for si in si_sel:
        SA['Current_si'] = si
        player['uptodate'] = False

        player_function('plotcurrent', player)

        fig = next((w for w in app.allWidgets() if w.objectName() == 'Table_figure'), None)
        if fig:
            filename = os.path.join(
                exportfolder,
                f"{SA['Table']['Filename'][si]}_{SA['Table']['ShotID'][si]:03d}_{SA['Table']['Mirrored'][si]}.png"
            )
            fig.grab().save(filename)
            print(f"Exported to {filename}")

    close_table_figure(param)
    player_function('plotcurrent', player)