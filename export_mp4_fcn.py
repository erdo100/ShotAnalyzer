import os
from close_table_figure import close_table_figure
from player_function import player_function

def export_mp4_fcn(SA, param, player):
    pathname, filename = os.path.split(SA['fullfilename'])
    exportfolder = 'MP4_export'

    if not os.path.isdir(exportfolder):
        os.mkdir(exportfolder)

    si_sel = [i for i, selected in enumerate(SA['Table']['Selected']) if selected]

    for si in si_sel:
        SA['Current_si'] = si
        player['uptodate'] = False

        file = f"{SA['Table']['Filename'][si]}_{SA['Table']['ShotID'][si]:03d}_{SA['Table']['Mirrored'][si]}.mp4"
        player['videofile'] = os.path.join(exportfolder, file)

        player_function('record_batch', player)

    close_table_figure(param)
    player_function('plotcurrent', player)