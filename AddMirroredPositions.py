class AddMirroredPositions:
    def __init__(self, SA, param, update_shot_list, player_function):
        self.SA = SA
        self.param = param
        self.update_shot_list = update_shot_list
        self.player_function = player_function

    def execute(self):
        print("Start mirroring ...")

        SAnew = {
            'Shot': [],
            'Table': [],
            'ShotIDsVisible': [],
            'ColumnsVisible': self.SA['ColumnsVisible'],
            'Current_si': self.SA['Current_si'],
            'Current_ti': self.SA['Current_ti'],
            'fullfilename': self.SA['fullfilename']
        }

        for si, shot in enumerate(self.SA['Shot']):
            if self.SA['Table']['Mirrored'][si] != 0:
                # Copy existing mirrored shot
                SAnew['Shot'].append(shot)
                SAnew['Table'].append(self.SA['Table'][si])
                SAnew['ShotIDsVisible'].append(self.SA['ShotIDsVisible'][si])
            else:
                # Create mirrored shots
                for i in range(1, 5):
                    mirrored_shot = self.mirror_shot(shot, i)
                    SAnew['Shot'].append(mirrored_shot)
                    mirrored_table_entry = self.SA['Table'][si].copy()
                    mirrored_table_entry['Mirrored'] = i
                    SAnew['Table'].append(mirrored_table_entry)
                    SAnew['ShotIDsVisible'].append(
                        self.SA['Table']['ShotID'][si] + i / 10
                    )

        # Overwrite SA with new data
        self.SA.update(SAnew)

        # Update the GUI
        self.update_shot_list()

        # Plot selected things
        self.player_function('plotcurrent')

        print(f"Mirroring done. Database has now {len(self.SA['Shot'])} shots")

    def mirror_shot(self, shot, mode):
        mirrored_shot = shot.copy()
        for bi in range(3):
            if mode in [2, 4]:
                # Mirror X-axis
                mirrored_shot['Route0'][bi]['x'] = self.param['size'][1] - shot['Route0'][bi]['x']
                mirrored_shot['Route'][bi]['x'] = self.param['size'][1] - shot['Route'][bi]['x']
            if mode in [3, 4]:
                # Mirror Y-axis
                mirrored_shot['Route0'][bi]['y'] = self.param['size'][0] - shot['Route0'][bi]['y']
                mirrored_shot['Route'][bi]['y'] = self.param['size'][0] - shot['Route'][bi]['y']
        return mirrored_shot