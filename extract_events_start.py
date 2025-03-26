from str2num_b1b2b3 import str2num_b1b2b3
from extract_events import extract_events
from eval_hit_events import eval_hit_events
from eval_point_and_kiss_control import eval_point_and_kiss_control
from create_varname import create_varname
from update_shot_list import update_shot_list
from player_function import player_function

def extract_events_start(SA, player):
    """
    Extract all ball-ball hit events and ball-cushion hit events.

    Args:
        SA (object): Global Shot Analyzer object.
        player (object): Global player object.
    """

    err = 0
    err_shots = []

    shotlength = len(SA['Shot'])
    for si in range(shotlength):
        if SA['Interpreted'][si] == 0:
            print(f"Shot {si + 1}/{shotlength}")
            if len(SA['B1B2B3'][si]) == 3:
                try:
                    b1b2b3, b1i, b2i, b3i = str2num_b1b2b3(SA['B1B2B3'][si])

                    # Extract all events
                    hit, _ = extract_events(si)

                    # Collect the events
                    hit = eval_hit_events(hit, si, b1b2b3)

                    # Evaluate hit accuracy and kiss control
                    hit = eval_point_and_kiss_control(si, hit)

                    # Create the SA.Table
                    SA = create_varname(SA, hit, si)

                    # Copy the hit data to SA
                    SA['Shot'][si]['hit'] = hit

                    # Set interpreted flag
                    SA['Interpreted'][si] = 1

                    # Reset flags and delete Route data
                    SA['ErrorID'][si] = None
                    SA['ErrorText'][si] = None
                    SA['Selected'][si] = False
                    SA['Shot'][si]['Route'] = None

                except Exception as e:
                    SA['ErrorID'][si] = 100
                    SA['ErrorText'][si] = 'Check diagram, correct or delete.'
                    SA['Selected'][si] = True

                    print("Some error occurred, probably ball routes are not continuous. Check diagram, correct or delete")
                    err += 1
                    err_shots.append(si)
            else:
                print(f"B1B2B3 has not 3 letters, skipping Shot {si + 1}")

    update_shot_list()

    print("These Shots are not interpreted:")
    for shot in err_shots:
        print(shot)

    # Update GUI
    update_shot_list()
    player['uptodate'] = 0
    player_function('plotcurrent', None)

    print("done (extract_events_start)")