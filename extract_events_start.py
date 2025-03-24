from extract_events import extract_events
from eval_hit_events import eval_hit_events
from eval_point_and_kiss_control import eval_point_and_kiss_control
from create_varname import create_varname
from update_shot_list import update_shot_list
from player_function import player_function

def extract_events_start(SA, player):
    if 'B1B2B3' not in SA['Table']:
        print(f"B1B2B3 is not identified yet ({__name__})")
        return

    err = 0
    err_shots = []

    for si, interpreted in enumerate(SA['Table']['Interpreted']):
        if not interpreted:
            print(f"Shot {si + 1}/{len(SA['Shot'])}")

            if len(SA['Table']['B1B2B3'][si]) == 3:
                try:
                    b1b2b3, b1i, b2i, b3i = str2num_B1B2B3(SA['Table']['B1B2B3'][si])

                    hit, _ = extract_events(si, SA)
                    hit = eval_hit_events(hit, si, b1b2b3, SA, param)
                    hit = eval_point_and_kiss_control(si, hit, SA, param)

                    SA = create_varname(SA, hit, si, param, str2num_B1B2B3, replace_colors_b1b2b3)
                    SA['Shot'][si]['hit'] = hit

                    SA['Table']['Interpreted'][si] = 1
                    SA['Table']['ErrorID'][si] = None
                    SA['Table']['ErrorText'][si] = None
                    SA['Table']['Selected'][si] = False
                    SA['Shot'][si]['Route'] = None

                except Exception as e:
                    SA['Table']['ErrorID'][si] = 100
                    SA['Table']['ErrorText'][si] = 'Check diagram, correct or delete.'
                    SA['Table']['Selected'][si] = True

                    print("Some error occurred, probably ball routes are not continuous. Check diagram, correct or delete")
                    err += 1
                    err_shots.append(si)
            else:
                print(f"B1B2B3 has not 3 letters, skipping Shot {si}")

    update_shot_list(SA)

    print("These Shots are not interpreted:")
    for shot in err_shots:
        print(shot)

    update_shot_list(SA)
    player['uptodate'] = False
    player_function('plotcurrent', player)

    print(f"done ({__name__})")