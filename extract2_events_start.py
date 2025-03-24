from extract_events_start import extract_events_start

def extract2_events_start(SA):
    for si in range(len(SA['Shot'])):
        SA['Table']['Interpreted'][si] = 0

    extract_events_start(SA, None)