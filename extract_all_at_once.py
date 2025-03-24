from extract_data_quality_start import extract_data_quality_start
from selection_menu_function import selection_menu_function
from add_mirrored_positions import AddMirroredPositions
from extract_b1b2b3_start import extract_b1b2b3_start
from extract_b1b2b3_position import extract_b1b2b3_position
from extract_events_start import extract_events_start

def extract_all_at_once(data, SA, param):
    data['Source']['Text'] = 'Delete selected shots'

    extract_data_quality_start(SA, param)
    selection_menu_function(0, data, SA, param)

    AddMirroredPositions(SA, param).execute()
    selection_menu_function(0, data, SA, param)

    extract_b1b2b3_start(SA, param)
    selection_menu_function(0, data, SA, param)

    extract_b1b2b3_position(SA, param)

    extract_events_start(SA, param)
    selection_menu_function(0, data, SA, param)