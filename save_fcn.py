import os

def save_function(mode):
    # Placeholder function for saving data
    print(f"Saving data with mode {mode}...")

    # Example: Simulate file saving
    file_path = os.path.join("..", "Gamedata", "saved_file.saf")

    if mode == 0:
        print(f"Saving to existing file: {file_path}")
    elif mode == 1:
        print(f"Saving all data to a new file: {file_path}")
    elif mode == 2:
        print(f"Saving selected data to a new file: {file_path}")
    else:
        print("Invalid save mode.")
        return

    # Simulate saving the file
    print(f"Data successfully saved to {file_path}")