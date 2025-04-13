import os

from extract_shotdata_start import extract_shotdata_start


# Example Usage:
if __name__ == "__main__":
    # Choose the file path to test
    # filepath = "D:/Programming/Shotdata/JSON/20210704_Ersin_Cemal.json" # Use forward slashes or raw string
    filepath = r"D:\Billard\0_AllDatabase\WebSport\20170906_match_01_Ersin_Cemal.txt" # Use raw string for Windows paths

    # Check if file exists before running
    if os.path.exists(filepath):
        SA = extract_shotdata_start(filepath)

        print("Shot data extraction completed.")
    else:
        print(f"Error: Test file not found at {filepath}")

